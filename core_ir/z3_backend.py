import z3

from .aliasing import AliasPolicy, AliasRelation, decompose_pointer
from .ir import (
    ConstVal,
    CoreVal,
    HoleVal,
    LoopOutVal,
    LoopVal,
    OpApp,
    ParamVal,
    SelectVal,
    SymVal,
    subst_val,
)
from .op_registry import OpRegistry
from .core_types import (
    BOOL,
    FLOAT,
    INT,
    STR,
    CoreType,
    TArray,
    TBitVec,
    TMem,
    TPtr,
    TStruct,
    TSym,
    TTuple,
    TVector,
)



def _as_bool_z3(x: z3.ExprRef) -> z3.BoolRef:
    if z3.is_bool(x):
        return x
    if z3.is_bv(x):
        return x != z3.BitVecVal(0, x.size())
    if z3.is_int(x):
        return x != 0
    return z3.BoolVal(True)


class CoreZ3Backend:
    """Denotational translation from CoreSSA-IR into Z3 5.4.1 with provenance-based
    pointer aliasing axioms, Memory-SSA arrays, bitvectors, structs, strings, and rigid UFs.
    """

    def __init__(self, reg: OpRegistry, alias_policy: AliasPolicy):
        self.reg = reg
        self.alias_policy = alias_policy
        self._sym_sorts: dict[str, z3.SortRef] = {}
        self._struct_sorts: dict[str, tuple[z3.DatatypeSortRef, z3.FuncDeclRef, dict[str, z3.FuncDeclRef]]] = {}
        self._ufs: dict[tuple, z3.FuncDeclRef] = {}
        self._params: dict[tuple[str, str], z3.ExprRef] = {}
        self._syms: dict[tuple[str, str], z3.ExprRef] = {}
        self._base_ids: dict[str, z3.ExprRef] = {}
        self._ptr_sort: z3.DatatypeSortRef | None = None
        self._ptr_ctor: z3.FuncDeclRef | None = None
        self._ptr_base: z3.FuncDeclRef | None = None
        self._ptr_off: z3.FuncDeclRef | None = None
        self._init_ptr_sort()

    def _init_ptr_sort(self) -> None:
        dt = z3.Datatype("CorePtr")
        dt.declare("mk_ptr", ("ptr_base", z3.IntSort()), ("ptr_off", z3.IntSort()))
        self._ptr_sort = dt.create()
        self._ptr_ctor = self._ptr_sort.mk_ptr
        self._ptr_base = self._ptr_sort.ptr_base
        self._ptr_off = self._ptr_sort.ptr_off

    def sym_sort(self, sort_name: str) -> z3.SortRef:
        if sort_name not in self._sym_sorts:
            self._sym_sorts[sort_name] = z3.DeclareSort(f"Sym_{sort_name}")
        return self._sym_sorts[sort_name]

    def sort_of(self, ty: CoreType) -> z3.SortRef:
        if ty == BOOL:
            return z3.BoolSort()
        if ty == INT:
            return z3.IntSort()
        if isinstance(ty, TBitVec):
            return z3.BitVecSort(ty.width)
        if ty == FLOAT:
            return z3.RealSort()
        if ty == STR:
            return z3.StringSort()
        if isinstance(ty, TSym):
            return self.sym_sort(ty.sort_name)
        if isinstance(ty, TPtr):
            assert self._ptr_sort is not None
            return self._ptr_sort
        if isinstance(ty, TMem):
            # Memory-SSA map from CorePtr -> Int (or polymorphic via per-element projection)
            assert self._ptr_sort is not None
            return z3.ArraySort(self._ptr_sort, z3.IntSort())
        if isinstance(ty, TArray):
            return z3.ArraySort(self.sort_of(ty.key), self.sort_of(ty.elem))
        if isinstance(ty, TStruct):
            return self._get_struct_info(ty)[0]
        if isinstance(ty, TVector):
            return z3.ArraySort(z3.IntSort(), self.sort_of(ty.elem))
        if isinstance(ty, TTuple):
            return self.sym_sort("Tuple_" + "_".join(str(e) for e in ty.elems))
        return z3.IntSort()

    def _get_struct_info(self, ty: TStruct):
        if ty.name not in self._struct_sorts:
            dt = z3.Datatype(f"CoreStruct_{ty.name}")
            fspecs = [(f"{ty.name}_{fname}", self.sort_of(fty)) for fname, fty in ty.fields]
            dt.declare(f"mk_{ty.name}", *fspecs)
            dt_sort = dt.create()
            ctor = getattr(dt_sort, f"mk_{ty.name}")
            accs = {fname: getattr(dt_sort, f"{ty.name}_{fname}") for fname, _ in ty.fields}
            self._struct_sorts[ty.name] = (dt_sort, ctor, accs)
        return self._struct_sorts[ty.name]

    def _get_uf(self, name: str, arg_sorts: list[z3.SortRef], ret_sort: z3.SortRef) -> z3.FuncDeclRef:
        key = (name, tuple(str(s) for s in arg_sorts), str(ret_sort))
        if key not in self._ufs:
            safe_name = "uf_" + name.replace(".", "_").replace(":", "_")
            self._ufs[key] = z3.Function(safe_name, *arg_sorts, ret_sort)
        return self._ufs[key]

    def _base_const(self, base_id: str) -> z3.ExprRef:
        if base_id not in self._base_ids:
            self._base_ids[base_id] = z3.Int(f"__base_{base_id.replace(':', '_')}")
        return self._base_ids[base_id]

    def _mem_array_for_elem(self, mem_expr: z3.ExprRef, elem_ty: CoreType) -> z3.ExprRef:
        """Project a Memory-SSA token into a typed array `CorePtr -> [[elem_ty]]`."""
        if elem_ty == INT:
            return mem_expr
        assert self._ptr_sort is not None
        arr_sort = z3.ArraySort(self._ptr_sort, self.sort_of(elem_ty))
        proj = self._get_uf(f"mem_view_{elem_ty}", [mem_expr.sort()], arr_sort)
        return proj(mem_expr)

    def _wrap_mem_from_elem(self, typed_arr: z3.ExprRef, elem_ty: CoreType, mem_ty: CoreType) -> z3.ExprRef:
        if elem_ty == INT:
            return typed_arr
        wrap = self._get_uf(f"mem_wrap_{elem_ty}", [typed_arr.sort()], self.sort_of(mem_ty))
        return wrap(typed_arr)

    def collect_aliasing_axioms(self, vals: list[CoreVal]) -> list[z3.BoolRef]:
        """Generate Z3 provenance-disjointness axioms implied by `AliasPolicy` for all pointers in `vals`."""
        ptrs: list[CoreVal] = []
        seen: set[int] = set()

        def walk(v: CoreVal) -> None:
            if id(v) in seen:
                return
            seen.add(id(v))
            if isinstance(v.ty, TPtr):
                ptrs.append(v)
            if isinstance(v, OpApp):
                for a in v.args:
                    walk(a)
            elif isinstance(v, SelectVal):
                walk(v.cond)
                walk(v.then_val)
                walk(v.else_val)
            elif isinstance(v, LoopVal):
                for a in v.init_vals:
                    walk(a)
                walk(v.cond)
                for b in v.body_next:
                    walk(b)
            elif isinstance(v, LoopOutVal):
                walk(v.loop)

        for v in vals:
            walk(v)

        axioms: list[z3.BoolRef] = []
        emitted_pairs: set[frozenset[str]] = set()
        for i in range(len(ptrs)):
            d1 = decompose_pointer(ptrs[i], self.reg)
            for j in range(i + 1, len(ptrs)):
                d2 = decompose_pointer(ptrs[j], self.reg)
                if d1.base_id and d2.base_id and d1.base_id != d2.base_id:
                    pair = frozenset((d1.base_id, d2.base_id))
                    if pair in emitted_pairs:
                        continue
                    if self.alias_policy.query(ptrs[i], ptrs[j], self.reg) == AliasRelation.NO_ALIAS:
                        emitted_pairs.add(pair)
                        axioms.append(self._base_const(d1.base_id) != self._base_const(d2.base_id))
        return axioms

    def to_z3(self, v: CoreVal, env: dict[str, z3.ExprRef] | None = None, _memo: dict[int, z3.ExprRef] | None = None) -> z3.ExprRef:
        """Translate a CoreSSA-IR value into a Z3 expression (DAG-memoized O(|V|))."""
        env = env or {}
        if _memo is None:
            _memo = {}
        if not env and id(v) in _memo:
            return _memo[id(v)][1]
        res = self._to_z3_impl(v, env, _memo)
        if not env:
            _memo[id(v)] = (v, res)
        return res

    def _to_z3_impl(self, v: CoreVal, env: dict[str, z3.ExprRef], _memo: dict[int, z3.ExprRef]) -> z3.ExprRef:

        if isinstance(v, ConstVal):
            if v.ty == BOOL:
                return z3.BoolVal(bool(v.value))
            if v.ty == INT:
                return z3.IntVal(int(v.value))
            if isinstance(v.ty, TBitVec):
                return z3.BitVecVal(int(v.value), v.ty.width)
            if v.ty == FLOAT:
                return z3.RealVal(str(v.value))
            if v.ty == STR:
                return z3.StringVal(str(v.value))
            return z3.Const(f"const_{v.value}", self.sort_of(v.ty))

        if isinstance(v, ParamVal):
            if v.name in env:
                return env[v.name]
            key = (v.name, str(v.ty))
            if key not in self._params:
                if isinstance(v.ty, TPtr):
                    assert self._ptr_ctor is not None
                    b = self._base_const(f"param:{v.name}")
                    off = z3.Int(f"{v.name}_off")
                    self._params[key] = self._ptr_ctor(b, off)
                else:
                    self._params[key] = z3.Const(v.name, self.sort_of(v.ty))
            return self._params[key]

        if isinstance(v, SymVal):
            key = (v.symbol, str(v.ty))
            if key not in self._syms:
                if isinstance(v.ty, TPtr):
                    assert self._ptr_ctor is not None
                    b = self._base_const(f"sym:{v.symbol}")
                    self._syms[key] = self._ptr_ctor(b, z3.IntVal(0))
                else:
                    self._syms[key] = z3.Const(f"sym_{v.symbol}", self.sort_of(v.ty))
            return self._syms[key]

        if isinstance(v, HoleVal):
            return z3.Const(f"hole_{v.hole_id}", self.sort_of(v.ty))

        if isinstance(v, SelectVal):
            c = _as_bool_z3(self.to_z3(v.cond, env, _memo))
            t = self.to_z3(v.then_val, env, _memo)
            e = self.to_z3(v.else_val, env, _memo)
            if t.sort() != e.sort():
                uf = self._get_uf(f"coerce_{e.sort()}_to_{t.sort()}", [e.sort()], t.sort())
                e = uf(e)
            return z3.If(c, t, e)

        if isinstance(v, LoopOutVal):
            # Bounded unrolling translation when a LoopOutVal reaches Z3 directly
            lp = v.loop
            cur_vals = [self.to_z3(a, env, _memo) for a in lp.init_vals]
            for _ in range(lp.unroll_hint):
                step_env = dict(env)
                for p, cv in zip(lp.iter_params, cur_vals):
                    step_env[p.name] = cv
                c_z3 = self.to_z3(lp.cond, step_env)
                next_vals = [self.to_z3(b, step_env) for b in lp.body_next]
                cur_vals = [z3.If(c_z3, nv, cv) for cv, nv in zip(cur_vals, next_vals)]
            return cur_vals[v.index]

        if isinstance(v, OpApp):
            spec = self.reg.get(v.op)
            if spec.is_transparent and len(v.args) == 1:
                return self.to_z3(v.args[0], env, _memo)

            zargs = [self.to_z3(a, env, _memo) for a in v.args]
            if spec.z3_builder is not None:
                return spec.z3_builder(zargs, v.attrs, v.ty, self)
            zk = spec.z3_kind

            if zk == "add":
                res = zargs[0]
                for a in zargs[1:]:
                    res = res + a
                return res
            if zk == "sub":
                return zargs[0] - zargs[1]
            if zk == "neg":
                return -zargs[0]
            if zk == "mul":
                res = zargs[0]
                for a in zargs[1:]:
                    res = res * a
                return res
            if zk == "div":
                return zargs[0] / zargs[1]
            if zk == "recip":
                return z3.RealVal(1) / zargs[0]
            if zk == "and":
                return z3.And(*[_as_bool_z3(a) for a in zargs])
            if zk == "or":
                return z3.Or(*[_as_bool_z3(a) for a in zargs])
            if zk == "not":
                return z3.Not(_as_bool_z3(zargs[0]))
            if zk == "xor":
                res = zargs[0]
                for a in zargs[1:]:
                    res = z3.Xor(res, a)
                return res
            if zk == "bv_add":
                res = zargs[0]
                for a in zargs[1:]:
                    res = res + a
                return res
            if zk == "bv_sub":
                return zargs[0] - zargs[1]
            if zk == "bv_neg":
                return -zargs[0]
            if zk == "bv_mul":
                res = zargs[0]
                for a in zargs[1:]:
                    res = res * a
                return res
            if zk == "bv_xor":
                res = zargs[0]
                for a in zargs[1:]:
                    res = res ^ a
                return res
            if zk == "bv_and":
                res = zargs[0]
                for a in zargs[1:]:
                    res = res & a
                return res
            if zk == "bv_or":
                res = zargs[0]
                for a in zargs[1:]:
                    res = res | a
                return res
            if zk == "bv_not":
                return ~zargs[0]
            if zk == "eq":
                return zargs[0] == zargs[1]
            if zk == "neq":
                return zargs[0] != zargs[1]
            if zk == "lt":
                return zargs[0] < zargs[1]
            if zk == "le":
                return zargs[0] <= zargs[1]
            if zk == "gt":
                return zargs[0] > zargs[1]
            if zk == "ge":
                return zargs[0] >= zargs[1]
            if zk == "str_concat":
                return z3.Concat(*zargs)
            if zk == "str_len":
                return z3.Length(zargs[0])
            if zk == "str_substr":
                return z3.SubString(zargs[0], zargs[1], zargs[2])
            if zk == "ptr_gep":
                assert self._ptr_ctor is not None and self._ptr_base is not None and self._ptr_off is not None
                p, off = zargs[0], zargs[1]
                return self._ptr_ctor(self._ptr_base(p), self._ptr_off(p) + off)
            if zk == "mem_alloc":
                assert self._ptr_ctor is not None
                rid = str(v.attr("region", "anon"))
                return self._ptr_ctor(self._base_const(f"alloc:{rid}"), z3.IntVal(0))
            if zk == "mem_load":
                mem_z, ptr_z = zargs[0], zargs[1]
                # Use typed array projection if elem type != Int
                if v.ty == INT:
                    return z3.Select(mem_z, ptr_z)
                return z3.Select(self._mem_array_for_elem(mem_z, v.ty), ptr_z)
            if zk == "mem_store":
                mem_z, ptr_z, val_z = zargs[0], zargs[1], zargs[2]
                elem_ty = v.args[2].ty
                if elem_ty == INT:
                    return z3.Store(mem_z, ptr_z, val_z)
                arr_t = self._mem_array_for_elem(mem_z, elem_ty)
                return self._wrap_mem_from_elem(z3.Store(arr_t, ptr_z, val_z), elem_ty, v.ty)
            if zk == "arr_load":
                return z3.Select(zargs[0], zargs[1])
            if zk == "arr_store":
                return z3.Store(zargs[0], zargs[1], zargs[2])
            if zk == "struct_mk" and isinstance(v.ty, TStruct):
                _, ctor, _ = self._get_struct_info(v.ty)
                return ctor(*zargs)
            if zk == "struct_get" and isinstance(v.args[0].ty, TStruct):
                _, _, accs = self._get_struct_info(v.args[0].ty)
                fname = str(v.attr("field"))
                return accs[fname](zargs[0])
            if zk == "struct_set" and isinstance(v.ty, TStruct):
                _, ctor, accs = self._get_struct_info(v.ty)
                fname = str(v.attr("field"))
                fields = [
                    zargs[1] if fn == fname else accs[fn](zargs[0])
                    for fn, _ in v.ty.fields
                ]
                return ctor(*fields)

            # Default: rigid uninterpreted function (includes mat.mul, seq.compose, uf.*, py.*, custom dialect ops)
            # If the op has associative-only traits, Z3 still treats binary applications as rigid UF
            # unless our equational anti-unifier already flattened and aligned them!
            attr_suffix = ""
            if v.attrs:
                attr_suffix = "_" + "_".join(f"{k}_{val}" for k, val in v.attrs)
            uf = self._get_uf(v.op + attr_suffix, [a.sort() for a in zargs], self.sort_of(v.ty))
            return uf(*zargs)

        raise ValueError(f"Unsupported CoreVal for Z3 lowering: {type(v)}")

    def prove_equal(
        self,
        lhs: CoreVal,
        rhs: CoreVal,
        assumptions: list[CoreVal] | None = None,
        extra_z3_assumptions: list[z3.BoolRef] | None = None,
        timeout_ms: int = 500,
    ) -> tuple[bool, str | None]:
        """Check whether `assumptions |= lhs == rhs` via Z3 refutation."""
        s = z3.Solver()
        s.set("timeout", timeout_ms)

        all_vals = [lhs, rhs] + (assumptions or [])
        for ax in self.collect_aliasing_axioms(all_vals):
            s.add(ax)
        for cond in (assumptions or []):
            s.add(self.to_z3(cond))
        for zc in (extra_z3_assumptions or []):
            s.add(zc)

        zl = self.to_z3(lhs)
        zr = self.to_z3(rhs)
        if zl.sort() != zr.sort():
            return False, f'sort mismatch: {zl.sort()} vs {zr.sort()}'
        s.add(zl != zr)
        res = s.check()
        if res == z3.unsat:
            return True, None
        if res == z3.sat:
            return False, str(s.model())
        return False, "unknown (timeout)"

    def prove_nonzero(
        self,
        val: CoreVal,
        assumptions: list[CoreVal] | None = None,
        timeout_ms: int = 200,
    ) -> bool:
        """Check whether `assumptions |= val != 0` (used as side-condition for div/mul inverse)."""
        zero = ConstVal(0.0, FLOAT) if val.ty == FLOAT else ConstVal(0, val.ty)
        s = z3.Solver()
        s.set("timeout", timeout_ms)
        for cond in (assumptions or []):
            s.add(self.to_z3(cond))
        s.add(self.to_z3(val) == self.to_z3(zero))
        return s.check() == z3.unsat
