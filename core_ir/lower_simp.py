from simp.simp_ast import (
    ArrLoad,
    ArrStoreExpr,
    ArrStoreStmt,
    Assign,
    BinOp,
    BuiltinCall,
    Const,
    DictInit,
    Expr,
    FieldGet,
    FieldSet,
    FuncCall,
    FuncDecl,
    IfStmt,
    IteExpr,
    LetDecl,
    Program,
    PtrDeref,
    PtrOffset,
    PtrStoreStmt,
    PyAttr,
    PyBox,
    PyCall,
    PyMethodCall,
    PyModuleRef,
    PyUnbox,
    ReturnStmt,
    Stmt,
    StrConcat,
    StrLen,
    StrSubstr,
    StructInit,
    UnOp,
    Var,
    WhileStmt,
)
from simp.simp_types import (
    BOOL as SIMP_BOOL,
    FLOAT as SIMP_FLOAT,
    INT as SIMP_INT,
    STR as SIMP_STR,
    TArray as SimpArray,
    TDict as SimpDict,
    TPtr as SimpPtr,
    TPyObj as SimpPyObj,
    TStruct as SimpStruct,
    Type as SimpType,
)

from .ir import (
    ConstVal,
    CoreFunc,
    CoreModule,
    CoreVal,
    LoopOutVal,
    LoopVal,
    OpApp,
    ParamVal,
    SelectVal,
    SymVal,
)
from .core_types import (
    BOOL,
    FLOAT,
    INT,
    MEM,
    PYOBJ,
    STR,
    CoreType,
    TArray,
    TPtr,
    TStruct,
    TSym,
    TTuple,
)

_BINOP_MAP = {
    "+": "add",
    "-": "sub",
    "*": "mul",
    "/": "div",
    "&&": "and",
    "||": "or",
    "==": "eq",
    "!=": "neq",
    "<": "lt",
    "<=": "le",
    ">": "gt",
    ">=": "ge",
}

_UNOP_MAP = {
    "-": "neg",
    "!": "not",
}


def lower_simp_type(ty: SimpType) -> CoreType:
    if ty == SIMP_BOOL:
        return BOOL
    if ty == SIMP_INT:
        return INT
    if ty == SIMP_FLOAT:
        return FLOAT
    if ty == SIMP_STR:
        return STR
    if isinstance(ty, SimpPyObj):
        return TSym(f"PyObj_{ty.tag}") if ty.tag != "object" else PYOBJ
    if isinstance(ty, SimpPtr):
        return TPtr(lower_simp_type(ty.elem), tbaa_tag=str(ty.elem))
    if isinstance(ty, SimpArray):
        return TArray(INT, lower_simp_type(ty.elem))
    if isinstance(ty, SimpDict):
        return TArray(lower_simp_type(ty.key), lower_simp_type(ty.val))
    if isinstance(ty, SimpStruct):
        return TStruct(ty.name, tuple((k, lower_simp_type(ft)) for k, ft in ty.fields))
    return INT


class SimpToCoreLowerer:
    """Lowers structured `s-imp` programs into the minimal CoreSSA-IR kernel with
    explicit Memory-SSA threading and structured theta-loop regions.
    """

    def __init__(self, prog: Program):
        self.prog = prog
        self.structs: dict[str, TStruct] = {
            sd.name: lower_simp_type(sd.as_type())  # type: ignore[misc]
            for sd in prog.structs
        }
        self._loop_counter = 0

    def _fresh_loop_id(self) -> str:
        lid = f"L{self._loop_counter}"
        self._loop_counter += 1
        return lid

    def _infer_binop_ty(self, op: str, lhs_ty: CoreType) -> CoreType:
        if op in ("==", "!=", "<", "<=", ">", ">=", "&&", "||"):
            return BOOL
        return lhs_ty

    def lower_expr(self, e: Expr, env: dict[str, CoreVal], mem: CoreVal) -> CoreVal:
        if isinstance(e, Const):
            return ConstVal(e.value, lower_simp_type(e.ty))
        if isinstance(e, Var):
            return env[e.name]
        if isinstance(e, UnOp):
            arg = self.lower_expr(e.arg, env, mem)
            return OpApp(_UNOP_MAP[e.op], (arg,), arg.ty)
        if isinstance(e, BinOp):
            l = self.lower_expr(e.left, env, mem)
            r = self.lower_expr(e.right, env, mem)
            return OpApp(_BINOP_MAP[e.op], (l, r), self._infer_binop_ty(e.op, l.ty))
        if isinstance(e, IteExpr):
            c = self.lower_expr(e.cond, env, mem)
            t = self.lower_expr(e.then_expr, env, mem)
            el = self.lower_expr(e.else_expr, env, mem)
            return SelectVal(c, t, el, t.ty)
        if isinstance(e, ArrLoad):
            arr = self.lower_expr(e.arr, env, mem)
            idx = self.lower_expr(e.index, env, mem)
            elem_ty = arr.ty.elem if isinstance(arr.ty, TArray) else INT
            return OpApp("arr.load", (arr, idx), elem_ty)
        if isinstance(e, ArrStoreExpr):
            arr = self.lower_expr(e.arr, env, mem)
            idx = self.lower_expr(e.index, env, mem)
            val = self.lower_expr(e.val, env, mem)
            return OpApp("arr.store", (arr, idx, val), arr.ty)
        if isinstance(e, PtrOffset):
            p = self.lower_expr(e.ptr, env, mem)
            off = self.lower_expr(e.offset, env, mem)
            return OpApp("ptr.gep", (p, off), p.ty)
        if isinstance(e, PtrDeref):
            p = self.lower_expr(e.ptr, env, mem)
            elem_ty = p.ty.elem if isinstance(p.ty, TPtr) else INT
            return OpApp("mem.load", (mem, p), elem_ty)
        if isinstance(e, StrConcat):
            l = self.lower_expr(e.left, env, mem)
            r = self.lower_expr(e.right, env, mem)
            return OpApp("str.concat", (l, r), STR)
        if isinstance(e, StrLen):
            a = self.lower_expr(e.arg, env, mem)
            return OpApp("str.len", (a,), INT)
        if isinstance(e, StrSubstr):
            s = self.lower_expr(e.s, env, mem)
            o = self.lower_expr(e.offset, env, mem)
            ln = self.lower_expr(e.length, env, mem)
            return OpApp("str.substr", (s, o, ln), STR)
        if isinstance(e, StructInit):
            st_ty = self.structs[e.struct_name]
            fvals = tuple(self.lower_expr(v, env, mem) for _, v in e.field_vals)
            return OpApp("struct.mk", fvals, st_ty, attrs=(("struct", e.struct_name),))
        if isinstance(e, FieldGet):
            tgt = self.lower_expr(e.target, env, mem)
            fty = tgt.ty.field_type(e.field) if isinstance(tgt.ty, TStruct) else INT
            return OpApp("struct.get", (tgt,), fty, attrs=(("field", e.field),))
        if isinstance(e, FieldSet):
            tgt = self.lower_expr(e.target, env, mem)
            val = self.lower_expr(e.val, env, mem)
            return OpApp("struct.set", (tgt, val), tgt.ty, attrs=(("field", e.field),))
        if isinstance(e, DictInit):
            dty = TArray(lower_simp_type(e.key_ty), lower_simp_type(e.val_ty))
            cur: CoreVal = SymVal(f"empty_dict_{dty.key}_{dty.elem}", dty)
            for k, v in e.entries:
                cur = OpApp("arr.store", (cur, self.lower_expr(k, env, mem), self.lower_expr(v, env, mem)), dty)
            return cur
        if isinstance(e, BuiltinCall):
            args = tuple(self.lower_expr(a, env, mem) for a in e.args)
            return OpApp(f"uf.{e.name}", args, FLOAT)
        if isinstance(e, FuncCall):
            args = tuple(self.lower_expr(a, env, mem) for a in e.args)
            callee = self.prog.get_func(e.name)
            return OpApp(f"fn.{e.name}", args, lower_simp_type(callee.ret_ty))
        if isinstance(e, PyModuleRef):
            return SymVal(f"pymod:{e.alias}", PYOBJ)
        if isinstance(e, PyAttr):
            obj = self.lower_expr(e.obj, env, mem)
            return OpApp("py.attr", (obj,), lower_simp_type(e.ret_ty), attrs=(("attr", e.attr),))
        if isinstance(e, PyBox):
            return OpApp("py.box", (self.lower_expr(e.expr, env, mem),), PYOBJ)
        if isinstance(e, PyUnbox):
            return OpApp("py.unbox", (self.lower_expr(e.expr, env, mem),), lower_simp_type(e.target_ty))
        if isinstance(e, PyCall):
            callee = self.lower_expr(e.callee, env, mem)
            args = tuple(self.lower_expr(a, env, mem) for a in e.args)
            return OpApp("py.call", (callee,) + args, lower_simp_type(e.ret_ty))
        if isinstance(e, PyMethodCall):
            obj = self.lower_expr(e.obj, env, mem)
            args = tuple(self.lower_expr(a, env, mem) for a in e.args)
            return OpApp("py.method", (obj,) + args, lower_simp_type(e.ret_ty), attrs=(("method", e.method),))
        raise ValueError(f"Unsupported s-imp expr: {type(e)}")

    def _assigned_vars(self, stmts: list[Stmt]) -> tuple[set[str], bool]:
        """Collect all variable names assigned in `stmts` and whether memory is written."""
        names: set[str] = set()
        writes_mem = False
        for st in stmts:
            if isinstance(st, Assign):
                names.add(st.name)
            elif isinstance(st, ArrStoreStmt):
                names.add(st.arr_name)
            elif isinstance(st, PtrStoreStmt):
                writes_mem = True
            elif isinstance(st, IfStmt):
                nt, mt = self._assigned_vars(st.then_body)
                ne, me = self._assigned_vars(st.else_body)
                names |= nt | ne
                writes_mem = writes_mem or mt or me
            elif isinstance(st, WhileStmt):
                nb, mb = self._assigned_vars(st.body)
                names |= nb
                writes_mem = writes_mem or mb
        return names, writes_mem

    def lower_stmts(
        self, stmts: list[Stmt], env: dict[str, CoreVal], mem: CoreVal
    ) -> tuple[dict[str, CoreVal], CoreVal, CoreVal | None]:
        cur_env = dict(env)
        cur_mem = mem
        ret_val: CoreVal | None = None

        for st in stmts:
            if isinstance(st, LetDecl):
                cur_env[st.name] = self.lower_expr(st.init, cur_env, cur_mem)
            elif isinstance(st, Assign):
                cur_env[st.name] = self.lower_expr(st.expr, cur_env, cur_mem)
            elif isinstance(st, ArrStoreStmt):
                idx = self.lower_expr(st.index, cur_env, cur_mem)
                val = self.lower_expr(st.val, cur_env, cur_mem)
                arr = cur_env[st.arr_name]
                cur_env[st.arr_name] = OpApp("arr.store", (arr, idx, val), arr.ty)
            elif isinstance(st, PtrStoreStmt):
                ptr = self.lower_expr(st.ptr_expr, cur_env, cur_mem)
                val = self.lower_expr(st.val, cur_env, cur_mem)
                cur_mem = OpApp("mem.store", (cur_mem, ptr, val), MEM)
            elif isinstance(st, IfStmt):
                cond = self.lower_expr(st.cond, cur_env, cur_mem)
                env_t, mem_t, ret_t = self.lower_stmts(st.then_body, cur_env, cur_mem)
                env_e, mem_e, ret_e = self.lower_stmts(st.else_body, cur_env, cur_mem)
                for k in list(cur_env.keys()):
                    vt, ve = env_t[k], env_e[k]
                    cur_env[k] = vt if vt == ve else SelectVal(cond, vt, ve, vt.ty)
                cur_mem = mem_t if mem_t == mem_e else SelectVal(cond, mem_t, mem_e, MEM)
                if ret_t is not None or ret_e is not None:
                    rt = ret_t if ret_t is not None else ret_val
                    re = ret_e if ret_e is not None else ret_val
                    if rt is not None and re is not None:
                        ret_val = rt if rt == re else SelectVal(cond, rt, re, rt.ty)
                    else:
                        ret_val = rt if rt is not None else re
            elif isinstance(st, WhileStmt):
                # Lower while loop into an exact unbounded SSA theta-loop!
                mod_vars, writes_mem = self._assigned_vars(st.body)
                carried_names = sorted(k for k in mod_vars if k in cur_env)
                iter_params: list[ParamVal] = [ParamVal(k, cur_env[k].ty) for k in carried_names]
                init_vals: list[CoreVal] = [cur_env[k] for k in carried_names]
                if writes_mem:
                    iter_params.append(ParamVal("__mem", MEM))
                    init_vals.append(cur_mem)

                body_env = dict(cur_env)
                for p in iter_params:
                    if p.name != "__mem":
                        body_env[p.name] = p
                body_mem: CoreVal = iter_params[-1] if writes_mem else cur_mem

                cond_val = self.lower_expr(st.cond, body_env, body_mem)
                next_env, next_mem, _ = self.lower_stmts(st.body, body_env, body_mem)
                body_next: list[CoreVal] = [next_env[k] for k in carried_names]
                if writes_mem:
                    body_next.append(next_mem)

                lp = LoopVal(
                    iter_params=tuple(iter_params),
                    init_vals=tuple(init_vals),
                    cond=cond_val,
                    body_next=tuple(body_next),
                    ty=TTuple(tuple(p.ty for p in iter_params)),
                    unroll_hint=st.unroll_bound,
                    loop_id=self._fresh_loop_id(),
                )
                for i, k in enumerate(carried_names):
                    cur_env[k] = LoopOutVal(lp, i, iter_params[i].ty)
                if writes_mem:
                    cur_mem = LoopOutVal(lp, len(carried_names), MEM)
            elif isinstance(st, ReturnStmt):
                ret_val = self.lower_expr(st.expr, cur_env, cur_mem)

        return cur_env, cur_mem, ret_val

    def lower_func(self, fn: FuncDecl) -> CoreFunc:
        params = [ParamVal(p, lower_simp_type(t)) for p, t in fn.params]
        mem0 = ParamVal("__mem0", MEM)
        env: dict[str, CoreVal] = {p.name: p for p in params}
        _, mem_out, ret_val = self.lower_stmts(fn.body, env, mem0)
        assert ret_val is not None, f"Function {fn.name} has no return value"
        return CoreFunc(
            name=fn.name,
            params=params + [mem0],
            ret_ty=lower_simp_type(fn.ret_ty),
            ret_val=ret_val,
            mem_out=mem_out,
        )

    def lower_program(self) -> CoreModule:
        mod = CoreModule(structs=dict(self.structs))
        for fn in self.prog.funcs:
            mod.add_func(self.lower_func(fn))
        return mod
