from dataclasses import dataclass

import z3

from simp.ast import (
    ArrLoad,
    ArrStoreExpr,
    ArrStoreStmt,
    Assign,
    BinOp,
    BuiltinCall,
    Const,
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
from simp.types import BOOL, FLOAT, INT, TArray, TPtr, TPyObj, TString, TStruct, Type

# Shared uninterpreted Z3 sort for all host-Python objects
PYOBJ_SORT = z3.DeclareSort("PyObj")


def z3_sort_of(ty: Type) -> z3.SortRef:
    if ty == BOOL:
        return z3.BoolSort()
    if ty == INT:
        return z3.IntSort()
    if ty == FLOAT:
        return z3.RealSort()
    if isinstance(ty, TString):
        return z3.StringSort()
    if isinstance(ty, TPyObj):
        return PYOBJ_SORT
    if isinstance(ty, TArray):
        return z3.ArraySort(z3.IntSort(), z3_sort_of(ty.elem))
    if isinstance(ty, TPtr):
        return z3.IntSort()
    raise ValueError(f"Unsupported sort: {ty}")


@dataclass
class SymState:
    env: dict[str, object]
    ret_val: z3.ExprRef | None = None
    has_returned: z3.BoolRef = None  # type: ignore

    def copy(self) -> "SymState":
        return SymState(dict(self.env), self.ret_val, self.has_returned)


class SymbolicExecutor:
    """
    Denotational translation [[.]]_Z3 from s-imp (+ host-Python extension) into
    hash-consed Z3 DAGs:
      - float -> RealSort, int -> IntSort, bool -> BoolSort
      - array<T> -> McCarthy Array(Int, [[T]])
      - ptr<T> -> (heap_array_name, offset_int)
      - pyobj -> uninterpreted PYOBJ_SORT
      - host-Python imports, attributes, calls, methods, box/unbox -> rigid Z3 UFs
      - transcendental builtins -> opaque Z3 UFs
      - while loops & recursion -> bounded unrolling + modular UF summary at depth cap
    """

    def __init__(self, prog: Program, rec_depth: int = 4):
        self.prog = prog
        self.rec_depth = rec_depth
        self.import_map: dict[str, str] = {imp.alias: imp.module for imp in prog.imports}
        self.uf_builtins: dict[str, z3.FuncDeclRef] = {
            name: z3.Function(f"uf_{name}", z3.RealSort(), z3.RealSort())
            for name in ("exp", "log", "sin", "cos", "sqrt")
        }
        self.uf_rec_summaries: dict[str, z3.FuncDeclRef] = {}
        self.uf_py_rigid: dict[tuple, z3.FuncDeclRef] = {}
        self.py_mod_consts: dict[str, z3.ExprRef] = {}
        self.struct_z3: dict[str, tuple[z3.DatatypeSortRef, z3.FuncDeclRef, dict[str, z3.FuncDeclRef], tuple]] = {}
        self._init_structs()

    def share_uf_tables_from(self, other: "SymbolicExecutor"):
        self.uf_builtins = other.uf_builtins
        self.uf_rec_summaries = other.uf_rec_summaries
        self.uf_py_rigid = other.uf_py_rigid
        self.py_mod_consts = other.py_mod_consts
        self.struct_z3 = other.struct_z3


    def _sort_of(self, ty: Type) -> z3.SortRef:
        if isinstance(ty, TStruct):
            return self.struct_z3[ty.name][0]
        return z3_sort_of(ty)

    def _init_structs(self):
        for sd in self.prog.structs:
            if sd.name in self.struct_z3:
                continue
            dt = z3.Datatype(sd.name)
            fields_spec = [(f"{sd.name}_{fname}", self._sort_of(fty)) for fname, fty in sd.fields]
            dt.declare(f"mk_{sd.name}", *fields_spec)
            dt_sort = dt.create()
            ctor = getattr(dt_sort, f"mk_{sd.name}")
            accessors = {fname: getattr(dt_sort, f"{sd.name}_{fname}") for fname, _ in sd.fields}
            self.struct_z3[sd.name] = (dt_sort, ctor, accessors, sd.fields)

    def _get_rec_uf(self, fn: FuncDecl) -> z3.FuncDeclRef:
        if fn.name not in self.uf_rec_summaries:
            arg_sorts = [self._sort_of(t) for _, t in fn.params]
            self.uf_rec_summaries[fn.name] = z3.Function(
                f"uf_rec_{fn.name}", *arg_sorts, self._sort_of(fn.ret_ty)
            )
        return self.uf_rec_summaries[fn.name]

    def _get_py_uf(self, key: tuple, fname: str, arg_sorts: list[z3.SortRef], ret_sort: z3.SortRef) -> z3.FuncDeclRef:
        if key not in self.uf_py_rigid:
            self.uf_py_rigid[key] = z3.Function(fname, *arg_sorts, ret_sort)
        return self.uf_py_rigid[key]

    def make_initial_inputs(self, entry_name: str) -> tuple[dict[str, object], list[z3.BoolRef]]:
        fn = self.prog.get_func(entry_name)
        env: dict[str, object] = {}
        preconds: list[z3.BoolRef] = []
        for pname, pty in fn.params:
            if isinstance(pty, TPtr):
                heap_name = f"__heap_{pname}"
                env[heap_name] = z3.Array(heap_name, z3.IntSort(), z3_sort_of(pty.elem))
                off = z3.Int(f"{pname}_off")
                env[pname] = ("ptr", heap_name, off)
            elif isinstance(pty, TArray):
                env[pname] = z3.Array(pname, z3.IntSort(), z3_sort_of(pty.elem))
            elif isinstance(pty, TPyObj):
                env[pname] = z3.Const(pname, PYOBJ_SORT)
            elif pty == BOOL:
                env[pname] = z3.Bool(pname)
            elif pty == INT:
                env[pname] = z3.Int(pname)
            elif pty == FLOAT:
                env[pname] = z3.Real(pname)
            elif isinstance(pty, TString):
                env[pname] = z3.String(pname)
            elif isinstance(pty, TStruct):
                env[pname] = z3.Const(pname, self._sort_of(pty))
        return env, preconds

    def lower_function(
        self,
        fn_name: str,
        init_env: dict[str, object],
        depth: int = 0,
    ) -> tuple[z3.ExprRef, dict[str, object]]:
        fn = self.prog.get_func(fn_name)
        st = SymState(dict(init_env), ret_val=None, has_returned=z3.BoolVal(False))
        st = self._exec_stmts(fn.body, st, depth)
        assert st.ret_val is not None, f"Function {fn_name} did not return a value"
        return st.ret_val, st.env

    def _eval_expr(self, e: Expr, st: SymState, depth: int):
        if isinstance(e, Const):
            if e.ty == BOOL:
                return z3.BoolVal(bool(e.value))
            if e.ty == INT:
                return z3.IntVal(int(e.value))
            if e.ty == FLOAT:
                return z3.RealVal(str(e.value))
            if isinstance(e.ty, TString):
                return z3.StringVal(str(e.value))
        if isinstance(e, Var):
            return st.env[e.name]
        if isinstance(e, UnOp):
            a = self._eval_expr(e.arg, st, depth)
            return z3.Not(a) if e.op == "!" else (-a)
        if isinstance(e, BinOp):
            l = self._eval_expr(e.left, st, depth)
            r = self._eval_expr(e.right, st, depth)
            ops = {
                "+": lambda a, b: a + b,
                "-": lambda a, b: a - b,
                "*": lambda a, b: a * b,
                "/": lambda a, b: a / b,
                "==": lambda a, b: a == b,
                "!=": lambda a, b: a != b,
                "<": lambda a, b: a < b,
                "<=": lambda a, b: a <= b,
                ">": lambda a, b: a > b,
                ">=": lambda a, b: a >= b,
                "&&": lambda a, b: z3.And(a, b),
                "||": lambda a, b: z3.Or(a, b),
            }
            return ops[e.op](l, r)
        if isinstance(e, IteExpr):
            c = self._eval_expr(e.cond, st, depth)
            t = self._eval_expr(e.then_expr, st, depth)
            f = self._eval_expr(e.else_expr, st, depth)
            return z3.If(c, t, f)
        if isinstance(e, ArrLoad):
            a = self._eval_expr(e.arr, st, depth)
            i = self._eval_expr(e.index, st, depth)
            return z3.Select(a, i)
        if isinstance(e, ArrStoreExpr):
            a = self._eval_expr(e.arr, st, depth)
            i = self._eval_expr(e.index, st, depth)
            v = self._eval_expr(e.val, st, depth)
            return z3.Store(a, i, v)
        if isinstance(e, PtrOffset):
            tag, heap_name, off = self._eval_expr(e.ptr, st, depth)
            delta = self._eval_expr(e.offset, st, depth)
            return (tag, heap_name, off + delta)
        if isinstance(e, PtrDeref):
            _, heap_name, off = self._eval_expr(e.ptr, st, depth)
            return z3.Select(st.env[heap_name], off)
        if isinstance(e, BuiltinCall):
            args = [self._eval_expr(a, st, depth) for a in e.args]
            return self.uf_builtins[e.name](*args)
        if isinstance(e, FuncCall):
            callee = self.prog.get_func(e.name)
            args = [self._eval_expr(a, st, depth) for a in e.args]
            if depth >= self.rec_depth:
                return self._get_rec_uf(callee)(*args)
            sub_env: dict[str, object] = {}
            for (pname, _), aval in zip(callee.params, args):
                sub_env[pname] = aval
            ret, _ = self.lower_function(e.name, sub_env, depth + 1)
            return ret
        # ---- Host-Python denotation as rigid Z3 UFs ----
        if isinstance(e, PyModuleRef):
            mod_name = self.import_map[e.alias]
            if mod_name not in self.py_mod_consts:
                self.py_mod_consts[mod_name] = z3.Const(f"py_mod_{mod_name}", PYOBJ_SORT)
            return self.py_mod_consts[mod_name]
        if isinstance(e, PyAttr):
            ov = self._eval_expr(e.obj, st, depth)
            rs = self._sort_of(e.ret_ty)
            uf = self._get_py_uf(("attr", e.attr, str(rs)), f"py_getattr_{e.attr}", [PYOBJ_SORT], rs)
            return uf(ov)
        if isinstance(e, PyBox):
            v = self._eval_expr(e.expr, st, depth)
            s_in = v.sort()
            uf = self._get_py_uf(("box", str(s_in)), f"py_box_{s_in}", [s_in], PYOBJ_SORT)
            return uf(v)
        if isinstance(e, PyUnbox):
            ov = self._eval_expr(e.expr, st, depth)
            rs = self._sort_of(e.target_ty)
            uf = self._get_py_uf(("unbox", str(rs)), f"py_unbox_{rs}", [PYOBJ_SORT], rs)
            return uf(ov)
        if isinstance(e, PyCall):
            cv = self._eval_expr(e.callee, st, depth)
            avs = [self._eval_expr(a, st, depth) for a in e.args]
            arg_sorts = [PYOBJ_SORT] + [a.sort() for a in avs]
            rs = self._sort_of(e.ret_ty)
            sig = tuple(str(s) for s in arg_sorts) + (str(rs),)
            uf = self._get_py_uf(("call", sig), f"py_call_{len(avs)}_{rs}", arg_sorts, rs)
            return uf(cv, *avs)
        if isinstance(e, PyMethodCall):
            ov = self._eval_expr(e.obj, st, depth)
            avs = [self._eval_expr(a, st, depth) for a in e.args]
            arg_sorts = [PYOBJ_SORT] + [a.sort() for a in avs]
            rs = self._sort_of(e.ret_ty)
            sig = (e.method,) + tuple(str(s) for s in arg_sorts) + (str(rs),)
            uf = self._get_py_uf(("method", sig), f"py_method_{e.method}_{rs}", arg_sorts, rs)
            return uf(ov, *avs)

        # ---- Native Strings & Structs ----
        if isinstance(e, StrConcat):
            return z3.Concat(self._eval_expr(e.left, st, depth), self._eval_expr(e.right, st, depth))
        if isinstance(e, StrLen):
            return z3.Length(self._eval_expr(e.arg, st, depth))
        if isinstance(e, StrSubstr):
            return z3.SubString(
                self._eval_expr(e.s, st, depth),
                self._eval_expr(e.offset, st, depth),
                self._eval_expr(e.length, st, depth),
            )
        if isinstance(e, StructInit):
            _, ctor, _, _ = self.struct_z3[e.struct_name]
            vals = [self._eval_expr(v, st, depth) for _, v in e.field_vals]
            return ctor(*vals)
        if isinstance(e, FieldGet):
            tv = self._eval_expr(e.target, st, depth)
            sname = tv.sort().name()
            _, _, accessors, _ = self.struct_z3[sname]
            return accessors[e.field](tv)
        if isinstance(e, FieldSet):
            tv = self._eval_expr(e.target, st, depth)
            nv = self._eval_expr(e.val, st, depth)
            sname = tv.sort().name()
            _, ctor, accessors, fields = self.struct_z3[sname]
            args = [nv if fnm == e.field else accessors[fnm](tv) for fnm, _ in fields]
            return ctor(*args)
        raise ValueError(f"Unhandled expr in symexec: {e}")

    def _merge_val(self, cond: z3.BoolRef, v_then, v_else):
        if v_then is None:
            return v_else
        if v_else is None:
            return v_then
        if isinstance(v_then, tuple) and v_then[0] == "ptr":
            return ("ptr", v_then[1], z3.If(cond, v_then[2], v_else[2]))
        if isinstance(v_then, z3.ExprRef) and isinstance(v_else, z3.ExprRef):
            if v_then.get_id() == v_else.get_id():
                return v_then
            return z3.If(cond, v_then, v_else)
        return v_then

    def _merge_states(self, cond: z3.BoolRef, st_t: SymState, st_e: SymState) -> SymState:
        merged_env: dict[str, object] = {}
        keys = set(st_t.env.keys()) & set(st_e.env.keys())
        for k in keys:
            merged_env[k] = self._merge_val(cond, st_t.env[k], st_e.env[k])
        ret = self._merge_val(cond, st_t.ret_val, st_e.ret_val)
        hr = z3.If(cond, st_t.has_returned, st_e.has_returned)
        return SymState(merged_env, ret, hr)

    def _exec_stmts(self, stmts: list[Stmt], st: SymState, depth: int) -> SymState:
        for stmt in stmts:
            if isinstance(stmt, LetDecl):
                val = self._eval_expr(stmt.init, st, depth)
                st.env[stmt.name] = val
            elif isinstance(stmt, Assign):
                val = self._eval_expr(stmt.expr, st, depth)
                old = st.env[stmt.name]
                if z3.is_false(st.has_returned):
                    st.env[stmt.name] = val
                else:
                    st.env[stmt.name] = self._merge_val(st.has_returned, old, val)
            elif isinstance(stmt, ArrStoreStmt):
                idx = self._eval_expr(stmt.index, st, depth)
                val = self._eval_expr(stmt.val, st, depth)
                old_arr = st.env[stmt.arr_name]
                new_arr = z3.Store(old_arr, idx, val)
                if z3.is_false(st.has_returned):
                    st.env[stmt.arr_name] = new_arr
                else:
                    st.env[stmt.arr_name] = z3.If(st.has_returned, old_arr, new_arr)
            elif isinstance(stmt, PtrStoreStmt):
                _, heap_name, off = self._eval_expr(stmt.ptr_expr, st, depth)
                val = self._eval_expr(stmt.val, st, depth)
                old_arr = st.env[heap_name]
                new_arr = z3.Store(old_arr, off, val)
                if z3.is_false(st.has_returned):
                    st.env[heap_name] = new_arr
                else:
                    st.env[heap_name] = z3.If(st.has_returned, old_arr, new_arr)
            elif isinstance(stmt, IfStmt):
                cond = self._eval_expr(stmt.cond, st, depth)
                st_t = self._exec_stmts(stmt.then_body, st.copy(), depth)
                st_e = self._exec_stmts(stmt.else_body, st.copy(), depth)
                st = self._merge_states(cond, st_t, st_e)
            elif isinstance(stmt, WhileStmt):
                for _ in range(stmt.unroll_bound):
                    cond = self._eval_expr(stmt.cond, st, depth)
                    st_body = self._exec_stmts(stmt.body, st.copy(), depth)
                    st = self._merge_states(cond, st_body, st)
            elif isinstance(stmt, ReturnStmt):
                val = self._eval_expr(stmt.expr, st, depth)
                if st.ret_val is None:
                    st.ret_val = val
                else:
                    st.ret_val = self._merge_val(st.has_returned, st.ret_val, val)
                st.has_returned = z3.BoolVal(True)
        return st
