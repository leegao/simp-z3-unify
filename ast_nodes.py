from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from .types import BOOL, FLOAT, INT, PYOBJ, STR, TArray, TPtr, TPyObj, TString, TStruct, Type, BUILTIN_SIGS


class Expr:
    pass


@dataclass(frozen=True)
class Const(Expr):
    value: Union[bool, int, float, str]
    ty: Type


@dataclass(frozen=True)
class Var(Expr):
    name: str


@dataclass(frozen=True)
class UnOp(Expr):
    op: str  # "-", "!"
    arg: Expr


@dataclass(frozen=True)
class BinOp(Expr):
    op: str  # "+", "-", "*", "/", "==", "!=", "<", "<=", ">", ">=", "&&", "||"
    left: Expr
    right: Expr


@dataclass(frozen=True)
class IteExpr(Expr):
    cond: Expr
    then_expr: Expr
    else_expr: Expr


@dataclass(frozen=True)
class ArrLoad(Expr):
    arr: Expr
    index: Expr


@dataclass(frozen=True)
class ArrStoreExpr(Expr):
    arr: Expr
    index: Expr
    val: Expr


@dataclass(frozen=True)
class PtrOffset(Expr):
    ptr: Expr
    offset: Expr


@dataclass(frozen=True)
class PtrDeref(Expr):
    ptr: Expr


@dataclass(frozen=True)
class BuiltinCall(Expr):
    name: str  # exp, log, sin, cos, sqrt
    args: Tuple[Expr, ...]


@dataclass(frozen=True)
class FuncCall(Expr):
    name: str
    args: Tuple[Expr, ...]


# ---- Host-Python Extension (all modeled as rigid uninterpreted functions) ----

@dataclass(frozen=True)
class PyImportDecl:
    """Host-Python module import: `import <module> as <alias>`."""
    module: str
    alias: str


@dataclass(frozen=True)
class PyModuleRef(Expr):
    """Reference to an imported host-Python module handle (constant of sort PyObj)."""
    alias: str


@dataclass(frozen=True)
class PyAttr(Expr):
    """Rigid attribute projection: `obj.<attr>` -> PyObj."""
    obj: Expr
    attr: str
    ret_ty: Type = PYOBJ


@dataclass(frozen=True)
class PyCall(Expr):
    """Rigid host-Python function call: `callee(args...) : ret_ty`."""
    callee: Expr
    args: Tuple[Expr, ...]
    ret_ty: Type = PYOBJ


@dataclass(frozen=True)
class PyMethodCall(Expr):
    """Rigid host-Python method invocation: `obj.<method>(args...) : ret_ty`."""
    obj: Expr
    method: str
    args: Tuple[Expr, ...]
    ret_ty: Type = PYOBJ


@dataclass(frozen=True)
class PyBox(Expr):
    """Rigid boxing bridge: primitive s-imp value -> PyObj."""
    expr: Expr


@dataclass(frozen=True)
class PyUnbox(Expr):
    """Rigid unboxing bridge: PyObj -> primitive s-imp type."""
    expr: Expr
    target_ty: Type



# ---- Native Strings & Structs ----

@dataclass(frozen=True)
class StrConcat(Expr):
    left: Expr
    right: Expr


@dataclass(frozen=True)
class StrLen(Expr):
    arg: Expr


@dataclass(frozen=True)
class StrSubstr(Expr):
    s: Expr
    offset: Expr
    length: Expr


@dataclass(frozen=True)
class StructDecl:
    name: str
    fields: Tuple[Tuple[str, Type], ...]

    def as_type(self) -> TStruct:
        return TStruct(self.name, self.fields)


@dataclass(frozen=True)
class StructInit(Expr):
    struct_name: str
    field_vals: Tuple[Tuple[str, Expr], ...]


@dataclass(frozen=True)
class FieldGet(Expr):
    target: Expr
    field: str


@dataclass(frozen=True)
class FieldSet(Expr):
    target: Expr
    field: str
    val: Expr


class Stmt:
    pass


@dataclass
class LetDecl(Stmt):
    name: str
    ty: Type
    init: Expr


@dataclass
class Assign(Stmt):
    name: str
    expr: Expr


@dataclass
class ArrStoreStmt(Stmt):
    arr_name: str
    index: Expr
    val: Expr


@dataclass
class PtrStoreStmt(Stmt):
    ptr_expr: Expr
    val: Expr


@dataclass
class IfStmt(Stmt):
    cond: Expr
    then_body: List[Stmt]
    else_body: List[Stmt] = field(default_factory=list)


@dataclass
class WhileStmt(Stmt):
    cond: Expr
    body: List[Stmt]
    unroll_bound: int = 4


@dataclass
class ReturnStmt(Stmt):
    expr: Expr


@dataclass
class FuncDecl:
    name: str
    params: List[Tuple[str, Type]]
    ret_ty: Type
    body: List[Stmt]


@dataclass
class Program:
    funcs: List[FuncDecl]
    imports: List[PyImportDecl] = field(default_factory=list)
    structs: List[StructDecl] = field(default_factory=list)

    def get_func(self, name: str) -> FuncDecl:
        for fn in self.funcs:
            if fn.name == name:
                return fn
        raise KeyError(f"Unknown function: {name}")


class TypeErrorSIMP(Exception):
    pass


def _is_pyobj(t: Type) -> bool:
    return isinstance(t, TPyObj)


def typecheck_program(prog: Program) -> Dict[str, Tuple[List[Type], Type]]:
    func_sigs: Dict[str, Tuple[List[Type], Type]] = {}
    import_aliases = {imp.alias: imp.module for imp in prog.imports}
    struct_defs = {sd.name: sd.as_type() for sd in prog.structs}
    for fn in prog.funcs:
        if fn.name in func_sigs:
            raise TypeErrorSIMP(f"Duplicate function: {fn.name}")
        func_sigs[fn.name] = ([t for _, t in fn.params], fn.ret_ty)

    for fn in prog.funcs:
        env: Dict[str, Type] = {p: t for p, t in fn.params}
        _tc_stmts(fn.body, env, func_sigs, import_aliases, struct_defs, fn.ret_ty)
    return func_sigs


def _types_compat(a: Type, b: Type) -> bool:
    return a == b or (_is_pyobj(a) and _is_pyobj(b))


def _tc_expr(
    e: Expr,
    env: Dict[str, Type],
    fsigs: Dict[str, Tuple[List[Type], Type]],
    imports: Dict[str, str],
    structs: Dict[str, TStruct],
) -> Type:
    if isinstance(e, Const):
        return e.ty
    if isinstance(e, Var):
        if e.name not in env:
            raise TypeErrorSIMP(f"Undefined variable: {e.name}")
        return env[e.name]
    if isinstance(e, UnOp):
        t = _tc_expr(e.arg, env, fsigs, imports, structs)
        if e.op == "!" and t == BOOL:
            return BOOL
        if e.op == "-" and t in (INT, FLOAT):
            return t
        raise TypeErrorSIMP(f"Invalid UnOp {e.op} on {t}")
    if isinstance(e, BinOp):
        tl = _tc_expr(e.left, env, fsigs, imports, structs)
        tr = _tc_expr(e.right, env, fsigs, imports, structs)
        if e.op in ("+", "-", "*", "/"):
            if tl == tr and tl in (INT, FLOAT):
                return tl
            raise TypeErrorSIMP(f"Arithmetic op {e.op} mismatch: {tl} vs {tr}")
        if e.op in ("<", "<=", ">", ">="):
            if tl == tr and tl in (INT, FLOAT):
                return BOOL
            raise TypeErrorSIMP(f"Comparison {e.op} mismatch: {tl} vs {tr}")
        if e.op in ("==", "!="):
            if _types_compat(tl, tr):
                return BOOL
            raise TypeErrorSIMP(f"Equality {e.op} mismatch: {tl} vs {tr}")
        if e.op in ("&&", "||"):
            if tl == BOOL and tr == BOOL:
                return BOOL
            raise TypeErrorSIMP(f"Logical op {e.op} requires bool, got {tl}, {tr}")
        raise TypeErrorSIMP(f"Unknown BinOp {e.op}")
    if isinstance(e, IteExpr):
        tc = _tc_expr(e.cond, env, fsigs, imports, structs)
        tt = _tc_expr(e.then_expr, env, fsigs, imports, structs)
        tf = _tc_expr(e.else_expr, env, fsigs, imports, structs)
        if tc != BOOL or not _types_compat(tt, tf):
            raise TypeErrorSIMP(f"Invalid IteExpr: cond={tc}, then={tt}, else={tf}")
        return tt
    if isinstance(e, ArrLoad):
        ta = _tc_expr(e.arr, env, fsigs, imports, structs)
        ti = _tc_expr(e.index, env, fsigs, imports, structs)
        if not isinstance(ta, TArray) or ti != INT:
            raise TypeErrorSIMP(f"Invalid ArrLoad: {ta}[{ti}]")
        return ta.elem
    if isinstance(e, ArrStoreExpr):
        ta = _tc_expr(e.arr, env, fsigs, imports, structs)
        ti = _tc_expr(e.index, env, fsigs, imports, structs)
        tv = _tc_expr(e.val, env, fsigs, imports, structs)
        if not isinstance(ta, TArray) or ti != INT or not _types_compat(tv, ta.elem):
            raise TypeErrorSIMP(f"Invalid ArrStoreExpr: {ta}[{ti}] = {tv}")
        return ta
    if isinstance(e, PtrOffset):
        tp = _tc_expr(e.ptr, env, fsigs, imports, structs)
        to = _tc_expr(e.offset, env, fsigs, imports, structs)
        if not isinstance(tp, TPtr) or to != INT:
            raise TypeErrorSIMP(f"Invalid PtrOffset: {tp} + {to}")
        return tp
    if isinstance(e, PtrDeref):
        tp = _tc_expr(e.ptr, env, fsigs, imports, structs)
        if not isinstance(tp, TPtr):
            raise TypeErrorSIMP(f"Invalid PtrDeref on {tp}")
        return tp.elem
    if isinstance(e, BuiltinCall):
        if e.name not in BUILTIN_SIGS:
            raise TypeErrorSIMP(f"Unknown builtin: {e.name}")
        ptys, rty = BUILTIN_SIGS[e.name]
        if len(e.args) != len(ptys):
            raise TypeErrorSIMP(f"Arity mismatch for {e.name}")
        for a, pt in zip(e.args, ptys):
            at = _tc_expr(a, env, fsigs, imports, structs)
            if at != pt:
                raise TypeErrorSIMP(f"Arg type mismatch in {e.name}: {at} vs {pt}")
        return rty
    if isinstance(e, FuncCall):
        if e.name not in fsigs:
            raise TypeErrorSIMP(f"Unknown function: {e.name}")
        ptys, rty = fsigs[e.name]
        if len(e.args) != len(ptys):
            raise TypeErrorSIMP(f"Arity mismatch for {e.name}")
        for a, pt in zip(e.args, ptys):
            at = _tc_expr(a, env, fsigs, imports, structs)
            if not _types_compat(at, pt):
                raise TypeErrorSIMP(f"Arg type mismatch in {e.name}: {at} vs {pt}")
        return rty
    # ---- Host-Python typing rules ----
    if isinstance(e, PyModuleRef):
        if e.alias not in imports:
            raise TypeErrorSIMP(f"Unimported host-Python module alias: {e.alias}")
        return TPyObj(imports[e.alias])
    if isinstance(e, PyAttr):
        to = _tc_expr(e.obj, env, fsigs, imports, structs)
        if not _is_pyobj(to):
            raise TypeErrorSIMP(f"PyAttr requires pyobj receiver, got {to}")
        return e.ret_ty
    if isinstance(e, PyCall):
        tc = _tc_expr(e.callee, env, fsigs, imports, structs)
        if not _is_pyobj(tc):
            raise TypeErrorSIMP(f"PyCall requires pyobj callee, got {tc}")
        for a in e.args:
            _tc_expr(a, env, fsigs, imports, structs)
        return e.ret_ty
    if isinstance(e, PyMethodCall):
        to = _tc_expr(e.obj, env, fsigs, imports, structs)
        if not _is_pyobj(to):
            raise TypeErrorSIMP(f"PyMethodCall requires pyobj receiver, got {to}")
        for a in e.args:
            _tc_expr(a, env, fsigs, imports, structs)
        return e.ret_ty
    if isinstance(e, PyBox):
        t = _tc_expr(e.expr, env, fsigs, imports, structs)
        if t not in (BOOL, INT, FLOAT):
            raise TypeErrorSIMP(f"PyBox requires primitive type, got {t}")
        return PYOBJ
    if isinstance(e, PyUnbox):
        t = _tc_expr(e.expr, env, fsigs, imports, structs)
        if not _is_pyobj(t) or e.target_ty not in (BOOL, INT, FLOAT):
            raise TypeErrorSIMP(f"PyUnbox requires pyobj -> primitive, got {t} -> {e.target_ty}")
        return e.target_ty

    if isinstance(e, StrConcat):
        if _tc_expr(e.left, env, fsigs, imports, structs) != STR or _tc_expr(e.right, env, fsigs, imports, structs) != STR:
            raise TypeErrorSIMP("StrConcat requires str, str")
        return STR
    if isinstance(e, StrLen):
        if _tc_expr(e.arg, env, fsigs, imports, structs) != STR:
            raise TypeErrorSIMP("StrLen requires str")
        return INT
    if isinstance(e, StrSubstr):
        if (_tc_expr(e.s, env, fsigs, imports, structs) != STR
                or _tc_expr(e.offset, env, fsigs, imports, structs) != INT
                or _tc_expr(e.length, env, fsigs, imports, structs) != INT):
            raise TypeErrorSIMP("StrSubstr requires (str, int, int)")
        return STR
    if isinstance(e, StructInit):
        if e.struct_name not in structs:
            raise TypeErrorSIMP(f"Unknown struct: {e.struct_name}")
        st = structs[e.struct_name]
        if len(e.field_vals) != len(st.fields):
            raise TypeErrorSIMP(f"StructInit {e.struct_name} field count mismatch")
        for (fnm, fexp), (enm, ety) in zip(e.field_vals, st.fields):
            if fnm != enm:
                raise TypeErrorSIMP(f"StructInit {e.struct_name}: expected field {enm}, got {fnm}")
            at = _tc_expr(fexp, env, fsigs, imports, structs)
            if not _types_compat(at, ety):
                raise TypeErrorSIMP(f"StructInit {e.struct_name}.{fnm}: expected {ety}, got {at}")
        return st
    if isinstance(e, FieldGet):
        ts = _tc_expr(e.target, env, fsigs, imports, structs)
        if not isinstance(ts, TStruct):
            raise TypeErrorSIMP(f"FieldGet requires struct, got {ts}")
        return ts.field_type(e.field)
    if isinstance(e, FieldSet):
        ts = _tc_expr(e.target, env, fsigs, imports, structs)
        if not isinstance(ts, TStruct):
            raise TypeErrorSIMP(f"FieldSet requires struct, got {ts}")
        ft = ts.field_type(e.field)
        vt = _tc_expr(e.val, env, fsigs, imports, structs)
        if not _types_compat(vt, ft):
            raise TypeErrorSIMP(f"FieldSet {ts.name}.{e.field}: expected {ft}, got {vt}")
        return ts
    raise TypeErrorSIMP(f"Unhandled expr: {e}")


def _tc_stmts(
    stmts: List[Stmt],
    env: Dict[str, Type],
    fsigs: Dict[str, Tuple[List[Type], Type]],
    imports: Dict[str, str],
    structs: Dict[str, TStruct],
    ret_ty: Type,
):
    for st in stmts:
        if isinstance(st, LetDecl):
            it = _tc_expr(st.init, env, fsigs, imports, structs)
            if not _types_compat(it, st.ty):
                raise TypeErrorSIMP(f"LetDecl {st.name}: declared {st.ty}, got {it}")
            env[st.name] = st.ty
        elif isinstance(st, Assign):
            if st.name not in env:
                raise TypeErrorSIMP(f"Assign to undeclared var: {st.name}")
            et = _tc_expr(st.expr, env, fsigs, imports, structs)
            if not _types_compat(et, env[st.name]):
                raise TypeErrorSIMP(f"Assign {st.name}: expected {env[st.name]}, got {et}")
        elif isinstance(st, ArrStoreStmt):
            if st.arr_name not in env or not isinstance(env[st.arr_name], TArray):
                raise TypeErrorSIMP(f"ArrStore target {st.arr_name} is not an array")
            ta = env[st.arr_name]
            ti = _tc_expr(st.index, env, fsigs, imports, structs)
            tv = _tc_expr(st.val, env, fsigs, imports, structs)
            if ti != INT or not _types_compat(tv, ta.elem):
                raise TypeErrorSIMP(f"ArrStore mismatch on {st.arr_name}")
        elif isinstance(st, PtrStoreStmt):
            tp = _tc_expr(st.ptr_expr, env, fsigs, imports, structs)
            tv = _tc_expr(st.val, env, fsigs, imports, structs)
            if not isinstance(tp, TPtr) or not _types_compat(tv, tp.elem):
                raise TypeErrorSIMP(f"PtrStore mismatch: *{tp} = {tv}")
        elif isinstance(st, IfStmt):
            tc = _tc_expr(st.cond, env, fsigs, imports, structs)
            if tc != BOOL:
                raise TypeErrorSIMP(f"If condition must be bool, got {tc}")
            _tc_stmts(st.then_body, dict(env), fsigs, imports, structs, ret_ty)
            _tc_stmts(st.else_body, dict(env), fsigs, imports, structs, ret_ty)
        elif isinstance(st, WhileStmt):
            tc = _tc_expr(st.cond, env, fsigs, imports, structs)
            if tc != BOOL:
                raise TypeErrorSIMP(f"While condition must be bool, got {tc}")
            _tc_stmts(st.body, dict(env), fsigs, imports, structs, ret_ty)
        elif isinstance(st, ReturnStmt):
            rt = _tc_expr(st.expr, env, fsigs, imports, structs)
            if not _types_compat(rt, ret_ty):
                raise TypeErrorSIMP(f"Return mismatch: expected {ret_ty}, got {rt}")
        else:
            raise TypeErrorSIMP(f"Unhandled stmt: {st}")
