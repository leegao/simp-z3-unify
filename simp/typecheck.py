"""Type checker for s-imp programs.

Discharged ASTs from :mod:`simp.simp_ast` are validated here. The entry point
is :func:`typecheck_program`, which raises :class:`TypeErrorSIMP` on the first
type error encountered.
"""

from .simp_ast import (
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
from .simp_types import (
    BOOL,
    BUILTIN_SIGS,
    FLOAT,
    INT,
    PYOBJ,
    STR,
    TArray,
    TDict,
    TPtr,
    TPyObj,
    TStruct,
    Type,
)


class TypeErrorSIMP(Exception):
    pass


def _is_pyobj(t: Type) -> bool:
    return isinstance(t, TPyObj)


def _is_unbox_target(t: Type) -> bool:
    """Types a host-Python object may be unboxed into (runtime-checked)."""
    return t in (BOOL, INT, FLOAT) or isinstance(t, (TStruct, TDict))


def typecheck_program(prog: Program) -> dict[str, tuple[list[Type], Type]]:
    func_sigs: dict[str, tuple[list[Type], Type]] = {}
    import_aliases = {imp.alias: imp.module for imp in prog.imports}
    struct_defs = {sd.name: sd.as_type() for sd in prog.structs}
    for fn in prog.funcs:
        if fn.name in func_sigs:
            raise TypeErrorSIMP(f"Duplicate function: {fn.name}")
        func_sigs[fn.name] = ([t for _, t in fn.params], fn.ret_ty)

    for fn in prog.funcs:
        env: dict[str, Type] = {p: t for p, t in fn.params}
        _tc_stmts(fn.body, env, func_sigs, import_aliases, struct_defs, fn.ret_ty)
    return func_sigs


def _types_compat(a: Type, b: Type) -> bool:
    return a == b or (_is_pyobj(a) and _is_pyobj(b))


def _tc_expr(
    e: Expr,
    env: dict[str, Type],
    fsigs: dict[str, tuple[list[Type], Type]],
    imports: dict[str, str],
    structs: dict[str, TStruct],
) -> Type:
    match e:
        case Const(_, ty):
            return ty
        case Var(name):
            if name not in env:
                raise TypeErrorSIMP(f"Undefined variable: {name}")
            return env[name]
        case UnOp(op, arg):
            t = _tc_expr(arg, env, fsigs, imports, structs)
            if op == "!" and t == BOOL:
                return BOOL
            if op == "-" and t in (INT, FLOAT):
                return t
            raise TypeErrorSIMP(f"Invalid UnOp {op} on {t}")
        case BinOp(op, left, right):
            tl = _tc_expr(left, env, fsigs, imports, structs)
            tr = _tc_expr(right, env, fsigs, imports, structs)
            if op in ("+", "-", "*", "/"):
                if tl == tr and tl in (INT, FLOAT):
                    return tl
                raise TypeErrorSIMP(f"Arithmetic op {op} mismatch: {tl} vs {tr}")
            if op in ("<", "<=", ">", ">="):
                if tl == tr and tl in (INT, FLOAT):
                    return BOOL
                raise TypeErrorSIMP(f"Comparison {op} mismatch: {tl} vs {tr}")
            if op in ("==", "!="):
                if _types_compat(tl, tr):
                    return BOOL
                raise TypeErrorSIMP(f"Equality {op} mismatch: {tl} vs {tr}")
            if op in ("&&", "||"):
                if tl == BOOL and tr == BOOL:
                    return BOOL
                raise TypeErrorSIMP(f"Logical op {op} requires bool, got {tl}, {tr}")
            raise TypeErrorSIMP(f"Unknown BinOp {op}")
        case IteExpr(cond, then_expr, else_expr):
            tc = _tc_expr(cond, env, fsigs, imports, structs)
            tt = _tc_expr(then_expr, env, fsigs, imports, structs)
            tf = _tc_expr(else_expr, env, fsigs, imports, structs)
            if tc != BOOL or not _types_compat(tt, tf):
                raise TypeErrorSIMP(f"Invalid IteExpr: cond={tc}, then={tt}, else={tf}")
            return tt
        case ArrLoad(arr, index):
            ta = _tc_expr(arr, env, fsigs, imports, structs)
            ti = _tc_expr(index, env, fsigs, imports, structs)
            if isinstance(ta, TArray):
                if ti != INT:
                    raise TypeErrorSIMP(f"Invalid ArrLoad: {ta}[{ti}]")
                return ta.elem
            if isinstance(ta, TDict):
                if not _types_compat(ti, ta.key):
                    raise TypeErrorSIMP(f"Invalid DictLoad: {ta}[{ti}]")
                return ta.val
            raise TypeErrorSIMP(f"Invalid ArrLoad: {ta}[{ti}]")
        case ArrStoreExpr(arr, index, val):
            ta = _tc_expr(arr, env, fsigs, imports, structs)
            ti = _tc_expr(index, env, fsigs, imports, structs)
            tv = _tc_expr(val, env, fsigs, imports, structs)
            if isinstance(ta, TArray):
                if ti != INT or not _types_compat(tv, ta.elem):
                    raise TypeErrorSIMP(f"Invalid ArrStoreExpr: {ta}[{ti}] = {tv}")
                return ta
            if isinstance(ta, TDict):
                if not _types_compat(ti, ta.key) or not _types_compat(tv, ta.val):
                    raise TypeErrorSIMP(f"Invalid DictStoreExpr: {ta}[{ti}] = {tv}")
                return ta
            raise TypeErrorSIMP(f"Invalid ArrStoreExpr: {ta}[{ti}] = {tv}")
        case PtrOffset(ptr, offset):
            tp = _tc_expr(ptr, env, fsigs, imports, structs)
            to = _tc_expr(offset, env, fsigs, imports, structs)
            if not isinstance(tp, TPtr) or to != INT:
                raise TypeErrorSIMP(f"Invalid PtrOffset: {tp} + {to}")
            return tp
        case PtrDeref(ptr):
            tp = _tc_expr(ptr, env, fsigs, imports, structs)
            if not isinstance(tp, TPtr):
                raise TypeErrorSIMP(f"Invalid PtrDeref on {tp}")
            return tp.elem
        case BuiltinCall(name, args):
            if name not in BUILTIN_SIGS:
                raise TypeErrorSIMP(f"Unknown builtin: {name}")
            ptys, rty = BUILTIN_SIGS[name]
            if len(args) != len(ptys):
                raise TypeErrorSIMP(f"Arity mismatch for {name}")
            for a, pt in zip(args, ptys):
                at = _tc_expr(a, env, fsigs, imports, structs)
                if at != pt:
                    raise TypeErrorSIMP(f"Arg type mismatch in {name}: {at} vs {pt}")
            return rty
        case FuncCall(name, args):
            if name not in fsigs:
                raise TypeErrorSIMP(f"Unknown function: {name}")
            ptys, rty = fsigs[name]
            if len(args) != len(ptys):
                raise TypeErrorSIMP(f"Arity mismatch for {name}")
            for a, pt in zip(args, ptys):
                at = _tc_expr(a, env, fsigs, imports, structs)
                if not _types_compat(at, pt):
                    raise TypeErrorSIMP(f"Arg type mismatch in {name}: {at} vs {pt}")
            return rty
        # ---- Host-Python typing rules ----
        case PyModuleRef(alias):
            if alias not in imports:
                raise TypeErrorSIMP(f"Unimported host-Python module alias: {alias}")
            return TPyObj(imports[alias])
        case PyAttr(obj, _, ret_ty):
            to = _tc_expr(obj, env, fsigs, imports, structs)
            if not _is_pyobj(to):
                raise TypeErrorSIMP(f"PyAttr requires pyobj receiver, got {to}")
            return ret_ty
        case PyCall(callee, args, ret_ty):
            tc = _tc_expr(callee, env, fsigs, imports, structs)
            if not _is_pyobj(tc):
                raise TypeErrorSIMP(f"PyCall requires pyobj callee, got {tc}")
            for a in args:
                _tc_expr(a, env, fsigs, imports, structs)
            return ret_ty
        case PyMethodCall(obj, _, args, ret_ty):
            to = _tc_expr(obj, env, fsigs, imports, structs)
            if not _is_pyobj(to):
                raise TypeErrorSIMP(f"PyMethodCall requires pyobj receiver, got {to}")
            for a in args:
                _tc_expr(a, env, fsigs, imports, structs)
            return ret_ty
        case PyBox(expr):
            t = _tc_expr(expr, env, fsigs, imports, structs)
            if t not in (BOOL, INT, FLOAT):
                raise TypeErrorSIMP(f"PyBox requires primitive type, got {t}")
            return PYOBJ
        case PyUnbox(expr, target_ty):
            t = _tc_expr(expr, env, fsigs, imports, structs)
            if not _is_pyobj(t) or not _is_unbox_target(target_ty):
                raise TypeErrorSIMP(f"PyUnbox requires pyobj -> unboxable, got {t} -> {target_ty}")
            return target_ty

        case StrConcat(left, right):
            if _tc_expr(left, env, fsigs, imports, structs) != STR or _tc_expr(right, env, fsigs, imports, structs) != STR:
                raise TypeErrorSIMP("StrConcat requires str, str")
            return STR
        case StrLen(arg):
            if _tc_expr(arg, env, fsigs, imports, structs) != STR:
                raise TypeErrorSIMP("StrLen requires str")
            return INT
        case StrSubstr(s, offset, length):
            if (_tc_expr(s, env, fsigs, imports, structs) != STR
                    or _tc_expr(offset, env, fsigs, imports, structs) != INT
                    or _tc_expr(length, env, fsigs, imports, structs) != INT):
                raise TypeErrorSIMP("StrSubstr requires (str, int, int)")
            return STR
        case StructInit(struct_name, field_vals):
            if struct_name not in structs:
                raise TypeErrorSIMP(f"Unknown struct: {struct_name}")
            st = structs[struct_name]
            if len(field_vals) != len(st.fields):
                raise TypeErrorSIMP(f"StructInit {struct_name} field count mismatch")
            for (fnm, fexp), (enm, ety) in zip(field_vals, st.fields):
                if fnm != enm:
                    raise TypeErrorSIMP(f"StructInit {struct_name}: expected field {enm}, got {fnm}")
                at = _tc_expr(fexp, env, fsigs, imports, structs)
                if not _types_compat(at, ety):
                    raise TypeErrorSIMP(f"StructInit {struct_name}.{fnm}: expected {ety}, got {at}")
            return st
        case DictInit(key_ty, val_ty, entries):
            if key_ty not in (STR, INT):
                raise TypeErrorSIMP(f"DictInit key type must be str or int, got {key_ty}")
            for k, v in entries:
                kt = _tc_expr(k, env, fsigs, imports, structs)
                if not _types_compat(kt, key_ty):
                    raise TypeErrorSIMP(f"DictInit key: expected {key_ty}, got {kt}")
                vt = _tc_expr(v, env, fsigs, imports, structs)
                if not _types_compat(vt, val_ty):
                    raise TypeErrorSIMP(f"DictInit val: expected {val_ty}, got {vt}")
            return TDict(key_ty, val_ty)
        case FieldGet(target, field):
            ts = _tc_expr(target, env, fsigs, imports, structs)
            if not isinstance(ts, TStruct):
                raise TypeErrorSIMP(f"FieldGet requires struct, got {ts}")
            return ts.field_type(field)
        case FieldSet(target, field, val):
            ts = _tc_expr(target, env, fsigs, imports, structs)
            if not isinstance(ts, TStruct):
                raise TypeErrorSIMP(f"FieldSet requires struct, got {ts}")
            ft = ts.field_type(field)
            vt = _tc_expr(val, env, fsigs, imports, structs)
            if not _types_compat(vt, ft):
                raise TypeErrorSIMP(f"FieldSet {ts.name}.{field}: expected {ft}, got {vt}")
            return ts
        case _:
            raise TypeErrorSIMP(f"Unhandled expr: {e}")


def _tc_stmts(
    stmts: list[Stmt],
    env: dict[str, Type],
    fsigs: dict[str, tuple[list[Type], Type]],
    imports: dict[str, str],
    structs: dict[str, TStruct],
    ret_ty: Type,
):
    for st in stmts:
        match st:
            case LetDecl(name, ty, init):
                it = _tc_expr(init, env, fsigs, imports, structs)
                if not _types_compat(it, ty):
                    raise TypeErrorSIMP(f"LetDecl {name}: declared {ty}, got {it}")
                env[name] = ty
            case Assign(name, expr):
                if name not in env:
                    raise TypeErrorSIMP(f"Assign to undeclared var: {name}")
                et = _tc_expr(expr, env, fsigs, imports, structs)
                if not _types_compat(et, env[name]):
                    raise TypeErrorSIMP(f"Assign {name}: expected {env[name]}, got {et}")
            case ArrStoreStmt(arr_name, index, val):
                if arr_name not in env or not isinstance(env[arr_name], (TArray, TDict)):
                    raise TypeErrorSIMP(f"ArrStore target {arr_name} is not an array/dict")
                ta = env[arr_name]
                ti = _tc_expr(index, env, fsigs, imports, structs)
                tv = _tc_expr(val, env, fsigs, imports, structs)
                if isinstance(ta, TArray):
                    if ti != INT or not _types_compat(tv, ta.elem):
                        raise TypeErrorSIMP(f"ArrStore mismatch on {arr_name}")
                else:
                    if not _types_compat(ti, ta.key) or not _types_compat(tv, ta.val):
                        raise TypeErrorSIMP(f"DictStore mismatch on {arr_name}")
            case PtrStoreStmt(ptr_expr, val):
                tp = _tc_expr(ptr_expr, env, fsigs, imports, structs)
                tv = _tc_expr(val, env, fsigs, imports, structs)
                if not isinstance(tp, TPtr) or not _types_compat(tv, tp.elem):
                    raise TypeErrorSIMP(f"PtrStore mismatch: *{tp} = {tv}")
            case IfStmt(cond, then_body, else_body):
                tc = _tc_expr(cond, env, fsigs, imports, structs)
                if tc != BOOL:
                    raise TypeErrorSIMP(f"If condition must be bool, got {tc}")
                _tc_stmts(then_body, dict(env), fsigs, imports, structs, ret_ty)
                _tc_stmts(else_body, dict(env), fsigs, imports, structs, ret_ty)
            case WhileStmt(cond, body, _):
                tc = _tc_expr(cond, env, fsigs, imports, structs)
                if tc != BOOL:
                    raise TypeErrorSIMP(f"While condition must be bool, got {tc}")
                _tc_stmts(body, dict(env), fsigs, imports, structs, ret_ty)
            case ReturnStmt(expr):
                rt = _tc_expr(expr, env, fsigs, imports, structs)
                if not _types_compat(rt, ret_ty):
                    raise TypeErrorSIMP(f"Return mismatch: expected {ret_ty}, got {rt}")
            case _:
                raise TypeErrorSIMP(f"Unhandled stmt: {st}")