import copy

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
from .simp_types import BOOL, FLOAT, INT, STR


def _expr_vars(e: Expr) -> set[str]:
    if isinstance(e, Var):
        return {e.name}
    if isinstance(e, Const):
        return set()
    # Native string & struct peephole simplifications
    if isinstance(e, StrConcat):
        if isinstance(e.left, Const) and e.left.ty == STR and e.left.value == "":
            return e.right
        if isinstance(e.right, Const) and e.right.ty == STR and e.right.value == "":
            return e.left
        if isinstance(e.left, Const) and isinstance(e.right, Const) and e.left.ty == STR and e.right.ty == STR:
            return Const(e.left.value + e.right.value, STR)
    if isinstance(e, StrLen) and isinstance(e.arg, Const) and e.arg.ty == STR:
        return Const(len(e.arg.value), INT)
    if isinstance(e, FieldGet):
        if isinstance(e.target, StructInit):
            for k, v in e.target.field_vals:
                if k == e.field:
                    return v
        if isinstance(e.target, FieldSet) and e.target.field == e.field:
            return e.target.val
    if isinstance(e, UnOp):
        return _expr_vars(e.arg)
    if isinstance(e, BinOp):
        return _expr_vars(e.left) | _expr_vars(e.right)
    if isinstance(e, IteExpr):
        return _expr_vars(e.cond) | _expr_vars(e.then_expr) | _expr_vars(e.else_expr)
    if isinstance(e, ArrLoad):
        return _expr_vars(e.arr) | _expr_vars(e.index)
    if isinstance(e, ArrStoreExpr):
        return _expr_vars(e.arr) | _expr_vars(e.index) | _expr_vars(e.val)
    if isinstance(e, PtrOffset):
        return _expr_vars(e.ptr) | _expr_vars(e.offset)
    if isinstance(e, PtrDeref):
        return _expr_vars(e.ptr)
    if isinstance(e, StrConcat):
        return _expr_vars(e.left) | _expr_vars(e.right)
    if isinstance(e, StrLen):
        return _expr_vars(e.arg)
    if isinstance(e, StrSubstr):
        return _expr_vars(e.s) | _expr_vars(e.offset) | _expr_vars(e.length)
    if isinstance(e, StructInit):
        s: set[str] = set()
        for _, v in e.field_vals:
            s |= _expr_vars(v)
        return s
    if isinstance(e, DictInit):
        s: set[str] = set()
        for k, v in e.entries:
            s |= _expr_vars(k) | _expr_vars(v)
        return s
    if isinstance(e, FieldGet):
        return _expr_vars(e.target)
    if isinstance(e, FieldSet):
        return _expr_vars(e.target) | _expr_vars(e.val)
    if isinstance(e, PyModuleRef):
        return set()
    if isinstance(e, (PyAttr, PyBox, PyUnbox)):
        return _expr_vars(e.obj if isinstance(e, PyAttr) else e.expr)
    if isinstance(e, PyCall):
        s: set[str] = _expr_vars(e.callee)
        for a in e.args:
            s |= _expr_vars(a)
        return s
    if isinstance(e, PyMethodCall):
        s: set[str] = _expr_vars(e.obj)
        for a in e.args:
            s |= _expr_vars(a)
        return s
    if isinstance(e, (BuiltinCall, FuncCall)):
        s: set[str] = set()
        for a in e.args:
            s |= _expr_vars(a)
        return s
    return set()


def _expr_is_pure(e: Expr) -> bool:
    """True if expression does not read memory (arrays/ptrs) or call user functions."""
    if isinstance(e, (Const, Var)):
        return True
    # Native string & struct peephole simplifications
    if isinstance(e, StrConcat):
        if isinstance(e.left, Const) and e.left.ty == STR and e.left.value == "":
            return e.right
        if isinstance(e.right, Const) and e.right.ty == STR and e.right.value == "":
            return e.left
        if isinstance(e.left, Const) and isinstance(e.right, Const) and e.left.ty == STR and e.right.ty == STR:
            return Const(e.left.value + e.right.value, STR)
    if isinstance(e, StrLen) and isinstance(e.arg, Const) and e.arg.ty == STR:
        return Const(len(e.arg.value), INT)
    if isinstance(e, FieldGet):
        if isinstance(e.target, StructInit):
            for k, v in e.target.field_vals:
                if k == e.field:
                    return v
        if isinstance(e.target, FieldSet) and e.target.field == e.field:
            return e.target.val
    if isinstance(e, UnOp):
        return _expr_is_pure(e.arg)
    if isinstance(e, BinOp):
        return _expr_is_pure(e.left) and _expr_is_pure(e.right)
    if isinstance(e, IteExpr):
        return _expr_is_pure(e.cond) and _expr_is_pure(e.then_expr) and _expr_is_pure(e.else_expr)
    if isinstance(e, BuiltinCall):
        return all(_expr_is_pure(a) for a in e.args)
    if isinstance(e, StrConcat):
        return _expr_is_pure(e.left) and _expr_is_pure(e.right)
    if isinstance(e, StrLen):
        return _expr_is_pure(e.arg)
    if isinstance(e, StrSubstr):
        return _expr_is_pure(e.s) and _expr_is_pure(e.offset) and _expr_is_pure(e.length)
    if isinstance(e, StructInit):
        return all(_expr_is_pure(v) for _, v in e.field_vals)
    if isinstance(e, DictInit):
        return all(_expr_is_pure(k) and _expr_is_pure(v) for k, v in e.entries)
    if isinstance(e, FieldGet):
        return _expr_is_pure(e.target)
    if isinstance(e, FieldSet):
        return _expr_is_pure(e.target) and _expr_is_pure(e.val)
    if isinstance(e, PyModuleRef):
        return True
    if isinstance(e, (PyAttr, PyBox, PyUnbox)):
        return _expr_is_pure(e.obj if isinstance(e, PyAttr) else e.expr)
    if isinstance(e, PyCall):
        return _expr_is_pure(e.callee) and all(_expr_is_pure(a) for a in e.args)
    if isinstance(e, PyMethodCall):
        return _expr_is_pure(e.obj) and all(_expr_is_pure(a) for a in e.args)
    return False


def _map_expr(e: Expr, fn) -> Expr:
    if isinstance(e, (Const, Var)):
        return fn(e)
    # Native string & struct peephole simplifications
    if isinstance(e, StrConcat):
        if isinstance(e.left, Const) and e.left.ty == STR and e.left.value == "":
            return e.right
        if isinstance(e.right, Const) and e.right.ty == STR and e.right.value == "":
            return e.left
        if isinstance(e.left, Const) and isinstance(e.right, Const) and e.left.ty == STR and e.right.ty == STR:
            return Const(e.left.value + e.right.value, STR)
    if isinstance(e, StrLen) and isinstance(e.arg, Const) and e.arg.ty == STR:
        return Const(len(e.arg.value), INT)
    if isinstance(e, FieldGet):
        if isinstance(e.target, StructInit):
            for k, v in e.target.field_vals:
                if k == e.field:
                    return v
        if isinstance(e.target, FieldSet) and e.target.field == e.field:
            return e.target.val
    if isinstance(e, UnOp):
        return fn(UnOp(e.op, _map_expr(e.arg, fn)))
    if isinstance(e, BinOp):
        return fn(BinOp(e.op, _map_expr(e.left, fn), _map_expr(e.right, fn)))
    if isinstance(e, IteExpr):
        return fn(IteExpr(_map_expr(e.cond, fn), _map_expr(e.then_expr, fn), _map_expr(e.else_expr, fn)))
    if isinstance(e, ArrLoad):
        return fn(ArrLoad(_map_expr(e.arr, fn), _map_expr(e.index, fn)))
    if isinstance(e, ArrStoreExpr):
        return fn(ArrStoreExpr(_map_expr(e.arr, fn), _map_expr(e.index, fn), _map_expr(e.val, fn)))
    if isinstance(e, PtrOffset):
        return fn(PtrOffset(_map_expr(e.ptr, fn), _map_expr(e.offset, fn)))
    if isinstance(e, PtrDeref):
        return fn(PtrDeref(_map_expr(e.ptr, fn)))
    if isinstance(e, BuiltinCall):
        return fn(BuiltinCall(e.name, tuple(_map_expr(a, fn) for a in e.args)))
    if isinstance(e, FuncCall):
        return fn(FuncCall(e.name, tuple(_map_expr(a, fn) for a in e.args)))
    if isinstance(e, StrConcat):
        return fn(StrConcat(_map_expr(e.left, fn), _map_expr(e.right, fn)))
    if isinstance(e, StrLen):
        return fn(StrLen(_map_expr(e.arg, fn)))
    if isinstance(e, StrSubstr):
        return fn(StrSubstr(_map_expr(e.s, fn), _map_expr(e.offset, fn), _map_expr(e.length, fn)))
    if isinstance(e, StructInit):
        return fn(StructInit(e.struct_name, tuple((k, _map_expr(v, fn)) for k, v in e.field_vals)))
    if isinstance(e, DictInit):
        return fn(DictInit(e.key_ty, e.val_ty, tuple((_map_expr(k, fn), _map_expr(v, fn)) for k, v in e.entries)))
    if isinstance(e, FieldGet):
        return fn(FieldGet(_map_expr(e.target, fn), e.field))
    if isinstance(e, FieldSet):
        return fn(FieldSet(_map_expr(e.target, fn), e.field, _map_expr(e.val, fn)))
    if isinstance(e, PyModuleRef):
        return fn(e)
    if isinstance(e, PyAttr):
        return fn(PyAttr(_map_expr(e.obj, fn), e.attr, e.ret_ty))
    if isinstance(e, PyBox):
        return fn(PyBox(_map_expr(e.expr, fn)))
    if isinstance(e, PyUnbox):
        return fn(PyUnbox(_map_expr(e.expr, fn), e.target_ty))
    if isinstance(e, PyCall):
        return fn(PyCall(_map_expr(e.callee, fn), tuple(_map_expr(a, fn) for a in e.args), e.ret_ty))
    if isinstance(e, PyMethodCall):
        return fn(PyMethodCall(_map_expr(e.obj, fn), e.method, tuple(_map_expr(a, fn) for a in e.args), e.ret_ty))
    return fn(e)


def _map_stmt_exprs(stmts: list[Stmt], fn) -> list[Stmt]:
    out: list[Stmt] = []
    for st in stmts:
        if isinstance(st, LetDecl):
            out.append(LetDecl(st.name, st.ty, _map_expr(st.init, fn)))
        elif isinstance(st, Assign):
            out.append(Assign(st.name, _map_expr(st.expr, fn)))
        elif isinstance(st, ArrStoreStmt):
            out.append(ArrStoreStmt(st.arr_name, _map_expr(st.index, fn), _map_expr(st.val, fn)))
        elif isinstance(st, PtrStoreStmt):
            out.append(PtrStoreStmt(_map_expr(st.ptr_expr, fn), _map_expr(st.val, fn)))
        elif isinstance(st, IfStmt):
            out.append(IfStmt(_map_expr(st.cond, fn), _map_stmt_exprs(st.then_body, fn), _map_stmt_exprs(st.else_body, fn)))
        elif isinstance(st, WhileStmt):
            out.append(WhileStmt(_map_expr(st.cond, fn), _map_stmt_exprs(st.body, fn), st.unroll_bound))
        elif isinstance(st, ReturnStmt):
            out.append(ReturnStmt(_map_expr(st.expr, fn)))
    return out


# ----------------------------------------------------------------------------
# Pass 1: Constant Folding + Algebraic & Peephole Rewrites
# ----------------------------------------------------------------------------

def _fold_peephole(e: Expr) -> Expr:
    # Native string & struct peephole simplifications
    if isinstance(e, StrConcat):
        if isinstance(e.left, Const) and e.left.ty == STR and e.left.value == "":
            return e.right
        if isinstance(e.right, Const) and e.right.ty == STR and e.right.value == "":
            return e.left
        if isinstance(e.left, Const) and isinstance(e.right, Const) and e.left.ty == STR and e.right.ty == STR:
            return Const(e.left.value + e.right.value, STR)
    if isinstance(e, StrLen) and isinstance(e.arg, Const) and e.arg.ty == STR:
        return Const(len(e.arg.value), INT)
    if isinstance(e, FieldGet):
        if isinstance(e.target, StructInit):
            for k, v in e.target.field_vals:
                if k == e.field:
                    return v
        if isinstance(e.target, FieldSet) and e.target.field == e.field:
            return e.target.val
    if isinstance(e, UnOp):
        if e.op == "!" and isinstance(e.arg, Const) and e.arg.ty == BOOL:
            return Const(not e.arg.value, BOOL)
        if e.op == "!" and isinstance(e.arg, UnOp) and e.arg.op == "!":
            return e.arg.arg
        if e.op == "-" and isinstance(e.arg, Const):
            return Const(-e.arg.value, e.arg.ty)
    if isinstance(e, IteExpr):
        if isinstance(e.cond, Const) and e.cond.ty == BOOL:
            return e.then_expr if e.cond.value else e.else_expr
        if isinstance(e.cond, UnOp) and e.cond.op == "!":
            return IteExpr(e.cond.arg, e.else_expr, e.then_expr)
        if e.then_expr == e.else_expr:
            return e.then_expr
    if isinstance(e, BinOp):
        l, r, op = e.left, e.right, e.op
        # Constant folding
        if isinstance(l, Const) and isinstance(r, Const) and l.ty == r.ty:
            if op == "+":
                return Const(l.value + r.value, l.ty)
            if op == "-":
                return Const(l.value - r.value, l.ty)
            if op == "*":
                return Const(l.value * r.value, l.ty)
            if op == "/" and r.value != 0:
                val = l.value // r.value if l.ty == INT else l.value / r.value
                return Const(val, l.ty)
            if op == "==":
                return Const(l.value == r.value, BOOL)
            if op == "!=":
                return Const(l.value != r.value, BOOL)
            if op == "<":
                return Const(l.value < r.value, BOOL)
            if op == "<=":
                return Const(l.value <= r.value, BOOL)
            if op == ">":
                return Const(l.value > r.value, BOOL)
            if op == ">=":
                return Const(l.value >= r.value, BOOL)
            if op == "&&":
                return Const(l.value and r.value, BOOL)
            if op == "||":
                return Const(l.value or r.value, BOOL)
        # Algebraic identities & strength reductions
        if op == "+":
            if isinstance(r, Const) and r.value == 0:
                return l
            if isinstance(l, Const) and l.value == 0:
                return r
        if op == "-":
            if isinstance(r, Const) and r.value == 0:
                return l
            if l == r:
                return Const(0.0, FLOAT)
        if op == "*":
            if isinstance(r, Const) and r.value == 1:
                return l
            if isinstance(l, Const) and l.value == 1:
                return r
            if (isinstance(r, Const) and r.value == 0) or (isinstance(l, Const) and l.value == 0):
                return Const(0 if (isinstance(r, Const) and r.ty == INT) else 0.0, r.ty if isinstance(r, Const) else l.ty)
            # Strength reduction: x * 2.0 -> x + x
            if isinstance(r, Const) and r.ty == FLOAT and r.value == 2.0:
                return BinOp("+", l, l)
            # Distributivity rewrite (intentionally creates parent mismatch Mul -> Add!):
            # (a + b) * c -> a * c + b * c
            if isinstance(l, BinOp) and l.op == "+":
                return BinOp("+", BinOp("*", l.left, r), BinOp("*", l.right, r))
        if op == "&&":
            if isinstance(l, Const) and l.value is True:
                return r
            if isinstance(r, Const) and r.value is True:
                return l
            if (isinstance(l, Const) and l.value is False) or (isinstance(r, Const) and r.value is False):
                return Const(False, BOOL)
    return e


def const_fold_pass(prog: Program) -> Program:
    p = copy.deepcopy(prog)
    for fn in p.funcs:
        fn.body = _map_stmt_exprs(fn.body, _fold_peephole)
        # Also prune statically known IfStmt conditions
        fn.body = _prune_const_ifs(fn.body)
    return p


def _prune_const_ifs(stmts: list[Stmt]) -> list[Stmt]:
    out: list[Stmt] = []
    for st in stmts:
        if isinstance(st, IfStmt):
            if isinstance(st.cond, Const) and st.cond.ty == BOOL:
                branch = st.then_body if st.cond.value else st.else_body
                out.extend(_prune_const_ifs(branch))
            else:
                out.append(IfStmt(st.cond, _prune_const_ifs(st.then_body), _prune_const_ifs(st.else_body)))
        elif isinstance(st, WhileStmt):
            out.append(WhileStmt(st.cond, _prune_const_ifs(st.body), st.unroll_bound))
        else:
            out.append(st)
    return out


# ----------------------------------------------------------------------------
# Pass 2: Local Available-Expression CSE (Common Subexpression Elimination)
# ----------------------------------------------------------------------------

def cse_pass(prog: Program) -> Program:
    p = copy.deepcopy(prog)
    for fn in p.funcs:
        fn.body = _cse_block(fn.body, {})
    return p


def _invalidate(avail: dict[Expr, str], mutated_var: str) -> dict[Expr, str]:
    return {e: v for e, v in avail.items() if mutated_var not in _expr_vars(e) and v != mutated_var}


def _cse_replace(e: Expr, avail: dict[Expr, str]) -> Expr:
    if e in avail:
        return Var(avail[e])
    return _map_expr(e, lambda sub: Var(avail[sub]) if sub in avail else sub)


def _cse_block(stmts: list[Stmt], avail_in: dict[Expr, str]) -> list[Stmt]:
    avail = dict(avail_in)
    out: list[Stmt] = []
    for st in stmts:
        if isinstance(st, LetDecl):
            new_init = _cse_replace(st.init, avail)
            out.append(LetDecl(st.name, st.ty, new_init))
            avail = _invalidate(avail, st.name)
            if _expr_is_pure(new_init) and not isinstance(new_init, (Var, Const)):
                avail[new_init] = st.name
        elif isinstance(st, Assign):
            new_e = _cse_replace(st.expr, avail)
            out.append(Assign(st.name, new_e))
            avail = _invalidate(avail, st.name)
            if _expr_is_pure(new_e) and not isinstance(new_e, (Var, Const)):
                avail[new_e] = st.name
        elif isinstance(st, ArrStoreStmt):
            out.append(ArrStoreStmt(st.arr_name, _cse_replace(st.index, avail), _cse_replace(st.val, avail)))
            avail = _invalidate(avail, st.arr_name)
        elif isinstance(st, PtrStoreStmt):
            out.append(PtrStoreStmt(_cse_replace(st.ptr_expr, avail), _cse_replace(st.val, avail)))
        elif isinstance(st, IfStmt):
            nc = _cse_replace(st.cond, avail)
            nt = _cse_block(st.then_body, dict(avail))
            ne = _cse_block(st.else_body, dict(avail))
            out.append(IfStmt(nc, nt, ne))
            avail = {}  # conservative join
        elif isinstance(st, WhileStmt):
            out.append(WhileStmt(st.cond, _cse_block(st.body, {}), st.unroll_bound))
            avail = {}
        elif isinstance(st, ReturnStmt):
            out.append(ReturnStmt(_cse_replace(st.expr, avail)))
    return out


# ----------------------------------------------------------------------------
# Pass 3: Store-to-Load Forwarding & Loop-Invariant Code Motion (Mem+LICM)
# ----------------------------------------------------------------------------

def mem_licm_pass(prog: Program) -> Program:
    p = copy.deepcopy(prog)
    counter = [0]
    for fn in p.funcs:
        fn.body = _mem_licm_block(fn.body, {}, counter)
    return p


def _mutated_in_stmts(stmts: list[Stmt]) -> set[str]:
    mut: set[str] = set()
    for st in stmts:
        if isinstance(st, (LetDecl, Assign)):
            mut.add(st.name)
        elif isinstance(st, ArrStoreStmt):
            mut.add(st.arr_name)
        elif isinstance(st, IfStmt):
            mut |= _mutated_in_stmts(st.then_body) | _mutated_in_stmts(st.else_body)
        elif isinstance(st, WhileStmt):
            mut |= _mutated_in_stmts(st.body)
    return mut


def _mem_licm_block(stmts: list[Stmt], known_stores: dict[tuple[str, Expr], Expr], counter: list[int]) -> list[Stmt]:
    stores = dict(known_stores)
    out: list[Stmt] = []

    def fwd(e: Expr) -> Expr:
        if isinstance(e, ArrLoad) and isinstance(e.arr, Var):
            key = (e.arr.name, e.index)
            if key in stores:
                return stores[key]
        return e

    for st in stmts:
        if isinstance(st, LetDecl):
            ne = _map_expr(st.init, fwd)
            out.append(LetDecl(st.name, st.ty, ne))
            stores = {k: v for k, v in stores.items() if st.name not in _expr_vars(k[1]) and st.name not in _expr_vars(v)}
        elif isinstance(st, Assign):
            ne = _map_expr(st.expr, fwd)
            out.append(Assign(st.name, ne))
            stores = {k: v for k, v in stores.items() if st.name not in _expr_vars(k[1]) and st.name not in _expr_vars(v)}
        elif isinstance(st, ArrStoreStmt):
            ni = _map_expr(st.index, fwd)
            nv = _map_expr(st.val, fwd)
            out.append(ArrStoreStmt(st.arr_name, ni, nv))
            # Clear potentially aliased entries for same array, record exact store
            stores = {k: v for k, v in stores.items() if k[0] != st.arr_name}
            stores[(st.arr_name, ni)] = nv
        elif isinstance(st, PtrStoreStmt):
            out.append(PtrStoreStmt(_map_expr(st.ptr_expr, fwd), _map_expr(st.val, fwd)))
            stores.clear()
        elif isinstance(st, IfStmt):
            nc = _map_expr(st.cond, fwd)
            nt = _mem_licm_block(st.then_body, dict(stores), counter)
            ne = _mem_licm_block(st.else_body, dict(stores), counter)
            out.append(IfStmt(nc, nt, ne))
            stores.clear()
        elif isinstance(st, WhileStmt):
            # LICM: hoist pure non-trivial BuiltinCall subexpressions whose vars are not mutated in loop
            mut = _mutated_in_stmts(st.body)
            hoisted: dict[Expr, str] = {}
            hoist_decls: list[Stmt] = []

            def hoist_fn(e: Expr) -> Expr:
                if isinstance(e, BuiltinCall) and _expr_is_pure(e) and (_expr_vars(e).isdisjoint(mut)):
                    if e not in hoisted:
                        tname = f"_licm_{counter[0]}"
                        counter[0] += 1
                        hoisted[e] = tname
                        hoist_decls.append(LetDecl(tname, FLOAT, e))
                    return Var(hoisted[e])
                return e

            new_body = _map_stmt_exprs(st.body, hoist_fn)
            out.extend(hoist_decls)
            out.append(WhileStmt(st.cond, _mem_licm_block(new_body, {}, counter), st.unroll_bound))
            stores.clear()
        elif isinstance(st, ReturnStmt):
            out.append(ReturnStmt(_map_expr(st.expr, fwd)))
    return out


# ----------------------------------------------------------------------------
# Pass 4: Liveness-Based Dead Code Elimination (DCE)
# ----------------------------------------------------------------------------

def dce_pass(prog: Program) -> Program:
    p = copy.deepcopy(prog)
    for fn in p.funcs:
        fn.body, _ = _dce_stmts(fn.body, set())
    return p


def _dce_stmts(stmts: list[Stmt], live_out: set[str]) -> tuple[list[Stmt], set[str]]:
    live = set(live_out)
    out_rev: list[Stmt] = []
    for st in reversed(stmts):
        if isinstance(st, ReturnStmt):
            live |= _expr_vars(st.expr)
            out_rev.append(st)
        elif isinstance(st, LetDecl):
            if st.name in live or not _expr_is_pure(st.init):
                live.discard(st.name)
                live |= _expr_vars(st.init)
                out_rev.append(st)
        elif isinstance(st, Assign):
            if st.name in live or not _expr_is_pure(st.expr):
                live.discard(st.name)
                live |= _expr_vars(st.expr)
                out_rev.append(st)
        elif isinstance(st, ArrStoreStmt):
            # Memory effect is observable if array may be live or returned
            live |= _expr_vars(st.index) | _expr_vars(st.val) | {st.arr_name}
            out_rev.append(st)
        elif isinstance(st, PtrStoreStmt):
            live |= _expr_vars(st.ptr_expr) | _expr_vars(st.val)
            out_rev.append(st)
        elif isinstance(st, IfStmt):
            nt, lt = _dce_stmts(st.then_body, live)
            ne, le = _dce_stmts(st.else_body, live)
            live = lt | le | _expr_vars(st.cond)
            if nt or ne:
                out_rev.append(IfStmt(st.cond, nt, ne))
        elif isinstance(st, WhileStmt):
            # Conservative fixed-point for loop liveness
            loop_live = set(live) | _expr_vars(st.cond)
            for _ in range(3):
                nb, lb = _dce_stmts(st.body, loop_live)
                loop_live |= lb | _expr_vars(st.cond)
            live = loop_live
            out_rev.append(WhileStmt(st.cond, nb, st.unroll_bound))
    out_rev.reverse()
    return out_rev, live


# ----------------------------------------------------------------------------
# Negative Test Pass: Deliberately Buggy Peephole (to verify Z3 catches SAT)
# ----------------------------------------------------------------------------

def buggy_opt_pass(prog: Program) -> Program:
    """Deliberately introduces an unsound rewrite: a - b -> a + b on first Sub."""
    p = copy.deepcopy(prog)
    done = [False]

    def corrupt(e: Expr) -> Expr:
        if not done[0] and isinstance(e, BinOp) and e.op == "-":
            done[0] = True
            return BinOp("+", e.left, e.right)
        return e

    for fn in p.funcs:
        fn.body = _map_stmt_exprs(fn.body, corrupt)
    return p
