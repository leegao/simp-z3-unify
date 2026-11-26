"""Concrete big-step interpreter for s-imp.

This is a direct, executable reading of the reference operational semantics in
``semantics.md``:

  * Values (``Val``) are represented by the classes below; the store ``σ`` is a
    ``dict`` of variables and the heap ``H`` is a ``dict`` of locations.
  * Expressions are **pure** (``eval_expr`` reads ``σ``/``H`` but never mutates
    them); array/dict/field "stores" are functional and return new values.
  * Statement execution (``exec_stmt``) threads ``(σ, H)`` and yields a result
    ``r ∈ {⊥, ret(v)}`` (``None`` is fall-through, ``Ret`` is an early return).
  * The host-Python extension is *uninterpreted* in the semantics; a concrete
    interpreter gives it concrete meaning by actually importing modules and
    invoking attributes/calls/methods, with ``box``/``unbox`` as the bridge.
    ``unbox`` into a struct/dict is **runtime-checked**.

The entry point is :func:`interpret`, which parses nothing itself -- pass it a
typed :class:`simp.simp_ast.Program` (e.g. from :func:`simp.parser.parse`).
"""

from __future__ import annotations

import argparse
import importlib
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

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
from .simp_types import (
    BOOL,
    FLOAT,
    INT,
    STR,
    TArray,
    TDict,
    TPtr,
    TPyObj,
    TStruct,
    Type,
)

# --------------------------------------------------------------------------- #
# Runtime values (semantic domain ``Val``)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PyObj:
    """An opaque host-Python object handle (carries the real Python value)."""
    value: object


@dataclass(frozen=True)
class Ptr:
    """A pointer value: a heap location ``ℓ ∈ Loc``."""
    loc: int


@dataclass
class Array:
    """A McCarthy-style array: a finite map ``ℤ ⇀ Val``."""
    data: dict[int, object] = field(default_factory=dict)

    def select(self, i: int) -> object:
        return self.data[i]

    def store(self, i: int, v: object) -> Array:
        nd = dict(self.data)
        nd[i] = v
        return Array(nd)


@dataclass
class Dict:
    """A dictionary value: a finite map ``Key ⇀ Val``."""
    data: dict[object, object] = field(default_factory=dict)

    def select(self, k: object) -> object:
        return self.data[k]

    def store(self, k: object, v: object) -> Dict:
        nd = dict(self.data)
        nd[k] = v
        return Dict(nd)


@dataclass(frozen=True)
class StructVal:
    """A struct value: a named field map ``{f_i ↦ v_i}``."""
    name: str
    fields: dict[str, object]


@dataclass
class Ret:
    """An early return carrying a value (``ret(v)``)."""
    value: object


# Sentinel for fall-through (⊥) is ``None``.


class SIMPRuntimeError(Exception):
    """Raised when execution cannot proceed (e.g. step limit, bad unbox)."""


# --------------------------------------------------------------------------- #
# Interpreter
# --------------------------------------------------------------------------- #


class Interp:
    """A concrete big-step interpreter for a single :class:`Program`."""

    def __init__(self, prog: Program, max_steps: int = 1_000_000):
        self.prog = prog
        self.funcs: dict[str, FuncDecl] = {fn.name: fn for fn in prog.funcs}
        self.structs = {sd.name: sd for sd in prog.structs}
        self.import_map: dict[str, str] = {imp.alias: imp.module for imp in prog.imports}
        self.max_steps = max_steps
        self._steps = 0
        self._heap_counter = 0
        self._modules: dict[str, object] = {}

    # -- helpers ----------------------------------------------------------- #

    def _tick(self) -> None:
        self._steps += 1
        if self._steps > self.max_steps:
            raise SIMPRuntimeError(f"step limit exceeded ({self.max_steps})")

    def _fresh_loc(self) -> int:
        loc = self._heap_counter
        self._heap_counter += 1
        return loc

    def _module(self, name: str) -> object:
        mod = self._modules.get(name)
        if mod is None:
            mod = importlib.import_module(name)
            self._modules[name] = mod
        return mod

    @staticmethod
    def _unwrap(v: object) -> object:
        return v.value if isinstance(v, PyObj) else v

    def _wrap(self, v: object, ty: Type) -> object:
        """Coerce a raw host result into the value domain for ``ty``."""
        if isinstance(ty, TPyObj):
            return PyObj(v)
        return self._convert(v, ty)

    def _convert(self, v: object, ty: Type) -> object:
        """Runtime-checked ``unbox`` conversion of a host value into ``ty``."""
        if ty == BOOL:
            if not isinstance(v, bool):
                raise SIMPRuntimeError(f"unbox: expected bool, got {type(v).__name__}")
            return v
        if ty == INT:
            if type(v) is not int:
                raise SIMPRuntimeError(f"unbox: expected int, got {type(v).__name__}")
            return v
        if ty == FLOAT:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise SIMPRuntimeError(f"unbox: expected float, got {type(v).__name__}")
            return float(v)
        if ty == STR:
            if not isinstance(v, str):
                raise SIMPRuntimeError(f"unbox: expected str, got {type(v).__name__}")
            return v
        if isinstance(ty, TStruct):
            fields: dict[str, object] = {}
            for fname, fty in ty.fields:
                if isinstance(v, dict):
                    if fname not in v:
                        raise SIMPRuntimeError(f"unbox: struct {ty.name} missing field {fname}")
                    raw = v[fname]
                else:
                    if not hasattr(v, fname):
                        raise SIMPRuntimeError(f"unbox: struct {ty.name} missing attribute {fname}")
                    raw = getattr(v, fname)
                fields[fname] = self._convert(raw, fty)
            return StructVal(ty.name, fields)
        if isinstance(ty, TDict):
            if not isinstance(v, dict):
                raise SIMPRuntimeError(f"unbox: expected dict, got {type(v).__name__}")
            out: dict[object, object] = {}
            for k, val in v.items():
                out[self._convert(k, ty.key)] = self._convert(val, ty.val)
            return Dict(out)
        raise SIMPRuntimeError(f"unbox: unsupported target type {ty}")

    # -- expression evaluation (§3.1) -------------------------------------- #

    def eval_expr(self, e: Expr, sigma: dict[str, object], H: dict[int, object]) -> object:
        self._tick()

        if isinstance(e, Const):
            return e.value

        if isinstance(e, Var):
            return sigma[e.name]

        if isinstance(e, UnOp):
            a = self.eval_expr(e.arg, sigma, H)
            return (not a) if e.op == "!" else (-a)

        if isinstance(e, BinOp):
            l = self.eval_expr(e.left, sigma, H)
            r = self.eval_expr(e.right, sigma, H)
            op = e.op
            if op == "+":
                return l + r
            if op == "-":
                return l - r
            if op == "*":
                return l * r
            if op == "/":
                # ℤ/ℤ is integer division; otherwise real division.
                if type(l) is int and type(r) is int:
                    return l // r
                return l / r
            if op == "==":
                return l == r
            if op == "!=":
                return l != r
            if op == "<":
                return l < r
            if op == "<=":
                return l <= r
            if op == ">":
                return l > r
            if op == ">=":
                return l >= r
            if op == "&&":
                return bool(l) and bool(r)
            if op == "||":
                return bool(l) or bool(r)
            raise SIMPRuntimeError(f"unknown BinOp {op}")

        if isinstance(e, IteExpr):
            c = self.eval_expr(e.cond, sigma, H)
            return self.eval_expr(e.then_expr if c else e.else_expr, sigma, H)

        if isinstance(e, ArrLoad):
            a = self.eval_expr(e.arr, sigma, H)
            i = self.eval_expr(e.index, sigma, H)
            return a.select(i)

        if isinstance(e, ArrStoreExpr):
            a = self.eval_expr(e.arr, sigma, H)
            i = self.eval_expr(e.index, sigma, H)
            v = self.eval_expr(e.val, sigma, H)
            return a.store(i, v)

        if isinstance(e, PtrOffset):
            p = self.eval_expr(e.ptr, sigma, H)
            off = self.eval_expr(e.offset, sigma, H)
            return Ptr(p.loc + off)

        if isinstance(e, PtrDeref):
            p = self.eval_expr(e.ptr, sigma, H)
            return H[p.loc]

        if isinstance(e, BuiltinCall):
            args = [self.eval_expr(a, sigma, H) for a in e.args]
            fn = getattr(math, e.name)
            return fn(*args)

        if isinstance(e, FuncCall):
            callee = self.funcs[e.name]
            args = [self.eval_expr(a, sigma, H) for a in e.args]
            sub = {pname: aval for (pname, _), aval in zip(callee.params, args)}
            _, _, r = self.exec_stmts(callee.body, sub, H)
            assert isinstance(r, Ret), f"function {e.name} did not return"
            return r.value

        # ---- Host-Python extension (rigid, concretely executed) ---------- #
        if isinstance(e, PyModuleRef):
            return PyObj(self._module(self.import_map[e.alias]))

        if isinstance(e, PyAttr):
            ov = self.eval_expr(e.obj, sigma, H)
            return self._wrap(getattr(self._unwrap(ov), e.attr), e.ret_ty)

        if isinstance(e, PyCall):
            cv = self.eval_expr(e.callee, sigma, H)
            args = [self._unwrap(self.eval_expr(a, sigma, H)) for a in e.args]
            return self._wrap(self._unwrap(cv)(*args), e.ret_ty)

        if isinstance(e, PyMethodCall):
            ov = self.eval_expr(e.obj, sigma, H)
            args = [self._unwrap(self.eval_expr(a, sigma, H)) for a in e.args]
            method = getattr(self._unwrap(ov), e.method)
            return self._wrap(method(*args), e.ret_ty)

        if isinstance(e, PyBox):
            return PyObj(self.eval_expr(e.expr, sigma, H))

        if isinstance(e, PyUnbox):
            ov = self.eval_expr(e.expr, sigma, H)
            return self._convert(self._unwrap(ov), e.target_ty)

        # ---- Native strings & structs & dicts ---------------------------- #
        if isinstance(e, StrConcat):
            return self.eval_expr(e.left, sigma, H) + self.eval_expr(e.right, sigma, H)

        if isinstance(e, StrLen):
            return len(self.eval_expr(e.arg, sigma, H))

        if isinstance(e, StrSubstr):
            s = self.eval_expr(e.s, sigma, H)
            off = self.eval_expr(e.offset, sigma, H)
            ln = self.eval_expr(e.length, sigma, H)
            return s[off:off + ln]

        if isinstance(e, StructInit):
            fields = {fname: self.eval_expr(v, sigma, H) for fname, v in e.field_vals}
            return StructVal(e.struct_name, fields)

        if isinstance(e, DictInit):
            d = Dict({})
            for k, v in e.entries:
                d = d.store(self.eval_expr(k, sigma, H), self.eval_expr(v, sigma, H))
            return d

        if isinstance(e, FieldGet):
            sv = self.eval_expr(e.target, sigma, H)
            return sv.fields[e.field]

        if isinstance(e, FieldSet):
            sv = self.eval_expr(e.target, sigma, H)
            nv = self.eval_expr(e.val, sigma, H)
            new_fields = dict(sv.fields)
            new_fields[e.field] = nv
            return StructVal(sv.name, new_fields)

        raise SIMPRuntimeError(f"unhandled expr: {e}")

    # -- statement execution (§3.2) ---------------------------------------- #

    def exec_stmt(
        self, st: Stmt, sigma: dict[str, object], H: dict[int, object]
    ) -> tuple[dict[str, object], dict[int, object], Ret | None]:
        self._tick()

        if isinstance(st, LetDecl):
            sigma[st.name] = self.eval_expr(st.init, sigma, H)
            return sigma, H, None

        if isinstance(st, Assign):
            sigma[st.name] = self.eval_expr(st.expr, sigma, H)
            return sigma, H, None

        if isinstance(st, ArrStoreStmt):
            idx = self.eval_expr(st.index, sigma, H)
            val = self.eval_expr(st.val, sigma, H)
            sigma[st.arr_name] = sigma[st.arr_name].store(idx, val)
            return sigma, H, None

        if isinstance(st, PtrStoreStmt):
            p = self.eval_expr(st.ptr_expr, sigma, H)
            H[p.loc] = self.eval_expr(st.val, sigma, H)
            return sigma, H, None

        if isinstance(st, IfStmt):
            c = self.eval_expr(st.cond, sigma, H)
            branch = st.then_body if c else st.else_body
            return self.exec_stmts(branch, sigma, H)

        if isinstance(st, WhileStmt):
            while self.eval_expr(st.cond, sigma, H):
                sigma, H, r = self.exec_stmts(st.body, sigma, H)
                if r is not None:
                    return sigma, H, r
            return sigma, H, None

        if isinstance(st, ReturnStmt):
            return sigma, H, Ret(self.eval_expr(st.expr, sigma, H))

        raise SIMPRuntimeError(f"unhandled stmt: {st}")

    def exec_stmts(
        self, stmts: list[Stmt], sigma: dict[str, object], H: dict[int, object]
    ) -> tuple[dict[str, object], dict[int, object], Ret | None]:
        r: Ret | None = None
        for st in stmts:
            sigma, H, r = self.exec_stmt(st, sigma, H)
            if r is not None:
                return sigma, H, r
        return sigma, H, None

    # -- program execution (§3.3) ------------------------------------------ #

    def run(
        self,
        entry: str,
        args: list[object] | None = None,
        heap: dict[int, object] | None = None,
    ) -> object:
        fn = self.prog.get_func(entry)
        if args is None:
            args, heap = self.default_args(entry, heap)
        H = heap if heap is not None else {}
        sigma: dict[str, object] = {p: a for (p, _), a in zip(fn.params, args)}
        _, _, r = self.exec_stmts(fn.body, sigma, H)
        if not isinstance(r, Ret):
            raise SIMPRuntimeError(f"function {entry} terminated without a return value")
        return r.value

    def default_args(
        self, entry: str, heap: dict[int, object] | None = None
    ) -> tuple[list[object], dict[int, object]]:
        """Synthesize concrete default arguments (and a fresh heap) for ``entry``."""
        H = heap if heap is not None else {}
        fn = self.prog.get_func(entry)
        args = [self._default_value(pty, H) for _, pty in fn.params]
        return args, H

    def _default_value(self, ty: Type, H: dict[int, object]) -> object:
        if ty == BOOL:
            return False
        if ty == INT:
            return 0
        if ty == FLOAT:
            return 0.0
        if ty == STR:
            return ""
        if isinstance(ty, TArray):
            return Array({})
        if isinstance(ty, TDict):
            return Dict({})
        if isinstance(ty, TPtr):
            loc = self._fresh_loc()
            H[loc] = self._default_value(ty.elem, H)
            return Ptr(loc)
        if isinstance(ty, TPyObj):
            return PyObj(None)
        if isinstance(ty, TStruct):
            return StructVal(ty.name, {f: self._default_value(ft, H) for f, ft in ty.fields})
        raise SIMPRuntimeError(f"no default value for type {ty}")


# --------------------------------------------------------------------------- #
# Convenience API
# --------------------------------------------------------------------------- #


def interpret(
    prog: Program,
    entry: str = "main",
    args: list[object] | None = None,
    heap: dict[int, object] | None = None,
    max_steps: int = 1_000_000,
) -> object:
    """Execute ``entry`` in ``prog`` and return its result value.

    If ``args`` is ``None``, concrete default arguments (and a fresh heap) are
    synthesized from the parameter types.
    """
    return Interp(prog, max_steps=max_steps).run(entry, args, heap)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="simp.interpreter",
        description="Parse, typecheck, and concretely execute a .simp program.",
    )
    p.add_argument("file", help="input .simp source file")
    p.add_argument("-e", "--entry", default="main", help="entry function (default: main)")
    p.add_argument("--max-steps", type=int, default=1_000_000)
    args = p.parse_args(argv)

    from .parser import parse
    from .typecheck import typecheck_program

    try:
        prog = parse(Path(args.file).read_text())
        typecheck_program(prog)
        result = interpret(prog, args.entry, max_steps=args.max_steps)
    except (OSError, SIMPRuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())