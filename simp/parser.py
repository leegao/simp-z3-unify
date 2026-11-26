"""Concrete parser for s-imp.

The grammar lives in ``simp/simp.lark`` (EBNF, Lark syntax) and the parse tree is
"discharged" directly into the dataclasses in :mod:`simp.ast`.

A few AST nodes share a single surface form and are therefore resolved in a
type-directed way while building the tree (mirroring ``typecheck_program``):

  * ``Var``        vs ``PyModuleRef``  -- a bare name that is an import alias.
  * ``FieldGet``   vs ``PyAttr``       -- ``e.f`` where ``e`` is a struct / pyobj.
  * ``BuiltinCall`` / ``FuncCall`` / ``PyCall`` / ``PyMethodCall`` -- ``callee(args)``.
  * ``StrLen`` / ``StrSubstr``        -- ``len(e)`` / ``substr(s, off, len)``.

Because those decisions need the *type* of sub-expressions, the builder threads a
type environment top-down (a plain bottom-up ``Transformer`` cannot see the types
of its children). The result is a fully-typed :class:`simp.ast.Program`.
"""

from __future__ import annotations

import ast as _pyast
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path

from lark import Lark, Token, Tree
from lark.exceptions import LarkError

from .ast import (
    ArrLoad,
    ArrStoreStmt,
    Assign,
    BinOp,
    BuiltinCall,
    Const,
    Expr,
    FieldGet,
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
    PyImportDecl,
    PyMethodCall,
    PyModuleRef,
    PyUnbox,
    ReturnStmt,
    StrConcat,
    StrLen,
    StrSubstr,
    StructDecl,
    StructInit,
    UnOp,
    Var,
    WhileStmt,
)
from .types import (
    BOOL,
    BUILTIN_SIGS,
    FLOAT,
    INT,
    PYOBJ,
    STR,
    VOID,
    TArray,
    TPtr,
    TPyObj,
    TStruct,
    Type,
)

# --------------------------------------------------------------------------- #
# Grammar (loaded from simp/simp.lark)
# --------------------------------------------------------------------------- #

GRAMMAR = (Path(__file__).with_name("simp.lark")).read_text()


class SIMPParseError(Exception):
    """Raised when source text cannot be parsed or discharged into the AST."""


# --------------------------------------------------------------------------- #
# Parser / AST builder
# --------------------------------------------------------------------------- #


class SIMPParser:
    """Parse s-imp source text into a typed :class:`Program`."""

    def __init__(self, parser: str = "earley") -> None:
        self._lark = Lark(GRAMMAR, parser=parser, start="program")
        # Symbol tables, populated per-parse in ``parse``.
        self._imports: dict[str, str] = {}      # alias -> module
        self._structs: dict[str, TStruct] = {}  # name  -> struct type
        self._func_sigs: dict[str, tuple[list[Type], Type]] = {}

    # -- public API -------------------------------------------------------- #

    def parse(self, src: str) -> Program:
        tree = self._lark.parse(src)
        top = list(tree.children)  # ``?top_level`` inlines each decl

        # Pass 1: collect symbol tables so expressions can be resolved.
        self._imports = {}
        self._structs = {}
        self._func_sigs = {}
        for node in top:
            if not isinstance(node, Tree):
                continue
            match node.data:
                case "import_decl":
                    module, alias = self._import_parts(node)
                    self._imports[alias] = module
                case "struct_decl":
                    name, fields = self._struct_parts(node)
                    self._structs[name] = TStruct(name, fields)
                case "func_decl":
                    name, params, ret_ty, _ = self._func_parts(node)
                    if name in self._func_sigs:
                        raise SIMPParseError(f"duplicate function: {name}")
                    self._func_sigs[name] = ([t for _, t in params], ret_ty)

        # Pass 2: build the actual AST nodes.
        imports: list[PyImportDecl] = []
        structs: list[StructDecl] = []
        funcs: list[FuncDecl] = []
        for node in top:
            if not isinstance(node, Tree):
                continue
            match node.data:
                case "import_decl":
                    module, alias = self._import_parts(node)
                    imports.append(PyImportDecl(module, alias))
                case "struct_decl":
                    name, fields = self._struct_parts(node)
                    structs.append(StructDecl(name, fields))
                case "func_decl":
                    funcs.append(self._func(node))
                case _:  # pragma: no cover - grammar guarantees only these three
                    raise SIMPParseError(f"unexpected top-level node: {node.data}")

        return Program(funcs, imports=imports, structs=structs)

    # -- top-level helpers ------------------------------------------------- #

    @staticmethod
    def _import_parts(node: Tree) -> tuple[str, str]:
        dotted, alias = node.children
        module = ".".join(str(p) for p in dotted.children)
        return module, str(alias)

    def _struct_parts(self, node: Tree) -> tuple[str, tuple[tuple[str, Type], ...]]:
        name = str(node.children[0])
        fields: list[tuple[str, Type]] = []
        for field in node.children[1:]:
            fields.append((str(field.children[0]), self._type(field.children[1])))
        return name, tuple(fields)

    def _func_parts(
        self, node: Tree
    ) -> tuple[str, list[tuple[str, Type]], Type, Tree]:
        name = str(node.children[0])
        params: list[tuple[str, Type]] = []
        ret_ty: Type = VOID
        block: Tree | None = None
        for child in node.children[1:]:
            if not isinstance(child, Tree):
                continue
            match child.data:
                case "params":
                    for p in child.children:
                        params.append((str(p.children[0]), self._type(p.children[1])))
                case "block":
                    block = child
                case _ if child.data.startswith("t_"):
                    ret_ty = self._type(child)
        assert block is not None, "func_decl without body"
        return name, params, ret_ty, block

    def _func(self, node: Tree) -> FuncDecl:
        name, params, ret_ty, block = self._func_parts(node)
        env: dict[str, Type] = {p: t for p, t in params}
        body = self._stmts(block.children, env, ret_ty)
        return FuncDecl(name=name, params=params, ret_ty=ret_ty, body=body)

    # -- types ------------------------------------------------------------- #

    def _type(self, node: Tree) -> Type:
        match node.data:
            case "t_bool":
                return BOOL
            case "t_int":
                return INT
            case "t_float":
                return FLOAT
            case "t_str":
                return STR
            case "t_void":
                return VOID
            case "t_ptr":
                return TPtr(self._type(node.children[0]))
            case "t_array":
                return TArray(self._type(node.children[0]))
            case "t_pyobj":
                tag = str(node.children[0]) if node.children else "object"
                return TPyObj(tag)
            case "t_struct":
                sname = str(node.children[0])
                if sname not in self._structs:
                    raise SIMPParseError(f"unknown struct type: {sname}")
                return self._structs[sname]
            case _:
                raise SIMPParseError(f"unknown type node: {node.data}")

    # -- statements -------------------------------------------------------- #

    def _stmts(self, nodes: list, env: dict[str, Type], ret_ty: Type) -> list:
        return [self._stmt(node, env, ret_ty) for node in nodes]

    def _stmt(self, node: Tree, env: dict[str, Type], ret_ty: Type):
        match node.data:
            case "let_stmt":
                name = str(node.children[0])
                ty = self._type(node.children[1])
                init, _ = self._expr(node.children[2], env)
                env[name] = ty
                return LetDecl(name, ty, init)
            case "assign_stmt":
                name = str(node.children[0])
                val, _ = self._expr(node.children[1], env)
                return Assign(name, val)
            case "arr_store_stmt":
                name = str(node.children[0])
                idx, _ = self._expr(node.children[1], env)
                val, _ = self._expr(node.children[2], env)
                return ArrStoreStmt(name, idx, val)
            case "ptr_store_stmt":
                ptr, _ = self._expr(node.children[0], env)
                val, _ = self._expr(node.children[1], env)
                return PtrStoreStmt(ptr, val)
            case "if_stmt":
                cond, _ = self._expr(node.children[0], env)
                then_block = node.children[1]
                else_block = node.children[2] if len(node.children) > 2 else None
                then_body = self._stmts(then_block.children, dict(env), ret_ty)
                else_body = (
                    self._stmts(else_block.children, dict(env), ret_ty)
                    if else_block is not None
                    else []
                )
                return IfStmt(cond, then_body, else_body)
            case "while_stmt":
                cond, _ = self._expr(self._child_expr(node, "block"), env)
                block = self._child_block(node)
                unroll = self._child_int(node)
                body = self._stmts(block.children, dict(env), ret_ty)
                return WhileStmt(cond, body, unroll_bound=unroll if unroll is not None else 4)
            case "return_stmt":
                val, _ = self._expr(node.children[0], env)
                return ReturnStmt(val)
            case _:
                raise SIMPParseError(f"unknown statement node: {node.data}")

    @staticmethod
    def _child_block(node: Tree) -> Tree:
        for c in node.children:
            if isinstance(c, Tree) and c.data == "block":
                return c
        raise SIMPParseError("missing block")

    @staticmethod
    def _child_int(node: Tree) -> int | None:
        for c in node.children:
            if isinstance(c, Token) and c.type == "INT":
                return int(c)
        return None

    @staticmethod
    def _child_expr(node: Tree, exclude: str) -> Tree:
        for c in node.children:
            if isinstance(c, Tree) and c.data != exclude:
                return c
        raise SIMPParseError("missing expression")

    # -- expressions ------------------------------------------------------- #

    def _expr(self, node: Tree, env: dict[str, Type]) -> tuple[Expr, Type]:
        match node.data:
            # ``postfix: primary`` base case -- just unwrap.
            case "postfix":
                return self._expr(node.children[0], env)

            case "var":
                name = str(node.children[0])
                if name in self._imports:
                    return PyModuleRef(name), TPyObj(self._imports[name])
                if name not in env:
                    raise SIMPParseError(f"undefined variable: {name}")
                return Var(name), env[name]

            case "true" | "false":
                return Const(node.data == "true", BOOL), BOOL
            case "int":
                return Const(int(str(node.children[0])), INT), INT
            case "float":
                return Const(float(str(node.children[0])), FLOAT), FLOAT
            case "string":
                return Const(_pyast.literal_eval(str(node.children[0])), STR), STR

            case "neg":
                e, t = self._expr(node.children[0], env)
                return UnOp("-", e), t
            case "not":
                e, _ = self._expr(node.children[0], env)
                return UnOp("!", e), BOOL
            case "deref":
                e, t = self._expr(node.children[0], env)
                if not isinstance(t, TPtr):
                    raise SIMPParseError(f"dereference of non-pointer type: {t}")
                return PtrDeref(e), t.elem

            case (
                "add" | "sub" | "mul" | "div" | "strcat" | "ptr_offset"
                | "lt" | "le" | "gt" | "ge" | "eq" | "ne" | "or_" | "and_"
            ):
                return self._binop(node.data, node, env)

            case "ite":
                cond, _ = self._expr(node.children[0], env)
                then_e, tt = self._expr(node.children[1], env)
                else_e, _ = self._expr(node.children[2], env)
                return IteExpr(cond, then_e, else_e), tt

            case "arr_load":
                arr, aty = self._expr(node.children[0], env)
                idx, _ = self._expr(node.children[1], env)
                if not isinstance(aty, TArray):
                    raise SIMPParseError(f"indexing non-array type: {aty}")
                return ArrLoad(arr, idx), aty.elem

            case "attr":
                obj, oty = self._expr(node.children[0], env)
                name = str(node.children[1])
                if isinstance(oty, TStruct):
                    return FieldGet(obj, name), oty.field_type(name)
                if isinstance(oty, TPyObj):
                    return PyAttr(obj, name), PYOBJ
                raise SIMPParseError(f"attribute access on non-struct/pyobj: {oty}")

            case "struct_init":
                sname = str(node.children[0])
                if sname not in self._structs:
                    raise SIMPParseError(f"unknown struct: {sname}")
                st = self._structs[sname]
                field_vals: list[tuple[str, Expr]] = []
                if len(node.children) > 1:
                    for fld in node.children[1].children:
                        field_vals.append(
                            (str(fld.children[0]), self._expr(fld.children[1], env)[0])
                        )
                return StructInit(sname, tuple(field_vals)), st

            case "box":
                e, t = self._expr(node.children[0], env)
                if t not in (BOOL, INT, FLOAT):
                    raise SIMPParseError(f"box requires primitive, got {t}")
                return PyBox(e), PYOBJ

            case "unbox":
                target = self._type(node.children[0])
                e, t = self._expr(node.children[1], env)
                if not isinstance(t, TPyObj) or target not in (BOOL, INT, FLOAT):
                    raise SIMPParseError(
                        f"unbox requires pyobj -> primitive, got {t} -> {target}"
                    )
                return PyUnbox(e, target), target

            case "call":
                return self._call(node, env)

            case _:
                raise SIMPParseError(f"unknown expression node: {node.data}")

    def _binop(self, d: str, node: Tree, env: dict[str, Type]) -> tuple[Expr, Type]:
        left, lt = self._expr(node.children[0], env)
        right, _rt = self._expr(node.children[1], env)

        match d:
            case "add":
                return BinOp("+", left, right), lt
            case "sub":
                return BinOp("-", left, right), lt
            case "mul":
                return BinOp("*", left, right), lt
            case "div":
                return BinOp("/", left, right), lt
            case "strcat":
                return StrConcat(left, right), STR
            case "ptr_offset":
                if not isinstance(lt, TPtr):
                    raise SIMPParseError(f"pointer offset on non-pointer: {lt}")
                return PtrOffset(left, right), lt
            case "lt" | "le" | "gt" | "ge":
                op = {"lt": "<", "le": "<=", "gt": ">", "ge": ">="}[d]
                return BinOp(op, left, right), BOOL
            case "eq" | "ne":
                op = "==" if d == "eq" else "!="
                return BinOp(op, left, right), BOOL
            case "or_" | "and_":
                op = "||" if d == "or_" else "&&"
                return BinOp(op, left, right), BOOL
            case _:  # pragma: no cover
                raise SIMPParseError(f"unknown binop: {d}")

    def _call(self, node: Tree, env: dict[str, Type]) -> tuple[Expr, Type]:
        callee = node.children[0]
        args = [
            self._expr(a, env)[0]
            for a in (node.children[1].children if len(node.children) > 1 else [])
        ]

        match callee:
            # Bare-name callee: builtin / user function / pyobj variable / string builtins.
            case Tree(data="postfix", children=[Tree(data="var", children=[Token() as name_tok])]):
                name = str(name_tok)
                match name:
                    case "len" if len(args) == 1:
                        return StrLen(args[0]), INT
                    case "substr" if len(args) == 3:
                        return StrSubstr(args[0], args[1], args[2]), STR
                if name in BUILTIN_SIGS:
                    ptys, rty = BUILTIN_SIGS[name]
                    if len(args) != len(ptys):
                        raise SIMPParseError(f"arity mismatch for builtin {name}")
                    return BuiltinCall(name, tuple(args)), rty
                if name in self._func_sigs:
                    ptys, rty = self._func_sigs[name]
                    if len(args) != len(ptys):
                        raise SIMPParseError(f"arity mismatch for function {name}")
                    return FuncCall(name, tuple(args)), rty
                if name in env and isinstance(env[name], TPyObj):
                    return PyCall(Var(name), tuple(args)), PYOBJ
                raise SIMPParseError(f"call to unknown name: {name}")

            # Attribute callee: module function (PyCall) vs method on a pyobj value.
            case Tree(data="attr", children=[obj_node, Token() as method_tok]):
                obj, oty = self._expr(obj_node, env)
                method = str(method_tok)
                if not isinstance(oty, TPyObj):
                    raise SIMPParseError(f"call on non-pyobj attribute: {oty}")
                if isinstance(obj, PyModuleRef):
                    return PyCall(PyAttr(obj, method), tuple(args)), PYOBJ
                return PyMethodCall(obj, method, tuple(args)), PYOBJ

            # General pyobj callee expression (e.g. result of another call).
            case _:
                callee_e, callee_t = self._expr(callee, env)
                if not isinstance(callee_t, TPyObj):
                    raise SIMPParseError(f"call of non-pyobj callee: {callee_t}")
                return PyCall(callee_e, tuple(args)), PYOBJ


# --------------------------------------------------------------------------- #
# Convenience API
# --------------------------------------------------------------------------- #

_DEFAULT_PARSER: SIMPParser | None = None


def parse(src: str) -> Program:
    """Parse s-imp source text into a typed :class:`Program`."""
    global _DEFAULT_PARSER
    if _DEFAULT_PARSER is None:
        _DEFAULT_PARSER = SIMPParser()
    return _DEFAULT_PARSER.parse(src)


# --------------------------------------------------------------------------- #
# Pretty-printing (ast.dump-style) + CLI
# --------------------------------------------------------------------------- #


def format_ast(obj: object, indent: int = 2, level: int = 0) -> str:
    """Render a (dataclass) AST as an indented, ``ast.dump``-style string."""
    pad = " " * (indent * level)
    pad_in = " " * (indent * (level + 1))

    if is_dataclass(obj) and not isinstance(obj, type):
        name = type(obj).__name__
        fs = fields(obj)
        if not fs:
            return f"{name}()"
        lines = [f"{name}("]
        for f in fs:
            lines.append(f"{pad_in}{f.name}={format_ast(getattr(obj, f.name), indent, level + 1)},")
        lines.append(f"{pad})")
        return "\n".join(lines)

    if isinstance(obj, list):
        if not obj:
            return "[]"
        lines = ["["]
        for item in obj:
            lines.append(f"{pad_in}{format_ast(item, indent, level + 1)},")
        lines.append(f"{pad}]")
        return "\n".join(lines)

    if isinstance(obj, tuple):
        if not obj:
            return "()"
        lines = ["("]
        for item in obj:
            lines.append(f"{pad_in}{format_ast(item, indent, level + 1)},")
        lines.append(f"{pad})")
        return "\n".join(lines)

    return repr(obj)


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="simp.parser",
        description="Parse a .simp file and pretty-print its AST.",
    )
    p.add_argument("file", help="input .simp source file")
    p.add_argument("-o", "--output", help="write to this file instead of stdout")
    p.add_argument("--indent", type=int, default=2, help="indentation width (default: 2)")
    args = p.parse_args(argv)

    try:
        src = Path(args.file).read_text()
        prog = parse(src)
    except (OSError, SIMPParseError, LarkError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    out = format_ast(prog, indent=args.indent)
    if args.output:
        Path(args.output).write_text(out + "\n")
    else:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())