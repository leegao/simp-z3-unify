from dataclasses import dataclass, field

from .simp_types import (
    PYOBJ,
    TStruct,
    Type,
)


class Expr:
    pass


@dataclass(frozen=True)
class Const(Expr):
    value: bool | int | float | str
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
    args: tuple[Expr, ...]


@dataclass(frozen=True)
class FuncCall(Expr):
    name: str
    args: tuple[Expr, ...]


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
    args: tuple[Expr, ...]
    ret_ty: Type = PYOBJ


@dataclass(frozen=True)
class PyMethodCall(Expr):
    """Rigid host-Python method invocation: `obj.<method>(args...) : ret_ty`."""
    obj: Expr
    method: str
    args: tuple[Expr, ...]
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
    fields: tuple[tuple[str, Type], ...]

    def as_type(self) -> TStruct:
        return TStruct(self.name, self.fields)


@dataclass(frozen=True)
class StructInit(Expr):
    struct_name: str
    field_vals: tuple[tuple[str, Expr], ...]


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
    then_body: list[Stmt]
    else_body: list[Stmt] = field(default_factory=list)


@dataclass
class WhileStmt(Stmt):
    cond: Expr
    body: list[Stmt]
    unroll_bound: int = 4


@dataclass
class ReturnStmt(Stmt):
    expr: Expr


@dataclass
class FuncDecl:
    name: str
    params: list[tuple[str, Type]]
    ret_ty: Type
    body: list[Stmt]


@dataclass
class Program:
    funcs: list[FuncDecl]
    imports: list[PyImportDecl] = field(default_factory=list)
    structs: list[StructDecl] = field(default_factory=list)

    def get_func(self, name: str) -> FuncDecl:
        for fn in self.funcs:
            if fn.name == name:
                return fn
        raise KeyError(f"Unknown function: {name}")
