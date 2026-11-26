from dataclasses import dataclass
from typing import Dict, List


class Type:
    pass


@dataclass(frozen=True)
class TBool(Type):
    def __str__(self) -> str:
        return "bool"


@dataclass(frozen=True)
class TInt(Type):
    def __str__(self) -> str:
        return "int"


@dataclass(frozen=True)
class TFloat(Type):
    """Modeled as exact Real in Z3 (representing fp64 values)."""
    def __str__(self) -> str:
        return "float"


@dataclass(frozen=True)
class TPtr(Type):
    elem: Type

    def __str__(self) -> str:
        return f"ptr<{self.elem}>"


@dataclass(frozen=True)
class TArray(Type):
    elem: Type

    def __str__(self) -> str:
        return f"array<{self.elem}>"


@dataclass(frozen=True)
class TPyObj(Type):
    """Opaque host-Python object handle (modeled as uninterpreted PyObj sort in Z3)."""
    tag: str = "object"

    def __str__(self) -> str:
        return f"pyobj<{self.tag}>"


@dataclass(frozen=True)
class TString(Type):
    """Native UTF-8 string (modeled as Z3 StringSort)."""
    def __str__(self) -> str:
        return "str"


@dataclass(frozen=True)
class TStruct(Type):
    """Named product struct type (modeled as Z3 algebraic Datatype)."""
    name: str
    fields: tuple  # Tuple[Tuple[str, Type], ...]

    def field_type(self, fname: str) -> Type:
        for k, t in self.fields:
            if k == fname:
                return t
        raise KeyError(f"Struct {self.name} has no field {fname}")

    def __str__(self) -> str:
        return f"struct {self.name}"


@dataclass(frozen=True)
class TVoid(Type):
    def __str__(self) -> str:
        return "void"


BOOL = TBool()
INT = TInt()
FLOAT = TFloat()
PYOBJ = TPyObj("object")
STR = TString()
VOID = TVoid()

# Transcendental builtins: float -> float (modeled as opaque UFs in Z3)
BUILTIN_SIGS: Dict[str, tuple[List[Type], Type]] = {
    "exp": ([FLOAT], FLOAT),
    "log": ([FLOAT], FLOAT),
    "sin": ([FLOAT], FLOAT),
    "cos": ([FLOAT], FLOAT),
    "sqrt": ([FLOAT], FLOAT),
}
