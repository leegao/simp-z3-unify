from dataclasses import dataclass


class CoreType:
    """Base class for all types in the minimal CoreSSA-IR kernel."""


@dataclass(frozen=True)
class TBool(CoreType):
    def __str__(self) -> str:
        return "bool"


@dataclass(frozen=True)
class TInt(CoreType):
    def __str__(self) -> str:
        return "int"


@dataclass(frozen=True)
class TBitVec(CoreType):
    width: int

    def __str__(self) -> str:
        return f"bv{self.width}"


@dataclass(frozen=True)
class TFloat(CoreType):
    """Exact real-modeled floating point (matching s-imp's Z3 RealSort)."""

    def __str__(self) -> str:
        return "float"


@dataclass(frozen=True)
class TStr(CoreType):
    def __str__(self) -> str:
        return "str"


@dataclass(frozen=True)
class TSym(CoreType):
    """Uninterpreted named sort for arbitrary symbols, handles, or opaque PL objects."""

    sort_name: str = "Sym"

    def __str__(self) -> str:
        return f"sym<{self.sort_name}>"


@dataclass(frozen=True)
class TPtr(CoreType):
    """Provenance-carrying pointer type with optional address space and TBAA tag."""

    elem: CoreType
    addr_space: int = 0
    tbaa_tag: str = "any"

    def __str__(self) -> str:
        tag = f", {self.tbaa_tag}" if self.tbaa_tag != "any" else ""
        space = f"@{self.addr_space}" if self.addr_space != 0 else ""
        return f"ptr{space}<{self.elem}{tag}>"


@dataclass(frozen=True)
class TMem(CoreType):
    """Explicit Memory-SSA state token for a memory region or global heap."""

    region: str = "global"

    def __str__(self) -> str:
        return f"mem<{self.region}>"


@dataclass(frozen=True)
class TArray(CoreType):
    key: CoreType
    elem: CoreType

    def __str__(self) -> str:
        return f"array<{self.key}, {self.elem}>"


@dataclass(frozen=True)
class TStruct(CoreType):
    name: str
    fields: tuple[tuple[str, CoreType], ...]

    def field_type(self, field_name: str) -> CoreType:
        for k, t in self.fields:
            if k == field_name:
                return t
        raise KeyError(f"Struct {self.name} has no field {field_name}")

    def __str__(self) -> str:
        return f"struct<{self.name}>"


@dataclass(frozen=True)
class TTuple(CoreType):
    elems: tuple[CoreType, ...]

    def __str__(self) -> str:
        inner = ", ".join(str(e) for e in self.elems)
        return f"({inner})"


@dataclass(frozen=True)
class TVoid(CoreType):
    def __str__(self) -> str:
        return "void"


BOOL = TBool()
INT = TInt()
BV32 = TBitVec(32)
BV64 = TBitVec(64)
FLOAT = TFloat()
STR = TStr()
MEM = TMem("global")
PYOBJ = TSym("PyObj")
VOID = TVoid()
