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
    """Floating-point type (modeled as exact Real or IEEE-754 bit-exact depending on mode)."""

    bits: int = 32

    def __str__(self) -> str:
        return "float" if self.bits == 32 else f"f{self.bits}"


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
class TVector(CoreType):
    """Fixed-length SIMD / shader vector type (e.g., V2F32, V3U32, V4F32, V2Bool)."""

    elem: CoreType
    lanes: int

    def __str__(self) -> str:
        return f"vec<{self.elem}x{self.lanes}>"


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
    """Explicit Memory-SSA / effect state token for a memory region or global heap."""

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
BV16 = TBitVec(16)
BV32 = TBitVec(32)
BV64 = TBitVec(64)
F16 = TFloat(16)
FLOAT = TFloat(32)
F64 = TFloat(64)
STR = TStr()
MEM = TMem("global")
TOKEN = TMem("effect")
PYOBJ = TSym("PyObj")
VOID = TVoid()
