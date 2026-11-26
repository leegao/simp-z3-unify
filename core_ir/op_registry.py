from dataclasses import dataclass, field
from enum import Flag, auto
from typing import Any, Callable


class OpTrait(Flag):
    """Algebraic, structural, and semantic traits annotating CoreSSA-IR operations."""

    NONE = 0
    ASSOC = auto()             # f(f(a, b), c) == f(a, f(b, c))
    COMM = auto()              # f(a, b) == f(b, a)
    IDEMPOTENT = auto()        # f(a, a) == a
    INVOLUTION = auto()        # unary: f(f(a)) == a
    SELF_INVERSE = auto()      # binary: f(f(a, b), b) == a and f(a, a) == identity
    TRANSPARENT_MOVE = auto()  # unary SSA copy/move/cast: f(x) == x (peeled during AU)
    VALUE_INTRO = auto()       # SSA value introduction wrapper (e.g., ssa.intro, freeze)
    RIGID_UF = auto()          # uninterpreted function / opaque call (congruence only)
    MEM_READ = auto()          # memory load: (mem, ptr) -> val
    MEM_WRITE = auto()         # memory store: (mem, ptr, val) -> mem
    MEM_ALLOC = auto()         # provenance-fresh region allocation
    PTR_OFFSET = auto()        # pointer arithmetic: gep(gep(p, a), b) == gep(p, a + b)
    LANE_OP = auto()           # GPU subgroup / quad / lane-dependent operation
    EFFECT_BARRIER = auto()    # ordered memory/control barrier, atomic, or discard effect
    BITCAST = auto()           # representation-preserving bitcast
    PURE = auto()              # side-effect-free operation

    AC = ASSOC | COMM


@dataclass(frozen=True)
class InverseSpec:
    """Declares an inverse relationship for a binary or unary operator."""

    forward_op: str
    inverse_binop: str | None = None
    unary_inverse: str | None = None
    nonzero_rhs: bool = False


@dataclass(frozen=True)
class OpSpec:
    """Metadata, equational annotations, optional Z3-counterpart aliasing, and
    optional concrete constant-folding evaluator for a single CoreSSA-IR operation.
    """

    name: str
    traits: OpTrait = OpTrait.PURE
    identity: object = None
    has_identity: bool = False
    absorbing: object = None
    has_absorbing: bool = False
    inverse: InverseSpec | None = None
    z3_kind: str | None = None
    z3_builder: Callable[..., Any] | None = field(default=None, compare=False, hash=False)
    concrete_eval: Callable[..., Any] | None = field(default=None, compare=False, hash=False)

    @property
    def is_ac(self) -> bool:
        return OpTrait.AC in self.traits

    @property
    def is_assoc_only(self) -> bool:
        return (OpTrait.ASSOC in self.traits) and (OpTrait.COMM not in self.traits)

    @property
    def is_comm(self) -> bool:
        return OpTrait.COMM in self.traits

    @property
    def is_transparent(self) -> bool:
        return bool(self.traits & (OpTrait.TRANSPARENT_MOVE | OpTrait.VALUE_INTRO))

    @property
    def is_involution(self) -> bool:
        return OpTrait.INVOLUTION in self.traits

    @property
    def is_self_inverse(self) -> bool:
        return OpTrait.SELF_INVERSE in self.traits

    @property
    def is_rigid_uf(self) -> bool:
        return OpTrait.RIGID_UF in self.traits


class OpRegistry:
    """Extensible registry of equational operation annotations, Z3 aliases, and concrete folders."""

    def __init__(self):
        self._ops: dict[str, OpSpec] = {}
        self._inv_binop_to_forward: dict[str, InverseSpec] = {}
        self._inv_unary_to_forward: dict[str, InverseSpec] = {}

    def register(self, spec: OpSpec) -> None:
        self._ops[spec.name] = spec
        if spec.inverse is not None:
            if spec.inverse.inverse_binop:
                self._inv_binop_to_forward[spec.inverse.inverse_binop] = spec.inverse
            if spec.inverse.unary_inverse:
                self._inv_unary_to_forward[spec.inverse.unary_inverse] = spec.inverse

    def alias_z3(self, op_name: str, builder: Callable[..., Any]) -> None:
        """Attach or override a Z3-counterpart builder for an existing or new op."""
        cur = self.get(op_name)
        self.register(
            OpSpec(
                name=cur.name,
                traits=cur.traits,
                identity=cur.identity,
                has_identity=cur.has_identity,
                absorbing=cur.absorbing,
                has_absorbing=cur.has_absorbing,
                inverse=cur.inverse,
                z3_kind=cur.z3_kind,
                z3_builder=builder,
                concrete_eval=cur.concrete_eval,
            )
        )

    def get(self, name: str) -> OpSpec:
        if name in self._ops:
            return self._ops[name]
        return OpSpec(name=name, traits=OpTrait.RIGID_UF | OpTrait.PURE, z3_kind=None)

    def has(self, name: str) -> bool:
        return name in self._ops

    def inverse_for_binop(self, binop_name: str) -> InverseSpec | None:
        return self._inv_binop_to_forward.get(binop_name)

    def inverse_for_unary(self, unary_name: str) -> InverseSpec | None:
        return self._inv_unary_to_forward.get(unary_name)

    def clone(self) -> "OpRegistry":
        reg = OpRegistry()
        for spec in self._ops.values():
            reg.register(spec)
        return reg

    def list_ops(self) -> list[OpSpec]:
        return list(self._ops.values())


def build_default_registry() -> OpRegistry:
    """Construct the standard CoreSSA-IR kernel operation registry."""
    reg = OpRegistry()

    for mv in ("ssa.mov", "ssa.copy", "ssa.intro", "ssa.cast_noop", "ssa.freeze"):
        reg.register(OpSpec(mv, OpTrait.TRANSPARENT_MOVE | OpTrait.PURE, z3_kind="identity"))

    add_inv = InverseSpec(forward_op="add", inverse_binop="sub", unary_inverse="neg", nonzero_rhs=False)
    reg.register(OpSpec("add", OpTrait.AC | OpTrait.PURE, identity=0, has_identity=True, inverse=add_inv, z3_kind="add"))
    reg.register(OpSpec("sub", OpTrait.PURE, identity=0, has_identity=True, inverse=add_inv, z3_kind="sub"))
    reg.register(OpSpec("neg", OpTrait.INVOLUTION | OpTrait.PURE, inverse=add_inv, z3_kind="neg"))

    mul_inv = InverseSpec(forward_op="mul", inverse_binop="div", unary_inverse="recip", nonzero_rhs=True)
    reg.register(
        OpSpec(
            "mul",
            OpTrait.AC | OpTrait.PURE,
            identity=1,
            has_identity=True,
            absorbing=0,
            has_absorbing=True,
            inverse=mul_inv,
            z3_kind="mul",
        )
    )
    reg.register(OpSpec("div", OpTrait.PURE, identity=1, has_identity=True, inverse=mul_inv, z3_kind="div"))
    reg.register(OpSpec("recip", OpTrait.INVOLUTION | OpTrait.PURE, inverse=mul_inv, z3_kind="recip"))

    reg.register(
        OpSpec(
            "and",
            OpTrait.AC | OpTrait.IDEMPOTENT | OpTrait.PURE,
            identity=True,
            has_identity=True,
            absorbing=False,
            has_absorbing=True,
            z3_kind="and",
        )
    )
    reg.register(
        OpSpec(
            "or",
            OpTrait.AC | OpTrait.IDEMPOTENT | OpTrait.PURE,
            identity=False,
            has_identity=True,
            absorbing=True,
            has_absorbing=True,
            z3_kind="or",
        )
    )
    reg.register(OpSpec("not", OpTrait.INVOLUTION | OpTrait.PURE, z3_kind="not"))
    reg.register(
        OpSpec(
            "xor",
            OpTrait.AC | OpTrait.SELF_INVERSE | OpTrait.PURE,
            identity=False,
            has_identity=True,
            z3_kind="xor",
        )
    )

    bvadd_inv = InverseSpec(forward_op="bv.add", inverse_binop="bv.sub", unary_inverse="bv.neg", nonzero_rhs=False)
    reg.register(OpSpec("bv.add", OpTrait.AC | OpTrait.PURE, identity=0, has_identity=True, inverse=bvadd_inv, z3_kind="bv_add"))
    reg.register(OpSpec("bv.sub", OpTrait.PURE, identity=0, has_identity=True, inverse=bvadd_inv, z3_kind="bv_sub"))
    reg.register(OpSpec("bv.neg", OpTrait.INVOLUTION | OpTrait.PURE, inverse=bvadd_inv, z3_kind="bv_neg"))
    reg.register(OpSpec("bv.mul", OpTrait.AC | OpTrait.PURE, identity=1, has_identity=True, absorbing=0, has_absorbing=True, z3_kind="bv_mul"))
    reg.register(OpSpec("bv.xor", OpTrait.AC | OpTrait.SELF_INVERSE | OpTrait.PURE, identity=0, has_identity=True, z3_kind="bv_xor"))
    reg.register(OpSpec("bv.and", OpTrait.AC | OpTrait.IDEMPOTENT | OpTrait.PURE, absorbing=0, has_absorbing=True, z3_kind="bv_and"))
    reg.register(OpSpec("bv.or", OpTrait.AC | OpTrait.IDEMPOTENT | OpTrait.PURE, identity=0, has_identity=True, z3_kind="bv_or"))
    reg.register(OpSpec("bv.not", OpTrait.INVOLUTION | OpTrait.PURE, z3_kind="bv_not"))

    reg.register(OpSpec("str.concat", OpTrait.ASSOC | OpTrait.PURE, identity="", has_identity=True, z3_kind="str_concat"))
    reg.register(OpSpec("str.len", OpTrait.PURE, z3_kind="str_len"))
    reg.register(OpSpec("str.substr", OpTrait.PURE, z3_kind="str_substr"))
    reg.register(OpSpec("mat.mul", OpTrait.ASSOC | OpTrait.RIGID_UF | OpTrait.PURE, z3_kind=None))
    reg.register(OpSpec("seq.compose", OpTrait.ASSOC | OpTrait.RIGID_UF | OpTrait.PURE, z3_kind=None))

    reg.register(OpSpec("eq", OpTrait.COMM | OpTrait.PURE, z3_kind="eq"))
    reg.register(OpSpec("neq", OpTrait.COMM | OpTrait.PURE, z3_kind="neq"))
    reg.register(OpSpec("lt", OpTrait.PURE, z3_kind="lt"))
    reg.register(OpSpec("le", OpTrait.PURE, z3_kind="le"))
    reg.register(OpSpec("gt", OpTrait.PURE, z3_kind="gt"))
    reg.register(OpSpec("ge", OpTrait.PURE, z3_kind="ge"))

    reg.register(OpSpec("ptr.gep", OpTrait.PTR_OFFSET | OpTrait.PURE, z3_kind="ptr_gep"))
    reg.register(OpSpec("mem.load", OpTrait.MEM_READ | OpTrait.PURE, z3_kind="mem_load"))
    reg.register(OpSpec("mem.store", OpTrait.MEM_WRITE | OpTrait.PURE, z3_kind="mem_store"))
    reg.register(OpSpec("mem.alloc", OpTrait.MEM_ALLOC | OpTrait.PURE, z3_kind="mem_alloc"))

    reg.register(OpSpec("arr.load", OpTrait.PURE, z3_kind="arr_load"))
    reg.register(OpSpec("arr.store", OpTrait.PURE, z3_kind="arr_store"))
    reg.register(OpSpec("struct.mk", OpTrait.PURE, z3_kind="struct_mk"))
    reg.register(OpSpec("struct.get", OpTrait.PURE, z3_kind="struct_get"))
    reg.register(OpSpec("struct.set", OpTrait.PURE, z3_kind="struct_set"))
    reg.register(OpSpec("tuple.mk", OpTrait.PURE, z3_kind="tuple_mk"))
    reg.register(OpSpec("tuple.get", OpTrait.PURE, z3_kind="tuple_get"))

    for fn_name in ("uf.exp", "uf.log", "uf.sin", "uf.cos", "uf.sqrt"):
        reg.register(OpSpec(fn_name, OpTrait.RIGID_UF | OpTrait.PURE, z3_kind=None))

    return reg
