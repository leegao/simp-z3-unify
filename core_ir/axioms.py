from dataclasses import dataclass, field
from typing import Any, Callable

from .ir import ConstVal, CoreVal, OpApp, free_param_names
from .op_registry import OpRegistry


@dataclass
class ExtensionalAxiom:
    """A sound extensional axiom attached to a specific OpRegistry / dialect.

    Can operate at two levels:
      1. `rewrite_fn(val, reg) -> CoreVal | None`:
         A sound term-level simplification applied during canonicalization
         (e.g., `Pack64(lo(Split64(x)), hi(Split64(x))) -> x`, constant folding,
         `VecExtract(VecConstruct(v0, v1), 1) -> v1`).
      2. `z3_axiom_fn(z3_backend, lhs, rhs) -> list[z3.BoolRef]`:
         Optional extra Z3 constraints asserted when discharging residual leaf holes.
    """

    name: str
    description: str
    rewrite_fn: Callable[[CoreVal, OpRegistry], CoreVal | None] | None = None
    z3_axiom_fn: Callable[[Any, CoreVal, CoreVal], list[Any]] | None = None
    dialect: str = "core"


class AxiomStore:
    """Inductively extensible store of known-true extensional axioms."""

    def __init__(self, axioms: list[ExtensionalAxiom] | None = None):
        self._axioms: dict[str, ExtensionalAxiom] = {}
        for ax in axioms or []:
            self.add(ax)

    def add(self, axiom: ExtensionalAxiom) -> None:
        self._axioms[axiom.name] = axiom

    def remove(self, name: str) -> None:
        self._axioms.pop(name, None)

    def has(self, name: str) -> bool:
        return name in self._axioms

    def list_axioms(self) -> list[ExtensionalAxiom]:
        return list(self._axioms.values())

    def clone(self) -> "AxiomStore":
        return AxiomStore(list(self._axioms.values()))

    def apply_rewrites(self, v: CoreVal, reg: OpRegistry) -> tuple[CoreVal, list[str]]:
        """Apply all registered extensional rewrite axioms to a node until fixed point."""
        fired: list[str] = []
        cur = v
        for _ in range(8):
            changed = False
            for ax in self._axioms.values():
                if ax.rewrite_fn is not None:
                    nxt = ax.rewrite_fn(cur, reg)
                    if nxt is not None and nxt != cur:
                        cur = nxt
                        fired.append(ax.name)
                        changed = True
                        break
            if not changed:
                break
        return cur, fired

    def collect_z3_axioms(self, z3_backend: Any, lhs: CoreVal, rhs: CoreVal) -> list[Any]:
        out: list[Any] = []
        for ax in self._axioms.values():
            if ax.z3_axiom_fn is not None:
                out.extend(ax.z3_axiom_fn(z3_backend, lhs, rhs))
        return out


# ---------------------------------------------------------------------------
# Standard GPU / Valhall SSA-IR / SPIR-V Extensional Axioms
# ---------------------------------------------------------------------------

def _rw_const_fold(v: CoreVal, reg: OpRegistry) -> CoreVal | None:
    if not isinstance(v, OpApp):
        return None
    spec = reg.get(v.op)
    if spec.concrete_eval is None:
        return None
    if not all(isinstance(a, ConstVal) for a in v.args):
        return None
    try:
        res = spec.concrete_eval([a.value for a in v.args], v.attrs, v.ty)  # type: ignore[union-attr]
    except Exception:
        return None
    if res is None:
        return None
    return ConstVal(res, v.ty)


def _rw_pack64_split64(v: CoreVal, _reg: OpRegistry) -> CoreVal | None:
    """Pack64(TupleExtract(Split64(x), 0), TupleExtract(Split64(x), 1)) -> x."""
    if not (isinstance(v, OpApp) and v.op == "vh.pack64" and len(v.args) == 2):
        return None
    lo, hi = v.args
    if (
        isinstance(lo, OpApp)
        and lo.op == "vh.tuple_extract"
        and lo.attr("index", 0) == 0
        and isinstance(hi, OpApp)
        and hi.op == "vh.tuple_extract"
        and hi.attr("index", 0) == 1
        and len(lo.args) == 1
        and len(hi.args) == 1
        and isinstance(lo.args[0], OpApp)
        and lo.args[0].op == "vh.split64"
        and lo.args[0] == hi.args[0]
        and len(lo.args[0].args) == 1
    ):
        return lo.args[0].args[0]
    return None


def _rw_split64_pack64_proj(v: CoreVal, _reg: OpRegistry) -> CoreVal | None:
    """TupleExtract(Split64(Pack64(lo, hi)), 0|1) -> lo|hi."""
    if not (isinstance(v, OpApp) and v.op == "vh.tuple_extract" and len(v.args) == 1):
        return None
    idx = v.attr("index", 0)
    inner = v.args[0]
    if (
        isinstance(inner, OpApp)
        and inner.op == "vh.split64"
        and len(inner.args) == 1
        and isinstance(inner.args[0], OpApp)
        and inner.args[0].op == "vh.pack64"
        and len(inner.args[0].args) == 2
        and idx in (0, 1)
    ):
        return inner.args[0].args[int(idx)]
    return None


def _rw_vec_extract_construct(v: CoreVal, _reg: OpRegistry) -> CoreVal | None:
    """VecExtract(VecConstructN(v0, ..., v_{N-1}), k) -> v_k."""
    if not (isinstance(v, OpApp) and v.op == "vh.vec_extract" and len(v.args) == 1):
        return None
    idx = int(v.attr("index", 0))
    inner = v.args[0]
    if isinstance(inner, OpApp) and inner.op in ("vh.vec2", "vh.vec3", "vh.vec4") and 0 <= idx < len(inner.args):
        return inner.args[idx]
    return None


def _rw_bitcast_roundtrip(v: CoreVal, _reg: OpRegistry) -> CoreVal | None:
    """Bitcast[T](x: T) -> x  and  Bitcast[T](Bitcast[U](x: T)) -> x."""
    if not (isinstance(v, OpApp) and v.op == "vh.bitcast" and len(v.args) == 1):
        return None
    arg = v.args[0]
    if arg.ty == v.ty:
        return arg
    if isinstance(arg, OpApp) and arg.op == "vh.bitcast" and len(arg.args) == 1 and arg.args[0].ty == v.ty:
        return arg.args[0]
    return None


def _rw_pack_unpack_half2x16(v: CoreVal, _reg: OpRegistry) -> CoreVal | None:
    if not (isinstance(v, OpApp) and len(v.args) == 1):
        return None
    inner = v.args[0]
    if v.op == "vh.pack_half2x16" and isinstance(inner, OpApp) and inner.op == "vh.unpack_half2x16":
        return inner.args[0]
    if v.op == "vh.unpack_half2x16" and isinstance(inner, OpApp) and inner.op == "vh.pack_half2x16":
        return inner.args[0]
    return None


def _rw_shift_zero(v: CoreVal, _reg: OpRegistry) -> CoreVal | None:
    if not (isinstance(v, OpApp) and len(v.args) == 2):
        return None
    if v.op in ("bv.shl", "bv.shru", "bv.shrs", "bv64.shl", "bv64.shru", "bv64.shrs"):
        if isinstance(v.args[1], ConstVal) and v.args[1].value == 0:
            return v.args[0]
    return None


def _rw_subgroup_shuffle_uniform(v: CoreVal, _reg: OpRegistry) -> CoreVal | None:
    """SubgroupShuffle(Const(c), lane) -> Const(c)."""
    if isinstance(v, OpApp) and v.op == "vh.subgroup_shuffle" and len(v.args) == 2:
        if isinstance(v.args[0], ConstVal):
            return v.args[0]
    return None


def _rw_fma_to_mul_add(v: CoreVal, _reg: OpRegistry) -> CoreVal | None:
    """Fast-math extensional axiom: Fma(a, b, c) -> FAdd(FMul(a, b), c)."""
    if isinstance(v, OpApp) and v.op == "vh.fma" and len(v.args) == 3:
        a, b, c = v.args
        return OpApp("f.add", (OpApp("f.mul", (a, b), v.ty), c), v.ty)
    return None



def _z3_ax_gid_nonneg(z3_backend, lhs: CoreVal, rhs: CoreVal):
    """Vulkan/Valhall CSF extensional axiom: gl_GlobalInvocationID.{x,y,z} < 2^31 (sign bit 0)."""
    import z3
    axioms = []
    for key, zexpr in list(z3_backend._syms.items()):
        sym_name = key[0]
        if sym_name in ("builtin:gl_GlobalInvocationID", "builtin:gl_LocalInvocationID", "builtin:gl_WorkGroupID"):
            if isinstance(zexpr.sort(), z3.ArraySortRef) and z3.is_bv(z3.Select(zexpr, z3.IntVal(0))):
                for lane in (0, 1, 2):
                    axioms.append(z3.Select(zexpr, z3.IntVal(lane)) >= 0)
    return axioms

def build_gpu_axiom_store(include_fast_math: bool = False) -> AxiomStore:
    """Construct the standard sound extensional axiom store for GPU SSA-IR / Valhall / SPIR-V."""
    store = AxiomStore(
        [
            ExtensionalAxiom("const_fold_op", "Evaluate pure OpApps on ConstVal arguments via OpSpec.concrete_eval", _rw_const_fold, dialect="gpu"),
            ExtensionalAxiom("pack64_split64_roundtrip", "Pack64(lo(Split64(x)), hi(Split64(x))) == x", _rw_pack64_split64, dialect="valhall"),
            ExtensionalAxiom("split64_pack64_proj", "TupleExtract(Split64(Pack64(lo, hi)), 0|1) == lo|hi", _rw_split64_pack64_proj, dialect="valhall"),
            ExtensionalAxiom("vec_extract_construct", "VecExtract(VecConstruct(v0..vk), i) == vi", _rw_vec_extract_construct, dialect="gpu"),
            ExtensionalAxiom("bitcast_roundtrip", "Bitcast[T](Bitcast[U](x: T)) == x and Bitcast[T](x: T) == x", _rw_bitcast_roundtrip, dialect="gpu"),
            ExtensionalAxiom("pack_unpack_half2x16", "PackHalf2x16(UnpackHalf2x16(x)) == x", _rw_pack_unpack_half2x16, dialect="valhall"),
            ExtensionalAxiom("shift_zero_identity", "x << 0 == x and x >> 0 == x", _rw_shift_zero, dialect="gpu"),
            ExtensionalAxiom("subgroup_shuffle_uniform", "SubgroupShuffle(Const(c), lane) == Const(c)", _rw_subgroup_shuffle_uniform, dialect="gpu"),
            ExtensionalAxiom("vulkan_gid_nonneg", "Vulkan/CSF gl_GlobalInvocationID.{x,y,z} < 2^31 (sconv32_64 == uconv32_64)", None, _z3_ax_gid_nonneg, dialect="valhall"),
        ]
    )
    if include_fast_math:
        store.add(ExtensionalAxiom("fma_to_mul_add", "Fast-math Fma(a, b, c) == FAdd(FMul(a, b), c)", _rw_fma_to_mul_add, dialect="gpu_fast_math"))
    return store


# ---------------------------------------------------------------------------
# Operator-in-the-Loop Interactive Proof & Falsification Session
# ---------------------------------------------------------------------------

@dataclass
class FailedObligation:
    """Structured report of an equivalence obligation that Z3/AU could not discharge."""

    obligation_id: str
    channel: str
    lhs: CoreVal
    rhs: CoreVal
    z3_counterexample: str | None
    free_vars: tuple[str, ...]
    suggested_hint: str = ""

    def summary(self) -> str:
        return (
            f"[{self.obligation_id}] channel={self.channel}\n"
            f"  LHS: {self.lhs.pretty()[:140]}\n"
            f"  RHS: {self.rhs.pretty()[:140]}\n"
            f"  Free vars: {', '.join(self.free_vars)}\n"
            f"  Z3 model: {(self.z3_counterexample or 'none')[:160]}\n"
            f"  Hint: {self.suggested_hint}"
        )


@dataclass
class FalsificationRecord:
    """Explicit operator-verified counterexample falsifying equivalence."""

    obligation_id: str
    channel: str
    concrete_witness: dict[str, Any]
    reason: str


@dataclass
class ProofSession:
    """Operator-in-the-loop inductive proof & falsification session.

    Workflow:
      1. Run `session.verify()` across all observable output channels of two CoreSSA-IR programs.
      2. Inspect `session.failed_obligations` if any channel failed to discharge.
      3. Resolve inductively by EITHER:
         - `session.add_axiom_and_reverify(axiom)` (scales to all future runs!), OR
         - `session.falsify(obligation_id, witness, reason)` (explicit counterexample).
    """

    lhs_channels: dict[str, CoreVal]
    rhs_channels: dict[str, CoreVal]
    reg: OpRegistry
    axioms: AxiomStore
    alias_policy: Any = None
    assumptions: list[CoreVal] = field(default_factory=list)
    proved_channels: list[str] = field(default_factory=list)
    failed_obligations: list[FailedObligation] = field(default_factory=list)
    falsifications: list[FalsificationRecord] = field(default_factory=list)
    history: list[str] = field(default_factory=list)

    def verify(self, z3_timeout_ms: int = 800) -> bool:
        from .equational_au import CoreSSAAntiUnifier

        self.proved_channels = []
        self.failed_obligations = []
        all_keys = sorted(set(self.lhs_channels.keys()) | set(self.rhs_channels.keys()))
        ob_idx = 0
        for ch in all_keys:
            if ch not in self.lhs_channels or ch not in self.rhs_channels:
                self.failed_obligations.append(
                    FailedObligation(
                        obligation_id=f"fail_{ob_idx}",
                        channel=ch,
                        lhs=self.lhs_channels[ch] if ch in self.lhs_channels else ConstVal(0, self.rhs_channels[ch].ty),
                        rhs=self.rhs_channels[ch] if ch in self.rhs_channels else ConstVal(0, self.lhs_channels[ch].ty),
                        z3_counterexample="Channel missing on one side",
                        free_vars=(),
                        suggested_hint="Missing observable output channel on LHS or RHS.",
                    )
                )
                ob_idx += 1
                continue

            au = CoreSSAAntiUnifier(
                self.reg,
                alias_policy=self.alias_policy,
                axioms=self.axioms,
                z3_timeout_ms=z3_timeout_ms,
            )
            res = au.verify_values(self.lhs_channels[ch], self.rhs_channels[ch], assumptions=self.assumptions)
            if res.equivalent:
                self.proved_channels.append(ch)
            else:
                fvs = tuple(sorted(free_param_names(self.lhs_channels[ch]) | free_param_names(self.rhs_channels[ch])))
                hint = _suggest_hint(self.lhs_channels[ch], self.rhs_channels[ch])
                self.failed_obligations.append(
                    FailedObligation(
                        obligation_id=f"fail_{ob_idx}",
                        channel=ch,
                        lhs=self.lhs_channels[ch],
                        rhs=self.rhs_channels[ch],
                        z3_counterexample=res.counterexample,
                        free_vars=fvs,
                        suggested_hint=hint,
                    )
                )
                ob_idx += 1

        ok = len(self.failed_obligations) == 0
        self.history.append(f"verify: proved={len(self.proved_channels)}/{len(all_keys)}, failed={len(self.failed_obligations)}")
        return ok

    def add_axiom_and_reverify(self, axiom: ExtensionalAxiom, z3_timeout_ms: int = 800) -> bool:
        """Inductively add a new known-true extensional axiom and re-verify."""
        self.axioms.add(axiom)
        self.history.append(f"add_axiom: {axiom.name} ({axiom.description})")
        return self.verify(z3_timeout_ms=z3_timeout_ms)

    def falsify(self, obligation_id: str, concrete_witness: dict[str, Any], reason: str) -> FalsificationRecord:
        """Explicitly falsify a failed obligation with a concrete counterexample witness."""
        match = [o for o in self.failed_obligations if o.obligation_id == obligation_id]
        ch = match[0].channel if match else "unknown"
        rec = FalsificationRecord(obligation_id, ch, concrete_witness, reason)
        self.falsifications.append(rec)
        self.history.append(f"falsify: {obligation_id} ({reason})")
        return rec


def _suggest_hint(lhs: CoreVal, rhs: CoreVal) -> str:
    lp, rp = lhs.pretty(), rhs.pretty()
    if "vh.pack64" in lp or "vh.pack64" in rp:
        return "Consider pack64/split64 roundtrip or 64-bit bit-twiddling extensional axiom."
    if "vh.fma" in lp or "vh.fma" in rp:
        return "Check strict IEEE-754 FMA vs. fast-math FAdd(FMul(a,b), c) axiom."
    if "vh.fau" in lp or "vh.fau" in rp:
        return "Unresolved FAU word: bind FAU slot in ShaderExecEnv or add FAU axiom."
    if "vh.subgroup" in lp or "vh.subgroup" in rp:
        return "Cross-lane subgroup op: check lane uniformity or add subgroup extensional axiom."
    return "Inspect Z3 counterexample model to decide between a new extensional axiom or genuine bug."
