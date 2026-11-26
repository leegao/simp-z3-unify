from dataclasses import dataclass, field
from enum import Enum

from .core_types import INT, TPtr
from .ir import ConstVal, CoreVal, OpApp, ParamVal, SymVal
from .op_registry import OpRegistry


class AliasRelation(Enum):
    NO_ALIAS = "no_alias"
    MUST_ALIAS = "must_alias"
    MAY_ALIAS = "may_alias"


@dataclass(frozen=True)
class PointerDecomposition:
    """Decomposes a pointer expression into `(base_id, base_val, const_offset, sym_offsets)`.

    Examples:
      - `%p` -> base=`param:p`, const_offset=0, sym_offsets=()
      - `ptr.gep(ptr.gep(%p, 4), 8)` -> base=`param:p`, const_offset=12, sym_offsets=()
      - `ptr.gep(%p, %i)` -> base=`param:p`, const_offset=0, sym_offsets=(%i,)
      - `mem.alloc[region=R1]` -> base=`alloc:R1`, const_offset=0, sym_offsets=()
    """

    base_id: str | None
    base_val: CoreVal
    const_offset: int
    sym_offsets: tuple[CoreVal, ...]
    addr_space: int = 0
    tbaa_tag: str = "any"


def peel_transparent(v: CoreVal, reg: OpRegistry) -> CoreVal:
    """Strip transparent SSA moves/intros/no-op casts."""
    while isinstance(v, OpApp) and reg.get(v.op).is_transparent and len(v.args) == 1:
        v = v.args[0]
    return v


def decompose_pointer(ptr: CoreVal, reg: OpRegistry) -> PointerDecomposition:
    """Fold nested `ptr.gep` chains and transparent moves into a canonical base + offset."""
    cur = peel_transparent(ptr, reg)
    const_off = 0
    sym_offs: list[CoreVal] = []

    while isinstance(cur, OpApp) and cur.op == "ptr.gep" and len(cur.args) == 2:
        base_arg, off_arg = peel_transparent(cur.args[0], reg), peel_transparent(cur.args[1], reg)
        if isinstance(off_arg, ConstVal) and isinstance(off_arg.value, int) and not isinstance(off_arg.value, bool):
            const_off += off_arg.value
        else:
            sym_offs.append(off_arg)
        cur = base_arg

    addr_space = cur.ty.addr_space if isinstance(cur.ty, TPtr) else 0
    tbaa_tag = cur.ty.tbaa_tag if isinstance(cur.ty, TPtr) else "any"

    base_id: str | None = None
    if isinstance(cur, ParamVal):
        base_id = f"param:{cur.name}"
    elif isinstance(cur, SymVal):
        base_id = f"sym:{cur.symbol}"
    elif isinstance(cur, OpApp) and cur.op == "mem.alloc":
        rid = cur.attr("region", cur.pretty())
        base_id = f"alloc:{rid}"

    # Sort symbolic offsets deterministically for canonical comparison
    sym_offs.sort(key=lambda x: x.pretty())
    return PointerDecomposition(
        base_id=base_id,
        base_val=cur,
        const_offset=const_off,
        sym_offsets=tuple(sym_offs),
        addr_space=addr_space,
        tbaa_tag=tbaa_tag,
    )


@dataclass
class AliasPolicy:
    """Configurable aliasing rules for pointer and Memory-SSA reasoning.

    Supports:
      1. Provenance disjointness: distinct `mem.alloc` regions never alias.
      2. Explicit `noalias` / `restrict` base pairs (e.g., function pointer parameters).
      3. Address-space disjointness (e.g., GPU global vs. shared vs. private memory).
      4. Type-Based Alias Analysis (TBAA): parent map `child_tag -> parent_tag`;
         two tags with no ancestor relationship cannot alias (unless either is `"any"`).
      5. Same-base constant-offset non-overlap (element-granular memory model).
    """

    noalias_pairs: set[frozenset[str]] = field(default_factory=set)
    disjoint_addr_spaces: bool = True
    tbaa_enabled: bool = True
    tbaa_parent: dict[str, str] = field(default_factory=dict)
    all_params_noalias: bool = False

    def add_noalias(self, base_a: str, base_b: str) -> None:
        if base_a != base_b:
            self.noalias_pairs.add(frozenset((base_a, base_b)))

    def add_tbaa_child(self, child_tag: str, parent_tag: str) -> None:
        self.tbaa_parent[child_tag] = parent_tag

    def _tbaa_compatible(self, tag_a: str, tag_b: str) -> bool:
        if not self.tbaa_enabled or tag_a == "any" or tag_b == "any" or tag_a == tag_b:
            return True

        def ancestors(t: str) -> set[str]:
            seen = {t}
            cur = t
            while cur in self.tbaa_parent and self.tbaa_parent[cur] not in seen:
                cur = self.tbaa_parent[cur]
                seen.add(cur)
            return seen

        return (tag_a in ancestors(tag_b)) or (tag_b in ancestors(tag_a))

    def query(self, p1: CoreVal, p2: CoreVal, reg: OpRegistry) -> AliasRelation:
        """Determine whether `p1` and `p2` MUST_ALIAS, NO_ALIAS, or MAY_ALIAS."""
        d1 = decompose_pointer(p1, reg)
        d2 = decompose_pointer(p2, reg)

        # 1. Address-space disjointness
        if self.disjoint_addr_spaces and d1.addr_space != d2.addr_space:
            return AliasRelation.NO_ALIAS

        # 2. TBAA disjointness
        if not self._tbaa_compatible(d1.tbaa_tag, d2.tbaa_tag):
            return AliasRelation.NO_ALIAS

        # 3. Same provenance base
        if d1.base_id is not None and d1.base_id == d2.base_id:
            if d1.sym_offsets == d2.sym_offsets:
                if d1.const_offset == d2.const_offset:
                    return AliasRelation.MUST_ALIAS
                return AliasRelation.NO_ALIAS
            return AliasRelation.MAY_ALIAS

        # 4. Distinct provenance bases
        if d1.base_id is not None and d2.base_id is not None:
            # Distinct stack/heap allocations never alias
            if d1.base_id.startswith("alloc:") or d2.base_id.startswith("alloc:"):
                return AliasRelation.NO_ALIAS
            # Explicit noalias / restrict pair
            b1_short = d1.base_id.split(":", 1)[-1]
            b2_short = d2.base_id.split(":", 1)[-1]
            if (
                frozenset((d1.base_id, d2.base_id)) in self.noalias_pairs
                or frozenset((b1_short, b2_short)) in self.noalias_pairs
            ):
                return AliasRelation.NO_ALIAS
            if self.all_params_noalias and d1.base_id.startswith("param:") and d2.base_id.startswith("param:"):
                return AliasRelation.NO_ALIAS

        return AliasRelation.MAY_ALIAS


def normalize_gep(v: CoreVal, reg: OpRegistry) -> CoreVal:
    """Fold nested `ptr.gep(ptr.gep(p, a), b)` into canonical `ptr.gep(p, a + b)`."""
    if not (isinstance(v, OpApp) and v.op == "ptr.gep" and len(v.args) == 2):
        return v
    dec = decompose_pointer(v, reg)
    if dec.const_offset == 0 and not dec.sym_offsets:
        return dec.base_val
    # Rebuild offset expression
    parts: list[CoreVal] = list(dec.sym_offsets)
    if dec.const_offset != 0:
        parts.append(ConstVal(dec.const_offset, INT))
    off = parts[0]
    for p in parts[1:]:
        off = OpApp("add", (off, p), INT)
    return OpApp("ptr.gep", (dec.base_val, off), v.ty, v.attrs)


def normalize_memory_ssa(v: CoreVal, reg: OpRegistry, policy: AliasPolicy) -> CoreVal:
    """Apply sound alias-directed Memory-SSA simplifications:

      1. Store-to-load forwarding: `mem.load(mem.store(M, p1, val), p2)`
         - if MUST_ALIAS(p1, p2) -> `val`
         - if NO_ALIAS(p1, p2)   -> `mem.load(M, p2)` (walk past irrelevant store)
      2. Dead-store elimination: `mem.store(mem.store(M, p1, v1), p2, v2)`
         - if MUST_ALIAS(p1, p2) -> `mem.store(M, p2, v2)`
      3. Disjoint store canonical ordering (bubble sort on NO_ALIAS adjacent stores):
         - if NO_ALIAS(p1, p2) and key(p2) < key(p1):
           `mem.store(mem.store(M, p1, v1), p2, v2)` -> `mem.store(mem.store(M, p2, v2), p1, v1)`
    """
    v = peel_transparent(v, reg)
    if not isinstance(v, OpApp):
        return v

    if v.op == "ptr.gep":
        return normalize_gep(v, reg)

    if v.op == "mem.load" and len(v.args) == 2:
        mem = normalize_memory_ssa(v.args[0], reg, policy)
        ptr = normalize_gep(peel_transparent(v.args[1], reg), reg)
        # Walk backwards through store chain
        cur_mem = mem
        while isinstance(cur_mem, OpApp) and cur_mem.op == "mem.store" and len(cur_mem.args) == 3:
            prev_mem, st_ptr, st_val = cur_mem.args
            rel = policy.query(st_ptr, ptr, reg)
            if rel == AliasRelation.MUST_ALIAS:
                return normalize_memory_ssa(st_val, reg, policy)
            if rel == AliasRelation.NO_ALIAS:
                cur_mem = prev_mem
                continue
            break
        return OpApp("mem.load", (cur_mem, ptr), v.ty, v.attrs)

    if v.op == "mem.store" and len(v.args) == 3:
        mem = normalize_memory_ssa(v.args[0], reg, policy)
        ptr = normalize_gep(peel_transparent(v.args[1], reg), reg)
        val = normalize_memory_ssa(v.args[2], reg, policy)

        # Dead-store elimination & disjoint store commutation
        if isinstance(mem, OpApp) and mem.op == "mem.store" and len(mem.args) == 3:
            prev_mem, p1, v1 = mem.args
            rel = policy.query(p1, ptr, reg)
            if rel == AliasRelation.MUST_ALIAS:
                return normalize_memory_ssa(OpApp("mem.store", (prev_mem, ptr, val), v.ty, v.attrs), reg, policy)
            if rel == AliasRelation.NO_ALIAS and ptr.pretty() < p1.pretty():
                swapped_inner = normalize_memory_ssa(
                    OpApp("mem.store", (prev_mem, ptr, val), v.ty, v.attrs), reg, policy
                )
                return OpApp("mem.store", (swapped_inner, p1, v1), v.ty, v.attrs)

        return OpApp("mem.store", (mem, ptr, val), v.ty, v.attrs)

    return v
