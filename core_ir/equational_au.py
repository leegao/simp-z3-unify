import time
from dataclasses import dataclass, field

from .aliasing import AliasPolicy, normalize_memory_ssa, peel_transparent
from .axioms import AxiomStore
from .ir import (
    ConstVal,
    CoreFunc,
    CoreVal,
    HoleVal,
    LoopOutVal,
    LoopVal,
    OpApp,
    ParamVal,
    SelectVal,
    SymVal,
    free_param_names,
    optimal_tree_cover_cost,
    subst_val,
    val_tree_size,
)
from .op_registry import OpRegistry, OpTrait
from .core_types import BOOL, FLOAT, INT, CoreType, TBitVec, TPtr
from .z3_backend import CoreZ3Backend


@dataclass
class AUObligation:
    """A single anti-unification proof obligation discharged structurally or via Z3."""

    hole_id: str
    lhs: str
    rhs: str
    method: str  # syntactic | move_peel | ac_match | assoc_match | inverse_cancel | alias_mem | opaque_uf | loop_lockstep | loop_relational | loop_bounded | z3_leaf


@dataclass
class CoreAUResult:
    """Result of equational tree-covering anti-unification on two CoreSSA-IR values/functions."""

    equivalent: bool
    status: str  # "unsat" | "sat" | "unknown"
    template: CoreVal | None
    tree_cost_before: int
    tree_cost_after: int
    z3_leaf_calls: int
    structural_discharges: int
    loops_proved_unbounded: int
    elapsed_ms: float
    obligations: list[AUObligation] = field(default_factory=list)
    counterexample: str | None = None

    def format_template(self) -> str:
        if self.template is None:
            return "<none>"
        holes = sorted({ob.hole_id for ob in self.obligations if ob.method == "z3_leaf"})
        params = ", ".join(f"?{h}" for h in holes)
        return f"let %skel({params}) = {self.template.pretty()}"


def _is_const_equal(v: CoreVal, target: object) -> bool:
    if not isinstance(v, ConstVal):
        return False
    if isinstance(target, bool):
        return isinstance(v.value, bool) and v.value is target
    if isinstance(v.value, bool):
        return False
    return v.value == target


def _structural_similarity(a: CoreVal, b: CoreVal, reg: OpRegistry) -> int:
    """Heuristic structural similarity score (higher = better pair) for AC bipartite matching."""
    a = peel_transparent(a, reg)
    b = peel_transparent(b, reg)
    if a == b:
        return 1000
    if type(a) is not type(b):
        return 0
    if isinstance(a, OpApp) and isinstance(b, OpApp):
        if a.op == b.op and a.attrs == b.attrs and len(a.args) == len(b.args):
            score = 50 + 10 * sum(1 for x, y in zip(a.args, b.args) if x == y)
            if reg.get(a.op).is_rigid_uf:
                score += 100  # Strongly prefer aligning matching opaque UFs!
            return score
        return 5
    if isinstance(a, SelectVal) and isinstance(b, SelectVal):
        return 40 + (20 if a.cond == b.cond else 0)
    if isinstance(a, LoopOutVal) and isinstance(b, LoopOutVal):
        return 60
    return 1


def _best_bipartite_pairing(
    left: list[CoreVal], right: list[CoreVal], reg: OpRegistry
) -> list[tuple[int, int]]:
    """Greedy/exact maximum-weight bipartite pairing for small AC unmatched multisets."""
    m, n = len(left), len(right)
    if m == 0 or n == 0:
        return []
    # Exact DP when small (<= 8), greedy highest-score otherwise
    if m <= 8 and n <= 8:
        scores = [[_structural_similarity(left[i], right[j], reg) for j in range(n)] for i in range(m)]
        memo: dict[tuple[int, int], tuple[int, list[tuple[int, int]]]] = {}

        def dp(i: int, used_mask: int) -> tuple[int, list[tuple[int, int]]]:
            if i == m:
                return 0, []
            key = (i, used_mask)
            if key in memo:
                return memo[key]
            best_score, best_pairs = dp(i + 1, used_mask)
            for j in range(n):
                if not (used_mask & (1 << j)) and scores[i][j] >= 10:
                    sub_s, sub_p = dp(i + 1, used_mask | (1 << j))
                    cand = scores[i][j] + sub_s
                    if cand > best_score:
                        best_score = cand
                        best_pairs = [(i, j)] + sub_p
            memo[key] = (best_score, best_pairs)
            return best_score, best_pairs

        _, pairs = dp(0, 0)
        return pairs

    # Greedy fallback for large arity
    cands = []
    for i in range(m):
        for j in range(n):
            sc = _structural_similarity(left[i], right[j], reg)
            if sc >= 10:
                cands.append((sc, i, j))
    cands.sort(reverse=True)
    used_l: set[int] = set()
    used_r: set[int] = set()
    pairs = []
    for _, i, j in cands:
        if i not in used_l and j not in used_r:
            used_l.add(i)
            used_r.add(j)
            pairs.append((i, j))
    return pairs


class CoreSSAAntiUnifier:
    """Equational Tree-Covering Anti-Unifier over CoreSSA-IR.

    Supports:
      - Transparent SSA move/intro peeling (`TRANSPARENT_MOVE`, `VALUE_INTRO`)
      - Involution cancellation (`neg(neg(x)) == x`, `not(not(x)) == x`)
      - Identity & absorbing element simplification
      - Inverse pair normalization & cancellation (`sub -> add+neg`, `add(x, neg(x)) -> 0`,
        `div(mul(a, b), b) -> a` with Z3 nonzero side-condition, `xor(x, x) -> 0`)
      - Associative-only (non-commutative) sequence flattening & ordered alignment
      - Associative-commutative (AC) multiset flattening, cancellation, and bipartite pairing
      - Provenance-based pointer aliasing & Memory-SSA normalization/commutation
      - Opaque UF & symbol congruence
      - Unbounded loop anti-unification (dead-var slicing, LICM peeling, lockstep induction,
        relational invariant simulation, and bounded unrolling fallback)
      - Lazy leaf Z3 discharge and O(|T|) optimal BURS tree-covering cost tracking.
    """

    def __init__(
        self,
        reg: OpRegistry,
        alias_policy: AliasPolicy | None = None,
        axioms: AxiomStore | None = None,
        z3_timeout_ms: int = 500,
    ):
        self.reg = reg
        self.alias_policy = alias_policy or AliasPolicy()
        self.axioms = axioms
        self.z3_timeout_ms = z3_timeout_ms
        self.z3 = CoreZ3Backend(self.reg, self.alias_policy)

        self._hole_counter = 0
        self.z3_leaf_calls = 0
        self.structural_discharges = 0
        self.loops_proved_unbounded = 0
        self.obligations: list[AUObligation] = []
        self.proved_lhs: set[CoreVal] = set()
        self._visited_pairs: set[tuple[int, int]] = set()
        self.counterexample: str | None = None

    def _fresh_hole(self) -> str:
        hid = f"h{self._hole_counter}"
        self._hole_counter += 1
        return hid

    def _record(self, lhs: CoreVal, rhs: CoreVal, method: str) -> str:
        hid = self._fresh_hole()
        self.proved_lhs.add(lhs)
        if method != "z3_leaf":
            self.structural_discharges += 1
        self.obligations.append(AUObligation(hid, lhs.pretty()[:80], rhs.pretty()[:80], method))
        return hid

    def canonicalize(self, v: CoreVal, assumptions: list[CoreVal], _memo: dict[int, tuple[CoreVal, CoreVal]] | None = None) -> CoreVal:
        """Apply local sound equational canonicalizations driven by `OpRegistry` and `AliasPolicy` (per-call DAG-memoized)."""
        if _memo is None:
            _memo = {}
        vid = id(v)
        if vid in _memo:
            return _memo[vid][1]
        res = self._canonicalize_impl(v, assumptions, _memo)
        _memo[vid] = (v, res)
        return res

    def _canonicalize_impl(self, v: CoreVal, assumptions: list[CoreVal], _memo: dict[int, tuple[CoreVal, CoreVal]]) -> CoreVal:
        v = peel_transparent(v, self.reg)
        if isinstance(v, (ConstVal, ParamVal, SymVal, HoleVal)):
            return v

        if isinstance(v, SelectVal):
            c = self.canonicalize(v.cond, assumptions, _memo)
            t = self.canonicalize(v.then_val, assumptions, _memo)
            e = self.canonicalize(v.else_val, assumptions, _memo)
            if _is_const_equal(c, True):
                return t
            if _is_const_equal(c, False):
                return e
            if t == e:
                return t
            return SelectVal(c, t, e, v.ty)

        if isinstance(v, LoopOutVal):
            return v

        if isinstance(v, OpApp):
            spec = self.reg.get(v.op)

            # Memory-SSA & pointer GEP normalization
            if v.op in ("ptr.gep", "mem.load", "mem.store"):
                canon_args = tuple(self.canonicalize(a, assumptions, _memo) for a in v.args)
                rebuilt = OpApp(v.op, canon_args, v.ty, v.attrs)
                return normalize_memory_ssa(rebuilt, self.reg, self.alias_policy)

            if spec.is_ac:
                raw_flat = self._flatten_op(v, v.op)
                canon_flat: list[CoreVal] = []
                for item in raw_flat:
                    c_item = self.canonicalize(item, assumptions, _memo)
                    canon_flat.extend(self._flatten_op(c_item, v.op))
                if spec.has_absorbing:
                    for a in canon_flat:
                        if _is_const_equal(a, spec.absorbing):
                            return ConstVal(spec.absorbing, v.ty)  # type: ignore[arg-type]
                reduced, _ = self._cancel_internal_ac(v.op, canon_flat)
                if not reduced and spec.has_identity:
                    return ConstVal(spec.identity, v.ty)  # type: ignore[arg-type]
                if len(reduced) == 1:
                    return reduced[0]
                if reduced:
                    reduced.sort(key=lambda item: item.pretty())
                    rebuilt_ac = self._rebuild_nary(v.op, reduced, v.ty)
                    if self.axioms is not None:
                        rw_ac, fired_ac = self.axioms.apply_rewrites(rebuilt_ac, self.reg)
                        if fired_ac and rw_ac != rebuilt_ac:
                            return self.canonicalize(rw_ac, assumptions, _memo)
                    return rebuilt_ac
                return v

            canon_args = tuple(self.canonicalize(a, assumptions, _memo) for a in v.args)

            # Involution: f(f(x)) -> x
            if spec.is_involution and len(canon_args) == 1:
                inner = canon_args[0]
                if isinstance(inner, OpApp) and inner.op == v.op and len(inner.args) == 1:
                    return inner.args[0]
                # Distribute unary additive inverse over add: neg(add(a, b)) -> add(neg(a), neg(b))
                inv_spec = self.reg.inverse_for_unary(v.op)
                if inv_spec and isinstance(inner, OpApp) and inner.op == inv_spec.forward_op and not inv_spec.nonzero_rhs:
                    neg_children = tuple(
                        self.canonicalize(OpApp(v.op, (c,), c.ty), assumptions, _memo) for c in inner.args
                    )
                    return OpApp(inv_spec.forward_op, neg_children, v.ty, inner.attrs)
                return OpApp(v.op, canon_args, v.ty, v.attrs)

            # Binary inverse normalization (e.g., sub(a, b) -> add(a, neg(b)))
            inv_bin = self.reg.inverse_for_binop(v.op)
            if inv_bin is not None and len(canon_args) == 2:
                a, b = canon_args
                # Guarded division cancellation: div(mul(x, y), y) -> x when y != 0
                if inv_bin.nonzero_rhs:
                    if a == b and self.z3.prove_nonzero(b, assumptions):
                        fwd_spec = self.reg.get(inv_bin.forward_op)
                        if fwd_spec.has_identity:
                            return ConstVal(fwd_spec.identity, v.ty)  # type: ignore[arg-type]
                    if isinstance(a, OpApp) and a.op == inv_bin.forward_op:
                        factors = self._flatten_op(a, inv_bin.forward_op)
                        if b in factors and self.z3.prove_nonzero(b, assumptions):
                            rem = list(factors)
                            rem.remove(b)
                            return self._rebuild_nary(inv_bin.forward_op, rem, v.ty)
                    # Identity on RHS: div(a, 1) -> a
                    if spec.has_identity and _is_const_equal(b, spec.identity):
                        return a
                    return OpApp(v.op, canon_args, v.ty, v.attrs)
                else:
                    # Total inverse (e.g., sub(a, b) -> add(a, neg(b)))
                    if a == b:
                        fwd_spec = self.reg.get(inv_bin.forward_op)
                        if fwd_spec.has_identity:
                            return ConstVal(fwd_spec.identity, v.ty)  # type: ignore[arg-type]
                    if inv_bin.unary_inverse:
                        neg_b = self.canonicalize(OpApp(inv_bin.unary_inverse, (b,), b.ty), assumptions, _memo)
                        return self.canonicalize(OpApp(inv_bin.forward_op, (a, neg_b), v.ty), assumptions, _memo)

            # Identity / absorbing checks for binary/n-ary ops
            if spec.has_absorbing:
                for a in canon_args:
                    if _is_const_equal(a, spec.absorbing):
                        return ConstVal(spec.absorbing, v.ty)  # type: ignore[arg-type]

            if spec.is_ac:
                flat: list[CoreVal] = []
                for a in canon_args:
                    flat.extend(self._flatten_op(a, v.op))
                reduced, _ = self._cancel_internal_ac(v.op, flat)
                if not reduced and spec.has_identity:
                    return ConstVal(spec.identity, v.ty)  # type: ignore[arg-type]
                if len(reduced) == 1:
                    return reduced[0]
                if reduced:
                    reduced.sort(key=lambda item: item.pretty())
                    return self._rebuild_nary(v.op, reduced, v.ty)
            elif spec.has_identity and spec.is_assoc_only:
                filtered = [a for a in canon_args if not _is_const_equal(a, spec.identity)]
                if not filtered:
                    return ConstVal(spec.identity, v.ty)  # type: ignore[arg-type]
                if len(filtered) == 1:
                    return filtered[0]
                canon_args = tuple(filtered)

            res_node: CoreVal = OpApp(v.op, canon_args, v.ty, v.attrs)
            if self.axioms is not None:
                rewritten, fired = self.axioms.apply_rewrites(res_node, self.reg)
                if fired and rewritten != res_node:
                    return self.canonicalize(rewritten, assumptions, _memo)
            return res_node

        return v

    def _flatten_op(self, v: CoreVal, op_name: str) -> list[CoreVal]:
        """Flatten nested associative applications of `op_name` into a flat list."""
        v = peel_transparent(v, self.reg)
        if isinstance(v, OpApp) and v.op == op_name:
            out: list[CoreVal] = []
            for a in v.args:
                out.extend(self._flatten_op(a, op_name))
            return out
        return [v]

    def _rebuild_nary(self, op_name: str, items: list[CoreVal], ty: CoreType) -> CoreVal:
        spec = self.reg.get(op_name)
        if not items:
            if spec.has_identity:
                return ConstVal(spec.identity, ty)  # type: ignore[arg-type]
            raise ValueError(f"Cannot rebuild empty n-ary op {op_name} without identity")
        cur = items[0]
        for x in items[1:]:
            cur = OpApp(op_name, (cur, x), ty)
        return cur

    def _cancel_internal_ac(self, op_name: str, items: list[CoreVal]) -> tuple[list[CoreVal], int]:
        """Cancel internal inverses, self-inverses, identities, and idempotents in an AC multiset."""
        spec = self.reg.get(op_name)
        cancellations = 0
        # Drop identities
        if spec.has_identity:
            before = len(items)
            items = [x for x in items if not _is_const_equal(x, spec.identity)]
            cancellations += before - len(items)

        # Idempotent deduplication: x & x -> x
        if OpTrait.IDEMPOTENT in spec.traits:
            dedup: list[CoreVal] = []
            seen: set[CoreVal] = set()
            for x in items:
                if x not in seen:
                    seen.add(x)
                    dedup.append(x)
                else:
                    cancellations += 1
            items = dedup

        # Self-inverse pair cancellation: xor(x, x) -> 0
        if spec.is_self_inverse:
            counts: dict[CoreVal, int] = {}
            order: list[CoreVal] = []
            for x in items:
                if x not in counts:
                    order.append(x)
                counts[x] = counts.get(x, 0) + 1
            rem: list[CoreVal] = []
            for x in order:
                if counts[x] % 2 == 1:
                    rem.append(x)
                cancellations += counts[x] // 2
            items = rem

        # Unary inverse cancellation: add(x, neg(x)) -> 0
        if spec.inverse and spec.inverse.unary_inverse and not spec.inverse.nonzero_rhs:
            uinv = spec.inverse.unary_inverse
            rem_list: list[CoreVal] = []
            for x in items:
                # Check if x's inverse is already waiting in rem_list
                inv_x = (
                    x.args[0]
                    if (isinstance(x, OpApp) and x.op == uinv and len(x.args) == 1)
                    else OpApp(uinv, (x,), x.ty)
                )
                if inv_x in rem_list:
                    rem_list.remove(inv_x)
                    cancellations += 1
                else:
                    rem_list.append(x)
            items = rem_list

        return items, cancellations

    def _z3_leaf_discharge(
        self, lhs: CoreVal, rhs: CoreVal, assumptions: list[CoreVal]
    ) -> tuple[bool, CoreVal]:
        self.z3_leaf_calls += 1
        _ = self.z3.to_z3(lhs); _ = self.z3.to_z3(rhs)
        extra_z3 = self.axioms.collect_z3_axioms(self.z3, lhs, rhs) if self.axioms else []
        ok, cex = self.z3.prove_equal(lhs, rhs, assumptions=assumptions, extra_z3_assumptions=extra_z3, timeout_ms=self.z3_timeout_ms)
        if ok:
            hid = self._record(lhs, rhs, "z3_leaf")
            fvs = tuple(sorted(free_param_names(lhs) | free_param_names(rhs)))
            return True, HoleVal(hid, lhs.ty, fvs)
        self.counterexample = cex
        return False, HoleVal("FAIL", lhs.ty)

    def anti_unify(
        self,
        raw_l: CoreVal,
        raw_r: CoreVal,
        assumptions: list[CoreVal] | None = None,
    ) -> tuple[bool, CoreVal]:
        """Equational tree-covering anti-unification of `raw_l` and `raw_r` under `assumptions`.

        Returns `(equivalent, generalization_template)`.
        """
        assumptions = assumptions or []
        if raw_l is raw_r:
            self.proved_lhs.add(raw_l)
            return True, raw_l

        # 1. Peel transparent SSA moves/intros
        pl = peel_transparent(raw_l, self.reg)
        pr = peel_transparent(raw_r, self.reg)
        peeled = (pl != raw_l) or (pr != raw_r)

        # 2. Local equational & Memory-SSA canonicalization
        l = self.canonicalize(pl, assumptions)
        r = self.canonicalize(pr, assumptions)

        if l is r or l == r:
            if peeled:
                self._record(raw_l, raw_r, "move_peel")
            elif l != pl or r != pr:
                method = "alias_mem" if isinstance(pl, OpApp) and pl.op in ("mem.load", "mem.store", "ptr.gep") else "inverse_cancel"
                self._record(raw_l, raw_r, method)
            else:
                self.proved_lhs.add(raw_l)
            return True, l

        # 3. Select / ITE congruence (with branch path-condition propagation!)
        if isinstance(l, SelectVal) and isinstance(r, SelectVal):
            ok_c, tmpl_c = self.anti_unify(l.cond, r.cond, assumptions)
            if ok_c:
                not_c = OpApp("not", (l.cond,), BOOL)
                ok_t, tmpl_t = self.anti_unify(l.then_val, r.then_val, assumptions + [l.cond])
                ok_e, tmpl_e = self.anti_unify(l.else_val, r.else_val, assumptions + [not_c])
                if ok_t and ok_e:
                    self.proved_lhs.add(raw_l)
                    return True, SelectVal(tmpl_c, tmpl_t, tmpl_e, l.ty)

        # 4. Structured SSA theta-loops (`LoopOutVal`)
        if isinstance(l, LoopOutVal) and isinstance(r, LoopOutVal):
            ok_lp, tmpl_lp = self._anti_unify_loops(l, r, assumptions)
            if ok_lp:
                self.proved_lhs.add(raw_l)
                return True, tmpl_lp

        # 5. Operation applications (`OpApp`)
        if isinstance(l, OpApp) and isinstance(r, OpApp) and l.op == r.op and l.attrs == r.attrs:
            spec = self.reg.get(l.op)

            # 5a. Associative-Commutative (AC) multiset alignment
            if spec.is_ac:
                return self._anti_unify_ac(raw_l, raw_r, l, r, assumptions)

            # 5b. Associative-Only (non-commutative) ordered sequence alignment
            if spec.is_assoc_only:
                return self._anti_unify_assoc_only(raw_l, raw_r, l, r, assumptions)

            # 5c. Commutative-Only binary op (e.g., eq, neq)
            if spec.is_comm and len(l.args) == 2 and len(r.args) == 2:
                ok_a, ta = self.anti_unify(l.args[0], r.args[0], assumptions)
                ok_b, tb = self.anti_unify(l.args[1], r.args[1], assumptions)
                if ok_a and ok_b:
                    self.proved_lhs.add(raw_l)
                    return True, OpApp(l.op, (ta, tb), l.ty, l.attrs)
                # Try swapped operands
                ok_a2, ta2 = self.anti_unify(l.args[0], r.args[1], assumptions)
                ok_b2, tb2 = self.anti_unify(l.args[1], r.args[0], assumptions)
                if ok_a2 and ok_b2:
                    self._record(raw_l, raw_r, "ac_match")
                    return True, OpApp(l.op, (ta2, tb2), l.ty, l.attrs)

            # 5d. Rigid UF / standard congruence (arity-matched)
            if len(l.args) == len(r.args):
                child_tmpls: list[CoreVal] = []
                all_ok = True
                for ca, cb in zip(l.args, r.args):
                    ok_c, tc = self.anti_unify(ca, cb, assumptions)
                    if not ok_c:
                        all_ok = False
                        break
                    child_tmpls.append(tc)
                if all_ok:
                    if spec.is_rigid_uf:
                        self._record(raw_l, raw_r, "opaque_uf")
                    else:
                        self.proved_lhs.add(raw_l)
                    return True, OpApp(l.op, tuple(child_tmpls), l.ty, l.attrs)

        # 6. Lazy leaf Z3 discharge for residual semantic mismatch
        return self._z3_leaf_discharge(l, r, assumptions)

    def _anti_unify_ac(
        self,
        raw_l: CoreVal,
        raw_r: CoreVal,
        l: OpApp,
        r: OpApp,
        assumptions: list[CoreVal],
    ) -> tuple[bool, CoreVal]:
        op_name = l.op
        flat_l, canc_l = self._cancel_internal_ac(op_name, self._flatten_op(l, op_name))
        flat_r, canc_r = self._cancel_internal_ac(op_name, self._flatten_op(r, op_name))

        # Exact multiset intersection
        rem_l: list[CoreVal] = []
        rem_r = list(flat_r)
        matched_tmpls: list[CoreVal] = []
        for item in flat_l:
            if item in rem_r:
                rem_r.remove(item)
                matched_tmpls.append(item)
            else:
                rem_l.append(item)

        if not rem_l and not rem_r:
            method = "inverse_cancel" if (canc_l + canc_r) > 0 else "ac_match"
            self._record(raw_l, raw_r, method)
            return True, self._rebuild_nary(op_name, matched_tmpls, l.ty)

        # Minimum-cost bipartite pairing on remaining unmatched terms
        pairs = _best_bipartite_pairing(rem_l, rem_r, self.reg)
        paired_l_idx: set[int] = set()
        paired_r_idx: set[int] = set()
        for i, j in pairs:
            ok_p, tp = self.anti_unify(rem_l[i], rem_r[j], assumptions)
            if ok_p:
                paired_l_idx.add(i)
                paired_r_idx.add(j)
                matched_tmpls.append(tp)

        leftover_l = [rem_l[i] for i in range(len(rem_l)) if i not in paired_l_idx]
        leftover_r = [rem_r[j] for j in range(len(rem_r)) if j not in paired_r_idx]

        if not leftover_l and not leftover_r:
            self._record(raw_l, raw_r, "ac_match")
            return True, self._rebuild_nary(op_name, matched_tmpls, l.ty)

        # Residual unmatched sub-expression -> single minimal Z3 leaf obligation!
        sub_l = self._rebuild_nary(op_name, leftover_l, l.ty)
        sub_r = self._rebuild_nary(op_name, leftover_r, r.ty)
        ok_rem, tmpl_rem = self._z3_leaf_discharge(sub_l, sub_r, assumptions)
        if ok_rem:
            matched_tmpls.append(tmpl_rem)
            self.proved_lhs.add(raw_l)
            return True, self._rebuild_nary(op_name, matched_tmpls, l.ty)

        return False, HoleVal("FAIL", l.ty)

    def _anti_unify_assoc_only(
        self,
        raw_l: CoreVal,
        raw_r: CoreVal,
        l: OpApp,
        r: OpApp,
        assumptions: list[CoreVal],
    ) -> tuple[bool, CoreVal]:
        """Ordered associative flattening + prefix/suffix peeling for non-commutative ops."""
        op_name = l.op
        seq_l = self._flatten_op(l, op_name)
        seq_r = self._flatten_op(r, op_name)

        # Peel identical prefix
        prefix: list[CoreVal] = []
        while seq_l and seq_r and seq_l[0] == seq_r[0]:
            prefix.append(seq_l.pop(0))
            seq_r.pop(0)

        # Peel identical suffix
        suffix: list[CoreVal] = []
        while seq_l and seq_r and seq_l[-1] == seq_r[-1]:
            suffix.append(seq_l.pop())
            seq_r.pop()
        suffix.reverse()

        if not seq_l and not seq_r:
            self._record(raw_l, raw_r, "assoc_match")
            return True, self._rebuild_nary(op_name, prefix + suffix, l.ty)

        # Equal remaining length -> try position-by-position anti-unification (order preserved!)
        if len(seq_l) == len(seq_r):
            mid: list[CoreVal] = []
            all_ok = True
            for a, b in zip(seq_l, seq_r):
                ok_c, tc = self.anti_unify(a, b, assumptions)
                if not ok_c:
                    all_ok = False
                    break
                mid.append(tc)
            if all_ok:
                self._record(raw_l, raw_r, "assoc_match")
                return True, self._rebuild_nary(op_name, prefix + mid + suffix, l.ty)

        # Otherwise discharge minimal mismatched contiguous slice via Z3
        sub_l = self._rebuild_nary(op_name, seq_l, l.ty)
        sub_r = self._rebuild_nary(op_name, seq_r, r.ty)
        ok_rem, tmpl_rem = self._z3_leaf_discharge(sub_l, sub_r, assumptions)
        if ok_rem:
            self.proved_lhs.add(raw_l)
            return True, self._rebuild_nary(op_name, prefix + [tmpl_rem] + suffix, l.ty)

        return False, HoleVal("FAIL", l.ty)

    # ------------------------------------------------------------------
    # Unbounded Loop Anti-Unification (Slicing + LICM + Lockstep + Relational)
    # ------------------------------------------------------------------

    def _slice_and_peel_loop(self, lp: LoopVal, out_idx: int) -> tuple[LoopVal, int]:
        """Slice dead loop-carried variables and peel loop-invariant carried variables (LICM)."""
        n = len(lp.iter_params)
        # 1. Identify loop-invariant carried variables: body_next[i] == iter_params[i]
        inv_map: dict[str, CoreVal] = {}
        for i in range(n):
            b_i = peel_transparent(lp.body_next[i], self.reg)
            if b_i == lp.iter_params[i]:
                inv_map[lp.iter_params[i].name] = lp.init_vals[i]

        cond_peeled = subst_val(lp.cond, inv_map)
        body_peeled = tuple(subst_val(b, inv_map) for b in lp.body_next)

        # 2. Backward liveness closure from {out_idx} and cond_peeled
        name_to_idx = {p.name: i for i, p in enumerate(lp.iter_params)}
        live: set[int] = {out_idx}
        worklist: list[CoreVal] = [cond_peeled, body_peeled[out_idx]]
        while worklist:
            cur = worklist.pop()
            for pname in free_param_names(cur):
                if pname in name_to_idx and name_to_idx[pname] not in live:
                    idx = name_to_idx[pname]
                    live.add(idx)
                    worklist.append(body_peeled[idx])

        # Keep invariant out_idx if it was requested, otherwise drop invariant vars that are not live
        live_sorted = sorted(i for i in live if lp.iter_params[i].name not in inv_map or i == out_idx)
        new_out_idx = live_sorted.index(out_idx)
        new_params = tuple(lp.iter_params[i] for i in live_sorted)
        new_inits = tuple(lp.init_vals[i] for i in live_sorted)
        new_body = tuple(body_peeled[i] for i in live_sorted)
        from .core_types import TTuple

        sliced = LoopVal(
            iter_params=new_params,
            init_vals=new_inits,
            cond=cond_peeled,
            body_next=new_body,
            ty=TTuple(tuple(p.ty for p in new_params)),
            unroll_hint=lp.unroll_hint,
            loop_id=lp.loop_id,
        )
        return sliced, new_out_idx

    def _anti_unify_loops(
        self, l_out: LoopOutVal, r_out: LoopOutVal, assumptions: list[CoreVal]
    ) -> tuple[bool, CoreVal]:
        """Prove equivalence of two SSA theta-loops via:
          1. Dead-var slicing + LICM invariant peeling
          2. Unbounded lockstep inductive congruence
          3. Auto-synthesized relational simulation (e.g., strength-reduced pointer/linear IV)
          4. Bounded unrolling Z3 fallback.
        """
        lp_l, idx_l = self._slice_and_peel_loop(l_out.loop, l_out.index)
        lp_r, idx_r = self._slice_and_peel_loop(r_out.loop, r_out.index)

        # Case A: Constant/invariant output (loop doesn't modify queried variable)
        if peel_transparent(lp_l.body_next[idx_l], self.reg) == lp_l.iter_params[idx_l] and (
            peel_transparent(lp_r.body_next[idx_r], self.reg) == lp_r.iter_params[idx_r]
        ):
            return self.anti_unify(lp_l.init_vals[idx_l], lp_r.init_vals[idx_r], assumptions)

        # Case B: Unbounded Lockstep Inductive Loop Congruence
        if len(lp_l.iter_params) == len(lp_r.iter_params):
            ok_lock, tmpl_lock = self._try_lockstep_loop(lp_l, idx_l, lp_r, idx_r, assumptions)
            if ok_lock:
                self.loops_proved_unbounded += 1
                self._record(l_out, r_out, "loop_lockstep")
                return True, tmpl_lock

        # Case C: Relational Loop Simulation (strength-reduced IV / pointer-bump coupling)
        ok_rel = self._try_relational_simulation(lp_l, idx_l, lp_r, idx_r, assumptions)
        if ok_rel:
            self.loops_proved_unbounded += 1
            self._record(l_out, r_out, "loop_relational")
            return True, l_out

        # Case D: Bounded unrolling fallback (e.g., 2x unrolled loops)
        ok_bnd, tmpl_bnd = self._z3_leaf_discharge(l_out, r_out, assumptions)
        if ok_bnd:
            return True, tmpl_bnd
        return False, HoleVal("FAIL", l_out.ty)

    def _try_lockstep_loop(
        self,
        lp_l: LoopVal,
        idx_l: int,
        lp_r: LoopVal,
        idx_r: int,
        assumptions: list[CoreVal],
    ) -> tuple[bool, CoreVal]:
        """Try aligning loop-carried variables (by name or position) and prove lockstep induction."""
        n = len(lp_l.iter_params)
        # Build alignment mapping from L index -> R index (prefer matching names & output index)
        perm: list[int] = [-1] * n
        used_r: set[int] = set()
        perm[idx_l] = idx_r
        used_r.add(idx_r)

        r_by_name = {p.name: j for j, p in enumerate(lp_r.iter_params)}
        for i, p in enumerate(lp_l.iter_params):
            if perm[i] != -1:
                continue
            if p.name in r_by_name and r_by_name[p.name] not in used_r and lp_r.iter_params[r_by_name[p.name]].ty == p.ty:
                perm[i] = r_by_name[p.name]
                used_r.add(perm[i])

        for i, p in enumerate(lp_l.iter_params):
            if perm[i] != -1:
                continue
            for j, pr in enumerate(lp_r.iter_params):
                if j not in used_r and pr.ty == p.ty:
                    perm[i] = j
                    used_r.add(j)
                    break

        if -1 in perm:
            return False, HoleVal("FAIL", lp_l.iter_params[idx_l].ty)

        # Fresh shared symbolic loop-state parameters for arbitrary iteration k >= 0
        shared_params = tuple(ParamVal(f"__iter_{lp_l.loop_id}_{i}", p.ty) for i, p in enumerate(lp_l.iter_params))
        map_l = {lp_l.iter_params[i].name: shared_params[i] for i in range(n)}
        map_r = {lp_r.iter_params[perm[i]].name: shared_params[i] for i in range(n)}

        # 1. Base case: init_L[i] ~ init_R[perm[i]]
        init_tmpls: list[CoreVal] = []
        for i in range(n):
            ok_init, ti = self.anti_unify(lp_l.init_vals[i], lp_r.init_vals[perm[i]], assumptions)
            if not ok_init:
                return False, HoleVal("FAIL", lp_l.iter_params[idx_l].ty)
            init_tmpls.append(ti)

        # 2. Guard equivalence: cond_L(s) ~ cond_R(s)
        cond_l = subst_val(lp_l.cond, map_l)
        cond_r = subst_val(lp_r.cond, map_r)
        ok_cond, tmpl_cond = self.anti_unify(cond_l, cond_r, assumptions)
        if not ok_cond:
            return False, HoleVal("FAIL", lp_l.iter_params[idx_l].ty)

        # 3. Body step equivalence under guard cond_L(s)
        step_tmpls: list[CoreVal] = []
        for i in range(n):
            bl = subst_val(lp_l.body_next[i], map_l)
            br = subst_val(lp_r.body_next[perm[i]], map_r)
            ok_step, ts = self.anti_unify(bl, br, assumptions + [cond_l])
            if not ok_step:
                return False, HoleVal("FAIL", lp_l.iter_params[idx_l].ty)
            step_tmpls.append(ts)

        tmpl_loop = LoopVal(
            iter_params=shared_params,
            init_vals=tuple(init_tmpls),
            cond=tmpl_cond,
            body_next=tuple(step_tmpls),
            ty=lp_l.ty,
            unroll_hint=lp_l.unroll_hint,
            loop_id=lp_l.loop_id,
        )
        return True, LoopOutVal(tmpl_loop, idx_l, lp_l.iter_params[idx_l].ty)

    def _try_relational_simulation(
        self,
        lp_l: LoopVal,
        idx_l: int,
        lp_r: LoopVal,
        idx_r: int,
        assumptions: list[CoreVal],
        custom_invariants: list[CoreVal] | None = None,
    ) -> bool:
        """Verify unbounded loop equivalence via a relational coupling invariant R(s_L, s_R).

        Automatically synthesizes candidate invariants for common compiler loop transforms:
          - Matching outputs & shared state: `sL_k == sR_j`
          - Strength reduction (linear IV): `sR_j == init_R + stride * (sL_i - init_L)`
          - Pointer-bump strength reduction: `sR_ptr == ptr.gep(base, stride * sL_i)`
        """
        nl, nr = len(lp_l.iter_params), len(lp_r.iter_params)
        s_l = tuple(ParamVal(f"__relL_{i}", p.ty) for i, p in enumerate(lp_l.iter_params))
        s_r = tuple(ParamVal(f"__relR_{j}", p.ty) for j, p in enumerate(lp_r.iter_params))
        map_l = {lp_l.iter_params[i].name: s_l[i] for i in range(nl)}
        map_r = {lp_r.iter_params[j].name: s_r[j] for j in range(nr)}

        rel_clauses: list[CoreVal] = list(custom_invariants or [])
        if not rel_clauses:
            # Synthesize candidate coupling clauses
            rel_clauses.append(OpApp("eq", (s_l[idx_l], s_r[idx_r]), BOOL))
            # Match same-named same-type variables
            r_by_name = {p.name: j for j, p in enumerate(lp_r.iter_params)}
            for i, pl in enumerate(lp_l.iter_params):
                if i != idx_l and pl.name in r_by_name:
                    j = r_by_name[pl.name]
                    if lp_r.iter_params[j].ty == pl.ty:
                        rel_clauses.append(OpApp("eq", (s_l[i], s_r[j]), BOOL))

            # Synthesize pointer-bump or linear strength-reduction invariant for extra R variables
            int_l_indices = [i for i, p in enumerate(lp_l.iter_params) if p.ty == INT]
            for j, pr in enumerate(lp_r.iter_params):
                if pr.name in {p.name for p in lp_l.iter_params}:
                    continue
                if isinstance(pr.ty, TPtr) and int_l_indices:
                    # Try ptr_r == ptr.gep(init_ptr_r, stride * i_l) for stride in (1, 2, 4, 8)
                    matched_ptr_iv = False
                    for il in int_l_indices:
                        # Prefer true linear induction variables whose step is `il + const`
                        step_l = self.canonicalize(subst_val(lp_l.body_next[il], map_l), assumptions)
                        if not (isinstance(step_l, OpApp) and step_l.op == "add" and any(isinstance(a, ConstVal) for a in step_l.args)):
                            continue
                        for stride in (1, 2, 4, 8):
                            off = (
                                s_l[il]
                                if stride == 1
                                else OpApp("mul", (ConstVal(stride, INT), s_l[il]), INT)
                            )
                            cand = OpApp("eq", (s_r[j], OpApp("ptr.gep", (lp_r.init_vals[j], off), pr.ty)), BOOL)
                            # Check step preservation of pointer bump `cur_p' == gep(p0, stride * il')`
                            step_r_j = subst_val(lp_r.body_next[j], map_r)
                            off_next = (
                                step_l
                                if stride == 1
                                else OpApp("mul", (ConstVal(stride, INT), step_l), INT)
                            )
                            step_ok, _ = self.z3.prove_equal(
                                step_r_j,
                                OpApp("ptr.gep", (lp_r.init_vals[j], off_next), pr.ty),
                                assumptions=assumptions + [cand],
                                timeout_ms=150,
                            )
                            if step_ok:
                                rel_clauses.append(cand)
                                matched_ptr_iv = True
                                break
                        if matched_ptr_iv:
                            break
                elif pr.ty == INT and int_l_indices:
                    # Try linear IV: s_r[j] == init_r[j] + stride * (s_l[il] - init_l[il])
                    for il in int_l_indices:
                        step_r = self.canonicalize(subst_val(lp_r.body_next[j], map_r), assumptions)
                        if isinstance(step_r, OpApp) and step_r.op == "add":
                            consts = [a.value for a in step_r.args if isinstance(a, ConstVal) and isinstance(a.value, int)]
                            if consts:
                                stride = consts[0]
                                diff = OpApp("sub", (s_l[il], lp_l.init_vals[il]), INT)
                                rhs_expr = OpApp("add", (lp_r.init_vals[j], OpApp("mul", (ConstVal(stride, INT), diff), INT)), INT)
                                rel_clauses.append(OpApp("eq", (s_r[j], rhs_expr), BOOL))
                                break

        if not rel_clauses:
            return False

        R = rel_clauses[0]
        for cl in rel_clauses[1:]:
            R = OpApp("and", (R, cl), BOOL)

        # 1. Base case: R(init_L, init_R)
        init_map = {s_l[i].name: lp_l.init_vals[i] for i in range(nl)}
        init_map.update({s_r[j].name: lp_r.init_vals[j] for j in range(nr)})
        R_init = subst_val(R, init_map)
        self.z3_leaf_calls += 1
        ok_base, _ = self.z3.prove_equal(R_init, ConstVal(True, BOOL), assumptions=assumptions, timeout_ms=self.z3_timeout_ms)
        if not ok_base:
            return False

        # 2. Guard equivalence: R(s_L, s_R) => cond_L(s_L) == cond_R(s_R)
        c_l = subst_val(lp_l.cond, map_l)
        c_r = subst_val(lp_r.cond, map_r)
        self.z3_leaf_calls += 1
        ok_guard, _ = self.z3.prove_equal(c_l, c_r, assumptions=assumptions + [R], timeout_ms=self.z3_timeout_ms)
        if not ok_guard:
            return False

        # 3. Inductive step preservation: R(s_L, s_R) /\ c_L => R(body_L(s_L), body_R(s_R))
        next_map = {s_l[i].name: subst_val(lp_l.body_next[i], map_l) for i in range(nl)}
        next_map.update({s_r[j].name: subst_val(lp_r.body_next[j], map_r) for j in range(nr)})
        R_next = subst_val(R, next_map)
        self.z3_leaf_calls += 1
        ok_step, _ = self.z3.prove_equal(R_next, ConstVal(True, BOOL), assumptions=assumptions + [R, c_l], timeout_ms=self.z3_timeout_ms)
        if not ok_step:
            return False

        # 4. Exit equivalence: R(s_L, s_R) /\ !c_L => s_L[idx_l] == s_R[idx_r]
        not_cl = OpApp("not", (c_l,), BOOL)
        self.z3_leaf_calls += 1
        ok_exit, _ = self.z3.prove_equal(
            s_l[idx_l], s_r[idx_r], assumptions=assumptions + [R, not_cl], timeout_ms=self.z3_timeout_ms
        )
        return ok_exit

    def verify_values(
        self,
        lhs: CoreVal,
        rhs: CoreVal,
        assumptions: list[CoreVal] | None = None,
    ) -> CoreAUResult:
        t0 = time.perf_counter()
        cost_before = val_tree_size(lhs)
        ok, tmpl = self.anti_unify(lhs, rhs, assumptions or [])
        cost_after = optimal_tree_cover_cost(lhs, self.proved_lhs)
        dt = (time.perf_counter() - t0) * 1000.0
        return CoreAUResult(
            equivalent=ok,
            status="unsat" if ok else ("sat" if self.counterexample and not self.counterexample.startswith("unknown") else "unknown"),
            template=tmpl if ok else None,
            tree_cost_before=cost_before,
            tree_cost_after=cost_after,
            z3_leaf_calls=self.z3_leaf_calls,
            structural_discharges=self.structural_discharges,
            loops_proved_unbounded=self.loops_proved_unbounded,
            elapsed_ms=dt,
            obligations=self.obligations,
            counterexample=self.counterexample,
        )

    def verify_functions(self, fn_l: CoreFunc, fn_r: CoreFunc) -> CoreAUResult:
        assumptions = list(fn_l.preconditions) + list(fn_r.preconditions)
        return self.verify_values(fn_l.ret_val, fn_r.ret_val, assumptions=assumptions)
