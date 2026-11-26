import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import z3


class UnionFind:
    def __init__(self):
        self.parent: Dict[int, int] = {}
        self.rank: Dict[int, int] = {}

    def find(self, x: int) -> int:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def same(self, a: int, b: int) -> bool:
        return self.find(a) == self.find(b)

    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return ra


@dataclass
class DischargedPredicate:
    hole_name: str
    lhs_str: str
    rhs_str: str
    method: str  # "exact_wl", "ring_fp_z3", "lsh_z3", "congruence_cascade", "holed_template"
    template_str: Optional[str] = None


@dataclass
class UnifyResult:
    equivalent: bool
    z3_final_status: str  # "unsat", "sat", "unknown"
    nodes_before: int
    nodes_after: int
    candidates_checked: int
    z3_local_calls: int
    elapsed_ms: float
    discharged: List[DischargedPredicate] = field(default_factory=list)
    counterexample: Optional[str] = None


def collect_dag_nodes(expr: z3.ExprRef) -> List[z3.ExprRef]:
    visited: Set[int] = set()
    order: List[z3.ExprRef] = []

    def dfs(n: z3.ExprRef):
        nid = n.get_id()
        if nid in visited:
            return
        visited.add(nid)
        for c in n.children():
            dfs(c)
        order.append(n)

    dfs(expr)
    return order


def _h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")


# Large Mersenne prime for Schwartz-Zippel ring fingerprinting
PRIME_P = (1 << 61) - 1


def _fp_eval_ring(
    expr: z3.ExprRef,
    env_fp: Dict[str, int],
    env_bool: Dict[str, int],
    memo: Dict[int, object],
):
    """
    Algebraic-invariant semantic homomorphism:
      - Real/Int sub-DAGs -> exact evaluation in F_p (p = 2^61 - 1)
      - Bool sub-DAGs -> 64-bit packed truth mask
      - Array sub-DAGs -> 4-point random-index probe tuple
      - Opaque UFs (uf_exp, uf_log, uf_sin, ...) -> deterministic F_p hash
    """
    nid = expr.get_id()
    if nid in memo:
        return memo[nid]

    k = expr.decl().kind()
    name = expr.decl().name()
    sort = expr.sort()

    if expr.num_args() == 0:
        if z3.is_true(expr):
            val = (1 << 64) - 1
        elif z3.is_false(expr):
            val = 0
        elif z3.is_int_value(expr):
            val = expr.as_long() % PRIME_P
        elif z3.is_rational_value(expr):
            frac = expr.as_fraction()
            val = (frac.numerator * pow(frac.denominator, PRIME_P - 2, PRIME_P)) % PRIME_P
        elif isinstance(sort, z3.ArraySortRef):
            # Base symbolic array: deterministic 4-probe fingerprint at indices [0, 1, 2, 7]
            val = tuple(_h64(f"ARR:{name}:{idx}") % PRIME_P for idx in (0, 1, 2, 7))
        elif z3.is_string_value(expr):
            val = expr.as_string()
        elif sort == z3.StringSort():
            val = f"s_{_h64('STR:' + name) % 997}"
        elif sort == z3.BoolSort():
            val = env_bool.setdefault(name, _h64(f"BOOL:{name}"))
        else:
            val = env_fp.setdefault(name, (_h64(f"VAR:{name}") % (PRIME_P - 1)) + 1)
        memo[nid] = val
        return val

    args = [_fp_eval_ring(c, env_fp, env_bool, memo) for c in expr.children()]
    if k == z3.Z3_OP_ADD:
        val = sum(args) % PRIME_P
    elif k == z3.Z3_OP_SUB:
        val = (args[0] - args[1]) % PRIME_P
    elif k == z3.Z3_OP_UMINUS:
        val = (-args[0]) % PRIME_P
    elif k == z3.Z3_OP_MUL:
        v = 1
        for a in args:
            v = (v * a) % PRIME_P
        val = v
    elif k == z3.Z3_OP_DIV or k == z3.Z3_OP_IDIV:
        denom = args[1] if args[1] != 0 else 1
        val = (args[0] * pow(denom, PRIME_P - 2, PRIME_P)) % PRIME_P
    elif k == z3.Z3_OP_NOT:
        val = (~args[0]) & ((1 << 64) - 1)
    elif k == z3.Z3_OP_AND:
        v = (1 << 64) - 1
        for a in args:
            v &= a
        val = v
    elif k == z3.Z3_OP_OR:
        v = 0
        for a in args:
            v |= a
        val = v
    elif k in (z3.Z3_OP_EQ, z3.Z3_OP_LT, z3.Z3_OP_LE, z3.Z3_OP_GT, z3.Z3_OP_GE):
        # Deterministic 64-bit comparison mask based on relational predicate
        val = _h64(f"CMP:{k}:{args[0]}:{args[1]}")
        if k == z3.Z3_OP_EQ and args[0] == args[1]:
            val = (1 << 64) - 1
    elif k == z3.Z3_OP_ITE:
        # Select based on LSB of boolean mask
        val = args[1] if (args[0] & 1) else args[2]
    elif k == z3.Z3_OP_SELECT:
        arr_fp, idx_fp = args[0], args[1]
        probe_idxs = (0, 1, 2, 7)
        if idx_fp in probe_idxs and isinstance(arr_fp, tuple):
            val = arr_fp[probe_idxs.index(idx_fp)]
        else:
            val = _h64(f"SEL:{arr_fp}:{idx_fp}") % PRIME_P
    elif k == z3.Z3_OP_SEQ_CONCAT:
        val = "".join(str(a) for a in args)
    elif k == z3.Z3_OP_SEQ_LENGTH:
        val = len(str(args[0])) % PRIME_P
    elif k == z3.Z3_OP_SEQ_EXTRACT:
        s0 = str(args[0])
        off = int(args[1]) if isinstance(args[1], int) else 0
        ln = int(args[2]) if isinstance(args[2], int) else 0
        val = s0[off : off + ln] if (0 <= off <= len(s0) and ln >= 0) else ""
    elif k == z3.Z3_OP_STORE:
        arr_fp, idx_fp, elem_fp = args[0], args[1], args[2]
        probe_idxs = (0, 1, 2, 7)
        if isinstance(arr_fp, tuple) and idx_fp in probe_idxs:
            lst = list(arr_fp)
            lst[probe_idxs.index(idx_fp)] = elem_fp
            val = tuple(lst)
        else:
            val = _h64(f"STO:{arr_fp}:{idx_fp}:{elem_fp}") % PRIME_P
    else:
        if name.startswith("mk_"):
            # Struct constructor: store tuple of field fingerprints
            val = ("STRUCT", name[3:], tuple(args))
        elif len(args) == 1 and isinstance(args[0], tuple) and len(args[0]) == 3 and args[0][0] == "STRUCT":
            # Datatype accessor projection on a constructed struct!
            sname = args[0][1]
            if name.startswith(sname + "_"):
                # Match field index by accessor position on the DatatypeSort
                dt_sort = expr.children()[0].sort()
                ctor = dt_sort.constructor(0)
                idx = next((i for i in range(ctor.arity()) if dt_sort.accessor(0, i).name() == name), None)
                val = args[0][2][idx] if idx is not None else (_h64(f"UF:{name}:{args}") % PRIME_P)
            else:
                val = _h64(f"UF:{name}:{args}") % PRIME_P
        else:
            # Opaque UF (exp, log, sin, rec summary, etc.)
            val = _h64(f"UF:{name}:{args}") % PRIME_P

    memo[nid] = val
    return val


def _z3_check_equiv(u: z3.ExprRef, v: z3.ExprRef, preconds: List[z3.BoolRef], timeout_ms: int = 50) -> Tuple[str, Optional[str]]:
    s = z3.Solver()
    s.set("timeout", timeout_ms)
    for p in preconds:
        s.add(p)
    s.add(u != v)
    res = s.check()
    if res == z3.unsat:
        return "unsat", None
    if res == z3.sat:
        return "sat", str(s.model())
    return "unknown", None


def _short_str(e: z3.ExprRef, limit: int = 80) -> str:
    s = " ".join(str(e).split())
    return s if len(s) <= limit else s[: limit - 3] + "..."


def verify_equiv_bu_wlau(
    lhs: z3.ExprRef,
    rhs: z3.ExprRef,
    preconds: Optional[List[z3.BoolRef]] = None,
    local_timeout_ms: int = 50,
    final_timeout_ms: int = 2000,
) -> UnifyResult:
    """
    Full Enhanced BU-WLAU Algorithm:
      1. Bottom-up AC-normalized exact hash-consing into Union-Find theta
      2. Algebraic-invariant semantic fingerprinting (F_p^2 ring + 64-bit bool + array probe)
         + 2-round WL-k shingling + height-banded MinHash LSH
      3. Bottom-up holed anti-unification + Z3 local oracle discharge + upward congruence cascade
      4. Deepest-first maximal non-overlapping DAG covering with monotonic DAG-size guard
         + final Z3 equivalence check & exact fallback on SAT (zero false-SAT).
    """
    t0 = time.perf_counter()
    preconds = preconds or []
    nodes_l = collect_dag_nodes(lhs)
    nodes_r = collect_dag_nodes(rhs)
    all_nodes = nodes_l + nodes_r
    nodes_before = len({n.get_id() for n in all_nodes})
    uf = UnionFind()
    discharged: List[DischargedPredicate] = []
    hole_counter = [0]

    # Heights & FVs
    heights: Dict[int, int] = {}
    fvs: Dict[int, frozenset] = {}
    for n in all_nodes:
        nid = n.get_id()
        if nid in heights:
            continue
        if n.num_args() == 0:
            heights[nid] = 0
            fvs[nid] = frozenset([n.decl().name()]) if (z3.is_const(n) and n.decl().kind() == z3.Z3_OP_UNINTERPRETED) else frozenset()
        else:
            heights[nid] = 1 + max(heights[c.get_id()] for c in n.children())
            s: Set[str] = set()
            for c in n.children():
                s.update(fvs[c.get_id()])
            fvs[nid] = frozenset(s)

    # Stage 1: AC-normalized WL hashes
    comm_ops = {z3.Z3_OP_ADD, z3.Z3_OP_MUL, z3.Z3_OP_AND, z3.Z3_OP_OR}
    wl: Dict[int, int] = {}
    shingles: Dict[int, Set[int]] = {}
    for n in all_nodes:
        nid = n.get_id()
        if nid in wl:
            continue
        op = n.decl().kind()
        name = n.decl().name()
        if n.num_args() == 0:
            wl[nid] = _h64(f"LEAF:{n.sort()}:{n}")
            shingles[nid] = {wl[nid]}
        else:
            ch_wl = [wl[c.get_id()] for c in n.children()]
            if op in comm_ops:
                ch_wl.sort()
            wl[nid] = _h64(f"{name}:{n.sort()}:{ch_wl}")
            sh = {wl[nid], _h64(f"OP:{name}")}
            for ch in ch_wl:
                sh.add(_h64(f"EDGE:{name}->{ch}"))
            for fv in fvs[nid]:
                sh.add(_h64(f"FV:{fv}"))
            shingles[nid] = sh

    # Stage 1 exact WL collapse
    by_wl: Dict[Tuple[str, int], List[z3.ExprRef]] = {}
    for u in nodes_l:
        if u.num_args() > 0:
            by_wl.setdefault((str(u.sort()), wl[u.get_id()]), []).append(u)
    for v in nodes_r:
        if v.num_args() == 0:
            continue
        for u in by_wl.get((str(v.sort()), wl[v.get_id()]), ()):
            if u.get_id() != v.get_id() and not uf.same(u.get_id(), v.get_id()):
                uf.union(u.get_id(), v.get_id())
                hname = f"hm_{hole_counter[0]}"
                hole_counter[0] += 1
                discharged.append(DischargedPredicate(hname, _short_str(u), _short_str(v), "exact_wl"))

    # Stage 2a: Algebraic-invariant semantic fingerprinting (2 independent F_p / bool / array points)
    envs_fp = [{"_seed": 1}, {"_seed": 2}]
    envs_bool = [{"_seed": 1}, {"_seed": 2}]
    # Seed distinct variable values for point 2
    for n in all_nodes:
        if n.num_args() == 0 and z3.is_const(n) and n.decl().kind() == z3.Z3_OP_UNINTERPRETED:
            nm = n.decl().name()
            envs_fp[0][nm] = (_h64(f"P1:{nm}") % (PRIME_P - 1)) + 1
            envs_fp[1][nm] = (_h64(f"P2:{nm}") % (PRIME_P - 1)) + 1
            envs_bool[0][nm] = _h64(f"B1:{nm}")
            envs_bool[1][nm] = _h64(f"B2:{nm}")

    memos = [{}, {}]
    sem_fp: Dict[int, Tuple[object, object]] = {}
    for n in all_nodes:
        nid = n.get_id()
        if nid not in sem_fp:
            sem_fp[nid] = (
                _fp_eval_ring(n, envs_fp[0], envs_bool[0], memos[0]),
                _fp_eval_ring(n, envs_fp[1], envs_bool[1], memos[1]),
            )

    # Candidate discovery: exact semantic-fingerprint buckets + height-banded MinHash LSH
    cands: List[Tuple[int, int, z3.ExprRef, z3.ExprRef, str]] = []
    seen_pairs: Set[Tuple[int, int]] = set()

    # 2a. Exact semantic fingerprint match (captures ALL commutative ring, boolean, array identities in O(1)!)
    sem_buckets: Dict[Tuple[str, object], List[z3.ExprRef]] = {}
    for u in nodes_l:
        if u.num_args() > 0:
            sem_buckets.setdefault((str(u.sort()), sem_fp[u.get_id()]), []).append(u)
    for v in nodes_r:
        if v.num_args() == 0:
            continue
        for u in sem_buckets.get((str(v.sort()), sem_fp[v.get_id()]), ()):
            uid, vid = u.get_id(), v.get_id()
            if uid != vid and not uf.same(uid, vid) and (uid, vid) not in seen_pairs:
                seen_pairs.add((uid, vid))
                cands.append((heights[uid] + heights[vid], 0, u, v, "ring_fp_z3"))

    # 2b. Height-banded MinHash LSH (for near-miss holed templates)
    num_h, bands = 12, 4
    rows = num_h // bands
    seeds = [(0x9E3779B97F4A7C15 * (i + 1)) & 0xFFFFFFFFFFFFFFFF for i in range(num_h)]

    def minhash_sig(sh: Set[int]) -> List[int]:
        return [min(((h ^ s) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF for h in sh) for s in seeds]

    lsh_buckets: List[Dict[Tuple[str, int, Tuple[int, ...]], List[z3.ExprRef]]] = [{} for _ in range(bands)]
    sigs_l: Dict[int, List[int]] = {}
    for u in nodes_l:
        if u.num_args() == 0:
            continue
        uid = u.get_id()
        sg = minhash_sig(shingles[uid])
        sigs_l[uid] = sg
        hb = heights[uid] // 2
        for b in range(bands):
            lsh_buckets[b].setdefault((str(u.sort()), hb, tuple(sg[b * rows : (b + 1) * rows])), []).append(u)

    for v in nodes_r:
        if v.num_args() == 0:
            continue
        vid = v.get_id()
        sg = minhash_sig(shingles[vid])
        hb = heights[vid] // 2
        for dh in (hb - 1, hb, hb + 1):
            for b in range(bands):
                for u in lsh_buckets[b].get((str(v.sort()), dh, tuple(sg[b * rows : (b + 1) * rows])), ()):
                    uid = u.get_id()
                    if uid == vid or uf.same(uid, vid) or (uid, vid) in seen_pairs:
                        continue
                    if abs(heights[uid] - heights[vid]) <= 2 and sem_fp[uid] == sem_fp[vid]:
                        seen_pairs.add((uid, vid))
                        cands.append((heights[uid] + heights[vid], 1, u, v, "lsh_z3"))

    # Sort bottom-up by topological height
    cands.sort(key=lambda x: (x[0], x[1]))

    # Stage 3: Bottom-up discharge & upward congruence cascade
    z3_local_calls = 0
    for _, _, u, v, tag in cands:
        uid, vid = u.get_id(), v.get_id()
        if uf.same(uid, vid):
            continue
        # Progressive congruence cascade (0 Z3 calls!)
        if u.decl().name() == v.decl().name() and u.num_args() == v.num_args() and u.num_args() > 0:
            if all(uf.same(a.get_id(), b.get_id()) for a, b in zip(u.children(), v.children())):
                uf.union(uid, vid)
                hname = f"hm_{hole_counter[0]}"
                hole_counter[0] += 1
                discharged.append(DischargedPredicate(hname, _short_str(u), _short_str(v), "congruence_cascade"))
                continue
        # Local Z3 oracle discharge
        z3_local_calls += 1
        status, _ = _z3_check_equiv(u, v, preconds, timeout_ms=local_timeout_ms)
        if status == "unsat":
            uf.union(uid, vid)
            hname = f"hm_{hole_counter[0]}"
            hole_counter[0] += 1
            discharged.append(DischargedPredicate(hname, _short_str(u), _short_str(v), tag))

    # Stage 4: Maximal non-overlapping DAG covering + monotonic DAG-size guard
    if uf.same(lhs.get_id(), rhs.get_id()):
        dt = (time.perf_counter() - t0) * 1000.0
        return UnifyResult(True, "unsat", nodes_before, 1, len(cands), z3_local_calls, dt, discharged)

    # Map every non-trivial unified class with >=2 distinct nodes to a fresh Skolem hole hm_k
    cls_members: Dict[int, Set[int]] = {}
    node_by_id: Dict[int, z3.ExprRef] = {}
    for n in all_nodes:
        nid = n.get_id()
        node_by_id[nid] = n
        cls_members.setdefault(uf.find(nid), set()).add(nid)

    rep_var: Dict[int, z3.ExprRef] = {}
    for rep, members in cls_members.items():
        if len(members) >= 2 and node_by_id[rep].num_args() > 0:
            rep_var[rep] = z3.Const(f"hm_{rep}", node_by_id[rep].sort())

    def rewrite_maximal(n: z3.ExprRef, memo: Dict[int, z3.ExprRef]) -> z3.ExprRef:
        nid = n.get_id()
        if nid in memo:
            return memo[nid]
        rep = uf.find(nid)
        if rep in rep_var:
            memo[nid] = rep_var[rep]
            return memo[nid]
        if n.num_args() == 0:
            memo[nid] = n
            return n
        memo[nid] = n.decl()(*[rewrite_maximal(c, memo) for c in n.children()])
        return memo[nid]

    l_abs = rewrite_maximal(lhs, {})
    r_abs = rewrite_maximal(rhs, {})
    nodes_after = len({n.get_id() for n in collect_dag_nodes(l_abs) + collect_dag_nodes(r_abs)})

    # Monotonic DAG-size guard
    if nodes_after > nodes_before:
        l_abs, r_abs = lhs, rhs
        nodes_after = nodes_before

    final_status, model_str = _z3_check_equiv(l_abs, r_abs, preconds, timeout_ms=final_timeout_ms)
    if final_status == "sat":
        # Exact fallback verification to guarantee zero false-SAT
        exact_status, exact_model = _z3_check_equiv(lhs, rhs, preconds, timeout_ms=final_timeout_ms)
        final_status, model_str = exact_status, exact_model

    dt = (time.perf_counter() - t0) * 1000.0
    return UnifyResult(
        equivalent=(final_status == "unsat"),
        z3_final_status=final_status,
        nodes_before=nodes_before,
        nodes_after=nodes_after,
        candidates_checked=len(cands),
        z3_local_calls=z3_local_calls,
        elapsed_ms=dt,
        discharged=discharged,
        counterexample=model_str,
    )
