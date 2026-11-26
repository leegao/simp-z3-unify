import json
import os
import time

from core_ir import (
    BOOL,
    BV32,
    FLOAT,
    INT,
    MEM,
    AliasPolicy,
    ConstVal,
    CoreSSAAntiUnifier,
    CoreZ3Backend,
    LoopOutVal,
    LoopVal,
    OpApp,
    ParamVal,
    SimpToCoreLowerer,
    TPtr,
    TSym,
    TTuple,
    build_default_registry,
)
from simp.passes import const_fold_pass, cse_pass, dce_pass
from simp.self_parser import build_self_hosted_simp_parser


def _build_scaling_pair(n_terms: int):
    """Build an AC + opaque-UF + inverse + SSA-move pair of size O(n_terms)."""
    xs = [ParamVal(f"x{i}", FLOAT) for i in range(n_terms)]
    ys = [ParamVal(f"y{i}", FLOAT) for i in range(n_terms)]
    terms_l = []
    terms_r = []
    for i in range(n_terms):
        # Left: uf.exp(ssa.mov((x_i + y_i) - y_i + y_i))
        inner_l = OpApp(
            "ssa.mov",
            (OpApp("add", (OpApp("sub", (OpApp("add", (xs[i], ys[i]), FLOAT), ys[i]), FLOAT), ys[i]), FLOAT),),
            FLOAT,
        )
        # Right: uf.exp(y_i + x_i)
        inner_r = OpApp("add", (ys[i], xs[i]), FLOAT)
        terms_l.append(OpApp("uf.exp", (inner_l,), FLOAT))
        terms_r.append(OpApp("uf.exp", (inner_r,), FLOAT))

    # Reverse order on RHS so AC matching must permute all n_terms!
    terms_r.reverse()
    lhs = terms_l[0]
    for t in terms_l[1:]:
        lhs = OpApp("add", (lhs, t), FLOAT)
    rhs = terms_r[0]
    for t in terms_r[1:]:
        rhs = OpApp("add", (rhs, t), FLOAT)
    return lhs, rhs


def run_benchmarks() -> dict:
    reg = build_default_registry()
    workloads = []

    def bench(name: str, category: str, lhs, rhs, policy=None, assumptions=None):
        pol = policy or AliasPolicy()
        au = CoreSSAAntiUnifier(reg, pol)
        res = au.verify_values(lhs, rhs, assumptions=assumptions or [])
        # Also time direct monolithic Z3 for comparison (skip unbounded loops where monolithic Z3 only unrolls)
        t0 = time.perf_counter()
        z3_b = CoreZ3Backend(reg, pol)
        z3_ok, _ = z3_b.prove_equal(lhs, rhs, assumptions=assumptions or [], timeout_ms=2000)
        z3_ms = (time.perf_counter() - t0) * 1000.0
        workloads.append(
            {
                "name": name,
                "category": category,
                "equivalent": res.equivalent,
                "tree_before": res.tree_cost_before,
                "tree_after": res.tree_cost_after,
                "structural": res.structural_discharges,
                "z3_leaves": res.z3_leaf_calls,
                "unbounded_loops": res.loops_proved_unbounded,
                "au_ms": round(res.elapsed_ms, 3),
                "mono_z3_ok": z3_ok,
                "mono_z3_ms": round(z3_ms, 3),
            }
        )

    # 1. Transparent SSA moves + AC + UF
    x, y, z = ParamVal("x", FLOAT), ParamVal("y", FLOAT), ParamVal("z", FLOAT)
    bench(
        "AC + SSA Moves + Opaque UF",
        "Equational (AC/Move)",
        OpApp("uf.exp", (OpApp("ssa.mov", (OpApp("add", (OpApp("add", (x, y), FLOAT), z), FLOAT),), FLOAT),), FLOAT),
        OpApp("uf.exp", (OpApp("add", (z, OpApp("ssa.copy", (OpApp("add", (y, x), FLOAT),), FLOAT)), FLOAT),), FLOAT),
    )

    # 2. Associative-only matrix chain (monolithic Z3 CANNOT prove this without assoc axioms!)
    mat_ty = TSym("Matrix")
    A, B, C, D = ParamVal("A", mat_ty), ParamVal("B", mat_ty), ParamVal("C", mat_ty), ParamVal("D", mat_ty)
    bench(
        "Assoc-Only Matrix Chain (4-way)",
        "Equational (A-only)",
        OpApp("mat.mul", (OpApp("mat.mul", (OpApp("mat.mul", (A, B), mat_ty), C), mat_ty), D), mat_ty),
        OpApp("mat.mul", (A, OpApp("mat.mul", (B, OpApp("mat.mul", (C, D), mat_ty)), mat_ty)), mat_ty),
    )

    # 3. Inverse + Self-Inverse + Bitvector XOR cancellation
    bv_a, bv_k, bv_m = ParamVal("a", BV32), ParamVal("k", BV32), ParamVal("m", BV32)
    bench(
        "BV32 XOR Self-Inverse + Add/Sub",
        "Equational (Inverse)",
        OpApp("bv.add", (OpApp("bv.xor", (OpApp("bv.xor", (bv_a, bv_k), BV32), bv_k), BV32), OpApp("bv.sub", (bv_m, bv_m), BV32)), BV32),
        bv_a,
    )

    # 4. Partial AC + Strength Reduction leaf
    bench(
        "Partial AC + Strength-Red Leaf",
        "Hybrid AU+Z3",
        OpApp("add", (OpApp("uf.sin", (x,), FLOAT), OpApp("add", (y, y), FLOAT)), FLOAT),
        OpApp("add", (OpApp("mul", (ConstVal(2.0, FLOAT), y), FLOAT), OpApp("uf.sin", (x,), FLOAT)), FLOAT),
    )

    # 5. Pointer Aliasing (noalias store reorder + GEP fold + store-to-load forward)
    p = ParamVal("p", TPtr(INT, addr_space=0, tbaa_tag="int"))
    q = ParamVal("q", TPtr(INT, addr_space=0, tbaa_tag="int"))
    m0 = ParamVal("m0", MEM)
    v1, v2 = ParamVal("v1", INT), ParamVal("v2", INT)
    pol = AliasPolicy()
    pol.add_noalias("p", "q")
    p23 = OpApp("ptr.gep", (OpApp("ptr.gep", (p, ConstVal(2, INT)), p.ty), ConstVal(3, INT)), p.ty)
    p5 = OpApp("ptr.gep", (p, ConstVal(5, INT)), p.ty)
    st1 = OpApp("mem.store", (OpApp("mem.store", (m0, p23, v1), MEM), q, v2), MEM)
    st2 = OpApp("mem.store", (OpApp("mem.store", (m0, q, v2), MEM), p5, v1), MEM)
    bench(
        "NoAlias Store Commutation + GEP",
        "Memory-SSA/Alias",
        OpApp("mem.load", (st1, p5), INT),
        OpApp("mem.load", (st2, p23), INT),
        policy=pol,
    )

    # 6. Unbounded Lockstep Loop + LICM + Dead-Var Slicing
    n, a_i, b_i = ParamVal("n", INT), ParamVal("a", INT), ParamVal("b", INT)
    i_l, acc_l = ParamVal("i", INT), ParamVal("acc", INT)
    i_r, acc_r, inv_r, dead_r = ParamVal("i", INT), ParamVal("acc", INT), ParamVal("inv", INT), ParamVal("dead", INT)
    lp_l = LoopVal(
        (i_l, acc_l),
        (ConstVal(0, INT), ConstVal(0, INT)),
        OpApp("lt", (i_l, n), BOOL),
        (OpApp("add", (i_l, ConstVal(1, INT)), INT), OpApp("add", (acc_l, OpApp("add", (a_i, b_i), INT)), INT)),
        TTuple((INT, INT)),
        loop_id="L1",
    )
    lp_r = LoopVal(
        (i_r, acc_r, inv_r, dead_r),
        (ConstVal(0, INT), ConstVal(0, INT), OpApp("add", (b_i, a_i), INT), ConstVal(0, INT)),
        OpApp("lt", (i_r, n), BOOL),
        (OpApp("add", (ConstVal(1, INT), i_r), INT), OpApp("add", (inv_r, acc_r), INT), inv_r, OpApp("add", (dead_r, ConstVal(7, INT)), INT)),
        TTuple((INT, INT, INT, INT)),
        loop_id="L2",
    )
    bench("Unbounded Loop (LICM + DeadVar)", "Loop (Lockstep)", LoopOutVal(lp_l, 1, INT), LoopOutVal(lp_r, 1, INT))

    # 7. Unbounded Relational Loop (Pointer-Bump Strength Reduction)
    p0 = ParamVal("p0", TPtr(INT))
    cur_p = ParamVal("cur_p", TPtr(INT))
    lp_idx = LoopVal(
        (i_l, acc_l),
        (ConstVal(0, INT), ConstVal(0, INT)),
        OpApp("lt", (i_l, n), BOOL),
        (OpApp("add", (i_l, ConstVal(1, INT)), INT), OpApp("add", (acc_l, OpApp("mem.load", (m0, OpApp("ptr.gep", (p0, i_l), p0.ty)), INT)), INT)),
        TTuple((INT, INT)),
        loop_id="Lidx",
    )
    lp_bmp = LoopVal(
        (i_r, acc_r, cur_p),
        (ConstVal(0, INT), ConstVal(0, INT), p0),
        OpApp("lt", (i_r, n), BOOL),
        (OpApp("add", (i_r, ConstVal(1, INT)), INT), OpApp("add", (acc_r, OpApp("mem.load", (m0, cur_p), INT)), INT), OpApp("ptr.gep", (cur_p, ConstVal(1, INT)), p0.ty)),
        TTuple((INT, INT, TPtr(INT))),
        loop_id="Lbmp",
    )
    bench("Pointer-Bump Strength Reduction", "Loop (Relational)", LoopOutVal(lp_idx, 1, INT), LoopOutVal(lp_bmp, 1, INT))

    # 8. Self-hosted s-imp parser lowered into CoreSSA-IR
    base = build_self_hosted_simp_parser()
    opt = dce_pass(cse_pass(const_fold_pass(base)))
    cm_l = SimpToCoreLowerer(base).lower_program()
    cm_r = SimpToCoreLowerer(opt).lower_program()
    bench("s-imp Self-Parser (next_token)", "Frontend (s-imp)", cm_l.get_func("next_token").ret_val, cm_r.get_func("next_token").ret_val)
    bench("s-imp Self-Parser (parse_assign)", "Frontend (s-imp)", cm_l.get_func("parse_assign_expr").ret_val, cm_r.get_func("parse_assign_expr").ret_val)

    # 9. Scaling benchmark: N = 4, 8, 16, 32, 64, 128 AC+UF+Inverse terms
    scaling = []
    for k in (4, 8, 16, 32, 64, 128):
        lhs_s, rhs_s = _build_scaling_pair(k)
        au_s = CoreSSAAntiUnifier(reg)
        res_s = au_s.verify_values(lhs_s, rhs_s)

        t0 = time.perf_counter()
        z3_b = CoreZ3Backend(reg, AliasPolicy())
        z3_ok, _ = z3_b.prove_equal(lhs_s, rhs_s, timeout_ms=5000)
        z3_ms = (time.perf_counter() - t0) * 1000.0
        scaling.append(
            {
                "k_terms": k,
                "tree_nodes": res_s.tree_cost_before,
                "tree_after": res_s.tree_cost_after,
                "au_ms": round(res_s.elapsed_ms, 3),
                "z3_leaves": res_s.z3_leaf_calls,
                "mono_z3_ms": round(z3_ms, 3),
                "mono_z3_ok": z3_ok,
                "speedup": round(z3_ms / max(res_s.elapsed_ms, 0.001), 2),
            }
        )

    return {"workloads": workloads, "scaling": scaling}


if __name__ == "__main__":
    data = run_benchmarks()
    out_path = os.environ.get("BENCH_OUT")
    if out_path:
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
    print(json.dumps(data, indent=2))
