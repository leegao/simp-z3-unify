import unittest

from core_ir import (
    BOOL,
    BV32,
    FLOAT,
    INT,
    MEM,
    STR,
    AliasPolicy,
    AliasRelation,
    CFGBlock,
    CFGBr,
    CFGCondBr,
    CFGFunction,
    CFGInstr,
    CFGPhi,
    CFGRet,
    ConstVal,
    CoreSSAAntiUnifier,
    InverseSpec,
    LoopOutVal,
    LoopVal,
    OpApp,
    OpSpec,
    OpTrait,
    ParamVal,
    SimpToCoreLowerer,
    TPtr,
    TSym,
    TTuple,
    build_default_registry,
    lower_cfg_to_core,
)
from simp.passes import const_fold_pass, cse_pass, dce_pass
from simp.self_parser import build_self_hosted_simp_parser
from simp.simp_ast import (
    Assign,
    BinOp,
    BuiltinCall,
    Const,
    FuncDecl,
    LetDecl,
    Program,
    ReturnStmt,
    Var,
    WhileStmt,
)
from simp.simp_types import FLOAT as SIMP_FLOAT
from simp.simp_types import INT as SIMP_INT


class TestCoreSSAAntiUnification(unittest.TestCase):
    def setUp(self):
        self.reg = build_default_registry()

    def test_01_transparent_ssa_moves_and_intros(self):
        x = ParamVal("x", INT)
        y = ParamVal("y", INT)
        lhs = OpApp(
            "ssa.mov",
            (OpApp("add", (OpApp("ssa.intro", (x,), INT), OpApp("ssa.copy", (y,), INT)), INT),),
            INT,
        )
        rhs = OpApp("ssa.cast_noop", (OpApp("add", (x, y), INT),), INT)
        au = CoreSSAAntiUnifier(self.reg)
        res = au.verify_values(lhs, rhs)
        self.assertTrue(res.equivalent)
        self.assertEqual(res.z3_leaf_calls, 0)

    def test_02_ac_matching_and_opaque_uf_congruence(self):
        x, y, z = ParamVal("x", FLOAT), ParamVal("y", FLOAT), ParamVal("z", FLOAT)
        # uf.exp((x + y) + z)  vs  uf.exp(z + (y + x))
        lhs = OpApp("uf.exp", (OpApp("add", (OpApp("add", (x, y), FLOAT), z), FLOAT),), FLOAT)
        rhs = OpApp("uf.exp", (OpApp("add", (z, OpApp("add", (y, x), FLOAT)), FLOAT),), FLOAT)
        au = CoreSSAAntiUnifier(self.reg)
        res = au.verify_values(lhs, rhs)
        self.assertTrue(res.equivalent)
        self.assertEqual(res.z3_leaf_calls, 0)

        # Partial AC + minimal Z3 leaf: add(uf.sin(x), add(y, y)) vs add(mul(2.0, y), uf.sin(x))
        lhs2 = OpApp("add", (OpApp("uf.sin", (x,), FLOAT), OpApp("add", (y, y), FLOAT)), FLOAT)
        rhs2 = OpApp("add", (OpApp("mul", (ConstVal(2.0, FLOAT), y), FLOAT), OpApp("uf.sin", (x,), FLOAT)), FLOAT)
        au2 = CoreSSAAntiUnifier(self.reg)
        res2 = au2.verify_values(lhs2, rhs2)
        self.assertTrue(res2.equivalent)
        self.assertEqual(res2.z3_leaf_calls, 1)

    def test_03_assoc_only_non_commutative(self):
        mat_ty = TSym("Matrix")
        A, B, C = ParamVal("A", mat_ty), ParamVal("B", mat_ty), ParamVal("C", mat_ty)
        # (A * B) * C == A * (B * C) under ASSOC-only (0 Z3 calls!)
        lhs = OpApp("mat.mul", (OpApp("mat.mul", (A, B), mat_ty), C), mat_ty)
        rhs = OpApp("mat.mul", (A, OpApp("mat.mul", (B, C), mat_ty)), mat_ty)
        au = CoreSSAAntiUnifier(self.reg)
        res = au.verify_values(lhs, rhs)
        self.assertTrue(res.equivalent)
        self.assertEqual(res.z3_leaf_calls, 0)

        # A * B != B * A (must NOT falsely commute!)
        bad_l = OpApp("mat.mul", (A, B), mat_ty)
        bad_r = OpApp("mat.mul", (B, A), mat_ty)
        au_bad = CoreSSAAntiUnifier(self.reg)
        res_bad = au_bad.verify_values(bad_l, bad_r)
        self.assertFalse(res_bad.equivalent)

        # String concat (ASSOC-only with identity "")
        s1, s2, s3 = ParamVal("s1", STR), ParamVal("s2", STR), ParamVal("s3", STR)
        sl = OpApp("str.concat", (OpApp("str.concat", (s1, ConstVal("", STR)), STR), OpApp("str.concat", (s2, s3), STR)), STR)
        sr = OpApp("str.concat", (OpApp("str.concat", (s1, s2), STR), s3), STR)
        res_s = CoreSSAAntiUnifier(self.reg).verify_values(sl, sr)
        self.assertTrue(res_s.equivalent)
        self.assertEqual(res_s.z3_leaf_calls, 0)

    def test_04_inverses_involutions_and_self_inverse(self):
        x, y, z = ParamVal("x", INT), ParamVal("y", INT), ParamVal("z", INT)
        # sub(add(x, y), y) == x
        lhs = OpApp("sub", (OpApp("add", (x, y), INT), y), INT)
        res = CoreSSAAntiUnifier(self.reg).verify_values(lhs, x)
        self.assertTrue(res.equivalent)
        self.assertEqual(res.z3_leaf_calls, 0)

        # sub(x, add(y, z)) == sub(sub(x, z), y)
        l2 = OpApp("sub", (x, OpApp("add", (y, z), INT)), INT)
        r2 = OpApp("sub", (OpApp("sub", (x, z), INT), y), INT)
        res2 = CoreSSAAntiUnifier(self.reg).verify_values(l2, r2)
        self.assertTrue(res2.equivalent)
        self.assertEqual(res2.z3_leaf_calls, 0)

        # Bitvector XOR self-inverse: bv.xor(bv.xor(a, k), k) == a
        a, k = ParamVal("a", BV32), ParamVal("k", BV32)
        xl = OpApp("bv.xor", (OpApp("bv.xor", (a, k), BV32), k), BV32)
        res_x = CoreSSAAntiUnifier(self.reg).verify_values(xl, a)
        self.assertTrue(res_x.equivalent)
        self.assertEqual(res_x.z3_leaf_calls, 0)

        # Guarded inverse: div(mul(fx, fy), fy) == fx requires fy != 0
        fx, fy = ParamVal("fx", FLOAT), ParamVal("fy", FLOAT)
        dl = OpApp("div", (OpApp("mul", (fx, fy), FLOAT), fy), FLOAT)
        nonzero_fy = OpApp("neq", (fy, ConstVal(0.0, FLOAT)), BOOL)
        res_guarded_ok = CoreSSAAntiUnifier(self.reg).verify_values(dl, fx, assumptions=[nonzero_fy])
        self.assertTrue(res_guarded_ok.equivalent)

        # Without fy != 0, Z3 must not blindly prove div(mul(fx, fy), fy) == fx
        res_unguarded = CoreSSAAntiUnifier(self.reg).verify_values(dl, fx)
        self.assertFalse(res_unguarded.equivalent)

    def test_05_pointers_aliasing_and_memory_ssa(self):
        p = ParamVal("p", TPtr(INT, addr_space=0, tbaa_tag="int"))
        q = ParamVal("q", TPtr(INT, addr_space=0, tbaa_tag="int"))
        m0 = ParamVal("m0", MEM)
        v1, v2 = ParamVal("v1", INT), ParamVal("v2", INT)

        # Store reordering: store(store(m0, p, v1), q, v2) vs store(store(m0, q, v2), p, v1)
        st_pq = OpApp("mem.store", (OpApp("mem.store", (m0, p, v1), MEM), q, v2), MEM)
        st_qp = OpApp("mem.store", (OpApp("mem.store", (m0, q, v2), MEM), p, v1), MEM)

        # With NO_ALIAS(p, q): EQUIVALENT with 0 Z3 calls!
        pol_noalias = AliasPolicy()
        pol_noalias.add_noalias("p", "q")
        res_ok = CoreSSAAntiUnifier(self.reg, pol_noalias).verify_values(st_pq, st_qp)
        self.assertTrue(res_ok.equivalent)
        self.assertEqual(res_ok.z3_leaf_calls, 0)

        # Without NO_ALIAS (MAY_ALIAS): NOT EQUIVALENT (SAT)!
        res_may = CoreSSAAntiUnifier(self.reg, AliasPolicy()).verify_values(st_pq, st_qp)
        self.assertFalse(res_may.equivalent)

        # GEP folding + store-to-load forwarding: load(store(m0, gep(gep(p, 2), 3), v1), gep(p, 5)) == v1
        p23 = OpApp("ptr.gep", (OpApp("ptr.gep", (p, ConstVal(2, INT)), p.ty), ConstVal(3, INT)), p.ty)
        p5 = OpApp("ptr.gep", (p, ConstVal(5, INT)), p.ty)
        ld = OpApp("mem.load", (OpApp("mem.store", (m0, p23, v1), MEM), p5), INT)
        res_fwd = CoreSSAAntiUnifier(self.reg).verify_values(ld, v1)
        self.assertTrue(res_fwd.equivalent)
        self.assertEqual(res_fwd.z3_leaf_calls, 0)

        # TBAA disjointness: ptr<int, tbaa="int"> vs ptr<float, tbaa="float">
        pf = ParamVal("pf", TPtr(FLOAT, addr_space=0, tbaa_tag="float"))
        fv = ParamVal("fv", FLOAT)
        pol_tbaa = AliasPolicy(tbaa_enabled=True)
        pol_tbaa.add_tbaa_child("int", "root")
        pol_tbaa.add_tbaa_child("float", "root")
        self.assertEqual(pol_tbaa.query(p, pf, self.reg), AliasRelation.NO_ALIAS)
        ld_past_f = OpApp("mem.load", (OpApp("mem.store", (m0, pf, fv), MEM), p), INT)
        ld_direct = OpApp("mem.load", (m0, p), INT)
        res_tbaa = CoreSSAAntiUnifier(self.reg, pol_tbaa).verify_values(ld_past_f, ld_direct)
        self.assertTrue(res_tbaa.equivalent)
        self.assertEqual(res_tbaa.z3_leaf_calls, 0)

        # Address-space disjointness: GPU shared (@3) vs global (@1)
        p_glob = ParamVal("pg", TPtr(INT, addr_space=1))
        p_shr = ParamVal("ps", TPtr(INT, addr_space=3))
        self.assertEqual(AliasPolicy().query(p_glob, p_shr, self.reg), AliasRelation.NO_ALIAS)

    def test_06_custom_dialect_annotations(self):
        reg = self.reg.clone()
        vec_ty = TSym("Vec")
        inv_enc = InverseSpec(forward_op="crypto.xor_pad", inverse_binop="crypto.unpad", unary_inverse="crypto.neg_pad")
        reg.register(OpSpec("crypto.xor_pad", OpTrait.AC | OpTrait.SELF_INVERSE | OpTrait.PURE, identity=0, has_identity=True, inverse=inv_enc))
        reg.register(OpSpec("dialect.kron", OpTrait.ASSOC | OpTrait.RIGID_UF | OpTrait.PURE))

        u, v, w = ParamVal("u", vec_ty), ParamVal("v", vec_ty), ParamVal("w", vec_ty)
        # Custom ASSOC-only op
        kl = OpApp("dialect.kron", (OpApp("dialect.kron", (u, v), vec_ty), w), vec_ty)
        kr = OpApp("dialect.kron", (u, OpApp("dialect.kron", (v, w), vec_ty)), vec_ty)
        res = CoreSSAAntiUnifier(reg).verify_values(kl, kr)
        self.assertTrue(res.equivalent)
        self.assertEqual(res.z3_leaf_calls, 0)

        # Custom SELF_INVERSE op: xor_pad(xor_pad(u, v), v) == u
        el = OpApp("crypto.xor_pad", (OpApp("crypto.xor_pad", (u, v), vec_ty), v), vec_ty)
        res_e = CoreSSAAntiUnifier(reg).verify_values(el, u)
        self.assertTrue(res_e.equivalent)
        self.assertEqual(res_e.z3_leaf_calls, 0)

    def test_07_unbounded_loops_lockstep_licm_deadvar(self):
        n, a, b = ParamVal("n", INT), ParamVal("a", INT), ParamVal("b", INT)
        i_l, acc_l = ParamVal("i", INT), ParamVal("acc", INT)
        i_r, acc_r, inv_r, dead_r = (
            ParamVal("i", INT),
            ParamVal("acc", INT),
            ParamVal("inv", INT),
            ParamVal("dead", INT),
        )

        # Loop L: while (i < n) { acc = acc + (a + b); i = i + 1; }
        lp_l = LoopVal(
            iter_params=(i_l, acc_l),
            init_vals=(ConstVal(0, INT), ConstVal(0, INT)),
            cond=OpApp("lt", (i_l, n), BOOL),
            body_next=(
                OpApp("add", (i_l, ConstVal(1, INT)), INT),
                OpApp("add", (acc_l, OpApp("add", (a, b), INT)), INT),
            ),
            ty=TTuple((INT, INT)),
            loop_id="L1",
        )
        # Loop R: LICM hoisted `inv = b + a` + dead counter `dead = dead + 7` + reassociated acc!
        lp_r = LoopVal(
            iter_params=(i_r, acc_r, inv_r, dead_r),
            init_vals=(ConstVal(0, INT), ConstVal(0, INT), OpApp("add", (b, a), INT), ConstVal(0, INT)),
            cond=OpApp("lt", (i_r, n), BOOL),
            body_next=(
                OpApp("add", (ConstVal(1, INT), i_r), INT),
                OpApp("add", (inv_r, acc_r), INT),
                inv_r,  # loop-invariant carried var!
                OpApp("add", (dead_r, ConstVal(7, INT)), INT),  # dead!
            ),
            ty=TTuple((INT, INT, INT, INT)),
            loop_id="L2",
        )
        au = CoreSSAAntiUnifier(self.reg)
        res = au.verify_values(LoopOutVal(lp_l, 1, INT), LoopOutVal(lp_r, 1, INT))
        self.assertTrue(res.equivalent)
        self.assertEqual(res.loops_proved_unbounded, 1)
        self.assertEqual(res.z3_leaf_calls, 0)

        # Negative test: off-by-one guard `i <= n` vs `i < n` -> MUST FAIL!
        lp_bug = LoopVal(
            iter_params=(i_l, acc_l),
            init_vals=(ConstVal(0, INT), ConstVal(0, INT)),
            cond=OpApp("le", (i_l, n), BOOL),
            body_next=lp_l.body_next,
            ty=lp_l.ty,
            loop_id="Lbug",
        )
        res_bug = CoreSSAAntiUnifier(self.reg).verify_values(LoopOutVal(lp_l, 1, INT), LoopOutVal(lp_bug, 1, INT))
        self.assertFalse(res_bug.equivalent)

    def test_08_relational_loop_strength_reduction(self):
        """Pointer-bump strength reduction inside an unbounded loop:
          L: while (i < n) { acc = acc + mem.load(M, gep(p0, i)); i = i + 1; }
          R: while (i < n) { acc = acc + mem.load(M, cur_p); cur_p = gep(cur_p, 1); i = i + 1; }
        """
        n = ParamVal("n", INT)
        p0 = ParamVal("p0", TPtr(INT))
        m0 = ParamVal("m0", MEM)
        i_l, acc_l = ParamVal("i", INT), ParamVal("acc", INT)
        i_r, acc_r, cur_p = ParamVal("i", INT), ParamVal("acc", INT), ParamVal("cur_p", TPtr(INT))

        lp_l = LoopVal(
            iter_params=(i_l, acc_l),
            init_vals=(ConstVal(0, INT), ConstVal(0, INT)),
            cond=OpApp("lt", (i_l, n), BOOL),
            body_next=(
                OpApp("add", (i_l, ConstVal(1, INT)), INT),
                OpApp("add", (acc_l, OpApp("mem.load", (m0, OpApp("ptr.gep", (p0, i_l), p0.ty)), INT)), INT),
            ),
            ty=TTuple((INT, INT)),
            loop_id="L_idx",
        )
        lp_r = LoopVal(
            iter_params=(i_r, acc_r, cur_p),
            init_vals=(ConstVal(0, INT), ConstVal(0, INT), p0),
            cond=OpApp("lt", (i_r, n), BOOL),
            body_next=(
                OpApp("add", (i_r, ConstVal(1, INT)), INT),
                OpApp("add", (acc_r, OpApp("mem.load", (m0, cur_p), INT)), INT),
                OpApp("ptr.gep", (cur_p, ConstVal(1, INT)), p0.ty),
            ),
            ty=TTuple((INT, INT, TPtr(INT))),
            loop_id="L_bump",
        )
        au = CoreSSAAntiUnifier(self.reg)
        res = au.verify_values(LoopOutVal(lp_l, 1, INT), LoopOutVal(lp_r, 1, INT))
        self.assertTrue(res.equivalent)
        self.assertEqual(res.loops_proved_unbounded, 1)

    def test_09_simp_frontend_lowering_and_self_parser(self):
        # 1. Structured s-imp with while loop + transcendental UF
        prog_l = Program(
            funcs=[
                FuncDecl(
                    name="kernel",
                    params=[("x", SIMP_FLOAT), ("y", SIMP_FLOAT), ("n", SIMP_INT)],
                    ret_ty=SIMP_FLOAT,
                    body=[
                        LetDecl("acc", SIMP_FLOAT, Const(0.0, SIMP_FLOAT)),
                        LetDecl("i", SIMP_INT, Const(0, SIMP_INT)),
                        WhileStmt(
                            cond=BinOp("<", Var("i"), Var("n")),
                            body=[
                                LetDecl("t", SIMP_FLOAT, BinOp("*", BinOp("+", Var("x"), Var("y")), Const(2.0, SIMP_FLOAT))),
                                Assign("acc", BinOp("+", Var("acc"), BuiltinCall("exp", (Var("t"),)))),
                                Assign("i", BinOp("+", Var("i"), Const(1, SIMP_INT))),
                            ],
                        ),
                        ReturnStmt(Var("acc")),
                    ],
                )
            ]
        )
        prog_r = dce_pass(cse_pass(const_fold_pass(prog_l)))
        mod_l = SimpToCoreLowerer(prog_l).lower_program()
        mod_r = SimpToCoreLowerer(prog_r).lower_program()
        au = CoreSSAAntiUnifier(self.reg)
        res = au.verify_functions(mod_l.get_func("kernel"), mod_r.get_func("kernel"))
        self.assertTrue(res.equivalent)
        self.assertEqual(res.loops_proved_unbounded, 1)

        # 2. Self-hosted s-imp parser lowered into CoreSSA-IR!
        base = build_self_hosted_simp_parser()
        opt = dce_pass(cse_pass(const_fold_pass(base)))
        cm_l = SimpToCoreLowerer(base).lower_program()
        cm_r = SimpToCoreLowerer(opt).lower_program()
        for fn_name in ("next_token", "parse_assign_expr"):
            res_p = CoreSSAAntiUnifier(self.reg).verify_functions(cm_l.get_func(fn_name), cm_r.get_func(fn_name))
            self.assertTrue(res_p.equivalent, f"Failed on {fn_name}: {res_p.counterexample}")

    def test_10_unstructured_cfg_frontend_lowering(self):
        """Lower two unstructured SSA-CFGs (with natural loop + header phis) into CoreSSA-IR."""
        n, a, b = ParamVal("n", INT), ParamVal("a", INT), ParamVal("b", INT)

        cfg1 = CFGFunction(
            name="sum_cfg",
            params=[n, a, b],
            ret_ty=INT,
            entry="entry",
            blocks={
                "entry": CFGBlock(
                    "entry",
                    instrs=[
                        CFGInstr("i0", "ssa.intro", (ConstVal(0, INT),), INT),
                        CFGInstr("acc0", "ssa.intro", (ConstVal(0, INT),), INT),
                    ],
                    term=CFGBr("header"),
                ),
                "header": CFGBlock(
                    "header",
                    phis=[
                        CFGPhi("i", INT, (("entry", "i0"), ("body", "i_next"))),
                        CFGPhi("acc", INT, (("entry", "acc0"), ("body", "acc_next"))),
                    ],
                    instrs=[CFGInstr("cond", "lt", ("i", "n"), BOOL)],
                    term=CFGCondBr("cond", "body", "exit"),
                ),
                "body": CFGBlock(
                    "body",
                    instrs=[
                        CFGInstr("s", "add", ("a", "b"), INT),
                        CFGInstr("acc_next", "add", ("acc", "s"), INT),
                        CFGInstr("i_next", "add", ("i", ConstVal(1, INT)), INT),
                    ],
                    term=CFGBr("header"),
                ),
                "exit": CFGBlock("exit", term=CFGRet("acc")),
            },
        )

        cfg2 = CFGFunction(
            name="sum_cfg_opt",
            params=[n, a, b],
            ret_ty=INT,
            entry="entry",
            blocks={
                "entry": CFGBlock(
                    "entry",
                    instrs=[
                        CFGInstr("i0", "ssa.mov", (ConstVal(0, INT),), INT),
                        CFGInstr("acc0", "ssa.mov", (ConstVal(0, INT),), INT),
                        CFGInstr("s_hoist", "add", ("b", "a"), INT),
                    ],
                    term=CFGBr("header"),
                ),
                "header": CFGBlock(
                    "header",
                    phis=[
                        CFGPhi("i", INT, (("entry", "i0"), ("body", "i_next"))),
                        CFGPhi("acc", INT, (("entry", "acc0"), ("body", "acc_next"))),
                    ],
                    instrs=[CFGInstr("cond", "lt", ("i", "n"), BOOL)],
                    term=CFGCondBr("cond", "body", "exit"),
                ),
                "body": CFGBlock(
                    "body",
                    instrs=[
                        CFGInstr("acc_next", "add", ("s_hoist", "acc"), INT),
                        CFGInstr("i_next", "add", (ConstVal(1, INT), "i"), INT),
                    ],
                    term=CFGBr("header"),
                ),
                "exit": CFGBlock("exit", term=CFGRet("acc")),
            },
        )

        fn1 = lower_cfg_to_core(cfg1)
        fn2 = lower_cfg_to_core(cfg2)
        au = CoreSSAAntiUnifier(self.reg)
        res = au.verify_functions(fn1, fn2)
        self.assertTrue(res.equivalent)
        self.assertEqual(res.loops_proved_unbounded, 1)
        self.assertEqual(res.z3_leaf_calls, 0)


if __name__ == "__main__":
    unittest.main()
