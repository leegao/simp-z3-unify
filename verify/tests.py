import unittest

from simp.passes import (
    buggy_opt_pass,
    const_fold_pass,
    cse_pass,
    dce_pass,
    mem_licm_pass,
)
from simp.simp_ast import (
    FuncDecl,
    Program,
    ReturnStmt,
    Var,
)
from simp.simp_types import FLOAT, INT
from simp.typecheck import TypeErrorSIMP, typecheck_program

from .e2e_demo import build_demo_program, verify_pass


class TestSIMPVerify(unittest.TestCase):
    def test_typecheck_valid(self):
        prog = build_demo_program()
        sigs = typecheck_program(prog)
        self.assertIn("kernel", sigs)
        self.assertIn("decay_rec", sigs)

    def test_typecheck_rejects_mismatch(self):
        bad = Program([
            FuncDecl("f", [("x", INT)], FLOAT, [ReturnStmt(Var("x"))])
        ])
        with self.assertRaises(TypeErrorSIMP):
            typecheck_program(bad)

    def test_each_pass_and_full_pipeline_equiv(self):
        base = build_demo_program()
        cur = base
        for name, fn in [
            ("const_fold", const_fold_pass),
            ("cse", cse_pass),
            ("mem_licm", mem_licm_pass),
            ("dce", dce_pass),
        ]:
            nxt = fn(cur)
            res = verify_pass(name, cur, nxt)
            self.assertTrue(res.equivalent, f"{name} should be equivalent")
            self.assertEqual(res.z3_final_status, "unsat")
            cur = nxt
        res_all = verify_pass("full", base, cur)
        self.assertTrue(res_all.equivalent)

    def test_buggy_pass_rejected_with_counterexample(self):
        base = build_demo_program()
        buggy = buggy_opt_pass(base)
        res = verify_pass("buggy", base, buggy)
        self.assertFalse(res.equivalent)
        self.assertEqual(res.z3_final_status, "sat")
        self.assertIsNotNone(res.counterexample)


if __name__ == "__main__":
    unittest.main()
