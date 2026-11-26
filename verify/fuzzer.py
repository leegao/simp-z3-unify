import random
import time
from dataclasses import dataclass

from simp.simp_ast import (
    ArrLoad,
    ArrStoreStmt,
    Assign,
    BinOp,
    BuiltinCall,
    Const,
    Expr,
    FuncCall,
    FuncDecl,
    IfStmt,
    IteExpr,
    LetDecl,
    Program,
    PtrDeref,
    PtrOffset,
    PtrStoreStmt,
    PyAttr,
    PyBox,
    PyCall,
    PyImportDecl,
    PyMethodCall,
    PyModuleRef,
    PyUnbox,
    ReturnStmt,
    Stmt,
    UnOp,
    Var,
    WhileStmt,
    typecheck_program,
)
from simp.passes import (
    buggy_opt_pass,
    const_fold_pass,
    cse_pass,
    dce_pass,
    mem_licm_pass,
)
from .symexec import SymbolicExecutor
from simp.simp_types import BOOL, FLOAT, INT, PYOBJ, TArray, TPtr
from .unifier import verify_equiv_bu_wlau


class SIMPFuzzer:
    """
    Type-directed fuzzer & stress-test synthesizer for s-imp programs with:
      - bool, int, float, array<float>, ptr<float>, pyobj
      - host-Python imports (math, numpy, scipy) & rigid PyAttr/PyMethodCall/PyBox/PyUnbox
      - transcendental builtins (exp, log, sin, cos, sqrt)
      - ite, if/else, bounded while loops, and recursive helpers
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.tmp_id = 0

    def _fresh(self, prefix: str = "t") -> str:
        self.tmp_id += 1
        return f"{prefix}_{self.tmp_id}"

    def _gen_float_expr(self, fvars: list[str], bvars: list[str], pyvars: list[str], depth: int) -> Expr:
        if depth <= 0 or self.rng.random() < 0.25:
            if self.rng.random() < 0.7 and fvars:
                return Var(self.rng.choice(fvars))
            return Const(self.rng.choice([0.0, 1.0, 2.0, 3.0, 0.5, -1.0]), FLOAT)

        choice = self.rng.choice([
            "add", "sub", "mul", "mul2", "dist", "builtin", "ite", "py_call", "py_method", "py_unbox"
        ])
        if choice == "add":
            return BinOp("+", self._gen_float_expr(fvars, bvars, pyvars, depth - 1),
                         self._gen_float_expr(fvars, bvars, pyvars, depth - 1))
        if choice == "sub":
            return BinOp("-", self._gen_float_expr(fvars, bvars, pyvars, depth - 1),
                         self._gen_float_expr(fvars, bvars, pyvars, depth - 1))
        if choice == "mul":
            return BinOp("*", self._gen_float_expr(fvars, bvars, pyvars, depth - 1),
                         self._gen_float_expr(fvars, bvars, pyvars, depth - 1))
        if choice == "mul2":
            # Triggers strength reduction x * 2.0 -> x + x (parent mismatch!)
            return BinOp("*", self._gen_float_expr(fvars, bvars, pyvars, depth - 1), Const(2.0, FLOAT))
        if choice == "dist":
            # Triggers distributivity (a + b) * c -> a*c + b*c (parent mismatch!)
            a = self._gen_float_expr(fvars, bvars, pyvars, depth - 1)
            b = self._gen_float_expr(fvars, bvars, pyvars, depth - 1)
            c = self._gen_float_expr(fvars, bvars, pyvars, depth - 1)
            return BinOp("*", BinOp("+", a, b), c)
        if choice == "builtin":
            bname = self.rng.choice(["exp", "log", "sin", "cos", "sqrt"])
            return BuiltinCall(bname, (self._gen_float_expr(fvars, bvars, pyvars, depth - 1),))
        if choice == "ite":
            cond = self._gen_bool_expr(fvars, bvars, depth - 1)
            return IteExpr(cond, self._gen_float_expr(fvars, bvars, pyvars, depth - 1),
                           self._gen_float_expr(fvars, bvars, pyvars, depth - 1))
        if choice == "py_call":
            # Call imported Python module attribute, e.g., np.tanh(expr)
            mod_fn = PyAttr(PyModuleRef("np"), self.rng.choice(["tanh", "sigmoid", "erf"]))
            arg = self._gen_float_expr(fvars, bvars, pyvars, depth - 1)
            return PyCall(mod_fn, (arg,), ret_ty=FLOAT)
        if choice == "py_method" and pyvars:
            obj = Var(self.rng.choice(pyvars))
            arg = self._gen_float_expr(fvars, bvars, pyvars, depth - 1)
            return PyMethodCall(obj, self.rng.choice(["eval", "project", "norm"]), (arg,), ret_ty=FLOAT)
        # py_unbox(py_method(py_box(x)))
        inner = PyBox(self._gen_float_expr(fvars, bvars, pyvars, depth - 1))
        return PyUnbox(PyMethodCall(inner, "clip", (Const(1.0, FLOAT),), ret_ty=PYOBJ), FLOAT)

    def _gen_bool_expr(self, fvars: list[str], bvars: list[str], depth: int) -> Expr:
        if depth <= 0 or self.rng.random() < 0.35:
            if bvars and self.rng.random() < 0.7:
                return Var(self.rng.choice(bvars))
            return Const(self.rng.choice([True, False]), BOOL)
        kind = self.rng.choice(["not", "notnot", "cmp", "and", "or"])
        if kind == "not":
            return UnOp("!", self._gen_bool_expr(fvars, bvars, depth - 1))
        if kind == "notnot":
            return UnOp("!", UnOp("!", self._gen_bool_expr(fvars, bvars, depth - 1)))
        if kind == "cmp":
            op = self.rng.choice(["<", "<=", ">", ">=", "=="])
            return BinOp(op, Var(self.rng.choice(fvars)), Const(self.rng.choice([0.0, 1.0]), FLOAT))
        if kind == "and":
            return BinOp("&&", self._gen_bool_expr(fvars, bvars, depth - 1), Const(True, BOOL))
        return BinOp("||", self._gen_bool_expr(fvars, bvars, depth - 1), Const(False, BOOL))

    def synthesize_program(self, num_stmts: int = 8, expr_depth: int = 2) -> Program:
        imports = [
            PyImportDecl("numpy", "np"),
            PyImportDecl("scipy.special", "sp"),
        ]
        # Recursive helper
        rec_fn = FuncDecl(
            name="rec_step",
            params=[("n", INT), ("v", FLOAT)],
            ret_ty=FLOAT,
            body=[
                IfStmt(
                    cond=BinOp("<=", Var("n"), Const(0, INT)),
                    then_body=[ReturnStmt(Var("v"))],
                    else_body=[
                        ReturnStmt(
                            FuncCall(
                                "rec_step",
                                (
                                    BinOp("-", Var("n"), Const(1, INT)),
                                    BinOp("+", BinOp("*", Var("v"), Const(0.5, FLOAT)), Const(0.0, FLOAT)),
                                ),
                            )
                        )
                    ],
                )
            ],
        )

        fvars = ["x", "y"]
        bvars = ["flag"]
        pyvars = ["ctx"]
        body: list[Stmt] = []

        # Create a host-Python object via module call
        obj_v = self._fresh("py")
        body.append(LetDecl(obj_v, PYOBJ, PyCall(PyAttr(PyModuleRef("sp"), "make_filter"), (Var("x"),), ret_ty=PYOBJ)))
        pyvars.append(obj_v)

        for _ in range(num_stmts):
            skind = self.rng.choice(["let_f", "cse_dup", "dead", "arr_rw", "ptr_rw", "if_stmt", "while_licm"])
            if skind == "let_f":
                vn = self._fresh("f")
                body.append(LetDecl(vn, FLOAT, self._gen_float_expr(fvars, bvars, pyvars, expr_depth)))
                fvars.append(vn)
            elif skind == "cse_dup":
                e = self._gen_float_expr(fvars, bvars, pyvars, expr_depth)
                v1, v2 = self._fresh("c1"), self._fresh("c2")
                body.append(LetDecl(v1, FLOAT, e))
                body.append(LetDecl(v2, FLOAT, e))
                fvars.extend([v1, v2])
            elif skind == "dead":
                vd = self._fresh("dead")
                body.append(LetDecl(vd, FLOAT, self._gen_float_expr(fvars, bvars, pyvars, expr_depth)))
                # Deliberately do NOT add vd to fvars so DCE removes it!
            elif skind == "arr_rw":
                idx = Const(self.rng.choice([0, 1, 2]), INT)
                val = self._gen_float_expr(fvars, bvars, pyvars, max(1, expr_depth - 1))
                vl = self._fresh("ld")
                body.append(ArrStoreStmt("buf", idx, val))
                body.append(LetDecl(vl, FLOAT, ArrLoad(Var("buf"), idx)))
                fvars.append(vl)
            elif skind == "ptr_rw":
                off = Const(self.rng.choice([0, 1]), INT)
                val = self._gen_float_expr(fvars, bvars, pyvars, max(1, expr_depth - 1))
                vp = self._fresh("pd")
                body.append(PtrStoreStmt(PtrOffset(Var("p"), off), val))
                body.append(LetDecl(vp, FLOAT, PtrDeref(PtrOffset(Var("p"), off))))
                fvars.append(vp)
            elif skind == "if_stmt":
                target = self.rng.choice(fvars)
                cond = self._gen_bool_expr(fvars, bvars, 1)
                body.append(IfStmt(
                    cond=cond,
                    then_body=[Assign(target, self._gen_float_expr(fvars, bvars, pyvars, max(1, expr_depth - 1)))],
                    else_body=[Assign(target, self._gen_float_expr(fvars, bvars, pyvars, max(1, expr_depth - 1)))],
                ))
            elif skind == "while_licm":
                iv = self._fresh("i")
                acc = self._fresh("acc")
                inv_b = BuiltinCall(self.rng.choice(["sin", "cos", "exp"]), (Var("y"),))
                body.append(LetDecl(iv, INT, Const(0, INT)))
                body.append(LetDecl(acc, FLOAT, Var(fvars[-1])))
                body.append(WhileStmt(
                    cond=BinOp("<", Var(iv), Const(2, INT)),
                    body=[
                        Assign(acc, BinOp("+", Var(acc), inv_b)),
                        Assign(iv, BinOp("+", Var(iv), Const(1, INT))),
                    ],
                    unroll_bound=2,
                ))
                fvars.append(acc)

        # Combine final expression with recursive call & host-Python call
        final_e = BinOp(
            "+",
            FuncCall("rec_step", (Const(1, INT), Var(fvars[-1]))),
            PyMethodCall(Var(pyvars[-1]), "finalize", (Var(fvars[0]),), ret_ty=FLOAT),
        )
        body.append(ReturnStmt(final_e))

        entry_fn = FuncDecl(
            name="fuzz_entry",
            params=[
                ("x", FLOAT),
                ("y", FLOAT),
                ("flag", BOOL),
                ("buf", TArray(FLOAT)),
                ("p", TPtr(FLOAT)),
                ("ctx", PYOBJ),
            ],
            ret_ty=FLOAT,
            body=body,
        )
        return Program([rec_fn, entry_fn], imports=imports)


@dataclass
class FuzzSummary:
    total_programs: int
    typecheck_ok: int
    sound_pipeline_unsat: int
    sound_pipeline_fail: int
    buggy_caught_sat: int
    total_discharged_preds: int
    avg_nodes_before: float
    avg_nodes_after: float
    total_ms: float


def run_fuzz_stress_test(num_programs: int = 12, seed: int = 2026) -> FuzzSummary:
    t0 = time.perf_counter()
    fuzzer = SIMPFuzzer(seed=seed)
    tc_ok = 0
    unsat_ok = 0
    sound_fail = 0
    buggy_sat = 0
    tot_preds = 0
    tot_before = 0
    tot_after = 0

    for i in range(num_programs):
        prog = fuzzer.synthesize_program(num_stmts=6, expr_depth=2)
        typecheck_program(prog)
        tc_ok += 1

        # Apply full 4-pass pipeline
        opt = dce_pass(mem_licm_pass(cse_pass(const_fold_pass(prog))))
        typecheck_program(opt)

        se_l = SymbolicExecutor(prog, rec_depth=2)
        se_r = SymbolicExecutor(opt, rec_depth=2)
        init_env, preconds = se_l.make_initial_inputs("fuzz_entry")
        se_r.share_uf_tables_from(se_l)

        lhs, _ = se_l.lower_function("fuzz_entry", init_env)
        rhs, _ = se_r.lower_function("fuzz_entry", init_env)

        res = verify_equiv_bu_wlau(lhs, rhs, preconds=preconds, local_timeout_ms=40, final_timeout_ms=1000)
        tot_preds += len(res.discharged)
        tot_before += res.nodes_before
        tot_after += res.nodes_after
        if res.equivalent:
            unsat_ok += 1
        else:
            sound_fail += 1

        # Negative check on mutated buggy pass
        bug = buggy_opt_pass(prog)
        se_b = SymbolicExecutor(bug, rec_depth=2)
        se_b.share_uf_tables_from(se_l)
        rhs_b, _ = se_b.lower_function("fuzz_entry", init_env)
        res_b = verify_equiv_bu_wlau(lhs, rhs_b, preconds=preconds, local_timeout_ms=40, final_timeout_ms=1000)
        if not res_b.equivalent and res_b.z3_final_status == "sat":
            buggy_sat += 1

    dt = (time.perf_counter() - t0) * 1000.0
    return FuzzSummary(
        total_programs=num_programs,
        typecheck_ok=tc_ok,
        sound_pipeline_unsat=unsat_ok,
        sound_pipeline_fail=sound_fail,
        buggy_caught_sat=buggy_sat,
        total_discharged_preds=tot_preds,
        avg_nodes_before=tot_before / max(1, num_programs),
        avg_nodes_after=tot_after / max(1, num_programs),
        total_ms=dt,
    )


if __name__ == "__main__":
    s = run_fuzz_stress_test(num_programs=15, seed=2026)
    print("=== s-imp Fuzzer Stress Test Summary ===")
    print(f"Synthesized Programs:    {s.total_programs} (typechecked: {s.typecheck_ok}/{s.total_programs})")
    print(f"Sound Pipeline UNSAT:    {s.sound_pipeline_unsat}/{s.total_programs} (failures: {s.sound_pipeline_fail})")
    print(f"Buggy Pass Caught (SAT): {s.buggy_caught_sat}/{s.total_programs}")
    print(f"Total Discharged Preds:  {s.total_discharged_preds}")
    print(f"Avg DAG Nodes:           {s.avg_nodes_before:.1f} -> {s.avg_nodes_after:.1f}")
    print(f"Total Wall Time:         {s.total_ms:.1f} ms")
