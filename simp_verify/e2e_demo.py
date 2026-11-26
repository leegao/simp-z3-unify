from collections.abc import Callable

from .ast_nodes import (
    ArrLoad,
    ArrStoreStmt,
    Assign,
    BinOp,
    BuiltinCall,
    Const,
    FuncCall,
    FuncDecl,
    IfStmt,
    IteExpr,
    LetDecl,
    Program,
    PtrDeref,
    PtrOffset,
    PtrStoreStmt,
    ReturnStmt,
    UnOp,
    Var,
    WhileStmt,
    typecheck_program,
)
from .passes import (
    buggy_opt_pass,
    const_fold_pass,
    cse_pass,
    dce_pass,
    mem_licm_pass,
)
from .symexec import SymbolicExecutor
from .types import BOOL, FLOAT, INT, TArray, TPtr
from .unifier import verify_equiv_bu_wlau


def build_demo_program() -> Program:
    """
    A rich s-imp program exercising:
      - Types: bool, int, float (Z3 Real), ptr<float>, array<float>
      - Control flow: recursive helper `decay_rec`, `while` loop, `if`/`else`, `ite`
      - Transcendental builtins: exp, log, sin (opaque Z3 UFs)
      - Optimization opportunities for const_fold, cse, mem_licm, and dce
    """
    # Recursive helper:
    # fn decay_rec(n: int, v: float) -> float:
    #   if (n <= 0) { return v; } else { return decay_rec(n - 1, v * 0.5 + 0.0); }
    decay_rec = FuncDecl(
        name="decay_rec",
        params=[("n", INT), ("v", FLOAT)],
        ret_ty=FLOAT,
        body=[
            IfStmt(
                cond=BinOp("<=", Var("n"), Const(0, INT)),
                then_body=[ReturnStmt(Var("v"))],
                else_body=[
                    ReturnStmt(
                        FuncCall(
                            "decay_rec",
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

    # Main kernel:
    # fn kernel(x: float, y: float, flag: bool, buf: array<float>, p: ptr<float>) -> float
    kernel = FuncDecl(
        name="kernel",
        params=[
            ("x", FLOAT),
            ("y", FLOAT),
            ("flag", BOOL),
            ("buf", TArray(FLOAT)),
            ("p", TPtr(FLOAT)),
        ],
        ret_ty=FLOAT,
        body=[
            # Dead code + constant folding opportunity
            LetDecl("dead_tmp", FLOAT, BinOp("*", BinOp("+", Var("x"), Const(3.0, FLOAT)), Const(1.0, FLOAT))),
            # Distributivity + algebraic simplification + transcendental UF:
            # base = exp((x + y) * 2.0)  --> const_fold rewrites * 2.0 -> (x+y)+(x+y) (parent mismatch!)
            LetDecl("s", FLOAT, BinOp("+", Var("x"), Var("y"))),
            LetDecl("e1", FLOAT, BuiltinCall("exp", (BinOp("*", BinOp("+", Var("x"), Var("y")), Const(2.0, FLOAT)),))),
            # Duplicate expression for CSE:
            LetDecl("e2", FLOAT, BuiltinCall("exp", (BinOp("*", BinOp("+", Var("x"), Var("y")), Const(2.0, FLOAT)),))),
            # Store to array + subsequent load at same index (store-to-load forwarding in mem_licm)
            ArrStoreStmt("buf", Const(1, INT), BinOp("+", Var("e1"), Var("e2"))),
            LetDecl("loaded", FLOAT, ArrLoad(Var("buf"), Const(1, INT))),
            # Pointer store & deref
            PtrStoreStmt(PtrOffset(Var("p"), Const(0, INT)), BuiltinCall("log", (BinOp("+", Var("x"), Const(1.0, FLOAT)),))),
            LetDecl("pval", FLOAT, PtrDeref(PtrOffset(Var("p"), Const(0, INT)))),
            # While loop with loop-invariant transcendental sin(y) for LICM
            LetDecl("i", INT, Const(0, INT)),
            LetDecl("acc", FLOAT, Var("loaded")),
            WhileStmt(
                cond=BinOp("<", Var("i"), Const(2, INT)),
                body=[
                    Assign("acc", BinOp("+", Var("acc"), BuiltinCall("sin", (Var("y"),)))),
                    Assign("i", BinOp("+", Var("i"), Const(1, INT))),
                ],
                unroll_bound=3,
            ),
            # Conditional with double negation + IteExpr + recursive call
            LetDecl(
                "scaled",
                FLOAT,
                IteExpr(
                    UnOp("!", UnOp("!", Var("flag"))),
                    FuncCall("decay_rec", (Const(2, INT), BinOp("*", BinOp("+", Var("acc"), Var("pval")), Var("s")))),
                    BinOp("-", Var("acc"), Const(0.0, FLOAT)),
                ),
            ),
            # Static dead branch for const_fold
            IfStmt(
                cond=BinOp("==", Const(1, INT), Const(2, INT)),
                then_body=[Assign("scaled", Const(999.0, FLOAT))],
                else_body=[],
            ),
            ReturnStmt(Var("scaled")),
        ],
    )
    return Program([decay_rec, kernel])


def verify_pass(name: str, prog_before: Program, prog_after: Program, entry: str = "kernel"):
    typecheck_program(prog_before)
    typecheck_program(prog_after)
    se_l = SymbolicExecutor(prog_before, rec_depth=3)
    se_r = SymbolicExecutor(prog_after, rec_depth=3)
    # Share exact initial Z3 symbolic variables & UF declarations across LHS and RHS
    init_env, preconds = se_l.make_initial_inputs(entry)
    se_r.uf_builtins = se_l.uf_builtins
    se_r.uf_rec_summaries = se_l.uf_rec_summaries

    lhs_ret, _ = se_l.lower_function(entry, init_env)
    rhs_ret, _ = se_r.lower_function(entry, init_env)

    res = verify_equiv_bu_wlau(lhs_ret, rhs_ret, preconds=preconds)
    print(f"\n=== Pass: {name} ===")
    print(
        f"  Status: {'EQUIVALENT (UNSAT)' if res.equivalent else 'NOT EQUIVALENT (' + res.z3_final_status.upper() + ')'} | "
        f"DAG nodes: {res.nodes_before} -> {res.nodes_after} | "
        f"Cands: {res.candidates_checked} | Local Z3: {res.z3_local_calls} | Time: {res.elapsed_ms:.2f} ms"
    )
    print(f"  Discharged Anti-Unification Predicates ({len(res.discharged)}):")
    for idx, dp in enumerate(res.discharged[:8]):
        print(f"    [{idx+1}] ({dp.method}) {dp.hole_name}:  {dp.lhs_str}  ~  {dp.rhs_str}")
    if len(res.discharged) > 8:
        print(f"    ... ({len(res.discharged) - 8} more discharged predicates)")
    print(f"  Final Z3 Equivalence Check: {res.z3_final_status.upper()}")
    if res.counterexample:
        cex_short = " ".join(res.counterexample.split())[:140]
        print(f"  Counterexample Model: {cex_short}...")
    return res


def run_e2e():
    base = build_demo_program()
    passes: list[tuple[str, Callable[[Program], Program]]] = [
        ("1. ConstFold + Algebraic Peephole", const_fold_pass),
        ("2. Common Subexpression Elimination (CSE)", cse_pass),
        ("3. Memory Forwarding + LICM", mem_licm_pass),
        ("4. Dead Code Elimination (DCE)", dce_pass),
    ]

    cur = base
    for pname, pfn in passes:
        nxt = pfn(cur)
        res = verify_pass(pname, cur, nxt)
        assert res.equivalent, f"Sound pass {pname} failed equivalence!"
        cur = nxt

    # Full end-to-end cumulative pipeline check (base ~ final optimized)
    res_all = verify_pass("5. Full Pipeline (Original ~ ConstFold+CSE+MemLICM+DCE)", base, cur)
    assert res_all.equivalent

    # Negative test: deliberately buggy optimization pass MUST return SAT!
    buggy = buggy_opt_pass(base)
    res_bug = verify_pass("6. Negative Soundness Test (Buggy Pass: a - b -> a + b)", base, buggy)
    assert not res_bug.equivalent and res_bug.z3_final_status == "sat"
    print("\nALL E2E VERIFICATION CHECKS PASSED (zero false-UNSAT, zero false-SAT)!")


if __name__ == "__main__":
    run_e2e()
