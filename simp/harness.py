"""Equivalence harness for s-imp programs.

Given a ``.simp`` file, this harness:

  1. parses and typechecks it,
  2. applies a selectable set of optimization passes (toggle on/off),
  3. lowers the original and optimized programs to Z3 (via
     :class:`verify.symexec.SymbolicExecutor`),
  4. runs the anti-unification unifier (:func:`verify.unifier.verify_equiv_bu_wlau`),
  5. prints a human-readable report:
       - every discharged unified pair (with the method that proved it),
       - the identified template expressions, pretty-printed in ``.simp``
         syntax with holes rendered as ``[[%xN]]``,
       - which pairs required a Z3 oracle (and how fast), plus the final
         equivalence check.

Run as a CLI::

    python -m simp.harness examples/demo.simp --passes const_fold,cse,mem_licm,dce
    python -m simp.harness examples/demo.simp --passes none      # identity check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import z3

from .passes import const_fold_pass, cse_pass, dce_pass, mem_licm_pass
from .simp_ast import Program, StructDecl
from .typecheck import typecheck_program

# Canonical pass registry + application order.
PASSES = {
    "const_fold": const_fold_pass,
    "cse": cse_pass,
    "mem_licm": mem_licm_pass,
    "dce": dce_pass,
}
PASS_ORDER = ["const_fold", "cse", "mem_licm", "dce"]

# Methods that discharge a pair with a Z3 oracle (vs. Z3-free structural rules).
Z3_METHODS = {"ring_fp_z3", "lsh_z3", "holed_template"}


# --------------------------------------------------------------------------- #
# Z3 -> .simp pretty-printer
# --------------------------------------------------------------------------- #


def _collect_holes(exprs: list[z3.ExprRef]) -> dict[str, str]:
    """Map each ``hm_*`` hole constant to ``[[%xN]]`` in first-appearance order."""
    holes: dict[str, str] = {}

    def dfs(n: z3.ExprRef) -> None:
        if n.num_args() == 0:
            nm = n.decl().name()
            if nm.startswith("hm_") and nm not in holes:
                holes[nm] = f"[[%x{len(holes)}]]"
        for c in n.children():
            dfs(c)

    for e in exprs:
        if e is not None:
            dfs(e)
    return holes


def _nary(args: list[str], op: str) -> str:
    s = args[0]
    for a in args[1:]:
        s = f"({s} {op} {a})"
    return s


def _simp_type(sort: z3.SortRef) -> str:
    s = str(sort)
    return {
        "Real": "float",
        "Int": "int",
        "Bool": "bool",
        "String": "str",
    }.get(s, s)


def z3_to_simp(e: z3.ExprRef, holes: dict[str, str], structs: dict[str, StructDecl]) -> str:
    """Render a Z3 expression as (approximate) ``.simp`` surface syntax."""
    if z3.is_true(e):
        return "true"
    if z3.is_false(e):
        return "false"
    if z3.is_int_value(e):
        return str(e.as_long())
    if z3.is_rational_value(e):
        return str(float(e.as_fraction()))
    if z3.is_string_value(e):
        return '"' + e.as_string() + '"'

    decl = e.decl()
    name = decl.name()
    kind = decl.kind()
    args = [z3_to_simp(c, holes, structs) for c in e.children()]

    if e.num_args() == 0:
        if name in holes:
            return holes[name]
        if name.startswith("dict_empty"):
            return "{}"
        return name

    # Arithmetic / logical / comparison operators.
    if kind == z3.Z3_OP_ADD:
        return _nary(args, "+")
    if kind == z3.Z3_OP_SUB:
        return f"({args[0]} - {args[1]})"
    if kind == z3.Z3_OP_UMINUS:
        return f"(-{args[0]})"
    if kind == z3.Z3_OP_MUL:
        return _nary(args, "*")
    if kind in (z3.Z3_OP_DIV, z3.Z3_OP_IDIV):
        return f"({args[0]} / {args[1]})"
    if kind == z3.Z3_OP_LT:
        return f"({args[0]} < {args[1]})"
    if kind == z3.Z3_OP_LE:
        return f"({args[0]} <= {args[1]})"
    if kind == z3.Z3_OP_GT:
        return f"({args[0]} > {args[1]})"
    if kind == z3.Z3_OP_GE:
        return f"({args[0]} >= {args[1]})"
    if kind == z3.Z3_OP_EQ:
        return f"({args[0]} == {args[1]})"
    if kind == z3.Z3_OP_DISTINCT:
        return f"({args[0]} != {args[1]})"
    if kind == z3.Z3_OP_AND:
        return _nary(args, "&&")
    if kind == z3.Z3_OP_OR:
        return _nary(args, "||")
    if kind == z3.Z3_OP_NOT:
        return f"(!{args[0]})"
    if kind == z3.Z3_OP_ITE:
        return f"({args[0]} ? {args[1]} : {args[2]})"
    if kind == z3.Z3_OP_SELECT:
        return f"{args[0]}[{args[1]}]"
    if kind == z3.Z3_OP_STORE:
        return f"store({args[0]}, {args[1]}, {args[2]})"
    if kind == z3.Z3_OP_SEQ_CONCAT:
        return f"({args[0]} ++ {args[1]})"
    if kind == z3.Z3_OP_SEQ_LENGTH:
        return f"len({args[0]})"
    if kind == z3.Z3_OP_SEQ_EXTRACT:
        return f"substr({args[0]}, {args[1]}, {args[2]})"

    # Datatype constructor / accessor (structs).
    if kind == z3.Z3_OP_DT_CONSTRUCTOR and name.startswith("mk_"):
        sname = name[3:]
        sd = structs.get(sname)
        if sd is not None:
            inner = ", ".join(f"{f}: {a}" for (f, _), a in zip(sd.fields, args))
            return f"{sname} {{ {inner} }}"
        return f"{name}({', '.join(args)})"
    if kind == z3.Z3_OP_DT_ACCESSOR:
        # Accessor name is "<Struct>_<field>".
        sname, _, field = name.partition("_")
        return f"{args[0]}.{field}"

    # Rigid host-Python UFs.
    if name.startswith("uf_exp"):
        return f"exp({args[0]})"
    if name.startswith("uf_log"):
        return f"log({args[0]})"
    if name.startswith("uf_sin"):
        return f"sin({args[0]})"
    if name.startswith("uf_cos"):
        return f"cos({args[0]})"
    if name.startswith("uf_sqrt"):
        return f"sqrt({args[0]})"
    if name.startswith("uf_rec_"):
        fn = name[len("uf_rec_"):]
        return f"{fn}({', '.join(args)})"
    if name.startswith("py_getattr_"):
        return f"{args[0]}.{name[len('py_getattr_'):]}"
    if name.startswith("py_box_"):
        return f"box({args[0]})"
    if name.startswith("py_unbox_"):
        ty = _simp_type(z3.RealSort() if name.endswith("Real") else z3.IntSort())
        return f"unbox<{ty}>({args[0]})"
    if name.startswith("py_call_"):
        return f"{args[0]}({', '.join(args[1:])})"
    if name.startswith("py_method_"):
        method = name[len("py_method_"):].split("_")[0]
        return f"{args[0]}.{method}({', '.join(args[1:])})"

    return f"{name}({', '.join(args)})"


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #


def _apply_passes(prog: Program, selected: list[str]) -> Program:
    cur = prog
    for name in PASS_ORDER:
        if name in selected:
            cur = PASSES[name](cur)
    return cur


def run_harness(
    path: str | Path,
    selected: list[str] | None = None,
    entry: str = "main",
    local_timeout_ms: int = 50,
    final_timeout_ms: int = 2000,
    verbose: bool = True,
):
    """Parse ``path``, apply ``selected`` passes, and prove equivalence.

    Returns the :class:`verify.unifier.UnifyResult`. Prints a report when
    ``verbose`` is true.
    """
    from verify.symexec import SymbolicExecutor
    from verify.unifier import verify_equiv_bu_wlau

    from .parser import parse

    if selected is None:
        selected = list(PASS_ORDER)

    src = Path(path).read_text()
    prog = parse(src)
    typecheck_program(prog)

    opt = _apply_passes(prog, selected)

    se_l = SymbolicExecutor(prog, rec_depth=3)
    se_r = SymbolicExecutor(opt, rec_depth=3)
    init_env, preconds = se_l.make_initial_inputs(entry)
    se_r.share_uf_tables_from(se_l)
    lhs, _ = se_l.lower_function(entry, init_env)
    rhs, _ = se_r.lower_function(entry, init_env)

    res = verify_equiv_bu_wlau(
        lhs, rhs, preconds=preconds,
        local_timeout_ms=local_timeout_ms, final_timeout_ms=final_timeout_ms,
    )

    if verbose:
        _report(path, entry, selected, prog, res)
    return res


def _report(path, entry, selected, prog: Program, res) -> None:
    structs = {sd.name: sd for sd in prog.structs}
    holes = _collect_holes([res.lhs_template, res.rhs_template])

    print("=" * 78)
    print(" s-imp Equivalence Harness")
    print("=" * 78)
    print(f"  file    : {path}")
    print(f"  entry   : {entry}")
    print(f"  passes  : {', '.join(selected) if selected else '(none / identity)'}")
    print(f"  typecheck: OK  (structs: {list(structs) or '-'})")
    print("-" * 78)
    verdict = "EQUIVALENT (UNSAT)" if res.equivalent else f"NOT EQUIVALENT ({res.z3_final_status})"
    print(f"  Result  : {verdict}")
    print(
        f"  DAG     : {res.nodes_before} -> {res.nodes_after} nodes | "
        f"candidates: {res.candidates_checked} | z3 local calls: {res.z3_local_calls} | "
        f"total: {res.elapsed_ms:.2f} ms"
    )
    if res.counterexample:
        print(f"  CEX     : {' '.join(res.counterexample.split())[:200]}")

    # --- Discharged unified pairs ------------------------------------- #
    print("-" * 78)
    print(f"  Discharged unified pairs ({len(res.discharged)}):")
    if not res.discharged:
        print("    (none)")
    for i, dp in enumerate(res.discharged):
        lhs_s = z3_to_simp(dp.lhs_expr, holes, structs) if dp.lhs_expr is not None else dp.lhs_str
        rhs_s = z3_to_simp(dp.rhs_expr, holes, structs) if dp.rhs_expr is not None else dp.rhs_str
        tag = f"  [z3 {dp.elapsed_ms:.3f} ms]" if dp.method in Z3_METHODS else ""
        print(f"    [{i}] ({dp.method}) {dp.hole_name}:")
        print(f"          {lhs_s}")
        print(f"        ~ {rhs_s}{tag}")

    # --- Identified templates ----------------------------------------- #
    print("-" * 78)
    print("  Identified templates (.simp syntax; holes = [[%xN]]):")
    if res.lhs_template is None:
        print("    (no template returned)")
    else:
        print("    LHS:")
        print("      " + z3_to_simp(res.lhs_template, holes, structs))
        print("    RHS:")
        print("      " + z3_to_simp(res.rhs_template, holes, structs))

    # --- Z3 oracle usage ---------------------------------------------- #
    print("-" * 78)
    z3_free = [dp for dp in res.discharged if dp.method not in Z3_METHODS]
    z3_used = [dp for dp in res.discharged if dp.method in Z3_METHODS]
    print("  Z3 oracle usage:")
    by_free: dict[str, int] = {}
    for dp in z3_free:
        by_free[dp.method] = by_free.get(dp.method, 0) + 1
    print(f"    Z3-free rules : {by_free or '(none)'}")
    if z3_used:
        print("    Z3-oracle discharges:")
        for dp in z3_used:
            print(f"      {dp.hole_name}  {dp.method:16s} {dp.elapsed_ms:.3f} ms")
    else:
        print("    Z3-oracle discharges: (none)")
    print(
        f"    Final check : {res.z3_final_status.upper()} "
        f"(local z3 calls={res.z3_local_calls}, total={res.elapsed_ms:.2f} ms)"
    )
    print("=" * 78)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="simp.harness",
        description="Prove equivalence of a .simp program across a toggleable set of passes.",
    )
    p.add_argument("file", help="input .simp source file")
    p.add_argument("-e", "--entry", default="main", help="entry function (default: main)")
    p.add_argument(
        "--passes",
        default="all",
        help="comma-separated passes to enable (subset of "
        + ",".join(PASS_ORDER)
        + "); 'all' (default) or 'none'",
    )
    p.add_argument("--local-timeout", type=int, default=50, help="per-pair Z3 timeout (ms)")
    p.add_argument("--final-timeout", type=int, default=2000, help="final Z3 timeout (ms)")
    args = p.parse_args(argv)

    sel = list(PASS_ORDER)
    if args.passes.strip().lower() == "none":
        sel = []
    elif args.passes.strip().lower() != "all":
        sel = [s.strip() for s in args.passes.split(",") if s.strip()]
        unknown = [s for s in sel if s not in PASSES]
        if unknown:
            print(f"error: unknown pass(es): {unknown}", file=sys.stderr)
            return 2

    try:
        res = run_harness(
            args.file, sel, args.entry,
            local_timeout_ms=args.local_timeout, final_timeout_ms=args.final_timeout,
        )
    except (OSError, Exception) as e:  # noqa: BLE001 - surface any failure cleanly
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0 if res.equivalent else 1


if __name__ == "__main__":
    raise SystemExit(main())