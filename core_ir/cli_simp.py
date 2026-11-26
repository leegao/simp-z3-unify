"""Equivalence checking CLI for `.simp` programs via CoreSSA-IR.

Parses and typechecks a `.simp` source file, applies a selectable set of
optimization passes to produce the RHS, lowers both the original (LHS) and
optimized (RHS) programs to CoreSSA-IR with `SimpToCoreLowerer`, and proves
each function equivalent with `CoreSSAAntiUnifier`.

Examples::

    # default: original vs const_fold+cse+dce, all functions
    python -m core_ir.cli_simp examples/demo.simp

    # identity check (no passes)
    python -m core_ir.cli_simp examples/demo.simp --passes none

    # pick passes and a single function, JSON output
    python -m core_ir.cli_simp examples/demo.simp --passes const_fold,dce -e main --json out.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import build_default_registry
from .cli_common import PairResult, dump_json, exit_code, print_result, print_summary
from .equational_au import CoreSSAAntiUnifier
from .lower_simp import SimpToCoreLowerer

# Canonical pass registry + application order (mirrors simp.harness).
PASSES = {
    "const_fold": "const_fold_pass",
    "cse": "cse_pass",
    "mem_licm": "mem_licm_pass",
    "dce": "dce_pass",
}
PASS_ORDER = ["const_fold", "cse", "mem_licm", "dce"]


def _apply_passes(prog, selected: list[str]):
    from simp import passes as P

    cur = prog
    for name in PASS_ORDER:
        if name in selected:
            cur = getattr(P, PASSES[name])(cur)
    return cur


def check_simp(
    path: str | Path,
    selected: list[str],
    entry: str | None,
    z3_timeout_ms: int,
) -> list[PairResult]:
    from simp.parser import parse
    from simp.typecheck import typecheck_program

    src = Path(path).read_text()
    prog = parse(src)
    typecheck_program(prog)
    opt = _apply_passes(prog, selected)

    mod_l = SimpToCoreLowerer(prog).lower_program()
    mod_r = SimpToCoreLowerer(opt).lower_program()

    reg = build_default_registry()
    au = CoreSSAAntiUnifier(reg, z3_timeout_ms=z3_timeout_ms)

    fn_names = [entry] if entry else list(mod_l.funcs)
    results: list[PairResult] = []
    pass_label = "+".join(selected) if selected else "none"
    for fn in fn_names:
        if fn not in mod_l.funcs or fn not in mod_r.funcs:
            results.append(PairResult(f"{fn} [{pass_label}]", fn, fn, False, "missing", error=f"function {fn!r} not found"))
            continue
        res = au.verify_functions(mod_l.get_func(fn), mod_r.get_func(fn))
        results.append(PairResult.from_au(f"{fn} [{pass_label}]", fn, fn, res))
    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="core_ir.cli_simp",
        description="Prove equivalence of a .simp program across a toggleable set of passes via CoreSSA-IR.",
    )
    p.add_argument("file", help="input .simp source file")
    p.add_argument("-e", "--entry", default=None, help="only check this function (default: all)")
    p.add_argument(
        "--passes",
        default="const_fold,cse,dce",
        help="comma-separated passes (subset of " + ",".join(PASS_ORDER) + "); 'all' or 'none' (default: const_fold,cse,dce)",
    )
    p.add_argument("--z3-timeout", type=int, default=500, help="per-leaf Z3 timeout (ms)")
    p.add_argument("--json", default=None, help="write JSON results to this path (default: stdout)")
    p.add_argument("-v", "--verbose", action="store_true", help="print per-obligation details")
    args = p.parse_args(argv)

    sel = list(PASS_ORDER)
    raw = args.passes.strip().lower()
    if raw == "none":
        sel = []
    elif raw != "all":
        sel = [s.strip() for s in args.passes.split(",") if s.strip()]
        unknown = [s for s in sel if s not in PASSES]
        if unknown:
            print(f"error: unknown pass(es): {unknown}", file=sys.stderr)
            return 2

    try:
        results = check_simp(args.file, sel, args.entry, args.z3_timeout)
    except Exception as ex:  # noqa: BLE001 - surface parse/type/lower errors cleanly
        print(f"error: {ex}", file=sys.stderr)
        return 1

    for r in results:
        print_result(r, verbose=args.verbose)
    print_summary(results)
    if args.json:
        dump_json(results, args.json)
    return exit_code(results)


if __name__ == "__main__":
    raise SystemExit(main())