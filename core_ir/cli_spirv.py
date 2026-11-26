"""2-way equivalence checking CLI for SPIR-V / ValhallSSA shaders via CoreSSA-IR.

Lowers a single `.spv` (or `.spvasm`) shader to CoreSSA-IR through up to four
frontends and proves a chosen ordered pair equivalent:

  routes (ValhallSSA, via `valhall2vulkan/build/spirv2valhall`):
    - `direct`   : spirv2ssa direct lowering        (--direct-ssa --opt-ssa)
    - `ddk`      : libMali-Gxx + De-FAU/SSA opt      (--dump-ssa --defau --opt-ssa)
    - `panfrost` : Mesa Panfrost (spirv -> isa -> ssa)

  extra frontends:
    - `spirv`    : system `spirv-dis` text -> CoreSSA-IR

Examples::

    # direct vs ddk (default pair)
    python -m core_ir.cli_spirv shader.spv

    # choose the pair explicitly
    python -m core_ir.cli_spirv shader.spv --pair direct:panfrost
    python -m core_ir.cli_spirv shader.spv --lhs ddk --rhs spirv

    # machine-readable output
    python -m core_ir.cli_spirv shader.spv --pair direct:spirv --json out.json
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from .axioms import ProofSession, build_gpu_axiom_store
from .cli_common import PairResult, dump_json, exit_code, print_result, print_summary
from .gpu_dialect import build_gpu_registry
from .lower_spirv import lower_spirv_dis_to_core
from .lower_valhall_ssa import CoreShaderProgram, lower_valhall_ssa_to_core

# Default tool locations (override with flags / env vars).
DEFAULT_S2V = "/home/leegao/meh/valhall2vulkan/build/spirv2valhall"
DEFAULT_MALI = "/home/leegao/meh/valhall2vulkan/mali_offline_compiler/graphics/libMali-Gxx_r55p0-00rel0.so"
DEFAULT_PAN = "/home/leegao/meh/valhall2vulkan/third_party/prebuilts/panfrost/panfrost_compile"

VALHALL_ROUTES = ("direct", "ddk", "panfrost")
ALL_ROUTES = VALHALL_ROUTES + ("spirv",)


def _run(cmd: list[str], timeout: int = 20) -> str | None:
    try:
        r = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as ex:
        print(f"  command failed ({cmd[0]}): {ex}", file=sys.stderr)
        return None
    if r.returncode != 0 or "; ValhallSSA" not in r.stdout:
        return None
    return r.stdout


def emit_valhall(spv: str, route: str, s2v: str, mali: str, pan: str) -> str | None:
    if route == "direct":
        return _run([s2v, spv, "--direct-ssa", "--opt-ssa"])
    if route == "ddk":
        return _run([s2v, spv, "--dump-ssa", "--defau", "--opt-ssa", "--mali-lib", mali])
    if route == "panfrost":
        return _run([s2v, spv, "--panfrost", "--dump-ssa", "--defau", "--opt-ssa", "--panfrost-bin", pan])
    raise ValueError(route)


def emit_spirv_dis(spv: str) -> str | None:
    exe = shutil.which("spirv-dis")
    if exe is None:
        print("  spirv-dis not found on PATH", file=sys.stderr)
        return None
    try:
        r = subprocess.run([exe, spv], check=False, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as ex:
        print(f"  spirv-dis failed: {ex}", file=sys.stderr)
        return None
    return r.stdout if r.returncode == 0 else None


def load_programs(
    spv: str, routes: tuple[str, ...], s2v: str, mali: str, pan: str
) -> dict[str, CoreShaderProgram]:
    progs: dict[str, CoreShaderProgram] = {}
    for route in routes:
        if route == "spirv":
            text = emit_spirv_dis(spv)
            if text is None:
                continue
            progs[route] = lower_spirv_dis_to_core(text, Path(spv).stem)
        else:
            text = emit_valhall(spv, route, s2v, mali, pan)
            if text is None:
                print(f"  route '{route}' produced no ValhallSSA for {spv}", file=sys.stderr)
                continue
            progs[route] = lower_valhall_ssa_to_core(text, Path(spv).stem, route)
    return progs


def check_pair(
    lhs: CoreShaderProgram,
    rhs: CoreShaderProgram,
    lhs_name: str,
    rhs_name: str,
    reg,
    axioms,
    z3_timeout_ms: int,
) -> PairResult:
    label = f"{lhs_name}:{rhs_name}"
    if not lhs.outputs and not rhs.outputs:
        return PairResult(label, lhs_name, rhs_name, True, "empty", elapsed_ms=0.0)
    sess = ProofSession(lhs.outputs, rhs.outputs, reg, axioms)
    ok = sess.verify(z3_timeout_ms=z3_timeout_ms)
    res = PairResult(label, lhs_name, rhs_name, ok, "unsat" if ok else "sat")
    if not ok:
        fo = sess.failed_obligations[0] if sess.failed_obligations else None
        res.counterexample = fo.z3_counterexample if fo else None
        res.obligations = [
            {"hole": o.obligation_id, "method": "failed", "lhs": o.lhs.pretty()[:80], "rhs": o.rhs.pretty()[:80]}
            for o in sess.failed_obligations
        ]
    return res


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="core_ir.cli_spirv",
        description="2-way equivalence checking for a SPIR-V / ValhallSSA shader via CoreSSA-IR.",
    )
    p.add_argument("spv", help="input .spv (or .spvasm) shader")
    p.add_argument("--pair", default="direct:ddk", help="ordered pair to check, LHS:RHS (default: direct:ddk)")
    p.add_argument("--lhs", default=None, help="LHS route (overrides --pair)")
    p.add_argument("--rhs", default=None, help="RHS route (overrides --pair)")
    p.add_argument(
        "--routes",
        default=",".join(ALL_ROUTES),
        help="comma-separated routes to lower (default: all). Only the chosen pair is checked.",
    )
    p.add_argument("--s2v", default=DEFAULT_S2V, help="path to spirv2valhall binary")
    p.add_argument("--mali", default=DEFAULT_MALI, help="path to libMali-Gxx .so")
    p.add_argument("--pan", default=DEFAULT_PAN, help="path to panfrost_compile")
    p.add_argument("--strict-fp", action="store_true", help="disable fast-math FP axioms")
    p.add_argument("--z3-timeout", type=int, default=1200, help="per-channel Z3 timeout (ms)")
    p.add_argument("--json", default=None, help="write JSON results to this path (default: stdout)")
    p.add_argument("-v", "--verbose", action="store_true", help="print per-obligation details")
    args = p.parse_args(argv)

    if args.lhs or args.rhs:
        lhs_name = args.lhs or "direct"
        rhs_name = args.rhs or "ddk"
    else:
        if ":" not in args.pair:
            print(f"error: --pair must be LHS:RHS, got {args.pair!r}", file=sys.stderr)
            return 2
        lhs_name, rhs_name = args.pair.split(":", 1)

    for nm in (lhs_name, rhs_name):
        if nm not in ALL_ROUTES:
            print(f"error: unknown route {nm!r} (choose from {', '.join(ALL_ROUTES)})", file=sys.stderr)
            return 2

    routes = tuple(r.strip() for r in args.routes.split(",") if r.strip())
    for nm in (lhs_name, rhs_name):
        if nm not in routes:
            routes = routes + (nm,)

    progs = load_programs(args.spv, routes, args.s2v, args.mali, args.pan)
    if lhs_name not in progs or rhs_name not in progs:
        missing = [n for n in (lhs_name, rhs_name) if n not in progs]
        print(f"error: could not lower route(s) {missing} for {args.spv}", file=sys.stderr)
        return 1

    reg = build_gpu_registry(fp_fast_math=not args.strict_fp)
    axioms = build_gpu_axiom_store(include_fast_math=not args.strict_fp)

    res = check_pair(progs[lhs_name], progs[rhs_name], lhs_name, rhs_name, reg, axioms, args.z3_timeout)
    print_result(res, verbose=args.verbose)
    results = [res]
    print_summary(results)
    if args.json:
        dump_json(results, args.json)
    return exit_code(results)


if __name__ == "__main__":
    raise SystemExit(main())