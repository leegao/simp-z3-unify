"""Shared helpers for the `core_ir` equivalence-checking CLIs.

Both `core_ir.cli_spirv` (SPIR-V / ValhallSSA) and `core_ir.cli_simp` (.simp)
print results in the same human-readable + `--json` format so they can be
composed in scripts.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Any

from .equational_au import CoreAUResult


@dataclass
class PairResult:
    """Outcome of checking one ordered pair of programs / functions."""

    label: str
    lhs_name: str
    rhs_name: str
    equivalent: bool
    status: str
    elapsed_ms: float = 0.0
    z3_leaf_calls: int = 0
    structural_discharges: int = 0
    loops_proved_unbounded: int = 0
    tree_cost_before: int = 0
    tree_cost_after: int = 0
    counterexample: str | None = None
    obligations: list[dict[str, str]] = field(default_factory=list)
    error: str | None = None

    @classmethod
    def from_au(cls, label: str, lhs_name: str, rhs_name: str, res: CoreAUResult) -> PairResult:
        return cls(
            label=label,
            lhs_name=lhs_name,
            rhs_name=rhs_name,
            equivalent=res.equivalent,
            status=res.status,
            elapsed_ms=round(res.elapsed_ms, 3),
            z3_leaf_calls=res.z3_leaf_calls,
            structural_discharges=res.structural_discharges,
            loops_proved_unbounded=res.loops_proved_unbounded,
            tree_cost_before=res.tree_cost_before,
            tree_cost_after=res.tree_cost_after,
            counterexample=res.counterexample,
            obligations=[{"hole": o.hole_id, "method": o.method, "lhs": o.lhs, "rhs": o.rhs} for o in res.obligations],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "lhs": self.lhs_name,
            "rhs": self.rhs_name,
            "equivalent": self.equivalent,
            "status": self.status,
            "elapsed_ms": self.elapsed_ms,
            "z3_leaf_calls": self.z3_leaf_calls,
            "structural_discharges": self.structural_discharges,
            "loops_proved_unbounded": self.loops_proved_unbounded,
            "tree_cost_before": self.tree_cost_before,
            "tree_cost_after": self.tree_cost_after,
            "counterexample": self.counterexample,
            "obligations": self.obligations,
            "error": self.error,
        }


def print_result(r: PairResult, verbose: bool = False) -> None:
    tag = "PROVED" if r.equivalent else ("ERROR" if r.error else "FAILED")
    print(
        f"  [{tag:6s}] {r.label:28s} | {r.elapsed_ms:8.2f} ms | "
        f"z3={r.z3_leaf_calls} struct={r.structural_discharges} loops={r.loops_proved_unbounded}"
    )
    if r.error:
        print(f"           error: {r.error}")
        return
    if not r.equivalent and r.counterexample:
        print(f"           cex: {' '.join(r.counterexample.split())[:160]}")
    if verbose:
        for ob in r.obligations:
            print(f"           [{ob['method']}] {ob['lhs'][:60]}  ~  {ob['rhs'][:60]}")


def print_summary(results: list[PairResult]) -> None:
    proved = sum(1 for r in results if r.equivalent)
    failed = sum(1 for r in results if not r.equivalent and r.error is None)
    errored = sum(1 for r in results if r.error is not None)
    print("-" * 78)
    print(f"  total={len(results)}  proved={proved}  failed={failed}  error={errored}")


def dump_json(results: list[PairResult], out_path: str | None) -> None:
    payload = {"results": [r.to_dict() for r in results]}
    text = json.dumps(payload, indent=2)
    if out_path:
        with open(out_path, "w") as f:
            f.write(text)
        print(f"  wrote {out_path}", file=sys.stderr)
    else:
        print(text)


def exit_code(results: list[PairResult]) -> int:
    """0 iff every pair was proved (or there were no pairs)."""
    return 0 if all(r.equivalent for r in results) else 1