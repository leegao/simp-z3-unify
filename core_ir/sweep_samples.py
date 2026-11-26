import signal
import glob
import json
import os
import subprocess
import time

from core_ir.axioms import ProofSession, build_gpu_axiom_store
from core_ir.gpu_dialect import build_gpu_registry
from core_ir.lower_valhall_ssa import lower_valhall_ssa_to_core


def emit_ssa(spv_path: str, route: str, s2v: str, mali: str, pan: str, shim: str) -> str | None:
    if route == "direct":
        cmd = [s2v, spv_path, "--direct-ssa", "--opt-ssa"]
        env = os.environ
    elif route == "ddk":
        cmd = [s2v, spv_path, "--dump-ssa", "--defau", "--opt-ssa", "--mali-lib", mali]
        env = os.environ
    elif route == "panfrost":
        cmd = [s2v, spv_path, "--panfrost", "--dump-ssa", "--defau", "--opt-ssa", "--panfrost-bin", pan]
        env = dict(os.environ, LD_PRELOAD=shim)
    else:
        raise ValueError(route)
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=15)
    if r.returncode != 0 or "; ValhallSSA" not in r.stdout:
        return None
    return r.stdout


def run_sweep(
    samples_dir: str,
    extra_spvs: list[str],
    s2v: str,
    mali: str,
    pan: str,
    shim: str,
    fp_fast_math: bool = True,
    z3_timeout_ms: int = 1200,
) -> dict:
    reg = build_gpu_registry(fp_fast_math=fp_fast_math)
    axioms = build_gpu_axiom_store(include_fast_math=fp_fast_math)

    spvs = sorted(glob.glob(os.path.join(samples_dir, "*.spv"))) + list(extra_spvs)
    results = []
    stats = {
        "total_pairs": 0,
        "proved": 0,
        "failed": 0,
        "empty_outputs": 0,
        "by_pair_kind": {},
    }

    for spv in spvs:
        name = os.path.basename(spv)
        ssa_direct = emit_ssa(spv, "direct", s2v, mali, pan, shim)
        ssa_ddk = emit_ssa(spv, "ddk", s2v, mali, pan, shim)
        ssa_pan = emit_ssa(spv, "panfrost", s2v, mali, pan, shim)

        pairs = []
        if ssa_direct and ssa_ddk:
            pairs.append(("direct_vs_ddk", "direct", ssa_direct, "ddk", ssa_ddk))
        if ssa_direct and ssa_pan:
            pairs.append(("direct_vs_panfrost", "direct", ssa_direct, "panfrost", ssa_pan))
        if ssa_ddk and ssa_pan:
            pairs.append(("ddk_vs_panfrost", "ddk", ssa_ddk, "panfrost", ssa_pan))

        for kind, rl, tl, rr, tr in pairs:
            t0 = time.perf_counter()
            def _alarm_handler(_sig, _frm):
                raise TimeoutError("pair verification exceeded 4s budget")
            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(4)
            try:
                pl = lower_valhall_ssa_to_core(tl, name=name, route=rl)
                pr = lower_valhall_ssa_to_core(tr, name=name, route=rr)

                stats["total_pairs"] += 1
                kstat = stats["by_pair_kind"].setdefault(kind, {"total": 0, "proved": 0, "failed": 0})
                kstat["total"] += 1

                if not pl.outputs and not pr.outputs:
                    stats["empty_outputs"] += 1
                    status = "EMPTY_NO_OUTPUTS"
                    proved = True
                    stats["proved"] += 1
                    kstat["proved"] += 1
                    fails = []
                else:
                    sess = ProofSession(pl.outputs, pr.outputs, reg, axioms)
                    proved = sess.verify(z3_timeout_ms=z3_timeout_ms)
                    status = "PROVED" if proved else "FAILED"
                    if proved:
                        stats["proved"] += 1
                        kstat["proved"] += 1
                    else:
                        stats["failed"] += 1
                        kstat["failed"] += 1
                    fails = [
                        {
                            "channel": fo.channel,
                            "lhs": fo.lhs.pretty()[:120],
                            "rhs": fo.rhs.pretty()[:120],
                            "hint": fo.suggested_hint,
                            "cex": (fo.z3_counterexample or "")[:120],
                        }
                        for fo in sess.failed_obligations
                    ]

                signal.alarm(0)
            except TimeoutError as te:
                signal.alarm(0)
                proved = False
                status = "TIMEOUT"
                stats["failed"] += 1
                kstat["failed"] += 1
                fails = [{"channel": "timeout", "lhs": "", "rhs": "", "hint": str(te), "cex": "timeout"}]
            except Exception as ex:
                signal.alarm(0)
                proved = False
                status = "ERROR"
                stats["failed"] += 1
                kstat["failed"] += 1
                fails = [{"channel": "error", "lhs": "", "rhs": "", "hint": str(ex)[:100], "cex": ""}]

            dt_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            results.append(
                {
                    "shader": name,
                    "pair": kind,
                    "status": status,
                    "elapsed_ms": dt_ms,
                    "lhs_blocks": pl.num_blocks,
                    "lhs_nodes": pl.num_nodes,
                    "rhs_blocks": pr.num_blocks,
                    "rhs_nodes": pr.num_nodes,
                    "channels_l": sorted(pl.outputs.keys()),
                    "channels_r": sorted(pr.outputs.keys()),
                    "failures": fails,
                }
            )
            tag = "OK" if proved else "FAIL"
            print(f"  [{tag:4s}] {name:32s} | {kind:18s} | {dt_ms:7.2f} ms | ch={sorted(pl.outputs.keys())}", flush=True)
            for f_item in fails[:2]:
                print(f"         -> ch={f_item['channel']} | L={f_item['lhs'][:65]} | R={f_item['rhs'][:65]}", flush=True)

    return {"stats": stats, "results": results}


if __name__ == "__main__":
    out = run_sweep(
        samples_dir="/working_dir/c_635d66aa892d7238/valhall_workdir/samples",
        extra_spvs=["/working_dir/c_8c7eadb161a76028/workdir/build/s2v_test_comp.spv"],
        s2v="/working_dir/c_8c7eadb161a76028/workdir/build/spirv2valhall",
        mali="/working_dir/c_dfa052f9061a19b3/workdir/mali_offline_compiler/graphics/libMali-Gxx_r55p0-00rel0.so",
        pan="/working_dir/c_8c7eadb161a76028/panfrost_inspect/clc/panfrost_compile.patched",
        shim="/working_dir/c_8c7eadb161a76028/spirv_tools_bin/libspv_compat.so",
    )
    out_path = os.environ.get("SWEEP_OUT")
    if out_path:
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
    print("\nSUMMARY:", json.dumps(out["stats"], indent=2))
