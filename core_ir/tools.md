# `core_ir` — CoreSSA-IR & Equivalence Checking Tools

`core_ir` is a self-contained Python package (no C++ build required) that defines a
minimal, immutable **CoreSSA-IR** kernel and an **equational tree-covering
anti-unifier** (`CoreSSAAntiUnifier`) for proving semantic equivalence between two
SSA programs. It is designed as the lightweight, embeddable counterpart to the
shader-level Z3 miter in `valhall2vulkan/tools/equivalence`: instead of one monolithic
Z3 query, it discharges proofs structurally (moves, AC/A-only matching, inverses,
Memory-SSA/aliasing, unbounded loops) and only falls back to Z3 at residual leaves.

For the full C++/Z3 shader pipeline, see `valhall2vulkan/tools/equivalence/tools.md`.

---

## Module Map

| Module | Role |
| :--- | :--- |
| `ir.py` | CoreSSA-IR value types: `ConstVal`, `ParamVal`, `SymVal`, `HoleVal`, `OpApp`, `SelectVal`, `LoopVal`, `LoopOutVal`, plus `CoreFunc`/`CoreModule` and DAG utilities (`val_tree_size`, `subst_val`, `free_param_names`, `optimal_tree_cover_cost`). |
| `core_types.py` | Type lattice: `TBool`, `TInt`, `TBitVec`, `TFloat`, `TStr`, `TSym`, `TVector`, `TPtr` (provenance + addr-space + TBAA), `TMem`, `TArray`, `TStruct`, `TTuple`, `TVoid` and the singletons `BOOL/INT/BV32/BV64/FLOAT/STR/MEM/TOKEN/VOID`. |
| `op_registry.py` | `OpSpec`/`OpTrait`/`InverseSpec` annotations (AC, ASSOC, COMM, INVOLUTION, SELF_INVERSE, TRANSPARENT_MOVE, RIGID_UF, MEM_*, PTR_OFFSET, …) and `build_default_registry()`. |
| `gpu_dialect.py` | `build_gpu_registry(fp_fast_math=...)` — expanded registry covering all 155 ValhallSSA opcodes + SPIR-V ops with exact traits, Z3 aliases, and concrete constant folders. |
| `axioms.py` | `AxiomStore` + `ExtensionalAxiom` (term rewrites and extra Z3 constraints), `build_gpu_axiom_store()`, and `ProofSession` for operator-in-the-loop proving/falsification. |
| `aliasing.py` | `AliasPolicy` (provenance, `noalias`, address-space, TBAA), `decompose_pointer`, `normalize_memory_ssa` (store-to-load forwarding, dead-store elimination, disjoint-store commutation). |
| `equational_au.py` | `CoreSSAAntiUnifier` — the equivalence engine (`verify_values`, `verify_functions`) returning a `CoreAUResult` with template, tree-cost, Z3-leaf count, and counterexample. |
| `z3_backend.py` | `CoreZ3Backend` — denotational translation to Z3 (`prove_equal`, `prove_nonzero`) with pointer sorts, Memory-SSA arrays, structs, strings, and rigid UFs. |
| `lower_spirv.py` | `lower_spirv_dis_to_core(spv_text)` — lower textual SPIR-V disassembly (`spirv-dis` output) into a `CoreShaderProgram`. |
| `lower_valhall_ssa.py` | `lower_valhall_ssa_to_core(text, route=...)` — lower ValhallSSA assembly (direct / ddk / panfrost routes) into observable output channels. |
| `lower_cfg.py` | `lower_cfg_to_core(cfg)` — lower a reducible unstructured SSA-CFG (LLVM/SPIR-V/MLIR-style) into `SelectVal` + `LoopVal`. |
| `lower_simp.py` | `SimpToCoreLowerer` — lower structured `s-imp` programs into CoreSSA-IR with explicit Memory-SSA threading. |
| `tests.py` | 10 unit tests for the core anti-unifier (moves, AC/A-only, inverses, aliasing, loops, s-imp, CFG). |
| `test_gpu_ir.py` | 3 unit tests for the GPU pipeline (direct vs ddk vs panfrost vs spirv-dis, constant folding, inductive axioms/falsification). |
| `benchmark.py` | `run_benchmarks()` — workloads + scaling study comparing AU vs monolithic Z3. |
| `sweep_samples.py` | Batch sweep over a directory of `.spv` shaders, verifying direct/ddk/panfrost pairwise. |

---

## Running the Tests

```bash
# Core anti-unifier tests (no external tools needed)
.venv/bin/python -m unittest core_ir.tests -v

# GPU / Valhall / SPIR-V lowering + equivalence tests
.venv/bin/python -m unittest core_ir.test_gpu_ir -v

# Benchmarks (AU vs monolithic Z3, incl. scaling k=4..128)
.venv/bin/python -m core_ir.benchmark
```

---

## Equivalence Checking for SPIR-V / SSA-IR

There are three supported frontends, all producing the same `CoreShaderProgram`
(or `CoreFunc`) that `CoreSSAAntiUnifier` / `ProofSession` consume:

1. **ValhallSSA** (three routes: `direct`, `ddk`, `panfrost`) via
   `lower_valhall_ssa_to_core`.
2. **SPIR-V disassembly** via `lower_spirv_dis_to_core` (uses system `spirv-dis`).
3. **Unstructured SSA-CFG** via `lower_cfg_to_core` (hand-built or from any SSA CFG).

### Option A — One-shot script (ValhallSSA direct vs ddk vs panfrost vs spirv-dis)

The snippet below reproduces `core_ir/test_gpu_ir.py::test_01` against any `.spv`.
It shells out to `valhall2vulkan/build/spirv2valhall` for the three Valhall routes
and to system `spirv-dis` for the SPIR-V route, then proves all four equivalent.

```python
import subprocess
from core_ir.axioms import ProofSession, build_gpu_axiom_store
from core_ir.gpu_dialect import build_gpu_registry
from core_ir.lower_valhall_ssa import lower_valhall_ssa_to_core
from core_ir.lower_spirv import lower_spirv_dis_to_core

S2V  = "/home/leegao/meh/valhall2vulkan/build/spirv2valhall"
MALI = "/home/leegao/meh/valhall2vulkan/mali_offline_compiler/graphics/libMali-Gxx_r55p0-00rel0.so"
PAN  = "/home/leegao/meh/valhall2vulkan/third_party/prebuilts/panfrost/panfrost_compile"
SPV  = "/home/leegao/meh/valhall2vulkan/samples/comp_rep_minimal.spv"

def emit(*args: str) -> str:
    return subprocess.run([S2V, SPV, *args], capture_output=True, text=True, timeout=15).stdout

ssa_direct = emit("--direct-ssa", "--opt-ssa")
ssa_ddk    = emit("--dump-ssa", "--defau", "--opt-ssa", "--mali-lib", MALI)
ssa_pan    = emit("--panfrost", "--dump-ssa", "--defau", "--opt-ssa", "--panfrost-bin", PAN)
spv_dis    = subprocess.run(["spirv-dis", SPV], capture_output=True, text=True).stdout

reg     = build_gpu_registry(fp_fast_math=True)
axioms  = build_gpu_axiom_store(include_fast_math=True)

progs = {
    "direct":   lower_valhall_ssa_to_core(ssa_direct, "comp", "direct"),
    "ddk":      lower_valhall_ssa_to_core(ssa_ddk,    "comp", "ddk"),
    "panfrost": lower_valhall_ssa_to_core(ssa_pan,    "comp", "panfrost"),
    "spirv":    lower_spirv_dis_to_core(spv_dis,      "comp"),
}

base = progs["direct"]
for label, rhs in progs.items():
    if label == "direct":
        continue
    sess = ProofSession(base.outputs, rhs.outputs, reg, axioms)
    ok = sess.verify()
    print(f"direct vs {label:9s}: {'PROVED' if ok else 'FAILED'}")
    for fo in sess.failed_obligations:
        print(fo.summary())
```

Expected output:

```
direct vs ddk      : PROVED
direct vs panfrost : PROVED
direct vs spirv    : PROVED
```

### Option B — Batch sweep over a corpus

`sweep_samples.py` automates Option A over every `*.spv` in a directory (plus
optional extras), verifying `direct_vs_ddk`, `direct_vs_panfrost`, and
`ddk_vs_panfrost` with a 4 s per-pair alarm and JSON output:

```bash
SWEEP_OUT=/tmp/sweep.json .venv/bin/python -m core_ir.sweep_samples
```

Edit the `__main__` block of `core_ir/sweep_samples.py` to point `samples_dir`,
`s2v`, `mali`, `pan`, and `shim` at your local `valhall2vulkan` checkout (the
defaults reference a sandbox path). `shim` may be `""` on hosts where
`panfrost_compile` does not need `libspv_compat.so`.

### Option C — Hand-built SSA / CFG (no external tools)

For unit tests or synthetic pairs, build `CoreVal` trees directly and call the
anti-unifier. This is what `core_ir/tests.py` does:

```python
from core_ir import (
    build_default_registry, CoreSSAAntiUnifier,
    OpApp, ParamVal, ConstVal, INT, BOOL, LoopVal, LoopOutVal, TTuple,
)

reg = build_default_registry()
n   = ParamVal("n", INT)
i   = ParamVal("i", INT)
acc = ParamVal("acc", INT)
lp  = LoopVal(
    iter_params=(i, acc),
    init_vals=(ConstVal(0, INT), ConstVal(0, INT)),
    cond=OpApp("lt", (i, n), BOOL),
    body_next=(OpApp("add", (i, ConstVal(1, INT)), INT),
               OpApp("add", (acc, i), INT)),
    ty=TTuple((INT, INT)),
    loop_id="L",
)
res = CoreSSAAntiUnifier(reg).verify_values(LoopOutVal(lp, 1, INT), LoopOutVal(lp, 1, INT))
assert res.equivalent
```

For unstructured CFGs, build a `CFGFunction` (see `core_ir/tests.py::test_10`)
and call `lower_cfg_to_core(cfg)` before `verify_functions`.

---

## Reading the Result

`CoreSSAAntiUnifier.verify_values(lhs, rhs, assumptions=...)` returns a
`CoreAUResult`:

| Field | Meaning |
| :--- | :--- |
| `equivalent` | `True` iff every obligation discharged (structural or Z3 `unsat`). |
| `status` | `"unsat"` (proved), `"sat"` (counterexample), or `"unknown"` (timeout). |
| `template` | Anti-unification generalization (holes `?hN`) when proved. |
| `tree_cost_before` / `tree_cost_after` | BURS tree-covering cost before/after proving. |
| `z3_leaf_calls` | Number of residual leaves discharged by Z3 (0 = fully structural). |
| `structural_discharges` | Moves, AC/A-only matches, inverse cancels, alias/mem, opaque-UF, loops. |
| `loops_proved_unbounded` | Loops closed by lockstep/relational induction (no unrolling). |
| `obligations` | Per-hole `AUObligation` log (`method` = `move_peel`, `ac_match`, `inverse_cancel`, `alias_mem`, `opaque_uf`, `loop_lockstep`, `loop_relational`, `z3_leaf`, …). |
| `counterexample` | Z3 model string when `status == "sat"`. |

`ProofSession` (Option A/B) verifies **all observable output channels** of two
`CoreShaderProgram`s (e.g. `__heap`, `frag_color:<loc>`, `builtin_out:<name>`,
`vert_varying:<loc>`, `__alive`) and supports inductive repair:

```python
sess = ProofSession(lhs.outputs, rhs.outputs, reg, axioms)
if not sess.verify():
    for fo in sess.failed_obligations:
        print(fo.summary())           # lhs/rhs/free vars/Z3 model/hint
    # Either add a known-true extensional axiom and re-verify:
    sess.add_axiom_and_reverify(my_axiom)
    # ...or record a concrete counterexample:
    sess.falsify(fo.obligation_id, witness, reason)
```

---

## Extending the Registry / Axioms

* **New opcode**: `reg.register(OpSpec("dialect.foo", OpTrait.AC | OpTrait.PURE, identity=..., inverse=...))`.
* **New Z3 semantics**: `reg.alias_z3("dialect.foo", lambda zargs, attrs, ty, b: ...)`.
* **New sound rewrite**: `axioms.add(ExtensionalAxiom("foo_roundtrip", "...", rewrite_fn))`.
* **Fast-math**: pass `fp_fast_math=True` to `build_gpu_registry` /
  `build_gpu_axiom_store` to enable AC/inverses on FP add/mul and the
  `Fma(a,b,c) -> FAdd(FMul(a,b),c)` axiom.

---

## Dependencies

Only `z3-solver` (already in `pyproject.toml`). No C++ build, no `spirv-tools`
required for Options A/C; Option A's SPIR-V route and `sweep_samples.py` use the
system `spirv-dis` and the prebuilt `valhall2vulkan/build/spirv2valhall`.