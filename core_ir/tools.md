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
| `cli_spirv.py` | **CLI**: 2-way equivalence checking for a `.spv` shader (`python -m core_ir.cli_spirv shader.spv --pair direct:ddk`). |
| `cli_simp.py` | **CLI**: equivalence checking for a `.simp` program across passes (`python -m core_ir.cli_simp examples/demo.simp`). |
| `cli_common.py` | Shared result/printing/JSON helpers used by both CLIs. |
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

## Quick CLI Reference

```bash
# SPIR-V / ValhallSSA: 2-way equivalence (default pair direct:ddk)
.venv/bin/python -m core_ir.cli_spirv shader.spv --pair direct:panfrost

# .simp: original vs optimized (default passes const_fold,cse,dce)
.venv/bin/python -m core_ir.cli_simp examples/demo.simp
```

---

## Equivalence Checking for SPIR-V / SSA-IR

There are three supported frontends, all producing the same `CoreShaderProgram`
(or `CoreFunc`) that `CoreSSAAntiUnifier` / `ProofSession` consume:

1. **ValhallSSA** (three routes: `direct`, `ddk`, `panfrost`) via
   `lower_valhall_ssa_to_core`.
2. **SPIR-V disassembly** via `lower_spirv_dis_to_core` (uses system `spirv-dis`).
3. **Unstructured SSA-CFG** via `lower_cfg_to_core` (hand-built or from any SSA CFG).

### Option A — `core_ir.cli_spirv` (2-way CLI, recommended)

The CLI lowers one `.spv` through the requested routes and proves a single
ordered pair equivalent. Pick the pair with `--pair LHS:RHS` (or `--lhs`/`--rhs`).
Routes: `direct`, `ddk`, `panfrost` (ValhallSSA via
`valhall2vulkan/build/spirv2valhall`) and `spirv` (system `spirv-dis`).

```bash
# default pair: direct vs ddk
.venv/bin/python -m core_ir.cli_spirv shader.spv

# choose the pair explicitly
.venv/bin/python -m core_ir.cli_spirv shader.spv --pair direct:panfrost
.venv/bin/python -m core_ir.cli_spirv shader.spv --lhs ddk --rhs spirv

# machine-readable output + per-obligation detail
.venv/bin/python -m core_ir.cli_spirv shader.spv --pair direct:spirv --json out.json -v
```

Flags of note: `--s2v` / `--mali` / `--pan` override the default
`valhall2vulkan` tool paths; `--strict-fp` disables fast-math FP axioms;
`--z3-timeout` sets the per-channel Z3 budget. Exit code is `0` iff the pair
is proved, `1` on a counterexample/lowering failure, `2` on bad arguments.

Example run against `samples/comp_rep_minimal.spv`:

```
  [PROVED] direct:ddk                   |     0.00 ms | z3=0 struct=0 loops=0
------------------------------------------------------------------------------
  total=1  proved=1  failed=0  error=0
```

### Option B — Batch sweep over a corpus

`sweep_samples.py` automates pairwise checking over every `*.spv` in a directory
(plus optional extras), verifying `direct_vs_ddk`, `direct_vs_panfrost`, and
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

## Equivalence Checking for `.simp` Programs — `core_ir.cli_simp`

`core_ir.cli_simp` parses and typechecks a `.simp` file, applies a selectable
set of optimization passes to produce the RHS, lowers both sides to CoreSSA-IR
with `SimpToCoreLowerer`, and proves every function (or a single `--entry`)
equivalent with `CoreSSAAntiUnifier`.

```bash
# default: original vs const_fold+cse+dce, all functions
.venv/bin/python -m core_ir.cli_simp examples/demo.simp

# identity check (no passes), single function
.venv/bin/python -m core_ir.cli_simp examples/demo.simp --passes none -e main

# pick passes, JSON output, per-obligation detail
.venv/bin/python -m core_ir.cli_simp examples/demo.simp --passes const_fold,dce --json out.json -v
```

Available passes: `const_fold`, `cse`, `mem_licm`, `dce` (or `all` / `none`).
Exit code is `0` iff every checked function is proved. Example:

```
  [PROVED] add [const_fold+cse+dce]     |     0.15 ms | z3=0 struct=0 loops=0
  [PROVED] main [const_fold+cse+dce]    |     4.64 ms | z3=1 struct=0 loops=0
------------------------------------------------------------------------------
  total=2  proved=2  failed=0  error=0
```

(The `main` proof discharges one residual `mul(x, 2.0) ~ add(x, x)` leaf in Z3;
`add` is fully structural.)

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