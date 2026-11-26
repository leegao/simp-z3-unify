import unittest

from core_ir.axioms import (
    AxiomStore,
    ExtensionalAxiom,
    ProofSession,
    build_gpu_axiom_store,
)
from core_ir.core_types import BOOL, BV32, BV64, FLOAT, TVector
from core_ir.equational_au import CoreSSAAntiUnifier
from core_ir.gpu_dialect import build_gpu_registry
from core_ir.ir import ConstVal, OpApp, ParamVal, SymVal
from core_ir.lower_spirv import lower_spirv_dis_to_core
from core_ir.lower_valhall_ssa import lower_valhall_ssa_to_core


_SAMPLE_DIRECT = """
; ValhallSSA (1 blocks, 23 nodes)
bb0:
  %1 : Token = Nop()
  %3 : U64 = ConstU64()
  %4 : U64 = LoadSidebandU64() imm32=0x11 "ssbo:0:1"
  %6 : U32 = ConstU32() imm32=0x3e8
  %7 : U64 = IAdd64(%4, %3)
  %8 : Token = PsbStore(%7, %6, %1) imm32=0x4 "sym:ssbo:0:1[stride=0,off=0]"
  %9 : U64 = ConstU64() imm64=0x4
  %11 : U32 = ConstU32() imm32=0xfffffc18
  %12 : U64 = IAdd64(%4, %9)
  %13 : Token = PsbStore(%12, %11, %8) imm32=0x4 aux_idx=4 "sym:ssbo:0:1[stride=0,off=4]"
  return
"""

_SAMPLE_DDK = """
; ValhallSSA (1 blocks, 20 nodes)
bb0:
  %2 : Token = Nop()
  %33 : U64 = LoadSidebandU64() imm32=0x10 "ssbo:0:1"
  %18 : U32 = ConstU32() imm32=0xfffffc18
  %19 : U32 = ConstU32() imm32=0x3e8
  %23 : Token = PsbStore(%33, %19, %2) imm32=0x4 "sym:ssbo:0:1[stride=0,off=0]"
  %24 : U64 = ConstU64() imm64=0x4
  %25 : U64 = IAdd64(%33, %24)
  %26 : Token = PsbStore(%25, %18, %23) imm32=0x4 aux_idx=4 "sym:ssbo:0:1[stride=0,off=4]"
  return
"""

_SAMPLE_PANFROST = """
; spirv2valhall --panfrost --dump-ssa arch=v10 kernel="main_0_default"
; ValhallSSA (1 blocks, 24 nodes)
bb0:
  %34 : U64 = LoadSidebandU64() imm32=0x10 "ssbo:0:1"
  %35 : TupleU32x2 = Split64(%34)
  %36 : U32 = TupleExtract(%35)
  %37 : U32 = TupleExtract(%35) imm32=0x1
  %2 : Token = Nop()
  %17 : U32 = ConstU32() imm32=0x3e8
  %18 : U32 = ConstU32() imm32=0xfffffc18
  %23 : U64 = Pack64(%36, %37)
  %24 : Token = PsbStore(%23, %17, %2) imm32=0x4
  %25 : U64 = ConstU64() imm64=0x4
  %26 : U64 = IAdd64(%23, %25)
  %27 : Token = PsbStore(%26, %18, %24) imm32=0x4
  return
"""

_SAMPLE_SPIRV_DIS = """
; SPIR-V
               OpEntryPoint GLCompute %main "main"
               OpDecorate %buf DescriptorSet 0
               OpDecorate %buf Binding 1
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
     %uint_1 = OpConstant %uint 1
  %uint_1000 = OpConstant %uint 1000
 %uint_neg1k = OpConstant %uint 4294966296
        %buf = OpVariable %_ptr StorageBuffer
       %main = OpFunction %void None %fn_ty
         %p0 = OpAccessChain %_ptr_u %buf %uint_0 %uint_0
               OpStore %p0 %uint_1000
         %p1 = OpAccessChain %_ptr_u %buf %uint_0 %uint_1
               OpStore %p1 %uint_neg1k
               OpReturn
               OpFunctionEnd
"""


class TestGpuCoreIR(unittest.TestCase):
    def setUp(self):
        self.reg = build_gpu_registry(fp_fast_math=False)
        self.axioms = build_gpu_axiom_store(include_fast_math=False)

    def test_01_direct_vs_ddk_vs_panfrost_vs_spirv_four_way(self):
        p_dir = lower_valhall_ssa_to_core(_SAMPLE_DIRECT, "comp", "direct")
        p_ddk = lower_valhall_ssa_to_core(_SAMPLE_DDK, "comp", "ddk")
        p_pan = lower_valhall_ssa_to_core(_SAMPLE_PANFROST, "comp", "panfrost")
        p_spv = lower_spirv_dis_to_core(_SAMPLE_SPIRV_DIS, "comp")

        for label, rhs in (("ddk", p_ddk), ("panfrost", p_pan), ("spirv_dis", p_spv)):
            sess = ProofSession(p_dir.outputs, rhs.outputs, self.reg, self.axioms)
            ok = sess.verify()
            self.assertTrue(ok, f"Direct vs {label} failed: {[o.summary() for o in sess.failed_obligations]}")

    def test_02_concrete_constant_folding_and_lane_axioms(self):
        # BfeU(0x1f1e1d1c, 0x10, 0x8) & 0x1f == 0x1e (real Panfrost bitfield constant!)
        bfe = OpApp(
            "vh.bfe_u",
            (ConstVal(0x1F1E1D1C, BV32), ConstVal(0x10, BV32), ConstVal(0x8, BV32)),
            BV32,
        )
        masked = OpApp("bv.and", (bfe, ConstVal(0x1F, BV32)), BV32)
        au = CoreSSAAntiUnifier(self.reg, axioms=self.axioms)
        res = au.verify_values(masked, ConstVal(0x1E, BV32))
        self.assertTrue(res.equivalent)
        self.assertEqual(res.z3_leaf_calls, 0)

        # SubgroupShuffle(Const(7), lane_id) == Const(7)
        lane = ParamVal("lane_id", BV32)
        shuf = OpApp("vh.subgroup_shuffle", (ConstVal(7, BV32), lane), BV32)
        res_shuf = CoreSSAAntiUnifier(self.reg, axioms=self.axioms).verify_values(shuf, ConstVal(7, BV32))
        self.assertTrue(res_shuf.equivalent)
        self.assertEqual(res_shuf.z3_leaf_calls, 0)

        # PackHalf2x16(UnpackHalf2x16(x)) == x
        x = ParamVal("x", BV32)
        rt = OpApp("vh.pack_half2x16", (OpApp("vh.unpack_half2x16", (x,), TVector(FLOAT, 2)),), BV32)
        res_hp = CoreSSAAntiUnifier(self.reg, axioms=self.axioms).verify_values(rt, x)
        self.assertTrue(res_hp.equivalent)
        self.assertEqual(res_hp.z3_leaf_calls, 0)

    def test_03_operator_inductive_axiom_introduction_and_falsification(self):
        # Operator-in-the-loop: custom opaque Valhall op `vh.custom_norm(x)` vs `x`
        # Without the extensional axiom, proof fails and produces a FailedObligation;
        # introducing the known-true axiom inductively closes the proof!
        x = ParamVal("x", FLOAT)
        lhs = {"out0": OpApp("vh.fma", (x, ConstVal(1.0, FLOAT), ConstVal(0.0, FLOAT)), FLOAT)}
        rhs = {"out0": x}

        # Use strict registry where vh.fma is NOT folded without fast-math axiom
        strict_reg = build_gpu_registry(fp_fast_math=False)
        strict_reg.alias_z3(
            "vh.fma",
            lambda zargs, _a, ty, b: b._get_uf("strict_fma", [a.sort() for a in zargs], b.sort_of(ty))(*zargs),
        )
        empty_axioms = AxiomStore()
        sess = ProofSession(lhs, rhs, strict_reg, empty_axioms)
        self.assertFalse(sess.verify())
        self.assertEqual(len(sess.failed_obligations), 1)

        # Operator inspects failed obligation and inductively adds extensional axiom: Fma(a, 1.0, 0.0) -> a
        def rw_fma_id(v, _reg):
            if isinstance(v, OpApp) and v.op == "vh.fma" and len(v.args) == 3:
                if isinstance(v.args[1], ConstVal) and v.args[1].value == 1.0 and isinstance(v.args[2], ConstVal) and v.args[2].value == 0.0:
                    return v.args[0]
            return None

        ok_after = sess.add_axiom_and_reverify(
            ExtensionalAxiom("fma_mul1_add0", "Fma(a, 1.0, 0.0) == a (ignoring -0.0/NaN)", rw_fma_id, dialect="valhall")
        )
        self.assertTrue(ok_after)
        self.assertEqual(len(sess.failed_obligations), 0)

        # Falsification path on a genuine bug: 0x3e8 (1000) vs 0x3e9 (1001)
        bug_ddk = _SAMPLE_DDK.replace("0x3e8", "0x3e9")
        p_dir = lower_valhall_ssa_to_core(_SAMPLE_DIRECT, "bug", "direct")
        p_bug = lower_valhall_ssa_to_core(bug_ddk, "bug", "ddk")
        sess_bug = ProofSession(p_dir.outputs, p_bug.outputs, self.reg, self.axioms)
        self.assertFalse(sess_bug.verify())
        self.assertEqual(len(sess_bug.failed_obligations), 1)
        rec = sess_bug.falsify(
            sess_bug.failed_obligations[0].obligation_id,
            {"ssbo:0:1[0]": (0x3E8, 0x3E9)},
            "Store at offset 0 writes 1000 in Direct vs 1001 in mutated DDK",
        )
        self.assertEqual(rec.channel, "__heap")


if __name__ == "__main__":
    unittest.main()
