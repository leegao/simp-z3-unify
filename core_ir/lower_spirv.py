import re

from .core_types import BOOL, BV32, BV64, FLOAT, CoreType, TArray, TVector, VOID
from .ir import ConstVal, CoreVal, OpApp, ParamVal, SelectVal, SymVal
from .lower_valhall_ssa import CoreShaderProgram, ShaderExecEnv


_SPV_OP_MAP: dict[str, str] = {
    "OpIAdd": "bv.add",
    "OpISub": "bv.sub",
    "OpIMul": "bv.mul",
    "OpSNegate": "bv.neg",
    "OpSDiv": "bv.sdiv",
    "OpUDiv": "bv.udiv",
    "OpSRem": "bv.srem",
    "OpUMod": "bv.umod",
    "OpBitwiseAnd": "bv.and",
    "OpBitwiseOr": "bv.or",
    "OpBitwiseXor": "bv.xor",
    "OpNot": "bv.not",
    "OpShiftLeftLogical": "bv.shl",
    "OpShiftRightLogical": "bv.shru",
    "OpShiftRightArithmetic": "bv.shrs",
    "OpIEqual": "bv.cmp_eq",
    "OpINotEqual": "bv.cmp_ne",
    "OpSLessThan": "bv.cmp_slt",
    "OpSLessThanEqual": "bv.cmp_sle",
    "OpSGreaterThan": "bv.cmp_sgt",
    "OpSGreaterThanEqual": "bv.cmp_sge",
    "OpULessThan": "bv.cmp_ult",
    "OpULessThanEqual": "bv.cmp_ule",
    "OpUGreaterThan": "bv.cmp_ugt",
    "OpUGreaterThanEqual": "bv.cmp_uge",
    "OpLogicalAnd": "and",
    "OpLogicalOr": "or",
    "OpLogicalNot": "not",
    "OpFAdd": "f.add",
    "OpFSub": "f.sub",
    "OpFMul": "f.mul",
    "OpFDiv": "f.div",
    "OpFNegate": "f.neg",
    "OpFOrdEqual": "f.cmp_oeq",
    "OpFOrdNotEqual": "f.cmp_one",
    "OpFUnordNotEqual": "f.cmp_une",
    "OpFOrdLessThan": "f.cmp_olt",
    "OpFOrdLessThanEqual": "f.cmp_ole",
    "OpFOrdGreaterThan": "f.cmp_ogt",
    "OpFOrdGreaterThanEqual": "f.cmp_oge",
    "OpBitcast": "vh.bitcast",
    "OpCopyObject": "spv.copy",
}


def lower_spirv_dis_to_core(spv_text: str, name: str = "spv_shader") -> CoreShaderProgram:
    """Bonus adapter: lower textual SPIR-V disassembly (`spirv-dis` output) into CoreSSA-IR.

    Supports types, constants, DescriptorSet/Binding decorations, SSBO/buffer AccessChain
    address calculation, Load/Store into Memory-SSA, scalar/vector ALU, CompositeConstruct/
    CompositeExtract, Select, and straight-line/entry-point outputs.
    """
    env = ShaderExecEnv(route="spirv_dis")
    types: dict[str, CoreType] = {}
    consts: dict[str, CoreVal] = {}
    desc_sets: dict[str, int] = {}
    bindings: dict[str, int] = {}
    array_strides: dict[str, int] = {}
    member_offsets: dict[tuple[str, int], int] = {}
    var_bases: dict[str, CoreVal] = {}
    vals: dict[str, CoreVal] = {}
    ptr_exprs: dict[str, CoreVal] = {}
    # Output variables (OpVariable with a StorageClass other than the private
    # Function storage class), keyed by result id. `OpStore` into one of these
    # becomes an observable output channel instead of a Memory-SSA heap write.
    output_vars: dict[str, tuple[str, str]] = {}
    locations: dict[str, int] = {}

    heap_ty = TArray(BV64, BV32)
    heap: CoreVal = ParamVal("__heap0", heap_ty)
    outputs: dict[str, CoreVal] = {}
    unsupported: list[str] = []
    in_func = False

    for raw in spv_text.splitlines():
        line = raw.strip()
        if not line or line.startswith(";"):
            continue

        # Entry point & decorations
        if line.startswith("OpEntryPoint "):
            parts = line.split()
            if len(parts) >= 2:
                env.stage = parts[1].lower()
            continue

        if line.startswith("OpDecorate "):
            parts = line.split()
            if len(parts) >= 4:
                tgt, dec, val = parts[1], parts[2], parts[3]
                if dec == "DescriptorSet":
                    desc_sets[tgt] = int(val)
                elif dec == "Binding":
                    bindings[tgt] = int(val)
                elif dec == "ArrayStride":
                    array_strides[tgt] = int(val)
                elif dec == "Location":
                    locations[tgt] = int(val)
            continue

        if line.startswith("OpMemberDecorate "):
            parts = line.split()
            if len(parts) >= 5 and parts[3] == "Offset":
                member_offsets[(parts[1], int(parts[2]))] = int(parts[4])
            continue

        if " = OpFunction " in line or line.startswith("OpFunction "):
            in_func = True
            continue
        if line.startswith("OpFunctionEnd"):
            in_func = False
            continue

        # Assignment form: %res = OpName args...
        m = re.match(r"^(%\S+)\s*=\s*(\w+)\s*(.*)$", line)
        if m:
            res_id, op, rest = m.group(1), m.group(2), m.group(3).strip()
            toks = rest.split() if rest else []

            if op == "OpTypeBool":
                types[res_id] = BOOL
            elif op == "OpTypeInt":
                w = int(toks[0]) if toks else 32
                types[res_id] = BV64 if w == 64 else BV32
            elif op == "OpTypeFloat":
                types[res_id] = FLOAT
            elif op == "OpTypeVoid":
                types[res_id] = VOID
            elif op == "OpTypeVector" and len(toks) >= 2:
                types[res_id] = TVector(types.get(toks[0], FLOAT), int(toks[1]))
            elif op == "OpConstant" and len(toks) >= 2:
                ty = types.get(toks[0], BV32)
                raw_v = toks[1]
                if ty == FLOAT:
                    consts[res_id] = ConstVal(float(raw_v), FLOAT)
                else:
                    iv = int(raw_v, 0)
                    consts[res_id] = ConstVal(iv & (0xFFFFFFFFFFFFFFFF if ty == BV64 else 0xFFFFFFFF), ty)
                vals[res_id] = consts[res_id]
            elif op == "OpConstantComposite" and len(toks) >= 2:
                ty = types.get(toks[0], BV32)
                cargs = tuple(vals.get(o, SymVal(o, ty)) for o in toks[1:])
                if isinstance(ty, TVector):
                    vop = {2: "vh.vec2", 3: "vh.vec3", 4: "vh.vec4"}.get(len(cargs), "vh.vec4")
                    consts[res_id] = OpApp(vop, cargs, ty)
                else:
                    consts[res_id] = OpApp("spv.const_composite", cargs, ty)
                vals[res_id] = consts[res_id]
            elif op == "OpConstantTrue":
                consts[res_id] = ConstVal(True, BOOL)
                vals[res_id] = consts[res_id]
            elif op == "OpConstantFalse":
                consts[res_id] = ConstVal(False, BOOL)
                vals[res_id] = consts[res_id]
            elif op == "OpVariable":
                # %res = OpVariable %ptr StorageClass [initializer]
                storage = toks[1] if len(toks) >= 2 else "Function"
                if storage in ("Output", "Input"):
                    output_vars[res_id] = (storage, res_id)
                dset = desc_sets.get(res_id, 0)
                bind = bindings.get(res_id)
                if bind is not None:
                    base = SymVal(f"sb64:ssbo:{dset}:{bind}", BV64)
                    var_bases[res_id] = base
                    ptr_exprs[res_id] = base
                else:
                    base = SymVal(f"spv_var:{res_id}", BV64)
                    var_bases[res_id] = base
                    ptr_exprs[res_id] = base
            elif in_func:
                ty = types.get(toks[0], BV32) if toks else BV32
                operands = toks[1:] if len(toks) > 1 else []

                if op == "OpAccessChain" and operands:
                    base_id = operands[0]
                    cur_addr = ptr_exprs.get(base_id, SymVal(f"spv_ptr:{base_id}", BV64))
                    # Sum constant byte offsets (stride=4 default for runtime uint/float arrays)
                    for idx_id in operands[2:]:
                        idx_v = vals.get(idx_id)
                        if isinstance(idx_v, ConstVal) and isinstance(idx_v.value, int):
                            byte_off = idx_v.value * 4
                            if byte_off != 0:
                                cur_addr = OpApp("bv64.add", (cur_addr, ConstVal(byte_off, BV64)), BV64)
                        elif idx_v is not None:
                            ext = OpApp("vh.uconv32_64", (idx_v,), BV64)
                            scaled = OpApp("bv64.shl", (ext, ConstVal(2, BV64)), BV64)
                            cur_addr = OpApp("bv64.add", (cur_addr, scaled), BV64)
                    ptr_exprs[res_id] = cur_addr
                    vals[res_id] = cur_addr
                elif op == "OpLoad" and operands:
                    addr = ptr_exprs.get(operands[0], vals.get(operands[0], SymVal(operands[0], BV64)))
                    vals[res_id] = OpApp("arr.load", (heap, addr), ty)
                elif op == "OpSelect" and len(operands) == 3:
                    vals[res_id] = SelectVal(
                        vals.get(operands[0], SymVal(operands[0], BOOL)),
                        vals.get(operands[1], SymVal(operands[1], ty)),
                        vals.get(operands[2], SymVal(operands[2], ty)),
                        ty,
                    )
                elif op == "OpCompositeConstruct":
                    cargs = tuple(vals.get(o, SymVal(o, FLOAT)) for o in operands)
                    vop = {2: "vh.vec2", 3: "vh.vec3", 4: "vh.vec4"}.get(len(cargs), "vh.vec4")
                    vals[res_id] = OpApp(vop, cargs, ty)
                elif op == "OpCompositeExtract" and len(operands) >= 2:
                    src = vals.get(operands[0], SymVal(operands[0], TVector(ty, 4)))
                    vals[res_id] = OpApp("vh.vec_extract", (src,), ty, attrs=(("index", int(operands[1])),))
                elif op in _SPV_OP_MAP:
                    cargs = tuple(vals.get(o, SymVal(o, ty)) for o in operands)
                    vals[res_id] = OpApp(_SPV_OP_MAP[op], cargs, ty)
                else:
                    if op not in unsupported:
                        unsupported.append(op)
                    cargs = tuple(vals.get(o, SymVal(o, ty)) for o in operands if o.startswith("%"))
                    vals[res_id] = OpApp(f"spv.{op.lower()}", cargs, ty)
            continue

        # Non-assignment instructions inside function
        if in_func:
            if line.startswith("OpStore "):
                toks = line.split()
                if len(toks) >= 3:
                    tgt = toks[1]
                    val = vals.get(toks[2], SymVal(toks[2], BV32))
                    if tgt in output_vars:
                        storage, var_id = output_vars[tgt]
                        if storage == "Output":
                            loc = locations.get(var_id)
                            chan = f"frag_color:{loc}" if loc is not None else f"frag_color:{var_id}"
                            outputs[chan] = val
                        else:
                            outputs[f"input:{var_id}"] = val
                    else:
                        addr = ptr_exprs.get(tgt, vals.get(tgt, SymVal(tgt, BV64)))
                        heap = OpApp("arr.store", (heap, addr, val), heap_ty)
                        outputs["__heap"] = heap
            elif line.startswith("OpKill") or line.startswith("OpDemoteToHelperInvocation"):
                outputs["__alive"] = ConstVal(False, BOOL)

    return CoreShaderProgram(
        name=name,
        route="spirv_dis",
        outputs=outputs,
        env=env,
        num_blocks=1,
        num_nodes=len(vals),
        unsupported_ops=unsupported,
    )
