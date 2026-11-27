from dataclasses import dataclass, field
import re
import struct

from .core_types import BOOL, BV32, BV64, FLOAT, TOKEN, CoreType, TArray, TTuple, TVector, VOID
from .ir import ConstVal, CoreVal, OpApp, ParamVal, SelectVal, SymVal


@dataclass
class ShaderExecEnv:
    """Execution environment metadata parsed from CSF / MBS2 / Panfrost headers."""

    route: str = "unknown"  # "direct" | "ddk" | "panfrost"
    stage: str = "compute"
    entry: str = "main"
    wg_size: tuple[int, int, int] = (1, 1, 1)
    fau_map: dict[int, str] = field(default_factory=dict)  # fau_word_idx -> symbolic binding name
    ssbo_map: dict[str, tuple[int, int]] = field(default_factory=dict)  # name -> (set, binding)


@dataclass
class ParsedSsaNode:
    id: int
    ty_name: str
    op: str
    args: list[int] = field(default_factory=list)
    phi_incomings: list[tuple[int, int]] = field(default_factory=list)  # (pred_bb_id, ssa_id)
    imm64: int = 0
    imm32: int = 0
    imm32_2: int = 0
    aux_idx: int = 0
    aux_str: str = ""


@dataclass
class ParsedSsaBlock:
    id: int
    preds: list[int] = field(default_factory=list)
    nodes: list[ParsedSsaNode] = field(default_factory=list)
    term_kind: str = "return"  # return | discard | br | cond_br | fallthrough
    term_cond: int = 0
    term_true: int = -1
    term_false: int = -1


@dataclass
class ParsedSsaModule:
    blocks: dict[int, ParsedSsaBlock]
    nodes: dict[int, ParsedSsaNode]
    env: ShaderExecEnv


@dataclass
class CoreShaderProgram:
    """Observable semantic outputs of a GPU shader lowered into CoreSSA-IR."""

    name: str
    route: str
    outputs: dict[str, CoreVal]
    env: ShaderExecEnv
    num_blocks: int = 0
    num_nodes: int = 0
    unsupported_ops: list[str] = field(default_factory=list)


_VAL_TYPE_MAP: dict[str, CoreType] = {
    "Void": VOID,
    "Bool": BOOL,
    "U32": BV32,
    "S32": BV32,
    "F32": FLOAT,
    "U64": BV64,
    "S64": BV64,
    "V2F32": TVector(FLOAT, 2),
    "V2U32": TVector(BV32, 2),
    "V2S32": TVector(BV32, 2),
    "V3U32": TVector(BV32, 3),
    "V4F32": TVector(FLOAT, 4),
    "V4U32": TVector(BV32, 4),
    "V4S32": TVector(BV32, 4),
    "V2Bool": TVector(BOOL, 2),
    "TupleU32x2": TTuple((BV32, BV32)),
    "TupleValToken": TTuple((BV32, TOKEN)),
    "Token": TOKEN,
}

# Mapping from all 155 ValhallSSA SsaOp names -> CoreSSA-IR canonical dialect op names
_SSA_OP_TO_CORE: dict[str, str] = {
    "Copy": "vh.copy",
    "Nop": "vh.nop",
    "IAdd": "bv.add",
    "ISub": "bv.sub",
    "IMul": "bv.mul",
    "SNeg": "bv.neg",
    "SAbs": "bv.sabs",
    "SDiv": "bv.sdiv",
    "UDiv": "bv.udiv",
    "SRem": "bv.srem",
    "SMod": "bv.smod",
    "UMod": "bv.umod",
    "IAdd64": "bv64.add",
    "ISub64": "bv64.sub",
    "IMul64": "bv64.mul",
    "BitAnd": "bv.and",
    "BitOr": "bv.or",
    "BitXor": "bv.xor",
    "BitNot": "bv.not",
    "BitAnd64": "bv64.and",
    "BitOr64": "bv64.or",
    "BitXor64": "bv64.xor",
    "BitNot64": "bv64.not",
    "Shl": "bv.shl",
    "ShrU": "bv.shru",
    "ShrS": "bv.shrs",
    "Shl64": "bv64.shl",
    "ShrU64": "bv64.shru",
    "ShrS64": "bv64.shrs",
    "RotL": "bv.rotl",
    "RotR": "bv.rotr",
    "BfeU": "vh.bfe_u",
    "BfeS": "vh.bfe_s",
    "Bfi": "vh.bfi",
    "BitCount": "vh.bit_count",
    "BitReverse": "vh.bit_reverse",
    "Clz": "vh.clz",
    "Pack64": "vh.pack64",
    "Split64": "vh.split64",
    "TupleExtract": "vh.tuple_extract",
    "UConvert32To64": "vh.uconv32_64",
    "SConvert32To64": "vh.sconv32_64",
    "UConvert64To32": "vh.uconv64_32",
    "Bitcast": "vh.bitcast",
    "ICmpEq": "bv.cmp_eq",
    "ICmpNe": "bv.cmp_ne",
    "ICmpSLt": "bv.cmp_slt",
    "ICmpSLe": "bv.cmp_sle",
    "ICmpSGt": "bv.cmp_sgt",
    "ICmpSGe": "bv.cmp_sge",
    "ICmpULt": "bv.cmp_ult",
    "ICmpULe": "bv.cmp_ule",
    "ICmpUGt": "bv.cmp_ugt",
    "ICmpUGe": "bv.cmp_uge",
    "LogicalAnd": "and",
    "LogicalOr": "or",
    "LogicalNot": "not",
    "FAdd": "f.add",
    "FSub": "f.sub",
    "FMul": "f.mul",
    "FDiv": "f.div",
    "FNeg": "f.neg",
    "FAbs": "vh.fabs",
    "Fma": "vh.fma",
    "FClamp": "vh.fclamp",
    "FMin": "vh.fmin",
    "FMax": "vh.fmax",
    "NMin": "vh.nmin",
    "NMax": "vh.nmax",
    "FCmpOrdEq": "f.cmp_oeq",
    "FCmpOrdNe": "f.cmp_one",
    "FCmpUnordNe": "f.cmp_une",
    "FCmpOrdLt": "f.cmp_olt",
    "FCmpOrdLe": "f.cmp_ole",
    "FCmpOrdGt": "f.cmp_ogt",
    "FCmpOrdGe": "f.cmp_oge",
    "FCmpTotal": "vh.fcmp_total",
    "VecConstruct2": "vh.vec2",
    "VecConstruct3": "vh.vec3",
    "VecConstruct4": "vh.vec4",
    "VecExtract": "vh.vec_extract",
    "VecShuffle": "vh.vec_shuffle",
    "PackHalf2x16": "vh.pack_half2x16",
    "UnpackHalf2x16": "vh.unpack_half2x16",
    "SubgroupBallot": "vh.subgroup_ballot",
    "SubgroupShuffle": "vh.subgroup_shuffle",
    "DPdx": "vh.dpdx",
    "DPdy": "vh.dpdy",
    "Fwidth": "vh.fwidth",
    "Acos": "vh.acos",
    "Asin": "vh.asin",
    "Atan": "vh.atan",
    "Atan2": "vh.atan2",
    "Ceil": "vh.ceil",
    "ConvertFToS": "vh.conv_ftos",
    "ConvertFToU": "vh.conv_ftou",
    "ConvertSToF": "vh.conv_stof",
    "ConvertUToF": "vh.conv_utof",
    "Cos": "vh.cos",
    "Cosh": "vh.cosh",
    "Exp": "vh.exp",
    "Exp2": "vh.exp2",
    "Floor": "vh.floor",
    "FrexpExp": "vh.frexp_exp",
    "FrexpSignificand": "vh.frexp_sig",
    "IDot4S": "vh.idot4s",
    "IDot4SU": "vh.idot4su",
    "IDot4U": "vh.idot4u",
    "ImageQueryLod": "vh.img_query_lod",
    "ImageQuerySize": "vh.img_query_size",
    "ImageTexelAddr": "vh.img_texel_addr",
    "InverseSqrt": "vh.inv_sqrt",
    "IsInf": "vh.isinf",
    "IsNan": "vh.isnan",
    "Ldexp": "vh.ldexp",
    "Log": "vh.log",
    "Log2": "vh.log2",
    "Pow": "vh.pow",
    "RoundEven": "vh.round_even",
    "Sin": "vh.sin",
    "Sinh": "vh.sinh",
    "Sqrt": "vh.sqrt",
    "Tanh": "vh.tanh",
    "Trunc": "vh.trunc",
    "TexFetch": "vh.tex_fetch",
    "TexGather": "vh.tex_gather",
    "TexSample": "vh.tex_sample",
    "TexSampleDref": "vh.tex_sample_dref",
    "TexSampleLod": "vh.tex_sample_lod",
    "ControlBarrier": "vh.control_barrier",
    "AtomicRmw": "vh.atomic_rmw",
    "AtomicExchange": "vh.atomic_xchg",
    "AtomicCmpXchg": "vh.atomic_cmpxchg",
    "Kill": "vh.kill",
}

_BB_RE = re.compile(r"^bb(\d+):(?:\s*;\s*preds=\[(.*?)\])?")
_NODE_RE = re.compile(r"^\s*%(\d+)\s*:\s*(\w+)\s*=\s*(\w+)\(([^)]*)\)(.*)$")
_FAU_RE = re.compile(r"FAU\+0x([0-9a-fA-F]+)\s+\([^)]+\)\s+\[[^\]]+\]\s+(.+)$")


def parse_valhall_ssa_text(text: str, route: str = "unknown") -> ParsedSsaModule:
    """Parse textual ValhallSSA assembly (and optional CSF Execution Context headers)."""
    env = ShaderExecEnv(route=route)
    blocks: dict[int, ParsedSsaBlock] = {}
    nodes: dict[int, ParsedSsaNode] = {}
    cur_blk: ParsedSsaBlock | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # Parse CSF Execution Context header metadata if present
        if line.startswith(";"):
            if "--panfrost" in line:
                env.route = "panfrost"
            m_fau = _FAU_RE.search(line)
            if m_fau:
                byte_off = int(m_fau.group(1), 16)
                env.fau_map[byte_off // 4] = m_fau.group(2).strip()
            if "FRAGMENT" in line or "CFRG" in line:
                env.stage = "fragment"
            elif "VERTEX" in line or "CVRT" in line:
                env.stage = "vertex"
            continue

        m_bb = _BB_RE.match(line)
        if m_bb:
            bid = int(m_bb.group(1))
            preds: list[int] = []
            if m_bb.group(2):
                preds = [int(x.strip()[2:]) for x in m_bb.group(2).split(",") if x.strip().startswith("bb")]
            cur_blk = ParsedSsaBlock(id=bid, preds=preds)
            blocks[bid] = cur_blk
            continue

        if cur_blk is None:
            continue

        m_node = _NODE_RE.match(raw_line)
        if m_node:
            nid = int(m_node.group(1))
            ty_name = m_node.group(2)
            op = m_node.group(3)
            arg_str = m_node.group(4).strip()
            rest = m_node.group(5)

            args: list[int] = []
            phi_incomings: list[tuple[int, int]] = []
            if arg_str:
                if op == "Phi":
                    for part in arg_str.split(","):
                        pm = re.match(r"\s*bb(\d+)\s*:\s*%(\d+)\s*", part)
                        if pm:
                            phi_incomings.append((int(pm.group(1)), int(pm.group(2))))
                else:
                    for part in arg_str.split(","):
                        p = part.strip()
                        if p.startswith("%"):
                            args.append(int(p[1:]))

            imm64 = 0
            imm32 = 0
            imm32_2 = 0
            aux_idx = 0
            aux_str = ""

            m64 = re.search(r"imm64=0x([0-9a-fA-F]+)", rest)
            if m64:
                imm64 = int(m64.group(1), 16)
            m32 = re.search(r"imm32=0x([0-9a-fA-F]+)", rest)
            if m32:
                imm32 = int(m32.group(1), 16)
            m32_2 = re.search(r"imm32_2=(\d+)", rest)
            if m32_2:
                imm32_2 = int(m32_2.group(1))
            m_aux = re.search(r"aux_idx=(\d+)", rest)
            if m_aux:
                aux_idx = int(m_aux.group(1))
            m_str = re.search(r'"([^"]*)"', rest)
            if m_str:
                aux_str = m_str.group(1)

            node = ParsedSsaNode(
                id=nid,
                ty_name=ty_name,
                op=op,
                args=args,
                phi_incomings=phi_incomings,
                imm64=imm64,
                imm32=imm32,
                imm32_2=imm32_2,
                aux_idx=aux_idx,
                aux_str=aux_str,
            )
            cur_blk.nodes.append(node)
            nodes[nid] = node
            continue

        # Block terminators
        if line == "return":
            cur_blk.term_kind = "return"
        elif line == "discard":
            cur_blk.term_kind = "discard"
        elif line.startswith("br bb"):
            cur_blk.term_kind = "br"
            cur_blk.term_true = int(line.split("bb", 1)[1])
        elif line.startswith("fallthrough bb"):
            cur_blk.term_kind = "fallthrough"
            cur_blk.term_true = int(line.split("bb", 1)[1])
        elif line.startswith("cond_br "):
            mc = re.match(r"cond_br\s+%(\d+),\s*bb(\d+),\s*bb(\d+)", line)
            if mc:
                cur_blk.term_kind = "cond_br"
                cur_blk.term_cond = int(mc.group(1))
                cur_blk.term_true = int(mc.group(2))
                cur_blk.term_false = int(mc.group(3))

    return ParsedSsaModule(blocks=blocks, nodes=nodes, env=env)


def _f32_bits_to_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def _normalize_sideband_tag(aux_str: str, imm32: int) -> str:
    """Canonicalize sideband/descriptor names across Direct, DDK, and Panfrost routes."""
    if aux_str:
        # e.g., "ssbo:0:1", "ubo:0:0", "push_const"
        return aux_str.strip()
    return f"sideband:{imm32:#x}"


def lower_valhall_ssa_to_core(
    text: str,
    name: str = "shader",
    route: str = "unknown",
    max_cfg_depth: int = 16,
) -> CoreShaderProgram:
    """Lower textual ValhallSSA assembly + execution environment into CoreSSA-IR.

    Produces a `CoreShaderProgram` whose `outputs` dictionary maps each observable
    shader output channel (`__heap`, `frag_color:<loc>`, `builtin_out:<name>`,
    `vert_varying:<loc>`, `__alive`) to its CoreSSA-IR value tree.
    """
    mod = parse_valhall_ssa_text(text, route=route)
    heap_ty = TArray(BV64, BV32)
    init_heap: CoreVal = ParamVal("__heap0", heap_ty)
    unsupported: list[str] = []

    def lower_node(
        node: ParsedSsaNode,
        vals: dict[int, CoreVal],
        heap: CoreVal,
        outputs: dict[str, CoreVal],
        alive: CoreVal,
        pred_bb: int | None,
    ) -> tuple[CoreVal, CoreVal, CoreVal]:
        ty = _VAL_TYPE_MAP.get(node.ty_name, BV32)
        op = node.op

        if op == "Phi":
            if pred_bb is not None:
                for pbb, sid in node.phi_incomings:
                    if pbb == pred_bb and sid in vals:
                        return vals[sid], heap, alive
            for _, sid in node.phi_incomings:
                if sid in vals:
                    return vals[sid], heap, alive
            return SymVal(f"phi_undef_{node.id}", ty), heap, alive

        if op == "ConstU32" or op == "ConstS32":
            return ConstVal(node.imm32 & 0xFFFFFFFF, BV32), heap, alive
        if op == "ConstU64":
            return ConstVal(node.imm64 & 0xFFFFFFFFFFFFFFFF, BV64), heap, alive
        if op == "ConstBool":
            return ConstVal(bool(node.imm32 != 0), BOOL), heap, alive
        if op == "ConstF32":
            return ConstVal(_f32_bits_to_float(node.imm32), FLOAT), heap, alive
        if op == "Nop":
            return ParamVal("__tok0", TOKEN), heap, alive

        if op == "LoadSidebandU64":
            tag = _normalize_sideband_tag(node.aux_str, node.imm32)
            return SymVal(f"sb64:{tag}", BV64), heap, alive
        if op == "LoadSidebandU32":
            tag = _normalize_sideband_tag(node.aux_str, node.imm32)
            return SymVal(f"sb32:{tag}", BV32), heap, alive
        if op == "LoadFau":
            if node.imm32 in mod.env.fau_map:
                return SymVal(f"fau_bind:{mod.env.fau_map[node.imm32]}", BV32), heap, alive
            return SymVal(f"fau:{node.imm32}", BV32), heap, alive
        if op == "LoadBuiltinInput":
            bname = node.aux_str or f"builtin_{node.imm32:#x}"
            return SymVal(f"builtin:{bname}", ty), heap, alive
        if op == "LoadFragVarying":
            return SymVal(f"frag_var:{node.imm32}:{node.aux_idx}", ty), heap, alive
        if op == "LoadVertAttr":
            return SymVal(f"vert_attr:{node.imm32}:{node.aux_idx}", ty), heap, alive

        arg_vals = [vals.get(a, SymVal(f"undef_{a}", BV32)) for a in node.args]

        if op == "Select" and len(arg_vals) == 3:
            return SelectVal(arg_vals[0], arg_vals[1], arg_vals[2], ty), heap, alive

        if op == "PsbLoad" and len(arg_vals) >= 1:
            addr = arg_vals[0]
            loaded = OpApp("arr.load", (heap, addr), BV32)
            # PsbLoad returns TupleValToken(val, tok)
            return OpApp("tuple.mk", (loaded, heap), TTuple((BV32, heap_ty))), heap, alive

        if op == "PsbStore" and len(arg_vals) >= 2:
            addr, val = arg_vals[0], arg_vals[1]
            new_heap = OpApp("arr.store", (heap, addr, val), heap_ty)
            outputs["__heap"] = new_heap
            return new_heap, new_heap, alive

        if op == "AtomicRmw" and len(arg_vals) >= 3:
            # Valhall AtomicRmw(addr, val, token) -> observable heap effect.
            # Model as arr.store into the shared `__heap` channel so it matches
            # the SPIR-V OpAtomicX lowering.
            addr, val = arg_vals[0], arg_vals[1]
            new_heap = OpApp("arr.store", (heap, addr, val), heap_ty)
            outputs["__heap"] = new_heap
            return OpApp("tuple.mk", (val, new_heap), TTuple((ty, heap_ty))), new_heap, alive

        if op == "TupleExtract" and len(arg_vals) == 1:
            src = arg_vals[0]
            idx = node.imm32
            if isinstance(src, OpApp) and src.op == "tuple.mk" and 0 <= idx < len(src.args):
                return src.args[idx], heap, alive
            return OpApp("vh.tuple_extract", (src,), ty, attrs=(("index", idx),)), heap, alive

        if op == "VecExtract" and len(arg_vals) == 1:
            return OpApp("vh.vec_extract", (arg_vals[0],), ty, attrs=(("index", node.imm32),)), heap, alive

        if op == "StoreFragColor" and len(arg_vals) >= 1:
            loc = node.imm32
            outputs[f"frag_color:{loc}"] = arg_vals[0]
            return heap, heap, alive

        if op == "StoreBuiltinOutput" and len(arg_vals) >= 1:
            bname = node.aux_str or f"builtin_out_{node.imm32:#x}"
            outputs[f"builtin_out:{bname}"] = arg_vals[0]
            return heap, heap, alive

        if op == "StoreVertVarying" and len(arg_vals) >= 1:
            outputs[f"vert_varying:{node.imm32}"] = arg_vals[0]
            return heap, heap, alive

        if op == "Kill":
            new_alive = ConstVal(False, BOOL)
            outputs["__alive"] = new_alive
            return heap, heap, new_alive

        # General annotated OpApp
        core_op = _SSA_OP_TO_CORE.get(op)
        if core_op is None:
            core_op = f"vh.{op.lower()}"
            if op not in unsupported:
                unsupported.append(op)

        attrs_list: list[tuple[str, object]] = []
        if node.imm32 != 0 and op in ("VecShuffle", "Bfi", "TexSample", "TexFetch", "AtomicRmw"):
            attrs_list.append(("imm32", node.imm32))
        if node.aux_str and op not in ("PsbStore", "PsbLoad"):
            attrs_list.append(("tag", node.aux_str))

        return OpApp(core_op, tuple(arg_vals), ty, attrs=tuple(attrs_list)), heap, alive

    # Linear O(|B| + |V|) Reverse-Post-Order (RPO) SSA lowering (zero exponential diamond blowup!)
    if not mod.blocks:
        return CoreShaderProgram(name=name, route=mod.env.route, outputs={}, env=mod.env)

    entry_bid = min(mod.blocks.keys())
    # Compute RPO order
    visited: set[int] = set()
    postorder: list[int] = []

    def dfs_rpo(b: int) -> None:
        if b in visited or b not in mod.blocks:
            return
        visited.add(b)
        blk = mod.blocks[b]
        succs: list[int] = []
        if blk.term_kind in ("br", "fallthrough") and blk.term_true >= 0:
            succs = [blk.term_true]
        elif blk.term_kind == "cond_br":
            succs = [s for s in (blk.term_true, blk.term_false) if s >= 0]
        for s in succs:
            dfs_rpo(s)
        postorder.append(b)

    dfs_rpo(entry_bid)
    rpo = list(reversed(postorder))

    vals: dict[int, CoreVal] = {}
    blk_reach: dict[int, CoreVal] = {entry_bid: ConstVal(True, BOOL)}
    blk_out_heap: dict[int, CoreVal] = {}
    blk_out_outputs: dict[int, dict[str, CoreVal]] = {}
    blk_out_alive: dict[int, CoreVal] = {}
    edge_conds: dict[tuple[int, int], CoreVal] = {}

    def _merge_by_preds(b: int, getter, default_val: CoreVal) -> CoreVal:
        preds = [p for p in mod.blocks[b].preds if p in blk_out_heap]
        if not preds:
            return default_val
        cur = getter(preds[0])
        for p in preds[1:]:
            v = getter(p)
            if v == cur:
                continue
            ec = edge_conds.get((p, b))
            if ec is None:
                ec = blk_reach.get(p, ConstVal(True, BOOL))
            cur = SelectVal(ec, v, cur, cur.ty)
        return cur

    for bid in rpo:
        blk = mod.blocks[bid]
        if bid != entry_bid:
            in_preds = [p for p in blk.preds if p in blk_out_heap]
            if in_preds:
                rc = edge_conds.get((in_preds[0], bid), blk_reach.get(in_preds[0], ConstVal(True, BOOL)))
                for p in in_preds[1:]:
                    ep = edge_conds.get((p, bid), blk_reach.get(p, ConstVal(True, BOOL)))
                    if ep != rc:
                        rc = OpApp("or", (rc, ep), BOOL)
                blk_reach[bid] = rc
            else:
                blk_reach[bid] = ConstVal(True, BOOL)

        cur_heap = _merge_by_preds(bid, lambda p: blk_out_heap[p], init_heap)
        cur_alive = _merge_by_preds(bid, lambda p: blk_out_alive[p], ConstVal(True, BOOL))
        # Merge output channels from predecessors
        cur_out: dict[str, CoreVal] = {}
        pred_done = [p for p in blk.preds if p in blk_out_outputs]
        all_chs: set[str] = set()
        for p in pred_done:
            all_chs.update(blk_out_outputs[p].keys())
        for ch in all_chs:
            present = [p for p in pred_done if ch in blk_out_outputs[p]]
            if present:
                v0 = blk_out_outputs[present[0]][ch]
                cur_out[ch] = _merge_by_preds(bid, lambda p, c=ch, d=v0: blk_out_outputs[p].get(c, d), v0)

        # Lower nodes in block
        for node in blk.nodes:
            if node.op == "Phi":
                ty = _VAL_TYPE_MAP.get(node.ty_name, BV32)
                resolved = [(pbb, vals[sid]) for pbb, sid in node.phi_incomings if sid in vals]
                if not resolved:
                    vals[node.id] = SymVal(f"phi_loop_{node.id}", ty)
                elif len(resolved) == 1:
                    vals[node.id] = resolved[0][1]
                else:
                    cur_v = resolved[0][1]
                    for pbb, pv in resolved[1:]:
                        if pv == cur_v:
                            continue
                        ec = edge_conds.get((pbb, bid), blk_reach.get(pbb, ConstVal(True, BOOL)))
                        cur_v = SelectVal(ec, pv, cur_v, ty)
                    vals[node.id] = cur_v
                continue
            v, cur_heap, cur_alive = lower_node(node, vals, cur_heap, cur_out, cur_alive, None)
            vals[node.id] = v

        if blk.term_kind == "discard":
            cur_alive = ConstVal(False, BOOL)
            cur_out["__alive"] = cur_alive
        elif blk.term_kind == "cond_br":
            cond = vals.get(blk.term_cond, SymVal(f"cond_{blk.term_cond}", BOOL))
            r_b = blk_reach[bid]
            t_edge = cond if r_b == ConstVal(True, BOOL) else OpApp("and", (r_b, cond), BOOL)
            not_c = OpApp("not", (cond,), BOOL)
            f_edge = not_c if r_b == ConstVal(True, BOOL) else OpApp("and", (r_b, not_c), BOOL)
            if blk.term_true >= 0:
                edge_conds[(bid, blk.term_true)] = t_edge
            if blk.term_false >= 0:
                edge_conds[(bid, blk.term_false)] = f_edge
        elif blk.term_kind in ("br", "fallthrough") and blk.term_true >= 0:
            edge_conds[(bid, blk.term_true)] = blk_reach[bid]

        blk_out_heap[bid] = cur_heap
        blk_out_outputs[bid] = cur_out
        blk_out_alive[bid] = cur_alive

    # Merge outputs across all exit blocks (return / discard)
    exit_bids = [b for b in rpo if mod.blocks[b].term_kind in ("return", "discard")]
    if not exit_bids:
        exit_bids = [rpo[-1]]
    final_outputs: dict[str, CoreVal] = {}
    all_exit_chs: set[str] = set()
    for eb in exit_bids:
        all_exit_chs.update(blk_out_outputs[eb].keys())
    for ch in all_exit_chs:
        present = [eb for eb in exit_bids if ch in blk_out_outputs[eb]]
        if present:
            cur_v = blk_out_outputs[present[0]][ch]
            for eb in present[1:]:
                ev = blk_out_outputs[eb][ch]
                if ev != cur_v:
                    cur_v = SelectVal(blk_reach.get(eb, ConstVal(True, BOOL)), ev, cur_v, cur_v.ty)
            final_outputs[ch] = cur_v

    return CoreShaderProgram(
        name=name,
        route=mod.env.route,
        outputs=final_outputs,
        env=mod.env,
        num_blocks=len(mod.blocks),
        num_nodes=len(mod.nodes),
        unsupported_ops=unsupported,
    )
