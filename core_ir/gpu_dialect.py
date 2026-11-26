import struct
import z3

from .core_types import BOOL, BV32, BV64, FLOAT, INT, TVector
from .op_registry import InverseSpec, OpRegistry, OpSpec, OpTrait, build_default_registry


def _u32(v: int) -> int:
    return int(v) & 0xFFFFFFFF


def _s32(v: int) -> int:
    v = int(v) & 0xFFFFFFFF
    return v - 0x100000000 if (v & 0x80000000) else v


def _u64(v: int) -> int:
    return int(v) & 0xFFFFFFFFFFFFFFFF


def _f32_bits_to_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", _u32(bits)))[0]


def _float_to_f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


# ---------------------------------------------------------------------------
# Concrete Evaluators (Constant Folding for GPU / Valhall SSA-IR Ops)
# ---------------------------------------------------------------------------

def _cf_iadd(args, _attrs, _ty):
    return _u32(args[0] + args[1])


def _cf_isub(args, _attrs, _ty):
    return _u32(args[0] - args[1])


def _cf_imul(args, _attrs, _ty):
    return _u32(args[0] * args[1])


def _cf_iadd64(args, _attrs, _ty):
    return _u64(args[0] + args[1])


def _cf_isub64(args, _attrs, _ty):
    return _u64(args[0] - args[1])


def _cf_bitand(args, _attrs, _ty):
    return _u32(args[0] & args[1])


def _cf_bitor(args, _attrs, _ty):
    return _u32(args[0] | args[1])


def _cf_bitxor(args, _attrs, _ty):
    return _u32(args[0] ^ args[1])


def _cf_bitnot(args, _attrs, _ty):
    return _u32(~args[0])


def _cf_shl(args, _attrs, _ty):
    return _u32(args[0] << (args[1] & 31))


def _cf_shru(args, _attrs, _ty):
    return _u32(_u32(args[0]) >> (args[1] & 31))


def _cf_shrs(args, _attrs, _ty):
    return _u32(_s32(args[0]) >> (args[1] & 31))


def _cf_shl64(args, _attrs, _ty):
    return _u64(args[0] << (args[1] & 63))


def _cf_shru64(args, _attrs, _ty):
    return _u64(_u64(args[0]) >> (args[1] & 63))


def _cf_bfe_u(args, _attrs, _ty):
    base, off, width = _u32(args[0]), _u32(args[1]) & 31, _u32(args[2]) & 31
    if width == 0:
        return 0
    mask = (1 << width) - 1
    return (base >> off) & mask


def _cf_bfe_s(args, _attrs, _ty):
    u = _cf_bfe_u(args, _attrs, _ty)
    width = _u32(args[2]) & 31
    if width == 0:
        return 0
    if u & (1 << (width - 1)):
        u |= (~((1 << width) - 1)) & 0xFFFFFFFF
    return _u32(u)


def _cf_bfi(args, _attrs, _ty):
    base, ins, off, width = _u32(args[0]), _u32(args[1]), _u32(args[2]) & 31, _u32(args[3]) & 31
    if width == 0:
        return base
    mask = ((1 << width) - 1) << off
    return _u32((base & ~mask) | ((ins << off) & mask))


def _cf_pack64(args, _attrs, _ty):
    return _u64(_u32(args[0]) | (_u32(args[1]) << 32))


def _cf_uconv32_64(args, _attrs, _ty):
    return _u64(_u32(args[0]))


def _cf_sconv32_64(args, _attrs, _ty):
    return _u64(_s32(args[0]))


def _cf_uconv64_32(args, _attrs, _ty):
    return _u32(args[0])


# ---------------------------------------------------------------------------
# Z3-Counterpart Builders (Aliasing GPU / Valhall SSA-IR Ops to Z3)
# ---------------------------------------------------------------------------

def _as_bv(x: z3.ExprRef, width: int) -> z3.BitVecRef:
    if z3.is_bv(x):
        if x.size() == width:
            return x
        if x.size() < width:
            return z3.ZeroExt(width - x.size(), x)
        return z3.Extract(width - 1, 0, x)
    if z3.is_int(x):
        return z3.Int2BV(x, width)
    if z3.is_bool(x):
        return z3.If(x, z3.BitVecVal(1, width), z3.BitVecVal(0, width))
    # Fallback UF coercion from Real / Sym
    f = z3.Function(f"coerce_to_bv{width}_{x.sort()}", x.sort(), z3.BitVecSort(width))
    return f(x)


def _z3_bv_shl(zargs, _attrs, ty, _backend):
    w = ty.width if hasattr(ty, "width") else 32
    a, b = _as_bv(zargs[0], w), _as_bv(zargs[1], w)
    mask = z3.BitVecVal(w - 1, w)
    return a << (b & mask)


def _z3_bv_shru(zargs, _attrs, ty, _backend):
    w = ty.width if hasattr(ty, "width") else 32
    a, b = _as_bv(zargs[0], w), _as_bv(zargs[1], w)
    mask = z3.BitVecVal(w - 1, w)
    return z3.LShR(a, b & mask)


def _z3_bv_shrs(zargs, _attrs, ty, _backend):
    w = ty.width if hasattr(ty, "width") else 32
    a, b = _as_bv(zargs[0], w), _as_bv(zargs[1], w)
    mask = z3.BitVecVal(w - 1, w)
    return a >> (b & mask)


def _z3_bfe_u(zargs, _attrs, _ty, _backend):
    base, off, width = _as_bv(zargs[0], 32), _as_bv(zargs[1], 32) & 31, _as_bv(zargs[2], 32) & 31
    mask = (z3.BitVecVal(1, 32) << width) - z3.BitVecVal(1, 32)
    return z3.If(width == 0, z3.BitVecVal(0, 32), z3.LShR(base, off) & mask)


def _z3_bfe_s(zargs, _attrs, _ty, _backend):
    base, off, width = _as_bv(zargs[0], 32), _as_bv(zargs[1], 32) & 31, _as_bv(zargs[2], 32) & 31
    shift_up = z3.BitVecVal(32, 32) - width
    shifted = (base << (shift_up - off)) >> shift_up
    return z3.If(width == 0, z3.BitVecVal(0, 32), shifted)


def _z3_bfi(zargs, _attrs, _ty, _backend):
    base, ins, off, width = (
        _as_bv(zargs[0], 32),
        _as_bv(zargs[1], 32),
        _as_bv(zargs[2], 32) & 31,
        _as_bv(zargs[3], 32) & 31,
    )
    mask = ((z3.BitVecVal(1, 32) << width) - z3.BitVecVal(1, 32)) << off
    return z3.If(width == 0, base, (base & ~mask) | ((ins << off) & mask))


def _z3_pack64(zargs, _attrs, _ty, _backend):
    lo, hi = _as_bv(zargs[0], 32), _as_bv(zargs[1], 32)
    return z3.Concat(hi, lo)


def _z3_split64(zargs, _attrs, _ty, _backend):
    # Represent Split64(x) directly as the 64-bit BV itself so TupleExtract(0/1) extracts lo/hi!
    return _as_bv(zargs[0], 64)


def _z3_tuple_extract(zargs, attrs, ty, backend):
    idx = 0
    for k, v in attrs:
        if k == "index":
            idx = int(v)
    src = zargs[0]
    if z3.is_bv(src) and src.size() == 64:
        return z3.Extract(31, 0, src) if idx == 0 else z3.Extract(63, 32, src)
    uf = backend._get_uf(f"tuple_extract_{idx}", [src.sort()], backend.sort_of(ty))
    return uf(src)


def _z3_uconv32_64(zargs, _attrs, _ty, _backend):
    return z3.ZeroExt(32, _as_bv(zargs[0], 32))


def _z3_sconv32_64(zargs, _attrs, _ty, _backend):
    return z3.SignExt(32, _as_bv(zargs[0], 32))


def _z3_uconv64_32(zargs, _attrs, _ty, _backend):
    return z3.Extract(31, 0, _as_bv(zargs[0], 64))


def _z3_icmp(kind: str):
    def _builder(zargs, _attrs, _ty, _backend):
        a, b = _as_bv(zargs[0], 32), _as_bv(zargs[1], 32)
        if kind == "eq":
            return a == b
        if kind == "ne":
            return a != b
        if kind == "slt":
            return a < b
        if kind == "sle":
            return a <= b
        if kind == "sgt":
            return a > b
        if kind == "sge":
            return a >= b
        if kind == "ult":
            return z3.ULT(a, b)
        if kind == "ule":
            return z3.ULE(a, b)
        if kind == "ugt":
            return z3.UGT(a, b)
        if kind == "uge":
            return z3.UGE(a, b)
        raise ValueError(kind)

    return _builder


def _z3_bitcast(zargs, _attrs, ty, backend):
    src = zargs[0]
    dst_sort = backend.sort_of(ty)
    if src.sort() == dst_sort:
        return src
    uf = backend._get_uf(f"bitcast_{src.sort()}_to_{dst_sort}", [src.sort()], dst_sort)
    return uf(src)


def _z3_vec_construct(zargs, _attrs, ty, backend):
    arr_sort = z3.ArraySort(z3.IntSort(), backend.sort_of(ty.elem if isinstance(ty, TVector) else FLOAT))
    base = z3.K(z3.IntSort(), zargs[0])
    cur = base
    for i, a in enumerate(zargs):
        cur = z3.Store(cur, z3.IntVal(i), a)
    return cur


def _z3_vec_extract(zargs, attrs, ty, backend):
    idx = 0
    for k, v in attrs:
        if k == "index":
            idx = int(v)
    src = zargs[0]
    if isinstance(src.sort(), z3.ArraySortRef):
        return z3.Select(src, z3.IntVal(idx))
    uf = backend._get_uf(f"vec_extract_{idx}", [src.sort()], backend.sort_of(ty))
    return uf(src)


def _z3_fma(zargs, _attrs, ty, backend):
    # Exact real-modeled FMA or rigid UF depending on sort
    if z3.is_real(zargs[0]) and z3.is_real(zargs[1]) and z3.is_real(zargs[2]):
        return zargs[0] * zargs[1] + zargs[2]
    uf = backend._get_uf("vh_fma", [a.sort() for a in zargs], backend.sort_of(ty))
    return uf(*zargs)


def _z3_fclamp(zargs, _attrs, _ty, _backend):
    x, lo, hi = zargs[0], zargs[1], zargs[2]
    return z3.If(x < lo, lo, z3.If(x > hi, hi, x))


def _z3_fmin(zargs, _attrs, _ty, _backend):
    return z3.If(zargs[0] <= zargs[1], zargs[0], zargs[1])


def _z3_fmax(zargs, _attrs, _ty, _backend):
    return z3.If(zargs[0] >= zargs[1], zargs[0], zargs[1])


def _z3_fabs(zargs, _attrs, _ty, _backend):
    return z3.If(zargs[0] < 0, -zargs[0], zargs[0])


def _z3_sabs(zargs, _attrs, _ty, _backend):
    x = _as_bv(zargs[0], 32)
    return z3.If(x < 0, -x, x)


def _z3_sdiv(zargs, _attrs, _ty, _backend):
    return _as_bv(zargs[0], 32) / _as_bv(zargs[1], 32)


def _z3_udiv(zargs, _attrs, _ty, _backend):
    return z3.UDiv(_as_bv(zargs[0], 32), _as_bv(zargs[1], 32))


def _z3_srem(zargs, _attrs, _ty, _backend):
    return z3.SRem(_as_bv(zargs[0], 32), _as_bv(zargs[1], 32))


def _z3_umod(zargs, _attrs, _ty, _backend):
    return z3.URem(_as_bv(zargs[0], 32), _as_bv(zargs[1], 32))


def _z3_rotl(zargs, _attrs, _ty, _backend):
    a, b = _as_bv(zargs[0], 32), _as_bv(zargs[1], 32) & 31
    return z3.RotateLeft(a, b)


def _z3_rotr(zargs, _attrs, _ty, _backend):
    a, b = _as_bv(zargs[0], 32), _as_bv(zargs[1], 32) & 31
    return z3.RotateRight(a, b)


def build_gpu_registry(fp_fast_math: bool = False) -> OpRegistry:
    """Build the expanded CoreSSA-IR OpRegistry covering all 155 ValhallSSA opcodes,
    Valhall ISA-IR dialect ops, and SPIR-V ops with exact algebraic traits, Z3 aliases,
    and concrete constant-folding evaluators.
    """
    reg = build_default_registry()

    # --- Transparent SSA / Valhall ISA Moves & Nops ---
    for mv in ("vh.copy", "vh.nop", "vh.mov", "spv.copy"):
        reg.register(OpSpec(mv, OpTrait.TRANSPARENT_MOVE | OpTrait.PURE, z3_kind="identity"))

    # --- 32-bit & 64-bit Integer / Bitvector ALU (AC + Inverses + Constant Folding) ---
    bvadd_inv = InverseSpec("bv.add", "bv.sub", "bv.neg", nonzero_rhs=False)
    reg.register(OpSpec("bv.add", OpTrait.AC | OpTrait.PURE, identity=0, has_identity=True, inverse=bvadd_inv, z3_kind="bv_add", concrete_eval=_cf_iadd))
    reg.register(OpSpec("bv.sub", OpTrait.PURE, identity=0, has_identity=True, inverse=bvadd_inv, z3_kind="bv_sub", concrete_eval=_cf_isub))
    reg.register(OpSpec("bv.neg", OpTrait.INVOLUTION | OpTrait.PURE, inverse=bvadd_inv, z3_kind="bv_neg"))
    reg.register(OpSpec("bv.mul", OpTrait.AC | OpTrait.PURE, identity=1, has_identity=True, absorbing=0, has_absorbing=True, z3_kind="bv_mul", concrete_eval=_cf_imul))

    bv64_inv = InverseSpec("bv64.add", "bv64.sub", "bv64.neg", nonzero_rhs=False)
    reg.register(OpSpec("bv64.add", OpTrait.AC | OpTrait.PURE, identity=0, has_identity=True, inverse=bv64_inv, z3_kind="bv_add", concrete_eval=_cf_iadd64))
    reg.register(OpSpec("bv64.sub", OpTrait.PURE, identity=0, has_identity=True, inverse=bv64_inv, z3_kind="bv_sub", concrete_eval=_cf_isub64))
    reg.register(OpSpec("bv64.neg", OpTrait.INVOLUTION | OpTrait.PURE, inverse=bv64_inv, z3_kind="bv_neg"))
    reg.register(OpSpec("bv64.mul", OpTrait.AC | OpTrait.PURE, identity=1, has_identity=True, absorbing=0, has_absorbing=True, z3_kind="bv_mul"))

    # Bitwise 32 & 64
    reg.register(OpSpec("bv.and", OpTrait.AC | OpTrait.IDEMPOTENT | OpTrait.PURE, absorbing=0, has_absorbing=True, z3_kind="bv_and", concrete_eval=_cf_bitand))
    reg.register(OpSpec("bv.or", OpTrait.AC | OpTrait.IDEMPOTENT | OpTrait.PURE, identity=0, has_identity=True, z3_kind="bv_or", concrete_eval=_cf_bitor))
    reg.register(OpSpec("bv.xor", OpTrait.AC | OpTrait.SELF_INVERSE | OpTrait.PURE, identity=0, has_identity=True, z3_kind="bv_xor", concrete_eval=_cf_bitxor))
    reg.register(OpSpec("bv.not", OpTrait.INVOLUTION | OpTrait.PURE, z3_kind="bv_not", concrete_eval=_cf_bitnot))
    reg.register(OpSpec("bv64.and", OpTrait.AC | OpTrait.IDEMPOTENT | OpTrait.PURE, absorbing=0, has_absorbing=True, z3_kind="bv_and"))
    reg.register(OpSpec("bv64.or", OpTrait.AC | OpTrait.IDEMPOTENT | OpTrait.PURE, identity=0, has_identity=True, z3_kind="bv_or"))
    reg.register(OpSpec("bv64.xor", OpTrait.AC | OpTrait.SELF_INVERSE | OpTrait.PURE, identity=0, has_identity=True, z3_kind="bv_xor"))
    reg.register(OpSpec("bv64.not", OpTrait.INVOLUTION | OpTrait.PURE, z3_kind="bv_not"))

    # Shifts, Rotates, Bitfields, Div/Rem
    reg.register(OpSpec("bv.shl", OpTrait.PURE, z3_builder=_z3_bv_shl, concrete_eval=_cf_shl))
    reg.register(OpSpec("bv.shru", OpTrait.PURE, z3_builder=_z3_bv_shru, concrete_eval=_cf_shru))
    reg.register(OpSpec("bv.shrs", OpTrait.PURE, z3_builder=_z3_bv_shrs, concrete_eval=_cf_shrs))
    reg.register(OpSpec("bv64.shl", OpTrait.PURE, z3_builder=_z3_bv_shl, concrete_eval=_cf_shl64))
    reg.register(OpSpec("bv64.shru", OpTrait.PURE, z3_builder=_z3_bv_shru, concrete_eval=_cf_shru64))
    reg.register(OpSpec("bv64.shrs", OpTrait.PURE, z3_builder=_z3_bv_shrs))
    reg.register(OpSpec("bv.rotl", OpTrait.PURE, z3_builder=_z3_rotl))
    reg.register(OpSpec("bv.rotr", OpTrait.PURE, z3_builder=_z3_rotr))
    reg.register(OpSpec("bv.sdiv", OpTrait.PURE, z3_builder=_z3_sdiv))
    reg.register(OpSpec("bv.udiv", OpTrait.PURE, z3_builder=_z3_udiv))
    reg.register(OpSpec("bv.srem", OpTrait.PURE, z3_builder=_z3_srem))
    reg.register(OpSpec("bv.smod", OpTrait.PURE, z3_builder=_z3_srem))
    reg.register(OpSpec("bv.umod", OpTrait.PURE, z3_builder=_z3_umod))
    reg.register(OpSpec("bv.sabs", OpTrait.PURE, z3_builder=_z3_sabs))
    reg.register(OpSpec("vh.bfe_u", OpTrait.PURE, z3_builder=_z3_bfe_u, concrete_eval=_cf_bfe_u))
    reg.register(OpSpec("vh.bfe_s", OpTrait.PURE, z3_builder=_z3_bfe_s, concrete_eval=_cf_bfe_s))
    reg.register(OpSpec("vh.bfi", OpTrait.PURE, z3_builder=_z3_bfi, concrete_eval=_cf_bfi))

    # 64-bit Pack / Split / Conversions
    reg.register(OpSpec("vh.pack64", OpTrait.PURE, z3_builder=_z3_pack64, concrete_eval=_cf_pack64))
    reg.register(OpSpec("vh.split64", OpTrait.PURE, z3_builder=_z3_split64))
    reg.register(OpSpec("vh.tuple_extract", OpTrait.PURE, z3_builder=_z3_tuple_extract))
    reg.register(OpSpec("vh.uconv32_64", OpTrait.PURE, z3_builder=_z3_uconv32_64, concrete_eval=_cf_uconv32_64))
    reg.register(OpSpec("vh.sconv32_64", OpTrait.PURE, z3_builder=_z3_sconv32_64, concrete_eval=_cf_sconv32_64))
    reg.register(OpSpec("vh.uconv64_32", OpTrait.PURE, z3_builder=_z3_uconv64_32, concrete_eval=_cf_uconv64_32))
    reg.register(OpSpec("vh.bitcast", OpTrait.BITCAST | OpTrait.PURE, z3_builder=_z3_bitcast))

    # Integer comparisons
    for cmp_name in ("eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"):
        tr = (OpTrait.COMM | OpTrait.PURE) if cmp_name in ("eq", "ne") else OpTrait.PURE
        reg.register(OpSpec(f"bv.cmp_{cmp_name}", tr, z3_builder=_z3_icmp(cmp_name)))

    # Floating-point ALU (strict = COMM only on add/mul; fast_math = AC + inverses)
    fp_ac = (OpTrait.AC | OpTrait.PURE) if fp_fast_math else (OpTrait.COMM | OpTrait.PURE)
    fadd_inv = InverseSpec("f.add", "f.sub", "f.neg", nonzero_rhs=False) if fp_fast_math else None
    reg.register(OpSpec("f.add", fp_ac, identity=0.0, has_identity=fp_fast_math, inverse=fadd_inv, z3_kind="add"))
    reg.register(OpSpec("f.sub", OpTrait.PURE, inverse=fadd_inv, z3_kind="sub"))
    reg.register(OpSpec("f.neg", OpTrait.INVOLUTION | OpTrait.PURE, inverse=fadd_inv, z3_kind="neg"))
    reg.register(OpSpec("f.mul", fp_ac, identity=1.0, has_identity=fp_fast_math, z3_kind="mul"))
    reg.register(OpSpec("f.div", OpTrait.PURE, z3_kind="div"))
    reg.register(OpSpec("vh.fma", OpTrait.PURE, z3_builder=_z3_fma))
    reg.register(OpSpec("vh.fclamp", OpTrait.PURE, z3_builder=_z3_fclamp))
    reg.register(OpSpec("vh.fmin", OpTrait.AC | OpTrait.IDEMPOTENT | OpTrait.PURE, z3_builder=_z3_fmin))
    reg.register(OpSpec("vh.fmax", OpTrait.AC | OpTrait.IDEMPOTENT | OpTrait.PURE, z3_builder=_z3_fmax))
    reg.register(OpSpec("vh.nmin", OpTrait.AC | OpTrait.IDEMPOTENT | OpTrait.PURE, z3_builder=_z3_fmin))
    reg.register(OpSpec("vh.nmax", OpTrait.AC | OpTrait.IDEMPOTENT | OpTrait.PURE, z3_builder=_z3_fmax))
    reg.register(OpSpec("vh.fabs", OpTrait.PURE, z3_builder=_z3_fabs))

    # FP comparisons
    for op_n, zk, comm in (
        ("f.cmp_oeq", "eq", True),
        ("f.cmp_one", "neq", True),
        ("f.cmp_une", "neq", True),
        ("f.cmp_olt", "lt", False),
        ("f.cmp_ole", "le", False),
        ("f.cmp_ogt", "gt", False),
        ("f.cmp_oge", "ge", False),
    ):
        reg.register(OpSpec(op_n, (OpTrait.COMM | OpTrait.PURE) if comm else OpTrait.PURE, z3_kind=zk))

    # Vectors & Lanes
    for vn in ("vh.vec2", "vh.vec3", "vh.vec4"):
        reg.register(OpSpec(vn, OpTrait.PURE, z3_builder=_z3_vec_construct))
    reg.register(OpSpec("vh.vec_extract", OpTrait.PURE, z3_builder=_z3_vec_extract))
    reg.register(OpSpec("vh.vec_shuffle", OpTrait.PURE | OpTrait.RIGID_UF))
    reg.register(OpSpec("vh.pack_half2x16", OpTrait.PURE | OpTrait.RIGID_UF))
    reg.register(OpSpec("vh.unpack_half2x16", OpTrait.PURE | OpTrait.RIGID_UF))

    # GPU Lane / Subgroup / Quad Derivative Ops (LANE_OP | RIGID_UF)
    for lop in ("vh.subgroup_ballot", "vh.subgroup_shuffle", "vh.dpdx", "vh.dpdy", "vh.fwidth", "vh.lane_swz"):
        reg.register(OpSpec(lop, OpTrait.LANE_OP | OpTrait.RIGID_UF | OpTrait.PURE))

    # Valhall ISA-IR & Shader Effects / Barriers / Atomics / Textures / Transcendentals
    for eff in ("vh.control_barrier", "vh.atomic_rmw", "vh.atomic_xchg", "vh.atomic_cmpxchg", "vh.wait_slot", "vh.kill"):
        reg.register(OpSpec(eff, OpTrait.EFFECT_BARRIER | OpTrait.RIGID_UF))

    for rigid in (
        "vh.acos", "vh.asin", "vh.atan", "vh.atan2", "vh.ceil", "vh.clz", "vh.bit_count",
        "vh.bit_reverse", "vh.conv_ftos", "vh.conv_ftou", "vh.conv_stof", "vh.conv_utof",
        "vh.cos", "vh.cosh", "vh.exp", "vh.exp2", "vh.floor", "vh.frexp_exp", "vh.frexp_sig",
        "vh.idot4s", "vh.idot4su", "vh.idot4u", "vh.img_query_lod", "vh.img_query_size",
        "vh.img_texel_addr", "vh.inv_sqrt", "vh.isinf", "vh.isnan", "vh.ldexp", "vh.log",
        "vh.log2", "vh.pow", "vh.round_even", "vh.sin", "vh.sinh", "vh.sqrt", "vh.tanh",
        "vh.trunc", "vh.tex_fetch", "vh.tex_gather", "vh.tex_sample", "vh.tex_sample_dref",
        "vh.tex_sample_lod", "vh.fcmp_total", "vh.fau_read", "vh.lea_buf_imm", "vh.blend",
    ):
        reg.register(OpSpec(rigid, OpTrait.RIGID_UF | OpTrait.PURE))

    return reg
