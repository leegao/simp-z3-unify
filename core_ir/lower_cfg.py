from dataclasses import dataclass, field

from .core_types import BOOL, CoreType, TTuple
from .ir import (
    ConstVal,
    CoreFunc,
    CoreVal,
    LoopOutVal,
    LoopVal,
    OpApp,
    ParamVal,
    SelectVal,
)


@dataclass(frozen=True)
class CFGPhi:
    """Standard SSA phi node at the top of a basic block: `%dest = phi [(pred_bb, val_name), ...]`."""

    dest: str
    ty: CoreType
    incoming: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class CFGInstr:
    """Three-address SSA instruction: `%dest = op(args...) [attrs]`."""

    dest: str
    op: str
    args: tuple[str | ConstVal, ...]
    ty: CoreType
    attrs: tuple[tuple[str, object], ...] = ()


class CFGTerminator:
    pass


@dataclass(frozen=True)
class CFGBr(CFGTerminator):
    target: str


@dataclass(frozen=True)
class CFGCondBr(CFGTerminator):
    cond: str
    true_target: str
    false_target: str


@dataclass(frozen=True)
class CFGRet(CFGTerminator):
    val: str


@dataclass
class CFGBlock:
    name: str
    phis: list[CFGPhi] = field(default_factory=list)
    instrs: list[CFGInstr] = field(default_factory=list)
    term: CFGTerminator = field(default_factory=lambda: CFGRet(""))


@dataclass
class CFGFunction:
    """Generic unstructured SSA-CFG IR (modeling LLVM IR / SPIR-V / MLIR / custom bytecode)."""

    name: str
    params: list[ParamVal]
    ret_ty: CoreType
    entry: str
    blocks: dict[str, CFGBlock]
    preconditions: list[CoreVal] = field(default_factory=list)


def compute_dominators(cfg: CFGFunction) -> dict[str, set[str]]:
    """Standard iterative dataflow dominator computation."""
    nodes = list(cfg.blocks.keys())
    preds: dict[str, set[str]] = {b: set() for b in nodes}
    for bname, blk in cfg.blocks.items():
        if isinstance(blk.term, CFGBr):
            preds[blk.term.target].add(bname)
        elif isinstance(blk.term, CFGCondBr):
            preds[blk.term.true_target].add(bname)
            preds[blk.term.false_target].add(bname)

    all_nodes = set(nodes)
    dom: dict[str, set[str]] = {n: ({cfg.entry} if n == cfg.entry else set(all_nodes)) for n in nodes}
    changed = True
    while changed:
        changed = False
        for n in nodes:
            if n == cfg.entry:
                continue
            if not preds[n]:
                new_dom = {n}
            else:
                new_dom = set.intersection(*(dom[p] for p in preds[n])) | {n}
            if new_dom != dom[n]:
                dom[n] = new_dom
                changed = True
    return dom


def find_natural_loops(cfg: CFGFunction) -> dict[str, tuple[str, str]]:
    """Detect reducible single-backedge natural loops: returns `header -> (latch, exit)`."""
    dom = compute_dominators(cfg)
    loops: dict[str, tuple[str, str]] = {}
    for bname, blk in cfg.blocks.items():
        succs: list[str] = []
        if isinstance(blk.term, CFGBr):
            succs = [blk.term.target]
        elif isinstance(blk.term, CFGCondBr):
            succs = [blk.term.true_target, blk.term.false_target]
        for s in succs:
            if s in dom[bname]:
                # Backedge bname -> s (header s dominates latch bname)
                hdr = cfg.blocks[s]
                if isinstance(hdr.term, CFGCondBr):
                    exit_blk = hdr.term.false_target if hdr.term.true_target != hdr.term.false_target else hdr.term.false_target
                    # Determine which successor of header leads to latch (body) vs exit
                    body_blk = hdr.term.true_target
                    if body_blk == bname or body_blk in dom[bname]:
                        loops[s] = (bname, hdr.term.false_target)
                    else:
                        loops[s] = (bname, hdr.term.true_target)
    return loops


def _eval_block_instrs(blk: CFGBlock, env: dict[str, CoreVal]) -> dict[str, CoreVal]:
    cur = dict(env)
    for ins in blk.instrs:
        arg_vals = tuple(a if isinstance(a, ConstVal) else cur[a] for a in ins.args)
        cur[ins.dest] = OpApp(ins.op, arg_vals, ins.ty, ins.attrs)
    return cur


def lower_cfg_to_core(cfg: CFGFunction) -> CoreFunc:
    """Lower a reducible unstructured SSA-CFG into CoreSSA-IR (gated SelectVal + LoopVal).

    Supports:
      - Straight-line SSA blocks with arbitrary dialect operations
      - Diamond if-then-else branches with SSA phi merges -> `SelectVal`
      - Natural loops (header with phi nodes + backedge from latch) -> unbounded `LoopVal`
    """
    natural_loops = find_natural_loops(cfg)
    init_env: dict[str, CoreVal] = {p.name: p for p in cfg.params}
    loop_idx = 0

    def lower_from(
        bname: str,
        pred_name: str | None,
        env: dict[str, CoreVal],
        stop_at: str | None = None,
    ) -> tuple[CoreVal | None, dict[str, CoreVal]]:
        nonlocal loop_idx
        if bname == stop_at:
            return None, env

        blk = cfg.blocks[bname]

        # Natural loop header -> lift directly to LoopVal!
        if bname in natural_loops and pred_name != natural_loops[bname][0]:
            latch_name, exit_name = natural_loops[bname]
            iter_params: list[ParamVal] = []
            init_vals: list[CoreVal] = []
            latch_val_names: list[str] = []

            for phi in blk.phis:
                inc_map = dict(phi.incoming)
                assert pred_name is not None and pred_name in inc_map
                assert latch_name in inc_map
                iter_params.append(ParamVal(phi.dest, phi.ty))
                init_vals.append(env[inc_map[pred_name]])
                latch_val_names.append(inc_map[latch_name])

            # Evaluate header instructions and guard in terms of iter_params
            hdr_env = dict(env)
            for p in iter_params:
                hdr_env[p.name] = p
            hdr_env = _eval_block_instrs(blk, hdr_env)
            assert isinstance(blk.term, CFGCondBr)
            cond_val = hdr_env[blk.term.cond]
            body_entry = blk.term.true_target if blk.term.false_target == exit_name else blk.term.false_target
            if blk.term.false_target != exit_name:
                cond_val = OpApp("not", (cond_val,), BOOL)

            # Evaluate loop body from body_entry back to header
            _, latch_env = lower_from(body_entry, bname, hdr_env, stop_at=bname)
            body_next = tuple(latch_env[vn] for vn in latch_val_names)

            lid = f"cfgL{loop_idx}"
            loop_idx += 1
            lp = LoopVal(
                iter_params=tuple(iter_params),
                init_vals=tuple(init_vals),
                cond=cond_val,
                body_next=body_next,
                ty=TTuple(tuple(p.ty for p in iter_params)),
                loop_id=lid,
            )
            post_env = dict(env)
            for i, p in enumerate(iter_params):
                post_env[p.name] = LoopOutVal(lp, i, p.ty)
            # Re-evaluate any non-phi header defs in post_env
            post_env = _eval_block_instrs(blk, post_env)
            return lower_from(exit_name, bname, post_env, stop_at=stop_at)

        # Standard block: resolve phis using pred_name
        cur = dict(env)
        if pred_name is not None:
            for phi in blk.phis:
                inc_map = dict(phi.incoming)
                if pred_name in inc_map:
                    cur[phi.dest] = env[inc_map[pred_name]]

        cur = _eval_block_instrs(blk, cur)

        if isinstance(blk.term, CFGRet):
            return cur[blk.term.val], cur

        if isinstance(blk.term, CFGBr):
            return lower_from(blk.term.target, bname, cur, stop_at=stop_at)

        if isinstance(blk.term, CFGCondBr):
            cond = cur[blk.term.cond]
            t_blk, f_blk = blk.term.true_target, blk.term.false_target
            # Find immediate join block if both branches converge
            t_term = cfg.blocks[t_blk].term
            f_term = cfg.blocks[f_blk].term
            join: str | None = None
            if isinstance(t_term, CFGBr) and isinstance(f_term, CFGBr) and t_term.target == f_term.target:
                join = t_term.target
            elif isinstance(t_term, CFGBr) and t_term.target == f_blk:
                join = f_blk
            elif isinstance(f_term, CFGBr) and f_term.target == t_blk:
                join = t_blk

            if join is not None:
                _, env_t = lower_from(t_blk, bname, cur, stop_at=join)
                _, env_f = lower_from(f_blk, bname, cur, stop_at=join)
                join_blk = cfg.blocks[join]
                merged = dict(cur)
                for phi in join_blk.phis:
                    inc_map = dict(phi.incoming)
                    vt = env_t[inc_map.get(t_blk, inc_map.get(bname, ""))]
                    vf = env_f[inc_map.get(f_blk, inc_map.get(bname, ""))]
                    merged[phi.dest] = vt if vt == vf else SelectVal(cond, vt, vf, phi.ty)
                # Continue from join (skipping join's phis since already merged)
                merged = _eval_block_instrs(join_blk, merged)
                if isinstance(join_blk.term, CFGRet):
                    return merged[join_blk.term.val], merged
                if isinstance(join_blk.term, CFGBr):
                    return lower_from(join_blk.term.target, join, merged, stop_at=stop_at)
                return None, merged

            # Non-converging branch (both arms return)
            rt, _ = lower_from(t_blk, bname, cur, stop_at=stop_at)
            rf, _ = lower_from(f_blk, bname, cur, stop_at=stop_at)
            assert rt is not None and rf is not None
            return (rt if rt == rf else SelectVal(cond, rt, rf, rt.ty)), cur

        return None, cur

    ret_val, _ = lower_from(cfg.entry, None, init_env)
    assert ret_val is not None, f"CFG function {cfg.name} did not return a value"
    return CoreFunc(
        name=cfg.name,
        params=cfg.params,
        ret_ty=cfg.ret_ty,
        ret_val=ret_val,
        preconditions=list(cfg.preconditions),
    )
