from dataclasses import dataclass, field

from .core_types import CoreType, TTuple


class CoreVal:
    """Base class for immutable SSA values in CoreSSA-IR.

    Every value represents a single SSA definition (either a leaf input/const/symbol,
    an algebraic/memory/UF operation application, a gated select, or a structured
    theta-loop region output).
    """

    ty: CoreType

    def pretty(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class ConstVal(CoreVal):
    value: bool | int | float | str
    ty: CoreType

    def pretty(self) -> str:
        if isinstance(self.value, str):
            return repr(self.value)
        if isinstance(self.value, bool):
            return "true" if self.value else "false"
        return str(self.value)


@dataclass(frozen=True)
class ParamVal(CoreVal):
    """Function parameter, loop-carried phi binder, or initial memory token."""

    name: str
    ty: CoreType

    def pretty(self) -> str:
        return f"%{self.name}"


@dataclass(frozen=True)
class SymVal(CoreVal):
    """Opaque named symbol / constant handle (e.g., module ref, global symbol, provenance base)."""

    symbol: str
    ty: CoreType

    def pretty(self) -> str:
        return f"Sym({self.symbol})"


@dataclass(frozen=True)
class HoleVal(CoreVal):
    """Anti-unification generalization hole variable (parameterized in templates)."""

    hole_id: str
    ty: CoreType
    free_vars: tuple[str, ...] = ()

    def pretty(self) -> str:
        if self.free_vars:
            fv = ", ".join(f"%{v}" for v in self.free_vars)
            return f"?{self.hole_id}({fv})"
        return f"?{self.hole_id}"


@dataclass(frozen=True)
class OpApp(CoreVal):
    """Application of an annotated CoreSSA-IR operation.

    `attrs` holds static compile-time attributes (e.g., struct name, field name,
    symbol callee, provenance region id, or bitvector width).
    """

    op: str
    args: tuple[CoreVal, ...]
    ty: CoreType
    attrs: tuple[tuple[str, object], ...] = ()

    def attr(self, key: str, default: object = None) -> object:
        for k, v in self.attrs:
            if k == key:
                return v
        return default

    def pretty(self) -> str:
        parts = [a.pretty() for a in self.args]
        if self.attrs:
            attr_str = ", ".join(f"{k}={v}" for k, v in self.attrs)
            parts.append(f"[{attr_str}]")
        return f"{self.op}({', '.join(parts)})"


@dataclass(frozen=True)
class SelectVal(CoreVal):
    """Gated SSA value merge (gamma-node / if-then-else)."""

    cond: CoreVal
    then_val: CoreVal
    else_val: CoreVal
    ty: CoreType

    def pretty(self) -> str:
        return f"Select({self.cond.pretty()}, {self.then_val.pretty()}, {self.else_val.pretty()})"


@dataclass(frozen=True)
class LoopVal(CoreVal):
    """Structured SSA theta-loop region.

    Semantics:
      Starting from `init_vals` bound to `iter_params`, while `cond(iter_params)` holds,
      simultaneously update `iter_params := body_next(iter_params)`.
      Upon termination (`not cond`), yields the final tuple of `iter_params` (type `TTuple`).
    """

    iter_params: tuple[ParamVal, ...]
    init_vals: tuple[CoreVal, ...]
    cond: CoreVal
    body_next: tuple[CoreVal, ...]
    ty: TTuple
    unroll_hint: int = 4
    loop_id: str = "L"

    def pretty(self) -> str:
        binds = ", ".join(f"{p.pretty()}={v.pretty()}" for p, v in zip(self.iter_params, self.init_vals))
        steps = ", ".join(f"{p.pretty()}'={v.pretty()}" for p, v in zip(self.iter_params, self.body_next))
        return f"Loop[{self.loop_id}](init=[{binds}], cond={self.cond.pretty()}, step=[{steps}])"


@dataclass(frozen=True)
class LoopOutVal(CoreVal):
    """Projection of the `index`-th loop-carried value after loop exit."""

    loop: LoopVal
    index: int
    ty: CoreType

    def pretty(self) -> str:
        pname = self.loop.iter_params[self.index].name
        return f"LoopOut({self.loop.loop_id}.{pname})"


@dataclass
class CoreFunc:
    """A single function in CoreSSA-IR."""

    name: str
    params: list[ParamVal]
    ret_ty: CoreType
    ret_val: CoreVal
    mem_out: CoreVal | None = None
    preconditions: list[CoreVal] = field(default_factory=list)


@dataclass
class CoreModule:
    """A CoreSSA-IR module containing functions and struct type declarations."""

    funcs: dict[str, CoreFunc] = field(default_factory=dict)
    structs: dict[str, CoreType] = field(default_factory=dict)

    def add_func(self, fn: CoreFunc) -> None:
        self.funcs[fn.name] = fn

    def get_func(self, name: str) -> CoreFunc:
        return self.funcs[name]


def val_children(v: CoreVal) -> list[CoreVal]:
    """Return immediate child SSA values of `v`."""
    if isinstance(v, (ConstVal, ParamVal, SymVal, HoleVal)):
        return []
    if isinstance(v, OpApp):
        return list(v.args)
    if isinstance(v, SelectVal):
        return [v.cond, v.then_val, v.else_val]
    if isinstance(v, LoopVal):
        return list(v.init_vals) + [v.cond] + list(v.body_next)
    if isinstance(v, LoopOutVal):
        return [v.loop]
    return []


def val_tree_size(v: CoreVal) -> int:
    """Total node count of `v` viewed as an expression tree."""
    if isinstance(v, (ConstVal, ParamVal, SymVal, HoleVal)):
        return 1
    if isinstance(v, OpApp):
        return 1 + sum(val_tree_size(a) for a in v.args)
    if isinstance(v, SelectVal):
        return 1 + val_tree_size(v.cond) + val_tree_size(v.then_val) + val_tree_size(v.else_val)
    if isinstance(v, LoopVal):
        return (
            1
            + sum(val_tree_size(a) for a in v.init_vals)
            + val_tree_size(v.cond)
            + sum(val_tree_size(a) for a in v.body_next)
        )
    if isinstance(v, LoopOutVal):
        return 1 + val_tree_size(v.loop)
    return 1


def subst_val(v: CoreVal, mapping: dict[str, CoreVal]) -> CoreVal:
    """Capture-avoiding substitution of `ParamVal(name)` in `v`."""
    if not mapping:
        return v
    if isinstance(v, ParamVal):
        return mapping.get(v.name, v)
    if isinstance(v, (ConstVal, SymVal, HoleVal)):
        return v
    if isinstance(v, OpApp):
        return OpApp(v.op, tuple(subst_val(a, mapping) for a in v.args), v.ty, v.attrs)
    if isinstance(v, SelectVal):
        return SelectVal(
            subst_val(v.cond, mapping),
            subst_val(v.then_val, mapping),
            subst_val(v.else_val, mapping),
            v.ty,
        )
    if isinstance(v, LoopVal):
        # Do not substitute bound iter_params inside cond/body_next
        bound = {p.name for p in v.iter_params}
        inner_map = {k: val for k, val in mapping.items() if k not in bound}
        return LoopVal(
            iter_params=v.iter_params,
            init_vals=tuple(subst_val(a, mapping) for a in v.init_vals),
            cond=subst_val(v.cond, inner_map),
            body_next=tuple(subst_val(a, inner_map) for a in v.body_next),
            ty=v.ty,
            unroll_hint=v.unroll_hint,
            loop_id=v.loop_id,
        )
    if isinstance(v, LoopOutVal):
        return LoopOutVal(subst_val(v.loop, mapping), v.index, v.ty)  # type: ignore[arg-type]
    return v


def free_param_names(v: CoreVal) -> set[str]:
    """Collect all free `ParamVal` names in `v`."""
    if isinstance(v, ParamVal):
        return {v.name}
    if isinstance(v, (ConstVal, SymVal, HoleVal)):
        return set()
    if isinstance(v, OpApp):
        out: set[str] = set()
        for a in v.args:
            out |= free_param_names(a)
        return out
    if isinstance(v, SelectVal):
        return free_param_names(v.cond) | free_param_names(v.then_val) | free_param_names(v.else_val)
    if isinstance(v, LoopVal):
        bound = {p.name for p in v.iter_params}
        out = set()
        for a in v.init_vals:
            out |= free_param_names(a)
        inner = free_param_names(v.cond)
        for b in v.body_next:
            inner |= free_param_names(b)
        return out | (inner - bound)
    if isinstance(v, LoopOutVal):
        return free_param_names(v.loop)
    return set()


def optimal_tree_cover_cost(v: CoreVal, proved_nodes: set[CoreVal]) -> int:
    """Exact O(|T|) bottom-up BURS dynamic programming tree-covering cost."""
    if v in proved_nodes:
        return 1
    children = val_children(v)
    if not children:
        return 1
    return 1 + sum(optimal_tree_cover_cost(c, proved_nodes) for c in children)
