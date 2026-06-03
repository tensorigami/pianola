"""Tensors — the catform AST.

Pure data: a `.cat` file parses into a `Module` (a dict of `Function`s, each a
typed sequence of `Op`s over `Tensor`s). Inert — no execution, no framework,
no interpreter machinery. The loaded-and-callable object that binds a Module
with config + weights + backend lives in `pianola.lower.interpreter` as
`Catform`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

# ── Tensor (type annotation from .cat) ────────────────────────────────────────

type Dim = int | str  # int = concrete size, str = name ("N", "param.hidden")


@dataclass(frozen=True)
class Tensor:
    """Type annotation from a .cat file: dtype + shape.

    Dims are structural: str for named dims (free variables like "N",
    config refs like "param.hidden"), int for concrete sizes (1, 2).
    Resolution to concrete ints happens at the runtime boundary.
    """

    dtype: str  # "bf16", "f32", "i32"
    shape: tuple[Dim, ...]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Tensor:
        return cls(dtype=d["dtype"], shape=tuple(d["shape"]))

    def __str__(self) -> str:
        return f"{self.dtype}[{', '.join(str(d) for d in self.shape)}]"

    @property
    def named_dims(self) -> tuple[str, ...]:
        """The symbolic (str) elements of shape — derived inputs for shape-driven ops."""
        out: list[str] = []
        for d in self.shape:
            match d:
                case str() as s:
                    out.append(s)
        return tuple(out)


# ── Introductions (inert data on the surface of the computation) ────────────

type Value = int | float | list[Any]


@dataclass(frozen=True)
class Literal:
    """Known value — entered inline by the .cat author."""

    _SPECIALS: ClassVar[dict[str, float]] = {
        "inf": float("inf"),
        "-inf": float("-inf"),
        "nan": float("nan"),
    }

    value: Value
    dtype: str

    @classmethod
    def from_dict(cls, d: dict[str, Any], ot: Tensor) -> tuple[Literal, tuple[str, ...]]:
        return cls(value=cls._parse(d["value"]), dtype=ot.dtype), ()

    @classmethod
    def _parse(cls, v: Any) -> Value:
        """JSON literal → Python value. 'inf'/'-inf'/'nan' arrive as strings (JSON can't encode)."""
        match v:
            case str() as s:
                return cls._SPECIALS[s]
            case list() as items:
                return [cls._parse(x) for x in items]
            case _:
                return v


@dataclass(frozen=True)
class Random:
    """Uniform random sample — drawn afresh each call."""

    lower: float
    upper: float
    dtype: str
    dims: tuple[Dim, ...]  # from type annotation — may be symbolic

    @classmethod
    def from_dict(cls, d: dict[str, Any], ot: Tensor) -> tuple[Random, tuple[str, ...]]:
        impl = cls(
            lower=float(d["lower"]),
            upper=float(d["upper"]),
            dtype=ot.dtype,
            dims=ot.shape,
        )
        return impl, ot.named_dims


@dataclass(frozen=True)
class Iota:
    """Index ramp [start, start+1, ..., start+S-1].

    Length S comes from the output type (`dims`, possibly symbolic); `start` is
    a literal int or a name resolved at runtime.
    """

    start: int | str
    dtype: str
    dims: tuple[Dim, ...]  # from output type — the length S (may be symbolic)

    @classmethod
    def from_dict(cls, d: dict[str, Any], ot: Tensor) -> tuple[Iota, tuple[str, ...]]:
        impl = cls(start=d["args"][0], dtype=ot.dtype, dims=ot.shape)
        return impl, ot.named_dims + impl.start_inputs

    @property
    def start_inputs(self) -> tuple[str, ...]:
        """The named (str) form of `start`, as a 1-tuple; empty if `start` is literal int."""
        match self.start:
            case str() as s:
                return (s,)
            case _:
                return ()


# ── Self-adjoint ops (View, Map) ─────────────────────────────────────────────


@dataclass(frozen=True)
class View:
    """Pure isomorphism (einops rearrange). Self-adjoint (inverse rearrangement)."""

    pattern: str
    axes: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any], _ot: Tensor | None) -> tuple[View, tuple[str, ...]]:
        impl = cls(
            pattern=d.get("pattern", ""),
            axes={k: int(v) for k, v in d.get("axes", {}).items()},
        )
        return impl, tuple(str(a) for a in d["args"])


@dataclass(frozen=True)
class Map:
    """Elementwise function lift. Self-adjoint (derivative of f is f')."""

    function: str

    @classmethod
    def from_dict(cls, d: dict[str, Any], _ot: Tensor | None) -> tuple[Map, tuple[str, ...]]:
        return cls(function=d["function"]), tuple(str(a) for a in d["args"])


# ── Dual pair: Fold ↔ Tile ──────────────────────────────────────────────────


@dataclass(frozen=True)
class Fold:
    """Fold over an index (einops reduce). Adjoint of Tile."""

    pattern: str
    reduction: str

    @classmethod
    def from_dict(cls, d: dict[str, Any], _ot: Tensor | None) -> tuple[Fold, tuple[str, ...]]:
        impl = cls(pattern=d["pattern"], reduction=d["reduction"])
        return impl, tuple(str(a) for a in d["args"])


@dataclass(frozen=True)
class Tile:
    """Replicate along an index (einops repeat). Adjoint of Fold."""

    pattern: str
    axes: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any], _ot: Tensor | None) -> tuple[Tile, tuple[str, ...]]:
        impl = cls(
            pattern=d.get("pattern", ""),
            axes={k: int(v) for k, v in d.get("axes", {}).items()},
        )
        return impl, tuple(str(a) for a in d["args"])


# ── Dual pair: Read ↔ Write ─────────────────────────────────────────────────


@dataclass(frozen=True)
class Read:
    """Data-dependent read: output[i] = data[indices[i]]. Adjoint of Write."""

    pattern: str

    @classmethod
    def from_dict(cls, d: dict[str, Any], _ot: Tensor | None) -> tuple[Read, tuple[str, ...]]:
        return cls(pattern=d["pattern"]), tuple(str(a) for a in d["args"])


@dataclass(frozen=True)
class Write:
    """Data-dependent write: combine source into template at indices, by reduction."""

    pattern: str
    reduction: str

    @classmethod
    def from_dict(cls, d: dict[str, Any], _ot: Tensor | None) -> tuple[Write, tuple[str, ...]]:
        impl = cls(pattern=d["pattern"], reduction=d["reduction"])
        return impl, tuple(str(a) for a in d["args"])


# ── Derived: Contract ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class Contract:
    """Bilinear contraction (einsum). Derived: map[mul] ; fold[sum]."""

    pattern: str

    @classmethod
    def from_dict(cls, d: dict[str, Any], _ot: Tensor | None) -> tuple[Contract, tuple[str, ...]]:
        return cls(pattern=d["pattern"]), tuple(str(a) for a in d["args"])


# ── Op ──────────────────────────────────────────────────────────────────────

type OpType = View | Map | Fold | Tile | Read | Write | Contract | Literal | Random | Iota


@dataclass(frozen=True)
class Op:
    """One catform instruction: a typed impl + its inputs/output names."""

    _IMPL_BY_KIND: ClassVar[dict[str, Any]] = {
        "view": View,
        "map": Map,
        "fold": Fold,
        "tile": Tile,
        "read": Read,
        "write": Write,
        "contract": Contract,
        "literal": Literal,
        "random": Random,
        "iota": Iota,
    }

    name: str
    impl: OpType
    inputs: tuple[str, ...]
    output: str
    out_type: Tensor | None = None  # output type annotation (shape resolution for Tile)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Op:
        out = d["outputs"][0]
        ot = cls._out_type(d)
        impl, inputs = cls._IMPL_BY_KIND[d["kind"]].from_dict(d, ot)
        return cls(name=out, impl=impl, inputs=inputs, output=out, out_type=ot)

    @staticmethod
    def _out_type(d: dict[str, Any]) -> Tensor | None:
        """The first non-None output_types entry, as a Tensor; else None."""
        match d.get("output_types") or [None]:
            case [dict() as t, *_]:
                return Tensor.from_dict(t)
            case _:
                return None


# ── Function, Module ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Param:
    """A named parameter or return: name + type annotation."""

    name: str
    ty: Tensor


@dataclass(frozen=True)
class Function:
    name: str
    params: tuple[Param, ...]
    returns: tuple[Param, ...]
    ops: tuple[Op, ...]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Function:
        def _params(key: str) -> tuple[Param, ...]:
            return tuple(Param(p["name"], Tensor.from_dict(p["ty"])) for p in d.get(key, []))

        return cls(
            name=d["name"],
            params=_params("params"),
            returns=_params("returns"),
            ops=tuple(Op.from_dict(op) for op in d["ops"]),
        )


@dataclass(frozen=True)
class Module:
    """A flattened .cat file: dict of functions. Pure data."""

    functions: dict[str, Function] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Module:
        return cls(
            functions={name: Function.from_dict(fn) for name, fn in d["functions"].items()},
        )
