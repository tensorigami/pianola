"""Interactive catform REPL.

Launches IPython with catform ops as callable Python objects.
Catform syntax executes directly as Python:

    x = literal([[1, 2, 3], [4, 5, 6]])
    y = fold["a b -> 1 b", sum](x)
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import einops
import torch

if TYPE_CHECKING:
    from tokenizers import Tokenizer

    from pianola.inference import Sampler as _PureSampler

# ── Helpers ──────────────────────────────────────────────────────────────────

_DTYPE: dict[str, torch.dtype] = {
    "f32": torch.float32,
    "bf16": torch.bfloat16,
    "i32": torch.int32,
}


def _coerce(x: Any) -> torch.Tensor:
    """Auto-convert Python values to tensors."""
    if isinstance(x, torch.Tensor):
        return x
    return torch.tensor(x)


def _resolve_pattern(pattern: str) -> tuple[str, dict[str, int]]:
    """Replace literal numbers in patterns with named axes for einops.

    '(2 b) -> 2 b' becomes ('(d0 b) -> d0 b', {'d0': 2})
    """
    axes: dict[str, int] = {}
    left, right = pattern.split("->")
    counter = 0

    # Left side: each number gets a unique name, tracked by value
    left_names: dict[int, list[str]] = {}

    def _sub_left(m: re.Match[str]) -> str:
        nonlocal counter
        n = int(m.group())
        if n == 1:
            return "1"
        name = f"d{counter}"
        counter += 1
        axes[name] = n
        left_names.setdefault(n, []).append(name)
        return name

    new_left = re.sub(r"\b(\d+)\b", _sub_left, left)

    # Right side: reuse left names by value + occurrence, or create new
    right_seen: dict[int, int] = {}

    def _sub_right(m: re.Match[str]) -> str:
        nonlocal counter
        n = int(m.group())
        if n == 1:
            return "1"
        idx = right_seen.get(n, 0)
        right_seen[n] = idx + 1
        if n in left_names and idx < len(left_names[n]):
            return left_names[n][idx]
        name = f"d{counter}"
        counter += 1
        axes[name] = n
        return name

    new_right = re.sub(r"\b(\d+)\b", _sub_right, right)
    return f"{new_left}->{new_right}", axes


def _wildcard_axis(pattern: str) -> int:
    """Find the position of '_' in a read/write pattern."""
    for side in pattern.split("->"):
        tokens = side.strip().split()
        if "_" in tokens:
            return tokens.index("_")
    raise ValueError(f"No '_' in pattern: {pattern}")


def _index_tuple(ndim: int, axis: int, indices: torch.Tensor) -> tuple[Any, ...]:
    """Build an indexing tuple that reads along one axis."""
    return tuple(indices if i == axis else slice(None) for i in range(ndim))


# ── Introductions ────────────────────────────────────────────────────────────


def literal(value: Any) -> torch.Tensor:
    """Create a tensor from a Python value. Dtype inferred from content."""
    return torch.tensor(value)


@dataclass(frozen=True)
class _Random:
    def __getitem__(self, spec: tuple[float | str, ...]) -> Callable[..., torch.Tensor]:
        lo, hi = float(spec[0]), float(spec[1])
        dtype = str(spec[2]) if len(spec) > 2 else "f32"

        def fn(*shape: int) -> torch.Tensor:
            return torch.empty(*shape, dtype=_DTYPE[dtype]).uniform_(lo, hi)

        return fn

    def __repr__(self) -> str:
        return "random"


random = _Random()


# ── Map dispatch ─────────────────────────────────────────────────────────────

_MAP_FN: dict[str, Callable[..., Any]] = {
    # Arithmetic
    "add": lambda *a: _coerce(a[0]) + _coerce(a[1]),
    "mul": lambda *a: _coerce(a[0]) * _coerce(a[1]),
    "sub": lambda *a: _coerce(a[0]) - _coerce(a[1]),
    "div": lambda *a: _coerce(a[0]) / _coerce(a[1]),
    "ge": lambda *a: _coerce(a[0]) >= _coerce(a[1]),
    "le": lambda *a: _coerce(a[0]) <= _coerce(a[1]),
    "pow": lambda *a: _coerce(a[0]) ** _coerce(a[1]),
    "square": lambda *a: _coerce(a[0]) ** 2,
    "lt1": lambda *a: (_coerce(a[0]) < 1.0).float(),
    # Unary
    "exp": lambda *a: torch.exp(_coerce(a[0])),
    "exp2": lambda *a: torch.exp2(_coerce(a[0])),
    "log": lambda *a: torch.log(_coerce(a[0])),
    "cos": lambda *a: torch.cos(_coerce(a[0])),
    "sin": lambda *a: torch.sin(_coerce(a[0])),
    "silu": lambda *a: torch.nn.functional.silu(_coerce(a[0])),
    "rsqrt": lambda *a: torch.rsqrt(_coerce(a[0])),
    # Ternary
    "where": lambda *a: torch.where(_coerce(a[0]), _coerce(a[1]), _coerce(a[2])),
    # Dtype casts
    "f32": lambda *a: _coerce(a[0]).float(),
    "bf16": lambda *a: _coerce(a[0]).bfloat16(),
    "i32": lambda *a: _coerce(a[0]).int(),
}


# ── Op objects ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _View:
    def __getitem__(self, pattern: str) -> Callable[..., torch.Tensor]:
        p, ax = _resolve_pattern(pattern)

        def fn(x: Any) -> torch.Tensor:
            return einops.rearrange(_coerce(x), p, **ax)

        return fn

    def __repr__(self) -> str:
        return "view"


@dataclass(frozen=True)
class _Map:
    def __getitem__(self, function: str) -> Callable[..., torch.Tensor]:
        fn = _MAP_FN.get(function)
        if fn is None:
            raise KeyError(f"Unknown map function: {function}")
        return fn

    def __repr__(self) -> str:
        return "map"


@dataclass(frozen=True)
class _Fold:
    def __getitem__(self, spec: tuple[str, str]) -> Callable[..., torch.Tensor]:
        pattern, reduction = spec
        p, ax = _resolve_pattern(pattern)

        def fn(x: Any) -> torch.Tensor:
            return einops.reduce(_coerce(x), p, reduction, **ax)

        return fn

    def __repr__(self) -> str:
        return "fold"


@dataclass(frozen=True)
class _Tile:
    def __getitem__(self, pattern: str) -> Callable[..., torch.Tensor]:
        p, ax = _resolve_pattern(pattern)

        def fn(x: Any) -> torch.Tensor:
            return einops.repeat(_coerce(x), p, **ax)

        return fn

    def __repr__(self) -> str:
        return "tile"


@dataclass(frozen=True)
class _Read:
    def __getitem__(self, pattern: str) -> Callable[..., torch.Tensor]:
        def fn(data: Any, indices: Any) -> torch.Tensor:
            d, i = _coerce(data), _coerce(indices)
            axis = _wildcard_axis(pattern)
            return d[_index_tuple(d.ndim, axis, i)]

        return fn

    def __repr__(self) -> str:
        return "read"


_WRITE_FN: dict[str, Callable[..., Any]] = {
    "sum": lambda out, ax, i, v: out.index_add_(ax, i, v),
    "copy": lambda out, ax, i, v: out.index_copy_(ax, i, v),
    "prod": lambda out, ax, i, v: out.index_reduce_(ax, i, v, "prod"),
    "max": lambda out, ax, i, v: out.index_reduce_(ax, i, v, "amax"),
    "min": lambda out, ax, i, v: out.index_reduce_(ax, i, v, "amin"),
}


@dataclass(frozen=True)
class _Write:
    def __getitem__(self, spec: tuple[str, str]) -> Callable[..., torch.Tensor]:
        pattern, reduction = spec
        axis = _wildcard_axis(pattern)
        op = _WRITE_FN[reduction]

        def fn(values: Any, indices: Any, template: Any) -> torch.Tensor:
            v, i, t = _coerce(values), _coerce(indices), _coerce(template)
            return op(t.clone(), axis, i, v)

        return fn

    def __repr__(self) -> str:
        return "write"


@dataclass(frozen=True)
class _Contract:
    def __getitem__(self, pattern: str) -> Callable[..., torch.Tensor]:
        def fn(a: Any, b: Any) -> torch.Tensor:
            return einops.einsum(_coerce(a), _coerce(b), pattern)

        return fn

    def __repr__(self) -> str:
        return "contract"


# ── Singletons ───────────────────────────────────────────────────────────────

view = _View()
fold = _Fold()
tile = _Tile()
read = _Read()
write = _Write()
contract = _Contract()
map = _Map()  # type: ignore[assignment]  # noqa: A001


# ── Catform names (specifier strings) ────────────────────────────────────────

# Each name maps to itself as a string so that `fold["a b -> 1 b", sum]`
# works in the REPL: `sum` resolves to the string "sum", not the builtin.
_NAMES: dict[str, str] = {
    k: k
    for k in (
        "sum",
        "mean",
        "max",
        "min",
        "prod",
        "copy",
        "mul",
        "add",
        "sub",
        "div",
        "exp",
        "exp2",
        "log",
        "cos",
        "sin",
        "silu",
        "rsqrt",
        "pow",
        "square",
        "lt1",
        "ge",
        "le",
        "where",
        "f32",
        "bf16",
        "i32",
    )
}


# ── Sampler wrapper (decoded display) ────────────────────────────────────────


class Sampler:
    """REPL wrapper around `inference.Sampler` that decodes tokens for display.

    The pure Sampler in `pianola.inference` does not know about tokens-as-text;
    that concern lives here so the interactive `repr` and `pick()` can show
    decoded strings. Behaves like the pure Sampler for `sample/topk/topp`.
    """

    def __init__(
        self,
        logits: Any = None,
        *,
        _wrap: Callable[..., _PureSampler],
        tokenizer: Tokenizer,
    ) -> None:
        self.s = _wrap(logits)
        self.tokenizer = tokenizer

    def sample(self, logits: Any) -> int:
        return self.s.sample(logits)

    def topk(self, k: int) -> None:
        self.s.topk(k)

    def topp(self, p: float) -> None:
        self.s.topp(p)

    def pick(self) -> str:
        """Draw one token from the current distribution and decode it."""
        return self.tokenizer.decode([self.s.B.multinomial(self.s.dist)])

    def __repr__(self) -> str:
        if self.s.dist is None:
            return "Sampler(no distribution — pass logits or call .sample())"
        vals, idxs = self.s.B.topk(self.s.dist, 10)
        return "\n".join(
            f"  {float(vals[i]):.4f}  {self.tokenizer.decode([int(idxs[i])])!r}"
            for i in range(10)
            if float(vals[i]) > 0
        )


# ── REPL startup ─────────────────────────────────────────────────────────────


def start(preload: dict[str, Any] | None = None) -> None:
    """Launch IPython with catform ops in the namespace."""
    try:
        from IPython.terminal.embed import InteractiveShellEmbed
    except ImportError:
        print("IPython required: uv add ipython")
        return

    ns: dict[str, Any] = {
        "view": view,
        "map": map,
        "fold": fold,
        "tile": tile,
        "read": read,
        "write": write,
        "contract": contract,
        "literal": literal,
        "random": random,
        "torch": torch,
        "einops": einops,
        **_NAMES,
    }

    if preload:
        ns.update(preload)

    shell = InteractiveShellEmbed(user_ns=ns, banner1="", banner2="")
    shell.enable_tip = False

    # Pretty-print torch tensors as plain Python values
    formatter = shell.display_formatter.formatters["text/plain"]
    formatter.for_type(torch.Tensor, lambda t, p, cycle: p.text(repr(t.tolist())))

    shell()
