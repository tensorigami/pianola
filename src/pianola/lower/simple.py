"""Lowering — catform's framework boundary and interpreter.

Two concerns living in this file:

  Backend  — a dataclass of opaque callables binding catform ops to one
             physical framework (torch or jax). Factories pull in the
             framework; nothing above this layer ever imports torch or jax.
  run      — the interpreter. Walks a flat op list, dispatches each op to
             Backend, threads the shape-env that lets tile resolve its
             target shape.

Catform (in `pianola.tensors`) hands its ops here and asks for results.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import einops

from pianola.tensors import (
    Contract,
    Fold,
    Iota,
    Literal,
    Map,
    Op,
    Random,
    Read,
    Tensor,
    Tile,
    View,
    Write,
)


class Framework(StrEnum):
    torch = "torch"
    jax = "jax"


@dataclass(frozen=True)
class Backend:
    """Catform ops + weight materialization → a single physical framework."""

    framework: Framework
    st_framework: str  # "pt" / "jax" — for safetensors.safe_open

    # ── Map (scalar functions lifted by map) ──────────────────────────────
    exp: Callable[..., Any]
    exp2: Callable[..., Any]
    log: Callable[..., Any]
    cos: Callable[..., Any]
    sin: Callable[..., Any]
    silu: Callable[..., Any]
    rsqrt: Callable[..., Any]
    where: Callable[..., Any]

    # ── Tensor creation ─────────────────────────────────────────────────
    array: Callable[..., Any]  # (data, dtype) → tensor
    arange: Callable[..., Any]  # (start, stop, dtype) → integer-sequence tensor
    zeros: Callable[..., Any]  # (shape, dtype) → zero tensor

    # ── Dtype translator + random ───────────────────────────────────────
    get_dtype: Callable[[str], Any]  # catform dtype name ("bf16", "i32", ...) → framework dtype
    rand: Callable[..., Any]  # (shape, dtype) → random [0,1) tensor

    # ── Sampling primitives ─────────────────────────────────────────────
    softmax: Callable[..., Any]  # (x, temperature: float) → probs
    argmax: Callable[..., int]  # (x) → Python int
    multinomial: Callable[..., int]  # (probs) → Python int (single draw)
    topk: Callable[..., tuple]  # (x, k) → (values, indices)
    sort_desc: Callable[..., tuple]  # (x) → (sorted_values, sorted_indices)
    cumsum: Callable[..., Any]  # (x) → cumulative sum
    scatter: Callable[..., Any]  # (dest, idx, src) → dest with src written at idx

    # ── Tensor shape ops ────────────────────────────────────────────────
    resize_axis: Callable[..., Any]  # (tensor, axis: int, target_size: int) → resized to target
    write: Callable[..., Any]  # (reduction, values, indices, template, axis) → tensor

    # ── Weight placement ────────────────────────────────────────────────
    to_device: Callable[..., Any]  # tensor → tensor (device placement)
    stack: Callable[..., Any]  # [tensor, ...] → stacked tensor


def _torch_resize_axis(t: Any, axis: int, target_size: int) -> Any:
    import torch.nn.functional as F

    sl = [slice(None)] * t.ndim
    sl[axis] = slice(0, min(t.shape[axis], target_size))
    t = t[tuple(sl)]
    pad = [0] * (2 * t.ndim)
    pad[2 * (t.ndim - 1 - axis) + 1] = max(0, target_size - t.shape[axis])
    return F.pad(t, pad)


def _jax_resize_axis(t: Any, axis: int, target_size: int) -> Any:
    import jax.numpy as jnp

    sl = [slice(None)] * t.ndim
    sl[axis] = slice(0, min(t.shape[axis], target_size))
    t = t[tuple(sl)]
    pad_width = [(0, 0)] * t.ndim
    pad_width[axis] = (0, max(0, target_size - t.shape[axis]))
    return jnp.pad(t, pad_width)


# ── Write dispatch (framework-specific scatter-with-reduction) ───────────────

_TORCH_WRITE: dict[str, Callable[..., Any]] = {
    "sum": lambda out, ax, i, v: out.index_add_(ax, i, v),
    "set": lambda out, ax, i, v: out.index_copy_(ax, i, v),
    "prod": lambda out, ax, i, v: out.index_reduce_(ax, i, v, "prod"),
    "max": lambda out, ax, i, v: out.index_reduce_(ax, i, v, "amax"),
    "min": lambda out, ax, i, v: out.index_reduce_(ax, i, v, "amin"),
}


def _write_torch(reduction: str, values: Any, indices: Any, template: Any, axis: int) -> Any:
    return _TORCH_WRITE[reduction](template.clone(), axis, indices, values)


_JAX_WRITE: dict[str, Callable[..., Any]] = {
    "sum": lambda t, idx, v: t.at[idx].add(v),
    "set": lambda t, idx, v: t.at[idx].set(v),
    "prod": lambda t, idx, v: t.at[idx].mul(v),
    "max": lambda t, idx, v: t.at[idx].max(v),
    "min": lambda t, idx, v: t.at[idx].min(v),
}


def _write_jax(reduction: str, values: Any, indices: Any, template: Any, axis: int) -> Any:
    idx = tuple(slice(None) if i != axis else indices for i in range(template.ndim))
    return _JAX_WRITE[reduction](template, idx, values)


# ── Backend factories ────────────────────────────────────────────────────────


def _torch_backend(device: str = "cpu") -> Backend:
    import torch
    import torch.nn.functional as F

    _torch_dtypes = {
        "f32": torch.float32,
        "f16": torch.float16,
        "bf16": torch.bfloat16,
        "i32": torch.long,
    }

    return Backend(
        framework=Framework.torch,
        st_framework="pt",
        exp=torch.exp,
        exp2=torch.exp2,
        log=torch.log,
        cos=torch.cos,
        sin=torch.sin,
        silu=F.silu,
        rsqrt=torch.rsqrt,
        where=torch.where,
        array=lambda data, dtype: torch.tensor(data, dtype=dtype, device=device),
        arange=lambda start, stop, dtype, _d=device: torch.arange(
            start, stop, dtype=dtype, device=_d
        ),
        zeros=lambda shape, dtype, _d=device: torch.zeros(shape, dtype=dtype, device=_d),
        get_dtype=lambda s, _m=_torch_dtypes: _m.get(s, torch.float32),
        rand=lambda shape, dtype, _d=device: torch.rand(shape, device=_d).to(dtype),
        softmax=lambda x, t: torch.softmax(x.float() / t, dim=-1),
        argmax=lambda x: int(torch.argmax(x).item()),
        multinomial=lambda p: int(torch.multinomial(p, 1).item()),
        topk=lambda x, k: tuple(torch.topk(x, k)),
        sort_desc=lambda x: tuple(torch.sort(x, descending=True)),
        cumsum=lambda x: torch.cumsum(x, dim=-1),
        scatter=lambda dest, idx, src: dest.scatter(0, idx, src),
        resize_axis=_torch_resize_axis,
        write=_write_torch,
        to_device=lambda t: t.to(device),
        stack=torch.stack,
    )


def _jax_backend(device: str = "cpu") -> Backend:
    import jax
    import jax.numpy as jnp

    _jax_dtypes = {
        "f32": jnp.float32,
        "f16": jnp.float16,
        "bf16": jnp.bfloat16,
        "i32": jnp.int32,
    }

    return Backend(
        framework=Framework.jax,
        st_framework="jax",
        exp=jnp.exp,
        exp2=jnp.exp2,
        log=jnp.log,
        cos=jnp.cos,
        sin=jnp.sin,
        silu=jax.nn.silu,
        rsqrt=jax.lax.rsqrt,
        where=jnp.where,
        array=lambda data, dtype: jnp.array(data, dtype=dtype),
        arange=lambda start, stop, dtype: jnp.arange(start, stop, dtype=dtype),
        zeros=lambda shape, dtype: jnp.zeros(shape, dtype=dtype),
        get_dtype=lambda s, _m=_jax_dtypes: _m.get(s, jnp.float32),
        rand=lambda shape, dtype: jax.random.uniform(jax.random.PRNGKey(0), shape, dtype=dtype),
        softmax=lambda x, t: jax.nn.softmax(x.astype(jnp.float32) / t, axis=-1),
        argmax=lambda x: int(jnp.argmax(x)),
        multinomial=lambda p: int(jax.random.choice(jax.random.PRNGKey(0), p.shape[0], p=p)),
        topk=lambda x, k: tuple(jax.lax.top_k(x, k)),
        sort_desc=lambda x: (jnp.sort(x)[::-1], jnp.argsort(x)[::-1]),
        cumsum=lambda x: jnp.cumsum(x),
        scatter=lambda dest, idx, src: dest.at[idx].set(src),
        resize_axis=_jax_resize_axis,
        write=_write_jax,
        to_device=lambda t: t,
        stack=jnp.stack,
    )


_BACKEND_FACTORY: dict[Framework, Callable[[str], Backend]] = {
    Framework.torch: _torch_backend,
    Framework.jax: _jax_backend,
}


def default_device() -> str:
    """Auto-detect best available device: mps → cuda → cpu."""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_backend(framework: Framework, device: str = "cpu") -> Backend:
    factory = _BACKEND_FACTORY.get(framework)
    if factory is None:
        raise ValueError(f"Unknown framework: {framework}")
    return factory(device)


# ── Map dispatch ─────────────────────────────────────────────────────────────

# All Map functions: (Backend, args) → result
_MAP_DISPATCH: dict[str, Callable[[Backend, tuple[Any, ...]], Any]] = {
    # Arithmetic (framework-agnostic)
    "add": lambda _B, a: a[0] + a[1],
    "mul": lambda _B, a: a[0] * a[1],
    "sub": lambda _B, a: a[0] - a[1],
    "div": lambda _B, a: a[0] / a[1],
    "ge": lambda _B, a: a[0] >= a[1],
    "le": lambda _B, a: a[0] <= a[1],
    "pow": lambda _B, a: a[0] ** a[1],
    "square": lambda _B, a: a[0] ** 2,
    "lt1": lambda _B, a: (a[0] < 1.0) * 1.0,
    # Backend-specific
    "exp": lambda B, a: B.exp(a[0]),
    "exp2": lambda B, a: B.exp2(a[0]),
    "log": lambda B, a: B.log(a[0]),
    "cos": lambda B, a: B.cos(a[0]),
    "sin": lambda B, a: B.sin(a[0]),
    "silu": lambda B, a: B.silu(a[0]),
    "rsqrt": lambda B, a: B.rsqrt(a[0]),
    "where": lambda B, a: B.where(a[0], a[1], a[2]),
    # Dtype casts
    "f32": lambda _B, a: a[0].float(),
    "bf16": lambda _B, a: a[0].bfloat16(),
}


# ── Shape environment (resolves Tile target shapes) ──────────────────────────
#
# Tile is the one op that fabricates shape, so it is the one op whose execution
# needs to know runtime sizes its single input doesn't carry. Rather than smuggle
# a second "shape source" tensor, the interpreter threads a `shapes` env: after
# each op runs, dynamic dim names (`N`, `M`, `S`, `D`, …) and the variadic `...`
# are bound from the produced tensor against its output-type annotation. A tile
# then resolves its target from its OWN out_type against that env. Names are
# bound by producers that run earlier in the (contiguous) op order, so every
# size a tile needs is present by the time it runs.

_TOK = re.compile(r"\([^)]*\)|\S+")  # tokenize a pattern side, keeping `(g r)` groups whole


def _wildcard_axis(pattern: str) -> int:
    """Find the position of '_' in a read/write pattern."""
    for side in pattern.split("->"):
        tokens = side.strip().split()
        if "_" in tokens:
            return tokens.index("_")
    raise ValueError(f"No '_' in pattern: {pattern}")


def _names_in(tok: str) -> set[str]:
    """Axis names in a pattern token: `(g r)`→{g,r}, `n`→{n}, `...`/`1`→∅."""
    if tok == "...":
        return set()
    return {p for p in tok.strip("()").split() if not p.isdigit()}


def _bind_shapes(out_type: Tensor | None, tensor: Any, shapes: dict[str, Any]) -> None:
    """Bind dynamic dim names and `...` from a produced tensor's actual shape."""
    if out_type is None or not hasattr(tensor, "ndim"):
        return
    shape = out_type.shape
    actual = tuple(int(s) for s in tensor.shape)
    if "..." in shape:
        i = shape.index("...")
        head, tail = shape[:i], shape[i + 1 :]
        for nm, sz in zip(head, actual[: len(head)]):
            if isinstance(nm, str):
                shapes[nm] = sz
        shapes["..."] = tuple(actual[len(head) : len(actual) - len(tail)])
        for nm, sz in zip(tail, actual[len(actual) - len(tail) :]):
            if isinstance(nm, str):
                shapes[nm] = sz
    else:
        for nm, sz in zip(shape, actual):
            if isinstance(nm, str):
                shapes[nm] = sz


def _resolve_target(shape: tuple[Any, ...], shapes: dict[str, Any]) -> tuple[int, ...]:
    """Resolve an output-type shape to concrete ints, expanding `...` from the env."""
    dims: list[int] = []
    for d in shape:
        if d == "...":
            dims.extend(shapes["..."])
        elif isinstance(d, int):
            dims.append(d)
        else:
            dims.append(int(shapes[d]))
    return tuple(dims)


def _tile(op: Op, args: tuple[Any, ...], shapes: dict[str, Any], backend: Backend) -> Any:
    """Single-input tile → `einops.repeat`. New axes sized from out_type + shape-env;
    a new `...` is materialized into fresh named axes. Multiplicative groups (`(g r)`)
    take their factor from the static `axes` dict."""
    impl = op.impl
    assert isinstance(impl, Tile) and op.out_type is not None
    data = args[0]
    if not hasattr(data, "ndim"):  # python scalar (config param / literal float)
        data = backend.array(data, backend.get_dtype(op.out_type.dtype))

    left_s, right_s = impl.pattern.split("->")
    left, right = _TOK.findall(left_s), _TOK.findall(right_s)
    left_names = {n for t in left for n in _names_in(t)}
    kw: dict[str, int] = {k: int(v) for k, v in impl.axes.items()}  # static (e.g. `r`)
    target = _resolve_target(op.out_type.shape, shapes)

    if "..." in right:
        e = right.index("...")
        before, after = right[:e], right[e + 1 :]
    else:
        before, after = right, []
    before_dims = target[: len(before)]
    after_dims = target[len(target) - len(after) :] if after else ()
    ell_dims = target[len(before) : len(target) - len(after)] if "..." in right else None

    for tok, dim in (*zip(before, before_dims), *zip(after, after_dims)):
        if tok == "..." or tok.startswith("(") or tok.isdigit() or tok in left_names:
            continue  # ellipsis / multiplicative group / literal / passthrough
        kw[tok] = int(dim)

    pattern = impl.pattern
    if ell_dims is not None and "..." not in left:  # new variadic axis → materialize
        names = [f"ax{i}" for i in range(len(ell_dims))]
        kw.update({nm: int(dim) for nm, dim in zip(names, ell_dims)})
        pattern = f"{left_s.strip()} -> {' '.join((*before, *names, *after))}"

    return einops.repeat(data, pattern, **kw)


def _resolve_op(op: Op, backend: Backend) -> Callable[..., Any]:
    """Resolve an Op to a concrete callable. Single source of truth for all dispatch."""
    match op.impl:
        case View(pattern=p, axes=ax):
            return lambda *a, _p=p, _ax=ax: einops.rearrange(a[0], _p, **_ax)
        case Map(function=f):
            fn = _MAP_DISPATCH.get(f)
            if fn is None:
                raise NotImplementedError(f"Unknown map: {f}")
            return lambda *a, _fn=fn, _B=backend: _fn(_B, a)
        case Fold(reduction=r, pattern=p):
            return lambda *a, _r=r, _p=p: einops.reduce(a[0], _p, _r)
        # Tile is handled in the interpreter loop (`_tile`), not here: it is the one
        # op whose target shape depends on the running shape-env, not just its inputs.
        case Read(pattern=p):
            axis = _wildcard_axis(p)
            if axis == 0:
                return lambda *a: a[0][a[1]]

            def _read(*a: Any, _ax: int = axis) -> Any:
                idx = tuple(slice(None) if i != _ax else a[1] for i in range(a[0].ndim))
                return a[0][idx]

            return _read
        case Write(pattern=p, reduction=r):
            axis = _wildcard_axis(p)
            return lambda *a, _ax=axis, _r=r, _B=backend: _B.write(_r, a[0], a[1], a[2], _ax)
        case Contract(pattern=p):
            return lambda *a, _p=p: einops.einsum(a[0], a[1], _p)
        case Literal(value=v, dtype=dt):
            t = backend.array(v, backend.get_dtype(dt))
            return lambda *a, _t=t: _t
        case Iota(start=start, dtype=dt, dims=dims):
            # Ramp [start .. start+S). Symbolic dims arrive as leading int args
            # (in `inputs` order); a named `start` is the trailing arg.
            _ndt = backend.get_dtype(dt)
            _nsym = sum(1 for d in dims if isinstance(d, str))
            _start_named = isinstance(start, str)

            def _iota(
                *a: Any,
                _start: Any = start,
                _dims: Any = dims,
                _ns: int = _nsym,
                _named: bool = _start_named,
                _dt: Any = _ndt,
                _B: Backend = backend,
            ) -> Any:
                syms = iter(int(x) for x in a[:_ns])
                shape = [next(syms) if isinstance(d, str) else int(d) for d in _dims]
                s = int(a[_ns]) if _named else int(_start)
                return _B.arange(s, s + shape[0], _dt)

            return _iota
        case Random(lower=lo, upper=hi, dtype=dt, dims=dims):
            native_dt = backend.get_dtype(dt)
            # Pre-split: positions of concrete dims to insert among symbolic args
            _concrete: list[tuple[int, int]] = []
            for i, d in enumerate(dims):
                match d:
                    case int(size):
                        _concrete.append((i, size))

            _ndt = native_dt

            def _make_random(*a: Any, _lo: float = lo, _hi: float = hi) -> Any:
                shape = [int(x) for x in a]
                for pos, size in _concrete:
                    shape.insert(pos, size)
                return backend.rand(shape, _ndt) * (_hi - _lo) + _lo

            return _make_random
        case _:
            raise NotImplementedError(f"Cannot resolve: {type(op.impl).__name__}")


# ── Interpreter ──────────────────────────────────────────────────────────────


def run(ops: Sequence[Op], env: dict[str, Any], backend: Backend) -> None:
    """Reference interpreter: walk flat op list, dispatch each to backend.

    Threads a `shapes` env so single-input tiles can resolve their target shape
    (see `_tile` / `_bind_shapes`)."""
    shapes: dict[str, Any] = {}
    for op in ops:
        args = tuple(env[name] for name in op.inputs)
        if isinstance(op.impl, Tile):
            env[op.output] = _tile(op, args, shapes, backend)
        else:
            env[op.output] = _resolve_op(op, backend)(*args)
        _bind_shapes(op.out_type, env[op.output], shapes)
