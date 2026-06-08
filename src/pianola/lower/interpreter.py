"""Catform — the loaded, callable .cat program.

Lives in `lower/` because it's the linker + execution dispatcher: it binds
a parsed `Module` (pure AST) with config + weights + backend, and hands
the flat ops to `run` in `pianola.lower.simple`. The AST classes themselves
stay in `pianola.tensors`.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pianola.tensors import Module, Op, Param, Tensor

if TYPE_CHECKING:
    from pianola.lower.simple import Backend


@dataclass(frozen=True)
class State:
    """The cache state: the tokens it reflects + the cached values.

    `cached` records which token ids are currently in the buffer (so a
    prefix check can decide what to re-feed). `cache` is the .cat-side
    `state: *` dict (a `state.seen` scalar plus the per-layer cache
    tensors that get consumed by the model).
    """

    cached: list[int] = field(default_factory=list)
    cache: dict[str, Any] = field(default_factory=dict)


class Catform:
    """A loaded, callable .cat program.

    Mirrors the .cat `main(weights, tokens, state) -> (logits, state)` signature.
    Weights are bound at construction; `__call__(tokens, state)` runs one
    forward pass and returns `(logits, state)`.

        model = Catform(family / "model.cat", size / "config.toml", backend)
        state = model.zero_state()
        logits, state = model([1, 2, 3], state)

    The cache buffer grows automatically inside `__call__` when the incoming
    tokens won't fit. Wildcard inputs (`name: *`) expand to flat per-member
    kwargs internally; callers see only the State wrapper.

    Free shape dims (names in input type annotations that aren't `param.X` and
    aren't yet bound) are inferred from the actual shapes of input tensors —
    `tokens: i32[N]` plus an input of shape `[3]` binds `N=3` automatically.
    """

    @dataclass(frozen=True)
    class _Program:
        """A Catform program at one cache-keys specialization.

        `params` / `returns` are the post-glob-expansion (flat) form used
        internally. `wildcards` lists the original `name: *` wildcard prefixes
        from the .cat signature, used to un-nest dict inputs."""

        params: tuple[Param, ...]
        returns: tuple[Param, ...]
        ops: tuple[Op, ...]
        wildcards: frozenset[str]

    def __init__(
        self,
        model_path: Path,
        config_path: Path,
        backend: Backend,
    ) -> None:
        from pianola.lower.weights import WeightStore, stack_experts

        with open(config_path, "rb") as f:
            config = tomllib.load(f)

        env: dict[str, Any] = {}
        for section in ("shape", "scalar"):
            for name, value in config.get(section, {}).items():
                env[f"param.{name}"] = value

        weights_dir = config_path.parent / "weights"
        if weights_dir.exists() and any(weights_dir.glob("model*.safetensors")):
            store = WeightStore(weights_dir, framework=backend.st_framework)
            for key in store.keys():
                env[key] = backend.to_device(store[key])
            stack_experts(env, backend.stack)

        self.backend = backend
        self.config = config
        self._cat_path = model_path
        self._config_path = config_path
        self._env = env

    def zero_state(self, cache_size: int = 0) -> State:
        """Initial State: empty cached, `state.seen=0`, cache buffers at the
        given seq-axis length (grown lazily inside `__call__` as needed)."""
        shape = self.config["shape"]
        slot = (shape["kv_heads"], cache_size, shape["head_dim"])
        B = self.backend
        cache: dict[str, Any] = {"state.seen": 0}
        for p in self.params:
            if p.name.startswith(("state.k.", "state.v.")):
                cache[p.name] = B.zeros(slot, B.get_dtype(p.ty.dtype))
        return State(cached=[], cache=cache)

    def reserve(self, state: State, total: int) -> State:
        """Grow the cache buffer once to fit `total` tokens total. Called by
        `complete` before the generation loop so each forward pass writes
        into an already-sized buffer (no per-step resize cascade)."""
        return State(cached=state.cached, cache=self._grow_to(state.cache, total))

    def _grow_to(self, cache: dict[str, Any], needed: int) -> dict[str, Any]:
        """Grow every cache buffer's seq-axis to fit `needed` tokens. Scalars
        pass through. Returns the same dict when buffers are already big
        enough — no copy."""
        is_buf = lambda k: k.startswith(("state.k.", "state.v."))  # noqa: E731
        sample = next((v for k, v in cache.items() if is_buf(k)), None)
        if sample is None or int(sample.shape[1]) >= needed:
            return cache
        return {
            k: (self.backend.resize_axis(v, 1, needed) if is_buf(k) else v)
            for k, v in cache.items()
        }

    @cache
    def _get_program(self, cache_keys: frozenset[str]) -> Catform._Program:
        from catform import load_flat

        flat = load_flat(
            str(self._cat_path),
            str(self._config_path),
            entry="main",
            cache_keys=list(cache_keys),
        )
        main = Module.from_dict(flat).functions["main"]
        produced = {op.output for op in main.ops}
        referenced = {i for op in main.ops for i in op.inputs}
        # Type table for wildcard members: any op output's type is the
        # canonical type for that SSA name (input occurrences of the same
        # name share the type — SSA preserves it across loop threading).
        types = {op.output: op.out_type for op in main.ops if op.out_type is not None}
        wildcards = frozenset(p.name for p in (*main.params, *main.returns) if p.ty.dtype == "*")
        env_keys = set(self._env)
        return Catform._Program(
            params=self._expand_globs(main.params, pool=referenced, exclude=env_keys, types=types),
            returns=self._expand_globs(main.returns, pool=produced, exclude=set(), types=types),
            ops=main.ops,
            wildcards=wildcards,
        )

    @property
    def params(self) -> tuple[Param, ...]:
        """Params of the cold flatten — the request-side inputs the user provides."""
        return self._get_program(frozenset()).params

    @property
    def returns(self) -> tuple[Param, ...]:
        """Returns of the cold flatten — includes `cache.X` outputs (init produces them)."""
        return self._get_program(frozenset()).returns

    def __call__(self, tokens: list[int], state: State) -> tuple[Any, State]:
        """Run `main(weights, tokens, state) → (logits, state)`. Grows the cache
        buffer to fit `len(state.cached) + len(tokens)` if needed."""
        cache = self._grow_to(state.cache, len(state.cached) + len(tokens))
        out = self._dispatch(**cache, tokens=tokens)
        new_cache = {k: out[k] for k in cache}
        return out["logits"], State(cached=state.cached + list(tokens), cache=new_cache)

    def _dispatch(self, **inputs: Any) -> dict[str, Any]:
        """Internal: take **kwargs (a flat or wildcard-nested input map), run
        main(), return the output dict. The kwargs-shaped surface for callers
        that don't use the State wrapper (e.g. correctness tests probing a
        single prefill)."""
        from pianola.lower.simple import run

        # Peek at wildcards via the cold flatten so we can un-nest dict inputs
        # before computing cache_keys (which may include unpacked member names).
        cold = self._get_program(frozenset())
        flat_inputs = self._flatten_dict_inputs(inputs, cold.wildcards)

        cache_keys = frozenset(
            k.removeprefix("cache.") for k in flat_inputs if k.startswith("cache.")
        )
        prog = self._get_program(cache_keys)

        # Auto-convert Python ints/lists/ranges to backend tensors via param dtypes.
        dtypes = {p.name: p.ty.dtype for p in prog.params}
        B = self.backend
        flat_inputs = {
            n: (
                B.array(v, B.get_dtype(dtypes[n]))
                if isinstance(v, (list, tuple, range)) and dtypes.get(n, "*") != "*"
                else v
            )
            for n, v in flat_inputs.items()
        }

        # Shape inference: bind free dims from input tensor shapes.
        bindings = self._infer_free_dims(flat_inputs, prog.params)

        env = {**self._env, **bindings, **flat_inputs}
        run(prog.ops, env, self.backend)
        return {r.name: env[r.name] for r in prog.returns}

    @staticmethod
    def _flatten_dict_inputs(inputs: dict[str, Any], wildcards: frozenset[str]) -> dict[str, Any]:
        """Un-nest wildcard inputs: `state={"k.0": t, ...}` → `state.k.0=t, ...`.

        Non-wildcard kwargs pass through unchanged."""
        flat: dict[str, Any] = {}
        for name, value in inputs.items():
            if name in wildcards and isinstance(value, dict):
                for member, member_value in value.items():
                    flat[f"{name}.{member}"] = member_value
            else:
                flat[name] = value
        return flat

    def _infer_free_dims(self, inputs: dict[str, Any], params: tuple[Param, ...]) -> dict[str, int]:
        """Bind free dims (named-str shape elements not in `self._env`) from
        the actual shapes of input tensors. Errors on conflicting bindings."""
        bindings: dict[str, int] = {}
        for p in params:
            if p.name not in inputs:
                continue
            tensor = inputs[p.name]
            if not hasattr(tensor, "shape"):
                continue
            for axis, dim in enumerate(p.ty.shape):
                if not isinstance(dim, str) or dim in self._env or dim.startswith("param."):
                    continue
                actual = int(tensor.shape[axis])
                prior = bindings.get(dim)
                if prior is not None and prior != actual:
                    raise ValueError(
                        f"Conflicting bindings for free dim {dim!r}: "
                        f"{prior} (earlier) vs {actual} (from {p.name})"
                    )
                bindings[dim] = actual
        return bindings

    @staticmethod
    def _expand_globs(
        params: tuple[Param, ...],
        *,
        pool: set[str],
        exclude: set[str],
        types: dict[str, Tensor],
    ) -> tuple[Param, ...]:
        """Expand `name: *` glob params into concrete `name.X` members from `pool`.

        A glob param (`ty.dtype == "*"`) stands for "every dotted member of `name`
        that the ops mention." Members in `exclude` are dropped (weights live in
        the static env, so they should not appear as request-side params). Each
        expanded member's type comes from `types[member]` (built from op outputs);
        unknown members fall back to `bf16[]` for back-compat.
        """
        fallback = Tensor(dtype="bf16", shape=())
        out: list[Param] = []
        for p in params:
            if p.ty.dtype != "*":
                out.append(p)
                continue
            base = p.name + "."
            for member in sorted(n for n in pool if n.startswith(base) and n not in exclude):
                out.append(Param(name=member, ty=types.get(member, fallback)))
        return tuple(out)
