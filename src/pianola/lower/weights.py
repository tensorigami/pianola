"""Lazy weight loading from safetensors."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from safetensors import safe_open


class WeightStore:
    """Memory-mapped lazy loading. Zero memory until a weight is accessed."""

    def __init__(self, model_dir: str | Path, framework: str = "pt"):
        self._handles: list[Any] = []
        self._index: dict[str, int] = {}

        for i, path in enumerate(sorted(Path(model_dir).glob("model*.safetensors"))):
            handle = safe_open(str(path), framework=framework)
            self._handles.append(handle)
            for key in handle.keys():
                self._index[key] = i

    def __getitem__(self, name: str) -> Any:
        i = self._index[name]
        return self._handles[i].get_tensor(name)

    def __contains__(self, name: str) -> bool:
        return name in self._index

    def keys(self) -> set[str]:
        return set(self._index.keys())


def stack_experts(env: dict[str, Any], stack: Callable[..., Any]) -> None:
    """Stack per-expert weights into batched tensors.

    HF stores: model.layers.{i}.block_sparse_moe.experts.{j}.w1.weight
    Catform needs: model.layers.{i}.block_sparse_moe.w1.weight [E, ...]

    Detects *.experts.{j}.* keys, pops them, stacks into batched tensor.
    """
    pattern = re.compile(r"^(.+)\.experts\.(\d+)\.(.+)$")
    groups: dict[str, dict[int, str]] = {}  # stacked_key → {expert_idx: original_key}

    for key in list(env.keys()):
        m = pattern.match(key)
        if m:
            prefix, idx, suffix = m.group(1), int(m.group(2)), m.group(3)
            stacked_key = f"{prefix}.{suffix}"
            groups.setdefault(stacked_key, {})[idx] = key

    for stacked_key, expert_keys in groups.items():
        n = max(expert_keys.keys()) + 1
        tensors = [env.pop(expert_keys[i]) for i in range(n)]
        env[stacked_key] = stack(tensors)
