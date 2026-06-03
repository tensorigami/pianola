"""Populate models from HuggingFace: fetch weights, tokenizer, generate config.toml.

The REGISTRY is the single source of truth for every model Pianola supports.
Each entry is a ModelSpec that carries enough structure for the fetcher and
config generator to operate without runtime discovery or branching.

Weight maps are flat: catform path → HF safetensors key. The tree structure
for config.toml is derived from model.cat's call hierarchy, not maintained
manually.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import tomli_w
import torch
from huggingface_hub import hf_hub_download, snapshot_download

import catform
from pianola import MODELS_DIR

# ── Weight tree derivation ───────────────────────────────────────────────────


def _weight_paths(module: dict[str, Any]) -> set[str]:
    """Extract global weight leaf paths from a parsed .cat module.

    Walks the call hierarchy starting from main, following * params through
    call/loop sites. Returns paths without the 'weights.' prefix, matching
    the flat weight map keys.
    """
    paths: list[str] = []
    _walk_weights(module["functions"], "main", "", paths)
    return set(paths)


def _walk_weights(functions: dict[str, Any], fn_name: str, prefix: str, paths: list[str]) -> None:
    fn = functions[fn_name]
    # Only the conventionally-named `weights` dict carries weight paths.
    # Other `*` params (like `state`) are runtime data, not weights.
    weight_params = {
        p["name"] for p in fn["params"] if p["ty"]["dtype"] == "*" and p["name"] == "weights"
    }

    for op in fn["ops"]:
        if op["kind"] not in ("call", "loop"):
            continue
        callee = functions.get(op["target"])
        if callee is None:
            continue

        for i, arg in enumerate(op["args"]):
            if not isinstance(arg, str):
                continue
            base = arg.split(".")[0]
            if base not in weight_params:
                continue

            local = arg[len(base) + 1 :] if len(arg) > len(base) else ""
            # `*` segments are loop-iteration markers (`weights.layer.*`);
            # the weight map uses the un-starred per-layer template.
            local = ".".join(s for s in local.split(".") if s != "*")
            full = f"{prefix}.{local}" if prefix else local

            # A `weights.X` dict at the LAST arg position fans out onto every
            # trailing callee param (the "dict is the trailing arg" convention
            # used by `weights.attn` → `wq, wk, wv, …`). At a non-last position
            # the dict consumes only the one matching callee param; subsequent
            # params are consumed by subsequent args.
            if i == len(op["args"]) - 1:
                remaining = callee["params"][i:]
            else:
                remaining = callee["params"][i : i + 1]

            for param in remaining:
                leaf = full if len(remaining) == 1 else f"{full}.{param['name']}"
                if param["ty"]["dtype"] == "*":
                    if param["name"] == "weights":
                        _walk_weights(functions, op["target"], leaf, paths)
                    # non-weights `*` params (state) are not weight paths — skip
                else:
                    paths.append(leaf)
            break


def _unflatten(flat: dict[str, str]) -> dict[str, Any]:
    """Convert flat dotted paths to nested dict for TOML."""
    tree: dict[str, Any] = {}
    for path, value in sorted(flat.items()):
        parts = path.split(".")
        d = tree
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return tree


# ── Spec ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RopeSpec:
    """Which HF config keys describe RoPE for this model."""

    head_dim_key: str = "head_dim"
    theta_key: str = "rope_theta"
    scaling_key: str | None = None  # None = plain RoPE, str = has scaling


# Flat weight maps: catform path → HF safetensors key.
# Tree structure is derived from model.cat, not maintained here.

_QWEN3_WEIGHT_MAP: dict[str, str] = {
    "embedding": "model.embed_tokens.weight",
    "lm_head": "lm_head.weight",
    "transformer.norm": "model.norm.weight",
    "transformer.layer.ln1": "model.layers.{}.input_layernorm.weight",
    "transformer.layer.ln2": "model.layers.{}.post_attention_layernorm.weight",
    "transformer.layer.attn.wq": "model.layers.{}.self_attn.q_proj.weight",
    "transformer.layer.attn.wk": "model.layers.{}.self_attn.k_proj.weight",
    "transformer.layer.attn.wv": "model.layers.{}.self_attn.v_proj.weight",
    "transformer.layer.attn.wo": "model.layers.{}.self_attn.o_proj.weight",
    "transformer.layer.attn.q_norm": "model.layers.{}.self_attn.q_norm.weight",
    "transformer.layer.attn.k_norm": "model.layers.{}.self_attn.k_norm.weight",
    "transformer.layer.ffn.gate": "model.layers.{}.mlp.gate_proj.weight",
    "transformer.layer.ffn.up": "model.layers.{}.mlp.up_proj.weight",
    "transformer.layer.ffn.down": "model.layers.{}.mlp.down_proj.weight",
}

_QWEN3_MOE_WEIGHT_MAP: dict[str, str] = {
    "embedding": "model.embed_tokens.weight",
    "lm_head": "lm_head.weight",
    "transformer.norm": "model.norm.weight",
    "transformer.layer.ln1": "model.layers.{}.input_layernorm.weight",
    "transformer.layer.ln2": "model.layers.{}.post_attention_layernorm.weight",
    "transformer.layer.attn.wq": "model.layers.{}.self_attn.q_proj.weight",
    "transformer.layer.attn.wk": "model.layers.{}.self_attn.k_proj.weight",
    "transformer.layer.attn.wv": "model.layers.{}.self_attn.v_proj.weight",
    "transformer.layer.attn.wo": "model.layers.{}.self_attn.o_proj.weight",
    "transformer.layer.attn.q_norm": "model.layers.{}.self_attn.q_norm.weight",
    "transformer.layer.attn.k_norm": "model.layers.{}.self_attn.k_norm.weight",
    "transformer.layer.ffn.router": "model.layers.{}.mlp.gate.weight",
    "transformer.layer.ffn.gate": "model.layers.{}.mlp.gate_proj.weight",
    "transformer.layer.ffn.up": "model.layers.{}.mlp.up_proj.weight",
    "transformer.layer.ffn.down": "model.layers.{}.mlp.down_proj.weight",
}


@dataclass(frozen=True)
class ModelSpec:
    """Everything we need to know about a model before fetching."""

    hf_id: str
    ffn_key: str = "intermediate_size"
    experts_key: str | None = None
    rope: RopeSpec = RopeSpec()
    weight_map: dict[str, str] = field(default_factory=lambda: dict(_QWEN3_WEIGHT_MAP))


# ── Registry ─────────────────────────────────────────────────────────────────
# family → size → spec. Add new models here.

REGISTRY: dict[str, dict[str, ModelSpec]] = {
    "qwen3": {
        "0_6b": ModelSpec(
            hf_id="Qwen/Qwen3-0.6B",
        ),
        "1_7b": ModelSpec(
            hf_id="Qwen/Qwen3-1.7B",
        ),
    },
    "qwen3_moe": {
        "30b_a3b": ModelSpec(
            hf_id="Qwen/Qwen3-30B-A3B",
            ffn_key="moe_intermediate_size",
            experts_key="num_experts",
            weight_map=_QWEN3_MOE_WEIGHT_MAP,
        ),
    },
}


def all_keys() -> list[str]:
    """All registered model keys as family/size strings."""
    return [f"{f}/{s}" for f in REGISTRY for s in REGISTRY[f]]


def get_spec(key: str) -> ModelSpec:
    """Look up a model spec by family/size key."""
    family, size = key.split("/")
    return REGISTRY[family][size]


# ── Config types ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ShapeConfig:
    """Catform shape parameters."""

    hidden: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    ffn_dim: int
    vocab: int
    rope_dim: int
    max_seq: int
    qd: int
    kd: int
    experts: int | None = None

    @classmethod
    def from_hf(cls, hf_config: dict, spec: ModelSpec) -> ShapeConfig:
        c = hf_config
        return cls(
            hidden=c["hidden_size"],
            layers=c["num_hidden_layers"],
            heads=c["num_attention_heads"],
            kv_heads=c["num_key_value_heads"],
            head_dim=c["head_dim"],
            ffn_dim=c[spec.ffn_key],
            vocab=c["vocab_size"],
            rope_dim=c["head_dim"] // 2,
            max_seq=c["max_position_embeddings"],
            qd=c["num_attention_heads"] * c["head_dim"],
            kd=c["num_key_value_heads"] * c["head_dim"],
            experts=c[spec.experts_key] if spec.experts_key else None,
        )

    def to_dict(self) -> dict[str, int]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass(frozen=True)
class RopeConfig:
    """RoPE frequency parameters."""

    head_dim: int
    theta: float
    factor: float | None = None
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    original_max: int = 8192

    @classmethod
    def from_hf(cls, hf_config: dict, spec: RopeSpec) -> RopeConfig:
        head_dim = hf_config[spec.head_dim_key]
        theta = hf_config[spec.theta_key]
        s = (hf_config.get(spec.scaling_key) if spec.scaling_key else None) or {}
        return cls(
            head_dim=head_dim,
            theta=theta,
            factor=s.get("factor"),
            low_freq_factor=s.get("low_freq_factor", 1.0),
            high_freq_factor=s.get("high_freq_factor", 4.0),
            original_max=s.get("original_max_position_embeddings", 8192),
        )

    def inv_freq(self) -> list[float]:
        """Inverse frequencies: 1/(theta^(2k/d)) for k in [0, d/2), matching HF's f32 path."""
        dim = self.head_dim
        k = torch.arange(0, dim, 2, dtype=torch.int64).to(dtype=torch.float32)
        freqs = (1.0 / (self.theta ** (k / dim))).tolist()

        # Plain RoPE: no scaling applied
        if self.factor is None:
            return freqs

        # Scaled RoPE: piecewise frequency adjustment
        low_wavelen = self.original_max / self.low_freq_factor
        high_wavelen = self.original_max / self.high_freq_factor

        scaled: list[float] = []
        for freq in freqs:
            wavelen = 2 * math.pi / freq
            if wavelen < high_wavelen:
                scaled.append(freq)
            elif wavelen > low_wavelen:
                scaled.append(freq / self.factor)
            else:
                smooth = (self.original_max / wavelen - self.low_freq_factor) / (
                    self.high_freq_factor - self.low_freq_factor
                )
                scaled.append((1 - smooth) * freq / self.factor + smooth * freq)

        return scaled


def _read_eos(hf_config: dict) -> tuple[int, ...]:
    """Read EOS token ID(s) from HuggingFace config.json."""
    eos = hf_config.get("eos_token_id")
    match eos:
        case int():
            return (eos,)
        case list():
            return tuple(eos)
        case _:
            return ()


@dataclass(frozen=True)
class CatformConfig:
    """Complete config.toml content for a model size."""

    shape: ShapeConfig
    rms_norm_eps: float
    attn_scale: float
    rope_base: float
    rope_step: float
    eos_token_id: tuple[int, ...]
    weights: dict[str, Any]

    @classmethod
    def from_hf(cls, hf_config: dict, spec: ModelSpec) -> CatformConfig:
        shape = ShapeConfig.from_hf(hf_config, spec)
        rope = RopeConfig.from_hf(hf_config, spec.rope)

        return cls(
            shape=shape,
            rms_norm_eps=hf_config.get("rms_norm_eps", 1e-6),
            attn_scale=1.0 / math.sqrt(shape.head_dim),
            rope_base=rope.theta,
            rope_step=rope.theta ** (-2.0 / shape.head_dim),
            eos_token_id=_read_eos(hf_config),
            weights=_unflatten(spec.weight_map),
        )

    def to_toml(self) -> str:
        return tomli_w.dumps(
            {
                "shape": self.shape.to_dict(),
                "scalar": {
                    "rms_norm_eps": self.rms_norm_eps,
                    "attn_scale": self.attn_scale,
                    "rope_base": self.rope_base,
                    "rope_step": self.rope_step,
                },
                "weights": self.weights,
                "tokenizer": {"eos_token_id": list(self.eos_token_id)},
            }
        )


# ── Weight layout ────────────────────────────────────────────────────────────


def _permute_into(src: Path, dst: Path, heads: int, kv_heads: int) -> None:
    """Copy safetensors shards src → dst, relaying weights to catform-native.

    HF stores everything for the `nn.Linear` primitive: 2-D `(out, in)`, with
    the multi-head structure of Q/K/V/O flattened away. Catform's spec carries
    the head axis explicitly and contracts in kissing form. We restore both
    here, once, at fetch — pure data layout, no numerical change — so the
    runtime never reshapes:

      q_proj  (heads·hd, hidden)      → (hidden, heads,    hd)
      k/v_proj (kv·hd, hidden)        → (hidden, kv_heads, hd)
      o_proj  (hidden, heads·hd)      → (hd, heads, hidden)
      embed_tokens (vocab, hidden)    → untouched (read along vocab axis,
                                         unembed contracts the same layout)
      other 2-D (ffn, router)         → transpose (out, in) → (in, out)
      1-D (norms)                     → untouched

    Q/K/V lead with the contracting axis (hidden) so the projection
    kisses: `"... n d, d h e -> ... h n e"`. O leads with head_dim so
    its contract kisses there: `"... h n e, e h d -> ... n d"`.

    Idempotent: callers download a fresh HF copy into `src` each run.
    """
    import re

    import einops
    from safetensors import safe_open
    from safetensors.torch import save_file

    # Some HF repos ship both sharded and single-file checkpoints; the two are
    # not always value-equivalent (e.g. one preserves an explicit lm_head while
    # the other relies on tying). Prefer the sharded set when present.
    all_files = sorted(src.glob("model*.safetensors"))
    sharded = [p for p in all_files if re.match(r"model-\d+-of-\d+\.safetensors$", p.name)]
    paths = sharded or all_files
    for path in paths:
        with safe_open(str(path), framework="pt") as f:
            meta = f.metadata()
            tensors = {k: f.get_tensor(k) for k in f.keys()}
        for k, t in tensors.items():
            if k.endswith(".q_proj.weight"):
                t = einops.rearrange(t, "(h e) d -> d h e", h=heads)
            elif k.endswith((".k_proj.weight", ".v_proj.weight")):
                t = einops.rearrange(t, "(g e) d -> d g e", g=kv_heads)
            elif k.endswith(".o_proj.weight"):
                t = einops.rearrange(t, "d (h e) -> e h d", h=heads)
            elif k.endswith("embed_tokens.weight") or k == "lm_head.weight":
                pass  # keep HF's (vocab, hidden) layout
            elif t.ndim == 2:
                t = einops.rearrange(t, "a b -> b a")
            tensors[k] = t.contiguous()
        save_file(tensors, str(dst / path.name), metadata=meta)


# ── Fetch ────────────────────────────────────────────────────────────────────


def populate(key: str) -> None:
    """Fetch tokenizer, config, and weights for a registered model."""
    spec = get_spec(key)
    family, size = key.split("/")
    family_dir = MODELS_DIR / family
    size_dir = family_dir / size

    # Verify weight map against model.cat
    module = catform.parse_file(family_dir / "model.cat")
    expected = _weight_paths(module)
    provided = set(spec.weight_map.keys())
    missing = expected - provided
    extra = provided - expected
    if missing:
        raise ValueError(f"Weight map missing paths from model.cat: {sorted(missing)}")
    if extra:
        raise ValueError(f"Weight map has extra paths not in model.cat: {sorted(extra)}")

    # Fetch HF config.json
    with tempfile.TemporaryDirectory() as tmp:
        config_path = hf_hub_download(spec.hf_id, "config.json", local_dir=tmp)
        with open(config_path) as f:
            hf_config = json.load(f)

    print(f"\n{spec.hf_id} → {family}/{size}")

    # Tokenizer
    for filename in ("tokenizer.json", "tokenizer_config.json"):
        hf_hub_download(spec.hf_id, filename, local_dir=str(family_dir))
        print(f"  {filename}")

    # config.toml
    size_dir.mkdir(parents=True, exist_ok=True)
    config = CatformConfig.from_hf(hf_config, spec)
    (size_dir / "config.toml").write_text(config.to_toml())
    print("  config.toml")

    # Weights — download raw HF shards to a temp dir, then transpose every
    # 2-D weight into weights_dir so the on-disk format is catform-native.
    weights_dir = size_dir / "weights"
    if weights_dir.exists():
        shutil.rmtree(weights_dir)
    weights_dir.mkdir(parents=True)
    with tempfile.TemporaryDirectory() as tmp:
        snapshot_download(
            spec.hf_id,
            local_dir=tmp,
            allow_patterns=["model*.safetensors"],
        )
        _permute_into(Path(tmp), weights_dir, config.shape.heads, config.shape.kv_heads)
    print("  weights/ (relaid → catform-native, head axis explicit)")

    # Clean up HF hub cache dirs
    for cache_dir in MODELS_DIR.rglob(".cache"):
        shutil.rmtree(cache_dir)
