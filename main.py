"""CLI entry point.

Two surfaces in one binary:
  - Model-running commands (generate / serve / repl) dispatch to `pianola.util.api`.
  - Catform tooling (run / fmt / flatten / check / populate) lives here.

Usage:
  uv run main.py generate qwen3/0_6b "It was the best of times,"
  uv run main.py serve    qwen3/0_6b
  uv run main.py repl
  uv run main.py run book/book.cat dot
  uv run main.py populate qwen3/0_6b
  uv run main.py fmt models/qwen3/model.cat
  uv run main.py flatten qwen3/0_6b
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from catform import format_cat, load_flat
from pianola import MODELS_DIR
from pianola.lower import Framework, default_device, get_backend, run
from pianola.tensors import Op
from pianola.util.api import generate_handler, repl_handler, serve_handler

app = typer.Typer()


def _fmt(name: str, val: Any) -> str:
    """Pretty-print `name = value` for the `run` command: 2D values get column-aligned rows."""
    v = val.tolist() if hasattr(val, "tolist") else val
    match v:
        case [list(), *_]:
            prefix = f"{name} = ["
            pad = " " * len(prefix)
            return prefix + (",\n" + pad).join(str(row) for row in v) + "]"
        case _:
            return f"{name} = {v}"


# ── Model-running commands (envelopes over pianola.util.api) ─────────────────


@app.command("generate")
def generate_cmd(
    model_path: str,
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    framework: Framework = Framework.torch,
) -> None:
    """Generate text. argv → Request → respond → stdout."""
    generate_handler(model_path, prompt, max_tokens, temperature, top_k, top_p, framework)


@app.command("serve")
def serve_cmd(
    model: str,
    framework: Framework = Framework.torch,
    port: int = 8000,
) -> None:
    """Start the OpenAI-compatible HTTP server.

    POST /v1/chat/completions on http://localhost:{port}.
    """
    serve_handler(model, framework, port)


@app.command("repl")
def repl_cmd() -> None:
    """Open an interactive catform REPL."""
    repl_handler()


# ── Catform tooling ──────────────────────────────────────────────────────────


@app.command("run")
def run_cmd(args: list[str]) -> None:
    """Run a catform function from a .cat file.

    Usage: run <cat_file> <function> [args...]
    """
    if len(args) < 2:
        print("Usage: run <cat_file> <function> [args...]")
        raise typer.Exit(1)

    cat_file = Path(args[0])
    fn_name = args[1]
    cli_args = args[2:]

    from catform import infer_axes, parse_file

    parsed = parse_file(str(cat_file))

    # Substitute CLI arg values into type dims before infer_axes
    fn_pre = parsed["functions"][fn_name]
    if cli_args:
        arg_values: dict[str, int] = {}
        for param, arg in zip(fn_pre["params"], cli_args):
            dtype_str = param["ty"]["dtype"]
            if dtype_str in ("i32",):
                arg_values[param["name"]] = int(arg)
        for op in fn_pre["ops"]:
            for ot in op["output_types"]:
                if ot is None:
                    continue
                ot["shape"] = [
                    arg_values[d] if isinstance(d, str) and d in arg_values else d
                    for d in ot["shape"]
                ]

    parsed = infer_axes(parsed)
    fn = parsed["functions"][fn_name]

    ops = tuple(Op.from_dict(op) for op in fn["ops"])

    device = default_device()
    B = get_backend(Framework.torch, device=device)

    env: dict[str, Any] = {}
    for param, arg in zip(fn["params"], cli_args):
        dtype_str = param["ty"]["dtype"]
        val = int(arg) if dtype_str in ("i32",) else float(arg)
        env[param["name"]] = B.array(val, B.get_dtype(dtype_str))

    run(ops, env, B)

    for ret in fn["returns"]:
        name = ret["name"]
        if name in env:
            print(_fmt(name, env[name]))


@app.command()
def fmt(files: list[Path], check: bool = False):
    """Format .cat files. --check exits 1 if any file would change."""
    import catform

    changed = False
    for path in files:
        source = path.read_text()
        formatted = catform.fmt_source(source)
        if source != formatted:
            if check:
                print(f"Would reformat: {path}")
                changed = True
            else:
                path.write_text(formatted)
                print(f"Formatted: {path}")
        else:
            print(f"Already formatted: {path}")

    if check and changed:
        raise typer.Exit(code=1)


@app.command("flatten")
def flatten_cmd(model: str, entry: str = "main", emit_cat: bool = False):
    """Flatten a catform module and print the op list.

    MODEL is family/size (e.g. qwen3/0_6b). Size needed for param-sensitive
    unrolling (e.g. layer count). --emit-cat writes flat.cat next to model.cat.
    """
    family, size = model.split("/")
    family_dir = MODELS_DIR / family
    size_dir = family_dir / size

    flat = load_flat(family_dir / "model.cat", size_dir / "config.toml", entry)
    ops = flat["functions"][entry]["ops"]

    if emit_cat:
        out = size_dir / "flat.cat"
        out.write_text(format_cat(flat))
        print(f"Wrote {len(ops)} ops to {out}")
    else:
        print(f"{len(ops)} ops from {entry}()\n")
        for op in ops:
            print(f"  {op['outputs'][0]} = {op['kind']}({', '.join(str(a) for a in op['args'])})")


@app.command("check")
def check_cmd(files: list[Path]):
    """Type-check .cat files and emit environment constraints."""
    import catform

    all_ok = True
    for path in files:
        result = catform.check_file(path)
        errors = result["errors"]
        constraints = result["constraints"]

        if errors:
            all_ok = False
            print(f"{path}:")
            for e in errors:
                print(f"  {e}")
        else:
            print(f"OK: {path}")

        if constraints["params"]:
            print(f"  params: {', '.join(sorted(constraints['params']))}")
        for fn_name, weights in sorted(constraints["weights"].items()):
            for wpath, wtype in sorted(weights.items()):
                dims = " × ".join(str(d) if isinstance(d, int) else d for d in wtype["shape"])
                print(f"  {fn_name}: {wpath} : {wtype['dtype']}[{dims}]")

    if not all_ok:
        raise typer.Exit(code=1)


@app.command("populate")
def populate_cmd(model: str = typer.Argument("")):
    """Fetch weights, tokenizer, and config from HuggingFace.

    Pass family/size (e.g. qwen3/0_6b) for one model, or just a family
    name (e.g. qwen3) to populate all sizes in the family.
    """
    from pianola.util.populate import REGISTRY, all_keys, populate

    if not model:
        print(f"Available: {', '.join(all_keys())}")
        raise typer.Exit(1)

    keys = [f"{model}/{s}" for s in REGISTRY[model]] if model in REGISTRY else [model]
    for key in keys:
        populate(key)

    print("\nDone.")


if __name__ == "__main__":
    app()
