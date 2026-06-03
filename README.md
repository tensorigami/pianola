# Pianola

A player piano for language models. Companion codebase to the book [*Structure and Execution of Language Models*](https://tensorigami.github.io/pianola/).

> **Note:** Pianola runs models end-to-end but does not yet do so performantly. The execution layer prioritizes correctness and pedagogical clarity. Performance work is planned.

## Setup

**1. Clone.**

```bash
git clone https://github.com/tensorigami/pianola.git
cd pianola
```

**2. Install [`uv`](https://docs.astral.sh/uv/)** (manages Python + deps; no separate Python install needed).

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**3. Fetch weights.** Qwen3 is open; no login required.

```bash
uv run main.py populate qwen3        # all sizes (~3 GB)
uv run main.py populate qwen3/0_6b   # just 0.6B (~1.2 GB)
```

**4. Verify.**

```bash
uv run main.py generate qwen3/0_6b "It was the best of times,"
```

## Repository layout

```
├── main.py                       CLI entry point
│
├── src/pianola/
│   ├── interface.py              CLI + HTTP API envelope (shared body)
│   ├── inference.py              ch 1 — infer / generate / complete / Sampler / Session
│   ├── tensors.py                M1 — types, ops, Module AST, Catform runtime shell
│   ├── lower/
│   │   ├── simple.py             Backend dispatch (PyTorch / JAX) + interpreter
│   │   └── weights.py            lazy safetensors loading
│   ├── repl/                     interactive catform REPL
│   └── util/
│       └── populate.py           Hugging Face → config.toml + weights
│
└── models/
    └── qwen3/
        ├── model.cat             architecture spec (shared across sizes)
        ├── tokenizer.json        ┐
        ├── tokenizer_config.json │
        ├── 0_6b/                 │ populated by
        │   ├── config.toml       │ uv run main.py populate qwen3
        │   └── weights/          │
        └── 1_7b/                 │
            ├── config.toml       │
            └── weights/          ┘
```

Models are organized by **family** (architecture, e.g. `qwen3`) and **size** (parameterization, e.g. `0_6b`). The `model.cat` file is the architecture, written by hand and shared across sizes. `config.toml` parameterizes it for a specific size. The catform toolchain (parser, type checker, flattener, formatter) lives in a separate [`catform`](https://github.com/tensorigami/catform) crate and is imported as a Python package.

## CLI

All commands run through `uv run main.py`.

### `generate`

Generate text from a model.

```bash
uv run main.py generate MODEL PROMPT [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--max-tokens` | 128 | Maximum tokens to generate |
| `--temperature` | 0.8 | Sampling temperature |
| `--top-k` | 50 | Top-k filtering |
| `--top-p` | 0.9 | Nucleus sampling threshold |
| `--framework` | `torch` | Backend (`torch`, `jax`); device auto-detected |

### `serve`

Start an OpenAI-compatible HTTP server.

```bash
uv run main.py serve MODEL [--framework torch|jax] [--port 8000]
```

`POST /v1/chat/completions` accepts an OpenAI-shaped request and returns an OpenAI-shaped response, plus a `session_id` field that, when passed back in a follow-up request, reuses the in-process KV cache for the matching prefix (visible in `usage.cached_tokens`).

```bash
uv run main.py serve qwen3/0_6b &

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3/0_6b",
       "messages": [{"role": "user", "content": "Hello"}],
       "max_tokens": 50}'
```

### `repl`

Open an interactive Python REPL with `catform`, `session`, `tokenizer`, `Sampler` preloaded (qwen3/0_6b if weights are present).

```bash
uv run main.py repl
```

### `populate`

Fetch weights, tokenizer, and config from Hugging Face. Pass a family to populate all sizes, or `family/size` for one.

```bash
uv run main.py populate qwen3          # all sizes in the family
uv run main.py populate qwen3/0_6b     # one size
```

### `run`

Run a catform function from a `.cat` file. Args are forwarded as inputs.

```bash
uv run main.py run FILE FUNCTION [ARGS...]
```

```bash
uv run main.py run book/book.cat dot
uv run main.py run book/book.cat ball_vol 5 10000
```

### `fmt`

Format `.cat` files in place. `--check` exits 1 if any file would change.

```bash
uv run main.py fmt FILES... [--check]
```

### `check`

Type-check `.cat` files for shape/dtype consistency. No config needed.

```bash
uv run main.py check FILES...
```

### `flatten`

Flatten a catform module: inline all `call` / `loop` ops into a single straight-line program.

```bash
uv run main.py flatten MODEL [--entry NAME] [--emit-cat]
```

| Option | Default | Description |
|---|---|---|
| `--entry` | `main` | Entry function to flatten from |
| `--emit-cat` | off | Write `flat.cat` next to `model.cat` |
