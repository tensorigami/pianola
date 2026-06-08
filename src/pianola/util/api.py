"""API — the surface that exposes inference to the outside world.

Sits above the pedagogy layer in `pianola.inference`. Defines the
Request/Response wire shapes and the `respond = unpack ; infer ; pack`
chain that bridges them to the book's `infer` pipeline.

Two endpoints share the surface: the CLI handlers (generate / serve /
repl) and the HTTP server (OpenAI-shaped /v1/chat/completions). Each
endpoint chooses a Backend at spin-up time and passes it to `respond`.
No module-level state; nothing crosses the inference boundary other than
Request, Response, and Backend.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from functools import cache
from typing import Any, Literal

from fastapi import FastAPI
from tokenizers import Tokenizer

from pianola import MODELS_DIR
from pianola.inference import (
    Context,
    Message,
    Sampler,
    Session,
    Template,
    Tool,
    ToolCall,
    infer,
)
from pianola.lower import Backend, Framework, default_device, get_backend
from pianola.lower.interpreter import Catform

# ── Request / Response ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class ToolChoice:
    """mode = auto | none | required | specific; `name` set iff mode == specific."""

    mode: Literal["auto", "none", "required", "specific"] = "auto"
    name: str = ""


@dataclass(frozen=True)
class Request:
    model: str
    messages: list[Message]
    # Sampler config
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    max_tokens: int = 128
    # Tools
    tools: tuple[Tool, ...] = ()
    tool_choice: ToolChoice = field(default_factory=ToolChoice)


@dataclass(frozen=True)
class Response:
    choices: list[Message] = field(default_factory=list)
    cached: int = 0


# ── respond = unpack ; infer ; pack ──────────────────────────────────────────


def respond(req: Request, B: Backend) -> Response:
    """respond = unpack ; infer ; pack."""
    template, tokenizer, sampler, session, context = unpack(req, B)
    reply = infer(template, tokenizer, sampler, session, context)
    return pack(reply, session)


def unpack(req: Request, B: Backend) -> tuple[Template, Tokenizer, Sampler, Session, Context]:
    model, tokenizer, template = load(req.model, B)
    sampler = load_sampler(tokenizer, req, B)
    session = the_session(model)
    context = Context(messages=req.messages, tools=req.tools)
    return template, tokenizer, sampler, session, context


def pack(reply: Message, session: Session) -> Response:
    """Message → Response."""
    return Response(choices=[reply], cached=session.cached)


# ── Loading ──────────────────────────────────────────────────────────────────


@cache
def load(model: str, B: Backend) -> tuple[Catform, Tokenizer, Template]:
    """Read a model from disk. Cached: same (model, B) → same loaded objects."""
    family, size = model.split("/")
    family_dir = MODELS_DIR / family
    config_path = family_dir / size / "config.toml"
    with open(family_dir / "tokenizer_config.json") as f:
        chat_template = json.load(f)["chat_template"]
    return (
        Catform(family_dir / "model.cat", config_path, B),
        Tokenizer.from_file(str(family_dir / "tokenizer.json")),
        Template(source=chat_template),
    )


def load_sampler(tokenizer: Tokenizer, req: Request, B: Backend) -> Sampler:
    """Build a Sampler for this request. EOS tokens come from the tokenizer's specials."""
    return Sampler(
        B=B,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        max_tokens=req.max_tokens,
        eos_tokens=frozenset(
            filter(None, (tokenizer.token_to_id(n) for n in ("<|endoftext|>", "<|im_end|>")))
        ),
    )


@cache
def the_session(catform: Catform) -> Session:
    """The session for a loaded catform. Cached → same catform always gives the same Session."""
    return Session(catform)


# ── CLI handlers ─────────────────────────────────────────────────────────────


def generate_handler(
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    framework: Framework,
) -> None:
    B = get_backend(framework, default_device())
    req = Request(
        model=model,
        messages=[Message(role="user", content=prompt)],
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    print(prompt + respond(req, B).choices[-1].content)


def serve_handler(model: str, framework: Framework, port: int) -> None:
    """Start the OpenAI-compatible HTTP server."""
    import uvicorn

    B = get_backend(framework, default_device())
    load(model, B)  # warm the cache
    uvicorn.run(_make_app(B), host="0.0.0.0", port=port)


def repl_handler(framework: Framework = Framework.torch) -> None:
    """Open IPython with the qwen3/0_6b model loaded."""
    from functools import partial

    from pianola.repl import Sampler as ReplSampler
    from pianola.repl import start

    B = get_backend(framework, default_device())
    catform, tokenizer, template = load("qwen3/0_6b", B)

    def forward_pass(tokens: list[int]) -> Any:
        state = {"state.seen": 0, **catform.kv_zeros(len(tokens))}
        out = catform(**state, tokens=tokens)
        return out["logits"]

    start(
        {
            "B": B,
            "catform": catform,
            "session": the_session(catform),
            "tokenizer": tokenizer,
            "template": template,
            "tokenize": lambda text: tokenizer.encode(text).tokens,
            "tokenize_ids": lambda text: tokenizer.encode(text).ids,
            "forward_pass": forward_pass,
            "Sampler": partial(ReplSampler, _wrap=partial(Sampler, B=B), tokenizer=tokenizer),
        }
    )


# ── HTTP API (OpenAI-shaped /v1/chat/completions) ────────────────────────────


def _make_app(B: Backend) -> FastAPI:
    """Build a FastAPI app whose handler closes over the chosen Backend."""
    api = FastAPI(title="Pianola", version="0.1")

    @api.post("/v1/chat/completions")
    def chat_completions(body: dict[str, Any]) -> dict[str, Any]:
        return _to_openai(respond(_from_openai(body), B), body["model"])

    return api


def _from_openai(body: dict[str, Any]) -> Request:
    return Request(
        model=body["model"],
        messages=[_from_openai_message(m) for m in body["messages"]],
        temperature=float(body.get("temperature", 1.0)),
        top_k=int(body.get("top_k", 0)),
        top_p=float(body.get("top_p", 1.0)),
        max_tokens=int(body.get("max_tokens", 128)),
        tools=tuple(
            Tool(
                name=t["function"]["name"],
                description=t["function"].get("description", ""),
                parameters=t["function"].get("parameters", {}),
            )
            for t in body.get("tools", [])
        ),
        tool_choice=_tool_choice_from_wire(body.get("tool_choice", "auto")),
    )


def _from_openai_message(m: dict[str, Any]) -> Message:
    return Message(
        role=m["role"],
        content=m.get("content", "") or "",
        tool_calls=tuple(
            ToolCall(
                id=tc["id"],
                name=tc["function"]["name"],
                arguments=json.loads(tc["function"]["arguments"]),
            )
            for tc in m.get("tool_calls", [])
        ),
        tool_call_id=m.get("tool_call_id", ""),
    )


def _tool_choice_from_wire(tc: Any) -> ToolChoice:
    """OpenAI tool_choice: str | {type: 'function', function: {name: X}} → ToolChoice."""
    match tc:
        case "auto" | "none" | "required" as mode:
            return ToolChoice(mode=mode)
        case {"function": {"name": str(name)}}:
            return ToolChoice(mode="specific", name=name)
        case _:
            return ToolChoice()


def _to_openai(resp: Response, model: str) -> dict[str, Any]:
    reply = resp.choices[-1]  # state-update view: last message is the assistant reply
    return {
        "id": f"chatcmpl-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": _to_openai_message(reply),
                "finish_reason": "tool_calls" if reply.tool_calls else "stop",
            }
        ],
        "cached_tokens": resp.cached,
    }


def _to_openai_message(m: Message) -> dict[str, Any]:
    # OpenAI compat: `tool_calls` / `tool_call_id` are spec-optional and
    # consumers expect them absent (not empty) when unused.
    out: dict[str, Any] = {"role": m.role, "content": m.content or None}
    if m.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            }
            for tc in m.tool_calls
        ]
    if m.tool_call_id:
        out["tool_call_id"] = m.tool_call_id
    return out
