"""The Inference matryoshka.

infer    : Context → Message  = template ; generate ; parse
generate : Text    → Text     = encode   ; complete ; decode
complete : Tokens  → Tokens   # autoregressively sample model
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from itertools import takewhile
from typing import Any

from tokenizers import Tokenizer

from pianola.lower import Backend
from pianola.lower.interpreter import Catform, State

type Token = int


@dataclass(frozen=True)
class ToolCall:
    """A request from the assistant to invoke a tool."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Tool:
    """A function the model may invoke."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)  # JSON Schema


@dataclass(frozen=True)
class Message:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()  # assistant only: tool invocations from the model
    tool_call_id: str = ""  # tool only: which call this answers


@dataclass(frozen=True)
class Context:
    """The structured input the harness sends: conversation + tool catalog."""

    messages: list[Message]
    tools: tuple[Tool, ...] = ()


# fmt: off
def infer(
    template: Template,
    tokenizer: Tokenizer,
    sampler: Sampler,
    model: Catform,
    context: Context,
    state: State,
) -> tuple[Message, State]:
    """infer = template ; generate ; parse."""
    text             = template(context)                                    # Context -> str
    inference, state = generate(tokenizer, sampler, model, text, state)     # str     -> str
    return template.parse(context, inference), state                        # str     -> Message


def generate(
    tokenizer: Tokenizer,
    sampler: Sampler,
    model: Catform,
    context: str,
    state: State,
) -> tuple[str, State]:
    """generate = encode ; complete ; decode."""
    tokens            = tokenizer.encode(context).ids                # str       -> list[int]
    completion, state = complete(sampler, model, tokens, state)      # list[int] -> list[int]
    return tokenizer.decode(completion), state                       # list[int] -> str


def complete(
    sampler: Sampler,
    model: Catform,
    context: list[int],
    state: State,
) -> tuple[list[int], State]:
    """Autoregressive completion."""
    context, state = prefix(context, state)
    completion = []
    while sampler.keep_generating(completion):
        logits, state = model(context, state)
        token = sampler.sample(logits)
        completion.append(token)
        context = [token]
    return completion, state
# fmt: on


# ── Prefix ───────────────────────────────────────────────────────────────────


def prefix(context: list[int], state: State) -> tuple[list[int], State]:
    """Truncate state.cached to the longest common prefix with context;
    return the uncached suffix and the truncated state."""
    matched = takewhile(lambda ab: ab[0] == ab[1], zip(state.cached, context))
    cached = [a for a, _ in matched]
    suffix = context[len(cached) :]
    return suffix, State(cached=cached, cache={**state.cache, "state.seen": len(cached)})


# ── Sampler ──────────────────────────────────────────────────────────────────


class Sampler:
    """Logits → probability distribution → token.

    Two surfaces:
      Loop:        sampler.sample(logits) → token id (one-shot, for `complete`)
      Interactive: Sampler(logits, ...) → inspect → .topk()/.topp() → .pick()
    """

    def __init__(
        self,
        logits: Any = None,
        *,
        B: Backend,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        max_tokens: int = 128,
        eos_tokens: frozenset[Token] = frozenset(),
    ) -> None:
        self.B = B
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.eos_tokens = eos_tokens
        self.dist: Any = None if logits is None else B.softmax(logits, temperature)

    def keep_generating(self, response: list[Token]) -> bool:
        if len(response) >= self.max_tokens:
            return False
        if response and response[-1] in self.eos_tokens:
            return False
        return True

    def sample(self, logits: Any) -> Token:
        """One-shot: logits → token id."""
        if self.temperature == 0:
            return self.B.argmax(logits)
        self.dist = self.B.softmax(logits, self.temperature)
        if self.top_k > 0:
            self.topk(self.top_k)
        if self.top_p < 1.0:
            self.topp(self.top_p)
        return self.B.multinomial(self.dist)

    def topk(self, k: int) -> None:
        """Keep only the k highest-probability tokens; renormalize."""
        vals, _ = self.B.topk(self.dist, k)
        threshold = vals[-1]
        self.dist = self.B.where(self.dist >= threshold, self.dist, 0.0)
        self.dist = self.dist / self.dist.sum()

    def topp(self, p: float) -> None:
        """Keep the smallest set whose cumulative probability exceeds p; renormalize."""
        sp, si = self.B.sort_desc(self.dist)
        keep = self.B.cumsum(sp) - sp <= p
        sp = self.B.where(keep, sp, 0.0)
        self.dist = self.B.scatter(self.B.zeros(self.dist.shape, self.dist.dtype), si, sp)
        self.dist = self.dist / self.dist.sum()


# ── Template ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Template:
    """Chat template: Jinja2 source from tokenizer_config.json."""

    source: str

    def __call__(self, context: Context) -> str:
        from jinja2 import BaseLoader, Environment

        env = Environment(loader=BaseLoader())
        tmpl = env.from_string(self.source)
        messages = [self._jinja_message(m) for m in context.messages]
        tools_data = [self._jinja_tool(t) for t in context.tools]
        return tmpl.render(messages=messages, tools=tools_data, add_generation_prompt=True)

    def parse(self, context: Context, response: str) -> Message:
        """Extract <tool_call>...</tool_call> blocks. Content is the residue."""
        tool_call = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
        parsed = [json.loads(raw) for raw in tool_call.findall(response)]
        return Message(
            role="assistant",
            content=tool_call.sub("", response).strip(),
            tool_calls=tuple(
                ToolCall(id=f"call_{i}", name=d["name"], arguments=d["arguments"])
                for i, d in enumerate(parsed)
            ),
        )

    @staticmethod
    def _jinja_message(m: Message) -> dict[str, Any]:
        return {
            "role": m.role,
            "content": m.content,
            "tool_calls": [{"name": tc.name, "arguments": tc.arguments} for tc in m.tool_calls],
            "tool_call_id": m.tool_call_id,
        }

    @staticmethod
    def _jinja_tool(t: Tool) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
        }
