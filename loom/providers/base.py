"""Provider-neutral model interface.

A provider is anything with a ``complete`` method that turns a system prompt,
a list of neutral messages, and a list of tool schemas into a ``ModelResponse``.
The kernel never imports a vendor SDK -- providers do, lazily.

Neutral message shape (a plain ``dict``):

    {"role": "user",      "content": "<text>"}
    {"role": "assistant", "content": "<text>", "tool_calls": [ToolCall, ...]}
    {"role": "tool",      "tool_call_id": "<id>", "name": "<tool>", "content": "<text>"}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ToolCall:
    """A model's request to call a tool."""

    id: str
    name: str
    input: dict

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "input": self.input}

    @staticmethod
    def from_dict(d: dict) -> "ToolCall":
        return ToolCall(id=d["id"], name=d["name"], input=d.get("input", {}))


@dataclass
class ModelResponse:
    """A normalized, JSON-serializable model reply.

    Keeping this vendor-neutral is what lets the Effect boundary record and
    replay a model call without depending on any SDK's response objects.
    """

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"  # "end_turn" | "tool_use"
    usage: dict = field(default_factory=dict)  # {"input_tokens": int, "output_tokens": int}

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "stop_reason": self.stop_reason,
            "usage": self.usage,
        }

    @staticmethod
    def from_dict(d: dict) -> "ModelResponse":
        return ModelResponse(
            text=d.get("text", ""),
            tool_calls=[ToolCall.from_dict(t) for t in d.get("tool_calls", [])],
            stop_reason=d.get("stop_reason", "end_turn"),
            usage=d.get("usage", {}),
        )


@runtime_checkable
class ModelProvider(Protocol):
    """The one interface the kernel depends on. Implement it for any model."""

    name: str
    model: str

    def complete(
        self, system: str, messages: list[dict], tools: list[dict]
    ) -> ModelResponse: ...
