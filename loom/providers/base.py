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
        # Tolerant of hand-edited / third-party traces: a tool_call missing id or
        # name (or with a null one) degrades to "" rather than crashing every
        # analyzer that reads the call. input coerces to {} when not a dict.
        inp = d.get("input")
        return ToolCall(id=d.get("id") or "", name=d.get("name") or "",
                        input=inp if isinstance(inp, dict) else {})


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
        # Tolerate hand-edited / third-party traces: a null tool_calls or usage
        # (some serializers emit `null` rather than omitting the key) must read
        # as empty, not crash. `or` covers both "missing" and "present but null".
        if not isinstance(d, dict):
            return ModelResponse()
        return ModelResponse(
            text=d.get("text") or "",
            tool_calls=[ToolCall.from_dict(t) for t in (d.get("tool_calls") or [])
                        if isinstance(t, dict)],
            stop_reason=d.get("stop_reason") or "end_turn",
            usage=d.get("usage") or {},
        )


@runtime_checkable
class ModelProvider(Protocol):
    """The one interface the kernel depends on. Implement it for any model."""

    name: str
    model: str

    def complete(
        self, system: str, messages: list[dict], tools: list[dict]
    ) -> ModelResponse: ...
