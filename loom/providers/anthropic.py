"""Live provider for Anthropic's Claude models (optional).

Install with ``pip install "loom-harness[anthropic]"`` and set ``ANTHROPIC_API_KEY``.
The kernel does not import this module unless you use it, so the anthropic SDK
stays an optional dependency.
"""

from __future__ import annotations

from typing import Callable  # noqa: F401 -- used in annotations

from .base import ModelResponse, ToolCall


class AnthropicProvider:
    """Adapts Loom's neutral interface to the Anthropic Messages API.

    With ``on_token`` set, responses stream and each text delta is passed to
    the callback as it arrives; the recorded effect is still the complete
    final response. (During trace replay providers are never called, so
    nothing re-streams -- replays return instantly.)
    """

    def __init__(
        self,
        model: str,
        api_key: "str | None" = None,
        max_tokens: int = 2048,
        on_token: "Callable[[str], None] | None" = None,
        base_url: "str | None" = None,
    ):
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover - import guard
            raise ImportError(
                "AnthropicProvider requires the anthropic SDK. "
                'Install it with: pip install "loom-harness[anthropic]"'
            ) from e
        # An explicit base_url overrides ANTHROPIC_BASE_URL -- used to make the
        # debugger's OWN meta calls (explain/copilot) bypass a recording proxy so
        # they never pollute the trace they are analyzing.
        self._client = anthropic.Anthropic(api_key=api_key, **({"base_url": base_url} if base_url else {}))
        self.model = model
        self.name = "anthropic"
        self.max_tokens = max_tokens
        self.on_token = on_token

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> ModelResponse:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self._to_anthropic_messages(messages),
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "input_schema": t.get("input_schema", {"type": "object", "properties": {}}),
                }
                for t in tools
            ]
        if self.on_token is not None:
            with self._client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    self.on_token(text)
                resp = stream.get_final_message()
        else:
            resp = self._client.messages.create(**kwargs)
        return self._from_anthropic(resp)

    # -- translation ------------------------------------------------------

    @staticmethod
    def _to_anthropic_messages(messages: list[dict]) -> list[dict]:
        """Convert neutral messages to Anthropic content-block format.

        Consecutive tool results are merged into a single user message, as the
        API requires all tool_result blocks to follow their tool_use turn.
        """
        out: list[dict] = []
        for m in messages:
            role = m["role"]
            if role == "user":
                out.append({"role": "user", "content": [{"type": "text", "text": m["content"]}]})
            elif role == "assistant":
                blocks: list[dict] = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for tc in m.get("tool_calls", []):
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.id if hasattr(tc, "id") else tc["id"],
                            "name": tc.name if hasattr(tc, "name") else tc["name"],
                            "input": tc.input if hasattr(tc, "input") else tc["input"],
                        }
                    )
                out.append({"role": "assistant", "content": blocks})
            elif role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": m["tool_call_id"],
                    "content": str(m["content"]),
                }
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list) \
                        and out[-1]["content"] and out[-1]["content"][0].get("type") == "tool_result":
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
        return out

    @staticmethod
    def _from_anthropic(resp) -> ModelResponse:
        text = ""
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, input=dict(block.input)))
        stop = "tool_use" if tool_calls else "end_turn"
        usage = {
            "input_tokens": getattr(resp.usage, "input_tokens", 0),
            "output_tokens": getattr(resp.usage, "output_tokens", 0),
        }
        return ModelResponse(text=text, tool_calls=tool_calls, stop_reason=stop, usage=usage)
