"""Live provider for any OpenAI-compatible Chat Completions API (optional).

Works with OpenAI and the many servers that speak the same protocol -- vLLM,
Ollama, LM Studio, Together, Groq, OpenRouter, and more -- by pointing
``base_url`` at them.

    pip install "loom-harness[openai]"

    # OpenAI
    Agent(provider=OpenAIProvider("gpt-4o"))
    # A local server (e.g. Ollama / vLLM)
    Agent(provider=OpenAIProvider("llama3.1", base_url="http://localhost:11434/v1", api_key="x"))
"""

from __future__ import annotations

import json

from .base import ModelResponse, ToolCall


class OpenAIProvider:
    """Adapts Loom's neutral interface to the OpenAI Chat Completions API."""

    def __init__(
        self,
        model: str,
        api_key: "str | None" = None,
        base_url: "str | None" = None,
        max_tokens: int = 2048,
    ):
        try:
            import openai
        except ImportError as e:  # pragma: no cover - import guard
            raise ImportError(
                "OpenAIProvider requires the openai SDK. "
                'Install it with: pip install "loom-harness[openai]"'
            ) from e
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.name = "openai"
        self.max_tokens = max_tokens

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> ModelResponse:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self._to_openai_messages(system, messages),
        }
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                    },
                }
                for t in tools
            ]
        resp = self._client.chat.completions.create(**kwargs)
        return self._from_openai(resp)

    # -- translation ------------------------------------------------------

    @staticmethod
    def _to_openai_messages(system: str, messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        if system:
            out.append({"role": "system", "content": system})
        for m in messages:
            role = m["role"]
            if role == "user":
                out.append({"role": "user", "content": m["content"]})
            elif role == "assistant":
                msg: dict = {"role": "assistant", "content": m.get("content") or ""}
                calls = m.get("tool_calls", [])
                if calls:
                    msg["tool_calls"] = [
                        {
                            "id": tc.id if hasattr(tc, "id") else tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc.name if hasattr(tc, "name") else tc["name"],
                                "arguments": json.dumps(
                                    tc.input if hasattr(tc, "input") else tc["input"]
                                ),
                            },
                        }
                        for tc in calls
                    ]
                out.append(msg)
            elif role == "tool":
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": m["tool_call_id"],
                        "content": str(m["content"]),
                    }
                )
        return out

    @staticmethod
    def _from_openai(resp) -> ModelResponse:
        choice = resp.choices[0]
        msg = choice.message
        text = msg.content or ""
        tool_calls: list[ToolCall] = []
        for tc in getattr(msg, "tool_calls", None) or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except (json.JSONDecodeError, TypeError):
                args = {}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, input=args))
        stop = "tool_use" if tool_calls else "end_turn"
        usage = {}
        if getattr(resp, "usage", None):
            usage = {
                "input_tokens": getattr(resp.usage, "prompt_tokens", 0),
                "output_tokens": getattr(resp.usage, "completion_tokens", 0),
            }
        return ModelResponse(text=text, tool_calls=tool_calls, stop_reason=stop, usage=usage)
