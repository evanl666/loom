"""The agent loop.

A tiny, readable loop: build messages from context, call the model (through the
Effect boundary), run any requested tools (also through the boundary), append
results, repeat until the model stops. Every nondeterministic step is an effect,
so the whole run is recorded and can be replayed, forked, and bisected.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from .context import Context
from .effect import Recorder
from .providers.base import ModelProvider, ModelResponse
from .tools import Tool


def _jsonable(value: Any) -> Any:
    """Coerce a tool result into something JSON-serializable for the trace."""
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _resolve_provider(model: Any, provider: "ModelProvider | None") -> ModelProvider:
    if provider is not None:
        return provider
    if hasattr(model, "complete"):  # a provider instance passed as `model`
        return model
    # Otherwise treat `model` as an Anthropic model id string.
    from .providers.anthropic import AnthropicProvider

    return AnthropicProvider(model=model)


class Agent:
    """An LLM plus a harness. Construct it, call ``run``, get a ``Run`` back.

        agent = Agent(model="claude-opus-4-8", tools=[add])
        run = agent.run("What is 2 + 2?")
        print(run.output)
    """

    def __init__(
        self,
        model: Any = "claude-opus-4-8",
        tools: "list[Tool] | None" = None,
        system: str = "",
        provider: "ModelProvider | None" = None,
        max_steps: int = 20,
        budget: "int | None" = None,
    ):
        self.provider = _resolve_provider(model, provider)
        self.model = getattr(self.provider, "model", str(model))
        self.tools: dict[str, Tool] = {t.name: t for t in (tools or [])}
        self.system = system
        self.max_steps = max_steps
        self.budget = budget

    # -- running ----------------------------------------------------------

    def run(
        self,
        prompt: str,
        *,
        recorder: "Recorder | None" = None,
        _edit: "Callable[[Context], None] | None" = None,
        _edit_at_turn: int = -1,
    ) -> "Run":
        """Run the agent to completion and return a recorded ``Run``."""
        from .trace import Run  # local import to avoid a cycle

        rec = recorder or Recorder.record()
        ctx = Context(system=self.system, budget=self.budget)
        ctx.add_user(prompt)

        tool_schemas = [t.schema() for t in self.tools.values()]
        resp = ModelResponse()
        truncated = True

        for turn in range(self.max_steps):
            if _edit is not None and turn == _edit_at_turn:
                _edit(ctx)  # fork intervention: mutate context before this model call

            messages = ctx.messages()
            resp = rec.run(
                "model",
                {"system": self.system, "messages": messages},
                lambda messages=messages: self.provider.complete(
                    self.system, messages, tool_schemas
                ),
                encode=lambda r: r.to_dict(),
                decode=ModelResponse.from_dict,
            )
            ctx.add_assistant(resp)

            if resp.stop_reason == "tool_use" and resp.tool_calls:
                for tc in resp.tool_calls:
                    result = rec.run(
                        f"tool:{tc.name}",
                        {"id": tc.id, "input": tc.input},
                        lambda tc=tc: self._call_tool(tc.name, tc.input),
                    )
                    ctx.add_tool_result(tc.id, tc.name, result)
                continue  # let the model observe tool results

            truncated = False
            break

        return Run(
            agent=self,
            recorder=rec,
            context=ctx,
            prompt=prompt,
            output=resp.text,
            truncated=truncated,
        )

    def _call_tool(self, name: str, args: dict) -> Any:
        tool = self.tools.get(name)
        if tool is None:
            return f"ERROR: unknown tool {name!r}"
        try:
            return _jsonable(tool(**args))
        except Exception as e:  # tools failing shouldn't crash the harness
            return f"ERROR: {type(e).__name__}: {e}"
