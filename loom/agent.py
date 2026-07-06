"""The agent loop.

A tiny, readable loop: build messages from context, call the model (through the
Effect boundary), run any requested tools (also through the boundary), append
results, repeat until the model stops. Every nondeterministic step is an effect,
so the whole run is recorded and can be replayed, forked, and bisected.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from .context import Context
from .effect import Recorder
from .providers.base import ModelProvider, ModelResponse
from .tools import Tool


@dataclass
class SubagentTool(Tool):
    """A tool that delegates to a child ``Agent`` with its own isolated context.

    The child's model and tool calls are recorded inline in the same trace (at a
    deeper ``depth``), so replay/fork/bisect keep working across delegation. The
    delegation itself is not a separate effect -- its nondeterminism lives in the
    child's leaf effects, per the Effect-boundary philosophy.
    """

    agent: "Agent | None" = None


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
        name: str = "agent",
    ):
        self.provider = _resolve_provider(model, provider)
        self.model = getattr(self.provider, "model", str(model))
        self.tools: dict[str, Tool] = {t.name: t for t in (tools or [])}
        self.system = system
        self.max_steps = max_steps
        self.budget = budget
        self.name = name

    # -- delegation -------------------------------------------------------

    def as_tool(self, name: "str | None" = None, description: "str | None" = None) -> SubagentTool:
        """Expose this agent as a tool another agent can delegate to.

            researcher = Agent(model=..., tools=[search], name="researcher")
            lead = Agent(model=..., tools=[researcher.as_tool()])
        """
        return SubagentTool(
            name=name or self.name,
            description=description
            or f"Delegate a task to the {self.name} subagent. Pass a 'task' string.",
            fn=lambda **_: None,  # never called; delegation is handled in the loop
            input_schema={
                "type": "object",
                "properties": {"task": {"type": "string"}},
                "required": ["task"],
            },
            agent=self,
        )

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
        output, ctx, truncated = self._loop(
            prompt, rec, depth=0, edit=_edit, edit_at_turn=_edit_at_turn
        )
        return Run(
            agent=self,
            recorder=rec,
            context=ctx,
            prompt=prompt,
            output=output,
            truncated=truncated,
        )

    def _loop(
        self,
        prompt: str,
        rec: Recorder,
        depth: int = 0,
        edit: "Callable[[Context], None] | None" = None,
        edit_at_turn: int = -1,
    ) -> "tuple[str, Context, bool]":
        """The core agent loop. Shared by top-level runs and nested subagents."""
        ctx = Context(system=self.system, budget=self.budget)
        ctx.add_user(prompt)

        tool_schemas = [t.schema() for t in self.tools.values()]
        resp = ModelResponse()
        truncated = True

        for turn in range(self.max_steps):
            # Fork intervention applies only at the top level, before this turn.
            if edit is not None and depth == 0 and turn == edit_at_turn:
                edit(ctx)

            rec.depth = depth
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
                    tool = self.tools.get(tc.name)
                    if isinstance(tool, SubagentTool) and tool.agent is not None:
                        # Delegate: the child records its own effects at depth+1.
                        task = str(tc.input.get("task", ""))
                        result, _child_ctx, _ = tool.agent._loop(task, rec, depth=depth + 1)
                    else:
                        rec.depth = depth
                        result = rec.run(
                            f"tool:{tc.name}",
                            {"id": tc.id, "input": tc.input},
                            lambda tc=tc: self._call_tool(tc.name, tc.input),
                        )
                    ctx.add_tool_result(tc.id, tc.name, result)
                continue  # let the model observe tool results

            truncated = False
            break

        return resp.text, ctx, truncated

    def _call_tool(self, name: str, args: dict) -> Any:
        tool = self.tools.get(name)
        if tool is None:
            return f"ERROR: unknown tool {name!r}"
        try:
            return _jsonable(tool(**args))
        except Exception as e:  # tools failing shouldn't crash the harness
            return f"ERROR: {type(e).__name__}: {e}"
