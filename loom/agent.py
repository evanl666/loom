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


@dataclass
class HumanTool(Tool):
    """A tool that asks the human operator for input.

    A human's answer is nondeterminism like any other, so it is recorded as a
    ``"human"`` effect -- replays include human decisions. If the agent has an
    ``on_human`` handler, the question is answered inline; otherwise the run
    pauses (``run.paused``) and can be continued later with ``run.resume(answer)``.
    """


def ask_human(
    name: str = "ask_human",
    description: str = "Ask the human operator a question and wait for their answer.",
) -> HumanTool:
    """Build the built-in human-in-the-loop tool. Add it to an agent's tools."""
    return HumanTool(
        name=name,
        description=description,
        fn=lambda **_: None,  # never called; the loop handles this tool
        input_schema={
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
    )


class HumanInputRequired(RuntimeError):
    """Raised internally when a human answer is needed and no handler is set."""

    def __init__(self, question: str, depth: int = 0):
        super().__init__(f"human input required: {question!r}")
        self.question = question
        self.depth = depth


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
        on_human: "Callable[[str], str] | None" = None,
        parallel_tools: bool = False,
        journal: "str | None" = None,
    ):
        self.provider = _resolve_provider(model, provider)
        self.model = getattr(self.provider, "model", str(model))
        self.tools: dict[str, Tool] = {t.name: t for t in (tools or [])}
        self.system = system
        self.max_steps = max_steps
        self.budget = budget
        self.name = name
        self.on_human = on_human
        self.parallel_tools = parallel_tools
        self.journal = journal  # write-ahead journal path; see loom/journal.py

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
        prompt: "str | list[str]",
        *,
        recorder: "Recorder | None" = None,
        _edit: "Callable[[Context], None] | None" = None,
        _edit_at_turn: int = -1,
    ) -> "Run":
        """Run the agent (or a whole conversation) and return a recorded ``Run``.

        ``prompt`` is one user message, or a list of them: each entry runs to
        completion in order, sharing one context and one trace.
        """
        from .trace import Run  # local import to avoid a cycle

        episodes = (
            [str(p) for p in prompt] if isinstance(prompt, (list, tuple)) else [str(prompt)]
        )
        rec = recorder or Recorder.record()
        if self.journal and rec.allow_live:
            # Write-ahead journal: header + any replayed prefix now, then every
            # new effect the moment it records. Pure replays never journal.
            from .journal import Journal

            j = Journal(self.journal)
            j.start(
                {"version": 1, "model": self.model, "system": self.system, "episodes": episodes},
                rec.log[: rec.replay_until],
            )
            rec.journal = j
        try:
            output, ctx, truncated = self._loop(
                episodes, rec, depth=0, edit=_edit, edit_at_turn=_edit_at_turn
            )
        except HumanInputRequired as e:
            # Pause: everything up to the question is already recorded, so
            # resume() can inject the answer as a "human" effect and continue.
            return Run(
                agent=self,
                recorder=rec,
                context=Context(),
                prompt=episodes[0],
                output="",
                truncated=True,
                episodes=episodes,
                paused=True,
                pending=e.question,
                pending_depth=e.depth,
            )
        return Run(
            agent=self,
            recorder=rec,
            context=ctx,
            prompt=episodes[0],
            output=output,
            truncated=truncated,
            episodes=episodes,
        )

    async def arun(self, prompt: "str | list[str]", **kwargs: Any) -> "Run":
        """Async convenience wrapper: runs the (synchronous) loop in a thread."""
        import asyncio

        return await asyncio.to_thread(self.run, prompt, **kwargs)

    def _loop(
        self,
        episodes: list[str],
        rec: Recorder,
        depth: int = 0,
        edit: "Callable[[Context], None] | None" = None,
        edit_at_turn: int = -1,
    ) -> "tuple[str, Context, bool]":
        """The core agent loop. Shared by top-level runs and nested subagents.

        ``episodes`` is the conversation: each entry is one user message, run
        to completion (end_turn) before the next begins.
        """
        ctx = Context(system=self.system, budget=self.budget)
        tool_schemas = [t.schema() for t in self.tools.values()]
        resp = ModelResponse()
        truncated = True
        turn = 0  # model calls at this depth, counted across all episodes

        for episode in episodes:
            ctx.add_user(episode)
            truncated = True
            for _ in range(self.max_steps):
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
                turn += 1

                if resp.stop_reason == "tool_use" and resp.tool_calls:
                    self._run_tools(resp.tool_calls, ctx, rec, depth)
                    continue  # let the model observe tool results

                truncated = False
                break

        return resp.text, ctx, truncated

    def _run_tools(self, calls: list, ctx: Context, rec: Recorder, depth: int) -> None:
        """Execute one turn's tool calls, recording each through the boundary."""
        plain = all(
            not isinstance(self.tools.get(tc.name), (SubagentTool, HumanTool)) for tc in calls
        )
        results: "dict[str, Any] | None" = None
        # Parallel execution is safe only for plain tools and only fully outside
        # the replay region (otherwise tools would re-execute during replay).
        # Results are still RECORDED in call order, so the trace stays
        # deterministic no matter which tool finished first.
        if self.parallel_tools and plain and len(calls) > 1 and rec.cursor >= rec.replay_until:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=len(calls)) as pool:
                futures = {tc.id: pool.submit(self._call_tool, tc.name, tc.input) for tc in calls}
                results = {cid: f.result() for cid, f in futures.items()}

        for tc in calls:
            tool = self.tools.get(tc.name)
            if isinstance(tool, SubagentTool) and tool.agent is not None:
                # Delegate: the child records its own effects at depth+1.
                task = str(tc.input.get("task", ""))
                result, _child_ctx, _ = tool.agent._loop([task], rec, depth=depth + 1)
            elif isinstance(tool, HumanTool):
                question = str(tc.input.get("question", ""))
                rec.depth = depth
                result = rec.run(
                    "human",
                    {"question": question},
                    lambda question=question, depth=depth: self._human(question, depth),
                )
            else:
                rec.depth = depth
                result = rec.run(
                    f"tool:{tc.name}",
                    {"id": tc.id, "input": tc.input},
                    (lambda tc=tc: results[tc.id])
                    if results is not None
                    else (lambda tc=tc: self._call_tool(tc.name, tc.input)),
                )
            ctx.add_tool_result(tc.id, tc.name, result)

    def _human(self, question: str, depth: int) -> str:
        """Answer via the on_human handler, or pause the run."""
        if self.on_human is not None:
            return str(self.on_human(question))
        raise HumanInputRequired(question, depth)

    def _call_tool(self, name: str, args: dict) -> Any:
        tool = self.tools.get(name)
        if tool is None:
            return f"ERROR: unknown tool {name!r}"
        try:
            return _jsonable(tool(**args))
        except Exception as e:  # tools failing shouldn't crash the harness
            return f"ERROR: {type(e).__name__}: {e}"
