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

from .context import Context, Item
from .effect import Recorder
from .providers.base import ModelProvider, ModelResponse
from .tools import Tool


def _tokens_spent(rec: Recorder) -> int:
    """Total tokens across every model call recorded so far (all depths)."""
    total = 0
    for e in rec.log:
        if e.kind == "model":
            usage = e.result.get("usage", {}) if isinstance(e.result, dict) else {}
            total += usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
    return total


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
        policy: "Any | None" = None,
        cache: "Any | None" = None,
        memory: "Any | None" = None,
        compact_after: "int | None" = None,
        compact_keep: int = 4,
        output_type: "Any | None" = None,
        output_retries: int = 2,
        clock: bool = False,
        critic: Any = None,
        critic_threshold: float = 0.6,
        critic_retries: int = 1,
        deliberate: int = 1,
    ):
        self.provider = _resolve_provider(model, provider)
        self.model = getattr(self.provider, "model", str(model))
        self.tools: dict[str, Tool] = {t.name: t for t in (tools or [])}
        # output_type is agent CONFIG (not a run() argument) so that replays
        # walk the same validation path -- the same-config rule.
        self.output_type = output_type
        self.output_retries = output_retries
        if output_type is not None:
            from .structured import format_instruction

            suffix = format_instruction(output_type)
            system = f"{system}\n\n{suffix}" if system else suffix
        self.system = system
        self.max_steps = max_steps
        self.budget = budget
        self.name = name
        self.on_human = on_human
        self.parallel_tools = parallel_tools
        self.journal = journal  # write-ahead journal path; see loom/journal.py
        self.policy = policy  # tool rules + token budget; see loom/policy.py
        self.cache = cache  # effect cache; see loom/cache.py
        self.memory = memory  # trace memory; see loom/memory.py
        self.compact_after = compact_after  # summarize history past this many tokens
        self.compact_keep = compact_keep  # recent items kept verbatim after compaction
        self.clock = clock  # tell the model today's date via a recorded time effect
        # Self-correction: a (usually cheaper) critic model scores final answers
        # as recorded "critic" effects; a low score rewinds the turn with the
        # critique in context. deliberate=N samples N candidate answers and lets
        # the critic pick ("sample"/"choose" effects) -- inference-time scaling
        # that stays fully replayable.
        self.critic_provider = _resolve_provider(critic, None) if critic is not None else None
        self.critic_threshold = critic_threshold
        self.critic_retries = critic_retries
        self.deliberate = max(1, deliberate)
        if self.deliberate > 1 and self.critic_provider is None:
            raise ValueError("deliberate=N needs a critic to pick the winner: Agent(critic=...)")

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
        if self.cache is not None and rec.allow_live:
            rec.cache = self.cache
        from .ambient import _activate, _deactivate

        ambient_token = _activate(rec)  # loom.now()/loom.random() route here
        try:
            output, ctx, truncated, stop_reason = self._loop(
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
        finally:
            _deactivate(ambient_token)
        run_obj = Run(
            agent=self,
            recorder=rec,
            context=ctx,
            prompt=episodes[0],
            output=output,
            truncated=truncated,
            episodes=episodes,
            stop_reason=stop_reason,
        )
        # Trace memory auto-store: completed live runs become future experience.
        if (
            self.memory is not None
            and getattr(self.memory, "auto_store", False)
            and rec.allow_live
            and not truncated
        ):
            self.memory.add(run_obj)
        return run_obj

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
    ) -> "tuple[str, Context, bool, str]":
        """The core agent loop. Shared by top-level runs and nested subagents.

        ``episodes`` is the conversation: each entry is one user message, run
        to completion (end_turn) before the next begins.
        """
        ctx = Context(system=self.system, budget=self.budget)
        tool_schemas = [t.schema() for t in self.tools.values()]
        resp = ModelResponse()
        truncated = True
        stop_reason = ""
        turn = 0  # model calls at this depth, counted across all episodes

        for ep_index, episode in enumerate(episodes):
            # The clock: today's date as a recorded "time" effect, so the model
            # knows when "now" is and replays still see the original moment.
            if ep_index == 0 and depth == 0 and self.clock:
                import time as _time
                from datetime import datetime, timezone

                rec.depth = depth
                ts = rec.run("time", {}, _time.time)
                stamp = datetime.fromtimestamp(ts, tz=timezone.utc).strftime(
                    "%Y-%m-%d %H:%M UTC"
                )
                ctx.add_user(f"Current datetime: {stamp}", source="clock")
            ctx.add_user(episode)
            # Trace memory: recall similar past runs once, at conversation start.
            # The store changes over time, so recall is a recorded effect.
            if ep_index == 0 and depth == 0 and self.memory is not None:
                rec.depth = depth
                recalled = rec.run(
                    "memory",
                    {"query": episode},
                    lambda episode=episode: self.memory.recall_text(episode),
                )
                if recalled:
                    ctx.add_user(recalled, source="memory")
            truncated = True
            format_retries = 0  # validation retries used in this episode
            critic_rounds = 0  # critic-triggered rewinds used in this episode
            for _ in range(self.max_steps):
                # Replay any recorded context edits (from earlier forks) first,
                # so rebuilt context reproduces past surgery exactly.
                while depth == 0 and rec.peek_kind() == "edit":
                    rec.depth = depth
                    snapshot = rec.run("edit", {"turn": turn}, lambda: None)
                    ctx.items[:] = [Item.from_dict(d) for d in snapshot]

                # Fork intervention applies only at the top level, before this
                # turn -- and is recorded as an "edit" effect (a full context
                # snapshot), so the fork is self-describing in the trace.
                if edit is not None and depth == 0 and turn == edit_at_turn:
                    edit(ctx)
                    rec.depth = depth
                    rec.run(
                        "edit",
                        {"turn": turn},
                        lambda ctx=ctx: [it.to_dict() for it in ctx.items],
                    )

                # Compaction: when history outgrows the threshold, summarize it
                # into one pinned item. The summary is a model call = an effect,
                # so compacted runs replay deterministically.
                if (
                    self.compact_after is not None
                    and depth == 0
                    and ctx.total_tokens() > self.compact_after
                    and len(ctx.items) > self.compact_keep
                ):
                    rec.depth = depth
                    summary = rec.run(
                        "compact", {"turn": turn}, lambda ctx=ctx: self._summarize(ctx)
                    )
                    self._compact(ctx, summary)

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

                # Hard token budget: stop cleanly in a resumable state.
                if (
                    depth == 0
                    and self.policy is not None
                    and self.policy.budget_tokens is not None
                    and _tokens_spent(rec) >= self.policy.budget_tokens
                ):
                    stop_reason = "budget"
                    truncated = True
                    break

                if resp.stop_reason == "tool_use" and resp.tool_calls:
                    self._run_tools(resp.tool_calls, ctx, rec, depth)
                    continue  # let the model observe tool results

                # Deliberate: sample extra candidate answers and let the critic
                # pick. Samples are "sample" effects, NOT "model" effects, so
                # turn semantics (fork points, num_turns) are untouched.
                if self.deliberate > 1 and depth == 0:
                    candidates = [resp.text]
                    for i in range(self.deliberate - 1):
                        rec.depth = depth
                        alt = rec.run(
                            "sample",
                            {"turn": turn, "i": i},
                            lambda messages=messages: self.provider.complete(
                                self.system, messages, tool_schemas
                            ),
                            encode=lambda r: r.to_dict(),
                            decode=ModelResponse.from_dict,
                        )
                        candidates.append(alt.text)
                    rec.depth = depth
                    choice = rec.run(
                        "choose",
                        {"turn": turn, "candidates": candidates},
                        lambda episode=episode, candidates=candidates: self._choose(
                            episode, candidates
                        ),
                    )
                    best = candidates[choice.get("best", 0) % len(candidates)]
                    if best != resp.text:
                        resp.text = best
                        ctx.items[-1].content = best  # the assistant item just added

                # Structured output: validate the final answer at the boundary.
                # Validation is a pure function of the recorded text, so replays
                # deterministically walk the same retry path.
                if self.output_type is not None and depth == 0:
                    from .structured import OutputInvalid, parse_as

                    try:
                        parse_as(self.output_type, resp.text)
                    except OutputInvalid as err:
                        if format_retries < self.output_retries:
                            format_retries += 1
                            ctx.add_user(
                                f"Your answer could not be parsed: {err}. Respond "
                                "again with ONLY a JSON object matching the schema.",
                                source="validation",
                            )
                            continue
                        stop_reason = "invalid_output"

                # Critic gate: a (cheaper) reviewer scores the final answer as a
                # recorded effect; a low score rewinds the turn with the
                # critique in context. The failed attempt, the verdict, and the
                # retry all stay in the trace -- self-correction you can replay.
                if (
                    self.critic_provider is not None
                    and depth == 0
                    and not stop_reason
                ):
                    rec.depth = depth
                    verdict = rec.run(
                        "critic",
                        {"turn": turn, "text": resp.text},
                        lambda episode=episode, text=resp.text: self._critique(episode, text),
                    )
                    if (
                        verdict.get("score", 1.0) < self.critic_threshold
                        and critic_rounds < self.critic_retries
                    ):
                        critic_rounds += 1
                        ctx.add_user(
                            f"A reviewer scored your answer {verdict.get('score')}/1.0: "
                            f"{verdict.get('critique', '')} Improve your answer.",
                            source="critique",
                        )
                        continue

                truncated = False
                break
            if stop_reason:
                break

        return resp.text, ctx, truncated, stop_reason

    def _run_tools(self, calls: list, ctx: Context, rec: Recorder, depth: int) -> None:
        """Execute one turn's tool calls, recording each through the boundary."""
        decisions: dict[str, str] = {}
        for tc in calls:
            tool = self.tools.get(tc.name)
            if isinstance(tool, (SubagentTool, HumanTool)):
                decisions[tc.id] = "special"  # handled by their own paths below
            elif self.policy is None:
                decisions[tc.id] = "allow"
            else:
                decisions[tc.id] = self.policy.decide(tc.name)

        results: "dict[str, Any] | None" = None
        # Parallel execution is safe only when EVERY call in the turn is a plain
        # allowed tool, fully outside the replay region (otherwise tools would
        # re-execute during replay, or policy gates would be bypassed). Results
        # are still RECORDED in call order, so the trace stays deterministic.
        if (
            self.parallel_tools
            and len(calls) > 1
            and all(d == "allow" for d in decisions.values())
            and rec.cursor >= rec.replay_until
        ):
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=len(calls)) as pool:
                futures = {tc.id: pool.submit(self._call_tool, tc.name, tc.input) for tc in calls}
                results = {cid: f.result() for cid, f in futures.items()}

        for tc in calls:
            tool = self.tools.get(tc.name)
            decision = decisions[tc.id]
            if isinstance(tool, SubagentTool) and tool.agent is not None:
                # Delegate: the child records its own effects at depth+1.
                task = str(tc.input.get("task", ""))
                result, _child_ctx, _, _ = tool.agent._loop([task], rec, depth=depth + 1)
            elif isinstance(tool, HumanTool):
                question = str(tc.input.get("question", ""))
                rec.depth = depth
                result = rec.run(
                    "human",
                    {"question": question},
                    lambda question=question, depth=depth: self._human(question, depth),
                )
            else:
                if decision == "confirm":
                    # Approval is nondeterminism -> a recorded human effect.
                    # No handler -> the run pauses; resume("yes") continues it.
                    from .policy import affirmative

                    q = f"Approve tool call {tc.name}({json.dumps(tc.input)})? Reply yes or no."
                    rec.depth = depth
                    answer = rec.run(
                        "human", {"question": q}, lambda q=q, depth=depth: self._human(q, depth)
                    )
                    decision = "allow" if affirmative(answer) else "rejected"

                if decision == "deny":
                    fn = lambda tc=tc: f"BLOCKED: {tc.name} denied by policy"  # noqa: E731
                elif decision == "rejected":
                    fn = lambda tc=tc: f"BLOCKED: {tc.name} rejected by operator"  # noqa: E731
                elif decision == "stub":
                    fn = (  # noqa: E731
                        lambda tc=tc: f"DRY-RUN: would call {tc.name}({json.dumps(tc.input)})"
                    )
                elif results is not None:
                    fn = lambda tc=tc: results[tc.id]  # noqa: E731
                else:
                    fn = lambda tc=tc: self._call_tool(tc.name, tc.input)  # noqa: E731
                rec.depth = depth
                result = rec.run(f"tool:{tc.name}", {"id": tc.id, "input": tc.input}, fn)
            ctx.add_tool_result(tc.id, tc.name, result)

    def _critique(self, question: str, text: str) -> dict:
        """Ask the critic to score an answer. Fails open (score 1.0) on junk replies."""
        from .structured import extract_json

        resp = self.critic_provider.complete(
            'You are a strict reviewer. Reply ONLY with JSON: {"score": <0.0-1.0>, "critique": "<one sentence>"}.',
            [
                {
                    "role": "user",
                    "content": f"Question:\n{question}\n\nAnswer:\n{text}\n\nScore this answer.",
                }
            ],
            [],
        )
        try:
            data = extract_json(resp.text)
            return {
                "score": float(data.get("score", 1.0)),
                "critique": str(data.get("critique", ""))[:500],
            }
        except Exception:
            return {"score": 1.0, "critique": "unparseable critic reply; letting the answer pass"}

    def _choose(self, question: str, candidates: list[str]) -> dict:
        """Ask the critic to pick the best candidate. Falls back to the first."""
        from .structured import extract_json

        numbered = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(candidates))
        resp = self.critic_provider.complete(
            'You judge candidate answers. Reply ONLY with JSON: {"best": <index>, "why": "<one sentence>"}.',
            [
                {
                    "role": "user",
                    "content": f"Question:\n{question}\n\nCandidates:\n{numbered}\n\nWhich index is best?",
                }
            ],
            [],
        )
        try:
            data = extract_json(resp.text)
            return {"best": int(data.get("best", 0)), "why": str(data.get("why", ""))[:500]}
        except Exception:
            return {"best": 0, "why": "unparseable critic reply; kept the first candidate"}

    def _human(self, question: str, depth: int) -> str:
        """Answer via the on_human handler, or pause the run."""
        if self.on_human is not None:
            return str(self.on_human(question))
        raise HumanInputRequired(question, depth)

    def _summarize(self, ctx: Context) -> str:
        """Ask the model to compress the history that compaction will drop."""
        old = ctx.items[: -self.compact_keep] if self.compact_keep else list(ctx.items)
        transcript = "\n".join(f"[{it.role}] {it.content}" for it in old if it.content)
        resp = self.provider.complete(
            "You compress conversation history. Keep facts, decisions, IDs, and open tasks.",
            [{"role": "user", "content": f"Summarize this history concisely:\n\n{transcript}"}],
            [],
        )
        return resp.text

    def _compact(self, ctx: Context, summary: str) -> None:
        """Replace old history with a pinned summary + the most recent items."""
        from .context import estimate_tokens

        tail = list(ctx.items[-self.compact_keep :]) if self.compact_keep else []
        while tail and tail[0].role == "tool":
            tail.pop(0)  # never leave an orphaned tool result at the front
        head = ctx.items[: len(ctx.items) - len(tail)] if tail else list(ctx.items)
        pinned = [it for it in head if it.pinned]
        summary_item = Item(
            "user",
            f"Summary of earlier conversation: {summary}",
            "compaction",
            pinned=True,
            tokens=estimate_tokens(summary),
        )
        ctx.items[:] = pinned + [summary_item] + tail

    def _call_tool(self, name: str, args: dict) -> Any:
        tool = self.tools.get(name)
        if tool is None:
            return f"ERROR: unknown tool {name!r}"
        try:
            return _jsonable(tool(**args))
        except Exception as e:  # tools failing shouldn't crash the harness
            return f"ERROR: {type(e).__name__}: {e}"
