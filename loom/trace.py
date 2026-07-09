"""The Run object: a recorded agent run you can inspect, replay, fork, and bisect.

The trace is the product. Everything a Run can do falls out of the effect log
captured through the Effect boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Callable

from .context import Context
from .effect import EffectEntry, Recorder
from .providers.base import ModelResponse

# The trace format version, embedded in every saved trace.
#
# Compatibility policy: fields are only ever ADDED at the same version --
# readers must ignore keys they don't know. The version bumps only when the
# meaning of existing data changes (so far: how effect keys are computed),
# because that silently breaks strict replay and impact verdicts.
#
#   1 -- original format; model-effect keys hash {system, messages} only
#   2 -- model-effect keys also hash the tool schemas (a tool added or a
#        schema edited now fails strict replay / shows up in impact)
TRACE_VERSION = 2


def trace_checksum(data: dict) -> str:
    """Content hash of a trace dict (excluding the checksum field itself)."""
    body = {k: v for k, v in sorted(data.items()) if k != "checksum"}
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True, default=str).encode()
    ).hexdigest()
    return f"sha256:{digest}"


def _check_integrity(data: dict, path: str) -> None:
    """Warn (never fail) when a stored checksum doesn't match the content.

    Tamper-EVIDENT, not tamper-proof: anyone can edit a trace and recompute
    the hash, but an edit that forgets to is visible -- which catches the
    common cases (hand-edited fixtures, truncated copies, merge damage).
    """
    stored = data.get("checksum")
    if stored and stored != trace_checksum(data):
        import warnings

        warnings.warn(
            f"{path} was modified after it was written (checksum mismatch). "
            f"If the edit was deliberate, re-stamp it: loom migrate {path}",
            stacklevel=3,
        )


def _check_trace_version(data: dict, path: str) -> None:
    """Warn (never fail) when a trace's format version isn't ours."""
    import warnings

    found = data.get("version", 1)
    if found > TRACE_VERSION:
        warnings.warn(
            f"{path} was written by a newer loom (trace version {found}, this loom "
            f"reads {TRACE_VERSION}); upgrade loom-harness if anything looks off.",
            stacklevel=3,
        )
    elif found < TRACE_VERSION:
        warnings.warn(
            f"{path} uses trace version {found} (current: {TRACE_VERSION}); effect keys "
            f"were computed differently, so strict replay and `loom impact` will report "
            f"inputs-differ. Re-record the trace, or use replay(strict=False).",
            stacklevel=3,
        )


class Run:
    """The result of ``Agent.run`` -- and the handle for time travel."""

    def __init__(
        self,
        agent: Any,
        recorder: Recorder,
        context: Context,
        prompt: str,
        output: str,
        truncated: bool = False,
        episodes: "list[str] | None" = None,
        paused: bool = False,
        pending: "str | None" = None,
        pending_depth: int = 0,
        stop_reason: str = "",
    ):
        self.agent = agent
        self.recorder = recorder
        self.context = context
        self.prompt = prompt
        self.output = output
        self.truncated = truncated
        self.episodes = episodes or [prompt]
        self.paused = paused
        self.pending = pending
        self.pending_depth = pending_depth
        self.stop_reason = stop_reason  # "" | "budget" -- why the run stopped early
        self.healed_by: "str | None" = None  # set on branches returned by heal()
        self.regression_path: "str | None" = None  # where heal() saved this branch

    # -- inspection -------------------------------------------------------

    @property
    def log(self) -> list[EffectEntry]:
        return self.recorder.log

    @property
    def num_turns(self) -> int:
        """Number of top-level model calls (the boundaries fork rewinds to)."""
        return sum(1 for e in self.log if e.kind == "model" and e.depth == 0)

    @property
    def num_model_calls(self) -> int:
        """Total model calls at every nesting level (including subagents)."""
        return sum(1 for e in self.log if e.kind == "model")

    @property
    def parsed(self) -> Any:
        """The final answer parsed as the agent's ``output_type``.

        None when the agent has no output_type, when the run is paused or
        truncated, or when validation retries ran out (stop_reason
        "invalid_output"). Recomputed from the recorded text, so it works on
        loaded and replayed traces alike.
        """
        output_type = getattr(self.agent, "output_type", None) if self.agent else None
        if output_type is None or self.paused or self.truncated or self.stop_reason:
            return None
        from .structured import parse_as

        return parse_as(output_type, self.output)

    def timeline(self) -> list[dict]:
        """A human-readable step-by-step summary of the run, with nesting depth."""
        out: list[dict] = []
        turn = 0
        for e in self.log:
            depth = e.depth
            if e.kind == "model":
                resp = ModelResponse.from_dict(e.result)
                if resp.tool_calls:
                    detail = "calls " + ", ".join(
                        f"{tc.name}({json.dumps(tc.input)})" for tc in resp.tool_calls
                    )
                else:
                    detail = (resp.text[:80] + "...") if len(resp.text) > 80 else resp.text
                out.append(
                    {"step": e.seq, "turn": turn, "depth": depth, "kind": "model", "detail": detail}
                )
                if depth == 0:
                    turn += 1
            else:  # tool
                result = e.result if isinstance(e.result, str) else json.dumps(e.result)
                detail = (result[:80] + "...") if len(result) > 80 else result
                out.append(
                    {"step": e.seq, "turn": turn - 1, "depth": depth, "kind": e.kind, "detail": detail}
                )
        return out

    def print_timeline(self) -> None:
        for row in self.timeline():
            indent = "  " * row["depth"]
            marker = "> " * row["depth"]
            print(
                f"  [{row['step']:>2}] {indent}{marker}{row['kind']:<14} {row['detail']}"
            )

    def cost(self, since: int = 0) -> dict:
        """Aggregate token usage across model calls.

        ``since`` restricts the sum to effects at seq >= since -- useful to
        measure only what a fork spent after its divergence point.
        """
        inp = out = 0
        for e in self.log:
            if e.kind == "model" and e.seq >= since:
                u = ModelResponse.from_dict(e.result).usage
                inp += u.get("input_tokens", 0)
                out += u.get("output_tokens", 0)
        return {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out}

    # -- persistence ------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "version": TRACE_VERSION,
            "model": self.agent.model,
            "system": self.agent.system,
            "prompt": self.prompt,
            "episodes": self.episodes,
            "output": self.output,
            "truncated": self.truncated,
            "paused": self.paused,
            "pending": self.pending,
            "pending_depth": self.pending_depth,
            "stop_reason": self.stop_reason,
            "log": [e.to_dict() for e in self.log],
            **({"healed_by": self.healed_by} if self.healed_by else {}),
        }

    def save(self, path: str) -> None:
        """Write the trace to a git-friendly JSON file (content-checksummed)."""
        data = self.to_dict()
        data["checksum"] = trace_checksum(data)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str, agent: "Any | None" = None) -> "Run":
        """Load a trace. Pass the same ``agent`` to replay/fork it live."""
        with open(path) as f:
            data = json.load(f)
        _check_trace_version(data, path)
        _check_integrity(data, path)
        log = [EffectEntry.from_dict(e) for e in data["log"]]
        rec = Recorder.replay(log)
        run = cls(
            agent=agent,
            recorder=rec,
            context=Context(),
            prompt=data["prompt"],
            output=data["output"],
            truncated=data.get("truncated", False),
            episodes=data.get("episodes"),
            paused=data.get("paused", False),
            pending=data.get("pending"),
            pending_depth=data.get("pending_depth", 0),
            stop_reason=data.get("stop_reason", ""),
        )
        run._loaded_model = data.get("model")
        run._loaded_system = data.get("system", "")
        run.healed_by = data.get("healed_by")
        return run

    @classmethod
    def recover(
        cls,
        journal_path: str,
        agent: Any,
        resume: bool = True,
        on_unfinished: str = "fail",
    ) -> "Run":
        """Recover a run from its write-ahead journal after a crash.

        The journaled prefix replays for free (no API calls, no re-executed
        tools), then the run continues live from the exact crash point. This is
        idempotent: recovering a journal of a run that actually finished simply
        replays it to the same result. Pass ``resume=False`` to get the partial
        run back for inspection without continuing it.

        If the journal ends in an unfinished TOOL intent -- the process died
        after starting a tool but before its result hit disk -- the tool may or
        may not have run, and only the outside world knows. Continuing would
        execute it (again?), so recovery raises ``UnfinishedEffect`` by
        default. Check the side effect yourself, then recover with
        ``on_unfinished="retry"`` to accept the re-execution. Harness-internal
        effects (model calls, memory, compaction...) are safe to retry and
        recover without ceremony.
        """
        from .journal import Journal, UnfinishedEffect

        header, log, unfinished = Journal.read_full(journal_path)
        episodes = list(header.get("episodes") or [""])
        if not resume:
            return cls(
                agent=agent,
                recorder=Recorder.replay(log),
                context=Context(),
                prompt=episodes[0],
                output="",
                truncated=True,
                episodes=episodes,
            )
        dangling = [d for d in unfinished if d.get("kind", "").startswith("tool:")]
        if dangling and on_unfinished != "retry":
            d = dangling[-1]
            raise UnfinishedEffect(
                f"journal ends with {d['kind']!r} (seq {d.get('seq')}) started but not "
                f"finished: the crash landed between executing the tool and persisting "
                f"its result, so it may or may not have run. Verify the side effect, "
                f'then Run.recover(..., on_unfinished="retry") to re-execute it.'
            )
        rec = Recorder.fork(log, at=len(log))
        return agent.run(episodes, recorder=rec)

    # -- time travel ------------------------------------------------------

    def replay(self, strict: bool = True) -> "Run":
        """Re-run deterministically from the log -- zero API calls.

        Strict by default: every effect's inputs are re-derived and verified
        against the recording, so a passing replay proves the agent as
        configured now is equivalent to the one that recorded -- a prompt or
        tool-schema change surfaces as ``ReplayMismatch`` at the first
        differing effect. ``strict=False`` merely walks the old log (useful
        for inspecting a trace with a deliberately changed configuration).
        """
        self._require_agent("replay")
        rec = Recorder.replay(self.log, strict=strict)
        return self.agent.run(self.episodes, recorder=rec)

    def ask(self, prompt: str) -> "Run":
        """Continue the conversation with a new user message.

        The entire recorded history replays for free; only the new exchange
        runs live. The result is one longer trace, so the whole conversation
        stays replayable, forkable, sweepable, and diffable.
        """
        self._require_agent("ask")
        if self.paused:
            raise ValueError("run is paused; answer it with resume() first")
        rec = Recorder.fork(self.log, at=len(self.log))
        return self.agent.run([*self.episodes, prompt], recorder=rec)

    def resume(self, answer: str) -> "Run":
        """Answer a paused run's pending question and continue it.

        The answer is appended to the log as a recorded ``"human"`` effect;
        the whole prefix (including the answer) replays for free, then the run
        continues live from exactly where it paused.
        """
        self._require_agent("resume")
        if not self.paused:
            raise ValueError("run is not paused")
        entry = EffectEntry(
            seq=len(self.log),
            kind="human",
            key="resumed",
            result=str(answer),
            depth=self.pending_depth,
        )
        rec = Recorder.fork([*self.log, entry], at=len(self.log) + 1)
        return self.agent.run(self.episodes, recorder=rec)

    def fork(
        self, at: int, edit: "Callable[[Context], None] | None" = None
    ) -> "Run":
        """Rewind to the start of turn ``at``, optionally edit context, continue live.

        Turns 0..at-1 are replayed from the log; then ``edit`` (if given) mutates
        the context, and the agent runs live from there -- a new branch.
        """
        self._require_agent("fork")
        seqs = self.recorder.model_seqs()
        if at < 0 or at >= len(seqs):
            raise IndexError(f"fork turn {at} out of range (run has {len(seqs)} turns)")
        replay_until = seqs[at]  # seq of the at-th model call
        rec = Recorder.fork(self.log, replay_until)
        return self.agent.run(self.episodes, recorder=rec, _edit=edit, _edit_at_turn=at)

    def sweep(
        self,
        at: int,
        variants: "list[Callable[[Context], None] | None]",
        labels: "list[str] | None" = None,
    ) -> "SweepResult":
        """Fork this run once per variant and compare the branches side by side.

        Every branch replays turns 0..at-1 from the log for free; only each
        divergent tail runs live. That makes counterfactual experiments cheap:
        N variants of a 20-turn run forked at turn 18 pay for N x 2 turns, not
        N x 20. Pass ``None`` as a variant for a no-edit control branch.
        """
        self._require_agent("sweep")
        seqs = self.recorder.model_seqs()
        if at < 0 or at >= len(seqs):
            raise IndexError(f"sweep turn {at} out of range (run has {len(seqs)} turns)")
        if labels is not None and len(labels) != len(variants):
            raise ValueError(f"{len(labels)} labels for {len(variants)} variants")
        runs = [self.fork(at=at, edit=v) for v in variants]
        return SweepResult(
            base=self,
            runs=runs,
            labels=list(labels) if labels else [f"v{i}" for i in range(len(variants))],
            fork_seq=seqs[at],
        )

    def diff(self, other: "Run") -> Any:
        """Compare this run's effect log with another's. See ``loom.diff``."""
        from .diff import diff_logs

        return diff_logs(self.log, other.log)

    def intents(self) -> list[dict]:
        """What the agent did (or tried to do) with tools, with policy outcomes.

        Statuses: ``executed`` | ``stubbed`` (dry-run) | ``blocked`` (denied or
        rejected). This is how you audit a dry run before granting real access.
        """
        out: list[dict] = []
        for e in self.log:
            if not e.kind.startswith("tool:"):
                continue
            status = "executed"
            if isinstance(e.result, str):
                if e.result.startswith("DRY-RUN:"):
                    status = "stubbed"
                elif e.result.startswith("BLOCKED:"):
                    status = "blocked"
            out.append({"tool": e.kind[5:], "status": status, "seq": e.seq})
        return out

    def proceed(self) -> "Run":
        """Continue an interrupted run (budget stop, max_steps truncation).

        Replays the whole recorded prefix for free and picks up live from the
        stopping point -- raise the agent's policy budget or max_steps first,
        or it will stop again at the same limit.
        """
        self._require_agent("proceed")
        rec = Recorder.fork(self.log, at=len(self.log))
        return self.agent.run(self.episodes, recorder=rec)

    def rerun(self, model: Any = None, system: "str | None" = None) -> "Run":
        """Re-run the same conversation live on a different model or system prompt.

        Same tools, same episodes, fresh trace -- then ``run.diff(other)`` shows
        exactly where and why the two models diverged. A/B testing for agents.
        """
        from .agent import Agent

        agent = Agent(
            model=model if model is not None else self.agent.provider,
            tools=list(self.agent.tools.values()),
            system=self.agent.system if system is None else system,
            max_steps=self.agent.max_steps,
            budget=self.agent.budget,
            name=self.agent.name,
            on_human=self.agent.on_human,
            parallel_tools=self.agent.parallel_tools,
            policy=self.agent.policy,
        )
        return agent.run(self.episodes)

    def checkup(self) -> Any:
        """Inspect this run's context for rot (oversized/unused/duplicate items).

        Returns a ``loom.health.HealthReport``; its ``experiments()`` produce
        sweep-ready repair edits, and ``heal()`` drives the whole loop.
        """
        from .health import analyze

        return analyze(self.episodes, self.log)

    def heal(
        self,
        check: Callable[[str], bool],
        at: "int | None" = None,
        variants: "list | None" = None,
        labels: "list[str] | None" = None,
        regression_dir: "str | None" = None,
    ) -> "Run | None":
        """Try to fix a failing run by testing context repairs, cheapest first.

        If ``check(self.output)`` already passes, returns self. Otherwise runs
        each experiment from ``checkup()`` (or the provided ``variants``) as a
        fork at turn ``at`` (default: the final turn), and returns the first
        branch whose output passes -- with ``healed_by`` naming the repair.
        Returns None if no experiment fixed it. Only divergent tails run live.

        With ``regression_dir``, every repair also grows your test suite: the
        healed branch is saved there as a golden trace (content-addressed
        filename, so re-healing is idempotent), ready for ``loom test`` and
        ``verify_replay``. The branch's ``regression_path`` says where.
        """
        self._require_agent("heal")
        if check(self.output):
            return self
        if variants is None:
            labels, variants = self.checkup().experiments()
        if labels is None:
            labels = [f"v{i}" for i in range(len(variants))]
        if at is None:
            at = max(0, self.num_turns - 1)
        for label, edit in zip(labels, variants):
            branch = self.fork(at=at, edit=edit)
            if not branch.paused and check(branch.output):
                branch.healed_by = label
                if regression_dir is not None:
                    data = branch.to_dict()
                    digest = hashlib.sha1(
                        json.dumps(data, sort_keys=True).encode()
                    ).hexdigest()[:10]
                    os.makedirs(regression_dir, exist_ok=True)
                    path = os.path.join(regression_dir, f"healed-{digest}.loom.json")
                    branch.save(path)
                    branch.regression_path = path
                return branch
        return None

    def bisect(self, check: Callable[[str], bool]) -> int:
        """Find the first turn whose assistant text fails ``check``.

        Walks the recorded model responses in order (no re-run needed) and
        returns the 1-based turn index where ``check`` first returns False,
        or -1 if every turn passes. This is how you locate where a run went bad.
        """
        turn = 0
        for e in self.log:
            if e.kind != "model":
                continue
            turn += 1
            text = ModelResponse.from_dict(e.result).text
            if not check(text):
                return turn
        return -1

    # -- helpers ----------------------------------------------------------

    def _require_agent(self, op: str) -> None:
        if self.agent is None:
            raise ValueError(
                f"{op}() needs the agent; load the trace with Run.load(path, agent=<agent>)"
            )


class SweepResult:
    """The branches produced by ``Run.sweep``, with comparison helpers."""

    def __init__(self, base: Run, runs: list[Run], labels: list[str], fork_seq: int):
        self.base = base
        self.runs = runs
        self.labels = labels
        self.fork_seq = fork_seq

    def __iter__(self):
        return iter(zip(self.labels, self.runs))

    def compare(self) -> list[dict]:
        """One row per branch (base first): output, turns, live spend, divergence."""
        from .diff import diff_logs

        rows = [
            {
                "label": "base",
                "output": self.base.output,
                "turns": self.base.num_turns,
                "live_tokens": 0,  # the base is already recorded; branches spend anew
                "diverged_at": None,
            }
        ]
        for label, run in zip(self.labels, self.runs):
            rows.append(
                {
                    "label": label,
                    "output": run.output,
                    "turns": run.num_turns,
                    # Only what this branch spent after the fork point.
                    "live_tokens": run.cost(since=self.fork_seq)["total_tokens"],
                    "diverged_at": diff_logs(self.base.log, run.log).first_divergence,
                }
            )
        return rows

    def print_compare(self) -> None:
        for row in self.compare():
            out = row["output"]
            out = (out[:56] + "...") if len(out) > 56 else out
            div = row["diverged_at"] if row["diverged_at"] is not None else "-"
            print(
                f"  {row['label']:<10} turns={row['turns']:<3} "
                f"live_tokens={row['live_tokens']:<6} diverged_at={div:<4} {out}"
            )
