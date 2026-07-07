"""The Run object: a recorded agent run you can inspect, replay, fork, and bisect.

The trace is the product. Everything a Run can do falls out of the effect log
captured through the Effect boundary.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from .context import Context
from .effect import EffectEntry, Recorder
from .providers.base import ModelResponse

TRACE_VERSION = 1


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
            "log": [e.to_dict() for e in self.log],
        }

    def save(self, path: str) -> None:
        """Write the trace to a git-friendly JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str, agent: "Any | None" = None) -> "Run":
        """Load a trace. Pass the same ``agent`` to replay/fork it live."""
        with open(path) as f:
            data = json.load(f)
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
        )
        run._loaded_model = data.get("model")
        run._loaded_system = data.get("system", "")
        return run

    # -- time travel ------------------------------------------------------

    def replay(self) -> "Run":
        """Re-run deterministically from the log -- zero API calls.

        The result must match the original; if the log is exhausted the run has
        diverged (a bug), which surfaces as ``ReplayExhausted``.
        """
        self._require_agent("replay")
        rec = Recorder.replay(self.log)
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
