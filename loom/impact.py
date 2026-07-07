"""Impact analysis: which recorded runs does a config change affect?

Snapshot testing for agents. You changed the system prompt, swapped a tool,
or reordered instructions -- before paying for a single API call, replay your
trace corpus against the new configuration and see exactly which runs are
touched and where:

    impacts = assess(["fixtures/a.loom.json", ...], agent=new_agent)
    print(report(impacts))

Dry mode (default) is free: it walks each trace in replay, recomputing every
effect's input key under the new config and comparing it with the recorded
key. The first mismatch is the first turn your change reaches. Live mode
(``live=True``) actually re-runs each conversation and diffs the outputs --
that is what it costs to know how behavior changes, not just where.
"""

from __future__ import annotations

from dataclasses import dataclass

from .effect import Recorder, ReplayExhausted, ReplayMismatch, _key


class _KeyCheckRecorder(Recorder):
    """A strict replayer that also notes every effect whose inputs changed."""

    def __init__(self, log):
        super().__init__(log=log, replay_until=len(log), allow_live=False)
        self.mismatches: "list[tuple[int, str]]" = []  # (seq, kind)

    def run(self, kind, payload, fn, encode=lambda x: x, decode=lambda x: x):
        seq = self._cursor
        if seq < self.replay_until:
            entry = self.log[seq]
            if entry.kind == kind and _key([kind, payload]) != entry.key:
                self.mismatches.append((seq, kind))
        return super().run(kind, payload, fn, encode=encode, decode=decode)


@dataclass
class Impact:
    """How one recorded trace is affected by the new configuration."""

    path: str
    verdict: str  # "unchanged" | "inputs-differ" | "structure-differs" | "outputs-differ"
    detail: str
    first_seq: "int | None" = None

    @property
    def changed(self) -> bool:
        return self.verdict != "unchanged"

    def describe(self) -> str:
        where = f" (first at seq {self.first_seq})" if self.first_seq is not None else ""
        return f"{self.verdict:<16} {self.path}{where}\n    {self.detail}"


def assess_trace(path: str, agent, live: bool = False) -> Impact:
    """Assess one saved trace against ``agent``'s current configuration."""
    from .trace import Run

    original = Run.load(path, agent=agent)
    if live:
        return _assess_live(path, original, agent)

    rec = _KeyCheckRecorder(original.log)
    try:
        agent.run(original.episodes, recorder=rec)
    except ReplayMismatch as e:
        return Impact(path, "structure-differs", str(e), first_seq=rec.cursor)
    except ReplayExhausted as e:
        return Impact(path, "structure-differs", str(e), first_seq=rec.cursor)
    if rec.cursor < len(original.log):
        return Impact(
            path,
            "structure-differs",
            f"new config finishes after {rec.cursor} of {len(original.log)} recorded effects",
            first_seq=rec.cursor,
        )
    if rec.mismatches:
        seq, kind = rec.mismatches[0]
        return Impact(
            path,
            "inputs-differ",
            f"{len(rec.mismatches)} effect(s) see different inputs, starting with {kind!r}",
            first_seq=seq,
        )
    return Impact(path, "unchanged", "every recorded effect gets identical inputs")


def _assess_live(path: str, original, agent) -> Impact:
    from .diff import diff_logs

    fresh = agent.run(original.episodes)
    diff = diff_logs(original.log, fresh.log)
    if fresh.output == original.output and diff.identical:
        return Impact(path, "unchanged", "re-run produced an identical trace")
    if fresh.output == original.output:
        return Impact(
            path,
            "inputs-differ",
            "same final output via a different path",
            first_seq=diff.first_divergence,
        )
    return Impact(
        path,
        "outputs-differ",
        f"output changed:\n      was: {original.output[:80]!r}\n      now: {fresh.output[:80]!r}",
        first_seq=diff.first_divergence,
    )


def assess(paths: list[str], agent, live: bool = False) -> list[Impact]:
    """Assess a corpus of saved traces. See ``assess_trace``."""
    return [assess_trace(p, agent, live=live) for p in paths]


def report(impacts: list[Impact]) -> str:
    """Human-readable impact report."""
    changed = [i for i in impacts if i.changed]
    lines = [i.describe() for i in impacts]
    lines.append("")
    lines.append(f"{len(changed)} of {len(impacts)} recorded run(s) affected")
    return "\n".join(lines)
