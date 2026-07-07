"""Trace diff: find where two runs diverged, at the effect level.

Because every effect entry carries a ``key`` (hash of its inputs) and a
``result``, a diff can tell three different failure stories apart:

  * kinds-differ    -> control flow diverged (a different action was taken)
  * inputs-differ   -> same action, but the context/inputs drifted
  * results-differ  -> same action, same inputs, different outcome
                       (model nondeterminism or a changed backend)

This is the answer to "it worked yesterday, why is it broken today?" -- and
the regression tester for model or prompt upgrades: record a fixture, re-run
live, diff.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .effect import EffectEntry
from .providers.base import ModelResponse

IDENTICAL = "identical"
KINDS_DIFFER = "kinds-differ"
INPUTS_DIFFER = "inputs-differ"
RESULTS_DIFFER = "results-differ"
ONLY_A = "only-a"
ONLY_B = "only-b"


def _detail(e: EffectEntry, width: int = 90) -> str:
    """One-line human summary of an effect entry."""
    if e.kind == "model":
        resp = ModelResponse.from_dict(e.result)
        if resp.tool_calls:
            text = "calls " + ", ".join(
                f"{tc.name}({json.dumps(tc.input)})" for tc in resp.tool_calls
            )
        else:
            text = resp.text
    else:
        text = e.result if isinstance(e.result, str) else json.dumps(e.result)
    return (text[:width] + "...") if len(text) > width else text


@dataclass
class StepDiff:
    """The comparison of one step across two traces."""

    seq: int
    status: str  # one of the module-level status constants
    a: "EffectEntry | None"
    b: "EffectEntry | None"

    def describe(self) -> str:
        lines = [f"step {self.seq} [{self.status}]"]
        if self.a is not None:
            lines.append(f"  a {self.a.kind}: {_detail(self.a)}")
        if self.b is not None:
            lines.append(f"  b {self.b.kind}: {_detail(self.b)}")
        return "\n".join(lines)


@dataclass
class TraceDiff:
    """The full comparison of two effect logs."""

    steps: list[StepDiff]

    @property
    def first_divergence(self) -> "int | None":
        """Seq of the first non-identical step, or None if fully identical."""
        for s in self.steps:
            if s.status != IDENTICAL:
                return s.seq
        return None

    @property
    def identical(self) -> bool:
        return self.first_divergence is None

    def counts(self) -> dict:
        out: dict = {}
        for s in self.steps:
            out[s.status] = out.get(s.status, 0) + 1
        return out

    def summary(self) -> str:
        if self.identical:
            return f"traces are identical ({len(self.steps)} steps)"
        first = self.first_divergence
        prefix = sum(1 for s in self.steps if s.seq < first)
        lines = [f"identical prefix: {prefix} step(s)", "first divergence:"]
        for s in self.steps:
            if s.seq == first:
                lines.append("  " + s.describe().replace("\n", "\n  "))
                break
        counts = ", ".join(f"{v} {k}" for k, v in sorted(self.counts().items()))
        lines.append(f"totals: {counts}")
        return "\n".join(lines)


def diff_logs(a: list[EffectEntry], b: list[EffectEntry]) -> TraceDiff:
    """Compare two effect logs step by step."""
    steps: list[StepDiff] = []
    for i in range(max(len(a), len(b))):
        ea = a[i] if i < len(a) else None
        eb = b[i] if i < len(b) else None
        if ea is None:
            status = ONLY_B
        elif eb is None:
            status = ONLY_A
        elif ea.kind != eb.kind:
            status = KINDS_DIFFER
        elif ea.key != eb.key:
            status = INPUTS_DIFFER
        elif ea.result != eb.result:
            status = RESULTS_DIFFER
        else:
            status = IDENTICAL
        steps.append(StepDiff(seq=i, status=status, a=ea, b=eb))
    return TraceDiff(steps=steps)
