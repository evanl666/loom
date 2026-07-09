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


# -- action-level behavior diff ----------------------------------------------

def run_risk_score(trace: dict) -> int:
    """0-100 score of the risk EXERCISED by a run (not just its tool surface).

    100 = no risky action taken; each risk category exercised deducts its
    weight once. Comparable across agent versions -- the PR-comment number.
    """
    from .action import actions as _actions
    from .impact import _SCORE_WEIGHTS

    exercised = {a.risk for a in _actions(trace) if a.type == "call" and a.risk}
    return max(0, 100 - sum(_SCORE_WEIGHTS.get(c, 0) for c in exercised))


def diff_actions(trace_a: dict, trace_b: dict) -> dict:
    """Behavior diff between two runs at the Action level.

    Answers the PR-review question: what does the new agent DO differently --
    which actions appeared, which disappeared, and how did exercised risk
    move? Actions are grouped by (tool, risk) so ten identical Reads collapse
    to one row with a count.
    """
    from collections import Counter

    from .action import actions as _actions
    from .impact import _RISK_LABELS
    from .packs import install_builtin

    install_builtin()
    calls_a = [x for x in _actions(trace_a) if x.type == "call"]
    calls_b = [x for x in _actions(trace_b) if x.type == "call"]
    sig_a = Counter((x.tool, x.risk) for x in calls_a)
    sig_b = Counter((x.tool, x.risk) for x in calls_b)

    def rows(counter):
        return [{"tool": t, "risk": r, "count": n}
                for (t, r), n in sorted(counter.items(), key=lambda kv: -kv[1])]

    added, removed = sig_b - sig_a, sig_a - sig_b
    score_a, score_b = run_risk_score(trace_a), run_risk_score(trace_b)
    risks_a = {x.risk for x in calls_a if x.risk}
    risks_b = {x.risk for x in calls_b if x.risk}
    return {
        "added": rows(added),
        "removed": rows(removed),
        "risk_gained": sorted(risks_b - risks_a),
        "risk_dropped": sorted(risks_a - risks_b),
        "score": {"a": score_a, "b": score_b},
        "calls": {"a": len(calls_a), "b": len(calls_b)},
        "labels": {c: _RISK_LABELS.get(c, c) for c in (risks_a | risks_b)},
    }


def describe_action_diff(d: dict) -> str:
    """Human/PR-comment rendering of a ``diff_actions`` result."""
    lines = []
    sa, sb = d["score"]["a"], d["score"]["b"]
    if sa != sb:
        arrow = "⬇" if sb < sa else "⬆"
        why = ""
        if d["risk_gained"]:
            why = " (" + ", ".join("+" + d["labels"].get(c, c) for c in d["risk_gained"]) + ")"
        lines.append(f"run risk score: {sa} → {sb} {arrow}{why}")
    else:
        lines.append(f"run risk score: {sa} (unchanged)")
    for row in d["added"]:
        risk = f"  ⚠ {row['risk']}" if row["risk"] else ""
        lines.append(f"  + {row['tool']} x{row['count']}{risk}")
    for row in d["removed"]:
        lines.append(f"  - {row['tool']} x{row['count']}")
    if not d["added"] and not d["removed"]:
        lines.append("  same actions on both sides "
                     f"({d['calls']['a']} vs {d['calls']['b']} calls)")
    return "\n".join(lines)
