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


def score_breakdown(trace: dict) -> dict:
    """An explainable behavior scorecard (each dimension 0-100, higher = safer).

    Not a toy number: every dimension says WHY it landed where it did, so a PR
    reviewer sees whether the score moved because of new risk, an irreversible
    action, an ungated call, an unsupported claim, or cost. Lighthouse for
    agent behavior.
    """
    from .action import actions as _actions
    from .impact import _SCORE_WEIGHTS
    from .packs import install_builtin, undo_plan

    install_builtin()
    acts = _actions(trace)
    calls = [a for a in acts if a.type == "call" and a.step >= 0]

    # security: risk categories exercised (the headline number).
    exercised = {a.risk for a in calls if a.risk}
    security = max(0, 100 - sum(_SCORE_WEIGHTS.get(c, 0) for c in exercised))

    # external side effect: how much the run reached OUT of the sandbox.
    external = [a for a in calls if "external_side_effect" in a.capabilities]
    ext_score = max(0, 100 - 12 * len(external))

    # reversibility: of the state-changing actions, how many can be undone?
    changing = [a for a in calls if a.state_diff is not None or (set(a.capabilities) & {"write", "destructive"})]
    reversible = sum(1 for a in changing
                     if (p := undo_plan(a, trace)) is not None and p.reversible)
    rev_score = round(100 * reversible / len(changing)) if changing else 100

    # policy coverage: of the risky actions, how many passed a firewall decision?
    risky = [a for a in calls if a.risky]
    if risky:
        gated = sum(1 for a in risky if a.policy is not None)
        pol_score = round(100 * gated / len(risky))
    else:
        pol_score = 100

    # evidence coverage: of the final answer's claims, how many are supported?
    from .insight import provenance

    claims = provenance(trace)
    if claims:
        supported = sum(1 for c in claims if c["evidence"])
        ev_score = round(100 * supported / len(claims))
    else:
        ev_score = 100

    # cost: cheap runs score high; ~one point per 1k tokens over a soft budget.
    tokens = 0
    for e in trace.get("log", []):
        if e.get("kind") == "model" and isinstance(e.get("result"), dict):
            u = e["result"].get("usage") or {}
            tokens += (u.get("input_tokens", 0) or 0) + (u.get("output_tokens", 0) or 0)
    cost_score = max(0, 100 - max(0, (tokens - 20_000)) // 1000)

    dims = {
        "security": {"score": security,
                     "why": ("no risky action" if not exercised
                             else "exercised " + ", ".join(sorted(exercised)))},
        "external_side_effect": {"score": ext_score,
                                 "why": f"{len(external)} action(s) reached off-box"},
        "reversibility": {"score": rev_score,
                          "why": (f"{len(changing) - reversible} of {len(changing)} "
                                  "world-changing action(s) can't be undone"
                                  if changing else "nothing changed the world")},
        "policy_coverage": {"score": pol_score,
                            "why": (f"{len(risky)} risky action(s), "
                                    f"{sum(1 for a in risky if a.policy is None)} ungated"
                                    if risky else "no risky actions to gate")},
        "evidence_coverage": {"score": ev_score,
                             "why": (f"{sum(1 for c in claims if not c['evidence'])} of "
                                     f"{len(claims)} claim(s) unsupported" if claims
                                     else "no claims to support")},
        "cost": {"score": cost_score, "why": f"{tokens:,} tokens"},
    }
    overall = round(sum(d["score"] for d in dims.values()) / len(dims))
    return {"overall": overall, "dimensions": dims}


def score_badge_svg(overall: int, label: str = "agent safety") -> str:
    """A shields.io-style SVG badge for the behavior score (self-contained)."""
    color = "#4c1" if overall >= 90 else "#97ca00" if overall >= 75 else \
            "#dfb317" if overall >= 60 else "#fe7d37" if overall >= 40 else "#e05d44"
    value = f"{overall}/100"
    lw, vw = 6 * len(label) + 12, 6 * len(value) + 12
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{lw + vw}" height="20" role="img" aria-label="{label}: {value}">
  <linearGradient id="s" x2="0" y2="100%"><stop offset="0" stop-color="#bbb" stop-opacity=".1"/><stop offset="1" stop-opacity=".1"/></linearGradient>
  <clipPath id="r"><rect width="{lw + vw}" height="20" rx="3" fill="#fff"/></clipPath>
  <g clip-path="url(#r)">
    <rect width="{lw}" height="20" fill="#555"/>
    <rect x="{lw}" width="{vw}" height="20" fill="{color}"/>
    <rect width="{lw + vw}" height="20" fill="url(#s)"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" font-size="11">
    <text x="{lw / 2}" y="14">{label}</text>
    <text x="{lw + vw / 2}" y="14">{value}</text>
  </g>
</svg>"""


def score_markdown(b: dict) -> str:
    """A PR-comment line + table for the scorecard."""
    lines = [f"### 🧭 Agent behavior score: **{b['overall']}/100**", "",
             "| dimension | score | why |", "|---|---:|---|"]
    for name, d in b["dimensions"].items():
        lines.append(f"| {name.replace('_', ' ')} | {d['score']} | {d['why']} |")
    return "\n".join(lines)


def describe_score(b: dict) -> str:
    lines = [f"behavior score: {b['overall']}/100"]
    for name, d in b["dimensions"].items():
        bar = "█" * (d["score"] // 10) + "·" * (10 - d["score"] // 10)
        lines.append(f"  {name:<20} {d['score']:>3}  {bar}  {d['why']}")
    return "\n".join(lines)


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
    bd_a, bd_b = score_breakdown(trace_a), score_breakdown(trace_b)
    risks_a = {x.risk for x in calls_a if x.risk}
    risks_b = {x.risk for x in calls_b if x.risk}
    # Which score dimensions moved, and by how much -- the "why score changed".
    moved = []
    for name in bd_a["dimensions"]:
        da, db = bd_a["dimensions"][name]["score"], bd_b["dimensions"][name]["score"]
        if da != db:
            moved.append({"dimension": name, "a": da, "b": db, "why": bd_b["dimensions"][name]["why"]})
    return {
        "added": rows(added),
        "removed": rows(removed),
        "risk_gained": sorted(risks_b - risks_a),
        "risk_dropped": sorted(risks_a - risks_b),
        "score": {"a": bd_a["overall"], "b": bd_b["overall"]},
        "score_moved": sorted(moved, key=lambda m: m["b"] - m["a"]),
        "calls": {"a": len(calls_a), "b": len(calls_b)},
        "labels": {c: _RISK_LABELS.get(c, c) for c in (risks_a | risks_b)},
    }


def describe_action_diff(d: dict) -> str:
    """Human/PR-comment rendering of a ``diff_actions`` result."""
    lines = []
    sa, sb = d["score"]["a"], d["score"]["b"]
    if sa != sb:
        arrow = "⬇" if sb < sa else "⬆"
        lines.append(f"behavior score: {sa} → {sb} {arrow}")
        for m in d.get("score_moved", []):
            mv = "⬇" if m["b"] < m["a"] else "⬆"
            lines.append(f"    {m['dimension']}: {m['a']} → {m['b']} {mv}  ({m['why']})")
    else:
        lines.append(f"behavior score: {sa} (unchanged)")
    for row in d["added"]:
        risk = f"  ⚠ {row['risk']}" if row["risk"] else ""
        lines.append(f"  + {row['tool']} x{row['count']}{risk}")
    for row in d["removed"]:
        lines.append(f"  - {row['tool']} x{row['count']}")
    if not d["added"] and not d["removed"]:
        lines.append("  same actions on both sides "
                     f"({d['calls']['a']} vs {d['calls']['b']} calls)")
    return "\n".join(lines)
