"""Offline debugger insights: why / side-effect map / provenance / flakiness.

Everything here is computed from saved traces -- no model, no API calls:

  why_action        "why did it do THAT?" -- the model's stated intent plus
                    the earlier observations the action most plausibly drew on
  provenance        each claim in the final answer linked to the tool results
                    that support it (evidence, not vibes)
  side_effect_map   one view of everything the run changed or reached: files,
                    database, browser, records, network
  causality_tree    who did what: the delegation tree across subagent depths
  flakiness         same task recorded N times: at which step do runs diverge?

The LLM-powered ``loom why`` remains for open questions; these answer the
common ones instantly and deterministically.
"""

from __future__ import annotations

import re
from typing import Any

from .action import Action, actions

_WORD = re.compile(r"[A-Za-z0-9_./-]{4,}")


def _words(text: str) -> "set[str]":
    return {w.lower() for w in _WORD.findall(str(text))}


def _snippet(text: str, n: int = 110) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


# -- why this action ---------------------------------------------------------

def why_action(source: Any, step: int) -> dict:
    """Explain one action: intent, risk, policy, and the evidence trail.

    Evidence = earlier observations (tool results, human answers) sharing the
    most distinctive words with this action's input -- the observations the
    model most plausibly acted on. Heuristic and honest about it: entries are
    ranked candidates, not proofs.
    """
    acts = actions(source)
    target = next((a for a in acts if a.step == step), None)
    if target is None:
        raise ValueError(f"no action at step {step}")
    probe = _words(target.input) | _words(target.tool)
    evidence = []
    for a in acts:
        if a.step >= step or a.step < 0 or a.observation is None:
            continue
        if a.type not in ("call", "ask-human"):
            continue
        overlap = probe & _words(a.observation.text)
        if overlap:
            evidence.append((len(overlap), a))
    evidence.sort(key=lambda pair: -pair[0])
    return {
        "step": step,
        "tool": target.tool,
        "type": target.type,
        "intent": target.intent,
        "risk": target.risk,
        "capabilities": target.capabilities,
        "policy": target.policy.to_dict() if target.policy else None,
        "evidence": [
            {"step": a.step, "tool": a.tool or a.type,
             "snippet": _snippet(a.observation.text)}
            for _, a in evidence[:3]
        ],
    }


def describe_why(w: dict) -> str:
    lines = [f"[{w['step']}] {w['tool'] or w['type']}"]
    if w["intent"]:
        lines.append(f'  stated intent: "{_snippet(w["intent"], 160)}"')
    if w["risk"]:
        lines.append(f"  risk: {w['risk']}  capabilities: {', '.join(w['capabilities'])}")
    if w["policy"]:
        p = w["policy"]
        lines.append(f"  firewall: {p['action']}" + (f" ({p['rule']})" if p.get("rule") else ""))
    if w["evidence"]:
        lines.append("  drew on (ranked candidates):")
        for e in w["evidence"]:
            lines.append(f"    [{e['step']}] {e['tool']}: {e['snippet']}")
    else:
        lines.append("  no earlier observation overlaps this input -- the model "
                     "acted on the prompt/context alone")
    return "\n".join(lines)


# -- claim provenance --------------------------------------------------------

def provenance(source: Any) -> "list[dict]":
    """Link each claim (sentence) of the final answer to supporting tool results."""
    acts = actions(source)
    answers = [a for a in acts if a.type == "answer" and a.intent]
    if not answers:
        return []
    final = answers[-1].intent
    observations = [
        a for a in acts
        if a.type == "call" and a.observation is not None and a.observation.text
    ]
    out = []
    for raw in re.split(r"(?<=[.!?])\s+|\n+", final):
        claim = raw.strip()
        cw = _words(claim)
        if len(cw) < 3:
            continue
        scored = []
        for a in observations:
            ow = _words(a.observation.text)
            overlap = cw & ow
            if len(overlap) >= 2:
                scored.append((len(overlap) / len(cw | ow), a))
        scored.sort(key=lambda pair: -pair[0])
        out.append({
            "claim": _snippet(claim, 140),
            "evidence": [
                {"step": a.step, "tool": a.tool,
                 "snippet": _snippet(a.observation.text)}
                for _, a in scored[:2]
            ],
        })
    return out


# -- side-effect map ----------------------------------------------------------

def side_effect_map(source: Any) -> dict:
    """Everything the run changed or reached, grouped by world.

    ``changes`` come from pack StateDiffs (file/database/dom/record/field);
    ``reached`` lists network touches; ``read`` counts pure reads. The
    at-a-glance answer to "what did this agent actually touch?"
    """
    from collections import Counter

    from .packs import install_builtin

    install_builtin()  # StateDiffs come from the domain packs
    changes: dict[str, Counter] = {}
    reached: Counter = Counter()
    reads = 0
    for a in actions(source):
        if a.type != "call" or a.step < 0:
            continue
        caps = set(a.capabilities)
        if a.state_diff is not None:
            changes.setdefault(a.state_diff.kind, Counter())[a.state_diff.summary] += 1
        elif "network" in caps:
            target = ""
            if isinstance(a.input, dict):
                target = str(a.input.get("url") or a.input.get("to") or "")
            reached[f"{a.tool}" + (f" -> {_snippet(target, 60)}" if target else "")] += 1
        elif caps <= {"read", "idempotent", "secret", "pii_access"} and caps:
            reads += 1
    return {
        "changes": {k: [s if n == 1 else f"{s} (x{n})" for s, n in c.most_common()]
                    for k, c in changes.items()},
        "reached": [s if n == 1 else f"{s} (x{n})" for s, n in reached.most_common()],
        "reads": reads,
    }


def describe_map(m: dict) -> str:
    lines = []
    label = {"file": "files", "database": "database", "dom": "browser",
             "record": "records", "field": "records"}
    for kind, items in sorted(m["changes"].items()):
        lines.append(f"{label.get(kind, kind)} changed:")
        lines += [f"  Δ {s}" for s in items]
    if m["reached"]:
        lines.append("network reached:")
        lines += [f"  ↗ {s}" for s in m["reached"]]
    if m["reads"]:
        lines.append(f"reads: {m['reads']} read-only call(s)")
    return "\n".join(lines) or "no side effects recorded"


# -- multi-agent causality tree ------------------------------------------------

def causality_tree(source: Any) -> str:
    """The delegation tree: which (sub)agent ran which actions, by depth."""
    lines = []
    for a in actions(source):
        if a.step < 0:
            continue
        indent = "  " * a.depth + ("└ " if a.depth else "")
        if a.type == "call":
            risk = f"  ⚠ {a.risk}" if a.risk else ""
            lines.append(f"[{a.step:>3}] {indent}{a.tool}{risk}")
        elif a.type == "answer":
            lines.append(f"[{a.step:>3}] {indent}✅ {_snippet(a.intent, 60)}")
        elif a.type == "reason" and a.depth > 0:
            lines.append(f"[{a.step:>3}] {indent}💭 {_snippet(a.intent, 60)}")
    return "\n".join(lines)


# -- flakiness across repeated runs ---------------------------------------------

def flakiness(traces: "list[dict]") -> dict:
    """Where do repeated runs of the same task diverge?

    The first trace is the baseline; every other is diffed against it, and
    the first-divergence steps make the heatmap. ``None`` = identical.
    """
    from collections import Counter

    from .diff import diff_logs
    from .effect import EffectEntry

    if len(traces) < 2:
        raise ValueError("flakiness needs at least two traces of the same task")
    logs = [[EffectEntry.from_dict(e) for e in t.get("log", [])] for t in traces]
    divergences = [diff_logs(logs[0], other).first_divergence for other in logs[1:]]
    hist = Counter(divergences)
    base_kinds = {e.seq: e.kind for e in logs[0]}
    return {
        "runs": len(traces),
        "identical": hist.pop(None, 0),
        "by_step": sorted(
            (step, n, base_kinds.get(step, "past-end")) for step, n in hist.items()
        ),
    }


def describe_flakiness(f: dict) -> str:
    total = f["runs"] - 1
    lines = [f"{f['runs']} run(s): {f['identical']}/{total} identical to the baseline"]
    if f["by_step"]:
        lines.append("first divergence (step -> runs):")
        peak = max(n for _, n, _ in f["by_step"])
        for step, n, kind in f["by_step"]:
            bar = "█" * max(1, int(24 * n / peak))
            lines.append(f"  step {step:>3} ({kind:<14}) {bar} {n}")
        worst = max(f["by_step"], key=lambda row: row[1])
        lines.append(f"flakiest step: {worst[0]} ({worst[2]}) -- "
                     f"{worst[1]}/{total} run(s) diverged there first")
    return "\n".join(lines)