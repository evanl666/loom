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
    return why_from_actions(actions(source), step)


def why_from_actions(acts: "list[Action]", step: int) -> dict:
    """``why_action`` over an already-built Action list (avoids re-lifting)."""
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
    top = evidence[0][0] if evidence else 0
    # Honest confidence: a strong word-overlap with an earlier observation is
    # "medium" (it's still correlation, not proof); a stated intent alone is
    # "low"; nothing at all is "none". Never "high" -- this is a heuristic.
    if top >= 3:
        confidence = "medium"
    elif top >= 1 or target.intent.strip():
        confidence = "low"
    else:
        confidence = "none"
    return {
        "step": step,
        "tool": target.tool,
        "type": target.type,
        "intent": target.intent,
        "risk": target.risk,
        "capabilities": target.capabilities,
        "policy": target.policy.to_dict() if target.policy else None,
        "confidence": confidence,
        "evidence": [
            {"step": a.step, "tool": a.tool or a.type,
             "snippet": _snippet(a.observation.text), "shared": n}
            for n, a in evidence[:3]
        ],
        "missing_evidence": not evidence,
    }


def describe_why(w: dict) -> str:
    lines = [f"[{w['step']}] {w['tool'] or w['type']}  (confidence: {w.get('confidence', '?')})"]
    if w["intent"]:
        lines.append(f'  stated intent: "{_snippet(w["intent"], 160)}"')
    if w["risk"]:
        lines.append(f"  risk: {w['risk']}  capabilities: {', '.join(w['capabilities'])}")
    if w["policy"]:
        p = w["policy"]
        lines.append(f"  firewall: {p['action']}" + (f" ({p['rule']})" if p.get("rule") else ""))
    if w["evidence"]:
        lines.append("  drew on (ranked candidates -- correlation, not proof):")
        for e in w["evidence"]:
            lines.append(f"    [{e['step']}] {e['tool']}: {e['snippet']}")
    else:
        lines.append("  ⚠ no earlier observation overlaps this input -- the model "
                     "acted on the prompt/context alone (explanation is weak)")
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


def evidence_coverage(source: Any) -> dict:
    """How well the final answer is backed by tool results.

    Returns total/supported/unsupported claim counts, a 0-1 coverage ratio,
    and the unsupported claims -- the input to a CI quality gate for
    research/support/data agents ("don't ship an answer that cites nothing")."""
    rows = provenance(source)
    supported = [r for r in rows if r["evidence"]]
    unsupported = [r["claim"] for r in rows if not r["evidence"]]
    return {
        "claims": len(rows),
        "supported": len(supported),
        "unsupported": unsupported,
        "coverage": round(len(supported) / len(rows), 3) if rows else 1.0,
    }


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

def _divergence_cause(a, b) -> str:
    """WHY two runs first diverged at a step -- the flake's root-cause class."""
    from .providers.base import ModelResponse

    if a is None or b is None:
        return "one run ended earlier"
    if a.kind != b.kind:
        return "control flow diverged (different action taken)"
    if a.kind == "model":
        ra, rb = ModelResponse.from_dict(a.result), ModelResponse.from_dict(b.result)
        ta, tb = [c.name for c in ra.tool_calls], [c.name for c in rb.tool_calls]
        if ta != tb:
            return f"model chose different tools ({ta or 'none'} vs {tb or 'none'})"
        if a.key != b.key:
            return "model saw different context (an earlier step leaked in)"
        return "model answered differently (sampling nondeterminism)"
    # a tool step
    err_a = isinstance(a.result, str) and a.result.startswith("ERROR")
    err_b = isinstance(b.result, str) and b.result.startswith("ERROR")
    if err_a != err_b:
        return "a tool erred in some runs (flaky tool)"
    if a.key != b.key:
        return "tool called with different input"
    return "tool returned different results (flaky tool/retrieval)"


def flakiness(traces: "list[dict]") -> dict:
    """Where do repeated runs of the same task diverge -- and WHY?

    The first trace is the baseline; every other is diffed against it. Each
    divergence carries a root-cause class (model sampling, different tool
    chosen, flaky tool result, tool error, control flow) so the histogram
    clusters by cause, not just position.
    """
    from collections import Counter

    from .diff import diff_logs
    from .effect import EffectEntry

    if len(traces) < 2:
        raise ValueError("flakiness needs at least two traces of the same task")
    logs = [[EffectEntry.from_dict(e) for e in t.get("log", [])] for t in traces]
    base = logs[0]
    hist: Counter = Counter()
    causes: dict = {}  # step -> Counter of causes
    identical = 0
    for other in logs[1:]:
        step = diff_logs(base, other).first_divergence
        if step is None:
            identical += 1
            continue
        hist[step] += 1
        a = base[step] if step < len(base) else None
        b = other[step] if step < len(other) else None
        causes.setdefault(step, Counter())[_divergence_cause(a, b)] += 1
    base_kinds = {e.seq: e.kind for e in base}
    return {
        "runs": len(traces),
        "identical": identical,
        "by_step": sorted(
            (step, n, base_kinds.get(step, "past-end")) for step, n in hist.items()
        ),
        "causes": {step: dict(c) for step, c in causes.items()},
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
            for cause, cn in sorted(f.get("causes", {}).get(step, {}).items(),
                                    key=lambda kv: -kv[1]):
                lines.append(f"        └ {cause} ×{cn}")
        worst = max(f["by_step"], key=lambda row: row[1])
        lines.append(f"flakiest step: {worst[0]} ({worst[2]}) -- "
                     f"{worst[1]}/{total} run(s) diverged there first")
        agg: dict = {}
        for c in f.get("causes", {}).values():
            for cause, n in c.items():
                agg[cause] = agg.get(cause, 0) + n
        if agg:
            cause, n = max(agg.items(), key=lambda kv: kv[1])
            lines.append(f"dominant cause: {cause} ({n}/{total} run(s))")
    return "\n".join(lines)