"""``loom memory forensics``: catch memory poisoning across time.

The scariest failure of a long-lived agent isn't a leak in the moment -- it's
that a web page it read on Monday planted an instruction in its memory that
steers it on Thursday. Value-lineage and injection scans look at one run;
memory forensics follows the contamination:

    loom memory forensics session.loom.json

It flags (1) POISONED RECALLS -- recalled memory that carries injected
instructions (contaminated in an earlier run, resurfacing now), (2) the
CONTAMINATION CHAIN when a poisoned recall precedes a risky/egress action, and
(3) FUTURE-POISON -- untrusted content in this run that, if auto-stored, would
poison later runs. Recommendation: quarantine / don't auto-store untrusted
results / expire memory.
"""

from __future__ import annotations

from typing import Any


def memory_forensics(source: Any) -> dict:
    """Poisoned recalls, contamination chains, and future-poison risk in a run."""
    from .action import actions, effect_dicts
    from .inject import _INJECTION, _is_untrusted, _snippet

    data = source if isinstance(source, dict) else source.to_dict()
    acts = actions(data)

    # 1. poisoned recalls: a 'memory' effect whose recalled text carries an
    #    injected instruction (planted in an earlier run, resurfacing now).
    poisoned: list[dict] = []
    for e in effect_dicts(data):
        if e.get("kind") != "memory":
            continue
        text = e["result"] if isinstance(e.get("result"), str) else ""
        m = _INJECTION.search(text)
        if m:
            poisoned.append({"step": e.get("seq"), "marker": _snippet(m.group(0)),
                             "context": _snippet(text[max(0, m.start() - 20):m.end() + 60])})

    # 2. contamination chain: a poisoned recall, then a later risky / egress action
    chains: list[dict] = []
    for p in poisoned:
        after = [a for a in acts if a.type == "call" and a.step > (p["step"] or -1)
                 and (a.risky or set(a.capabilities) & {"network", "user_communication", "money_movement"})]
        if after:
            chains.append({"recall_step": p["step"],
                           "action": {"step": after[0].step, "tool": after[0].tool,
                                      "risk": after[0].risk or "egress"}})

    # 3. future-poison: untrusted results in THIS run that, if stored, poison later
    future: list[dict] = []
    for a in acts:
        if a.type == "call" and a.step >= 0 and _is_untrusted(a):
            text = a.observation.text if a.observation else ""
            m = _INJECTION.search(text)
            if m:
                future.append({"step": a.step, "tool": a.tool, "marker": _snippet(m.group(0))})

    sev = ("critical" if chains else "high" if poisoned else
           "medium" if future else "none")
    return {
        "poisoned_recalls": poisoned, "contamination_chains": chains,
        "future_poison": future, "severity": sev,
        "recommendations": _recommend(poisoned, chains, future),
    }


def _recommend(poisoned, chains, future) -> "list[str]":
    recs = []
    if chains:
        recs.append("QUARANTINE the poisoned memory now -- it already steered a risky action")
    if poisoned:
        recs.append("don't auto-store untrusted (network/fetch/browser) results: "
                    "TraceMemory(auto_store=False) or filter before add()")
        recs.append("add a memory TTL so stale poison expires")
    if future:
        recs.append("scan untrusted results for injection before they enter memory "
                    "(loom inject --gate in CI)")
    return recs or ["no memory-poisoning signals found"]


def describe_memforensics(r: dict) -> str:
    if r["severity"] == "none":
        return "memory forensics: no poisoning signals (no poisoned recalls or chains)"
    lines = [f"memory forensics — severity {r['severity']}"]
    if r["contamination_chains"]:
        lines.append(f"  🔴 {len(r['contamination_chains'])} contamination chain(s) "
                     "(poisoned recall → risky action):")
        for c in r["contamination_chains"]:
            lines.append(f"      recall@{c['recall_step']} → [{c['action']['step']}] "
                         f"{c['action']['tool']} ({c['action']['risk']})")
    if r["poisoned_recalls"]:
        lines.append(f"  🟠 {len(r['poisoned_recalls'])} poisoned recall(s):")
        for p in r["poisoned_recalls"][:3]:
            lines.append(f"      step {p['step']}: “{p['marker']}”")
    if r["future_poison"]:
        lines.append(f"  🟡 {len(r['future_poison'])} untrusted result(s) that would poison memory")
    lines += ["  recommendations:"] + [f"    · {x}" for x in r["recommendations"]]
    return "\n".join(lines)
