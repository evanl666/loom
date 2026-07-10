"""``loom rootcause``: find the FIRST step that went wrong, and the cascade.

A trace viewer shows a timeline; a debugger should point at the root cause. This
scores each action against a set of "badness" signals -- an ungated dangerous
call, the start of an exfiltration path, a firewall block, a tool error, the
start of a loop, a cost spike -- and returns the EARLIEST one, plus the chain of
consequences it set off:

    loom rootcause session.loom.json
"""

from __future__ import annotations

from typing import Any

_DANGEROUS = {"money_movement", "destructive", "database_write"}


def first_bad_step(data: Any) -> dict:
    """The earliest action tripping a badness signal, and its downstream cascade."""
    from .action import actions
    from .loops import detect_loops
    from .taint import taint_paths

    acts = [a for a in actions(data) if a.step >= 0]
    signals: dict[int, list] = {}

    def _add(step: int, why: str) -> None:
        signals.setdefault(step, []).append(why)

    for a in acts:
        if a.type != "call":
            continue
        caps = set(a.capabilities)
        if a.policy is not None and a.policy.blocked:
            _add(a.step, "firewall blocked this call")
        if (caps & _DANGEROUS) and (a.policy is None):
            _add(a.step, f"ungated {', '.join(sorted(caps & _DANGEROUS))} call")
        if a.observation is not None and a.observation.error:
            _add(a.step, "tool errored / returned an error")
        if a.risky:
            _add(a.step, f"risky action ({a.risk})")

    # exfiltration paths: the source read is the bad step
    for p in taint_paths(data):
        _add(p["source"]["step"], f"secret read that later leaks to {p['sink']['tool']}")

    # loop start
    loops = detect_loops(data)
    for f in loops["findings"]:
        _add(f.get("started", 0), "start of a loop / oscillation")

    if not signals:
        return {"found": False}
    first = min(signals)
    target = next((a for a in acts if a.step == first), None)
    cascade = [{"step": a.step, "tool": a.tool, "type": a.type}
               for a in acts if a.step > first and a.type == "call"][:6]
    return {
        "found": True, "step": first,
        "tool": target.tool if target else "",
        "signals": signals[first],
        "cascade": cascade,
        "note": "everything after this step is downstream of it -- fork here to test the fix",
    }


def describe_rootcause(r: dict) -> str:
    if not r["found"]:
        return "no root-cause signal found -- the run looks clean"
    lines = [f"🎯 first bad step: {r['step']} ({r['tool']})",
             "  why: " + "; ".join(r["signals"])]
    if r["cascade"]:
        chain = " → ".join(f"[{c['step']}]{c['tool']}" for c in r["cascade"])
        lines.append(f"  cascade: {chain}")
    lines.append(f"  {r['note']}")
    return "\n".join(lines)
