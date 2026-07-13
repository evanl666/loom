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
    """The earliest step that went wrong, and its downstream cascade.

    Signals are TIERED. A genuine FAILURE -- a firewall block, a tool error, a loop
    -- is what "went wrong". A RISK signal -- an ungated dangerous call, PII/secret
    access, a taint path to an egress -- is security-notable but NOT, on its own, a
    failure: a legitimate refund flow reads a customer record and emails that same
    customer. So the earliest FAILURE wins; only when there is no failure do we
    surface the earliest RISK, clearly marked ``kind: "risk"`` ("the run completed;
    review this") instead of alarming as a "first bad step".
    """
    from .action import actions
    from .loops import detect_loops
    from .taint import taint_paths

    acts = [a for a in actions(data) if a.step >= 0]
    fails: dict[int, list] = {}
    risks: dict[int, list] = {}

    def _fail(step: int, why: str) -> None:
        fails.setdefault(step, []).append(why)

    def _risk(step: int, why: str) -> None:
        risks.setdefault(step, []).append(why)

    for a in acts:
        if a.type != "call":
            continue
        caps = set(a.capabilities)
        if a.policy is not None and a.policy.blocked:
            _fail(a.step, "firewall blocked this call")
        if a.observation is not None and a.observation.error:
            _fail(a.step, "tool errored / returned an error")
        if (caps & _DANGEROUS) and (a.policy is None):
            _risk(a.step, f"ungated {', '.join(sorted(caps & _DANGEROUS))} call")
        if a.risky:
            _risk(a.step, f"risky action ({a.risk})")

    # taint: data read here later flows OUT -- an exfiltration signal, but the sink
    # may be a legitimate recipient, so it's a RISK to review, not a failure.
    for p in taint_paths(data):
        _risk(p["source"]["step"],
              f"data read here later flows to {p['sink']['tool']} (verify the recipient)")

    loops = detect_loops(data)
    for f in loops["findings"]:
        _fail(f.get("started", 0), "start of a loop / oscillation")

    if fails:
        signals, kind = fails, "failure"
        note = "everything after this step is downstream of it -- fork here to test the fix"
    elif risks:
        signals, kind = risks, "risk"
        note = ("the run completed WITHOUT a failure -- this is just its most "
                "security-notable step; review it, it isn't necessarily a bug")
    else:
        return {"found": False}
    first = min(signals)
    target = next((a for a in acts if a.step == first), None)
    cascade = [{"step": a.step, "tool": a.tool, "type": a.type}
               for a in acts if a.step > first and a.type == "call"][:6]
    return {
        "found": True, "kind": kind, "step": first,
        "tool": target.tool if target else "",
        "signals": signals[first],
        "cascade": cascade,
        "note": note,
    }


def describe_rootcause(r: dict) -> str:
    if not r["found"]:
        return "no root-cause signal found -- the run looks clean"
    head = (f"🎯 first bad step: {r['step']} ({r['tool']})" if r.get("kind") != "risk"
            else f"✅ no failure -- most security-notable step: {r['step']} ({r['tool']})")
    lines = [head, "  why: " + "; ".join(r["signals"])]
    if r["cascade"]:
        chain = " → ".join(f"[{c['step']}]{c['tool']}" for c in r["cascade"])
        lines.append(f"  cascade: {chain}")
    lines.append(f"  {r['note']}")
    return "\n".join(lines)
