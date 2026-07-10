"""Conditional breakpoints for a trace -- the debugger feature agents lacked.

Set a condition and find the first action that trips it, like a debugger's
conditional breakpoint. Conditions:

    tool:send_email        the first call to send_email* (glob)
    cap:network            the first action with the network capability
    risk:destructive       the first destructive-risk action
    blocked                the first firewall-blocked call
    after cap:secret       the first egress AFTER a secret was read (sequence)
    text:password          the first result containing "password"

    loom replay run.loom.json --break "cap:network after cap:secret"
    loom debug  run.loom.json --break tool:issue_refund     # jump there in the UI
"""

from __future__ import annotations

from fnmatch import fnmatchcase
from typing import Any

_EGRESS = {"network", "user_communication", "browser_submit", "money_movement"}


def _matches(a: Any, cond: str) -> bool:
    cond = cond.strip()
    if cond.startswith("tool:"):
        return a.type in ("call",) and fnmatchcase(a.tool, cond[5:] if "*" in cond else cond[5:] + "*")
    if cond.startswith("cap:"):
        return cond[4:] in set(a.capabilities)
    if cond.startswith("risk:"):
        return a.risk == cond[5:]
    if cond == "blocked":
        return a.policy is not None and a.policy.blocked
    if cond == "risky":
        return bool(a.risky)
    if cond.startswith("text:"):
        return a.observation is not None and cond[5:].lower() in (a.observation.text or "").lower()
    return False


def find_break(data: Any, condition: str) -> dict:
    """The first action that trips ``condition`` (supports 'X after Y' sequence)."""
    from .action import actions

    acts = actions(data)
    after = None
    cond = condition
    if " after " in condition:
        cond, after = [p.strip() for p in condition.split(" after ", 1)]

    seen_pre = after is None
    for a in acts:
        if after is not None and not seen_pre and _matches(a, after):
            seen_pre = True
            continue
        if seen_pre and _matches(a, cond):
            return {"hit": True, "step": a.step, "tool": a.tool, "type": a.type,
                    "capabilities": a.capabilities, "risk": a.risk,
                    "condition": condition}
    return {"hit": False, "condition": condition}


def find_all_breaks(data: Any, condition: str) -> "list[int]":
    """Every step that trips ``condition`` (no sequence gate) -- for UI highlighting."""
    from .action import actions

    cond = condition.split(" after ", 1)[0].strip()
    return [a.step for a in actions(data) if _matches(a, cond)]


def describe_break(r: dict) -> str:
    if not r["hit"]:
        return f"breakpoint “{r['condition']}” never tripped"
    return (f"⏹ breakpoint “{r['condition']}” hit at step {r['step']}: "
            f"{r['type']} {r['tool']} "
            f"[{', '.join(r['capabilities'])}]{' ⚠' + r['risk'] if r['risk'] else ''}")
