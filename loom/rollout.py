"""``loom policy rollout/promote/rollback``: a safe policy lifecycle.

Enterprises don't flip a firewall to full-deny on day one. A policy moves
through stages -- draft → canary → enforce -- and each promotion is gated on
evidence from a real corpus of runs:

    loom policy rollout policy.yml --traces runs/    # assess: blast radius + gate
    loom policy promote policy.yml --traces runs/ --by alice   # advance a stage
    loom policy rollback policy.yml                  # step back

The gate that matters: **would this policy deny runs that previously completed
successfully?** Those are the false positives that break working agents. A
policy with zero such breakages is safe to enforce; one with breakages can
canary (observe-only) but not enforce without --force. Stage + owner + history
live in a ``<policy>.rollout.json`` sidecar, so the policy file stays clean and
the lifecycle is auditable.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

STAGES = ["draft", "canary", "enforce"]
_STAGE_DESC = {
    "draft": "not applied -- being written and simulated",
    "canary": "observe-only -- decisions logged, calls NOT blocked",
    "enforce": "live -- the firewall blocks/gates matching calls",
}


def _sidecar(policy_path: str) -> str:
    return policy_path + ".rollout.json"


def _expand(paths: "list[str]") -> "list[str]":
    """Expand directories to their *.loom.json files (files pass through)."""
    from glob import glob

    out: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            out.extend(sorted(glob(os.path.join(p, "**", "*.loom.json"), recursive=True)))
        else:
            out.append(p)
    return out


def read_stage(policy_path: str) -> dict:
    try:
        with open(_sidecar(policy_path)) as f:
            d = json.load(f)
        if isinstance(d, dict) and d.get("stage") in STAGES:
            return d
    except (OSError, json.JSONDecodeError):
        pass
    return {"stage": "draft", "owner": "", "updated": "", "history": []}


def _write_stage(policy_path: str, state: dict) -> None:
    with open(_sidecar(policy_path), "w") as f:
        json.dump(state, f, indent=2)


def assess(policy_path: str, paths: "list[str]") -> dict:
    """Lifecycle status for a policy against a corpus: stage, blast radius, gate."""
    from .policy_file import load_document, simulate, to_shield_kwargs
    from .shield import Shield

    doc = load_document(policy_path)
    shield = Shield(**to_shield_kwargs(doc))
    sim = simulate(shield, _expand(paths))
    state = read_stage(policy_path)
    stage = state["stage"]
    idx = STAGES.index(stage)
    next_stage = STAGES[idx + 1] if idx + 1 < len(STAGES) else None

    breakages = sim["false_positives"]  # completed runs this policy would deny
    safe_to_enforce = len(breakages) == 0
    # advancing to enforce needs zero breakages; canary is always safe (no block)
    gate_ok = True if (next_stage == "canary") else safe_to_enforce
    if next_stage == "enforce" and not safe_to_enforce:
        rec = (f"HOLD at canary: enforcing would break {len(breakages)} run(s) that "
               "completed successfully -- review them (or narrow the rule) first")
    elif next_stage:
        rec = f"safe to promote → {next_stage}"
    else:
        rec = "already enforcing"
    return {
        "policy": os.path.basename(policy_path),
        "stage": stage, "stage_desc": _STAGE_DESC[stage],
        "next_stage": next_stage, "owner": state.get("owner", ""),
        "runs": sim["runs"], "calls": sim["calls"],
        "would_deny_runs": len(sim["denied"]), "would_confirm_runs": len(sim["confirm_only"]),
        "untouched": sim["untouched"],
        "breakages": [b["name"] for b in breakages],
        "rule_hits": sim["rule_hits"][:8],
        "capabilities": sim["capabilities"],
        "gate_ok": gate_ok, "safe_to_enforce": safe_to_enforce,
        "recommendation": rec,
        "history": state.get("history", []),
    }


def promote(policy_path: str, paths: "list[str]", by: str = "", force: bool = False) -> dict:
    """Advance the policy one stage (draft→canary→enforce), gated by the corpus."""
    status = assess(policy_path, paths)
    nxt = status["next_stage"]
    if nxt is None:
        raise ValueError("policy is already at 'enforce'")
    if not status["gate_ok"] and not force:
        raise ValueError(
            f"refusing to promote to '{nxt}': {len(status['breakages'])} completed "
            f"run(s) would be denied ({', '.join(status['breakages'][:5])}). "
            "Review them, narrow the rule, or pass --force.")
    state = read_stage(policy_path)
    entry = {"stage": nxt, "by": by, "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "forced": bool(force and not status["gate_ok"])}
    state.update({"stage": nxt, "owner": by or state.get("owner", ""),
                  "updated": entry["at"]})
    state.setdefault("history", []).append(entry)
    _write_stage(policy_path, state)
    return {**status, "stage": nxt, "promoted": True, "forced": entry["forced"]}


def rollback(policy_path: str, by: str = "") -> dict:
    """Step the policy back one stage (enforce→canary→draft)."""
    state = read_stage(policy_path)
    idx = STAGES.index(state["stage"])
    if idx == 0:
        raise ValueError("policy is already at 'draft'")
    prev = STAGES[idx - 1]
    at = time.strftime("%Y-%m-%dT%H:%M:%S")
    state.setdefault("history", []).append({"stage": prev, "by": by, "at": at, "rollback": True})
    state.update({"stage": prev, "updated": at})
    _write_stage(policy_path, state)
    return {"policy": os.path.basename(policy_path), "stage": prev, "rolled_back": True}


def describe(status: dict) -> str:
    lines = [
        f"policy: {status['policy']}   stage: {status['stage'].upper()} "
        f"({status['stage_desc']})" + (f"   owner: {status['owner']}" if status.get("owner") else ""),
        f"  corpus: {status['runs']} run(s), {status['calls']} tool call(s)",
        f"  blast radius: would DENY {status['would_deny_runs']} run(s), "
        f"CONFIRM {status['would_confirm_runs']}, leave {status['untouched']} untouched",
    ]
    if status["breakages"]:
        lines.append(f"  ⚠ {len(status['breakages'])} completed run(s) would BREAK: "
                     f"{', '.join(status['breakages'][:6])}")
    if status["rule_hits"]:
        lines.append("  top rules:")
        for h in status["rule_hits"][:5]:
            lines.append(f"    {h['action']:<8} {h['rule']:<28} ×{h['count']}  e.g. {h['example']}")
    if status.get("next_stage"):
        icon = "✓" if status["gate_ok"] else "⛔"
        lines.append(f"  {icon} {status['recommendation']}")
    else:
        lines.append(f"  · {status['recommendation']}")
    return "\n".join(lines)
