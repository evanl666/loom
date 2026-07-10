"""``loom scan``: a security posture report for an agent's tool surface.

Agents accrete tools -- local functions, MCP servers, packs -- and each one is
a capability the agent (or a prompt injection) can reach. ``scan`` reads a run
(or a whole corpus) and reports the surface and its gaps:

    loom scan session.loom.json
    loom scan runs/ --gate         # exit 1 if any high finding

For every tool it exercised it shows the inferred capabilities and whether the
firewall ever gated it, then flags the combinations that matter: a
money-movement / destructive / db-write tool that ran with **no firewall rule**,
a secret read that shared a run with an egress sink (exfiltration surface), and
tools that couldn't be classified at all (unknown = not safe). A letter grade
summarizes it. Point it at an MCP manifest's tools too, via a trace of a run
that used them.
"""

from __future__ import annotations

import json
import os
from glob import glob
from typing import Any

_DANGEROUS_CAPS = {"money_movement", "destructive", "database_write", "browser_submit"}
_EGRESS_CAPS = {"network", "user_communication", "browser_submit", "money_movement"}


def _tool_surface(data: dict) -> dict:
    """{tool: {capabilities, calls, risky, gated, blocked}} across a run."""
    from .action import actions

    surface: dict[str, dict] = {}
    for a in actions(data):
        if a.type not in ("call", "blocked"):
            continue
        t = surface.setdefault(a.tool, {
            "capabilities": set(), "calls": 0, "risky": False,
            "gated": False, "blocked": False, "risk": ""})
        t["capabilities"] |= set(a.capabilities)
        t["calls"] += 1
        t["risky"] = t["risky"] or a.risky
        t["risk"] = t["risk"] or a.risk
        if a.policy is not None:
            t["gated"] = True
        if a.type == "blocked":
            t["blocked"] = True
    return surface


def scan(source: Any) -> dict:
    """Security posture for one trace (dict) or a directory of traces.

    Returns {"tools": [...], "findings": [...], "grade": "A".."F", "runs": N}.
    """
    datas: list[dict] = []

    def _add_path(p: str) -> None:
        if os.path.isdir(p):
            for f in sorted(glob(os.path.join(p, "**", "*.loom.json"), recursive=True)):
                try:
                    with open(f) as fh:
                        datas.append(json.load(fh))
                except (OSError, json.JSONDecodeError):
                    pass
        else:
            try:
                with open(p) as fh:
                    datas.append(json.load(fh))
            except (OSError, json.JSONDecodeError):
                pass

    if isinstance(source, str):
        _add_path(source)
    elif isinstance(source, (list, tuple)):
        for item in source:
            (_add_path(item) if isinstance(item, str) else datas.append(item))
    else:
        datas.append(source)

    from .taint import taint_paths

    merged: dict[str, dict] = {}
    exfil_paths = 0
    for data in datas:
        if not isinstance(data, dict):
            continue
        for name, t in _tool_surface(data).items():
            m = merged.setdefault(name, {
                "capabilities": set(), "calls": 0, "risky": False,
                "gated": False, "blocked": False, "risk": ""})
            m["capabilities"] |= t["capabilities"]
            m["calls"] += t["calls"]
            m["risky"] = m["risky"] or t["risky"]
            m["gated"] = m["gated"] or t["gated"]
            m["blocked"] = m["blocked"] or t["blocked"]
            m["risk"] = m["risk"] or t["risk"]
        exfil_paths += len(taint_paths(data))

    findings: list[dict] = []
    for name, t in sorted(merged.items()):
        caps = t["capabilities"]
        dangerous = caps & _DANGEROUS_CAPS
        # a dangerous tool that ran and was never gated by the firewall
        if dangerous and not t["gated"] and not t["blocked"]:
            findings.append({
                "severity": "high", "tool": name,
                "issue": f"{', '.join(sorted(dangerous))} tool ran with no firewall rule",
                "detail": f"{name} ({', '.join(sorted(caps))}) executed {t['calls']}x, "
                          f"never gated -- add a deny/confirm/approver rule"})
        # unknown-capability tool: couldn't be classified, so not provably safe
        elif not caps:
            findings.append({
                "severity": "medium", "tool": name,
                "issue": "unknown capability (unclassified tool)",
                "detail": f"{name} matched no capability heuristic; declare "
                          f"@tool(capabilities=...) so policy can trust it"})
        # a plain egress or secret tool with no gate -- lower, but worth noting
        elif (caps & _EGRESS_CAPS) and not t["gated"]:
            findings.append({
                "severity": "low", "tool": name,
                "issue": f"ungated egress ({', '.join(sorted(caps & _EGRESS_CAPS))})",
                "detail": f"{name} reaches off the box and is not firewall-gated"})

    if exfil_paths:
        findings.insert(0, {
            "severity": "high", "tool": "",
            "issue": f"{exfil_paths} exfiltration path(s) by value lineage",
            "detail": "a sensitive value read in a run reappeared in an egress "
                      "action -- run `loom taint` for the lineage"})

    highs = sum(1 for f in findings if f["severity"] == "high")
    meds = sum(1 for f in findings if f["severity"] == "medium")
    grade = "A" if not findings else ("F" if highs >= 3 else "D" if highs >= 1
                                      else "C" if meds >= 2 else "B")
    return {
        "runs": len(datas),
        "tools": [{"name": n, "capabilities": sorted(t["capabilities"]),
                   "calls": t["calls"], "risky": t["risky"], "gated": t["gated"],
                   "blocked": t["blocked"], "risk": t["risk"]}
                  for n, t in sorted(merged.items())],
        "findings": findings,
        "grade": grade,
        "high": highs,
    }


def describe_scan(report: dict) -> str:
    lines = [f"agent supply-chain scan — grade {report['grade']} "
             f"({report['runs']} run(s), {len(report['tools'])} tool(s))", ""]
    lines.append(f"  {'tool':<24} {'capabilities':<40} {'calls':>5} gated")
    lines.append("  " + "-" * 78)
    for t in report["tools"]:
        flag = "⚠" if t["risky"] else " "
        gate = "✓" if t["gated"] else ("blocked" if t["blocked"] else "—")
        caps = ", ".join(t["capabilities"])[:38]
        lines.append(f"  {flag}{t['name']:<23} {caps:<40} {t['calls']:>5} {gate}")
    if report["findings"]:
        lines += ["", f"  {len(report['findings'])} finding(s):"]
        icon = {"high": "🔴", "medium": "🟠", "low": "🟡"}
        for f in report["findings"]:
            where = f" [{f['tool']}]" if f["tool"] else ""
            lines.append(f"    {icon.get(f['severity'], '·')} {f['issue']}{where}")
            lines.append(f"        {f['detail']}")
    else:
        lines += ["", "  ✓ no gaps found"]
    return "\n".join(lines)
