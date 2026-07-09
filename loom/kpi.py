"""``loom kpi``: platform-team trends across a corpus of agent runs.

A single trace is a debugging artifact; a *corpus* is an operational signal.
This aggregates a directory of runs into the numbers a platform team watches:
how often the firewall fires, how much PII / money the fleet touches, cost
tails, failure rate. Offline, built on the Action schema so business risk is
first-class.

    loom kpi runs/
    loom kpi runs/ --html kpis.html
"""

from __future__ import annotations

import json
import os

# Capabilities worth trending for a platform/security owner.
_WATCH = ["pii_access", "money_movement", "database_write", "user_communication",
          "browser_submit", "exec", "network", "secret"]


def _percentile(values: "list[int]", p: float) -> int:
    if not values:
        return 0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100) * (len(s) - 1)))))
    return s[k]


def compute_kpis(paths: "list[str]") -> dict:
    """Aggregate KPIs over every trace in ``paths`` (files and/or directories)."""
    from .action import actions as _actions, effect_dicts as _effect_dicts
    from .packs import install_builtin

    install_builtin()

    runs = 0
    failed = 0
    total_calls = 0
    risky_calls = 0
    blocked = 0
    tokens_per_run: list[int] = []
    cap_actions = {c: 0 for c in _WATCH}    # action count per capability
    cap_runs = {c: 0 for c in _WATCH}       # run count touching each capability
    tool_risk = {}                          # tool -> risky-call count

    for p in paths:
        try:
            with open(p) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        runs += 1
        if (data.get("stop_reason") not in ("", "end_turn")
                or data.get("truncated") or data.get("paused")):
            failed += 1
        tokens = 0
        for e in _effect_dicts(data):
            if e.get("kind") == "model" and isinstance(e.get("result"), dict):
                u = e["result"].get("usage") or {}
                tokens += (u.get("input_tokens", 0) or 0) + (u.get("output_tokens", 0) or 0)
        tokens_per_run.append(tokens)
        blocked += sum(1 for ev in (data.get("shield_events") or [])
                       if ev.get("action") == "deny")

        run_caps = set()
        for a in _actions(data):
            if a.type != "call" or a.step < 0:
                continue
            total_calls += 1
            if a.risky:
                risky_calls += 1
                tool_risk[a.tool] = tool_risk.get(a.tool, 0) + 1
            for c in a.capabilities:
                if c in cap_actions:
                    cap_actions[c] += 1
                    run_caps.add(c)
        for c in run_caps:
            cap_runs[c] += 1

    return {
        "runs": runs,
        "failed": failed,
        "failure_rate": round(100 * failed / runs) if runs else 0,
        "total_calls": total_calls,
        "risky_calls": risky_calls,
        "blocked_actions": blocked,
        "tokens": {"total": sum(tokens_per_run),
                   "mean": round(sum(tokens_per_run) / runs) if runs else 0,
                   "p50": _percentile(tokens_per_run, 50),
                   "p95": _percentile(tokens_per_run, 95),
                   "max": max(tokens_per_run) if tokens_per_run else 0},
        "capabilities": [
            {"capability": c, "actions": cap_actions[c], "runs": cap_runs[c]}
            for c in _WATCH if cap_actions[c]
        ],
        "top_risky_tools": sorted(
            ({"tool": t, "count": n} for t, n in tool_risk.items()),
            key=lambda r: -r["count"])[:10],
    }


def kpi_text(k: dict) -> str:
    lines = [f"KPIs over {k['runs']} run(s), {k['total_calls']} tool call(s):", ""]
    lines.append(f"  failure rate       {k['failure_rate']}%  ({k['failed']}/{k['runs']})")
    lines.append(f"  risky actions      {k['risky_calls']}")
    lines.append(f"  firewall denies    {k['blocked_actions']}")
    t = k["tokens"]
    lines.append(f"  tokens/run         mean {t['mean']:,}  p50 {t['p50']:,}  "
                 f"p95 {t['p95']:,}  max {t['max']:,}")
    if k["capabilities"]:
        lines.append("\n  capability exposure (actions / runs touched):")
        for c in k["capabilities"]:
            lines.append(f"    {c['capability']:<20} {c['actions']:>5} / {c['runs']}")
    if k["top_risky_tools"]:
        lines.append("\n  top risky tools:")
        for r in k["top_risky_tools"]:
            lines.append(f"    {r['tool']:<24} {r['count']}")
    return "\n".join(lines)


def kpi_html(k: dict) -> str:
    from .lake import _esc

    tiles = [
        ("runs", k["runs"], ""),
        ("failure rate", f"{k['failure_rate']}%", "warn" if k["failure_rate"] else ""),
        ("risky actions", k["risky_calls"], ""),
        ("firewall denies", k["blocked_actions"], "warn" if k["blocked_actions"] else ""),
        ("tokens p95", f"{k['tokens']['p95']:,}", ""),
    ]
    tile_html = "".join(
        f'<div class="tile"><div class="v {cls}">{_esc(v)}</div>'
        f'<div class="k">{_esc(label)}</div></div>' for label, v, cls in tiles)
    cap_rows = "".join(
        f'<tr><td><code>{_esc(c["capability"])}</code></td>'
        f'<td class="num">{c["actions"]}</td><td class="num">{c["runs"]}</td></tr>'
        for c in k["capabilities"])
    tool_rows = "".join(
        f'<tr><td><code>{_esc(r["tool"])}</code></td><td class="num">{r["count"]}</td></tr>'
        for r in k["top_risky_tools"])
    css = """body{font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
    max-width:820px;margin:0 auto;padding:32px;color:#0b0b0b;background:#f9f9f7}
    @media(prefers-color-scheme:dark){body{color:#fff;background:#0d0d0d}
    table,.tile{background:#1a1a19!important;border-color:#2c2c2a!important}}
    h1{font-size:20px}.sub{color:#898781;margin-bottom:20px}
    .tiles{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:24px}
    .tile{background:#fff;border:1px solid #e1e0d9;border-radius:10px;padding:14px 18px;min-width:130px}
    .tile .v{font-size:26px;font-weight:650}.tile .v.warn{color:#b3261e}
    .tile .k{color:#898781;font-size:11px;text-transform:uppercase;letter-spacing:.06em}
    h2{font-size:13px;color:#52514e;margin:22px 0 6px}
    table{width:100%;border-collapse:collapse;background:#fff;border:1px solid #e1e0d9;border-radius:10px}
    th{text-align:left;font-size:11px;color:#898781;text-transform:uppercase;padding:8px 12px;border-bottom:1px solid #e1e0d9}
    td{padding:7px 12px;border-bottom:1px solid #e1e0d9}td.num{text-align:right;font-variant-numeric:tabular-nums}"""
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Loom KPIs</title><style>{css}</style></head><body>
<h1>📊 Loom agent KPIs</h1>
<p class="sub">{k['runs']} run(s), {k['total_calls']} tool call(s)</p>
<div class="tiles">{tile_html}</div>
<h2>Capability exposure</h2>
<table><tr><th>capability</th><th>actions</th><th>runs touched</th></tr>
{cap_rows or '<tr><td colspan=3>none</td></tr>'}</table>
<h2>Top risky tools</h2>
<table><tr><th>tool</th><th>risky calls</th></tr>
{tool_rows or '<tr><td colspan=2>none</td></tr>'}</table>
</body></html>"""


def tool_trust(paths: "list[str]") -> "list[dict]":
    """Per-tool reputation across a corpus -- the tool/MCP-server risk profile.

    For each tool: how often it errored, was blocked by the firewall, needed
    approval, exercised risk, and whether it supports undo. A trust score
    (0-100) folds these into one number so a pack registry / MCP gateway can
    rank tools and servers.
    """
    import json

    from .action import actions as _actions
    from .packs import install_builtin, undo_plan

    install_builtin()
    stats: dict[str, dict] = {}
    for p in paths:
        try:
            with open(p) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        for a in _actions(data):
            if a.type != "call" or a.step < 0:
                continue
            s = stats.setdefault(a.tool, {"calls": 0, "errors": 0, "blocked": 0,
                                          "approved": 0, "risky": 0, "undoable": 0})
            s["calls"] += 1
            if a.observation is not None and a.observation.error:
                s["errors"] += 1
            if a.policy is not None:
                if a.policy.blocked:
                    s["blocked"] += 1
                elif a.policy.action in ("approve",):
                    s["approved"] += 1
            if a.risky:
                s["risky"] += 1
            if undo_plan(a, data) is not None:
                s["undoable"] += 1

    rows = []
    for name, s in stats.items():
        n = s["calls"] or 1
        # Penalize error/blocked/risky rates; reward undo support. Bounded 0-100.
        err = 40 * s["errors"] / n
        blk = 30 * s["blocked"] / n
        risk = 25 * s["risky"] / n
        undo_bonus = 10 * s["undoable"] / n if s["risky"] else 0
        trust = max(0, min(100, round(100 - err - blk - risk + undo_bonus)))
        rows.append({"tool": name, "calls": s["calls"],
                     "error_rate": round(100 * s["errors"] / n),
                     "blocked_rate": round(100 * s["blocked"] / n),
                     "risky_rate": round(100 * s["risky"] / n),
                     "undo_support": round(100 * s["undoable"] / n),
                     "trust": trust})
    rows.sort(key=lambda r: r["trust"])   # least-trusted first (what to review)
    return rows


def tool_trust_text(rows: "list[dict]") -> str:
    if not rows:
        return "no tools found"
    lines = [f"{'tool':<26} {'trust':>6} {'calls':>6} {'err%':>5} "
             f"{'blk%':>5} {'risk%':>6} {'undo%':>6}", "-" * 64]
    for r in rows:
        lines.append(f"{r['tool']:<26} {r['trust']:>6} {r['calls']:>6} "
                     f"{r['error_rate']:>4}% {r['blocked_rate']:>4}% "
                     f"{r['risky_rate']:>5}% {r['undo_support']:>5}%")
    return "\n".join(lines)
