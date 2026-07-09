"""``loom leaderboard``: compare agents by safety, not just success.

Point it at a directory laid out one sub-folder per agent::

    runs/
      claude-code/  *.loom.json
      codex/        *.loom.json
      aider/        *.loom.json

and it ranks them on the numbers that matter for putting an agent in
production -- behavior safety score, cost, how often they take risky actions,
how often the firewall had to step in, and failure rate:

    loom leaderboard runs/
    loom leaderboard runs/ --html board.html

Offline; the safety benchmark tool the community can standardize on. (The
HOSTED public leaderboard is a separate, commercial thing.)
"""

from __future__ import annotations

import json
import os
from glob import glob


def _agent_dirs(directory: str) -> "list[tuple[str, list[str]]]":
    """(agent name, its trace paths) for each sub-folder holding traces."""
    out = []
    for name in sorted(os.listdir(directory)):
        sub = os.path.join(directory, name)
        if not os.path.isdir(sub):
            continue
        traces = sorted(glob(os.path.join(sub, "**", "*.loom.json"), recursive=True))
        if traces:
            out.append((name, traces))
    return out


def compute_leaderboard(directory: str) -> "list[dict]":
    """Aggregate per-agent safety/cost/risk metrics; ranked safest first."""
    from .action import actions as _actions, effect_dicts as _effect_dicts
    from .diff import score_breakdown
    from .packs import install_builtin

    install_builtin()
    rows = []
    for name, traces in _agent_dirs(directory):
        scores, tokens, risky_runs, blocked, failed = [], [], 0, 0, 0
        for p in traces:
            try:
                with open(p) as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            scores.append(score_breakdown(data)["overall"])
            t = 0
            for e in _effect_dicts(data):
                if e.get("kind") == "model" and isinstance(e.get("result"), dict):
                    u = e["result"].get("usage") or {}
                    t += (u.get("input_tokens", 0) or 0) + (u.get("output_tokens", 0) or 0)
            tokens.append(t)
            if any(a.risky for a in _actions(data) if a.type == "call"):
                risky_runs += 1
            blocked += sum(1 for ev in (data.get("shield_events") or [])
                           if ev.get("action") == "deny")
            if (data.get("stop_reason") not in ("", "end_turn")
                    or data.get("truncated") or data.get("paused")):
                failed += 1
        n = len(scores) or 1
        rows.append({
            "agent": name, "runs": len(scores),
            "safety": round(sum(scores) / n),
            "cost": round(sum(tokens) / n),
            "risky_rate": round(100 * risky_runs / n),
            "blocked": blocked,
            "failure_rate": round(100 * failed / n),
        })
    rows.sort(key=lambda r: (-r["safety"], r["cost"]))
    return rows


def leaderboard_text(rows: "list[dict]") -> str:
    if not rows:
        return "no agents found (expected <dir>/<agent-name>/*.loom.json)"
    lines = [f"{'agent':<18} {'safety':>6} {'cost':>8} {'risky%':>7} "
             f"{'blocked':>7} {'fail%':>6} {'runs':>5}",
             "-" * 62]
    for r in rows:
        lines.append(f"{r['agent']:<18} {r['safety']:>6} {r['cost']:>8,} "
                     f"{r['risky_rate']:>6}% {r['blocked']:>7} {r['failure_rate']:>5}% "
                     f"{r['runs']:>5}")
    lines.append(f"\n🏆 safest: {rows[0]['agent']} ({rows[0]['safety']}/100)")
    return "\n".join(lines)


def leaderboard_html(rows: "list[dict]") -> str:
    from .diff import score_badge_svg

    body = "".join(
        f"<tr><td>{i + 1}</td><td><b>{r['agent']}</b></td>"
        f"<td>{score_badge_svg(r['safety'])}</td>"
        f"<td class='num'>{r['cost']:,}</td><td class='num'>{r['risky_rate']}%</td>"
        f"<td class='num'>{r['blocked']}</td><td class='num'>{r['failure_rate']}%</td>"
        f"<td class='num'>{r['runs']}</td></tr>"
        for i, r in enumerate(rows))
    css = """body{font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
    max-width:820px;margin:0 auto;padding:32px;color:#0b0b0b;background:#f9f9f7}
    @media(prefers-color-scheme:dark){body{color:#fff;background:#0d0d0d}
    table{background:#1a1a19!important;border-color:#2c2c2a!important}}
    h1{font-size:22px}.sub{color:#898781;margin-bottom:18px}
    table{width:100%;border-collapse:collapse;background:#fff;border:1px solid #e1e0d9;border-radius:10px}
    th{text-align:left;font-size:11px;color:#898781;text-transform:uppercase;padding:9px 12px;border-bottom:1px solid #e1e0d9}
    td{padding:8px 12px;border-bottom:1px solid #eee;vertical-align:middle}
    td.num{text-align:right;font-variant-numeric:tabular-nums}"""
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Loom agent safety leaderboard</title><style>{css}</style></head><body>
<h1>🏆 Agent safety leaderboard</h1>
<p class="sub">ranked by behavior safety score, then cost</p>
<table><tr><th>#</th><th>agent</th><th>safety</th><th>cost</th><th>risky%</th>
<th>blocked</th><th>fail%</th><th>runs</th></tr>{body}</table>
</body></html>"""
