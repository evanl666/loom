"""``loom incident``: an agent postmortem, written from the flight recording.

Every section is computed FROM THE TRACE, offline, no model required: what
happened, the suspect timeline (errors, blocks, the final words), blast
radius (cost, tools touched), firewall decisions, context health, whether the
run ever SAW credentials -- and concrete prevention flags plus the exact
commands that turn this incident into a regression test.

    loom incident failed.loom.json
    loom incident failed.loom.json -o postmortem.md
    loom incident failed.loom.json --why --model claude-opus-4-8   # + AI root cause

``--why`` adds a narrative root-cause section from the ``loom why`` debugger
agent (costs API calls; its diagnosis run is itself recorded).
"""

from __future__ import annotations

import json

_SNIP = 160


def classify_all_for_event(ev: dict) -> bool:
    """Did a blocked call carry network/exfiltration risk?"""
    from .risk import classify_all

    return "network-egress" in classify_all(ev.get("tool", ""), ev.get("input", {}))


def _clip(s: str, n: int = _SNIP) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 3] + "..."


def _analyze(data: dict) -> dict:
    """Everything the report needs, mined from the trace dict."""
    log = data.get("log") or []
    facts: dict = {
        "episodes": data.get("episodes") or [data.get("prompt", "")],
        "output": str(data.get("output", "")),
        "stop_reason": data.get("stop_reason", ""),
        "truncated": data.get("truncated", False),
        "paused": data.get("paused", False),
        "model_calls": 0, "input_tokens": 0, "output_tokens": 0,
        "tool_counts": {}, "suspects": [], "final_words": "",
        "shield_events": data.get("shield_events") or [],
    }

    from .risk import classify_all

    facts["risk"] = {}  # category -> example signature
    for e in log:
        kind, result = e.get("kind", ""), e.get("result")
        if kind == "model" and isinstance(result, dict):
            facts["model_calls"] += 1
            usage = result.get("usage") or {}
            facts["input_tokens"] += usage.get("input_tokens", 0) or 0
            facts["output_tokens"] += usage.get("output_tokens", 0) or 0
            for tc in result.get("tool_calls") or []:
                name = tc.get("name", "?")
                facts["tool_counts"][name] = facts["tool_counts"].get(name, 0) + 1
                cats = classify_all(name, tc.get("input", {}))
                if cats:
                    sig = f"{name}({json.dumps(tc.get('input', {}), sort_keys=True, default=str)})"
                    for cat in cats:
                        facts["risk"].setdefault(cat, _clip(sig, 90))
            if result.get("text"):
                facts["final_words"] = (e.get("seq"), _clip(result["text"]))
        elif kind.startswith("tool:") and isinstance(result, str):
            if result.startswith("ERROR:") or result.startswith("BLOCKED:"):
                facts["suspects"].append(
                    (e.get("seq"), f"{kind} -> {_clip(result)}")
                )

    # Context health: the doctor's findings, inline.
    try:
        from .effect import EffectEntry
        from .health import analyze

        report = analyze(facts["episodes"], [EffectEntry.from_dict(e) for e in log])
        facts["health"] = [] if report.ok else [
            f"[{f.severity}] {f.kind}: {_clip(f.message)}" for f in report.findings
        ]
    except Exception:
        facts["health"] = []

    # Did the run SEE credentials? Scan tool results with the scrub patterns.
    from .scrub import scrub_text

    secrets: dict = {}
    for e in log:
        if e.get("kind", "").startswith("tool:") and isinstance(e.get("result"), str):
            _, found = scrub_text(e["result"])
            for k, v in found.items():
                secrets[k] = secrets.get(k, 0) + v
    facts["secrets"] = secrets

    facts["denied"] = [ev for ev in facts["shield_events"] if ev.get("action") == "deny"]
    failed = (
        facts["stop_reason"] not in ("", "end_turn")
        or facts["truncated"] or facts["paused"]
        or "ERROR" in facts["output"] or "FAILED" in facts["output"]
        or bool(facts["suspects"])
    )
    facts["failed"] = failed
    facts["classification"] = _classify_incident(facts)
    facts["severity"] = _severity(facts)
    return facts


# Roughly "this many tokens in one run is worth a mention".
_COST_ALERT_TOKENS = 200_000


def _classify_incident(facts: dict) -> "list[str]":
    """Human labels for what kind of incident this is."""
    from .risk import ALARMING

    tags = []
    risk = facts["risk"]
    # Any channel that carries data off the box: network, an outbound message
    # to a user, or a browser form submit -- a secret leaving by email is
    # exfiltration just as much as one leaving by curl.
    outbound = {"network-egress", "user-comm", "browser-submit"} & set(risk)
    if facts["secrets"] or "secret-read" in risk:
        tags.append("secret exposure")
    if ("secret-read" in risk or facts["secrets"]) and outbound:
        tags.append("possible exfiltration")
    if "pii-access" in risk and outbound:
        tags.append("PII exfiltration")
    if "money-movement" in risk:
        tags.append("money movement")
    if "db-write" in risk:
        tags.append("database mutation")
    if "fs-destructive" in risk:
        tags.append("destructive filesystem action")
    if any("curl" in s or "| sh" in s or "| bash" in s for _, s in facts["suspects"]):
        tags.append("unsafe shell")
    if facts["input_tokens"] + facts["output_tokens"] >= _COST_ALERT_TOKENS:
        tags.append("cost blowup")
    facts["_alarming"] = ALARMING & set(risk)
    return tags


def _severity(facts: dict) -> str:
    """critical / high / medium / low, from what actually happened."""
    cls = facts["classification"]
    if ("possible exfiltration" in cls or "destructive filesystem action" in cls
            or "PII exfiltration" in cls):
        return "critical"
    if ("secret exposure" in cls or "unsafe shell" in cls or "money movement" in cls
            or facts["denied"]):
        return "high"
    # medium needs a genuinely dangerous capability or an actual failure --
    # a clean run that merely executed pytest (code-exec) stays low.
    if facts["failed"] or facts.get("_alarming") or "cost blowup" in cls:
        return "medium"
    return "low"


def _prevention(facts: dict) -> list[str]:
    tips = []
    if facts["secrets"]:
        kinds = ", ".join(sorted(facts["secrets"]))
        tips.append(
            f"this run's tool results contained credentials ({kinds}): record with "
            f"`--scrub` so they never reach disk, and consider "
            f"`--rule 'taint sk-*: confirm *'` to gate everything after an exposure"
        )
    if any("Timeout" in s for _, s in facts["suspects"]):
        tips.append("a tool timed out: cap them explicitly with `Agent(tool_timeout=30)`")
    for f in facts["health"]:
        if "oversized" in f:
            tips.append(
                "an oversized tool result dominated the context: `loom heal` can verify "
                "a redaction fix; `Agent(compact_after=...)` prevents the buildup"
            )
            break
    denied = [ev for ev in facts["shield_events"] if ev.get("action") == "deny"]
    if denied:
        tips.append(
            f"the firewall blocked {len(denied)} call(s) -- the rules held; keep them"
        )
    elif facts["failed"]:
        tips.append(
            "no firewall was active: `loom record --deny/--confirm/--rule` would put "
            "one between the model and the next incident"
        )
    return tips


def _executive_summary(facts: dict, data: dict) -> str:
    """One paragraph a human can paste into a Slack thread or GitHub issue."""
    outcome = "failed" if facts["failed"] else "completed"
    parts = [f"The agent run **{outcome}**"]
    if facts["suspects"]:
        parts[0] += f" after {len(facts['suspects'])} failing tool call(s)"
    blocked = [ev for ev in facts["shield_events"] if ev.get("action") == "deny"]
    if blocked:
        parts.append(f"the firewall blocked {len(blocked)} call(s)")
    if facts["classification"]:
        parts.append("flagged as " + ", ".join(facts["classification"]))
    if facts["secrets"]:
        parts.append(f"credentials were visible in {sum(facts['secrets'].values())} tool result(s)")
    ch = (data.get("workspace") or {}).get("changes") or {}
    if ch.get("files"):
        parts.append(f"{len(ch['files'])} file(s) changed")
    sentence = "; ".join(parts) + "."
    return f"{sentence} Severity **{facts['severity']}**."


def _affected_systems(data: dict) -> "dict[str, list[str]]":
    """StateDiff summaries grouped by world kind (file/database/dom/record).

    This is what makes the incident report agent-type aware: the same section
    reads "files affected: wrote src/app.py" for a coding agent and
    "customers/records affected: moved money: 50 (A-17)" for a support agent.
    """
    from collections import Counter

    try:
        from .action import actions as _actions
        from .packs import install_builtin

        install_builtin()
        acts = _actions(data)
    except Exception:
        return {}
    grouped: dict[str, Counter] = {}
    for a in acts:
        if a.type == "call" and a.state_diff is not None:
            grouped.setdefault(a.state_diff.kind, Counter())[a.state_diff.summary] += 1
    return {
        kind: [s if n == 1 else f"{s} (x{n})" for s, n in counts.most_common()]
        for kind, counts in grouped.items()
    }


def build_report(data: dict, path: str, why_output: str = "") -> str:
    """The incident report as markdown. Deterministic given the trace."""
    facts = _analyze(data)
    prompt = _clip(facts["episodes"][0] if facts["episodes"] else "", 80)
    verdict = "❌ failed" if facts["failed"] else "✅ completed"
    reasons = []
    if facts["stop_reason"] not in ("", "end_turn"):
        reasons.append(f"stop_reason={facts['stop_reason']}")
    if facts["truncated"]:
        reasons.append("truncated")
    if facts["paused"]:
        reasons.append("paused mid-run")
    if facts["suspects"]:
        reasons.append(f"{len(facts['suspects'])} failing tool call(s)")

    sev_badge = {"critical": "🔴 critical", "high": "🟠 high",
                 "medium": "🟡 medium", "low": "⚪ low"}[facts["severity"]]
    lines = [
        f"# Incident report: {prompt}",
        "",
        f"**Severity:** {sev_badge}"
        + (f" — {', '.join(facts['classification'])}" if facts["classification"] else ""),
        f"**Verdict:** {verdict}" + (f" ({', '.join(reasons)})" if reasons else ""),
    ]
    ws = data.get("workspace")
    if ws:
        bits = []
        if ws.get("git"):
            g = ws["git"]
            bits.append(f"commit `{g.get('commit', '')[:10]}`"
                        + (" (dirty tree)" if g.get("dirty") else "")
                        + (f" on {g['branch']}" if g.get("branch") else ""))
        if ws.get("cwd"):
            bits.append(f"cwd `{ws['cwd']}`")
        if ws.get("os"):
            bits.append(ws["os"])
        if ws.get("argv"):
            bits.append("`" + " ".join(ws["argv"][:6]) + "`")
        if bits:
            lines.append(f"**Where:** {' · '.join(bits)}")
        ch = ws.get("changes") or {}
        if ch.get("stat"):
            lines.append(f"**Workspace changes:** {ch['stat']}"
                         + (f" · dirty-hash `{ch['dirty_hash']}`" if ch.get("dirty_hash") else ""))
    # --- fixed five-section skeleton, paste-ready for Slack / a GitHub issue.

    # 1. Executive summary
    lines += ["", "## Executive summary", _executive_summary(facts, data)]

    # 2. Timeline of suspects
    lines += ["", "## Timeline of suspects"]
    for seq, s in facts["suspects"]:
        lines.append(f"- [seq {seq}] {s}")
    if facts["final_words"]:
        seq, words = facts["final_words"]
        lines.append(f"- [seq {seq}] final words: “{words}”")
    if not facts["suspects"] and not facts["final_words"]:
        lines.append("- (nothing obviously wrong in the effect log)")

    # 3. What Loom prevented
    lines += ["", "## What Loom prevented"]
    blocked = [ev for ev in facts["shield_events"] if ev.get("action") == "deny"]
    if blocked:
        for ev in blocked:
            sig = f"{ev.get('tool', '?')}({json.dumps(ev.get('input', {}), sort_keys=True, default=str)})"
            rule, via = ev.get("rule", ""), ev.get("via", "")
            how = (f"sequence rule `{rule}`" if via == "sequence"
                   else f"deny rule `{rule}`" if rule else "the shield default")
            lines.append(f"- 🛡️ blocked `{_clip(sig, 90)}` — {how}")
        egress = any(classify_all_for_event(ev) for ev in blocked)
        lines.append(
            f"\n{len(blocked)} call(s) stopped before reaching the agent; "
            + ("a network/exfiltration attempt was cut off." if egress
               else "none of them executed."))
    else:
        lines.append("- nothing was blocked (no firewall rule fired)"
                     + ("" if facts["shield_events"] else " — no firewall was active"))

    # 4. Blast radius
    lines += ["", "## Blast radius",
              f"- {facts['input_tokens'] + facts['output_tokens']:,} tokens across "
              f"{facts['model_calls']} model call(s); tools: "
              + (", ".join(f"{n}×{c}" for n, c in sorted(facts["tool_counts"].items())) or "none")]
    changes = (data.get("workspace") or {}).get("changes") or {}
    if changes.get("files"):
        shown = ", ".join(
            f"`{f['path']}`" + ("*" if f.get("pre_existing") else "")
            for f in changes["files"][:12])
        lines.append(f"- files changed: {shown}" + (" … " if len(changes["files"]) > 12 else "")
                     + "  _(* = already dirty before the run)_")
        if changes.get("agent_exit_code") is not None:
            lines.append(f"- agent process exited {changes['agent_exit_code']}")
    # Exfiltration PATH: ordered evidence beats co-occurrence -- "read the
    # secret, THEN reached the network" names the actual leak chain.
    try:
        from .action import sequence_hits

        for first, then in (("secret", "network"), ("secret", "user_communication"),
                            ("pii_access", "network"), ("pii_access", "user_communication"),
                            ("pii_access", "browser_submit")):
            hits = sequence_hits(data, first, then)
            if hits:
                a, b = hits[0]
                lines.append(f"- ⛓ exfiltration path: [{a.step}] {a.tool} → "
                             f"[{b.step}] {b.tool}")
                break
    except Exception:
        pass

    # Affected systems, from each owning pack's StateDiff -- the generic view:
    # a support agent's blast radius is customers and money, not files.
    affected = _affected_systems(data)
    for kind, summaries in sorted(affected.items()):
        label = {"file": "files", "database": "database", "dom": "browser",
                 "record": "customers/records", "field": "records"}.get(kind, kind)
        lines.append(f"- {label} affected: " + "; ".join(summaries[:8])
                     + (" …" if len(summaries) > 8 else ""))
    if facts["risk"]:
        lines.append("- risky capabilities exercised: " + ", ".join(sorted(facts["risk"])))
    if facts["secrets"]:
        lines.append("- credentials seen in tool results: "
                     + ", ".join(f"{c}× {k}" for k, c in sorted(facts["secrets"].items())))
    if facts["health"]:
        lines += [f"- {f}" for f in facts["health"]]

    # 5. How to prevent this again
    lines += ["", "## How to prevent this again"]
    if why_output:
        lines += [why_output, ""]
    from .risk import recommended_rule

    rules = [recommended_rule(c) for c in facts["risk"] if recommended_rule(c)]
    for r in rules:
        lines.append(f"- add firewall rule: `--{r}`")
    for t in _prevention(facts):
        lines.append(f"- {t}")
    lines += [
        "- turn this incident into a regression test:",
        "  ```",
        f"  loom heal {path} --agent <module:attr> --forbid ERROR --save-regression tests/traces",
        "  loom test tests/traces",
        "  ```",
    ]
    if not why_output:
        lines.append("_For an investigated root cause that cites seqs, rerun with `--why`._")
    return "\n".join(lines)
