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
    if facts["secrets"] or "secret-read" in risk:
        tags.append("secret exposure")
    if ("secret-read" in risk or facts["secrets"]) and "network-egress" in risk:
        tags.append("possible exfiltration")
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
    if "possible exfiltration" in cls or "destructive filesystem action" in cls:
        return "critical"
    if "secret exposure" in cls or "unsafe shell" in cls or facts["denied"]:
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
    lines += [
        f"**Blast radius:** {facts['input_tokens'] + facts['output_tokens']:,} tokens "
        f"across {facts['model_calls']} model call(s); tools touched: "
        + (", ".join(f"{n}×{c}" for n, c in sorted(facts["tool_counts"].items())) or "none"),
        "",
        "## Timeline of suspects",
    ]
    for seq, s in facts["suspects"]:
        lines.append(f"- [seq {seq}] {s}")
    if facts["final_words"]:
        seq, words = facts["final_words"]
        lines.append(f"- [seq {seq}] final words: “{words}”")
    if not facts["suspects"] and not facts["final_words"]:
        lines.append("- (nothing obviously wrong in the effect log)")

    if facts["shield_events"]:
        lines += ["", "## Firewall decisions"]
        for ev in facts["shield_events"]:
            sig = f"{ev.get('tool', '?')}({json.dumps(ev.get('input', {}), sort_keys=True, default=str)})"
            lines.append(
                f"- {ev.get('action', '?')} {_clip(sig, 90)} — rule `{ev.get('rule', '')}`"
                f" via {ev.get('via', '?')}"
            )

    if facts["health"]:
        lines += ["", "## Context health"]
        lines += [f"- {f}" for f in facts["health"]]

    if facts["secrets"]:
        lines += ["", "## Secrets sighted"]
        lines += [f"- {count}× {kind} in tool results" for kind, count in sorted(facts["secrets"].items())]

    if facts["risk"]:
        from .risk import recommended_rule

        lines += ["", "## Risky capabilities exercised"]
        for cat, example in facts["risk"].items():
            lines.append(f"- **{cat}**: `{example}`")
        rules = [recommended_rule(c) for c in facts["risk"] if recommended_rule(c)]
        if rules:
            lines += ["", "## Recommended firewall rules"]
            lines.append("Add to `loom record` (or a `--profile`):")
            lines += [f"- `--{r}`" for r in rules]

    lines += ["", "## Root cause"]
    if why_output:
        lines.append(why_output)
    else:
        lines.append(
            "_(heuristics above; for an investigated narrative that cites seqs, rerun "
            "with `--why` -- a debugger agent reads the trace through tools)_"
        )

    tips = _prevention(facts)
    if tips:
        lines += ["", "## Prevention"]
        lines += [f"- {t}" for t in tips]

    lines += [
        "",
        "## Turn this incident into a regression test",
        "```",
        f"loom heal {path} --agent <module:attr> --forbid ERROR --save-regression tests/traces",
        "loom test tests/traces          # now it's CI",
        "```",
        f"_Or scrub and keep the recording itself: `loom scrub {path}` then commit it "
        f"for `loom impact` / `verify_replay`._",
    ]
    return "\n".join(lines)
