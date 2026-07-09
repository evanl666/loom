"""``loom diagnose``: from debugger to debugging assistant.

``loom incident`` describes what happened; ``loom diagnose`` says what to DO
about it. It classifies the failure offline, proposes a fix in a concrete
category (policy / config / tool / schema / context / security), and -- with
``--plan`` -- emits the exact replay/fork commands to verify the fix before
you ship it.

    loom diagnose failed.loom.json
    loom diagnose failed.loom.json --plan
"""

from __future__ import annotations


def diagnose(data: dict, path: str = "trace.loom.json") -> dict:
    """Classify a run's failure and propose a categorized fix + verify plan."""
    from .incident import _analyze

    facts = _analyze(data)
    stop = facts["stop_reason"]
    suspects = facts["suspects"]
    health = facts["health"]
    cls = facts["classification"]
    denied = facts["denied"]

    symptoms: list[str] = []
    if stop and stop != "end_turn":
        symptoms.append(f"stopped with stop_reason={stop!r}")
    for _seq, s in suspects[:3]:
        symptoms.append(s)
    symptoms += cls[:3]
    if health:
        symptoms.append(health[0])

    # -- classify + prescribe, most-specific first ------------------------
    last_good = _last_top_turn_before_trouble(data)

    if facts["paused"]:
        d = _fix("paused", "config",
                 "the run paused for a human answer that never came",
                 "answer it and continue, or give the agent an on_human handler",
                 [f"# in Python: Run.load({path!r}, agent=agent).resume('your answer')"])
    elif "possible exfiltration" in cls or "PII exfiltration" in cls:
        d = _fix("exfiltration", "security",
                 "the run read a secret/PII and then reached an egress channel",
                 "add a firewall rule that gates egress after a sensitive read, "
                 "e.g. --rule 'taint sk-*: confirm *' or --deny cap:network",
                 [f"loom taint {path}",
                  f"loom policy simulate {path} --profile prod-data-safe"])
    elif "destructive filesystem action" in cls:
        d = _fix("destructive", "security",
                 "the run took a destructive action (rm -rf / force-push / DROP)",
                 "deny it: --deny 'Bash(*rm -rf*)' (or cap:destructive)",
                 [f"loom policy simulate {path} --deny 'Bash(*rm -rf*)'"])
    elif denied:
        d = _fix("blocked-by-firewall", "policy",
                 f"the firewall blocked {len(denied)} call(s)",
                 "if the block was correct the agent needs a different approach; if it "
                 "was a false positive, relax the rule (see the simulation)",
                 [f"loom policy explain {path} --profile claude-code-safe",
                  f"loom fork {path} --from-step {last_good} --agent <module:attr> "
                  "--inject 'avoid the blocked action; try another way'"])
    elif stop == "budget":
        d = _fix("budget-stop", "config",
                 "the run hit its token budget",
                 "raise Agent(budget=...) or trim context with Agent(compact_after=...)",
                 [f"loom heal {path} --agent <module:attr> --forbid ERROR",
                  f"# then: Run.load({path!r}, agent=bigger_budget_agent).proceed()"])
    elif stop == "max_steps":
        d = _fix("max-steps", "config",
                 "the agent kept calling tools past max_steps (often a loop)",
                 "raise Agent(max_steps=...), or fork with a hint if it was looping",
                 [f"loom fork {path} --from-step {last_good} --agent <module:attr> "
                  "--inject 'you are repeating yourself; produce the final answer now'"])
    elif stop == "invalid_output":
        d = _fix("invalid-output", "schema",
                 "the model's final answer never matched output_type after retries",
                 "loosen the schema, raise output_retries, or clarify the format in the prompt",
                 [f"loom fork {path} --from-step {last_good} --agent <module:attr>"])
    elif any("Timeout" in s for _s, s in suspects):
        d = _fix("tool-timeout", "tool",
                 "a tool exceeded its wall-clock cap",
                 "raise Agent(tool_timeout=...) or make the tool faster/async",
                 [f"loom fork {path} --from-step {last_good} --agent <module:attr>"])
    elif suspects:
        d = _fix("tool-error", "tool",
                 "a tool returned an error the agent couldn't recover from",
                 "fix the tool or handle the error; fork past it to test a change",
                 [f"loom why {path} --step {_first_error_step(data)}",
                  f"loom fork {path} --from-step {last_good} --agent <module:attr>"])
    elif any("oversized" in f or "unused" in f or "duplicate" in f for f in health):
        d = _fix("context-rot", "context",
                 "context rot: an oversized/unused/duplicate item is degrading attention",
                 "let loom heal verify a redaction, or set Agent(compact_after=...)",
                 [f"loom heal {path} --agent <module:attr> --forbid ERROR "
                  "--save-regression tests/traces"])
    else:
        d = _fix("no-clear-failure", "none",
                 "no failure signature found -- the run completed",
                 "if the OUTPUT is wrong, inspect why it acted as it did / whether its "
                 "claims are supported",
                 [f"loom why {path} --step <N>", f"loom provenance {path}"])

    d["symptoms"] = symptoms
    d["severity"] = facts["severity"]
    return d


def _fix(failure, category, diagnosis, suggestion, verify):
    return {"failure": failure, "fix_category": category, "diagnosis": diagnosis,
            "suggestion": suggestion, "verify": verify}


def _last_top_turn_before_trouble(data: dict) -> int:
    """A reasonable fork point: the last top-level turn index (so a fork
    re-runs the final decision). Falls back to 0."""
    from .action import effect_dicts

    turns = sum(1 for e in effect_dicts(data)
                if e.get("kind") == "model" and not e.get("depth", 0))
    return max(0, turns - 1)


def _first_error_step(data: dict) -> int:
    from .action import effect_dicts

    for e in effect_dicts(data):
        if e.get("kind", "").startswith("tool:") and isinstance(e.get("result"), str):
            if e["result"].startswith(("ERROR", "BLOCKED")):
                return e.get("seq", 0)
    return 0


def describe_diagnosis(d: dict, plan: bool = False) -> str:
    lines = [f"diagnosis: {d['failure']}  (severity: {d['severity']}, "
             f"fix category: {d['fix_category']})",
             f"  {d['diagnosis']}"]
    if d["symptoms"]:
        lines.append("  symptoms:")
        for s in d["symptoms"]:
            lines.append(f"    - {s}")
    lines.append(f"  suggested fix: {d['suggestion']}")
    if plan:
        lines.append("  verify the fix:")
        for c in d["verify"]:
            lines.append(f"    {c}")
    return "\n".join(lines)
