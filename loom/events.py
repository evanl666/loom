"""Observability export: turn a trace into standard events for your stack.

A company already runs Datadog / Splunk / Grafana / an OTel collector -- it
won't watch a Loom HTML file. ``loom export --jsonl`` flattens a trace (or a
whole directory of them) into one normalized JSON event per effect, ready to
ship:

    loom export session.loom.json --jsonl events.jsonl
    loom export runs/ --jsonl - | vector   # a corpus, streamed to stdout

Each event is a flat record: run id, seq, kind, tool, tokens, risk category,
shield action. ``--otel`` emits the same as OpenTelemetry-style log records
(one JSON object per line with ``resource``/``attributes``) that an OTel
collector's file receiver can read.
"""

from __future__ import annotations

import hashlib
import json
import os


def _run_id(path: str, data: dict) -> str:
    seed = (data.get("checksum") or "") + os.path.basename(path)
    return hashlib.sha256(seed.encode()).hexdigest()[:12]


def events_for(path: str, data: dict) -> "list[dict]":
    """Flatten one trace into normalized per-effect events."""
    from .capabilities import capabilities
    from .risk import classify_all

    run = _run_id(path, data)
    model = data.get("model", "")
    prompt = (data.get("episodes") or [data.get("prompt", "")])[0]
    shield_by_tool: dict = {}
    for ev in data.get("shield_events") or []:
        shield_by_tool.setdefault(ev.get("tool", ""), []).append(ev.get("action"))

    out: list[dict] = []
    for e in data.get("log") or []:
        kind = e.get("kind", "")
        base = {"run": run, "seq": e.get("seq"), "kind": kind, "model": model,
                "prompt": prompt[:120]}
        result = e.get("result")
        if kind == "model" and isinstance(result, dict):
            usage = result.get("usage") or {}
            base.update(input_tokens=usage.get("input_tokens", 0) or 0,
                        output_tokens=usage.get("output_tokens", 0) or 0)
            calls = result.get("tool_calls") or []
            base["tool_calls"] = [c.get("name") for c in calls]
            out.append(base)
        elif kind.startswith("tool:"):
            tool = kind[5:]
            caps = sorted(capabilities(tool, {}))
            base.update(tool=tool, capabilities=caps,
                        error=isinstance(result, str) and result.startswith("ERROR:"),
                        blocked=isinstance(result, str) and result.startswith("BLOCKED:"))
            out.append(base)
        else:
            out.append(base)
    # Firewall decisions as their own events.
    for ev in data.get("shield_events") or []:
        out.append({"run": run, "kind": "shield", "tool": ev.get("tool"),
                    "action": ev.get("action"), "rule": ev.get("rule"),
                    "via": ev.get("via"),
                    "risk": sorted(classify_all(ev.get("tool", ""), ev.get("input", {})))})

    # Action-level events: the STABLE semantic layer (docs/action-schema.md).
    # One event per Action with the fields dashboards key on -- action type,
    # capabilities, risk, the firewall's decision, and the state diff kind.
    try:
        from .action import actions as _actions
        from .packs import install_builtin

        install_builtin()
        acts = _actions(data)
    except Exception:
        acts = []
    for a in acts:
        ev = {"run": run, "kind": "action", "seq": a.step, "model": model,
              "action_type": a.type, "depth": a.depth}
        if a.tool:
            ev["tool"] = a.tool
        if a.capabilities:
            ev["capabilities"] = a.capabilities
        if a.risk:
            ev["risk"] = a.risk
        if a.policy is not None:
            ev["policy_action"] = a.policy.action
            if a.policy.rule:
                ev["policy_rule"] = a.policy.rule
            if a.policy.via:
                ev["policy_via"] = a.policy.via
        if a.state_diff is not None:
            ev["state_diff_kind"] = a.state_diff.kind
            ev["state_diff_summary"] = a.state_diff.summary
        if a.observation is not None and a.observation.error:
            ev["error"] = True
        out.append(ev)
    return out


# Flat event keys -> stable OTel attribute names. These are the semantic
# contract (docs/action-schema.md): fields are only ever ADDED, never renamed,
# so a Datadog/Splunk/Grafana dashboard keyed on them doesn't break.
_OTEL_KEYS = {
    "input_tokens": "loom.tokens.input",
    "output_tokens": "loom.tokens.output",
    "action_type": "loom.action.type",
    "capabilities": "loom.capability",
    "policy_action": "loom.policy.action",
    "policy_rule": "loom.policy.rule",
    "policy_via": "loom.policy.via",
    "state_diff_kind": "loom.state_diff.kind",
    "state_diff_summary": "loom.state_diff.summary",
    "model": "loom.agent.id",
}


def to_otel(event: dict) -> dict:
    """Wrap a flat event as an OpenTelemetry-style log record.

    Attribute keys are namespaced ``loom.*`` and STABLE (see the mapping
    above and docs/action-schema.md); token usage lands in the semantic-
    convention shape ``loom.tokens.input``/``.output``.
    """
    attrs: dict = {}
    for k, v in event.items():
        if k in ("run", "kind"):
            continue
        attrs[_OTEL_KEYS.get(k, f"loom.{k}")] = v
    return {
        "resource": {"service.name": "loom-agent", "loom.run": event.get("run")},
        "name": f"loom.{event.get('kind', 'effect')}",
        "attributes": attrs,
    }


def export_events(paths: "list[str]", out, otel: bool = False) -> int:
    """Write JSONL events for every trace in ``paths`` to file-like ``out``.

    Returns the number of events written. ``out`` may be a real file or
    ``sys.stdout`` (for streaming into a collector).
    """
    n = 0
    for path in paths:
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        for ev in events_for(path, data):
            record = to_otel(ev) if otel else ev
            out.write(json.dumps(record, default=str) + "\n")
            n += 1
    return n
