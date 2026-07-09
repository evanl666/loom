"""``loom alert``: thresholds over the fleet, for CI/cron -- not just a dashboard.

A platform team doesn't watch dashboards; it gets paged. This evaluates a
corpus against thresholds and exits 1 (and/or POSTs a Slack-compatible
webhook) when any breaches:

    # alerts.yml
    alerts:
      - {metric: failure_rate, max: 10}          # percent
      - {metric: tokens_p95, max: 50000}
      - {metric: money_movement_actions, max: 0}
      - {metric: pii_to_comm_paths, max: 0}      # ordered PII -> outbound message
      - {metric: blocked_actions, max: 5}
    webhook: https://hooks.slack.com/...

    loom alert runs/ --config alerts.yml         # cron/CI: exit 1 on breach
"""

from __future__ import annotations

import json


def _metrics(paths: "list[str]") -> dict:
    """Every alertable number over the corpus."""
    from .action import sequence_hits
    from .kpi import compute_kpis
    from .packs import install_builtin

    install_builtin()
    k = compute_kpis(paths)
    caps = {c["capability"]: c for c in k["capabilities"]}

    pii_to_comm = 0
    for p in paths:
        try:
            with open(p) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if sequence_hits(data, "pii_access", "user_communication"):
            pii_to_comm += 1

    return {
        "runs": k["runs"],
        "failure_rate": k["failure_rate"],
        "risky_actions": k["risky_calls"],
        "blocked_actions": k["blocked_actions"],
        "tokens_p95": k["tokens"]["p95"],
        "tokens_total": k["tokens"]["total"],
        "money_movement_actions": caps.get("money_movement", {}).get("actions", 0),
        "pii_access_actions": caps.get("pii_access", {}).get("actions", 0),
        "database_write_actions": caps.get("database_write", {}).get("actions", 0),
        "user_communication_actions": caps.get("user_communication", {}).get("actions", 0),
        "pii_to_comm_paths": pii_to_comm,
    }


def evaluate(paths: "list[str]", config: dict) -> "tuple[list[dict], dict]":
    """Evaluate alert rules. Returns (results, metrics); breached rules first."""
    metrics = _metrics(paths)
    results = []
    for rule in config.get("alerts", []):
        name = rule.get("metric", "")
        if name not in metrics:
            results.append({"metric": name, "error": f"unknown metric (have: "
                            f"{', '.join(sorted(metrics))})", "breached": True})
            continue
        value, cap = metrics[name], rule.get("max")
        breached = cap is not None and value > float(cap)
        results.append({"metric": name, "value": value, "max": cap, "breached": breached})
    results.sort(key=lambda r: not r["breached"])
    return results, metrics


def post_webhook(url: str, breached: "list[dict]", runs: int) -> bool:
    """POST a Slack-compatible alert payload. Best-effort; True on 2xx."""
    import urllib.request

    lines = [f"🚨 loom alert: {len(breached)} threshold(s) breached over {runs} run(s)"]
    for r in breached:
        lines.append(f"• {r['metric']}: {r.get('value')} > {r.get('max')}"
                     if "value" in r else f"• {r['metric']}: {r.get('error')}")
    payload = {"text": "\n".join(lines)}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"content-type": "application/json"},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return 200 <= resp.status < 300
    except OSError:
        return False


def describe(results: "list[dict]", metrics: dict) -> str:
    lines = [f"alerts over {metrics['runs']} run(s):"]
    for r in results:
        if r.get("error"):
            lines.append(f"  ⚠️ {r['metric']}: {r['error']}")
        else:
            mark = "🚨" if r["breached"] else "ok"
            lines.append(f"  {mark:>3}  {r['metric']:<28} {r['value']}"
                         + (f"  (max {r['max']})" if r.get("max") is not None else ""))
    return "\n".join(lines)
