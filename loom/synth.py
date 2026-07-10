"""``loom policy synthesize``: auto-generate a least-privilege policy from history.

Production teams rarely know what policy to write. Given a corpus of *successful*
runs, Loom already knows every tool the agent used and what each can do -- so it
can propose a minimal-privilege firewall: read-only tools the agent actually
needs are allowed, anything that writes / reaches the network / messages a user
is gated behind confirm, and destructive / money-moving tools are denied by
default.

    loom policy synthesize runs/ --goal least-privilege -o policy.yml
    loom policy rollout policy.yml --traces runs/     # then canary it, gated

The synthesized policy pairs with the rollout center: it won't break the runs it
was learned from (they succeeded using exactly these tools), and the rollout gate
proves it before you enforce.
"""

from __future__ import annotations

from typing import Any

_DENY_CAPS = {"destructive", "money_movement"}
_CONFIRM_CAPS = {"database_write", "browser_submit", "exec", "network",
                 "user_communication", "write"}


def synthesize_policy(source: Any, goal: str = "least-privilege", model: Any = None) -> dict:
    """A policy document (dict) proposed from the tool surface of a corpus.

    ``goal``: "least-privilege" (default deny/confirm unseen tools) or
    "observed" (default allow, only gate the risky tools that appeared).

    With ``model`` set, an LLM refines the capability-based baseline: it can
    reason about a tool's *purpose* (a ``send_email_external`` deserves confirm
    even if its caps look benign) and attach a rationale, validated back against
    the real tool set. Falls back to the deterministic policy on any failure."""
    from .scan import scan

    rep = scan(source)
    if model is not None:
        smart = _synthesize_with_llm(rep, goal, model)
        if smart is not None:
            return smart
    allow, confirm, deny = [], [], []
    for t in rep["tools"]:
        name = t["name"]
        if not name:
            continue
        caps = set(t["capabilities"])
        glob = f"{name}*"
        if caps & _DENY_CAPS:
            deny.append(glob)
        elif caps & _CONFIRM_CAPS:
            confirm.append(glob)
        elif caps:  # read / idempotent -- the agent needs these to work
            allow.append(glob)
        else:  # unclassified: gate it under least-privilege, allow under observed
            (confirm if goal == "least-privilege" else allow).append(glob)
    default = "confirm" if goal == "least-privilege" else "allow"
    return {"default": default, "allow": sorted(set(allow)),
            "confirm": sorted(set(confirm)), "deny": sorted(set(deny)),
            "_synthesized": {"goal": goal, "from_runs": rep["runs"],
                             "tools_seen": len(rep["tools"])}}


def _synthesize_with_llm(rep: dict, goal: str, model: Any) -> "dict | None":
    """LLM-refined least-privilege policy. Every rule is validated to reference a
    real tool; the default and structure are sanitized. None on any failure."""
    import json
    import re

    from .judge import _resolve

    tools = [{"name": t["name"], "capabilities": sorted(t["capabilities"])}
             for t in rep["tools"] if t.get("name")]
    if not tools:
        return None
    names = {t["name"] for t in tools}
    try:
        provider = _resolve(model)
        system = (
            "You are a security engineer writing a least-privilege firewall policy "
            "for an AI agent. For each tool decide: allow (safe/needed), confirm "
            "(gate on human approval), or deny (should never run unattended). "
            "Reason about the tool's PURPOSE and how tools COMBINE into an exfil / "
            "destruction / money-movement path, not just isolated capabilities. Use "
            "glob rules like 'send_email*'. Prefer the tightest safe policy.\n"
            f"Goal: {goal}. Reply with ONLY JSON: "
            '{"default":"allow|confirm|deny","allow":[...],"confirm":[...],'
            '"deny":[...],"rationale":{"<rule>":"<why>"}}'
        )
        user = "TOOLS:\n" + "\n".join(
            f"- {t['name']} (caps: {', '.join(t['capabilities']) or 'none'})" for t in tools)
        resp = provider.complete(system, [{"role": "user", "content": user}], [])
        m = re.search(r"\{.*\}", resp.text or "", re.S)
        doc = json.loads(m.group(0)) if m else None
        if not isinstance(doc, dict):
            return None
    except Exception:  # noqa: BLE001 -- fall back to the deterministic policy
        return None

    def _clean(rules) -> list:
        out = []
        for r in rules if isinstance(rules, list) else []:
            r = str(r)
            base = r[:-1] if r.endswith("*") else r
            if base in names or any(re.fullmatch(r.replace("*", ".*"), n) for n in names):
                out.append(r)
        return sorted(set(out))

    default = doc.get("default")
    if default not in ("allow", "confirm", "deny"):
        default = "confirm" if goal == "least-privilege" else "allow"
    rationale = {k: str(v)[:160] for k, v in (doc.get("rationale") or {}).items()
                 if isinstance(v, (str, int))}
    return {"default": default, "allow": _clean(doc.get("allow")),
            "confirm": _clean(doc.get("confirm")), "deny": _clean(doc.get("deny")),
            "rationale": rationale,
            "_synthesized": {"goal": goal, "from_runs": rep["runs"],
                             "tools_seen": len(rep["tools"]), "by": "llm"}}


def to_yaml(doc: dict) -> str:
    """A hand-rolled YAML dump (no pyyaml dependency) for a policy document."""
    meta = doc.get("_synthesized", {})
    by = " · llm-refined" if meta.get("by") == "llm" else ""
    lines = [f"# synthesized by loom ({meta.get('goal', '')}{by}) from "
             f"{meta.get('from_runs', '?')} run(s), {meta.get('tools_seen', '?')} tool(s)",
             "# review before enforcing:  loom policy rollout <this> --traces runs/"]
    rationale = doc.get("rationale") or {}
    if rationale:
        lines.append("# why:")
        lines += [f"#   {rule}: {why}" for rule, why in rationale.items()]
    lines.append(f"default: {doc['default']}")
    for section in ("allow", "confirm", "deny"):
        items = doc.get(section) or []
        if items:
            lines.append(f"{section}:")
            lines += [f'  - "{p}"' if ": " in p else f"  - {p}" for p in items]
    return "\n".join(lines) + "\n"


def describe_synth(doc: dict) -> str:
    meta = doc.get("_synthesized", {})
    return (f"synthesized {meta.get('goal', '')} policy from {meta.get('from_runs', '?')} run(s): "
            f"default {doc['default']}, {len(doc['allow'])} allow, "
            f"{len(doc['confirm'])} confirm, {len(doc['deny'])} deny")
