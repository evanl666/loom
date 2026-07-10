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


def synthesize_policy(source: Any, goal: str = "least-privilege") -> dict:
    """A policy document (dict) proposed from the tool surface of a corpus.

    ``goal``: "least-privilege" (default deny/confirm unseen tools) or
    "observed" (default allow, only gate the risky tools that appeared).
    """
    from .scan import scan

    rep = scan(source)
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


def to_yaml(doc: dict) -> str:
    """A hand-rolled YAML dump (no pyyaml dependency) for a policy document."""
    meta = doc.get("_synthesized", {})
    lines = [f"# synthesized by loom ({meta.get('goal', '')}) from "
             f"{meta.get('from_runs', '?')} run(s), {meta.get('tools_seen', '?')} tool(s)",
             "# review before enforcing:  loom policy rollout <this> --traces runs/",
             f"default: {doc['default']}"]
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
