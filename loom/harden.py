"""``loom harden``: a deployment recommendation from the threat model.

``docs/threat-model.md`` says what Shield and the sandbox stop. ``loom harden``
turns that into an executable starting point: for a given scenario it emits a
recommended policy file, the ``loom record`` flags to run with, and a plain
explanation of what each choice defends against.

    loom harden --scenario support        # prints the recommendation
    loom harden --scenario coding -o loom-policy.yml   # writes the policy
"""

from __future__ import annotations

# scenario -> (profile, record-flags, rationale lines)
_SCENARIOS = {
    "coding": {
        "profile": "claude-code-safe",
        "flags": ["--sandbox", "--scrub"],
        "why": [
            "claude-code-safe: reads/tests flow, network/installs/pushes ask, "
            "secrets and destructive shell are blocked.",
            "--sandbox (macOS) or --container: the proxy becomes the only network "
            "door, so Shield can't be bypassed.",
            "--scrub: secrets never reach the trace on disk.",
        ],
    },
    "data": {
        "profile": "prod-data-safe",
        "flags": ["--scrub", "--require-approver", "cap:pii_access=data-steward"],
        "why": [
            "prod-data-safe: reads ask, writes/deletes/egress denied, a credential "
            "sighting locks everything to confirm.",
            "--require-approver cap:pii_access=data-steward: only a data steward may "
            "approve reads of personal data.",
            "point tools at a read-replica / test DB (see loom packs safe-runtime).",
        ],
    },
    "browser": {
        "profile": "prod-data-safe",
        "flags": ["--confirm", "cap:browser_submit"],
        "why": [
            "gate every form submit (cap:browser_submit) -- a submit is an "
            "irreversible external side effect.",
            "drive a staging profile with submission disabled while debugging "
            "(loom packs safe-runtime).",
        ],
    },
    "support": {
        "profile": "customer-data-safe",
        "flags": ["--require-approver", "cap:money_movement=2:manager,finance",
                  "--require-approver", "cap:user_communication=support-lead",
                  "--sign-approvals-key-env", "LOOM_APPROVAL_KEY"],
        "why": [
            "customer-data-safe: aggregate reads ask, exports blocked, a PII/"
            "credential sighting cuts egress.",
            "money movement needs TWO approvers (manager + finance); outbound "
            "messages need the support lead.",
            "sign the decisions (--sign-approvals-key-env) so the approval trail is "
            "tamper-proof; verify with loom trace verify-approvals.",
            "point tools at a sandbox tenant with payment idempotency keys.",
        ],
    },
    "ci": {
        "profile": "github-actions-safe",
        "flags": ["--shield-default", "deny"],
        "why": [
            "github-actions-safe: fully non-interactive (nothing waits for a human), "
            "read/build/test allowed, secrets and egress denied.",
            "default deny: anything not explicitly allowed is blocked -- an agent in "
            "CI should never do something novel.",
            "gate PRs with loom policy simulate --fail-on-deny and the GitHub Action's "
            "fail-on-new-risk.",
        ],
    },
}


def scenarios() -> "list[str]":
    return sorted(_SCENARIOS)


def harden(scenario: str) -> dict:
    if scenario not in _SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; try: {', '.join(scenarios())}")
    return _SCENARIOS[scenario]


def policy_yaml(scenario: str) -> str:
    """The recommended policy file for a scenario (extends its profile)."""
    from .policy_file import PROFILES

    rec = _SCENARIOS[scenario]
    prof = PROFILES[rec["profile"]]
    lines = [f"# loom harden --scenario {scenario}",
             f"# {prof.get('description', '')}",
             f"profile: {rec['profile']}"]
    # Fold any --require-approver flags into the file as require_approver.
    approvers: dict = {}
    it = iter(rec["flags"])
    for flag in it:
        if flag == "--require-approver":
            spec = next(it, "")
            pattern, _, names = spec.partition("=")
            head, sep, rest = names.partition(":")
            if sep and head.isdigit():
                approvers[pattern] = {"names": [n.strip() for n in rest.split(",")],
                                      "min": int(head)}
            else:
                approvers[pattern] = {"names": [n.strip() for n in names.split(",")], "min": 1}
    if approvers:
        lines.append("require_approver:")
        for pat, spec in approvers.items():
            lines.append(f'  "{pat}":')
            lines.append(f"    names: [{', '.join(spec['names'])}]")
            lines.append(f"    min: {spec['min']}")
    return "\n".join(lines) + "\n"


def describe(scenario: str) -> str:
    rec = _SCENARIOS[scenario]
    # Show record flags minus the approver specs (those live in the policy file).
    shown, it = [], iter(rec["flags"])
    for flag in it:
        if flag == "--require-approver":
            next(it, "")
            continue
        shown.append(flag)
    cmd = "loom record --policy loom-policy.yml " + " ".join(shown) + " -- <your agent>"
    lines = [f"Hardening for a {scenario} agent:", "",
             f"  recommended profile: {rec['profile']}",
             f"  run with:\n    {cmd}", "", "  why:"]
    lines += [f"    - {w}" for w in rec["why"]]
    lines += ["", "  write the policy: loom harden --scenario "
              f"{scenario} -o loom-policy.yml"]
    return "\n".join(lines)
