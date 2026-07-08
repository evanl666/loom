"""Secret redaction for traces: store the conversation, not the credentials.

Traces record everything an agent saw -- which can include API keys read from
a .env file, tokens in tool output, passwords in pasted configs. Before a
trace leaves your machine (committed as a CI fixture, attached to a bug
report), scrub it:

    loom scrub session.loom.json                 # writes session.scrubbed.loom.json
    loom scrub session.loom.json --in-place
    loom scrub session.loom.json --check         # CI gate: exit 1 if secrets found

Detection is pattern-based (known key shapes: Anthropic, OpenAI, GitHub, AWS,
Slack, Google, JWTs, PEM blocks, ``password=...`` assignments) so it does not
mangle ordinary high-entropy content like hashes. ``--aggressive`` adds an
entropy detector for long mixed-case tokens; expect some false positives.

Recording can scrub at the persist boundary too (``loom record --scrub``):
the agent sees real values, the trace stores redacted ones.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter

# (kind, compiled pattern). Order matters: specific before generic.
PATTERNS: "list[tuple[str, re.Pattern]]" = [
    ("anthropic-key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{16,}")),
    ("openai-key", re.compile(r"sk-proj-[A-Za-z0-9_\-]{16,}|sk-[A-Za-z0-9]{32,}")),
    ("github-token", re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}")),
    ("aws-key-id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("slack-token", re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}")),
    ("google-key", re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b")),
    ("bearer", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9_\-.=]{16,}")),
    (
        "private-key",
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
    ),
    (
        # Any identifier whose last _-separated component is a credential word:
        # PASSWORD=..., DB_PASSWORD=..., AWS_SECRET_ACCESS_KEY=..., authToken: ...
        "credential-assignment",
        re.compile(
            r"(?i)\b((?:[a-z0-9]+[_-])*(?:api[_-]?key|access[_-]?key|secret[_-]?key"
            r"|private[_-]?key|(?:access|auth|api|session|refresh)[_-]?token|token"
            r"|secret|client[_-]?secret|password|passwd|credentials?))"
            r"(\s*[=:]\s*)['\"]?([^\s'\"]{8,})"
        ),
    ),
]

_ENTROPY_TOKEN = re.compile(r"[A-Za-z0-9+/=_\-]{32,}")
_HEX_ONLY = re.compile(r"^[0-9a-fA-F]+$")


def _shannon(s: str) -> float:
    counts = Counter(s)
    return -sum(c / len(s) * math.log2(c / len(s)) for c in counts.values())


def scrub_text(text: str, aggressive: bool = False) -> "tuple[str, Counter]":
    """Redact secrets in one string. Returns (clean text, counts by kind)."""
    found: Counter = Counter()

    for kind, pattern in PATTERNS:
        def _hit(m: "re.Match", kind=kind) -> str:
            found[kind] += 1
            if kind == "credential-assignment":
                return f"{m.group(1)}{m.group(2)}[scrubbed:{kind}]"
            return f"[scrubbed:{kind}]"

        text = pattern.sub(_hit, text)

    if aggressive:
        def _entropy_hit(m: "re.Match") -> str:
            token = m.group(0)
            # Hex-only strings are almost always hashes (incl. loom's own
            # effect keys), not credentials -- leave them alone.
            if _HEX_ONLY.match(token) or "[scrubbed:" in token:
                return token
            has_mix = (
                any(c.islower() for c in token)
                and any(c.isupper() for c in token)
                and any(c.isdigit() for c in token)
            )
            if has_mix and _shannon(token) >= 4.2:
                found["high-entropy"] += 1
                return "[scrubbed:high-entropy]"
            return token

        text = _ENTROPY_TOKEN.sub(_entropy_hit, text)

    return text, found


def scrub_obj(obj, aggressive: bool = False):
    """Recursively scrub every string value in a JSON-shaped object.

    Returns (new object, counts). The input is never mutated -- callers may
    still need the real values (e.g. the proxy answering its client).
    """
    found: Counter = Counter()

    def walk(x):
        if isinstance(x, str):
            clean, hits = scrub_text(x, aggressive=aggressive)
            found.update(hits)
            return clean
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v) for v in x]
        return x

    return walk(obj), found


def scrub_trace(data: dict, aggressive: bool = False) -> "tuple[dict, Counter]":
    """Scrub a whole trace dict (episodes, effects, wire, shield events...)."""
    return scrub_obj(data, aggressive=aggressive)
