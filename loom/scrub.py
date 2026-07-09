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


class ScrubConfig:
    """Enterprise scrub policy: custom detectors + an allowlist.

    ``detectors`` are extra ``(name, regex)`` pairs (a company's own secret
    shapes); ``allow`` is a list of literal strings or globs that must never be
    redacted (documented example keys, known-safe values) -- the false-positive
    escape hatch. Loaded from ``loom-scrub.yml``/``.json``.
    """

    def __init__(self, detectors: "list[tuple[str, str]] | None" = None,
                 allow: "list[str] | None" = None):
        self.detectors = [(n, re.compile(p)) for n, p in (detectors or [])]
        self.allow = list(allow or [])

    def allowed(self, value: str) -> bool:
        from fnmatch import fnmatchcase as fnmatch

        return any(value == a or fnmatch(value, a) for a in self.allow)


def load_scrub_config(path: str) -> ScrubConfig:
    """Load a scrub config (YAML or JSON) via the bounded policy parser."""
    from .policy_file import _parse

    with open(path) as f:
        doc = _parse(f.read(), path) or {}
    detectors = list((doc.get("detectors") or {}).items())
    return ScrubConfig(detectors=detectors, allow=doc.get("allow") or [])


def _shannon(s: str) -> float:
    counts = Counter(s)
    return -sum(c / len(s) * math.log2(c / len(s)) for c in counts.values())


def scrub_text(text: str, aggressive: bool = False,
               config: "ScrubConfig | None" = None) -> "tuple[str, Counter]":
    """Redact secrets in one string. Returns (clean text, counts by kind).

    Custom detectors from ``config`` run first (most specific), and any match
    in the config's allowlist is left intact.
    """
    found: Counter = Counter()
    allow = config.allowed if config is not None else (lambda v: False)
    patterns = ((config.detectors if config is not None else []) + PATTERNS)

    for kind, pattern in patterns:
        def _hit(m: "re.Match", kind=kind) -> str:
            matched = m.group(0)
            if kind == "credential-assignment":
                if allow(m.group(3)):        # the value is allowlisted -> keep
                    return matched
                found[kind] += 1
                return f"{m.group(1)}{m.group(2)}[scrubbed:{kind}]"
            if allow(matched):               # the whole match is allowlisted -> keep
                return matched
            found[kind] += 1
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


def scrub_obj(obj, aggressive: bool = False, config: "ScrubConfig | None" = None,
              audit: "list | None" = None):
    """Recursively scrub every string value in a JSON-shaped object.

    Returns (new object, counts). The input is never mutated -- callers may
    still need the real values (e.g. the proxy answering its client). Pass an
    ``audit`` list to collect ``{path, kind}`` records (paths, never values) of
    where redactions happened.
    """
    found: Counter = Counter()

    def walk(x, path):
        if isinstance(x, str):
            clean, hits = scrub_text(x, aggressive=aggressive, config=config)
            found.update(hits)
            if audit is not None:
                for kind, n in hits.items():
                    audit.append({"path": path or "(root)", "kind": kind, "count": n})
            return clean
        if isinstance(x, dict):
            return {k: walk(v, f"{path}.{k}" if path else k) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v, f"{path}[{i}]") for i, v in enumerate(x)]
        return x

    return walk(obj, ""), found


def scrub_trace(data: dict, aggressive: bool = False,
                config: "ScrubConfig | None" = None) -> "tuple[dict, Counter]":
    """Scrub a whole trace dict (episodes, effects, wire, shield events...)."""
    return scrub_obj(data, aggressive=aggressive, config=config)


def audit_report(data: dict, aggressive: bool = False,
                 config: "ScrubConfig | None" = None) -> dict:
    """What would be redacted, and where -- for a compliance record. No values."""
    entries: list = []
    scrub_obj(data, aggressive=aggressive, config=config, audit=entries)
    by_kind: Counter = Counter()
    for e in entries:
        by_kind[e["kind"]] += e["count"]
    # Collapse to unique (path, kind) with summed counts, sorted by kind.
    return {
        "total": sum(by_kind.values()),
        "by_kind": dict(sorted(by_kind.items())),
        "locations": sorted(entries, key=lambda e: (e["kind"], e["path"])),
    }
