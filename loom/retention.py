"""``loom retention``: age-based lifecycle for a trace corpus.

Traces accumulate, and raw ones hold secrets and PII. A retention policy keeps
the corpus useful without keeping raw sensitive data forever:

    # retention.yml
    scrub_after: 30d     # redact secrets (and --pii) in place past this age
    delete_after: 90d    # remove the trace entirely past this age
    redact_pii: true     # also redact emails/SSNs/cards/phones when scrubbing

    loom retention runs/ --config retention.yml            # dry-run: what would happen
    loom retention runs/ --config retention.yml --apply    # do it, write an audit log

Offline and reversible-until-applied. Legal hold, access audit, and remote
storage remain the commercial layer; this is the local lifecycle + a
per-file audit record of what was kept, scrubbed, or purged.
"""

from __future__ import annotations

import json
import os
import re
import time
from glob import glob

_DUR = re.compile(r"^\s*(\d+)\s*([dhw])\s*$")
_UNIT_SECONDS = {"h": 3600, "d": 86400, "w": 604800}

# PII value shapes, redacted when redact_pii is set (credentials are covered by
# the scrub detectors already).
_PII_DETECTORS = [
    ("email", r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    ("ssn", r"\b\d{3}-\d{2}-\d{4}\b"),
    ("credit-card", r"\b(?:\d[ -]?){13,16}\b"),
    ("phone", r"\b\+?\d[\d\-() ]{8,}\d\b"),
]


def _seconds(duration: str) -> "int | None":
    m = _DUR.match(str(duration))
    return int(m.group(1)) * _UNIT_SECONDS[m.group(2)] if m else None


def load_retention(path: str) -> dict:
    from .policy_file import _parse

    with open(path) as f:
        doc = _parse(f.read(), path) or {}
    if not isinstance(doc, dict):
        raise ValueError(
            f"{path} is not a valid retention config (expected a mapping of "
            f"settings like scrub_after/delete_after, got {type(doc).__name__})"
        )
    return doc


def plan_retention(directory: str, config: dict, now: "float | None" = None) -> "list[dict]":
    """What retention WOULD do to each trace (no changes made).

    ``legal_hold`` patterns (path globs) exempt matching traces from BOTH
    scrubbing and deletion -- a held trace is evidence; age doesn't apply.
    """
    from fnmatch import fnmatchcase

    now = now if now is not None else time.time()
    scrub_after = _seconds(config.get("scrub_after", ""))
    delete_after = _seconds(config.get("delete_after", ""))
    holds = config.get("legal_hold") or []
    out = []
    for path in sorted(glob(os.path.join(directory, "**", "*.loom.json"), recursive=True)):
        try:
            age = now - os.path.getmtime(path)
        except OSError:
            continue
        age_days = round(age / 86400, 1)
        already = _is_scrubbed(path)
        base = os.path.basename(path)
        held = any(fnmatchcase(base, h) or fnmatchcase(path, h) for h in holds)
        if held:
            action = "hold"
        elif delete_after is not None and age >= delete_after:
            action = "delete"
        elif scrub_after is not None and age >= scrub_after and not already:
            action = "scrub"
        else:
            action = "keep"
        out.append({"path": path, "age_days": age_days, "action": action,
                    "already_scrubbed": already})
    return out


def dsar(directory: str, value: str, mode: str = "plan") -> "list[dict]":
    """Data-subject request: find (and optionally purge) a person's identifier.

    ``value`` is the subject's identifier (an email, an ID). ``mode``:
    "plan" lists matching traces; "scrub" redacts every occurrence in place
    (to ``[scrubbed:dsar]``, re-checksummed); "delete" removes the files.
    Returns per-file audit records. Substring match over the raw JSON, so it
    catches the value wherever it sits -- inputs, results, or wire data.
    """
    from .trace import trace_checksum

    if len(value) < 4:
        raise ValueError("a DSAR identifier under 4 characters would over-match")
    audit = []
    for path in sorted(glob(os.path.join(directory, "**", "*.loom.json"), recursive=True)):
        try:
            with open(path) as f:
                raw = f.read()
        except OSError:
            continue
        count = raw.count(value)
        if not count:
            continue
        record = {"path": path, "occurrences": count, "mode": mode,
                  "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        if mode == "delete":
            try:
                os.remove(path)
            except OSError as e:
                record["error"] = str(e)
        elif mode == "scrub":
            try:
                data = json.loads(raw.replace(value, "[scrubbed:dsar]"))
                if "checksum" in data:
                    data["checksum"] = trace_checksum(data)
                with open(path, "w") as f:
                    json.dump(data, f, indent=2)
            except (ValueError, OSError) as e:
                record["error"] = str(e)
        audit.append(record)
    return audit


def _is_scrubbed(path: str) -> bool:
    try:
        with open(path) as f:
            return bool(json.load(f).get("scrubbed"))
    except (OSError, json.JSONDecodeError):
        return False


def apply_retention(directory: str, config: dict, dry_run: bool = True,
                    now: "float | None" = None) -> "list[dict]":
    """Execute the retention plan (unless dry_run). Returns the audit records."""
    from .scrub import ScrubConfig, scrub_trace
    from .trace import trace_checksum

    plan = plan_retention(directory, config, now=now)
    scrub_config = None
    if config.get("redact_pii"):
        scrub_config = ScrubConfig(detectors=_PII_DETECTORS)

    audit = []
    for item in plan:
        record = {"path": item["path"], "age_days": item["age_days"],
                  "action": item["action"], "applied": not dry_run,
                  "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        if not dry_run and item["action"] == "delete":
            try:
                os.remove(item["path"])
            except OSError as e:
                record["error"] = str(e)
        elif not dry_run and item["action"] == "scrub":
            try:
                with open(item["path"]) as f:
                    data = json.load(f)
                clean, found = scrub_trace(data, config=scrub_config)
                clean["scrubbed"] = True
                if "checksum" in clean:
                    clean["checksum"] = trace_checksum(clean)
                with open(item["path"], "w") as f:
                    json.dump(clean, f, indent=2)
                record["secrets_redacted"] = sum(found.values())
            except (OSError, json.JSONDecodeError) as e:
                record["error"] = str(e)
        audit.append(record)
    return audit


def summarize(audit: "list[dict]") -> str:
    from collections import Counter

    counts = Counter(a["action"] for a in audit)
    redacted = sum(a.get("secrets_redacted", 0) for a in audit)
    verb = "applied" if any(a["applied"] for a in audit) else "would"
    lines = [f"retention ({verb}): {counts.get('keep', 0)} kept, "
             f"{counts.get('scrub', 0)} scrubbed, {counts.get('delete', 0)} deleted"
             + (f", {counts['hold']} on legal hold" if counts.get("hold") else "")]
    if redacted:
        lines.append(f"  {redacted} secret(s) redacted")
    errs = [a for a in audit if a.get("error")]
    if errs:
        lines.append(f"  ⚠️ {len(errs)} error(s)")
    return "\n".join(lines)
