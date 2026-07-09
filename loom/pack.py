"""``loom pack``: a self-contained incident bundle you can hand to anyone.

A production incident needs one thing above all: *reproducibility*. ``loom
pack`` rolls everything a colleague needs to understand and rerun a failed
agent run into a single ``.loompack`` (a zip):

    loom pack session.loom.json          # -> session.loompack

Inside:

  trace.loom.json     the run, SCRUBBED (safe to share) and checksummed
  incident.md         the five-section postmortem
  studio.html         the visual, offline time-travel viewer
  workspace.patch     the git diff the agent produced (if recorded)
  manifest.json       loom version, git commit, OS, checksum, contents
  README.md           how to replay and inspect it

Only the scrubbed trace goes in, so a pack is safe to attach to an issue.
"""

from __future__ import annotations

import io
import json
import os
import platform
import zipfile

from . import __version__


def build_pack(trace_path: str, out: "str | None" = None) -> "tuple[str, int]":
    """Write a .loompack for a trace. Returns (path, secrets_redacted)."""
    from .export import trace_to_html
    from .incident import build_report
    from .scrub import scrub_trace
    from .trace import trace_checksum

    with open(trace_path) as f:
        data = json.load(f)

    clean, found = scrub_trace(data)
    clean["scrubbed"] = True
    if "checksum" in clean:
        clean["checksum"] = trace_checksum(clean)

    incident = build_report(clean, "trace.loom.json")
    studio = trace_to_html(clean, path="trace.loom.json")
    ws = (data.get("workspace") or {})
    patch = (ws.get("changes") or {}).get("diff", "")

    manifest = {
        "loom_version": __version__,
        "created": _now(),
        "trace_checksum": clean.get("checksum", ""),
        "secrets_redacted": sum(found.values()),
        "workspace": {k: ws.get(k) for k in ("git", "os", "cwd") if ws.get(k)},
        "system": {"python": platform.python_version(), "platform": platform.platform()},
        "contents": ["trace.loom.json", "incident.md", "studio.html", "manifest.json",
                     "README.md"] + (["workspace.patch"] if patch else []),
    }

    if out is None:
        base = trace_path[: -len(".loom.json")] if trace_path.endswith(".loom.json") else trace_path
        out = base + ".loompack"

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("trace.loom.json", json.dumps(clean, indent=2))
        z.writestr("incident.md", incident + "\n")
        z.writestr("studio.html", studio)
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
        z.writestr("README.md", _readme(manifest))
        if patch:
            z.writestr("workspace.patch", patch)
    return out, sum(found.values())


def _readme(manifest: dict) -> str:
    g = (manifest.get("workspace") or {}).get("git") or {}
    where = f"commit `{g.get('commit', '?')[:10]}`" + (" (dirty)" if g.get("dirty") else "") if g else ""
    return f"""# Loom incident pack

Produced by loom {manifest['loom_version']} on {manifest['created']}.
{('Recorded at ' + where) if where else ''}

- **incident.md** — the postmortem (severity, timeline, what was blocked,
  blast radius, how to prevent it again).
- **studio.html** — open in a browser to time-travel through the run.
- **trace.loom.json** — the recording, secrets scrubbed
  ({manifest['secrets_redacted']} redacted). Replay it offline:

  ```
  pip install loom-harness
  loom replay trace.loom.json         # reconstruct the run, zero API calls
  loom studio trace.loom.json         # the viewer
  loom proxy --replay trace.loom.json # serve it back as a mock API
  ```

- **workspace.patch** — the file changes the agent made (`git apply` to
  reproduce them). Present only if the run was recorded with `--capture-diff`.
- **manifest.json** — versions, git metadata, checksum, contents.
"""


def _now() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
