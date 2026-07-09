"""Workspace metadata: what the recording was OF, beyond the API traffic.

A coding-agent trace is far more useful for debugging when it also remembers
where it ran: the directory, the git commit (and whether the tree was dirty),
the command line, the OS and Python, the API dialect. All best-effort and
non-fatal -- a missing git or an odd platform just leaves fields out.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys


def _git(args: "list[str]", cwd: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def collect(command: "list[str] | None" = None, target: str = "") -> dict:
    """Snapshot the current workspace. Every field is best-effort."""
    cwd = os.getcwd()
    info: dict = {
        "cwd": cwd,
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "recorded_at": _now(),
    }
    if command:
        info["argv"] = list(command)
    if target:
        info["dialect"] = "openai" if "openai" in target else "anthropic"

    commit = _git(["rev-parse", "HEAD"], cwd)
    if commit:
        info["git"] = {
            "commit": commit,
            "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd),
            # A dirty tree is the single most useful field: "it reproduced on
            # THIS uncommitted state" is exactly what a bug report needs.
            "dirty": bool(_git(["status", "--porcelain"], cwd)),
        }
    return info


def _now() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
