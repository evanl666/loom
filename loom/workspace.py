"""Workspace metadata: what the recording was OF, beyond the API traffic.

A coding-agent trace is far more useful for debugging when it remembers not
just *where* it ran -- directory, git commit, dirty tree, argv, OS -- but
*what the agent did to the workspace*. Replaying the API traffic reproduces
what the model said; it does not reproduce the files it changed. So around a
``loom record`` we snapshot ``git diff`` before and after the agent runs and
record the delta: which files changed, a diff-stat, and a hash of the working
tree (so you can tell whether a trace still matches the repo).

All best-effort and non-fatal -- no git, an odd platform, or a non-repo just
leaves fields out. Read-only: we never touch the index or stash.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys

# Full patches can be large and can carry secrets; capped, and only embedded
# when explicitly asked for (--capture-diff). The summary always fits.
_DIFF_CAP = 200_000


def _git(args: "list[str]", cwd: str, strip: bool = True) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10
        )
        if out.returncode != 0:
            return ""
        # porcelain output is column-sensitive: stripping eats the leading
        # status space on the first line. Callers that parse columns pass
        # strip=False.
        return out.stdout.strip() if strip else out.stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _porcelain_files(cwd: str) -> "list[tuple[str, str]]":
    """(status, path) for every change incl. UNTRACKED, via git status.

    git diff omits untracked files; status --porcelain doesn't, so a brand-new
    file the agent wrote still shows up. XY codes are normalized to M/A/D.
    """
    out = []
    for line in _git(["status", "--porcelain"], cwd, strip=False).splitlines():
        if len(line) < 4:
            continue
        code, path = line[:2], line[3:]
        if " -> " in path:  # a rename: report the new name
            path = path.split(" -> ", 1)[1]
        if code == "??":
            status = "A"  # untracked == newly added
        else:
            letters = code.replace(" ", "")
            status = "D" if "D" in letters else ("A" if "A" in letters else letters[:1] or "M")
        out.append((status, path.strip('"')))
    return out


def diff_snapshot(cwd: str) -> dict:
    """The working tree's state vs HEAD: file list (incl untracked), stat, hash."""
    porcelain = _git(["status", "--porcelain"], cwd, strip=False)
    full = _git(["diff", "HEAD"], cwd)
    return {
        "stat": _git(["diff", "HEAD", "--stat"], cwd),
        "files": _porcelain_files(cwd),
        # Hash the whole porcelain state (covers untracked too) -- the
        # fingerprint for "does this trace still match the repo?".
        "hash": hashlib.sha256(porcelain.encode()).hexdigest()[:16] if porcelain else "",
        "full": full,
    }


def _file_sha(cwd: str, path: str) -> str:
    """Content hash of a file, so undo can tell if it still holds what the
    agent left. '' if it doesn't exist (a deletion)."""
    try:
        with open(os.path.join(cwd, path), "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except OSError:
        return ""


def changes_since(before: dict, after: dict, agent_exit_code=None,
                  capture_diff: bool = False, cwd: "str | None" = None) -> dict:
    """What the agent did to the workspace, from before/after snapshots.

    Files present in the after-diff but not the before-diff are the agent's;
    files dirty in both are flagged ``pre_existing`` so the reader can tell
    the agent's work from what was already uncommitted. Each file also gets a
    content ``sha`` (at record time) so ``loom undo`` can revert only files
    that still hold exactly what the agent left.
    """
    cwd = cwd or os.getcwd()
    before_paths = {p for _, p in before.get("files", [])}
    changed = [
        {"status": s, "path": p, "pre_existing": p in before_paths,
         "sha": _file_sha(cwd, p)}
        for s, p in after.get("files", [])
    ]
    stat_summary = ""
    for line in reversed(after.get("stat", "").splitlines()):
        if "changed" in line:  # e.g. " 3 files changed, 40 insertions(+), 2 deletions(-)"
            stat_summary = line.strip()
            break
    if not stat_summary and changed:  # untracked-only edits have no diff --stat
        stat_summary = f"{len(changed)} file(s) changed"
    out: dict = {
        "files": changed,
        "stat": stat_summary,
        "dirty_hash": after.get("hash", ""),
        "agent_exit_code": agent_exit_code,
    }
    if capture_diff and after.get("full"):
        out["diff"] = after["full"][:_DIFF_CAP]
        out["diff_truncated"] = len(after["full"]) > _DIFF_CAP
    return out


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
