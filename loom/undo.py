"""``loom undo``: revert the file changes an agent made.

The workspace metadata on a trace records exactly which files the agent
touched (and, with ``--capture-diff``, the patch). ``loom undo`` puts the
tree back:

    loom undo session.loom.json            # revert the agent's edits
    loom undo session.loom.json --dry-run  # show what would be reverted
    loom undo session.loom.json --only src/  # only under this path

Tracked files the agent modified are ``git checkout``'d back to HEAD; files it
added are removed. Files that were *already dirty before the run* are left
alone -- undo touches the agent's work, not yours. Each target is reverted
only if it still holds exactly what the agent left (content hash match); if a
file changed again after the recording, undo skips it (or ``--force`` reverts
anyway).
"""

from __future__ import annotations

import os
import subprocess


def _git(args: "list[str]", cwd: str) -> "tuple[int, str]":
    try:
        p = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=30)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except (OSError, subprocess.SubprocessError) as e:
        return 1, str(e)


def plan_undo(data: dict, only: str = "") -> "list[dict]":
    """The files loom undo would touch (the agent's own, in scope)."""
    changes = (data.get("workspace") or {}).get("changes") or {}
    return [
        f for f in changes.get("files", [])
        if not f.get("pre_existing")  # leave what was already dirty alone
        and (not only or f.get("path", "").startswith(only))
    ]


def undo(data: dict, cwd: str, only: str = "", dry_run: bool = False,
         force: bool = False) -> "tuple[bool, list[str]]":
    """Revert the agent's changes. Returns (ok, log lines)."""
    from .workspace import _file_sha

    files = plan_undo(data, only=only)
    if not files:
        return True, ["nothing to undo (no agent file changes recorded)"]

    # A file that changed again since the recording no longer holds what the
    # agent left -- reverting it would clobber that newer work.
    moved = [f for f in files if f.get("sha") and _file_sha(cwd, f["path"]) != f["sha"]]
    if moved and not force:
        return False, [
            "these files changed since the recording, so undo would clobber newer work "
            "(pass --force to revert anyway):",
            *[f"  {f['path']}" for f in moved],
        ]

    if dry_run:
        return True, ["would undo:", *[f"  {f['status']} {f['path']}" for f in files]]

    ok, log = True, []
    for f in files:
        path, status = f["path"], f.get("status", "")[:1]
        if status == "A":  # the agent created it -> remove it
            try:
                os.remove(os.path.join(cwd, path))
                log.append(f"removed {path}")
            except OSError as e:
                ok = False
                log.append(f"could not remove {path}: {e}")
        else:  # modified or deleted -> restore from HEAD
            code, out = _git(["checkout", "HEAD", "--", path], cwd)
            if code == 0:
                log.append(f"reverted {path}")
            else:
                ok = False
                log.append(f"could not revert {path}: {out.strip()[:80]}")
    return ok, log
