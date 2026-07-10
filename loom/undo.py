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


def _in_scope(path: str, only: str) -> bool:
    """Path-segment-aware prefix match: --only src must not match src2/x."""
    if not only:
        return True
    only = only.rstrip("/")
    return path == only or path.startswith(only + "/")


def _contained(cwd: str, path: str) -> bool:
    """Does ``path`` resolve to somewhere INSIDE ``cwd``? A recorded path is
    always repo-relative, but a hostile/hand-crafted trace could carry an
    absolute path or ``../..`` -- undo runs os.remove, so anything that escapes
    the tree must never be touched."""
    if not path or os.path.isabs(path):
        return False
    root = os.path.realpath(cwd)
    target = os.path.realpath(os.path.join(root, path))
    return target == root or target.startswith(root + os.sep)


def plan_undo(data: dict, only: str = "") -> "list[dict]":
    """The files loom undo would touch (the agent's own, in scope)."""
    changes = (data.get("workspace") or {}).get("changes") or {}
    return [
        f for f in changes.get("files", [])
        if not f.get("pre_existing")  # leave what was already dirty alone
        and _in_scope(f.get("path", ""), only)
    ]


def undo(data: dict, cwd: str, only: str = "", dry_run: bool = False,
         force: bool = False) -> "tuple[bool, list[str]]":
    """Revert the agent's changes. Returns (ok, log lines)."""
    from .workspace import _file_sha

    files = plan_undo(data, only=only)
    escaping = [f for f in files if not _contained(cwd, f.get("path", ""))]
    files = [f for f in files if _contained(cwd, f.get("path", ""))]
    if not files:
        msg = "nothing to undo (no agent file changes recorded)"
        if escaping:
            return False, [f"refused: {len(escaping)} recorded path(s) resolve outside "
                           f"the working tree and were skipped"]
        return True, [msg]

    # A file that changed again since the recording no longer holds what the
    # agent left -- reverting it would clobber that newer work.
    moved = [f for f in files if f.get("sha") and _file_sha(cwd, f["path"]) != f["sha"]]
    if moved and not force:
        return False, [
            "these files changed since the recording, so undo would clobber newer work "
            "(pass --force to revert anyway):",
            *[f"  {f['path']}" for f in moved],
        ]

    skipped = [f"  skipped (outside the tree): {f.get('path', '')}" for f in escaping]
    if dry_run:
        return True, ["would undo:", *[f"  {f['status']} {f['path']}" for f in files], *skipped]

    ok, log = True, list(skipped)
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
