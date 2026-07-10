"""The Coding Pack -- Loom's built-in domain pack for coding agents.

Everything that used to be baked into the core as "the coding story" -- file
edits as state changes, ``git`` undo, shell/secret risk -- is now expressed
through the generic Pack interface, so coding is simply the *first* pack
rather than a special case. Browser, SQL, and Support packs plug in exactly
the same way (``loom.packs.Pack``); this one just ships built-in and is
registered on import.

What it teaches Loom:

  owns        file / shell / git / read actions (by risk class or tool name)
  state_diff  a file edit -> "wrote src/x.py" (from the action's own input)
  undo        restore that file to HEAD (git checkout), or remove a created
              file; a destructive shell action gets a manual-review plan
"""

from __future__ import annotations

from fnmatch import fnmatchcase as fnmatch

from ..action import Action, StateDiff
from . import Pack, RestorePlan, UndoPlan, register

# Risk categories that are unmistakably coding-infrastructure.
_CODING_RISK = {"secret-read", "code-exec", "fs-destructive", "fs-write"}

# Tool-name shapes for coding actions that carry no risky arguments (a plain
# Read, an ls, a git status) and so wouldn't be caught by risk alone.
_CODING_NAMES = [
    "Read*", "Write*", "Edit*", "MultiEdit*", "NotebookEdit*", "Bash*", "Shell*",
    "sh", "zsh", "Glob*", "Grep*", "ls", "cat", "git*", "apply_patch*",
    "str_replace*", "view*", "create_file*", "*write_file*", "run_command*",
]

# Input keys that name the file an edit touched.
_PATH_KEYS = ("file_path", "path", "filename", "file", "notebook_path")


def _path_of(tool_input) -> str:
    if isinstance(tool_input, dict):
        for k in _PATH_KEYS:
            v = tool_input.get(k)
            if isinstance(v, str) and v:
                return v
    return ""


class CodingPack(Pack):
    name = "coding"

    def owns(self, action: Action) -> bool:
        if action.type != "call":
            return False
        if action.risk in _CODING_RISK:
            return True
        return any(fnmatch(action.tool, g) for g in _CODING_NAMES)

    def debugger_panels(self, action: Action, trace: dict) -> "list[dict]":
        if action.type != "call":
            return []
        path = _path_of(action.input)
        content = (action.input or {}).get("content") or (action.input or {}).get("new_str")
        if path and content is not None:
            return [{"title": f"📄 file · {path}", "code": str(content)[:6000]}]
        cmd = self._command(action)
        if cmd:
            return [{"title": "▶ shell command", "code": cmd}]
        return []

    def state_diff(self, action: Action, trace: dict) -> "StateDiff | None":
        # Only actions that WRITE change the world; derive the file from the
        # action's own input, so the diff is per-step, not run-level.
        caps = set(action.capabilities)
        if not (caps & {"write", "destructive"}):
            return None
        path = _path_of(action.input)
        if path:
            status = self._recorded_status(trace, path)
            verb = {"A": "created", "D": "deleted"}.get(status, "wrote")
            return StateDiff("file", f"{verb} {path}", detail={"path": path, "status": status})
        if "destructive" in caps:  # a destructive shell action with no file arg
            return StateDiff("file", f"destructive shell: {self._command(action)[:60]}",
                             detail={"command": self._command(action)})
        return None

    def undo(self, action: Action, trace: dict) -> "UndoPlan | None":
        caps = set(action.capabilities)
        path = _path_of(action.input)
        # A destructive action with no file target (rm -rf, force-push) can't be
        # auto-undone from a snapshot -- flag it for manual review, don't pretend.
        if "destructive" in caps and not path:
            return UndoPlan(
                "noop",
                "destructive shell action -- no automatic undo; review manually "
                "(`loom undo` reverts recorded file changes)",
                reversible=False,
            )
        if not (caps & {"write", "destructive"}):
            return None  # a read/exec that wrote nothing -- nothing to undo
        if not path:
            return None
        status = self._recorded_status(trace, path)
        if status == "A":  # the agent created it -> removing restores prior state
            return UndoPlan("revert", f"remove created file {path}", [f"rm {path}"])
        return UndoPlan("revert", f"restore {path} to HEAD",
                        [f"git checkout HEAD -- {path}"])

    @staticmethod
    def _command(action: Action) -> str:
        inp = action.input
        return inp.get("command", "") if isinstance(inp, dict) else ""

    def snapshot(self, trace: dict) -> "dict | None":
        # The world a coding run started from IS pinned in the trace: the git
        # commit it recorded against (plus whether the tree was dirty).
        g = (trace.get("workspace") or {}).get("git") or {}
        if not g.get("commit"):
            return None
        return {"commit": g["commit"], "branch": g.get("branch", ""),
                "dirty": bool(g.get("dirty"))}

    def restore(self, snapshot: "dict | None") -> "RestorePlan | None":
        if not snapshot or not snapshot.get("commit"):
            return RestorePlan(
                "manual", "this trace didn't record a git commit -- check out the "
                          "source state the run assumed before replaying", executable=False)
        commit = snapshot["commit"][:12]
        cmds = [f"git checkout {commit}"]
        note = f"check out the commit the run recorded against ({commit})"
        if snapshot.get("dirty"):
            # A dirty record can't be reproduced from git alone -- be honest.
            return RestorePlan("git", note + " -- WARNING: recorded on a dirty tree, "
                               "so uncommitted changes can't be reproduced from git",
                               cmds, executable=False)
        return RestorePlan("git", note, cmds, executable=True)

    def safe_runtime(self) -> str:
        return ("run the agent in a container (loom record --container) or the macOS "
                "sandbox (loom record --sandbox) so its file writes and network go "
                "through the proxy and can't escape the workspace")

    @staticmethod
    def _recorded_status(trace: dict, path: str) -> str:
        """The git status the workspace recorded for ``path`` (''/'A'/'M'/'D')."""
        changes = (trace.get("workspace") or {}).get("changes") or {}
        for f in changes.get("files", []):
            if f.get("path") == path:
                return (f.get("status") or "")[:1]
        return ""


# Ship built-in: registered on import so the coding story keeps working while
# being, mechanically, just another pack.
register(CodingPack())
