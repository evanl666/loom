"""The Remote-Agent Pack -- teach Loom about calls to remote (HTTP/gRPC) agents.

A ``RemoteAgent`` call is a network hop to code you don't control. This pack
lets the debugger and analyzers treat it as what it is: a network + remote_agent
action whose effect (whatever the remote did) can't be reversed from here.
"""

from __future__ import annotations

from fnmatch import fnmatchcase as fnmatch

from ..action import Action, StateDiff
from . import Pack, UndoPlan, register

_REMOTE_NAMES = ["remote_*", "*_remote", "call_remote*", "delegate_remote*"]


def _is_remote(action: Action) -> bool:
    if "remote_agent" in (action.capabilities or []):
        return True
    return any(fnmatch(action.tool, g) for g in _REMOTE_NAMES)


class RemoteAgentPack(Pack):
    name = "remote"

    def owns(self, action: Action) -> bool:
        return action.type == "call" and _is_remote(action)

    def capabilities(self, name: str, tool_input) -> "set[str]":
        if any(fnmatch(name, g) for g in _REMOTE_NAMES):
            return {"network", "remote_agent"}
        return set()

    def state_diff(self, action: Action, trace: dict) -> "StateDiff | None":
        target = action.tool.replace("remote_", "") or "remote agent"
        return StateDiff("remote", f"called remote agent {target}",
                         detail={"tool": action.tool})

    def undo(self, action: Action, trace: dict) -> "UndoPlan | None":
        return UndoPlan(
            "noop",
            "a remote agent call can't be undone from here -- whatever the remote "
            "did happened on its side; compensate on the remote if needed",
            reversible=False)

    def debugger_panels(self, action: Action, trace: dict) -> "list[dict]":
        ep = trace.get("endpoint") or ""
        transport = trace.get("transport") or "http"
        text = f"transport: {transport}" + (f"\nendpoint: {ep}" if ep else "")
        prompt = (action.input or {}).get("prompt") if isinstance(action.input, dict) else None
        if prompt:
            text += f"\nsent: {str(prompt)[:2000]}"
        return [{"title": "🛰 remote agent call", "text": text}]

    def safe_runtime(self) -> str:
        return ("point the remote agent at a staging/sandbox endpoint (or a mock) "
                "while debugging, so replays and forks don't hit the real remote")

    def replay_hint(self, action: Action) -> str:
        return "the recorded response serves this call on replay -- no network needed"


register(RemoteAgentPack())
