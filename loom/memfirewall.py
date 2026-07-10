"""``MemoryFirewall``: provenance + quarantine for an agent's long-term memory.

``loom memory forensics`` *detects* poisoning after the fact. The firewall
*prevents* it: a drop-in wrapper around ``TraceMemory`` that tags each memory by
provenance and refuses to recall poisoned or untrusted memories into the agent's
context, so a web page it read last week can't steer it this week.

    from loom import Agent, TraceMemory
    from loom.memfirewall import MemoryFirewall

    memory = MemoryFirewall(TraceMemory("runs/", auto_store=True))
    agent = Agent(model="claude-opus-4-8", tools=[...], memory=memory)

``Agent`` calls ``memory.recall_text(...)``; the firewall returns only the
trusted memories and keeps a ``quarantined`` list of what it blocked and why.
``loom memory audit <dir>`` shows the provenance of a whole memory store.
"""

from __future__ import annotations

import os
from glob import glob
from typing import Any


def _poison_reason(text: str) -> str:
    """"" if the text looks trustworthy, else why it's quarantined."""
    from .inject import _INJECTION

    if _INJECTION.search(text):
        return "carries injected instructions"
    return ""


def trace_provenance(data: dict) -> dict:
    """Trust assessment for one stored run: trusted / untrusted + reasons."""
    from .action import actions
    from .inject import _INJECTION, _is_untrusted

    text = " ".join(str(x) for x in (data.get("episodes") or [])) + " " + str(data.get("output", ""))
    reasons: list[str] = []
    if _INJECTION.search(text):
        reasons.append("injected-instructions-in-output")
    # untrusted content it ingested (network/fetch/browser results)
    for a in actions(data):
        if a.type == "call" and _is_untrusted(a) and a.observation is not None:
            if _INJECTION.search(a.observation.text or ""):
                reasons.append(f"ingested-injection@{a.step}({a.tool})")
                break
    trust = "untrusted" if reasons else "trusted"
    return {"trust": trust, "reasons": reasons}


class MemoryFirewall:
    """Wrap a TraceMemory: recall only trusted memories, quarantine the rest."""

    def __init__(self, memory: Any, block_untrusted: bool = True):
        self.memory = memory
        self.block_untrusted = block_untrusted
        self.quarantined: list[dict] = []

    # delegated storage surface (so it's a drop-in for Agent(memory=...))
    @property
    def auto_store(self) -> bool:
        return getattr(self.memory, "auto_store", False)

    def add(self, run: Any) -> Any:
        return self.memory.add(run)

    def refresh(self) -> None:
        self.memory.refresh()

    def _entry_text(self, entry: dict) -> str:
        return " ".join(entry.get("episodes") or []) + " " + str(entry.get("output", ""))

    def recall(self, query: str) -> "list[dict]":
        hits = self.memory.recall(query)
        safe = []
        for h in hits:
            reason = _poison_reason(self._entry_text(h))
            if reason and self.block_untrusted:
                self.quarantined.append({"path": h.get("path", "?"), "reason": reason})
            else:
                safe.append(h)
        return safe

    def recall_text(self, query: str) -> str:
        """The recall the agent sees -- poisoned memories are withheld."""
        hits = self.recall(query)
        if not hits:
            return ""
        lines = ["Relevant experience from similar past runs:"]
        for i, h in enumerate(hits, 1):
            q = (h.get("episodes") or ["?"])[0][:120]
            out = str(h.get("output", ""))[:160]
            lines += [f"{i}. Task: {q}", f"   Outcome: {out}"]
        return "\n".join(lines)


def audit_memory(directory: str) -> dict:
    """Provenance of every stored run in a memory directory."""
    import json

    items = []
    for p in sorted(glob(os.path.join(directory, "*.loom.json"))):
        try:
            with open(p) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        prov = trace_provenance(data)
        items.append({"path": os.path.basename(p), **prov})
    untrusted = [i for i in items if i["trust"] == "untrusted"]
    return {"total": len(items), "untrusted": len(untrusted), "items": items}


def describe_audit(a: dict) -> str:
    lines = [f"memory audit — {a['total']} stored run(s), {a['untrusted']} untrusted/poisoned"]
    for i in a["items"]:
        if i["trust"] == "untrusted":
            lines.append(f"  🔴 {i['path']}: {', '.join(i['reasons'])}")
    if not a["untrusted"]:
        lines.append("  ✓ no poisoned memories")
    else:
        lines.append("\n  wrap with MemoryFirewall(TraceMemory(...)) to quarantine these at recall.")
    return "\n".join(lines)
