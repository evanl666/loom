"""Trace memory: agents that learn from their own history.

Every Loom run leaves a complete trace. ``TraceMemory`` turns a directory of
them into recallable experience: before a run starts, the most similar past
runs are summarized into a context item, so the agent walks in knowing what
worked (and what failed) last time.

    memory = TraceMemory("runs/", auto_store=True)
    agent = Agent(model=..., tools=[...], memory=memory)
    agent.run("Migrate the staging database.")   # recalls similar past runs
    # auto_store=True saves each completed live run back into runs/

Recall is nondeterminism (the store changes over time), so it is recorded as a
``"memory"`` effect -- replays reproduce exactly what was recalled at the time.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from glob import glob

_WORD = re.compile(r"[A-Za-z0-9]{4,}")


def _words(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text)}


class TraceMemory:
    """Similarity recall over a directory of saved traces."""

    def __init__(self, directory: str, k: int = 3, auto_store: bool = False):
        self.directory = directory
        self.k = k
        self.auto_store = auto_store
        os.makedirs(directory, exist_ok=True)
        self._index: list[dict] = []
        self.refresh()

    def refresh(self) -> None:
        """Re-scan the directory. Called automatically after add()."""
        self._index = []
        for path in sorted(glob(os.path.join(self.directory, "*.loom.json"))):
            try:
                with open(path) as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            episodes = data.get("episodes") or [data.get("prompt", "")]
            output = data.get("output", "")
            self._index.append(
                {
                    "path": path,
                    "episodes": episodes,
                    "output": output,
                    "words": _words(" ".join(episodes) + " " + output),
                }
            )

    def add(self, run) -> str:
        """Store a finished run as experience. Returns the saved path."""
        blob = json.dumps(run.to_dict(), sort_keys=True)
        digest = hashlib.sha256(blob.encode()).hexdigest()[:12]
        path = os.path.join(self.directory, f"{digest}.loom.json")
        run.save(path)
        self.refresh()
        return path

    def recall(self, query: str) -> list[dict]:
        """Top-k most similar past runs, by word overlap with the query."""
        qw = _words(query)
        if not qw:
            return []
        scored = []
        for entry in self._index:
            overlap = len(qw & entry["words"])
            if overlap:
                scored.append((overlap / len(qw | entry["words"]), entry))
        scored.sort(key=lambda pair: -pair[0])
        return [e for _, e in scored[: self.k]]

    def recall_text(self, query: str) -> str:
        """A context-ready summary of similar past runs ("" when none)."""
        hits = self.recall(query)
        if not hits:
            return ""
        lines = ["Relevant experience from similar past runs:"]
        for i, h in enumerate(hits, 1):
            question = h["episodes"][0][:120]
            outcome = h["output"][:160]
            lines.append(f"{i}. Task: {question}")
            lines.append(f"   Outcome: {outcome}")
        return "\n".join(lines)
