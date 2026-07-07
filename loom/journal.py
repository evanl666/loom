"""Write-ahead journaling: crash-safe agent runs.

With ``Agent(journal="task.jsonl")``, every effect is appended to disk the
moment it is recorded -- one JSON line per effect, flushed immediately. If the
process dies mid-run (crash, kill, power loss), nothing paid for is lost:

    run = Run.recover("task.jsonl", agent=agent)

replays the journaled prefix for free and continues live from the exact crash
point. Model calls and tool side effects that already happened are never
re-executed -- the same guarantee replay gives, extended across process death.

Recovery is idempotent: recovering a journal of a run that actually finished
just replays it to the same result with zero live calls.
"""

from __future__ import annotations

import json

from .effect import EffectEntry


class Journal:
    """An append-only JSONL sink for one run: a header line, then effects."""

    def __init__(self, path: str):
        self.path = path
        self._f = None

    def start(self, header: dict, prefix: list[EffectEntry]) -> None:
        """(Re)write the journal: the header plus any already-committed prefix.

        Called at run start. For continued runs (ask/resume/recover) the
        replayed prefix is written up front so the file is always a complete,
        self-contained record of the run in progress.
        """
        self._f = open(self.path, "w")
        self._f.write(json.dumps({"type": "header", **header}) + "\n")
        for e in prefix:
            self._f.write(json.dumps({"type": "effect", **e.to_dict()}) + "\n")
        self._f.flush()

    def append(self, entry: EffectEntry) -> None:
        """Persist one freshly recorded effect. Flushed so a crash loses nothing."""
        self._f.write(json.dumps({"type": "effect", **entry.to_dict()}) + "\n")
        self._f.flush()

    @staticmethod
    def read(path: str) -> "tuple[dict, list[EffectEntry]]":
        """Parse a journal, tolerating a torn final line (crash mid-write)."""
        header: dict = {}
        entries: list[EffectEntry] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    break  # torn tail from a crash -- everything before it is good
                if d.get("type") == "header":
                    header = d
                elif d.get("type") == "effect":
                    d.pop("type")
                    entries.append(EffectEntry.from_dict(d))
        return header, entries
