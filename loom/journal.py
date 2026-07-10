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

The journal is **two-phase**: before a live effect executes, an ``intent``
line is flushed; once the result is on disk, the matching ``effect`` line
supersedes it. So an intent with no effect after it marks the exact window
where the process died *between starting a side effect and persisting its
result* -- the one case where "did it run?" is genuinely unknowable from the
log. Recovery refuses to silently re-execute a tool in that state (see
``Run.recover``'s ``on_unfinished``); harness-internal effects (model calls,
memory recalls...) are safe to retry and recover silently.
"""

from __future__ import annotations

import json

from .effect import EffectEntry


class UnfinishedEffect(RuntimeError):
    """A journal ends in an intent whose side effect may or may not have run."""


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

    def intent(self, seq: int, kind: str, key: str, depth: int = 0) -> None:
        """Declare an effect is ABOUT to execute (phase one of two).

        Flushed before ``fn()`` runs, so if the process dies mid-effect the
        journal shows exactly which side effect was in flight. A torn intent
        line means the crash happened before execution started -- safe.
        """
        self._f.write(
            json.dumps({"type": "intent", "seq": seq, "kind": kind, "key": key, "depth": depth})
            + "\n"
        )
        self._f.flush()

    def append(self, entry: EffectEntry) -> None:
        """Persist one freshly recorded effect. Flushed so a crash loses nothing."""
        self._f.write(json.dumps({"type": "effect", **entry.to_dict()}) + "\n")
        self._f.flush()

    def close(self) -> None:
        """Release the file handle. Idempotent; safe if start() never ran.

        Every write already flushes, so closing changes nothing on disk -- it
        just stops leaking the descriptor once the run is over.
        """
        if self._f is not None:
            self._f.close()
            self._f = None

    @staticmethod
    def read_full(path: str) -> "tuple[dict, list[EffectEntry], list[dict]]":
        """Parse a journal, tolerating a torn final line (crash mid-write).

        Returns (header, effects, unfinished intents) -- intents that never got
        their effect line: side effects that started but whose outcome the
        journal never saw.
        """
        header: dict = {}
        entries: list[EffectEntry] = []
        pending: dict[int, dict] = {}
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
                    pending.clear()  # a rewrite (ask/resume/recover) resets the run
                elif d.get("type") == "intent":
                    pending[d.get("seq", -1)] = d
                elif d.get("type") == "effect":
                    d.pop("type")
                    entry = EffectEntry.from_dict(d)
                    pending.pop(entry.seq, None)  # phase two arrived: intent fulfilled
                    entries.append(entry)
        return header, entries, sorted(pending.values(), key=lambda d: d.get("seq", -1))

    @staticmethod
    def read(path: str) -> "tuple[dict, list[EffectEntry]]":
        """Parse a journal: (header, effects). See ``read_full`` for intents."""
        header, entries, _ = Journal.read_full(path)
        return header, entries
