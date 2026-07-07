"""The Effect boundary -- the soul of Loom.

Every nondeterministic action in the harness (model calls, tool calls, clock,
randomness, network) flows through a single chokepoint: ``Recorder.run``.

In RECORD mode it executes the effect and logs the result. In REPLAY mode it
returns the logged result without executing. Because control flow is otherwise
deterministic, this one primitive is what makes an agent run:

  * reproducible  -- replay a run with zero API calls,
  * forkable      -- rewind to any step, change context, and continue,
  * bisectable    -- walk the recorded turns to find where it went wrong,
  * testable      -- record once, replay in CI for free.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable


class ReplayMismatch(RuntimeError):
    """Raised when the replayed control flow diverges from the recorded log."""


class ReplayExhausted(RuntimeError):
    """Raised when a strict replay needs an effect past the end of the log."""


def _key(payload: Any) -> str:
    """Stable short hash of an effect's inputs, used to validate replays."""
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


@dataclass
class EffectEntry:
    """One recorded effect: an ordered, JSON-serializable unit of the trace."""

    seq: int
    kind: str  # "model" | "tool:<name>" | ...
    key: str  # hash of the inputs at record time
    result: Any  # JSON-serializable result payload
    depth: int = 0  # 0 = top-level agent, 1+ = nested subagent

    def to_dict(self) -> dict:
        return {
            "seq": self.seq,
            "kind": self.kind,
            "key": self.key,
            "result": self.result,
            "depth": self.depth,
        }

    @staticmethod
    def from_dict(d: dict) -> "EffectEntry":
        return EffectEntry(
            seq=d["seq"],
            kind=d["kind"],
            key=d["key"],
            result=d["result"],
            depth=d.get("depth", 0),
        )


class Recorder:
    """Serves effects from a log and/or records them live.

    A single ``replay_until`` cursor unifies all three modes:

      * ``record()``       -> replay_until=0                (log everything)
      * ``replay(log)``    -> replay_until=len(log)         (serve everything)
      * ``fork(log, at)``  -> replay_until=at, allow_live   (serve, then diverge)
    """

    def __init__(
        self,
        log: list[EffectEntry] | None = None,
        replay_until: int = 0,
        allow_live: bool = True,
    ):
        self.log: list[EffectEntry] = list(log or [])
        self.replay_until = replay_until
        self.allow_live = allow_live
        self._cursor = 0
        self.depth = 0  # current nesting depth; agents set this before recording
        self.journal = None  # optional write-ahead Journal; set by the agent

    # -- constructors -----------------------------------------------------

    @classmethod
    def record(cls) -> "Recorder":
        return cls(log=None, replay_until=0, allow_live=True)

    @classmethod
    def replay(cls, log: list[EffectEntry]) -> "Recorder":
        return cls(log=log, replay_until=len(log), allow_live=False)

    @classmethod
    def fork(cls, log: list[EffectEntry], at: int) -> "Recorder":
        return cls(log=log, replay_until=at, allow_live=True)

    # -- the chokepoint ---------------------------------------------------

    def run(
        self,
        kind: str,
        payload: Any,
        fn: Callable[[], Any],
        encode: Callable[[Any], Any] = lambda x: x,
        decode: Callable[[Any], Any] = lambda x: x,
    ) -> Any:
        """Execute (or replay) a single effect.

        ``payload`` is the effect's inputs (hashed for validation). ``fn`` is the
        thunk that actually performs the side effect -- it is never called when
        the effect is served from the log.
        """
        seq = self._cursor

        if seq < self.replay_until:
            entry = self.log[seq]
            if entry.kind != kind:
                raise ReplayMismatch(
                    f"at seq {seq}: recorded kind {entry.kind!r} but replay wants {kind!r}"
                )
            self._cursor += 1
            return decode(entry.result)

        if not self.allow_live:
            raise ReplayExhausted(
                f"replay log exhausted at seq {seq} (kind={kind!r}); the run diverged"
            )

        result = fn()
        entry = EffectEntry(
            seq=seq, kind=kind, key=_key([kind, payload]), result=encode(result), depth=self.depth
        )
        # Forking overwrites the tail of the original log from this point on.
        del self.log[seq:]
        self.log.append(entry)
        if self.journal is not None:
            self.journal.append(entry)  # write-ahead: on disk before we move on
        self._cursor += 1
        return result

    # -- introspection ----------------------------------------------------

    @property
    def cursor(self) -> int:
        return self._cursor

    def model_seqs(self, depth: "int | None" = 0) -> list[int]:
        """Seq indices of model calls -- the turn boundaries.

        By default only top-level (depth 0) turns, which is what forking rewinds
        to. Pass ``depth=None`` for every model call at any nesting level.
        """
        return [
            e.seq
            for e in self.log
            if e.kind == "model" and (depth is None or e.depth == depth)
        ]
