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

    def to_dict(self) -> dict:
        return {"seq": self.seq, "kind": self.kind, "key": self.key, "result": self.result}

    @staticmethod
    def from_dict(d: dict) -> "EffectEntry":
        return EffectEntry(seq=d["seq"], kind=d["kind"], key=d["key"], result=d["result"])


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
        entry = EffectEntry(seq=seq, kind=kind, key=_key([kind, payload]), result=encode(result))
        # Forking overwrites the tail of the original log from this point on.
        del self.log[seq:]
        self.log.append(entry)
        self._cursor += 1
        return result

    # -- introspection ----------------------------------------------------

    @property
    def cursor(self) -> int:
        return self._cursor

    def model_seqs(self) -> list[int]:
        """Seq indices of every model call -- i.e. the turn boundaries."""
        return [e.seq for e in self.log if e.kind == "model"]
