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
from dataclasses import dataclass, field
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
    # Optional per-call provenance used for multi-agent attribution: for native
    # runs the agent name; for proxied (wire) runs a fingerprint of the request
    # (system-prompt hash, tool set, model) so a sub-agent can be recovered even
    # when the framework never told us about it. Absent on old/simple traces.
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "seq": self.seq,
            "kind": self.kind,
            "key": self.key,
            "result": self.result,
            "depth": self.depth,
        }
        if self.meta:
            d["meta"] = self.meta
        return d

    @staticmethod
    def from_dict(d: dict) -> "EffectEntry":
        # Tolerant of hand-edited / corrupted traces: a missing field degrades
        # (breaks strict-replay matching at worst) rather than crashing every
        # analyzer that loads the trace through this chokepoint.
        return EffectEntry(
            seq=d.get("seq", 0),
            kind=d.get("kind", ""),
            key=d.get("key", ""),
            result=d.get("result"),
            depth=d.get("depth", 0),
            meta=d.get("meta") or {},
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
        self.agent_name = ""  # name of the agent currently recording (multi-agent attribution)
        self.journal = None  # optional write-ahead Journal; set by the agent
        self.cache = None  # optional EffectCache; set by the agent
        self.executing = False  # True while an effect's thunk runs (see loom/ambient.py)
        self.strict = False  # replay-time key verification; set by Recorder.replay

    # -- constructors -----------------------------------------------------

    @classmethod
    def record(cls) -> "Recorder":
        return cls(log=None, replay_until=0, allow_live=True)

    @classmethod
    def replay(cls, log: list[EffectEntry], strict: bool = True) -> "Recorder":
        """Serve every effect from the log; never go live.

        ``strict`` (the default) also recomputes each effect's input key and
        raises ``ReplayMismatch`` if it differs from the recording -- proof
        that the agent as configured NOW is equivalent to the one that
        recorded, not merely that the old log can be walked to the end. Pass
        ``strict=False`` to inspect a trace with a changed configuration.
        """
        rec = cls(log=log, replay_until=len(log), allow_live=False)
        rec.strict = strict
        return rec

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
            # "resumed" marks a human answer injected by Run.resume(): its key
            # is a sentinel, not an input hash -- nothing to verify against.
            if self.strict and entry.key != "resumed":
                computed = _key([kind, payload])
                if computed != entry.key:
                    raise ReplayMismatch(
                        f"at seq {seq} ({kind}): inputs differ from the recording "
                        f"(recorded key {entry.key[:12]}, current {computed[:12]}). "
                        f"The agent's configuration is not equivalent to the one that "
                        f"recorded this trace -- `loom impact` shows a per-run report; "
                        f"replay(strict=False) walks the old log anyway."
                    )
            self._cursor += 1
            return decode(entry.result)

        if not self.allow_live:
            raise ReplayExhausted(
                f"replay log exhausted at seq {seq} (kind={kind!r}); the run diverged"
            )

        key = _key([kind, payload])

        # Effect cache: identical inputs reuse an earlier run's encoded result.
        cached = None
        if self.cache is not None and self.cache.wants(kind):
            cached = self.cache.get(key)

        if cached is not None:
            encoded, result = cached, decode(cached)
        else:
            if self.journal is not None:
                # Phase one of the two-phase journal: declare the effect before
                # it runs, so a crash mid-effect is distinguishable from a
                # crash between effects (cache hits execute nothing -- no intent).
                self.journal.intent(seq, kind, key, self.depth)
            self.executing = True
            try:
                result = fn()
            finally:
                self.executing = False
            encoded = encode(result)
            if self.cache is not None and self.cache.wants(kind):
                self.cache.put(key, encoded)

        meta = {"agent": self.agent_name} if self.agent_name else {}
        entry = EffectEntry(seq=seq, kind=kind, key=key, result=encoded, depth=self.depth, meta=meta)
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

    def peek_kind(self) -> "str | None":
        """Kind of the next entry to be replayed, or None outside the replay region."""
        if self._cursor < self.replay_until:
            return self.log[self._cursor].kind
        return None

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
