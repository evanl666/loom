"""``loom migrate``: bring a trace to the current format version.

Old traces stay readable forever (loaders warn, never fail), but their effect
keys may predate the current key semantics -- which makes strict replay and
``loom impact`` report inputs-differ on runs that are actually fine. Migration
recomputes what changed and re-stamps the checksum:

    loom migrate old.loom.json --agent mypkg.agents:build   # harness traces
    loom migrate session.loom.json                          # proxy traces

Harness traces need the agent (keys are derived from the rebuilt context, so
the same-config rule applies). Proxy traces migrate without one -- their key
semantics never changed. ``loom migrate`` on a current-version trace just
re-stamps the checksum, which is also how you bless a deliberate hand-edit.
"""

from __future__ import annotations

import json

from .effect import Recorder, _key
from .trace import TRACE_VERSION, Run, trace_checksum


class _RekeyRecorder(Recorder):
    """Replays non-strictly while rewriting each entry's key from the
    recomputed payload -- the write-side twin of impact's key checker."""

    def __init__(self, log):
        super().__init__(log=log, replay_until=len(log), allow_live=False)
        self.rekeyed = 0

    def run(self, kind, payload, fn, encode=lambda x: x, decode=lambda x: x):
        if self._cursor < self.replay_until:
            entry = self.log[self._cursor]
            if entry.kind == kind and entry.key != "resumed":
                new_key = _key([kind, payload])
                if new_key != entry.key:
                    entry.key = new_key
                    self.rekeyed += 1
        return super().run(kind, payload, fn, encode=encode, decode=decode)


def migrate(path: str, agent=None, out: "str | None" = None) -> "tuple[int, str]":
    """Migrate one trace file. Returns (effects re-keyed, output path)."""
    import warnings

    out = out or path
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # we ARE the fix the warnings point at
        with open(path) as f:
            data = json.load(f)

        rekeyed = 0
        if data.get("recorded_via") != "proxy" and data.get("version", 1) < TRACE_VERSION:
            if agent is None:
                raise ValueError(
                    "harness traces need the recording agent to recompute keys: "
                    "loom migrate <trace> --agent module:attr"
                )
            run = Run.load(path, agent=agent)
            rec = _RekeyRecorder(run.log)
            agent.run(run.episodes, recorder=rec)  # non-strict walk, rewrites keys
            data["log"] = [e.to_dict() for e in run.log]
            rekeyed = rec.rekeyed

    data["version"] = TRACE_VERSION
    data["checksum"] = trace_checksum(data)
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    return rekeyed, out
