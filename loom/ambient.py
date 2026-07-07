"""Ambient effects: the clock and randomness, through the boundary.

    from loom import now, random

Called at harness level while a run is active, these are recorded effects --
replays serve the recorded value, so time- and randomness-dependent harness
logic is deterministic like everything else. ``Agent(clock=True)`` uses this
to tell the model today's date in a replay-stable way.

Called inside a tool, they return REAL values on purpose: a tool executes
either live (fresh time is correct) or not at all (replay serves the tool's
recorded result, which already embeds whatever time it saw). Recording a
nested effect there would corrupt the log's replay order. Outside any run,
they fall back to the standard library.
"""

from __future__ import annotations

import contextvars
import random as _stdlib_random
import time as _stdlib_time

_active: contextvars.ContextVar = contextvars.ContextVar("loom_recorder", default=None)


def _activate(rec) -> contextvars.Token:
    """Make ``rec`` the ambient recorder for this context. Returns a reset token."""
    return _active.set(rec)


def _deactivate(token: contextvars.Token) -> None:
    _active.reset(token)


def _recorder():
    rec = _active.get()
    if rec is None or rec.executing:
        return None  # outside a run, or inside a tool: use real values
    return rec


def now() -> float:
    """The current UNIX timestamp -- recorded at harness level, real elsewhere."""
    rec = _recorder()
    if rec is None:
        return _stdlib_time.time()
    return rec.run("time", {}, _stdlib_time.time)


def random() -> float:
    """A float in [0, 1) -- recorded at harness level, real elsewhere."""
    rec = _recorder()
    if rec is None:
        return _stdlib_random.random()
    return rec.run("random", {}, _stdlib_random.random)
