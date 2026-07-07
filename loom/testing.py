"""Trace verification: regression testing for agents.

``loom test fixtures/`` checks every saved trace for internal consistency --
the cheapest possible CI gate for agent behavior. For full behavioral replays
(which need your live tools), call ``verify_replay`` from your own test suite:

    def test_support_agent_fixtures():
        for path in glob("fixtures/*.loom.json"):
            verify_replay(path, agent=build_agent())   # zero API calls
"""

from __future__ import annotations

import json

from .providers.base import ModelResponse


def verify_trace(path: str) -> list[str]:
    """Structural checks on a saved trace. Returns problems (empty = OK)."""
    problems: list[str] = []
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return [f"unreadable: {e}"]

    log = data.get("log")
    if not isinstance(log, list):
        return ["missing log"]

    for i, e in enumerate(log):
        if e.get("seq") != i:
            problems.append(f"log entry {i} has seq {e.get('seq')} (expected {i})")
            break

    if not data.get("paused"):
        model_entries = [e for e in log if e.get("kind") == "model"]
        if model_entries:
            final = ModelResponse.from_dict(model_entries[-1]["result"]).text
            if final != data.get("output", ""):
                problems.append("stored output does not match the final model text")

    if not data.get("episodes") and not data.get("prompt"):
        problems.append("missing episodes/prompt")
    return problems


def verify_replay(path: str, agent) -> None:
    """Replay a trace against your agent and assert the output matches.

    Raises AssertionError on mismatch and ReplayMismatch/ReplayExhausted if the
    agent's control flow no longer matches the recording -- both mean behavior
    changed. Costs zero API calls.
    """
    from .trace import Run

    original = Run.load(path, agent=agent)
    replayed = original.replay()
    assert replayed.output == original.output, (
        f"replayed output diverged for {path}:\n"
        f"  stored:   {original.output!r}\n"
        f"  replayed: {replayed.output!r}"
    )
