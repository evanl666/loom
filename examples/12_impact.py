"""Impact analysis: which recorded runs does a prompt change affect?

    python examples/12_impact.py

Offline: record two runs, then ask "if I change the system prompt, which of my
recorded runs are touched?" -- without a single model call.
"""

import tempfile

from loom import Agent
from loom.impact import assess, report
from loom.providers import ModelResponse, ScriptedProvider


def build_agent(system: str) -> Agent:
    return Agent(
        model=ScriptedProvider(
            [ModelResponse(text="All good.", stop_reason="end_turn")] * 2
        ),
        system=system,
    )


with tempfile.TemporaryDirectory() as tmp:
    # Record a small corpus with today's prompt.
    old = build_agent("You are a support agent.")
    for i, q in enumerate(["Where is my order?", "Cancel my subscription."]):
        old.run(q).save(f"{tmp}/run{i}.loom.json")
    paths = [f"{tmp}/run0.loom.json", f"{tmp}/run1.loom.json"]

    print("=== same config: nothing affected ===")
    print(report(assess(paths, build_agent("You are a support agent."))))

    print()
    print("=== changed prompt: every run flagged, free ===")
    print(report(assess(paths, build_agent("You are a VERY TERSE support agent."))))
