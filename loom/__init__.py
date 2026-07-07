"""Loom -- the agent harness you can read, replay, and rewind.

One primitive (the Effect boundary), five superpowers: reproducible replay,
fork-at-any-step, bisect, free CI testing, and exact cost accounting.

Quickstart:

    from loom import Agent, tool
    from loom.providers import ScriptedProvider  # offline; swap for a real model

    @tool
    def add(a: int, b: int) -> int:
        "Add two numbers."
        return a + b

    agent = Agent(model="claude-opus-4-8", tools=[add])
    run = agent.run("What is 2 + 2?")
    print(run.output)
"""

from .agent import Agent, SubagentTool
from .context import Context, Item
from .diff import StepDiff, TraceDiff, diff_logs
from .effect import EffectEntry, Recorder, ReplayExhausted, ReplayMismatch
from .providers.base import ModelProvider, ModelResponse, ToolCall
from .tools import Tool, tool
from .trace import Run, SweepResult

__version__ = "0.2.0"

__all__ = [
    "Agent",
    "Run",
    "SweepResult",
    "TraceDiff",
    "StepDiff",
    "diff_logs",
    "Context",
    "Item",
    "Recorder",
    "EffectEntry",
    "ReplayMismatch",
    "ReplayExhausted",
    "tool",
    "Tool",
    "SubagentTool",
    "ModelProvider",
    "ModelResponse",
    "ToolCall",
]
