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

from .action import Action, Observation, PolicyDecision, ReplayPoint, StateDiff, actions
from .agent import Agent, HumanInputRequired, HumanTool, SubagentTool, ask_human
from .ambient import now, random
from .context import Context, Item
from .diff import StepDiff, TraceDiff, diff_logs
from .effect import EffectEntry, Recorder, ReplayExhausted, ReplayMismatch
from .export import trace_to_html
from .cache import EffectCache
from .health import Finding, HealthReport
from .impact import Impact
from .journal import Journal
from .memory import TraceMemory
from .policy import Policy
from .structured import OutputInvalid, parse_as, schema_for
from .testing import verify_replay, verify_trace
from .providers.base import ModelProvider, ModelResponse, ToolCall
from .tools import Tool, tool
from .trace import Run, SweepResult
from . import packs  # noqa: F401  (imports & registers the built-in Coding Pack)
from .packs import coding  # noqa: F401

__version__ = "0.31.1"

__all__ = [
    "Agent",
    "Action",
    "Observation",
    "StateDiff",
    "PolicyDecision",
    "ReplayPoint",
    "actions",
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
    "HumanTool",
    "HumanInputRequired",
    "ask_human",
    "trace_to_html",
    "HealthReport",
    "Finding",
    "Journal",
    "Policy",
    "EffectCache",
    "TraceMemory",
    "OutputInvalid",
    "parse_as",
    "schema_for",
    "Impact",
    "verify_trace",
    "verify_replay",
    "now",
    "random",
    "ModelProvider",
    "ModelResponse",
    "ToolCall",
]
