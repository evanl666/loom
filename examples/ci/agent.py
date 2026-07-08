"""The demo agent used by the Agent CI workflow -- offline, deterministic.

The Loom GitHub Action replays the traces in ``examples/ci/traces`` against
this module on every PR: change the system prompt below and the check fails
with an impact report showing exactly which recorded runs are touched.
"""

from loom import Agent, tool
from loom.providers import ModelResponse, RuleProvider, ToolCall


@tool
def lookup(city: str) -> str:
    "Look up a city."
    return f"{city}: ok"


def _wants_tool(messages):
    if not any(m["role"] == "tool" for m in messages):
        return ModelResponse(
            tool_calls=[ToolCall("t1", "lookup", {"city": "Berlin"})],
            stop_reason="tool_use",
        )
    return None


def _answer(messages):
    return ModelResponse(text="Berlin looks fine.", stop_reason="end_turn")


def build() -> Agent:
    return Agent(
        model=RuleProvider(rules=[_wants_tool, _answer]),
        tools=[lookup],
        system="You are a city inspector.",
    )
