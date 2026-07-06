"""The smallest useful Loom agent -- runs offline, no API key.

    python examples/01_hello.py
"""

from loom import Agent, tool
from loom.providers import ModelResponse, ScriptedProvider, ToolCall


@tool
def add(a: int, b: int) -> int:
    "Add two numbers."
    return a + b


# A deterministic offline "model": call `add`, then answer. Swap this line for
#   agent = Agent(model="claude-opus-4-8", tools=[add])
# to use a real model (needs ANTHROPIC_API_KEY).
provider = ScriptedProvider(
    [
        ModelResponse(
            tool_calls=[ToolCall("t1", "add", {"a": 2, "b": 3})], stop_reason="tool_use"
        ),
        ModelResponse(text="2 + 3 = 5.", stop_reason="end_turn"),
    ]
)

agent = Agent(model=provider, tools=[add])
run = agent.run("What is 2 + 3?")

print("output:", run.output)
print("\ntimeline:")
run.print_timeline()
