"""Subagents: a lead agent delegates to a researcher. Offline, no API key.

    python examples/04_subagents.py

The researcher runs with its own isolated context; its steps nest into the same
trace, so the whole thing still replays deterministically.
"""

from loom import Agent, tool
from loom.providers import ModelResponse, ScriptedProvider, ToolCall


@tool
def search(q: str) -> str:
    "Search a tiny knowledge base."
    return f"'{q}' -> Loom routes every effect through one boundary."


# Child: search, then summarize.
researcher = Agent(
    model=ScriptedProvider(
        [
            ModelResponse(
                tool_calls=[ToolCall("s1", "search", {"q": "loom effect boundary"})],
                stop_reason="tool_use",
            ),
            ModelResponse(
                text="Loom records every model/tool call at one chokepoint.",
                stop_reason="end_turn",
            ),
        ]
    ),
    tools=[search],
    name="researcher",
)

# Parent: delegate to the researcher, then answer.
lead = Agent(
    model=ScriptedProvider(
        [
            ModelResponse(
                tool_calls=[ToolCall("d1", "researcher", {"task": "how does loom work"})],
                stop_reason="tool_use",
            ),
            ModelResponse(
                text="Loom works by recording every effect at a single boundary, "
                "which makes runs replayable.",
                stop_reason="end_turn",
            ),
        ]
    ),
    tools=[researcher.as_tool()],
    name="lead",
)

run = lead.run("How does Loom work?")

print("answer:", run.output)
print(f"\ntop-level turns: {run.num_turns}   |   total model calls: {run.num_model_calls}")
print("\nnested timeline (indentation = subagent depth):")
run.print_timeline()

print("\nparent context stayed isolated -- it only saw the delegated result:")
for p in run.context.provenance():
    print(f"  {p['source']}")

print("\nreplay through the subagent (zero live calls):", run.replay().output == run.output)
