"""Context-rot detection and self-healing. Offline, no API key.

    python examples/08_checkup_heal.py

An oversized junk tool result corrupts the agent's final answer. checkup()
finds the suspect, heal() redacts it in a fork, re-runs only the final turn,
and returns the fixed branch -- diagnosis to verified fix, automatically.
"""

from loom import Agent, tool
from loom.providers import ModelResponse, RuleProvider, ToolCall

POISON = "POISONMARKER junkdata noisepayload telemetry " * 60  # ~600 tokens of rot


@tool
def fetch_context() -> str:
    "Fetch background data for the question."
    return POISON  # a real agent might get this from a bloated RAG hit


# A deterministic "model": corrupted whenever the junk is in its context.
def wants_tool(messages):
    if not any(m["role"] == "tool" for m in messages):
        return ModelResponse(
            tool_calls=[ToolCall("t1", "fetch_context", {})], stop_reason="tool_use"
        )
    return None


def poisoned(messages):
    if any("POISONMARKER" in str(m.get("content", "")) for m in messages):
        return ModelResponse(text="ERROR: reasoning corrupted by junk", stop_reason="end_turn")
    return None


def clean(messages):
    return ModelResponse(text="The answer is 42.", stop_reason="end_turn")


agent = Agent(model=RuleProvider(rules=[wants_tool, poisoned, clean]), tools=[fetch_context])

print("=" * 64)
print("1. A run goes wrong")
print("=" * 64)
run = agent.run("What is the answer?")
print("output:", run.output)

print()
print("=" * 64)
print("2. CHECKUP: what does the context look like?")
print("=" * 64)
report = run.checkup()
print(report.summary())

print()
print("=" * 64)
print("3. HEAL: test each suspected repair in a fork, keep the first fix")
print("=" * 64)
healed = run.heal(check=lambda text: "ERROR" not in text)
print("healed output:", healed.output)
print("fixed by     :", healed.healed_by)
print("original run untouched:", run.output)
