"""Policy at the boundary: deny / confirm / dry-run / budget. Offline, no API key.

    python examples/10_policy.py

One Policy object gates every tool call at the single chokepoint -- see what an
agent WOULD do (dry run), block the dangerous parts, and require human approval
for the rest.
"""

from loom import Agent, Policy, tool
from loom.providers import ModelResponse, ScriptedProvider, ToolCall


@tool
def read_orders() -> str:
    "Read the orders table."
    return "42 stale orders found"


@tool
def delete_orders() -> str:
    "Delete stale orders. DESTRUCTIVE."
    return "deleted 42 orders"


def provider():
    return ScriptedProvider(
        [
            ModelResponse(
                text="I'll check the data, then clean it up.",
                tool_calls=[
                    ToolCall("t1", "read_orders", {}),
                    ToolCall("t2", "delete_orders", {}),
                ],
                stop_reason="tool_use",
            ),
            ModelResponse(text="Cleanup complete.", stop_reason="end_turn"),
        ]
    )


print("=" * 64)
print("1. DRY RUN: see what the agent WOULD do -- nothing destructive executes")
print("=" * 64)
agent = Agent(
    model=provider(),
    tools=[read_orders, delete_orders],
    policy=Policy(allow=["read_*"], dry_run=True),
)
run = agent.run("Clean up stale orders.")
for intent in run.intents():
    print(f"  {intent['tool']:<15} -> {intent['status']}")

print()
print("=" * 64)
print("2. CONFIRM: destructive tools pause for human approval")
print("=" * 64)
agent = Agent(
    model=provider(),
    tools=[read_orders, delete_orders],
    policy=Policy(confirm=["delete_*"]),
)
paused = agent.run("Clean up stale orders.")
print("paused  :", paused.paused)
print("question:", paused.pending)
done = paused.resume("yes, approved")
print("resumed :", done.output)
for intent in done.intents():
    print(f"  {intent['tool']:<15} -> {intent['status']}")

print()
print("=" * 64)
print("3. The approval is IN the trace -- auditable forever")
print("=" * 64)
done.print_timeline()
