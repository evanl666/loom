"""Human-in-the-loop: pause, persist, resume. Offline, no API key.

    python examples/07_human_in_the_loop.py

A human answer is nondeterminism like any other, so Loom records it as an
effect: paused runs are saved to disk and resumed later, and replays include
the human decision -- an auditable approval flow with no extra machinery.
"""

from loom import Agent, Run, ask_human
from loom.providers import ModelResponse, ScriptedProvider, ToolCall

provider = ScriptedProvider(
    [
        ModelResponse(
            text="This refund exceeds my authority; asking the operator.",
            tool_calls=[ToolCall("h1", "ask_human", {"question": "Approve $500 refund for A123?"})],
            stop_reason="tool_use",
        ),
        ModelResponse(text="Refund approved and processed.", stop_reason="end_turn"),
    ]
)

agent = Agent(model=provider, tools=[ask_human()], system="You are a support agent.")

print("=" * 60)
print("1. The run PAUSES when the agent needs a human")
print("=" * 60)
run = agent.run("Customer wants a $500 refund on order A123.")
print("paused:  ", run.paused)
print("question:", run.pending)

print()
print("=" * 60)
print("2. Persist the paused run -- answer it whenever")
print("=" * 60)
run.save("pending-approval.loom.json")
print("saved -> pending-approval.loom.json (answer it tomorrow if you like)")

loaded = Run.load("pending-approval.loom.json", agent=agent)
done = loaded.resume("yes, approved by evan")
print("resumed ->", done.output)

print()
print("=" * 60)
print("3. The decision is IN the trace -- auditable and replayable")
print("=" * 60)
done.print_timeline()
print("\nreplay (human answer included, zero live calls):",
      done.replay().output == done.output)
