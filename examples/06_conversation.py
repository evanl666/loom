"""Multi-turn conversation with run.ask(). Offline, no API key.

    python examples/06_conversation.py

Each ask() replays the whole recorded history for free and only runs the new
exchange live -- the conversation is one growing trace that stays replayable,
forkable, and diffable.
"""

from loom import Agent
from loom.providers import ModelResponse, RuleProvider


def users(messages):
    return [m["content"] for m in messages if m["role"] == "user"]


# A deterministic "model" that needs memory of earlier episodes to answer.
def refund(messages):
    if "refund" in users(messages)[-1].lower():
        if any("A123" in u for u in users(messages)[:-1]):
            return ModelResponse(text="Refund for order A123 initiated.", stop_reason="end_turn")
        return ModelResponse(text="Which order do you mean?", stop_reason="end_turn")
    return None


def status(messages):
    if "A123" in users(messages)[-1]:
        return ModelResponse(text="Order A123 shipped yesterday.", stop_reason="end_turn")
    return None


def default(messages):
    return ModelResponse(text="How can I help?", stop_reason="end_turn")


agent = Agent(model=RuleProvider(rules=[refund, status, default]))

print("=" * 60)
print("A conversation: context carries across turns")
print("=" * 60)
run1 = agent.run("Where is order A123?")
print("user : Where is order A123?")
print("agent:", run1.output)

run2 = run1.ask("I want a refund.")
print("user : I want a refund.")
print("agent:", run2.output, " <- knew the order id from turn 1")

print("\nwhole conversation is ONE trace:")
run2.print_timeline()

print("\nreplay of the full conversation (zero model calls):",
      run2.replay().output == run2.output)

print("\n" + "=" * 60)
print("Contrast: a cold run without the first exchange")
print("=" * 60)
cold = agent.run("I want a refund.")
print("agent:", cold.output)
