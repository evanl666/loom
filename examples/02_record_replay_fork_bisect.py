"""The flagship demo: record -> replay -> fork -> bisect. Offline, no API key.

    python examples/02_record_replay_fork_bisect.py

Uses a context-sensitive RuleProvider so that editing the context at a fork point
genuinely changes the downstream branch -- exactly what you can't do with any
other agent framework.
"""

from loom import Agent
from loom.providers import ModelResponse, RuleProvider


def last_user(messages):
    for m in reversed(messages):
        if m["role"] == "user":
            return m["content"].lower()
    return ""


# A tiny deterministic "model": it answers differently depending on the question.
def wants_celsius(messages):
    if "celsius" in last_user(messages):
        return ModelResponse(text="It is 20 degrees celsius today.", stop_reason="end_turn")
    return None


def default(messages):
    return ModelResponse(text="It is 68 degrees fahrenheit today.", stop_reason="end_turn")


agent = Agent(model=RuleProvider(rules=[wants_celsius, default]))

print("=" * 60)
print("1. RECORD a run")
print("=" * 60)
run = agent.run("What's the temperature in celsius?")
print("output:", run.output)
run.print_timeline()

print("\n" + "=" * 60)
print("2. SAVE + REPLAY (zero model calls, identical output)")
print("=" * 60)
run.save("demo.loom.json")
replay = run.replay()
print("replayed output:", replay.output)
print("match:", replay.output == run.output)

print("\n" + "=" * 60)
print("3. FORK: rewind to turn 0, rewrite the question, new branch")
print("=" * 60)


def rewrite(ctx):
    ctx.items[0].content = "What's the temperature?"  # drop 'celsius'


branch = run.fork(at=0, edit=rewrite)
print("original branch:", run.output)
print("forked branch:  ", branch.output)

print("\n" + "=" * 60)
print("4. BISECT: find the first turn that mentions fahrenheit")
print("=" * 60)
bad = run.bisect(lambda text: "fahrenheit" not in text.lower())
print("original run never mentions fahrenheit -> bisect returns:", bad)
bad2 = branch.bisect(lambda text: "fahrenheit" not in text.lower())
print("forked run mentions fahrenheit at turn:", bad2)
