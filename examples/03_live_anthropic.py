"""Run a real Claude agent, then replay the trace for free.

Requires:  pip install "loom-agent[anthropic]"  and  ANTHROPIC_API_KEY set.

    python examples/03_live_anthropic.py
"""

import os
import sys

from loom import Agent, tool


@tool
def get_weather(city: str) -> str:
    "Get the current weather for a city."
    # A real tool would call an API; this keeps the example self-contained.
    return f"It is 22 degrees and sunny in {city}."


def main() -> int:
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY to run this live example.", file=sys.stderr)
        return 1

    agent = Agent(
        model="claude-opus-4-8",
        tools=[get_weather],
        system="You are a concise travel assistant.",
    )
    run = agent.run("What's the weather in Tokyo, and should I bring an umbrella?")

    print("output:\n", run.output, "\n")
    print("timeline:")
    run.print_timeline()
    print("\ncost:", run.cost())

    # The whole run is now reproducible -- replay it with zero API calls.
    run.save("weather.loom.json")
    replay = run.replay()
    print("\nreplayed output matches:", replay.output == run.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
