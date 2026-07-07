"""Shared agent for the README demo GIF. Offline and deterministic.

A context-sensitive RuleProvider stands in for the model so the demo runs
without an API key -- swap it for model="claude-opus-4-8" to go live.
"""

from loom import Agent, tool
from loom.providers import ModelResponse, RuleProvider, ToolCall

WEATHER = {"Berlin": "rain, 12C", "Lisbon": "sunny, 24C"}


@tool
def get_weather(city: str) -> str:
    "Look up today's weather for a city."
    report = WEATHER.get(city, "clear, 18C")
    print(f"  [tool] get_weather({city!r}) -> {report!r}")
    return report


def policy(messages):
    text = str(messages).lower()
    if "rain" in text:
        return ModelResponse(
            text="It's raining and 12C in Berlin -- take the U-Bahn.",
            stop_reason="end_turn",
        )
    if "sunny" in text:
        return ModelResponse(
            text="Sunny and 24C in Lisbon -- perfect day to bike!",
            stop_reason="end_turn",
        )
    city = "Lisbon" if "lisbon" in text else "Berlin"
    return ModelResponse(
        tool_calls=[ToolCall("t1", "get_weather", {"city": city})],
        stop_reason="tool_use",
    )


def make_agent() -> Agent:
    return Agent(model=RuleProvider(rules=[policy]), tools=[get_weather])
