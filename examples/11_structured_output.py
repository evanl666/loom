"""Structured output: a validated object back, with retries at the boundary.

    python examples/11_structured_output.py

Offline: a ScriptedProvider first answers with prose (invalid), then with good
JSON -- watch the harness feed the parse error back and retry, all recorded.
"""

from dataclasses import dataclass

from loom import Agent
from loom.providers import ModelResponse, ScriptedProvider


@dataclass
class Weather:
    city: str
    temp_c: float
    rain: bool


provider = ScriptedProvider(
    [
        # First attempt: prose. The harness rejects it and asks again.
        ModelResponse(text="It's about twelve degrees and rainy.", stop_reason="end_turn"),
        # Second attempt: valid JSON matching the schema.
        ModelResponse(text='{"city": "Berlin", "temp_c": 12.0, "rain": true}', stop_reason="end_turn"),
    ]
)

agent = Agent(model=provider, output_type=Weather)
run = agent.run("Weather in Berlin?")

print("parsed:", run.parsed)
print("type:  ", type(run.parsed).__name__)
print("calls: ", run.num_model_calls, "(the failed attempt and the retry are both in the trace)")

# The retry path is recorded, so the whole thing replays deterministically:
replay = run.replay()
assert replay.parsed == run.parsed
print("replayed parsed matches:", replay.parsed == run.parsed)
