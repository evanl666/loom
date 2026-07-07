"""Structured output: schema in the system prompt, validation at the boundary."""

from dataclasses import dataclass, field
from typing import Optional, TypedDict

import pytest

from loom import Agent, OutputInvalid, Run, parse_as, schema_for
from loom.providers import ModelResponse, ScriptedProvider


@dataclass
class Weather:
    city: str
    temp_c: float
    rain: bool
    tags: list[str] = field(default_factory=list)


class Verdict(TypedDict):
    answer: str
    confidence: float


GOOD = '{"city": "Berlin", "temp_c": 12.5, "rain": true, "tags": ["wind"]}'


# -- parse_as / schema_for ---------------------------------------------------


def test_parse_dataclass():
    w = parse_as(Weather, GOOD)
    assert w == Weather(city="Berlin", temp_c=12.5, rain=True, tags=["wind"])


def test_parse_tolerates_prose_and_fences():
    for text in (
        f"Here you go:\n{GOOD}",
        f"```json\n{GOOD}\n```",
    ):
        assert parse_as(Weather, text).city == "Berlin"


def test_parse_coerces_int_to_float():
    w = parse_as(Weather, '{"city": "Oslo", "temp_c": 3, "rain": false}')
    assert w.temp_c == 3.0 and isinstance(w.temp_c, float)


def test_parse_rejects_missing_and_mistyped():
    with pytest.raises(OutputInvalid, match="missing required field 'rain'"):
        parse_as(Weather, '{"city": "Oslo", "temp_c": 3}')
    with pytest.raises(OutputInvalid, match="expected a boolean"):
        parse_as(Weather, '{"city": "Oslo", "temp_c": 3, "rain": "yes"}')
    with pytest.raises(OutputInvalid, match="no JSON object"):
        parse_as(Weather, "I would rather write prose.")


def test_parse_typeddict_and_optional():
    v = parse_as(Verdict, '{"answer": "42", "confidence": 0.9}')
    assert v == {"answer": "42", "confidence": 0.9}
    with pytest.raises(OutputInvalid, match="missing required key"):
        parse_as(Verdict, '{"answer": "42"}')
    assert parse_as(Optional[Weather], "null") is None


def test_schema_lists_required_fields():
    schema = schema_for(Weather)
    assert schema["type"] == "object"
    assert set(schema["required"]) == {"city", "temp_c", "rain"}  # tags has a default
    assert schema["properties"]["temp_c"] == {"type": "number"}


def test_pydantic_model_when_installed():
    pydantic = pytest.importorskip("pydantic")

    class Invoice(pydantic.BaseModel):
        total: float
        paid: bool

    inv = parse_as(Invoice, '{"total": 99.5, "paid": false}')
    assert inv.total == 99.5
    assert schema_for(Invoice)["required"] == ["total", "paid"]
    with pytest.raises(OutputInvalid):
        parse_as(Invoice, '{"total": "lots"}')


# -- the agent loop ----------------------------------------------------------


def make_agent(responses):
    return Agent(
        model=ScriptedProvider(responses),
        output_type=Weather,
        system="You are a weather reporter.",
    )


def test_valid_answer_parses_and_schema_reaches_system():
    agent = make_agent([ModelResponse(text=GOOD, stop_reason="end_turn")])
    assert "temp_c" in agent.system  # schema appended to the system prompt
    run = agent.run("Weather in Berlin?")
    assert run.parsed == Weather(city="Berlin", temp_c=12.5, rain=True, tags=["wind"])


def test_invalid_answer_retries_and_the_retry_is_recorded():
    agent = make_agent(
        [
            ModelResponse(text="It is rainy, about twelve degrees.", stop_reason="end_turn"),
            ModelResponse(text=GOOD, stop_reason="end_turn"),
        ]
    )
    run = agent.run("Weather in Berlin?")
    assert run.parsed.city == "Berlin"
    assert run.num_model_calls == 2  # the retry is part of the trace
    # The error feedback landed in context for the second call.
    assert any(item.source == "validation" for item in run.context.items)


def test_retries_exhausted_sets_stop_reason():
    bad = ModelResponse(text="prose only", stop_reason="end_turn")
    agent = Agent(
        model=ScriptedProvider([bad, bad, bad]),
        output_type=Weather,
        output_retries=2,
    )
    run = agent.run("Weather?")
    assert run.stop_reason == "invalid_output"
    assert run.parsed is None
    assert run.num_model_calls == 3  # initial + 2 retries


def test_validated_run_replays_deterministically(tmp_path):
    agent = make_agent(
        [
            ModelResponse(text="not json", stop_reason="end_turn"),
            ModelResponse(text=GOOD, stop_reason="end_turn"),
        ]
    )
    run = agent.run("Weather in Berlin?")
    path = str(tmp_path / "weather.loom.json")
    run.save(path)

    fresh_agent = make_agent([])  # replay needs config, not scripted responses
    loaded = Run.load(path, agent=fresh_agent)
    replayed = loaded.replay()
    assert replayed.output == run.output
    assert replayed.parsed == run.parsed
    assert replayed.num_model_calls == 2  # retry path walked again, zero API calls
