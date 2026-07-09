"""Skill crystallization: proven tool sequences become macro-tools."""

from loom import Agent, tool
from loom.providers import ModelResponse, RuleProvider, ScriptedProvider, ToolCall
from loom.skills import Skill, load, mine, save

CALLS = []


@tool
def geocode(city: str) -> str:
    "Find coordinates for a city."
    CALLS.append(("geocode", city))
    return f"{city}@52.5,13.4"


@tool
def forecast(coords: str, units: str) -> str:
    "Forecast weather at coordinates."
    CALLS.append(("forecast", coords))
    return f"sunny at {coords} ({units})"


def record_run(city: str):
    """One successful run that geocodes then forecasts -- the provable pattern."""
    provider = ScriptedProvider(
        [
            ModelResponse(
                tool_calls=[ToolCall("t1", "geocode", {"city": city})],
                stop_reason="tool_use",
            ),
            ModelResponse(
                tool_calls=[
                    ToolCall("t2", "forecast", {"coords": f"{city}@52.5,13.4", "units": "metric"})
                ],
                stop_reason="tool_use",
            ),
            ModelResponse(text=f"Sunny in {city}.", stop_reason="end_turn"),
        ]
    )
    return Agent(model=provider, tools=[geocode, forecast]).run(f"Weather in {city}?")


def test_mine_finds_the_pattern_and_learns_parameters():
    runs = [record_run("Berlin"), record_run("Lisbon")]
    skills = mine(runs)
    assert len(skills) == 1
    s = skills[0]
    assert s.name == "skill_geocode_then_forecast"
    assert s.support == 2
    # city and coords varied across runs -> parameters; units never did -> baked.
    assert set(s.params) == {"city", "coords"}
    assert s.steps[1]["args"]["units"] == "metric"


def test_skill_executes_as_one_tool_and_replays_free(tmp_path):
    skills = mine([record_run("Berlin"), record_run("Lisbon")])
    skills[0].approved = True
    macro = skills[0].as_tool([geocode, forecast])

    provider = ScriptedProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        "t1",
                        macro.name,
                        {"city": "Tokyo", "coords": "Tokyo@52.5,13.4"},
                    )
                ],
                stop_reason="tool_use",
            ),
            ModelResponse(text="Done.", stop_reason="end_turn"),
        ]
    )
    agent = Agent(model=provider, tools=[geocode, forecast, macro])
    CALLS.clear()
    run = agent.run("Weather in Tokyo?")
    # Both real tools ran, but the trace shows ONE macro effect.
    assert ("geocode", "Tokyo") in CALLS and ("forecast", "Tokyo@52.5,13.4") in CALLS
    kinds = [e.kind for e in run.log]
    assert f"tool:{macro.name}" in kinds
    assert "tool:geocode" not in kinds

    CALLS.clear()
    replay = run.replay()  # recorded result served; nothing re-executes
    assert replay.output == run.output
    assert CALLS == []


def test_failed_runs_prove_nothing():
    good = record_run("Berlin")
    bad = record_run("Lisbon")
    bad.truncated = True  # simulate a run that never finished
    assert mine([good, bad]) == []


def test_mining_needs_min_support():
    assert mine([record_run("Berlin")]) == []  # one occurrence is an anecdote


def test_skill_library_round_trips(tmp_path):
    skills = mine([record_run("Berlin"), record_run("Lisbon")])
    path = str(tmp_path / "skills.json")
    save(skills, path)
    loaded = load(path)
    assert loaded == skills
    # And the loaded skill still binds and runs (once a human arms it).
    loaded[0].approved = True
    macro = loaded[0].as_tool([geocode, forecast])
    out = macro(city="Oslo", coords="Oslo@52.5,13.4")
    assert "metric" in out


def test_missing_tool_is_a_clear_error():
    skills = mine([record_run("Berlin"), record_run("Lisbon")])
    skills[0].approved = True
    try:
        skills[0].as_tool([geocode])  # forecast missing
        assert False, "expected KeyError"
    except KeyError as e:
        assert "forecast" in str(e)


# ---------------------------------------------------------------- safety gate


def test_unapproved_skills_refuse_to_arm():
    import pytest

    skills = mine([record_run("Berlin"), record_run("Lisbon")])
    assert skills[0].approved is False  # mined skills are born unapproved
    with pytest.raises(PermissionError, match="not approved"):
        skills[0].as_tool([geocode, forecast])
    skills[0].as_tool([geocode, forecast], force=True)  # experiments can override


def test_mine_check_defines_success():
    good, bad = record_run("Berlin"), record_run("Lisbon")
    # Both runs finished -- but the check says only Berlin's answer was right.
    assert mine([good, bad], check=lambda r: "Berlin" in r.output) == []
    assert len(mine([good, bad], check=lambda r: "in" in r.output)) == 1


def test_approve_updates_the_library(tmp_path):
    from loom.skills import approve

    skills = mine([record_run("Berlin"), record_run("Lisbon")])
    path = str(tmp_path / "skills.json")
    save(skills, path)
    assert approve(path, skills[0].name) is True
    assert load(path)[0].approved is True
    assert approve(path, "no_such_skill") is False
