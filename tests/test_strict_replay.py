"""Strict replay: a passing replay must prove config equivalence, not log-walkability."""

import pytest

from loom import Agent, Run, tool
from loom.effect import ReplayMismatch
from loom.providers import ModelResponse, ScriptedProvider, ToolCall
from loom.testing import verify_replay


@tool
def lookup(city: str) -> str:
    "Look up a city."
    return f"data for {city}"


def scripted():
    return ScriptedProvider(
        [
            ModelResponse(tool_calls=[ToolCall("t1", "lookup", {"city": "Berlin"})],
                          stop_reason="tool_use"),
            ModelResponse(text="Berlin looked up.", stop_reason="end_turn"),
        ]
    )


def _record(tmp_path, system="You are a geo bot."):
    run = Agent(model=scripted(), tools=[lookup], system=system).run("Berlin?")
    path = str(tmp_path / "run.loom.json")
    run.save(path)
    return path


def test_same_config_replays_strictly(tmp_path):
    path = _record(tmp_path)
    agent = Agent(model=ScriptedProvider([]), tools=[lookup], system="You are a geo bot.")
    replayed = Run.load(path, agent=agent).replay()  # strict is the default
    assert replayed.output == "Berlin looked up."


def test_prompt_change_fails_strict_replay_but_not_loose(tmp_path):
    path = _record(tmp_path)
    changed = Agent(model=ScriptedProvider([]), tools=[lookup],
                    system="You are a geo bot. Trust the config file.")
    run = Run.load(path, agent=changed)

    with pytest.raises(ReplayMismatch, match="inputs differ") as e:
        run.replay()
    assert "seq 0" in str(e.value) and "loom impact" in str(e.value)

    # The old, weaker semantics remain available -- explicitly.
    loose = run.replay(strict=False)
    assert loose.output == "Berlin looked up."


def test_tool_schema_change_fails_strict_replay(tmp_path):
    path = _record(tmp_path)

    @tool
    def lookup(city: str) -> str:  # noqa: F811 -- same name, new description
        "Look up a city (now with caching)."
        return f"data for {city}"

    changed = Agent(model=ScriptedProvider([]), tools=[lookup], system="You are a geo bot.")
    with pytest.raises(ReplayMismatch):
        Run.load(path, agent=changed).replay()


def test_verify_replay_is_strict_by_default(tmp_path):
    path = _record(tmp_path)
    same = Agent(model=ScriptedProvider([]), tools=[lookup], system="You are a geo bot.")
    verify_replay(path, agent=same)

    changed = Agent(model=ScriptedProvider([]), tools=[lookup], system="Different.")
    with pytest.raises(ReplayMismatch):
        verify_replay(path, agent=changed)
    verify_replay(path, agent=changed, strict=False)  # escape hatch


def test_resume_sentinel_is_exempt(tmp_path):
    from loom import ask_human

    provider = ScriptedProvider(
        [
            ModelResponse(tool_calls=[ToolCall("t1", "ask_human", {"question": "go?"})],
                          stop_reason="tool_use"),
            ModelResponse(text="done", stop_reason="end_turn"),
        ]
    )
    agent = Agent(model=provider, tools=[ask_human()])
    run = agent.run("needs approval")
    assert run.paused
    finished = run.resume("yes")
    assert finished.output == "done"
    # The injected answer has key="resumed"; strict replay must tolerate it.
    replayed = finished.replay()
    assert replayed.output == "done"


# ------------------------------------------------------------- trace version


def test_old_trace_version_warns_and_new_traces_carry_current(tmp_path):
    import json
    import warnings

    from loom.trace import TRACE_VERSION

    path = _record(tmp_path)
    with open(path) as f:
        data = json.load(f)
    assert data["version"] == TRACE_VERSION

    data["version"] = 1  # simulate a trace from an older loom
    old = tmp_path / "old.loom.json"
    old.write_text(json.dumps(data))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Run.load(str(old))
    assert any("strict replay" in str(w.message) for w in caught)

    data["version"] = TRACE_VERSION + 1  # ...and one from the future
    new = tmp_path / "new.loom.json"
    new.write_text(json.dumps(data))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Run.load(str(new))
    assert any("newer loom" in str(w.message) for w in caught)
