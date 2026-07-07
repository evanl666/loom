"""Clock and randomness through the boundary (loom.now / loom.random)."""

import time

import loom
from loom import Agent, tool
from loom.ambient import _activate, _deactivate
from loom.effect import Recorder
from loom.providers import ModelResponse, ScriptedProvider, ToolCall


def test_fallback_outside_any_run():
    before = time.time()
    assert before <= loom.now() <= time.time()
    assert 0.0 <= loom.random() < 1.0


def test_harness_level_calls_are_recorded_and_replayed():
    rec = Recorder.record()
    token = _activate(rec)
    try:
        t1, r1 = loom.now(), loom.random()
    finally:
        _deactivate(token)
    assert [e.kind for e in rec.log] == ["time", "random"]

    replay = Recorder.replay(rec.log)
    token = _activate(replay)
    try:
        assert loom.now() == t1  # served from the log, not the wall clock
        assert loom.random() == r1
    finally:
        _deactivate(token)


def test_clock_agent_records_time_and_replays_same_date():
    agent = Agent(
        model=ScriptedProvider([ModelResponse(text="hi", stop_reason="end_turn")]),
        clock=True,
    )
    run = agent.run("hello")
    assert run.log[0].kind == "time"
    stamped = [i for i in run.context.items if i.source == "clock"]
    assert len(stamped) == 1 and "UTC" in stamped[0].content

    replay = run.replay()  # the replayed run sees the ORIGINAL moment
    replayed_stamp = [i for i in replay.context.items if i.source == "clock"]
    assert replayed_stamp[0].content == stamped[0].content


def test_inside_a_tool_returns_real_values_and_records_no_effect():
    seen = {}

    @tool
    def stamp() -> str:
        "Timestamp something."
        seen["t"] = loom.now()  # inside a tool: real time, no nested effect
        return f"stamped at {seen['t']}"

    provider = ScriptedProvider(
        [
            ModelResponse(tool_calls=[ToolCall("t1", "stamp", {})], stop_reason="tool_use"),
            ModelResponse(text="done", stop_reason="end_turn"),
        ]
    )
    run = Agent(model=provider, tools=[stamp]).run("stamp it")
    kinds = [e.kind for e in run.log]
    assert "time" not in kinds  # no ambient effect leaked from tool execution
    assert kinds == ["model", "tool:stamp", "model"]
    # Determinism still holds: the timestamp lives inside the recorded result.
    assert str(seen["t"]) in run.log[1].result
    assert run.replay().output == run.output
