"""Bisect locates the turn where a run first went wrong."""

from loom import Agent, tool
from loom.providers import ModelResponse, ScriptedProvider, ToolCall


@tool
def step() -> str:
    "Advance one step."
    return "ok"


def _tooluse(cid: str) -> ModelResponse:
    return ModelResponse(
        tool_calls=[ToolCall(id=cid, name="step", input={})], stop_reason="tool_use"
    )


def test_bisect_over_multi_turn_loop():
    provider = ScriptedProvider(
        [
            _tooluse("t1"),  # turn 1: fine
            _tooluse("t2"),  # turn 2: fine
            ModelResponse(text="ERROR: lost the thread", stop_reason="end_turn"),  # turn 3: bad
        ]
    )
    run = Agent(model=provider, tools=[step]).run("loop")
    assert run.num_turns == 3
    bad_turn = run.bisect(lambda text: "ERROR" not in text)
    assert bad_turn == 3


def test_bisect_returns_minus_one_when_all_pass():
    provider = ScriptedProvider([ModelResponse(text="all good", stop_reason="end_turn")])
    run = Agent(model=provider).run("go")
    assert run.bisect(lambda text: "ERROR" not in text) == -1
