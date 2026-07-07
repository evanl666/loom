"""Streaming callbacks and parallel tool execution (deterministic recording)."""

import time

from loom import Agent, tool
from loom.providers import ModelResponse, ScriptedProvider, ToolCall


def test_scripted_streaming_emits_tokens():
    tokens = []
    provider = ScriptedProvider(
        [ModelResponse(text="hello streaming world", stop_reason="end_turn")],
        on_token=tokens.append,
    )
    run = Agent(model=provider).run("hi")
    assert "".join(tokens) == "hello streaming world"
    assert run.output == "hello streaming world"


def test_arun_wraps_run():
    import asyncio

    provider = ScriptedProvider([ModelResponse(text="async ok", stop_reason="end_turn")])
    run = asyncio.run(Agent(model=provider).arun("hi"))
    assert run.output == "async ok"


@tool
def slow_a() -> str:
    "Slow tool A."
    time.sleep(0.2)
    return "a done"


@tool
def slow_b() -> str:
    "Slow tool B."
    time.sleep(0.2)
    return "b done"


def parallel_agent(parallel: bool):
    provider = ScriptedProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall("t1", "slow_a", {}),
                    ToolCall("t2", "slow_b", {}),
                ],
                stop_reason="tool_use",
            ),
            ModelResponse(text="both done", stop_reason="end_turn"),
        ]
    )
    return Agent(model=provider, tools=[slow_a, slow_b], parallel_tools=parallel)


def test_parallel_tools_run_concurrently():
    start = time.monotonic()
    run = parallel_agent(parallel=True).run("do both")
    elapsed = time.monotonic() - start
    assert elapsed < 0.35  # sequential would be >= 0.4s
    assert run.output == "both done"


def test_parallel_recording_order_is_deterministic_and_replayable():
    run = parallel_agent(parallel=True).run("do both")
    kinds = [e.kind for e in run.log]
    # Recorded in call order regardless of completion order.
    assert kinds == ["model", "tool:slow_a", "tool:slow_b", "model"]

    # Replay serves everything from the log: instant (no sleeps re-run).
    start = time.monotonic()
    replayed = run.replay()
    assert time.monotonic() - start < 0.1
    assert replayed.output == run.output
