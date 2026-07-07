"""Write-ahead journaling and crash recovery."""

import json

import pytest

from loom import Agent, Journal, Run, tool
from loom.providers import ModelResponse, ScriptedProvider, ToolCall


class CrashingProvider:
    """Succeeds for the first N calls, then dies like a lost connection."""

    def __init__(self, inner, crash_after: int):
        self.inner = inner
        self.model = inner.model
        self.name = "crashing"
        self.calls = 0
        self.crash_after = crash_after

    def complete(self, system, messages, tools):
        if self.calls >= self.crash_after:
            raise ConnectionError("simulated crash: network died mid-run")
        self.calls += 1
        return self.inner.complete(system, messages, tools)


EXPENSIVE_CALLS = {"count": 0}


@tool
def expensive() -> str:
    "An expensive side-effectful operation that must run exactly once."
    EXPENSIVE_CALLS["count"] += 1
    return "expensive result"


def scripted():
    return ScriptedProvider(
        [
            ModelResponse(
                tool_calls=[ToolCall("t1", "expensive", {})], stop_reason="tool_use"
            ),
            ModelResponse(text="final answer after tool", stop_reason="end_turn"),
        ]
    )


def test_journal_written_incrementally(tmp_path):
    path = str(tmp_path / "run.jsonl")
    agent = Agent(model=scripted(), tools=[expensive], journal=path)
    run = agent.run("do the thing")

    lines = [json.loads(line) for line in open(path)]
    assert lines[0]["type"] == "header"
    assert lines[0]["episodes"] == ["do the thing"]
    effects = [line for line in lines if line["type"] == "effect"]
    assert len(effects) == len(run.log) == 3  # model, tool, model


def test_crash_then_recover_exactly_once(tmp_path):
    EXPENSIVE_CALLS["count"] = 0
    path = str(tmp_path / "crash.jsonl")

    # First attempt: model call 1 + tool succeed, then the 2nd model call dies.
    provider = CrashingProvider(scripted(), crash_after=1)
    agent = Agent(model=provider, tools=[expensive], journal=path)
    with pytest.raises(ConnectionError):
        agent.run("do the thing")
    assert EXPENSIVE_CALLS["count"] == 1  # the tool ran before the crash

    # The journal preserved everything paid for before the crash.
    _, entries = Journal.read(path)
    assert [e.kind for e in entries] == ["model", "tool:expensive"]

    # Recovery: healthy provider, same journal. Prefix replays; run finishes.
    agent2 = Agent(model=scripted(), tools=[expensive], journal=path)
    # Skip the first scripted response: it was already consumed pre-crash and
    # will be served from the journal, not requested from the provider.
    agent2.provider.responses = agent2.provider.responses[1:]
    run = Run.recover(path, agent=agent2)

    assert run.output == "final answer after tool"
    assert EXPENSIVE_CALLS["count"] == 1  # the expensive tool NEVER re-ran


def test_recover_tolerates_torn_final_line(tmp_path):
    path = str(tmp_path / "torn.jsonl")
    agent = Agent(model=scripted(), tools=[expensive], journal=path)
    agent.run("do the thing")

    with open(path, "a") as f:
        f.write('{"type": "effect", "seq": 99, "ki')  # crash mid-write

    _, entries = Journal.read(path)
    assert len(entries) == 3  # the torn tail is ignored, prefix intact

    run = Run.recover(path, agent=Agent(model=scripted(), tools=[expensive]))
    assert run.output == "final answer after tool"


def test_recover_completed_run_is_idempotent(tmp_path):
    path = str(tmp_path / "done.jsonl")
    agent = Agent(model=scripted(), tools=[expensive], journal=path)
    original = agent.run("do the thing")

    # Recovering a finished journal replays it: same output, zero live calls.
    fresh = Agent(model=ScriptedProvider([]), tools=[expensive])  # would fail if called
    recovered = Run.recover(path, agent=fresh)
    assert recovered.output == original.output


def test_replay_does_not_touch_the_journal(tmp_path):
    path = str(tmp_path / "run.jsonl")
    agent = Agent(model=scripted(), tools=[expensive], journal=path)
    run = agent.run("do the thing")
    before = open(path).read()

    run.replay()  # pure replay: allow_live=False -> no journaling
    assert open(path).read() == before


def test_ask_rewrites_journal_with_full_conversation(tmp_path):
    path = str(tmp_path / "convo.jsonl")
    provider = ScriptedProvider(
        [
            ModelResponse(text="first answer", stop_reason="end_turn"),
            ModelResponse(text="second answer", stop_reason="end_turn"),
        ]
    )
    agent = Agent(model=provider, journal=path)
    run = agent.run("first question").ask("second question")

    header, entries = Journal.read(path)
    assert header["episodes"] == ["first question", "second question"]
    assert len(entries) == len(run.log) == 2
