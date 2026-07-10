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


# ------------------------------------------------------------------ two-phase


def test_intents_precede_effects_and_are_fulfilled(tmp_path):
    path = str(tmp_path / "run.jsonl")
    Agent(model=scripted(), tools=[expensive], journal=path).run("do the thing")

    lines = [json.loads(line) for line in open(path)]
    kinds = [(d["type"], d.get("kind")) for d in lines]
    # every effect is announced first: intent(model), effect(model), intent(tool)...
    assert kinds[1] == ("intent", "model") and kinds[2] == ("effect", "model")
    assert kinds[3] == ("intent", "tool:expensive") and kinds[4] == ("effect", "tool:expensive")

    header, entries, unfinished = Journal.read_full(path)
    assert len(entries) == 3 and unfinished == []  # all intents fulfilled


def test_crash_inside_a_tool_leaves_an_unfinished_intent(tmp_path):
    # Build the exact crash artifact: a journal that stops right after a tool
    # intent. (Exceptions inside tools are handled by the loop and recorded;
    # only real process death -- kill -9, power loss -- produces this file.)
    full = str(tmp_path / "full.jsonl")
    Agent(model=scripted(), tools=[expensive], journal=full).run("do the thing")
    lines = open(full).read().splitlines(keepends=True)
    cut = next(
        i for i, line in enumerate(lines)
        if '"intent"' in line and '"tool:expensive"' in line
    )
    path = str(tmp_path / "boom.jsonl")
    with open(path, "w") as f:
        f.writelines(lines[: cut + 1])

    _, entries, unfinished = Journal.read_full(path)
    assert unfinished and unfinished[-1]["kind"] == "tool:expensive"

    # Recovery refuses to guess whether the side effect happened...
    from loom.journal import UnfinishedEffect

    with pytest.raises(UnfinishedEffect, match="may or may not have run"):
        Run.recover(path, agent=Agent(model=scripted(), tools=[expensive]))

    # ...unless told the re-execution is acceptable. (The first scripted
    # response was consumed before the crash and replays from the journal.)
    EXPENSIVE_CALLS["count"] = 0
    remaining = ScriptedProvider(
        [ModelResponse(text="final answer after tool", stop_reason="end_turn")]
    )
    done = Run.recover(
        path,
        agent=Agent(model=remaining, tools=[expensive]),
        on_unfinished="retry",
    )
    assert done.output == "final answer after tool"
    assert EXPENSIVE_CALLS["count"] == 1  # the accepted re-execution

    # Inspection without resuming never raises: reading is always safe.
    partial = Run.recover(path, agent=Agent(model=scripted()), resume=False)
    assert partial.truncated


def test_unfinished_model_intent_recovers_silently(tmp_path):
    path = str(tmp_path / "crash.jsonl")
    provider = CrashingProvider(scripted(), crash_after=0)  # dies on first model call
    with pytest.raises(ConnectionError):
        Agent(model=provider, tools=[expensive], journal=path).run("do the thing")

    _, _, unfinished = Journal.read_full(path)
    assert unfinished and unfinished[-1]["kind"] == "model"
    # model calls are harness-internal: retrying costs tokens, not correctness
    run = Run.recover(path, agent=Agent(model=scripted(), tools=[expensive]))
    assert run.output == "final answer after tool"


def test_journal_file_handle_is_closed_after_run(tmp_path):
    """The write-ahead journal must release its fd when the run ends, or a
    long-lived process that keeps Run objects leaks a descriptor per run."""
    from loom import Agent, tool
    from loom.providers import ModelResponse, ScriptedProvider, ToolCall

    @tool
    def add(a: int, b: int) -> int:
        "add"
        return a + b

    jp = str(tmp_path / "task.jsonl")
    agent = Agent(
        model=ScriptedProvider([
            ModelResponse(tool_calls=[ToolCall("t1", "add", {"a": 1, "b": 2})], stop_reason="tool_use"),
            ModelResponse(text="3", stop_reason="end_turn"),
        ]),
        tools=[add],
        journal=jp,
    )
    run = agent.run("go")
    j = run.recorder.journal
    assert j._f is None  # handle released
    j.close()  # idempotent, no error
