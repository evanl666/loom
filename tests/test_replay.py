"""Replay is deterministic and makes zero live model calls."""

from loom import Agent, tool
from loom.providers import ModelResponse, ScriptedProvider, ToolCall


@tool
def add(a: int, b: int) -> int:
    "Add two numbers."
    return a + b


class CountingProvider:
    """Wraps a provider and counts how many times the model is actually called."""

    def __init__(self, inner):
        self.inner = inner
        self.model = inner.model
        self.name = "counting"
        self.calls = 0

    def complete(self, system, messages, tools):
        self.calls += 1
        return self.inner.complete(system, messages, tools)


def build():
    provider = CountingProvider(
        ScriptedProvider(
            [
                ModelResponse(
                    tool_calls=[ToolCall(id="t1", name="add", input={"a": 2, "b": 40})],
                    stop_reason="tool_use",
                ),
                ModelResponse(text="42", stop_reason="end_turn"),
            ]
        )
    )
    return Agent(model=provider, tools=[add]), provider


def test_replay_matches_and_is_free():
    agent, provider = build()
    run = agent.run("meaning of life?")
    assert provider.calls == 2

    replayed = run.replay()
    assert replayed.output == run.output == "42"
    # Replay served everything from the log -- no new model calls.
    assert provider.calls == 2


def test_save_load_roundtrip(tmp_path):
    agent, _ = build()
    run = agent.run("meaning of life?")
    path = tmp_path / "trace.loom.json"
    run.save(str(path))

    loaded = Run_load(agent, str(path))
    assert loaded.output == "42"
    assert loaded.num_turns == 2

    replayed = loaded.replay()
    assert replayed.output == "42"


def Run_load(agent, path):
    from loom import Run

    return Run.load(path, agent=agent)
