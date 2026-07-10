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


def test_load_rejects_non_trace_json_with_a_clear_error(tmp_path):
    """Run.load powers nearly every CLI command; pointing it at a JSON file that
    isn't a loom trace must give a clear ValueError, not a cryptic KeyError/
    AttributeError from deep inside."""
    import pytest

    from loom import Run

    for name, content in [
        ("empty.json", "{}"),
        ("random.json", '{"foo": 1}'),
        ("array.json", "[1, 2, 3]"),
        ("scalar.json", '"hello"'),
        ("null.json", "null"),
        ("badlog.json", '{"log": "nope", "prompt": "p", "output": "o"}'),
    ]:
        p = tmp_path / name
        p.write_text(content)
        with pytest.raises(ValueError, match="not a loom trace"):
            Run.load(str(p))


def test_openai_adapter_handles_empty_choices():
    """A content-filtered response can carry no choices; the adapter must return
    an empty end-of-turn, not IndexError on choices[0]."""
    from types import SimpleNamespace as NS

    from loom.providers.openai import OpenAIProvider

    r = OpenAIProvider._from_openai(NS(choices=[], usage=None))
    assert r.text == "" and r.stop_reason == "end_turn" and not r.tool_calls
    r2 = OpenAIProvider._from_openai(NS(choices=None))
    assert r2.text == "" and r2.stop_reason == "end_turn"
