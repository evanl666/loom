"""Effect cache, model A/B rerun, and edits persisted as effects."""

from loom import Agent, EffectCache, Run, tool
from loom.providers import ModelResponse, RuleProvider, ScriptedProvider, ToolCall


class Counting:
    def __init__(self, inner):
        self.inner, self.model, self.name, self.calls = inner, inner.model, "counting", 0

    def complete(self, system, messages, tools):
        self.calls += 1
        return self.inner.complete(system, messages, tools)


# -- cache -------------------------------------------------------------------

TOOL_CALLS = {"n": 0}


@tool
def fetch() -> str:
    "Fetch data (side effect -- should NOT be cached by default)."
    TOOL_CALLS["n"] += 1
    return "fetched"


def scripted():
    return ScriptedProvider(
        [
            ModelResponse(tool_calls=[ToolCall("t1", "fetch", {})], stop_reason="tool_use"),
            ModelResponse(text="answer", stop_reason="end_turn"),
        ]
    )


def test_cache_serves_identical_model_calls_across_runs():
    TOOL_CALLS["n"] = 0
    cache = EffectCache()
    p1 = Counting(scripted())
    run1 = Agent(model=p1, tools=[fetch], cache=cache).run("same question")
    assert p1.calls == 2

    p2 = Counting(scripted())
    run2 = Agent(model=p2, tools=[fetch], cache=cache).run("same question")
    assert p2.calls == 0  # both model calls served from cache
    assert run2.output == run1.output == "answer"
    # Tools are NOT cached by default: the side effect ran in both runs.
    assert TOOL_CALLS["n"] == 2


def test_cache_persists_to_disk(tmp_path):
    path = str(tmp_path / "cache.jsonl")
    cache = EffectCache(path)
    Agent(model=Counting(scripted()), tools=[fetch], cache=cache).run("q")

    fresh = EffectCache(path)  # a new process would load the same file
    p = Counting(scripted())
    run = Agent(model=p, tools=[fetch], cache=fresh).run("q")
    assert p.calls == 0
    assert run.output == "answer"


def test_cache_misses_on_different_inputs():
    cache = EffectCache()
    p1 = Counting(ScriptedProvider([ModelResponse(text="a")]))
    Agent(model=p1, cache=cache).run("question one")
    p2 = Counting(ScriptedProvider([ModelResponse(text="b")]))
    run = Agent(model=p2, cache=cache).run("question two")  # different inputs
    assert p2.calls == 1  # no false hit
    assert run.output == "b"


# -- rerun -------------------------------------------------------------------


def test_rerun_same_conversation_on_other_model():
    run_a = Agent(model=ScriptedProvider([ModelResponse(text="claude says X")])).run("q")
    run_b = run_a.rerun(model=ScriptedProvider([ModelResponse(text="other says Y")]))

    assert run_b.episodes == run_a.episodes
    assert run_b.output == "other says Y"
    d = run_a.diff(run_b)
    assert d.steps[0].status == "results-differ"  # same inputs, different model


def test_rerun_preserves_output_type_config():
    # A/B rerun of a structured-output agent must stay apples-to-apples: the
    # new agent keeps output_type (and doesn't double-append its instruction).
    from dataclasses import dataclass

    @dataclass
    class Ans:
        value: int

    agent_a = Agent(
        model=ScriptedProvider([ModelResponse(text='{"value": 1}')]),
        output_type=Ans,
        system="Be terse.",
    )
    run_a = agent_a.run("q")
    run_b = run_a.rerun(model=ScriptedProvider([ModelResponse(text='{"value": 2}')]))

    assert run_b.agent.output_type is Ans
    assert run_b.parsed == Ans(value=2)
    # the format instruction appears exactly once, not doubled
    assert run_b.agent.system.count("respond with ONLY a JSON object") == 1
    assert run_b.agent.system.startswith("Be terse.")


# -- edit-as-effect ----------------------------------------------------------


def _last_user(messages):
    for m in reversed(messages):
        if m["role"] == "user":
            return m["content"]
    return ""


def build_rule_agent():
    return Agent(
        model=RuleProvider(
            rules=[
                lambda ms: ModelResponse(text="celsius answer", stop_reason="end_turn")
                if "celsius" in _last_user(ms).lower()
                else None,
                lambda ms: ModelResponse(text="fahrenheit answer", stop_reason="end_turn"),
            ]
        )
    )


def test_fork_edit_is_recorded_and_replayed(tmp_path):
    agent = build_rule_agent()
    run = agent.run("Temperature in celsius?")
    branch = run.fork(
        at=0, edit=lambda ctx: setattr(ctx.items[0], "content", "Temperature?")
    )
    assert branch.output == "fahrenheit answer"
    assert [e.kind for e in branch.log] == ["edit", "model"]  # the surgery is in the trace

    # Replay rebuilds the EDITED context -- the fork is self-describing now.
    replayed = branch.replay()
    assert replayed.output == branch.output
    assert replayed.context.items[0].content == "Temperature?"

    # And it survives save/load across processes.
    path = str(tmp_path / "branch.loom.json")
    branch.save(path)
    loaded = Run.load(path, agent=agent)
    re2 = loaded.replay()
    assert re2.context.items[0].content == "Temperature?"
    assert re2.output == "fahrenheit answer"
