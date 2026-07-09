"""The agent loop actually runs tools and produces output (offline, deterministic)."""

from loom import Agent, tool
from loom.providers import ModelResponse, ScriptedProvider, ToolCall


@tool
def add(a: int, b: int) -> int:
    "Add two numbers."
    return a + b


def make_agent():
    provider = ScriptedProvider(
        [
            ModelResponse(
                tool_calls=[ToolCall(id="t1", name="add", input={"a": 2, "b": 3})],
                stop_reason="tool_use",
                usage={"input_tokens": 10, "output_tokens": 5},
            ),
            ModelResponse(
                text="The answer is 5.",
                stop_reason="end_turn",
                usage={"input_tokens": 20, "output_tokens": 6},
            ),
        ]
    )
    return Agent(model=provider, tools=[add], system="You are precise.")


def test_agent_runs_tool_and_answers():
    run = make_agent().run("What is 2 + 3?")
    assert run.output == "The answer is 5."
    assert not run.truncated
    assert run.num_turns == 2


def test_tool_result_recorded_and_in_context():
    run = make_agent().run("What is 2 + 3?")
    tool_entries = [e for e in run.log if e.kind.startswith("tool:")]
    assert len(tool_entries) == 1
    assert tool_entries[0].result == 5
    sources = [p["source"] for p in run.context.provenance()]
    assert "tool:add" in sources


def test_cost_aggregates_usage():
    run = make_agent().run("What is 2 + 3?")
    assert run.cost() == {"input_tokens": 30, "output_tokens": 11, "total_tokens": 41}


def test_timeline_shape():
    run = make_agent().run("What is 2 + 3?")
    tl = run.timeline()
    assert tl[0]["kind"] == "model"
    assert "add" in tl[0]["detail"]
    assert tl[1]["kind"] == "tool:add"
    assert tl[2]["detail"] == "The answer is 5."


def test_unknown_tool_is_handled_gracefully():
    provider = ScriptedProvider(
        [
            ModelResponse(
                tool_calls=[ToolCall(id="x", name="nope", input={})],
                stop_reason="tool_use",
            ),
            ModelResponse(text="done", stop_reason="end_turn"),
        ]
    )
    run = Agent(model=provider, tools=[add]).run("go")
    tool_entry = [e for e in run.log if e.kind.startswith("tool:")][0]
    assert "unknown tool" in tool_entry.result


def test_run_with_empty_prompt_list_is_a_clear_error():
    import pytest
    from loom import Agent
    from loom.providers import ModelResponse, ScriptedProvider

    with pytest.raises(ValueError, match="at least one prompt"):
        Agent(model=ScriptedProvider([ModelResponse(text="hi")])).run([])
