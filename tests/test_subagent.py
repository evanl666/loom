"""Subagents: isolated context, nested trace, and replay/fork through delegation."""

from loom import Agent, tool
from loom.providers import ModelResponse, ScriptedProvider, ToolCall


@tool
def search(q: str) -> str:
    "Search a tiny knowledge base."
    return f"result for {q}"


def build_lead():
    # Child researcher: search once, then summarize.
    researcher = Agent(
        model=ScriptedProvider(
            [
                ModelResponse(
                    tool_calls=[ToolCall("s1", "search", {"q": "loom"})], stop_reason="tool_use"
                ),
                ModelResponse(text="Loom is an agent harness.", stop_reason="end_turn"),
            ]
        ),
        tools=[search],
        name="researcher",
    )
    # Parent lead: delegate to the researcher, then answer.
    lead = Agent(
        model=ScriptedProvider(
            [
                ModelResponse(
                    tool_calls=[ToolCall("d1", "researcher", {"task": "what is loom"})],
                    stop_reason="tool_use",
                ),
                ModelResponse(text="Per research: Loom is an agent harness.", stop_reason="end_turn"),
            ]
        ),
        tools=[researcher.as_tool()],
        name="lead",
    )
    return lead


def test_subagent_delegates_and_answers():
    run = build_lead().run("Explain Loom.")
    assert run.output == "Per research: Loom is an agent harness."
    # Two top-level turns; more model calls total because of the subagent.
    assert run.num_turns == 2
    assert run.num_model_calls == 4  # 2 parent + 2 child


def test_subagent_effects_are_nested_in_trace():
    run = build_lead().run("Explain Loom.")
    depths = {e.kind: e.depth for e in run.log if e.kind == "model"}
    # There are model calls at both depth 0 and depth 1.
    all_depths = sorted({e.depth for e in run.log if e.kind == "model"})
    assert all_depths == [0, 1]
    # The child's search tool ran at depth 1.
    search_entry = [e for e in run.log if e.kind == "tool:search"][0]
    assert search_entry.depth == 1


def test_subagent_context_is_isolated():
    run = build_lead().run("Explain Loom.")
    # The parent's context must NOT contain the child's internal search result;
    # it only sees the delegated tool result.
    parent_sources = [p["source"] for p in run.context.provenance()]
    assert "tool:researcher" in parent_sources
    assert "tool:search" not in parent_sources


def test_replay_through_subagent_is_deterministic_and_free():
    run = build_lead().run("Explain Loom.")
    replay = run.replay()
    assert replay.output == run.output
    assert replay.num_model_calls == run.num_model_calls
