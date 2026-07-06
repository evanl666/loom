"""Context tracks provenance and trims to a token budget without orphaning tools."""

from loom import Context
from loom.providers import ModelResponse, ToolCall


def test_provenance_records_sources():
    ctx = Context(system="sys")
    ctx.add_user("hello")
    ctx.add_assistant(ModelResponse(text="hi", tool_calls=[ToolCall("t1", "add", {"a": 1})]))
    ctx.add_tool_result("t1", "add", "1")
    sources = [p["source"] for p in ctx.provenance()]
    assert sources == ["user", "model", "tool:add"]


def test_pinned_user_survives_budget_trim():
    ctx = Context(budget=5)  # ~5 tokens total
    ctx.add_user("PINNED SYSTEM CONTEXT " * 2, pinned=True)
    for i in range(10):
        ctx.add_user(f"chatter message number {i} " * 2)
    msgs = ctx.messages()
    # The pinned item must still be present after trimming.
    assert any("PINNED SYSTEM CONTEXT" in m["content"] for m in msgs)


def test_trim_does_not_orphan_tool_results():
    ctx = Context(budget=3)
    ctx.add_user("old question " * 5)
    ctx.add_assistant(ModelResponse(text="", tool_calls=[ToolCall("t1", "add", {})]))
    ctx.add_tool_result("t1", "add", "result " * 5)
    ctx.add_user("new question")
    msgs = ctx.messages()
    # A tool message must never be the first message (it would be an API error).
    assert not msgs or msgs[0]["role"] != "tool"
