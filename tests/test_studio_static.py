"""Studio = the interactive debugger UI frozen into a self-contained file."""

import json

from loom import Agent, tool
from loom.debugger import static_data, static_page
from loom.providers import ModelResponse, ScriptedProvider, ToolCall


@tool
def search(q: str) -> str:
    "Search."
    return f"result for {q}"


def _multi_agent_trace():
    researcher = Agent(model=ScriptedProvider([
        ModelResponse(tool_calls=[ToolCall("s", "search", {"q": "x"})], stop_reason="tool_use"),
        ModelResponse(text="done", stop_reason="end_turn")]), tools=[search], name="researcher")
    lead = Agent(model=ScriptedProvider([
        ModelResponse(tool_calls=[ToolCall("d", "researcher", {"task": "t"})], stop_reason="tool_use"),
        ModelResponse(text="final", stop_reason="end_turn")]),
        tools=[researcher.as_tool()], name="lead")
    return lead.run("go").to_dict()


def test_static_data_inlines_what_the_page_fetches():
    sd = static_data(_multi_agent_trace())
    assert sd["run"]["steps"] and sd["run"]["can_fork"] is False and sd["run"]["live"] is False
    # multi-agent recovery is inlined
    assert sd["agents"]["multi"] and {a["label"] for a in sd["agents"]["agents"]} == {"lead", "researcher"}
    # per-step context frames are precomputed for every real step
    for s in sd["run"]["steps"]:
        if s.get("step", -1) >= 0:
            assert str(s["step"]) in sd["context"]


def test_static_page_is_self_contained_and_is_the_debugger():
    html = static_page(_multi_agent_trace())
    assert "window.LOOM_STATIC=" in html          # data inlined
    assert "const STATIC=" in html                # static branch
    assert "function renderTree" in html          # the debugger's tree UI
    assert "function showAgents" in html          # agents overview
    assert "127.0.0.1" not in html                # no server/localhost dependency
    # the inlined blob is valid JSON
    blob = html.split("window.LOOM_STATIC=", 1)[1].split(";</script>", 1)[0]
    json.loads(blob.replace("<\\/", "</"))


def test_static_page_survives_a_hostile_trace():
    # a trace with a </script> in its content must not break out of the inline blob
    data = {"log": [{"seq": 0, "kind": "model",
                     "result": {"text": "</script><b>x", "stop_reason": "end_turn"}}],
            "prompt": "</script>", "output": "</script>", "tools": {}}
    html = static_page(data)
    assert "</script><b>x" not in html  # escaped as <\/script>
