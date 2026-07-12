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
    # the full conversation is inlined ONCE (the page filters by step client-side)
    assert sd["context_all"] and all("step" in m for m in sd["context_all"])


def test_static_page_gates_off_every_server_only_feature():
    """A frozen studio page must never reach a server. Every gate that hides
    fork / explain / copilot / blame / live-ask is OFF, every read feature's data
    is inlined, and every button wired unconditionally at load exists in the HTML
    (a missing id -> getElementById(null).onclick throws -> all later JS breaks)."""
    data = _multi_agent_trace()
    sd = static_data(data)
    assert sd["run"]["can_fork"] is False   # fork panel / fault-inject / autofix hidden
    assert sd["run"]["can_chat"] is False   # explain button + copilot not rendered
    assert sd["run"]["live"] is False       # live ask bar + polling off
    for k in ("run", "agents", "panels", "context_all", "rootcause", "branches"):
        assert k in sd                       # every read feature's data is inlined
    assert sd["branches"] == []              # no forking in a frozen file -> no branches

    html = static_page(data)
    for bid in ("copilot", "assertbtn", "export", "brk", "rootcause", "branches",
                "agentsbtn", "palettebtn", "swim", "play", "prompt", "model"):
        assert f'id="{bid}"' in html, f"#{bid} is wired at load but missing from the HTML"
    assert "!STATIC" in html                 # memory blame is gated off when static


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


def test_static_data_parses_the_log_a_bounded_number_of_times():
    """Regression: static_data used to call actions() once PER STEP (O(n^2), a
    300-step trace took seconds). It must parse the log a constant number of
    times regardless of trace length."""
    import loom.action as A
    from loom.debugger import static_data

    log = []
    for i in range(80):
        log.append({"seq": len(log), "kind": "model", "result": {"tool_calls": [
            {"id": str(i), "name": "search", "input": {"q": str(i)}}], "stop_reason": "tool_use"}})
        log.append({"seq": len(log), "kind": "tool:search", "result": "r" * 40})
    data = {"log": log, "prompt": "go", "output": "d", "tools": {"search": ["network"]}}

    orig = A.actions
    calls = {"n": 0}
    A.actions = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), orig(*a, **k))[1]
    try:
        sd = static_data(data)
    finally:
        A.actions = orig
    assert len(sd["run"]["steps"]) > 80
    assert calls["n"] <= 6, f"static_data re-parsed the log {calls['n']}x (should be O(1))"
    # correctness preserved: the conversation is inlined once, filterable by step
    assert sd["context_all"] and all("step" in m for m in sd["context_all"])


def test_static_html_size_is_linear_not_quadratic():
    """Regression: cumulative context frames inlined per step made the HTML
    O(n^2) in size (a 300-step run was ~23 MB). Inlining the conversation once
    keeps a 200-step run well under a couple MB."""
    from loom.debugger import static_page

    log = []
    for i in range(200):
        log.append({"seq": len(log), "kind": "model", "result": {"tool_calls": [
            {"id": str(i), "name": "search", "input": {"q": str(i)}}], "stop_reason": "tool_use"}})
        log.append({"seq": len(log), "kind": "tool:search", "result": "result data " * 30})
    data = {"log": log, "prompt": "go", "output": "d", "tools": {"search": ["network"]}}
    assert len(static_page(data)) < 2_000_000    # < 2 MB (was tens of MB)


def test_static_context_filter_matches_server_frame():
    from loom.debugger import context_at, static_data

    steps = []
    for i in range(6):
        steps.append({"seq": len(steps), "kind": "model", "result": {"tool_calls": [
            {"id": str(i), "name": "get", "input": {"i": i}}], "stop_reason": "tool_use"}})
        steps.append({"seq": len(steps), "kind": "tool:get", "result": f"r{i}"})
    data = {"log": steps, "prompt": "go", "output": "d", "tools": {"get": ["read"]}}
    allc = static_data(data)["context_all"]
    for st in (0, 3, 11):
        assert [m for m in allc if m["step"] <= st] == context_at(data, st)


def test_agent_frame_includes_follow_up_dialogue_turns_for_the_root():
    """The multi-agent context panel builds the frame client-side in agentFrame().
    It must fold EVERY plain user node (each follow-up ask), not only RUN.prompt,
    into the root agent's frame -- otherwise a live follow-up ('good morning!')
    never shows in the latest model turn's context. Lock the handling in place."""
    html = static_page(_multi_agent_trace())
    assert "function agentFrame" in html
    # a top-level agent (root OR peer) adopts dialogue turns; a delegated sub-agent
    # sees only its task -- keyed on "was delegated to", so peers show the dialogue.
    assert 'x.type==="user"&&!x.injected' in html
    assert 'if(!isSub) frame.push({role:"user"' in html
    assert "const isSub=!!deleg;" in html
