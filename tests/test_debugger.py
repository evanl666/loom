"""The interactive step-debugger: steps_for + server API + live fork callback.

Uses a ScriptedProvider so the fork is deterministic and needs no API key.
"""
import json
import threading
import urllib.request

from loom import Agent, tool
from loom.debugger import DebugServer, DebugSession, steps_for
from loom.providers import ModelResponse, ScriptedProvider, ToolCall


@tool
def add(a: int, b: int) -> int:
    "Add."
    return a + b


def _script():
    return ScriptedProvider([
        ModelResponse(tool_calls=[ToolCall("t1", "add", {"a": 2, "b": 3})], stop_reason="tool_use"),
        ModelResponse(text="the answer is 5", stop_reason="end_turn"),
        # extra responses the fork's live tail will consume after the edit:
        ModelResponse(text="revised: 5 (noted)", stop_reason="end_turn"),
    ])


def _make_trace(tmp_path):
    run = Agent(model=_script(), tools=[add]).run("what is 2+3?")
    p = tmp_path / "t.loom.json"
    run.save(str(p))
    return str(p)


def test_steps_for_lifts_actions(tmp_path):
    data = json.load(open(_make_trace(tmp_path)))
    steps = steps_for(data)
    assert steps and any(s["type"] == "call" and s.get("tool") == "add" for s in steps)
    assert any((s.get("replay") or {}).get("forkable") for s in steps)


def test_debug_session_fork_runs_the_edited_tail(tmp_path):
    path = _make_trace(tmp_path)
    agent = Agent(model=_script(), tools=[add])
    sess = DebugSession(path, agent=agent)
    res = sess.fork(at=1, append="please revise")
    # the fork replays turn 0 from the log for free, applies the context edit,
    # and runs the tail live -- producing a branch with steps + an output.
    assert res["branch_output"] and len(res["branch_steps"]) >= 1
    assert "diverge" in res and res["forked_at"] == 1


def test_debug_server_serves_page_and_api(tmp_path):
    server = DebugServer(_make_trace(tmp_path), agent=None, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{server.port}"
        page = urllib.request.urlopen(base + "/", timeout=5).read().decode()
        assert "Loom debugger" in page
        run = json.load(urllib.request.urlopen(base + "/api/run", timeout=5))
        assert run["can_fork"] is False and len(run["steps"]) >= 2
    finally:
        server.shutdown()


def test_context_at_reconstructs_the_frame(tmp_path):
    from loom.debugger import context_at

    path = _make_trace(tmp_path)
    import json
    data = json.load(open(path))
    frame = context_at(data, 99)  # whole run
    roles = [m["role"] for m in frame]
    assert roles[0] == "user"  # the prompt
    assert "assistant" in roles and "tool" in roles  # the add call + its result
    # a mid-step frame is a prefix (never shows later steps)
    early = context_at(data, 0)
    assert len(early) <= len(frame)


def test_context_endpoint(tmp_path):
    import json
    import threading
    import urllib.request

    server = DebugServer(_make_trace(tmp_path), agent=None, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        r = json.load(urllib.request.urlopen(
            f"http://127.0.0.1:{server.port}/api/context?step=0", timeout=5))
        assert r["frame"] and r["frame"][0]["role"] == "user"
    finally:
        server.shutdown()


def test_copilot_report_points_at_suspicious_steps(tmp_path):
    from loom import Agent, tool
    from loom.debugger import copilot_report
    from loom.providers import ModelResponse, ScriptedProvider, ToolCall

    @tool(capabilities={"money_movement"})
    def refund(x: int) -> str:
        "Refund."
        return "ok"

    prov = ScriptedProvider([
        ModelResponse(tool_calls=[ToolCall("t", "refund", {"x": 1})], stop_reason="tool_use"),
        ModelResponse(text="done", stop_reason="end_turn"),
    ])
    data = Agent(model=prov, tools=[refund]).run("do it").to_dict()
    rep = copilot_report(data)
    assert any(s["tool"] == "refund" for s in rep["suspicious"])
    assert "refund*" in rep["policy_suggestion"]
    assert rep["fork_edits"] and rep["summary"]


def test_copilot_endpoint(tmp_path):
    import json
    import threading
    import urllib.request

    server = DebugServer(_make_trace(tmp_path), agent=None, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        r = json.load(urllib.request.urlopen(
            f"http://127.0.0.1:{server.port}/api/copilot", timeout=5))
        assert "summary" in r and "grade" in r
    finally:
        server.shutdown()
