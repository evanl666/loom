"""First-bad-step, context delta, branch tree, auto-fix."""

import json

from loom import Agent, tool
from loom.debugger import DebugSession, context_delta
from loom.providers import ModelResponse, ScriptedProvider, ToolCall
from loom.rootcause import first_bad_step


def _exfil_trace():
    secret = "sk-ant-api03-" + "Z" * 40
    return {"log": [
        {"seq": 0, "kind": "model", "result": {"tool_calls": [{"id": "1", "name": "read_secret", "input": {}}], "stop_reason": "tool_use"}},
        {"seq": 1, "kind": "tool:read_secret", "result": f"KEY={secret}"},
        {"seq": 2, "kind": "model", "result": {"tool_calls": [{"id": "2", "name": "http_post", "input": {"data": f"KEY={secret}"}}], "stop_reason": "tool_use"}},
        {"seq": 3, "kind": "tool:http_post", "result": "200"}],
        "prompt": "x", "output": "d", "tools": {"http_post": ["network"], "read_secret": ["secret"]}}


def test_first_bad_step_finds_the_leak_source():
    r = first_bad_step(_exfil_trace())
    assert r["found"] and r["step"] == 1  # the secret read is the root cause
    assert any("leak" in s for s in r["signals"])
    assert r["cascade"]  # downstream actions listed


def test_context_delta_flags_untrusted_and_dominant():
    d = context_delta(_exfil_trace(), 3)
    assert d["items"]
    assert d["dominant"] is not None


class _Echo:
    model = "echo"
    def complete(self, system, messages, tools):
        n = sum(1 for m in messages if m.get("role") == "assistant")
        if n == 0:
            return ModelResponse(tool_calls=[ToolCall("1", "get", {})], stop_reason="tool_use")
        return ModelResponse(text="answer", stop_reason="end_turn")


def test_fork_records_a_branch_node(tmp_path):
    @tool
    def get() -> str:
        "get"
        return "data"
    agent = Agent(model=_Echo(), tools=[get])
    p = tmp_path / "t.loom.json"
    agent.run("go").save(str(p))
    sess = DebugSession(str(p), agent=agent)
    assert sess.branches == []
    res = sess.fork(at=1, append="try again")
    assert len(sess.branches) == 1
    node = sess.branches[0]
    assert node["id"] == 1 and node["label"] == "try again" and "score" in node
    assert res["branch_id"] == 1
