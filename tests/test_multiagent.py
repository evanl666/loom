"""Multi-agent attribution: native names + wire-fingerprint recovery."""

import hashlib

from loom import Agent, tool
from loom.action import actions
from loom.multiagent import best_role, infer_agents
from loom.providers import ModelResponse, ScriptedProvider, ToolCall


@tool
def search(q: str) -> str:
    "Search."
    return f"result for {q}"


def _lead():
    researcher = Agent(model=ScriptedProvider([
        ModelResponse(tool_calls=[ToolCall("s1", "search", {"q": "loom"})], stop_reason="tool_use"),
        ModelResponse(text="Loom is an agent harness.", stop_reason="end_turn")]),
        tools=[search], name="researcher")
    return Agent(model=ScriptedProvider([
        ModelResponse(tool_calls=[ToolCall("d1", "researcher", {"task": "x"})], stop_reason="tool_use"),
        ModelResponse(text="Per research: harness.", stop_reason="end_turn")]),
        tools=[researcher.as_tool()], name="lead")


def test_native_names_and_delegation_step():
    data = _lead().run("Explain Loom.").to_dict()
    # the delegation call is no longer dropped
    tools = [a.tool for a in actions(data) if a.type == "call"]
    assert "researcher" in tools and "search" in tools
    ia = infer_agents(data)
    assert ia["multi"] and ia["source"] == "native"
    labels = {a["label"] for a in ia["agents"]}
    assert labels == {"lead", "researcher"}
    # the parent delegates to the child
    assert any(e for e in ia["edges"])
    # each step maps to an agent
    assert ia["step_agent"]


def _fp(system, tools):
    return {"sys_hash": hashlib.sha1(system.encode()).hexdigest()[:12],
            "sys_head": system, "tools": tools, "model": "claude"}


def test_wire_fingerprint_recovers_two_agents():
    SUP = "You are the supervisor. Delegate to workers."
    COD = "You are the coder worker. Write code."
    data = {"recorded_via": "proxy", "model": "claude", "output": "done", "tools": {}, "log": [
        {"seq": 0, "kind": "model", "depth": 0, "meta": _fp(SUP, ["delegate"]),
         "result": {"tool_calls": [{"id": "1", "name": "delegate", "input": {}}], "stop_reason": "tool_use"}},
        {"seq": 1, "kind": "tool:delegate", "depth": 0, "result": "ok"},
        {"seq": 2, "kind": "model", "depth": 0, "meta": _fp(COD, ["write_file"]),
         "result": {"tool_calls": [{"id": "2", "name": "write_file", "input": {}}], "stop_reason": "tool_use"}},
        {"seq": 3, "kind": "tool:write_file", "depth": 0, "result": "ok"},
        {"seq": 4, "kind": "model", "depth": 0, "meta": _fp(COD, ["write_file"]),
         "result": {"text": "done", "stop_reason": "end_turn"}},
    ]}
    ia = infer_agents(data)
    assert ia["multi"] and ia["source"] == "wire"
    labels = [a["label"] for a in ia["agents"]]
    assert "supervisor" in labels and "coder worker" in labels
    # the coder's tool call is attributed to the coder, not the supervisor
    assert ia["step_agent"]["3"] == ia["step_agent"]["2"]
    assert ia["step_agent"]["0"] != ia["step_agent"]["2"]


def test_best_role_finds_specific_role_past_generic_preamble():
    # a framework preamble ("You are Claude Code...") must not mask the sub-agent's
    # real role stated later in the same (possibly long) system prompt.
    assert best_role("You are Claude Code. [800 chars] You are a Landmark Researcher. "
                     "Be terse.") == "Landmark Researcher"
    assert best_role("You are the Supervisor. Delegate to workers.") == "Supervisor"
    assert best_role("You are a Math Specialist. Use calculate.") == "Math Specialist"
    # generic identities give no label (caller falls back to "agent N")
    assert best_role("You are a helpful assistant.") is None
    assert best_role("You are Claude, made by Anthropic.") is None


def test_single_agent_is_not_multi():
    @tool
    def get() -> str:
        "get"
        return "x"
    a = Agent(model=ScriptedProvider([
        ModelResponse(tool_calls=[ToolCall("1", "get", {})], stop_reason="tool_use"),
        ModelResponse(text="ok", stop_reason="end_turn")]), tools=[get], name="solo")
    ia = infer_agents(a.run("go").to_dict())
    assert ia["multi"] is False
