"""Failure paths, deterministically. Real agents error, get firewalled, loop, or
blow a budget -- the debugger and analyzers must record each failure, point at the
FIRST bad step, and survive it. (Most other tests exercise the happy path.)"""
import pytest

from loom import Agent, Policy, tool
from loom.action import actions
from loom.autopsy import autopsy_html
from loom.debugger import steps_for
from loom.diagnose import diagnose
from loom.diff import score_breakdown
from loom.incident import build_report
from loom.providers import ModelResponse, ScriptedProvider, ToolCall
from loom.rootcause import first_bad_step


@tool
def flaky(x: int) -> str:
    "A tool that raises."
    raise ValueError(f"boom on {x}")


@tool(capabilities={"money_movement"})
def issue_refund(amt: int) -> str:
    "Refund money."
    return f"refunded {amt}"


def _tool_error_run():
    prov = ScriptedProvider([
        ModelResponse(tool_calls=[ToolCall("t1", "flaky", {"x": 5})], stop_reason="tool_use"),
        ModelResponse(text="couldn't recover", stop_reason="end_turn")])
    return Agent(model=prov, tools=[flaky]).run("do it")


def _firewall_block_run():
    prov = ScriptedProvider([
        ModelResponse(tool_calls=[ToolCall("t1", "issue_refund", {"amt": 9999})], stop_reason="tool_use"),
        ModelResponse(text="ok", stop_reason="end_turn")])
    return Agent(model=prov, tools=[issue_refund], policy=Policy(deny=["issue_refund*"])).run("refund")


def test_tool_error_is_recorded_and_root_caused():
    run = _tool_error_run()
    data = run.to_dict()
    # the error effect is in the log, verbatim
    errs = [e for e in data["log"] if str(e.get("result", "")).startswith("ERROR")]
    assert errs and "boom on 5" in errs[0]["result"]
    # rootcause points at the FIRST bad step and calls it a failure
    rc = first_bad_step(data)
    assert rc["found"] and rc["kind"] == "failure" and rc["tool"] == "flaky"
    # diagnose categorizes it
    dg = diagnose(data)
    assert dg["failure"] == "tool-error" and dg["fix_category"] == "tool"
    # replay is still byte-identical (a failure trace replays like any other)
    assert run.replay().output == run.output
    # the debugger shows the errored call
    assert any(s.get("type") == "call" and s.get("tool") == "flaky" for s in steps_for(data))


def test_firewall_block_is_recorded_and_root_caused():
    data = _firewall_block_run().to_dict()
    blocked = [e for e in data["log"] if str(e.get("result", "")).startswith("BLOCKED")]
    assert blocked, "the denied tool call must be recorded as BLOCKED"
    rc = first_bad_step(data)
    assert rc["found"] and rc["kind"] == "failure"
    # a blocked money-movement call is a firewall signal, not a silent success
    assert any("issue_refund" in str(e.get("result", "")) for e in data["log"])


@pytest.mark.parametrize("run_fn", [_tool_error_run, _firewall_block_run])
def test_reports_survive_failure_traces(run_fn):
    """autopsy / incident / score / actions must not crash on a failed run."""
    data = run_fn().to_dict()
    actions(data)
    autopsy_html(data)
    build_report(data, "x.loom.json")
    sb = score_breakdown(data)
    assert 0 <= sb["overall"] <= 100


def test_subagent_failure_is_attributed_to_the_subagent():
    """A failure inside a delegated sub-agent must root-cause to the sub-agent's
    step, not the coordinator's."""
    SUB = "You are the Worker. Use flaky."
    COORD = "You are the Coordinator. Delegate to the worker."
    import hashlib
    def h(s):
        return hashlib.sha1(s.encode()).hexdigest()[:12]
    data = {"recorded_via": "proxy", "episodes": ["go"], "output": "done",
            "systems": {h(COORD): COORD, h(SUB): SUB}, "model": "m", "log": [
                {"seq": 0, "kind": "model", "depth": 0,
                 "meta": {"sys_hash": h(COORD), "sys_head": COORD, "tools": ["ask_worker"]},
                 "result": {"tool_calls": [{"id": "d1", "name": "ask_worker", "input": {}}],
                            "stop_reason": "tool_use"}},
                {"seq": 1, "kind": "tool:ask_worker", "depth": 0, "result": "ok"},
                {"seq": 2, "kind": "model", "depth": 0,
                 "meta": {"sys_hash": h(SUB), "sys_head": SUB, "tools": ["flaky"]},
                 "result": {"tool_calls": [{"id": "f1", "name": "flaky", "input": {}}],
                            "stop_reason": "tool_use"}},
                {"seq": 3, "kind": "tool:flaky", "depth": 0, "result": "ERROR: boom", "meta": {"tuid": "f1"}},
                {"seq": 4, "kind": "model", "depth": 0,
                 "meta": {"sys_hash": h(SUB), "sys_head": SUB, "tools": ["flaky"]},
                 "result": {"text": "failed", "stop_reason": "end_turn"}},
            ]}
    from loom.multiagent import infer_agents
    rc = first_bad_step(data)
    assert rc["found"] and rc["tool"] == "flaky" and rc["step"] == 3
    # the failing step belongs to the Worker sub-agent, not the Coordinator
    ia = infer_agents(data)
    worker = next(a for a in ia["agents"] if "Worker" in a["label"])
    assert ia["step_agent"].get("2") == worker["id"]     # the flaky-calling model turn


@tool
def spin(q: str) -> str:
    "A tool that always returns the same thing (agent gets stuck)."
    return "same result"


def test_a_loop_is_detected_and_root_caused():
    """An agent that calls the same tool with the same args over and over is a
    loop -- rootcause must flag it as a failure, not report a clean run."""
    resps = [ModelResponse(tool_calls=[ToolCall(f"t{i}", "spin", {"q": "x"})], stop_reason="tool_use")
             for i in range(12)]
    resps.append(ModelResponse(text="done", stop_reason="end_turn"))
    data = Agent(model=ScriptedProvider(resps), tools=[spin]).run("go").to_dict()
    rc = first_bad_step(data)
    assert rc["found"] and rc["kind"] == "failure"
    assert any("loop" in s.lower() for s in rc["signals"])
