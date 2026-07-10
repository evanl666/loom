"""RemoteAgent: record a black-box remote (HTTP/gRPC) call as one Loom action."""

from loom.action import actions
from loom.debugger import steps_for
from loom.packs import install_builtin, pack_for, undo_plan
from loom.remote import RemoteAgent


def _ra():
    return RemoteAgent("planner", call=lambda p: f"plan for {p}",
                       transport="grpc", endpoint="127.0.0.1:50051")


def test_record_captures_the_call_as_one_action():
    trace = _ra().record("launch")
    calls = [s for s in steps_for(trace) if s["type"] == "call"]
    assert len(calls) == 1
    c = calls[0]
    assert c["tool"] == "remote_planner"
    assert set(c["capabilities"]) >= {"network", "remote_agent"}
    assert c["input"] == {"prompt": "launch"}
    assert trace["output"] == "plan for launch"
    assert trace["recorded_via"] == "remote" and trace["transport"] == "grpc"


def test_as_tool_carries_remote_capabilities():
    t = _ra().as_tool()
    assert t.name == "remote_planner"
    assert t.capabilities and {"network", "remote_agent"} <= set(t.capabilities)
    assert t(prompt="x") == "plan for x"


def test_pack_owns_and_advises_remote_calls():
    install_builtin()
    trace = _ra().record("launch")
    call = next(a for a in actions(trace) if a.type == "call")
    p = pack_for(call)
    assert p is not None and p.name == "remote"
    assert call.state_diff and "remote agent" in call.state_diff.summary
    plan = undo_plan(call, trace)
    assert plan is not None and plan.reversible is False  # can't undo a remote effect


def test_record_is_replayable_offline():
    # the recorded response serves the call -- a second read never hits the network
    calls = {"n": 0}

    def once(p):
        calls["n"] += 1
        return "remote said hi"

    trace = RemoteAgent("svc", call=once).record("hello")
    assert calls["n"] == 1
    # re-deriving actions from the saved trace makes no further calls
    _ = actions(trace)
    assert calls["n"] == 1
