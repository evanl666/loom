"""Policy rollout lifecycle: draft -> canary -> enforce, gated by breakages."""

import json

import pytest

from loom import Agent, tool
from loom.providers import ModelResponse, ScriptedProvider, ToolCall
from loom.rollout import assess, promote, read_stage, rollback


@tool(capabilities={"money_movement"})
def issue_refund(order_id: str, amount: float) -> str:
    "Refund."
    return "REFUNDED"


def _corpus(tmp_path):
    # a COMPLETED run that used issue_refund -> a policy denying it would break it
    prov = ScriptedProvider([
        ModelResponse(tool_calls=[ToolCall("t", "issue_refund", {"order_id": "1", "amount": 9})], stop_reason="tool_use"),
        ModelResponse(text="done", stop_reason="end_turn"),
    ])
    run = Agent(model=prov, tools=[issue_refund]).run("refund it")
    run.save(str(tmp_path / "r.loom.json"))
    pol = tmp_path / "p.yml"
    pol.write_text("default: allow\ndeny:\n  - issue_refund*\n")
    return str(pol), [str(tmp_path)]


def test_assess_flags_breakage_and_gates_enforce(tmp_path):
    pol, traces = _corpus(tmp_path)
    st = assess(pol, traces)
    assert st["stage"] == "draft" and st["next_stage"] == "canary"
    assert st["breakages"]  # the completed refund run would break
    assert st["gate_ok"] is True  # -> canary is always safe


def test_lifecycle_promote_gate_force_rollback(tmp_path):
    pol, traces = _corpus(tmp_path)
    assert promote(pol, traces, by="alice")["stage"] == "canary"
    # canary -> enforce is gated because a completed run would be denied
    with pytest.raises(ValueError, match="refusing to promote"):
        promote(pol, traces, by="alice")
    forced = promote(pol, traces, by="alice", force=True)
    assert forced["stage"] == "enforce" and forced["forced"]
    assert read_stage(pol)["stage"] == "enforce"
    assert rollback(pol)["stage"] == "canary"
