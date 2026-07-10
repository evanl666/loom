"""Dry-run, policy preview, plugin panels, .loomdebug artifact."""

import json

from loom import Agent, tool
from loom.debugger import DebugSession
from loom.providers import ModelResponse, ScriptedProvider, ToolCall


@tool(capabilities={"money_movement"})
def refund(order: str) -> str:
    "Refund."
    return f"REFUNDED {order}"


def _sess(tmp_path):
    prov = ScriptedProvider([
        ModelResponse(tool_calls=[ToolCall("1", "refund", {"order": "A"})], stop_reason="tool_use"),
        ModelResponse(text="done", stop_reason="end_turn"),
    ])
    agent = Agent(model=prov, tools=[refund])
    p = tmp_path / "t.loom.json"
    agent.run("refund order A").save(str(p))
    return DebugSession(str(p), agent=agent), p


def test_dry_run_runs_only_the_tool(tmp_path):
    sess, p = _sess(tmp_path)
    tool_step = next(e["seq"] for e in json.load(open(p))["log"] if e["kind"].startswith("tool:refund"))
    r = sess.dry_run(tool_step, {"order": "DIFFERENT"})
    assert r["result"] == "REFUNDED DIFFERENT"  # ran with the edited args, no model call
    assert "ms" in r


def test_policy_preview_flags_blocked_calls(tmp_path):
    sess, _ = _sess(tmp_path)
    r = sess.policy_preview(deny=["refund*"])
    assert r["this_run"] and r["this_run"][0]["tool"] == "refund"
    assert r["this_run"][0]["action"] == "deny"


def test_plugin_panels_from_a_pack(tmp_path):
    from loom.packs import Pack, register, unregister

    class _MyPack(Pack):
        name = "testpanel"
        def debugger_panels(self, action, trace):
            if action.type == "call":
                return [{"title": "custom", "text": f"panel for {action.tool}"}]
            return []

    register(_MyPack())
    try:
        sess, p = _sess(tmp_path)
        tool_step = next(e["seq"] for e in json.load(open(p))["log"] if e["kind"].startswith("tool:refund"))
        panels = sess.panels_for(tool_step)
        assert any(pl["title"] == "custom" for pl in panels)
    finally:
        unregister("testpanel")


def test_loomdebug_export_and_reload(tmp_path):
    sess, _ = _sess(tmp_path)
    sess.fork(at=1, append="try again")
    sess.comments.append({"step": 1, "text": "the refund", "label": "root-cause"})
    bundle = sess.export_session()
    assert bundle["loomdebug"] == 1 and bundle["branches"] and bundle["comments"]
    # write it and reload as a .loomdebug
    dbg = tmp_path / "s.loomdebug"
    dbg.write_text(json.dumps(bundle))
    reloaded = DebugSession(str(dbg))
    assert reloaded.branches and reloaded.comments
    assert reloaded.data == sess.data  # base trace restored
