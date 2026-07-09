"""The built-in Coding Pack -- coding expressed through the generic Pack API."""

from loom import actions
from loom.packs import packs, pack_for, undo_plan


def test_coding_pack_is_registered_by_default():
    assert "coding" in [p.name for p in packs()]


def _edit_trace(status="M"):
    return {
        "log": [
            {"seq": 0, "kind": "model", "key": "k",
             "result": {"text": "fixing the bug",
                        "tool_calls": [{"id": "t", "name": "Edit",
                                        "input": {"file_path": "src/app.py", "old": "a", "new": "b"}}],
                        "stop_reason": "tool_use", "usage": {}}},
            {"seq": 1, "kind": "tool:Edit", "key": "k2", "result": "edited"},
        ],
        "workspace": {"changes": {"files": [
            {"path": "src/app.py", "status": status, "pre_existing": False}]}},
    }


def test_file_edit_gets_a_per_step_state_diff():
    call = [a for a in actions(_edit_trace()) if a.type == "call"][0]
    assert call.state_diff.kind == "file"
    assert call.state_diff.summary == "wrote src/app.py"
    assert call.state_diff.detail["path"] == "src/app.py"


def test_modified_file_undo_is_git_checkout():
    call = [a for a in actions(_edit_trace("M")) if a.type == "call"][0]
    plan = undo_plan(call, _edit_trace("M"))
    assert plan.kind == "revert" and plan.reversible is True
    assert plan.commands == ["git checkout HEAD -- src/app.py"]


def test_created_file_undo_removes_it():
    plan = undo_plan([a for a in actions(_edit_trace("A")) if a.type == "call"][0],
                     _edit_trace("A"))
    assert plan.commands == ["rm src/app.py"]


def _bash_trace(cmd):
    return {"log": [
        {"seq": 0, "kind": "model", "key": "k",
         "result": {"text": "x", "tool_calls": [{"id": "t", "name": "Bash",
                    "input": {"command": cmd}}], "stop_reason": "tool_use", "usage": {}}},
        {"seq": 1, "kind": "tool:Bash", "key": "k2", "result": "ok"},
    ]}


def test_destructive_shell_gets_manual_review_undo_not_a_lie():
    call = [a for a in actions(_bash_trace("rm -rf build/")) if a.type == "call"][0]
    plan = undo_plan(call, _bash_trace("rm -rf build/"))
    assert plan.kind == "noop" and plan.reversible is False
    assert call.state_diff.summary.startswith("destructive shell")


def test_plain_shell_read_has_nothing_to_undo():
    call = [a for a in actions(_bash_trace("ls -la")) if a.type == "call"][0]
    assert undo_plan(call, _bash_trace("ls -la")) is None


def test_coding_pack_ignores_business_actions():
    # A refund is not the coding pack's concern -- that's the Support pack's.
    from loom.packs.coding import CodingPack

    refund = {"log": [
        {"seq": 0, "kind": "model", "key": "k",
         "result": {"text": "x", "tool_calls": [{"id": "t", "name": "issue_refund",
                    "input": {"amount": 9}}], "stop_reason": "tool_use", "usage": {}}},
        {"seq": 1, "kind": "tool:issue_refund", "key": "k2", "result": "ok"},
    ]}
    call = [a for a in actions(refund) if a.type == "call"][0]
    assert CodingPack().owns(call) is False
