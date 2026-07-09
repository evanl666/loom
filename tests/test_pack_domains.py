"""The opt-in domain packs: SQL, Browser, Support."""

import pytest

from loom import actions
from loom.packs import register, undo_plan, unregister
from loom.packs.browser import BrowserPack
from loom.packs.sql import SqlPack
from loom.packs.support import SupportPack


@pytest.fixture
def domain_packs():
    register(SqlPack())
    register(BrowserPack())
    register(SupportPack())
    yield
    for name in ("sql", "browser", "support"):
        unregister(name)


def _trace(tool, tool_input, result="ok"):
    return {"log": [
        {"seq": 0, "kind": "model", "key": "k",
         "result": {"text": "doing it", "tool_calls": [
             {"id": "t", "name": tool, "input": tool_input}],
             "stop_reason": "tool_use", "usage": {}}},
        {"seq": 1, "kind": f"tool:{tool}", "key": "k2", "result": result},
    ]}


def _call(trace):
    return [a for a in actions(trace) if a.type == "call"][0]


# -- SQL ---------------------------------------------------------------------

def test_sql_select_with_pii_is_flagged(domain_packs):
    a = _call(_trace("run_query", {"query": "SELECT ssn, email FROM customers"}))
    assert "pii_access" in a.capabilities
    assert a.state_diff is None  # a read changes nothing


def test_sql_insert_gets_state_diff_and_compensating_undo(domain_packs):
    tr = _trace("run_query", {"query": "INSERT INTO orders VALUES (1)"}, "1 row inserted")
    a = _call(tr)
    assert "database_write" in a.capabilities
    assert a.state_diff.kind == "database"
    assert a.state_diff.summary == "INSERT on orders (1 rows)"
    plan = undo_plan(a, tr)
    assert plan.kind == "compensate" and "DELETE FROM orders" in plan.commands[0]


def test_sql_drop_is_honestly_irreversible(domain_packs):
    tr = _trace("db_exec", {"sql": "DROP TABLE users"})
    a = _call(tr)
    assert "destructive" in a.capabilities
    plan = undo_plan(a, tr)
    assert plan.reversible is False and "backup" in plan.summary


# -- Browser -----------------------------------------------------------------

def test_browser_submit_is_external_and_unsubmittable(domain_packs):
    tr = _trace("click", {"selector": "#buy-now"})
    a = _call(tr)
    assert "browser_submit" in a.capabilities
    assert "external_side_effect" in a.capabilities
    assert a.state_diff.summary == "submitted #buy-now"
    plan = undo_plan(a, tr)
    assert plan.reversible is False and "cannot be unsubmitted" in plan.summary


def test_browser_navigation_is_reversible(domain_packs):
    tr = _trace("navigate", {"url": "https://example.com/checkout"})
    a = _call(tr)
    assert a.state_diff.summary == "navigated to https://example.com/checkout"
    assert undo_plan(a, tr).kind == "revert"


def test_browser_dom_snapshots_become_a_real_diff(domain_packs):
    tr = _trace("click", {"selector": "#add"},
                {"dom_before": "<ul></ul>", "dom_after": "<ul><li>x</li></ul>"})
    a = _call(tr)
    assert "DOM 9 -> 19 chars" in a.state_diff.summary
    assert a.state_diff.detail["dom_after"] == "<ul><li>x</li></ul>"


# -- Support -----------------------------------------------------------------

def test_refund_state_diff_and_compensation(domain_packs):
    tr = _trace("issue_refund", {"amount": 50, "order_id": "A-17"})
    a = _call(tr)
    assert "money_movement" in a.capabilities  # from the core taxonomy
    assert a.state_diff.summary == "moved money: 50 (A-17)"
    plan = undo_plan(a, tr)
    assert plan.kind == "compensate" and plan.reversible is False


def test_sent_email_cannot_be_unsent(domain_packs):
    tr = _trace("send_email", {"to": "jane@x.com", "body": "hi"})
    a = _call(tr)
    assert a.state_diff.summary == "messaged jane@x.com"
    assert undo_plan(a, tr).reversible is False


def test_crm_field_write_reverts_only_with_prior_value(domain_packs):
    with_old = _trace("update_field", {"record_id": "C-9", "field": "tier",
                                       "value": "free", "old": "pro"})
    a = _call(with_old)
    assert a.state_diff.summary == "updated C-9: tier pro -> free"
    assert undo_plan(a, with_old).kind == "revert"

    without_old = _trace("update_field", {"record_id": "C-9", "field": "tier",
                                          "value": "free"})
    plan = undo_plan(_call(without_old), without_old)
    assert plan.kind == "noop" and plan.reversible is False


def test_domain_packs_are_opt_in():
    # With none of them registered, a SQL call gets only core inference.
    for name in ("sql", "browser", "support"):  # order-independent: force-clear
        unregister(name)
    try:
        a = _call(_trace("run_query", {"query": "INSERT INTO orders VALUES (1)"}))
        assert a.state_diff is None
    finally:
        from loom.packs import install_builtin

        install_builtin()  # restore the default registry for later tests


# -- snapshot / restore contract ---------------------------------------------

def test_coding_snapshot_restore_is_runnable_on_a_clean_tree(domain_packs):
    from loom.packs.coding import CodingPack

    trace = {"workspace": {"git": {"commit": "abc1234567890", "branch": "main",
                                   "dirty": False}}}
    snap = CodingPack().snapshot(trace)
    assert snap["commit"] == "abc1234567890"
    plan = CodingPack().restore(snap)
    assert plan.kind == "git" and plan.executable is True
    assert plan.commands == ["git checkout abc123456789"]


def test_coding_restore_on_a_dirty_tree_is_not_executable(domain_packs):
    from loom.packs.coding import CodingPack

    plan = CodingPack().restore(
        CodingPack().snapshot({"workspace": {"git": {"commit": "deadbeef1234", "dirty": True}}}))
    assert plan.executable is False and "dirty" in plan.summary


def test_domain_packs_give_honest_manual_restore(domain_packs):
    # sql/browser/support can't be reproduced from the trace: manual, advisory.
    from loom.packs.sql import SqlPack

    plan = SqlPack().restore(SqlPack().snapshot({}))
    assert plan.kind == "manual" and plan.executable is False
    assert "database" in plan.summary


def test_restore_plans_one_per_touched_domain(domain_packs):
    from loom.action import actions
    from loom.packs import restore_plans

    trace = {
        "workspace": {"git": {"commit": "abc1234567890", "dirty": False}},
        "log": [
            {"seq": 0, "kind": "model", "key": "k",
             "result": {"tool_calls": [{"id": "t", "name": "Edit",
                        "input": {"file_path": "a.py"}}], "stop_reason": "tool_use", "usage": {}}},
            {"seq": 1, "kind": "tool:Edit", "key": "k2", "result": "edited"},
            {"seq": 2, "kind": "model", "key": "k3",
             "result": {"tool_calls": [{"id": "t2", "name": "run_query",
                        "input": {"query": "INSERT INTO t VALUES (1)"}}],
                        "stop_reason": "tool_use", "usage": {}}},
            {"seq": 3, "kind": "tool:run_query", "key": "k4", "result": "1 row"},
        ],
    }
    plans = dict(restore_plans([a for a in actions(trace) if a.type == "call"], trace))
    assert plans["coding"].executable is True       # git checkout
    assert plans["sql"].executable is False         # manual DB snapshot
