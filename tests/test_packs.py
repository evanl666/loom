"""The Pack interface -- teaching Loom about a domain of agent."""

import pytest

from loom import actions
from loom.action import Action, StateDiff
from loom.packs import Pack, UndoPlan, enrich, pack_for, packs, register, undo_plan, unregister


class SqlPack(Pack):
    name = "sql-test"

    def owns(self, action):
        return action.tool == "run_sql"

    def capabilities(self, name, tool_input):
        return {"database_write"} if "INSERT" in str(tool_input) else set()

    def state_diff(self, action, trace):
        return StateDiff("database", "+1 row in customers")

    def undo(self, action, trace):
        return UndoPlan("compensate", "delete the inserted row",
                        ["DELETE FROM customers WHERE id=99"], reversible=False)


@pytest.fixture
def sql_pack():
    register(SqlPack())
    yield
    unregister("sql-test")


def _sql_trace():
    return {"log": [
        {"seq": 0, "kind": "model", "key": "k",
         "result": {"text": "inserting", "tool_calls": [
             {"id": "t", "name": "run_sql", "input": {"query": "INSERT INTO customers VALUES (99)"}}],
             "stop_reason": "tool_use", "usage": {}}},
        {"seq": 1, "kind": "tool:run_sql", "key": "k2", "result": "ok"},
    ]}


def test_registered_pack_enriches_actions(sql_pack):
    call = [a for a in actions(_sql_trace()) if a.type == "call"][0]
    assert "database_write" in call.capabilities
    assert call.state_diff.kind == "database"
    assert call.state_diff.summary == "+1 row in customers"


def test_undo_plan_comes_from_owning_pack(sql_pack):
    call = [a for a in actions(_sql_trace()) if a.type == "call"][0]
    plan = undo_plan(call, _sql_trace())
    assert plan.kind == "compensate" and plan.reversible is False
    assert plan.commands == ["DELETE FROM customers WHERE id=99"]


def test_registry_is_idempotent_by_name(sql_pack):
    before = len(packs())
    register(SqlPack())  # same name -> replaces, doesn't duplicate
    assert len(packs()) == before


def test_no_packs_is_a_noop():
    # With nothing registered, enrichment leaves Actions untouched.
    call = [a for a in actions(_sql_trace()) if a.type == "call"][0]
    assert call.state_diff is None
    assert enrich([call], _sql_trace()) == [call]


def test_pack_for_finds_the_owner(sql_pack):
    call = [a for a in actions(_sql_trace()) if a.type == "call"][0]
    assert pack_for(call).name == "sql-test"
    # a plain business read owned by no registered pack (coding owns fs/shell)
    other = Action(step=9, depth=0, type="call", tool="get_weather")
    assert pack_for(other) is None
