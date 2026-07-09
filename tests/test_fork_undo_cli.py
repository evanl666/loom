"""`loom fork` and the generic undo/compensation surface."""

import json

from loom import Agent, tool
from loom.cli import main
from loom.providers import ModelResponse, ScriptedProvider, ToolCall


@tool
def Edit(file_path: str, new: str) -> str:
    "edit"
    return "edited"


@tool
def run_sql(query: str) -> str:
    "sql"
    return "1 row inserted"


def _mixed_run():
    prov = ScriptedProvider([
        ModelResponse(text="edit first",
                      tool_calls=[ToolCall("t1", "Edit", {"file_path": "a.py", "new": "x"})],
                      stop_reason="tool_use"),
        ModelResponse(text="now insert",
                      tool_calls=[ToolCall("t2", "run_sql",
                                           {"query": "INSERT INTO orders VALUES (1)"})],
                      stop_reason="tool_use"),
        ModelResponse(text="done"),
    ])
    return Agent(model=prov, tools=[Edit, run_sql]).run("do both")


def test_run_undo_plans_cover_every_domain_newest_first():
    plans = _mixed_run().undo_plans()
    tools = [a.tool for a, _ in plans]
    assert tools == ["run_sql", "Edit"]  # newest first: undo runs backwards
    sql_plan = plans[0][1]
    assert sql_plan.kind == "compensate" and "DELETE FROM orders" in sql_plan.commands[0]
    assert plans[1][1].commands == ["git checkout HEAD -- a.py"]


def test_cli_undo_plan_prints_all_domains(tmp_path, capsys):
    path = str(tmp_path / "r.loom.json")
    _mixed_run().save(path)
    assert main(["undo", path, "--plan"]) == 0
    out = capsys.readouterr().out
    assert "run_sql: DELETE the rows inserted into orders" in out
    assert "Edit: restore a.py to HEAD" in out
    assert "$ git checkout HEAD -- a.py" in out


def test_cli_fork_plan_shows_replay_hints_without_an_agent(tmp_path, capsys):
    path = str(tmp_path / "r.loom.json")
    _mixed_run().save(path)
    assert main(["fork", path, "--from-step", "2"]) == 0
    out = capsys.readouterr().out
    assert "fork at turn 1 (step 2)" in out
    assert "[sql] (manual) restore the database" in out   # the pack's restore plan
    assert "run.fork(at=1" in out                          # the Python snippet


def test_cli_fork_continues_live_with_an_agent(tmp_path, capsys, monkeypatch):
    path = str(tmp_path / "r.loom.json")
    _mixed_run().save(path)
    (tmp_path / "forkmod.py").write_text(
        "from loom import Agent, tool\n"
        "from loom.providers import RuleProvider, ModelResponse, ToolCall\n"
        "@tool\n"
        "def Edit(file_path: str, new: str) -> str:\n"
        "    'edit'\n"
        "    return 'edited'\n"
        "@tool\n"
        "def run_sql(query: str) -> str:\n"
        "    'sql'\n"
        "    return '1 row inserted'\n"
        "def _final(ms):\n"
        "    return ModelResponse(text='forked ending')\n"
        "agent = Agent(model=RuleProvider(rules=[_final]), tools=[Edit, run_sql])\n"
    )
    monkeypatch.chdir(tmp_path)
    out_path = str(tmp_path / "branch.loom.json")
    assert main(["fork", path, "--turn", "1", "--agent", "forkmod:agent",
                 "-o", out_path]) == 0
    out = capsys.readouterr().out
    assert "forked ending" in out
    saved = json.loads((tmp_path / "branch.loom.json").read_text())
    assert saved["output"] == "forked ending"
    # the replayed prefix is intact: the first Edit call is still step 1
    assert saved["log"][1]["kind"] == "tool:Edit"


def test_cli_fork_inject_edits_the_context(tmp_path, capsys, monkeypatch):
    path = str(tmp_path / "r.loom.json")
    _mixed_run().save(path)
    (tmp_path / "forkmod2.py").write_text(
        "from loom import Agent, tool\n"
        "from loom.providers import RuleProvider, ModelResponse\n"
        "@tool\n"
        "def Edit(file_path: str, new: str) -> str:\n"
        "    'edit'\n"
        "    return 'edited'\n"
        "@tool\n"
        "def run_sql(query: str) -> str:\n"
        "    'sql'\n"
        "    return 'ok'\n"
        "def _echo_note(ms):\n"
        "    users = [m['content'] for m in ms if m['role'] == 'user']\n"
        "    return ModelResponse(text='saw: ' + users[-1])\n"
        "agent = Agent(model=RuleProvider(rules=[_echo_note]), tools=[Edit, run_sql])\n"
    )
    monkeypatch.chdir(tmp_path)
    assert main(["fork", path, "--turn", "1", "--agent", "forkmod2:agent",
                 "--inject", "skip the database write", "-o",
                 str(tmp_path / "b2.loom.json")]) == 0
    assert "saw: skip the database write" in capsys.readouterr().out
