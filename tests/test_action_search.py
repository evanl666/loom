"""Ordered action-path search: sequence_hits + `loom search "path:A->B"`."""

import json

from loom import Agent, tool
from loom.action import sequence_hits
from loom.cli import main
from loom.providers import ModelResponse, ScriptedProvider, ToolCall


@tool
def get_customer(id: int) -> str:
    "lookup"
    return "Jane"


@tool
def send_email(to: str) -> str:
    "email"
    return "sent"


def _save(tmp_path, name, order):
    calls = {"pii": ToolCall("t1", "get_customer", {"id": 7}),
             "email": ToolCall("t2", "send_email", {"to": "x"})}
    responses = [ModelResponse(tool_calls=[calls[k]], stop_reason="tool_use") for k in order]
    responses.append(ModelResponse(text="done"))
    run = Agent(model=ScriptedProvider(responses),
                tools=[get_customer, send_email]).run("go")
    path = str(tmp_path / name)
    run.save(path)
    return path


def test_sequence_hits_respects_order(tmp_path):
    leak = json.load(open(_save(tmp_path, "leak.loom.json", ["pii", "email"])))
    fine = json.load(open(_save(tmp_path, "fine.loom.json", ["email", "pii"])))
    hits = sequence_hits(leak, "pii_access", "user_communication")
    assert len(hits) == 1
    first, then = hits[0]
    assert first.tool == "get_customer" and then.tool == "send_email"
    assert sequence_hits(fine, "pii_access", "user_communication") == []


def test_sequence_hits_matches_risk_and_tool_names(tmp_path):
    leak = json.load(open(_save(tmp_path, "leak.loom.json", ["pii", "email"])))
    assert sequence_hits(leak, "pii-access", "send_email")  # risk + tool name


def test_cli_search_path_filters_and_shows_evidence(tmp_path, capsys):
    _save(tmp_path, "leak.loom.json", ["pii", "email"])
    _save(tmp_path, "fine.loom.json", ["email", "pii"])
    assert main(["search", str(tmp_path), "path:pii_access->user_communication"]) == 0
    out = capsys.readouterr().out
    assert "leak.loom.json" in out and "fine.loom.json" not in out
    assert "path: [1] get_customer → [3] send_email" in out
    assert "1 run(s)" in out


def test_cli_packs_lists_builtins(capsys):
    assert main(["packs"]) == 0
    out = capsys.readouterr().out
    for name in ("coding", "sql", "browser", "support"):
        assert name in out
    assert "built-in" in out and "loom.packs" in out
