"""The loom test and loom watch CLI subcommands."""

import json

from loom import Agent
from loom.cli import main
from loom.providers import ModelResponse, ScriptedProvider


def make_trace(path, text="hello"):
    run = Agent(model=ScriptedProvider([ModelResponse(text=text)])).run("question")
    run.save(str(path))
    return run


def test_loom_test_passes_good_traces(tmp_path, capsys):
    make_trace(tmp_path / "a.loom.json")
    make_trace(tmp_path / "b.loom.json")
    code = main(["test", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "2/2 traces passed" in out


def test_loom_test_fails_corrupted_trace(tmp_path, capsys):
    make_trace(tmp_path / "good.loom.json")
    p = tmp_path / "bad.loom.json"
    make_trace(p)
    data = json.loads(p.read_text())
    data["output"] = "TAMPERED"  # no longer matches the final model text
    p.write_text(json.dumps(data))

    code = main(["test", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 1
    assert "FAIL" in out and "1/2 traces passed" in out


def test_loom_watch_once_prints_journal(tmp_path, capsys):
    journal = str(tmp_path / "run.jsonl")
    agent = Agent(
        model=ScriptedProvider([ModelResponse(text="watched answer")]), journal=journal
    )
    agent.run("watch me")

    code = main(["watch", journal, "--once"])
    out = capsys.readouterr().out
    assert code == 0
    assert "watch me" in out  # header with the prompt
    assert "watched answer" in out  # the recorded effect
