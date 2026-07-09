"""Every CLI error names the problem AND the next step -- no tracebacks."""

import json

import pytest

from loom.cli import main


@pytest.fixture
def not_a_trace(tmp_path):
    p = tmp_path / "notes.loom.json"
    p.write_text('{"hello": 1}')
    return str(p)


def _err(capsys):
    return capsys.readouterr().err


def test_missing_file_is_a_sentence(capsys):
    assert main(["replay", "/definitely/not/here.loom.json"]) == 2
    assert "no such file" in _err(capsys)


def test_invalid_json_names_the_line(tmp_path, capsys):
    p = tmp_path / "bad.loom.json"
    p.write_text("garbage{")
    assert main(["replay", str(p)]) == 2
    assert "not valid JSON (line 1)" in _err(capsys)


@pytest.mark.parametrize("command", ["replay", "timeline", "doctor"])
def test_json_without_log_says_not_a_trace(command, not_a_trace, capsys):
    assert main([command, not_a_trace]) == 2
    err = _err(capsys)
    assert "not a loom trace" in err and "run.save()" in err


def test_directory_points_at_corpus_commands(tmp_path, capsys):
    assert main(["replay", str(tmp_path)]) == 2
    assert "loom test" in _err(capsys)


def test_proxy_replay_of_harness_trace_names_the_fix(tmp_path, capsys):
    from loom import Agent
    from loom.providers import ModelResponse, ScriptedProvider

    p = str(tmp_path / "harness.loom.json")
    Agent(model=ScriptedProvider([ModelResponse(text="x")])).run("q").save(p)
    assert main(["proxy", "--replay", p]) == 2
    err = _err(capsys)
    assert "harness trace" in err and f"loom replay {p}" in err


def test_record_zero_steps_hints_at_target(tmp_path, capsys):
    import sys as _sys

    code = main(["record", "--save", str(tmp_path / "s.loom.json"),
                 "--", _sys.executable, "-c", "pass"])
    assert code == 0  # the child's exit code passes through
    err = _err(capsys)
    assert "no traffic recorded" in err
    assert "--target https://api.openai.com" in err
