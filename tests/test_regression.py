"""loom regression from: a bad run becomes a test that stays red."""

import json
import os
import subprocess
import sys

from loom import Agent, tool
from loom.cli import main
from loom.providers import ModelResponse, ScriptedProvider, ToolCall
from loom.regression import build_regression


SECRET = "sk-ant-api03-" + "a1B2" * 8


@tool
def Read(file_path: str) -> str:
    "read"
    return f"ANTHROPIC_API_KEY={SECRET}"


@tool
def Bash(command: str) -> str:
    "sh"
    return "ok"


def _bad_run():
    return Agent(model=ScriptedProvider([
        ModelResponse(tool_calls=[ToolCall("t1", "Read", {"file_path": "/app/.env"})],
                      stop_reason="tool_use"),
        ModelResponse(tool_calls=[ToolCall("t2", "Bash", {"command": "rm -rf /tmp/x"})],
                      stop_reason="tool_use"),
        ModelResponse(text="done"),
    ]), tools=[Read, Bash]).run("go")


def test_build_regression_writes_the_bundle(tmp_path):
    trace = str(tmp_path / "bad.loom.json")
    _bad_run().save(trace)
    result = build_regression(trace, str(tmp_path / "reg"))
    assert result["risky"] == 2
    for f in result["files"]:
        assert os.path.isfile(os.path.join(result["outdir"], f))

    # the fixture is SCRUBBED (the secret never reaches the guard)
    fixture = json.load(open(os.path.join(result["outdir"], result["fixture"])))
    assert fixture.get("scrubbed") is True
    assert SECRET not in json.dumps(fixture)

    # the policy denies the .env read
    policy = open(os.path.join(result["outdir"], result["policy"])).read()
    assert "Read(*.env*)" in policy and "deny" in policy


def test_generated_pytest_passes(tmp_path):
    trace = str(tmp_path / "bad.loom.json")
    _bad_run().save(trace)
    result = build_regression(trace, str(tmp_path / "reg"))
    # run the generated guard as its own pytest process
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", result["test"]],
        cwd=result["outdir"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "2 passed" in proc.stdout


def test_cli_regression_from(tmp_path, capsys):
    trace = str(tmp_path / "bad.loom.json")
    _bad_run().save(trace)
    assert main(["regression", "from", trace, "-o", str(tmp_path / "reg")]) == 0
    out = capsys.readouterr().out
    assert "regression guard ->" in out and "test_bad.py" in out
    assert "2 risky action(s) captured" in out
