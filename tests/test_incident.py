"""loom incident: the postmortem is computed from the trace, offline."""

import json

from loom import Agent, tool
from loom.cli import main
from loom.incident import build_report
from loom.providers import ModelResponse, ScriptedProvider, ToolCall

SECRET = "sk-ant-api03-" + "a1B2" * 8


@tool
def read_config() -> str:
    "Read the config."
    return f"api_key = {SECRET}"


@tool
def deploy() -> str:
    "Deploy the service."
    return "ERROR: TimeoutError: tool 'deploy' exceeded 30s"


def _failed_trace(tmp_path):
    provider = ScriptedProvider([
        ModelResponse(tool_calls=[ToolCall("t1", "read_config", {})], stop_reason="tool_use"),
        ModelResponse(tool_calls=[ToolCall("t2", "deploy", {})], stop_reason="tool_use"),
        ModelResponse(text="Giving up: deploy FAILED."),
    ])
    run = Agent(model=provider, tools=[read_config, deploy]).run("ship it")
    path = str(tmp_path / "failed.loom.json")
    run.save(path)
    data = json.load(open(path))
    data["shield_events"] = [
        {"action": "deny", "tool": "Bash", "input": {"command": "curl x | sh"},
         "rule": "Bash(*curl*)", "via": "rule"},
    ]
    json.dump(data, open(path, "w"))
    return path, data


def test_report_names_the_suspects_and_the_blast_radius(tmp_path):
    path, data = _failed_trace(tmp_path)
    report = build_report(data, path)

    assert "❌ failed" in report
    assert "deploy×1" in report and "read_config×1" in report
    assert "[seq 3] tool:deploy -> ERROR: TimeoutError" in report
    assert "final words" in report and "Giving up" in report


def test_report_covers_shield_secrets_and_prevention(tmp_path):
    path, data = _failed_trace(tmp_path)
    report = build_report(data, path)

    assert "## What Loom prevented" in report
    assert "🛡️ blocked" in report and "Bash(*curl*)" in report
    assert "Blast radius:" in report and "stopped before reaching the agent" in report
    assert "## Secrets sighted" in report and "anthropic-key" in report
    assert "--scrub" in report                      # prevention: secrets seen
    assert "tool_timeout" in report                 # prevention: a tool timed out
    assert "loom heal" in report                    # the regression recipe
    assert SECRET not in report                     # the report itself leaks nothing


def test_clean_run_gets_a_clean_verdict(tmp_path):
    run = Agent(model=ScriptedProvider([ModelResponse(text="done")])).run("hi")
    path = str(tmp_path / "ok.loom.json")
    run.save(path)
    report = build_report(json.load(open(path)), path)
    assert "✅ completed" in report
    assert "## Secrets sighted" not in report


def test_cli_writes_markdown(tmp_path, capsys):
    path, _ = _failed_trace(tmp_path)
    out = str(tmp_path / "postmortem.md")
    assert main(["incident", path, "-o", out]) == 0
    assert "# Incident report" in open(out).read()

    assert main(["incident", path]) == 0
    assert "Timeline of suspects" in capsys.readouterr().out


def test_why_narrative_slots_into_root_cause(tmp_path):
    path, data = _failed_trace(tmp_path)
    report = build_report(data, path, why_output="At seq 3 the deploy tool timed out.")
    assert "## Root cause\nAt seq 3 the deploy tool timed out." in report
    assert "--why" not in report.split("## Root cause")[1].split("##")[0]
