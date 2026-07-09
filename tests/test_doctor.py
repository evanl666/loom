"""loom doctor: environment preflight (no args) and trace rot check (path)."""

import json

from loom.cli import main


def test_doctor_environment_reports_readiness(capsys):
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "agents (loom record" in out
    assert "optional extras" in out
    assert "claude-code-safe" in out  # lists profiles
    assert "ready to record." in out


def test_doctor_trace_still_checks_context_rot(tmp_path, capsys):
    # a trace with an oversized tool result -> doctor(path) finds rot, exit 1
    big = "x " * 400
    trace = {
        "model": "m", "episodes": ["hi"], "output": "done", "stop_reason": "end_turn",
        "log": [
            {"seq": 0, "kind": "model", "key": "k", "depth": 0,
             "result": {"text": "", "stop_reason": "tool_use",
                        "tool_calls": [{"id": "t", "name": "fetch", "input": {}}], "usage": {}}},
            {"seq": 1, "kind": "tool:fetch", "key": "k", "depth": 0, "result": big},
            {"seq": 2, "kind": "model", "key": "k", "depth": 0,
             "result": {"text": "done", "stop_reason": "end_turn", "usage": {}}},
        ],
    }
    path = tmp_path / "r.loom.json"
    path.write_text(json.dumps(trace))
    code = main(["doctor", str(path)])
    out = capsys.readouterr().out
    assert code == 1 and "oversized" in out


def test_doctor_missing_trace_is_a_friendly_error(capsys):
    assert main(["doctor", "/nope.loom.json"]) == 2
    assert "no such file" in capsys.readouterr().err
