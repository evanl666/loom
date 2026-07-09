"""loom trace validate / verify / explain-version: the format contract."""

import json

from loom import Agent
from loom.cli import main
from loom.providers import ModelResponse, ScriptedProvider


def _trace(tmp_path):
    run = Agent(model=ScriptedProvider([ModelResponse(text="hi")])).run("q")
    path = str(tmp_path / "t.loom.json")
    run.save(path)
    return path


def test_validate_passes_a_fresh_trace(tmp_path, capsys):
    assert main(["trace", "validate", _trace(tmp_path)]) == 0
    assert "valid" in capsys.readouterr().out


def test_verify_detects_tampering(tmp_path, capsys):
    path = _trace(tmp_path)
    assert main(["trace", "verify", path]) == 0

    data = json.load(open(path))
    data["output"] = "tampered"
    json.dump(data, open(path, "w"))
    assert main(["trace", "verify", path]) == 1
    assert "MODIFIED" in capsys.readouterr().err


def test_explain_version_reports_current(tmp_path, capsys):
    assert main(["trace", "explain-version", _trace(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "version 2" in out and "current" in out


def test_validate_flags_a_future_version(tmp_path, capsys):
    path = _trace(tmp_path)
    data = json.load(open(path))
    data["version"] = 99
    json.dump(data, open(path, "w"))
    assert main(["trace", "validate", path]) == 1
    assert "newer than" in capsys.readouterr().out


def test_trace_on_missing_file_is_friendly(capsys):
    assert main(["trace", "verify", "/nope.loom.json"]) == 2
    assert "no such file" in capsys.readouterr().err


def test_sign_and_verify_with_a_key(tmp_path, capsys, monkeypatch):
    path = _trace(tmp_path)
    monkeypatch.setenv("LOOM_KEY", "shared-secret")
    assert main(["trace", "sign", path, "--key-env", "LOOM_KEY"]) == 0
    assert "signed" in capsys.readouterr().out
    assert json.load(open(path))["signature"].startswith("hmac-sha256:")

    assert main(["trace", "verify", path, "--key-env", "LOOM_KEY"]) == 0
    assert "signature valid" in capsys.readouterr().out

    # wrong key fails
    monkeypatch.setenv("LOOM_KEY", "not-the-key")
    assert main(["trace", "verify", path, "--key-env", "LOOM_KEY"]) == 1
    assert "INVALID" in capsys.readouterr().err


def test_verify_signature_detects_tampering(tmp_path, capsys, monkeypatch):
    path = _trace(tmp_path)
    monkeypatch.setenv("LOOM_KEY", "k")
    main(["trace", "sign", path, "--key-env", "LOOM_KEY"])
    capsys.readouterr()

    data = json.load(open(path))
    data["output"] = "tampered"
    json.dump(data, open(path, "w"))
    assert main(["trace", "verify", path, "--key-env", "LOOM_KEY"]) == 1
    assert "INVALID" in capsys.readouterr().err


def test_sign_without_key_is_a_clear_error(tmp_path, capsys):
    assert main(["trace", "sign", _trace(tmp_path)]) == 2
    assert "needs a key" in capsys.readouterr().err
