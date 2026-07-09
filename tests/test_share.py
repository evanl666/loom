"""loom share: scrub, refuse if secrets remain, mark the copy safe."""

import json

from loom import Agent
from loom.cli import main
from loom.export import trace_to_html
from loom.providers import ModelResponse, ScriptedProvider

SECRET = "sk-ant-api03-" + "a1B2" * 8


def _trace(tmp_path, output):
    run = Agent(model=ScriptedProvider([ModelResponse(text=output)])).run("q")
    path = str(tmp_path / "s.loom.json")
    run.save(path)
    return path


def test_share_redacts_and_marks_safe(tmp_path, capsys):
    path = _trace(tmp_path, f"your key: {SECRET}")
    out = str(tmp_path / "shared.loom.json")
    assert main(["share", path, "-o", out]) == 0

    shared = json.load(open(out))
    assert SECRET not in json.dumps(shared)
    assert shared["scrubbed"] is True
    assert "safe to share" in capsys.readouterr().out


def test_share_default_output_path(tmp_path):
    path = _trace(tmp_path, "nothing secret here")
    assert main(["share", path]) == 0
    assert (tmp_path / "s.shared.loom.json").exists()


def test_studio_banner_reflects_scrubbed_flag(tmp_path):
    path = _trace(tmp_path, "hello")
    unsafe = trace_to_html(json.load(open(path)))
    assert "Not scrubbed" in unsafe and "loom share" in unsafe

    main(["share", path, "-o", str(tmp_path / "shared.loom.json")])
    safe = trace_to_html(json.load(open(tmp_path / "shared.loom.json")))
    assert "Scrubbed" in safe and "safe to share" in safe
    assert "Not scrubbed" not in safe


def test_share_refuses_when_secrets_survive(tmp_path, capsys, monkeypatch):
    # Simulate a scrub that leaves a secret behind: patch scrub_trace to no-op
    # so the residual scan trips and share refuses to emit.
    path = _trace(tmp_path, f"leak {SECRET}")
    monkeypatch.setattr("loom.scrub.scrub_trace",
                        lambda data, aggressive=False: (data, {}))
    assert main(["share", path, "-o", str(tmp_path / "out.loom.json")]) == 1
    assert "survived scrubbing" in capsys.readouterr().err
    assert not (tmp_path / "out.loom.json").exists()  # nothing written
