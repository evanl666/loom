"""CLI: loom trust / loom why."""

import json

from loom.cli import main
from loom.shield import TrustLedger


def _seeded_ledger(tmp_path):
    path = str(tmp_path / "trust.json")
    ledger = TrustLedger(path)
    ledger.record("Bash", True, {"id": "a1"})
    ledger.record("Bash", True, {"id": "b2"})
    return path


def test_trust_shows_streaks_with_evidence(tmp_path, capsys):
    path = _seeded_ledger(tmp_path)
    assert main(["trust", "--ledger", path]) == 0
    out = capsys.readouterr().out
    assert "Bash: streak 2" in out and "a1, b2" in out


def test_trust_demote_resets_a_tool(tmp_path, capsys):
    path = _seeded_ledger(tmp_path)
    assert main(["trust", "--ledger", path, "--demote", "Bash"]) == 0
    assert json.load(open(path))["Bash"]["streak"] == 0
    assert main(["trust", "--ledger", path, "--demote", "NeverSeen"]) == 1
    assert "no trust recorded" in capsys.readouterr().err


def test_trust_with_empty_ledger(tmp_path, capsys):
    assert main(["trust", "--ledger", str(tmp_path / "none.json")]) == 0
    assert "no trust recorded yet" in capsys.readouterr().out


def test_why_surfaces_provider_errors_cleanly(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    trace = tmp_path / "run.loom.json"
    trace.write_text(json.dumps({"episodes": ["hi"], "output": "", "log": []}))
    assert main(["why", str(trace), "what happened?"]) == 1
    assert "why failed" in capsys.readouterr().err
