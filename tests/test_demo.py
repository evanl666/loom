"""loom demo: the one-command shock demo."""

import os

from loom.cli import main
from loom.demo import run_demo, scenarios


def test_run_demo_emits_all_artifacts(tmp_path):
    for scenario in scenarios():
        r = run_demo(scenario, str(tmp_path / scenario))
        assert os.path.isfile(r["trace"])
        assert os.path.isfile(r["movie"])
        assert os.path.isfile(r["autopsy"])
        assert os.path.isdir(r["fix_dir"])


def test_demo_redteam_verdict_present_for_attacks(tmp_path):
    r = run_demo("secret-leak", str(tmp_path / "d"))
    assert r["redteam"] is not None
    assert r["redteam"]["stopped"] is True          # claude-code-safe stops it


def test_movie_and_autopsy_never_leak_the_secret(tmp_path):
    from loom.demo import SECRET

    r = run_demo("secret-leak", str(tmp_path / "d"))
    assert SECRET not in open(r["movie"]).read()
    assert SECRET not in open(r["autopsy"]).read()


def test_cli_demo(tmp_path, capsys):
    assert main(["demo", "--scenario", "refund", "-o", str(tmp_path / "d")]) == 0
    out = capsys.readouterr().out
    assert "loom demo: refund" in out and "movie:" in out
    assert "red team:" in out
