"""Workspace metadata: what the recording was OF, beyond the API traffic."""

import json

from loom.incident import build_report
from loom.proxy import WireRecorder
from loom.workspace import collect


def test_collect_captures_the_essentials():
    ws = collect(command=["claude", "-p", "fix"], target="https://api.openai.com")
    assert ws["cwd"] and ws["os"] and ws["python"]
    assert ws["argv"] == ["claude", "-p", "fix"]
    assert ws["dialect"] == "openai"
    assert "recorded_at" in ws


def test_collect_reads_git_when_present(tmp_path, monkeypatch):
    import subprocess

    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path)
    (tmp_path / "f.txt").write_text("hi")
    subprocess.run(["git", "add", "."], cwd=tmp_path)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path)

    ws = collect()
    assert len(ws["git"]["commit"]) == 40
    assert ws["git"]["dirty"] is False
    (tmp_path / "f.txt").write_text("changed")  # now the tree is dirty
    assert collect()["git"]["dirty"] is True


def test_collect_survives_no_git(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # a bare temp dir: not a repo
    ws = collect()
    assert "git" not in ws
    assert ws["cwd"]  # the rest still works


def test_workspace_rides_the_trace_and_shows_in_incident(tmp_path):
    rec = WireRecorder()
    rec.episodes = ["fix the deploy"]
    rec.output = "DEPLOY FAILED"
    rec.workspace = {
        "cwd": "/repo", "os": "Linux 6.1", "argv": ["claude", "-p", "fix"],
        "git": {"commit": "abc1234567deadbeef", "branch": "main", "dirty": True},
    }
    path = str(tmp_path / "s.loom.json")
    rec.save(path)

    data = json.load(open(path))
    assert data["workspace"]["git"]["dirty"] is True

    report = build_report(data, path)
    assert "**Where:**" in report
    assert "abc1234567" in report and "dirty tree" in report and "/repo" in report
