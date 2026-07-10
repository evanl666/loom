"""loom undo: revert the agent's file changes, leave the human's alone."""

import json
import subprocess

from loom.cli import main
from loom.workspace import changes_since, diff_snapshot


def _repo(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path)
    (tmp_path / "app.py").write_text("v1\n")
    (tmp_path / "mine.py").write_text("keep\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path)
    return tmp_path


def _record_agent_changes(tmp_path, capture_diff=True):
    """Human dirties mine.py; then the 'agent' edits app.py + adds helper.py."""
    (tmp_path / "mine.py").write_text("my work\n")           # pre-existing (human)
    before = diff_snapshot(str(tmp_path))
    (tmp_path / "app.py").write_text("agent broke it\n")     # agent modifies
    (tmp_path / "helper.py").write_text("agent junk\n")      # agent adds
    after = diff_snapshot(str(tmp_path))
    ch = changes_since(before, after, agent_exit_code=1, capture_diff=capture_diff,
                       cwd=str(tmp_path))
    path = str(tmp_path / "run.loom.json")
    json.dump({"workspace": {"changes": ch}, "log": [], "episodes": ["x"], "output": "FAILED"},
              open(path, "w"))
    return path


def test_undo_reverts_agent_edits_and_spares_human(tmp_path, monkeypatch, capsys):
    repo = _repo(tmp_path)
    path = _record_agent_changes(repo)
    monkeypatch.chdir(repo)

    assert main(["undo", path]) == 0
    assert (repo / "app.py").read_text() == "v1\n"          # agent edit reverted
    assert not (repo / "helper.py").exists()                # agent addition removed
    assert (repo / "mine.py").read_text() == "my work\n"    # human work untouched


def test_undo_dry_run_changes_nothing(tmp_path, monkeypatch, capsys):
    repo = _repo(tmp_path)
    path = _record_agent_changes(repo)
    monkeypatch.chdir(repo)

    assert main(["undo", path, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "would undo" in out and "app.py" in out and "helper.py" in out
    assert (repo / "app.py").read_text() == "agent broke it\n"  # untouched


def test_undo_only_scopes_the_revert(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    path = _record_agent_changes(repo, capture_diff=False)  # per-file path
    monkeypatch.chdir(repo)

    assert main(["undo", path, "--only", "helper.py"]) == 0
    assert not (repo / "helper.py").exists()                # in scope: removed
    assert (repo / "app.py").read_text() == "agent broke it\n"  # out of scope: kept


def test_undo_refuses_when_tree_moved(tmp_path, monkeypatch, capsys):
    repo = _repo(tmp_path)
    path = _record_agent_changes(repo)
    # someone keeps working after the recording -> state hash no longer matches
    (repo / "app.py").write_text("even newer work\n")
    monkeypatch.chdir(repo)

    assert main(["undo", path]) == 1
    assert "changed since the recording" in capsys.readouterr().out
    assert (repo / "app.py").read_text() == "even newer work\n"  # nothing clobbered
    # --force overrides
    assert main(["undo", path, "--force"]) == 0


def test_only_is_segment_aware():
    from loom.undo import _in_scope

    assert _in_scope("src/a.py", "src")
    assert _in_scope("src", "src")
    assert not _in_scope("src2/a.py", "src")    # the old startswith footgun
    assert _in_scope("src/a.py", "src/")


def test_undo_refuses_paths_outside_the_working_tree(tmp_path):
    """A hostile/hand-crafted trace could record an absolute or ../.. path;
    undo runs os.remove, so anything escaping the tree must be refused."""
    import os
    import subprocess

    from loom.undo import undo

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo)
    victim = tmp_path / "victim.txt"
    victim.write_text("precious")

    data = {"workspace": {"changes": {"files": [{"path": str(victim), "status": "A"}]}}}
    ok, log = undo(data, str(repo), force=True)
    assert not ok
    assert victim.exists() and victim.read_text() == "precious"
