"""Workspace mutations: before/after git-diff capture around a recording."""

import json
import subprocess
import sys
import threading

from loom.cli import main
from loom.incident import build_report
from loom.workspace import changes_since, diff_snapshot
from tests.test_proxy import FINAL_ANSWER, _FakeUpstream


def _git_repo(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path)
    (tmp_path / "app.py").write_text("print('v1')\n")
    (tmp_path / "already.py").write_text("baseline\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path)
    return tmp_path


def test_changes_since_isolates_the_agents_edits(tmp_path, monkeypatch):
    repo = _git_repo(tmp_path)
    monkeypatch.chdir(repo)

    # a file already dirty before the "agent" runs
    (repo / "already.py").write_text("baseline changed by human\n")
    before = diff_snapshot(str(repo))

    # the agent edits app.py and creates new.py
    (repo / "app.py").write_text("print('v2')\n")
    (repo / "new.py").write_text("agent made this\n")
    after = diff_snapshot(str(repo))

    changes = changes_since(before, after, agent_exit_code=0)
    by_path = {f["path"]: f for f in changes["files"]}
    assert by_path["app.py"]["status"] == "M"
    assert by_path["new.py"]["status"] == "A"  # untracked -> added
    assert by_path["already.py"]["pre_existing"] is True   # dirty before the run
    assert by_path["app.py"]["pre_existing"] is False       # the agent's own edit
    assert by_path["new.py"]["pre_existing"] is False
    assert changes["stat"] and changes["dirty_hash"]
    assert changes["agent_exit_code"] == 0


# The "agent": a child that edits a file and talks to the API through the proxy.
CHILD = """
import json, os, urllib.request
open("app.py", "w").write("print('edited by agent')\\n")
open("brand_new.py", "w").write("new\\n")
url = os.environ["ANTHROPIC_BASE_URL"] + "/v1/messages"
req = urllib.request.Request(url,
    data=json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}]}).encode(),
    headers={"content-type": "application/json"}, method="POST")
urllib.request.urlopen(req, timeout=10).read()
"""


def test_record_captures_file_changes(tmp_path, monkeypatch):
    repo = _git_repo(tmp_path)
    monkeypatch.chdir(repo)
    (repo / "child.py").write_text(CHILD)

    upstream = _FakeUpstream([FINAL_ANSWER])
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    save = str(repo / "session.loom.json")
    code = main([
        "record", "--save", save,
        "--target", f"http://127.0.0.1:{upstream.server_address[1]}",
        "--", sys.executable, str(repo / "child.py"),
    ])
    upstream.shutdown()
    assert code == 0

    changes = json.load(open(save))["workspace"]["changes"]
    paths = {f["path"] for f in changes["files"]}
    assert "app.py" in paths and "brand_new.py" in paths
    assert changes["agent_exit_code"] == 0
    assert changes["dirty_hash"]
    assert "diff" not in changes  # not captured without --capture-diff

    report = build_report(json.load(open(save)), save)
    assert "## Files the agent changed" in report
    assert "app.py" in report and "exited **0**" in report


def test_capture_diff_embeds_the_patch(tmp_path, monkeypatch):
    repo = _git_repo(tmp_path)
    monkeypatch.chdir(repo)
    (repo / "child.py").write_text(CHILD)

    upstream = _FakeUpstream([FINAL_ANSWER])
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    save = str(repo / "session.loom.json")
    main([
        "record", "--save", save, "--capture-diff",
        "--target", f"http://127.0.0.1:{upstream.server_address[1]}",
        "--", sys.executable, str(repo / "child.py"),
    ])
    upstream.shutdown()

    changes = json.load(open(save))["workspace"]["changes"]
    assert "edited by agent" in changes["diff"]  # the actual patch content
