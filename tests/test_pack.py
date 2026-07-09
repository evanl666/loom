"""loom pack: a self-contained, scrubbed incident bundle."""

import json
import zipfile

from loom import Agent
from loom.cli import main
from loom.pack import build_pack
from loom.providers import ModelResponse, ScriptedProvider

SECRET = "sk-ant-api03-" + "a1B2" * 8


def _trace(tmp_path, with_workspace=False):
    run = Agent(model=ScriptedProvider([ModelResponse(text=f"done, key {SECRET}")])).run("q")
    path = str(tmp_path / "run.loom.json")
    run.save(path)
    if with_workspace:
        data = json.load(open(path))
        data["workspace"] = {"os": "Linux", "git": {"commit": "abc123", "dirty": True},
                             "changes": {"diff": "--- a/x\n+++ b/x\n+edit\n",
                                          "files": [{"status": "M", "path": "x"}]}}
        json.dump(data, open(path, "w"))
    return path


def test_pack_bundles_everything_scrubbed(tmp_path):
    path = _trace(tmp_path, with_workspace=True)
    out, redacted = build_pack(path)
    assert out.endswith(".loompack") and redacted >= 1

    with zipfile.ZipFile(out) as z:
        names = set(z.namelist())
        assert names >= {"trace.loom.json", "incident.md", "studio.html",
                         "manifest.json", "README.md", "workspace.patch"}
        assert SECRET not in z.read("trace.loom.json").decode()   # scrubbed
        assert SECRET not in z.read("studio.html").decode()
        manifest = json.loads(z.read("manifest.json"))
        assert manifest["secrets_redacted"] >= 1
        assert manifest["workspace"]["git"]["commit"] == "abc123"
        assert "## Executive summary" in z.read("incident.md").decode()


def test_pack_without_workspace_omits_patch(tmp_path):
    out, _ = build_pack(_trace(tmp_path))
    with zipfile.ZipFile(out) as z:
        assert "workspace.patch" not in z.namelist()


def test_cli_pack(tmp_path, capsys):
    path = _trace(tmp_path)
    assert main(["pack", path]) == 0
    assert "packed ->" in capsys.readouterr().out
    assert (tmp_path / "run.loompack").exists()
