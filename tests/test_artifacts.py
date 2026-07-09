"""External artifact store: externalize/inline oversized tool results."""

import json

from loom import Agent, tool
from loom.artifacts import externalize, inline, artifact_pointer
from loom.cli import main
from loom.providers import ModelResponse, ScriptedProvider, ToolCall


@tool
def screenshot(url: str) -> str:
    "shot"
    return "X" * 50_000


def _big_run():
    return Agent(model=ScriptedProvider([
        ModelResponse(tool_calls=[ToolCall("t", "screenshot", {"url": "x"})], stop_reason="tool_use"),
        ModelResponse(text="done"),
    ]), tools=[screenshot]).run("go")


def test_externalize_then_inline_roundtrips(tmp_path):
    data = _big_run().to_dict()
    blobdir = str(tmp_path / "blobs")
    small, manifest = externalize(data, blobdir, threshold=32_768)
    assert len(manifest) == 1 and manifest[0]["bytes"] == 50_000
    # the trace now holds a pointer, not the content
    ptr = artifact_pointer(small["log"][1]["result"])
    assert ptr and ptr["kind"] == "tool:screenshot"
    assert len(json.dumps(small)) < len(json.dumps(data)) / 10   # much smaller

    restored, missing = inline(small, blobdir)
    assert not missing
    assert restored["log"][1]["result"] == "X" * 50_000          # byte-perfect


def test_small_results_are_left_inline(tmp_path):
    data = {"log": [{"seq": 0, "kind": "tool:x", "key": "k", "result": "small"}]}
    out, manifest = externalize(data, str(tmp_path / "b"), threshold=32_768)
    assert manifest == [] and out["log"][0]["result"] == "small"


def test_inline_reports_missing_blobs(tmp_path):
    data = _big_run().to_dict()
    small, _ = externalize(data, str(tmp_path / "blobs"), threshold=32_768)
    _restored, missing = inline(small, str(tmp_path / "empty"))   # wrong dir
    assert len(missing) == 1


def test_studio_renders_the_pointer(tmp_path):
    from loom.export import trace_to_html

    small, _ = externalize(_big_run().to_dict(), str(tmp_path / "b"), threshold=32_768)
    html = trace_to_html(small)
    assert "externalized artifact" in html and "KB" in html
    assert "XXXX" not in html   # the blob itself is NOT in the page


def test_cli_externalize_and_inline(tmp_path, capsys):
    trace = str(tmp_path / "r.loom.json")
    _big_run().save(trace)
    before = len(open(trace).read())
    assert main(["artifacts", "externalize", trace, "--threshold", "32kb"]) == 0
    assert "externalized 1 artifact" in capsys.readouterr().out
    assert len(open(trace).read()) < before / 5

    assert main(["artifacts", "inline", trace]) == 0
    assert "inlined artifacts" in capsys.readouterr().out
    data = json.load(open(trace))
    assert data["log"][1]["result"] == "X" * 50_000
