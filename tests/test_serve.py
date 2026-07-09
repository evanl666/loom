"""loom serve: the single-tenant team trace server."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from loom import Agent, tool
from loom.providers import ModelResponse, ScriptedProvider, ToolCall
from loom.serve import TraceServer


@pytest.fixture
def server(tmp_path):
    @tool
    def Read(file_path: str) -> str:
        "read"
        return "x"

    for name, target in [("env.loom.json", "/app/.env"), ("ok.loom.json", "src/x.py")]:
        Agent(model=ScriptedProvider([
            ModelResponse(tool_calls=[ToolCall("t1", "Read", {"file_path": target})],
                          stop_reason="tool_use"),
            ModelResponse(text="done"),
        ]), tools=[Read]).run(f"handle {target}").save(str(tmp_path / name))

    srv = TraceServer(str(tmp_path), port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()


def _get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.read().decode()


def test_index_lists_runs_and_search_filters(server):
    html = _get(server.url + "/")
    assert "Loom trace server" in html
    assert "env.loom.json" in html and "ok.loom.json" in html
    filtered = _get(server.url + "/?q=risk%3Asecret-read")
    assert "env.loom.json" in filtered and "ok.loom.json" not in filtered


def test_per_run_studio_and_incident_views(server):
    studio = _get(server.url + "/run?p=env.loom.json")
    assert "Actions — what it did, why, what changed" in studio  # the Action Debugger
    incident = _get(server.url + "/run?p=env.loom.json&view=incident")
    assert "Incident report" in incident


def test_api_returns_json_rows(server):
    rows = json.loads(_get(server.url + "/api/runs"))
    assert len(rows) == 2
    assert {r["path"].rsplit("/", 1)[-1] for r in rows} == {"env.loom.json", "ok.loom.json"}


def test_path_traversal_is_blocked(server):
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(server.url + "/run?p=../../etc/passwd")
    assert e.value.code == 404
