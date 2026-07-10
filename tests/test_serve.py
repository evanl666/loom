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
    studio = _get(server.url + "/run/studio?p=env.loom.json")
    assert "window.LOOM_STATIC=" in studio  # the embedded debugger UI
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


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def test_replay_room_comments_and_triage(server):
    base = server.url
    # the run page is a room: embedded studio + comment form
    room = _get(base + "/run?p=env.loom.json")
    assert "/run/studio?p=env.loom.json" in room
    assert "Step comments (0)" in room and "copy permalink" in room

    assert _post(base + "/api/notes?p=env.loom.json",
                 {"step": 1, "text": "the leak starts here", "by": "evan"})["ok"]
    assert _post(base + "/api/room?p=env.loom.json",
                 {"owner": "evan", "root_cause": "stale config", "resolved": True})["ok"]

    room2 = _get(base + "/run?p=env.loom.json")
    assert "the leak starts here" in room2 and "Step comments (1)" in room2
    assert "✓ resolved" in room2 and "stale config" in room2
    # the run list surfaces triage state
    idx = _get(base + "/")
    assert "✓ resolved" in idx and "👤 evan" in idx
    # the raw studio remains available for embedding
    assert "window.LOOM_STATIC=" in _get(base + "/run/studio?p=env.loom.json")


def test_room_notes_interoperate_with_the_cli(server, tmp_path):
    from loom.cli import main

    base = server.url
    _post(base + "/api/notes?p=ok.loom.json", {"step": 0, "text": "from the room", "by": "web"})
    # `loom note` reads the same sidecar
    trace = str(tmp_path / "ok.loom.json")
    assert main(["note", trace]) == 0


def test_post_traversal_and_bad_body_rejected(server):
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(server.url + "/api/notes?p=../../etc/passwd", {"text": "x"})
    assert e.value.code == 404
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(server.url + "/api/notes?p=env.loom.json", {"text": ""})
    assert e.value.code == 400


def test_concurrent_comments_do_not_drop(server):
    import concurrent.futures

    base = server.url

    def add(i):
        return _post(base + "/api/notes?p=env.loom.json",
                     {"step": i, "text": f"comment {i}", "by": "u"})

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(add, range(30)))

    room = _get(base + "/run?p=env.loom.json")
    assert "Step comments (30)" in room          # all 30 survived the race
