"""Live Studio: the real-time viewer endpoints served by the proxy."""

import json
import threading
import urllib.error
import urllib.request

from loom.proxy import ProxyServer
from loom.shield import Shield
from tests.test_proxy import WEATHER_TOOL_USE, FINAL_ANSWER, _FakeUpstream


def _serve(server):
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _post(port, payload):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/messages",
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json", "x-api-key": "k"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def _get(port, path, token=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 headers={"x-loom-token": token} if token else {})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.read(), r.headers.get("content-type", "")


def test_live_page_and_state(tmp_path):
    upstream = _serve(_FakeUpstream([WEATHER_TOOL_USE, FINAL_ANSWER]))
    proxy = _serve(ProxyServer(port=0, target=f"http://127.0.0.1:{upstream.server_address[1]}",
                               save_path=str(tmp_path / "s.loom.json")))
    try:
        # the page renders and wires the state feed
        page, ctype = _get(proxy.port, "/loom/live")
        assert "text/html" in ctype and b"Loom Live" in page and b"/loom/live/state" in page

        # empty state before any traffic (no shield -> ungated)
        body, _ = _get(proxy.port, "/loom/live/state")
        state = json.loads(body)
        assert state["effects"] == [] and state["shield_denied"] == 0

        # drive one exchange; the effect shows up in the live feed
        _post(proxy.port, {"model": "m", "messages": [{"role": "user", "content": "weather?"}]})
        state = json.loads(_get(proxy.port, "/loom/live/state")[0])
        assert state["effects"] and any(e["kind"] == "model" for e in state["effects"])
        assert state["episodes"] == ["weather?"]
    finally:
        proxy.shutdown()
        upstream.shutdown()


def test_live_state_is_token_gated_with_a_shield(tmp_path):
    upstream = _serve(_FakeUpstream([WEATHER_TOOL_USE]))
    proxy = _serve(ProxyServer(port=0, target=f"http://127.0.0.1:{upstream.server_address[1]}",
                               save_path=str(tmp_path / "s.loom.json"),
                               shield=Shield(deny=["Never*"])))
    try:
        # no token -> 403 (the feed carries approve/deny controls)
        try:
            _get(proxy.port, "/loom/live/state")
            assert False, "expected 403"
        except urllib.error.HTTPError as e:
            assert e.code == 403
        # with the token -> ok
        body, _ = _get(proxy.port, "/loom/live/state", token=proxy.control_token)
        assert "effects" in json.loads(body)
    finally:
        proxy.shutdown()
        upstream.shutdown()


def test_live_shows_pending_approvals(tmp_path):
    import time

    upstream = _serve(_FakeUpstream([WEATHER_TOOL_USE, FINAL_ANSWER]))
    proxy = _serve(ProxyServer(port=0, target=f"http://127.0.0.1:{upstream.server_address[1]}",
                               save_path=str(tmp_path / "s.loom.json"),
                               shield=Shield(confirm=["get_weather*"], timeout=5)))
    token = proxy.control_token
    result = {}

    def client():
        result["resp"] = _post(proxy.port, {"model": "m",
                                            "messages": [{"role": "user", "content": "w"}]})
    t = threading.Thread(target=client)
    t.start()
    try:
        # the held call appears in live state's pending list
        pending = []
        deadline = time.time() + 5
        while not pending and time.time() < deadline:
            state = json.loads(_get(proxy.port, "/loom/live/state", token=token)[0])
            pending = state["pending"]
            time.sleep(0.05)
        assert pending and pending[0]["tool"] == "get_weather"

        # decide it via the same control plane the page uses
        req = urllib.request.Request(
            f"http://127.0.0.1:{proxy.port}/loom/shield/decide",
            data=json.dumps({"id": pending[0]["id"], "decision": "approve"}).encode(),
            headers={"content-type": "application/json", "x-loom-token": token}, method="POST")
        urllib.request.urlopen(req, timeout=5).close()
        t.join(timeout=5)
    finally:
        proxy.shutdown()
        upstream.shutdown()
