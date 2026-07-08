"""loom proxy: record any Anthropic-API agent's traffic into a loom trace."""

import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from loom import Run
from loom.proxy import ProxyServer, reconstruct_sse
from loom.testing import verify_trace

WEATHER_TOOL_USE = {
    "content": [
        {"type": "text", "text": "Let me check."},
        {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"city": "Berlin"}},
    ],
    "stop_reason": "tool_use",
    "usage": {"input_tokens": 12, "output_tokens": 8},
}
FINAL_ANSWER = {
    "content": [{"type": "text", "text": "It is raining in Berlin."}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 30, "output_tokens": 9},
}


class _FakeUpstream(ThreadingHTTPServer):
    """Stands in for api.anthropic.com: serves scripted responses in order."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests_seen = []
        super().__init__(("127.0.0.1", 0), _FakeHandler)

    def server_bind(self):
        import socketserver

        socketserver.TCPServer.server_bind(self)  # skip slow getfqdn()
        self.server_name = "127.0.0.1"
        self.server_port = self.server_address[1]


class _FakeHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("content-length", 0))
        self.server.requests_seen.append(json.loads(self.rfile.read(length)))
        body = json.dumps(self.server.responses.pop(0)).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _serve(server):
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


def _post(port, payload):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/messages",
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json", "x-api-key": "sk-test"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def drive_agent_conversation(port):
    """What any Anthropic-SDK agent's traffic looks like: two API calls."""
    first = _post(
        port,
        {
            "model": "claude-opus-4-8",
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "Weather in Berlin?"}],
        },
    )
    second = _post(
        port,
        {
            "model": "claude-opus-4-8",
            "system": "You are helpful.",
            "messages": [
                {"role": "user", "content": "Weather in Berlin?"},
                {"role": "assistant", "content": first["content"]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_1",
                            "content": "rain, 12C",
                        }
                    ],
                },
            ],
        },
    )
    return first, second


@pytest.fixture()
def recorded_trace(tmp_path):
    upstream = _serve(_FakeUpstream([WEATHER_TOOL_USE, FINAL_ANSWER]))
    path = str(tmp_path / "session.loom.json")
    proxy = _serve(
        ProxyServer(port=0, target=f"http://127.0.0.1:{upstream.server_address[1]}", save_path=path)
    )
    drive_agent_conversation(proxy.port)
    proxy.shutdown()
    upstream.shutdown()
    return path


def test_proxy_reconstructs_a_full_loom_trace(recorded_trace):
    with open(recorded_trace) as f:
        data = json.load(f)
    assert data["recorded_via"] == "proxy"
    assert data["episodes"] == ["Weather in Berlin?"]
    assert data["output"] == "It is raining in Berlin."
    kinds = [e["kind"] for e in data["log"]]
    assert kinds == ["model", "tool:get_weather", "model"]  # tool call recovered
    assert data["log"][1]["result"] == "rain, 12C"
    assert verify_trace(recorded_trace) == []  # standard tooling accepts it


def test_recorded_trace_works_with_run_load(recorded_trace):
    run = Run.load(recorded_trace)
    assert run.num_turns == 2
    assert run.cost()["total_tokens"] == 12 + 8 + 30 + 9
    assert run.bisect(lambda t: "raining" not in t) == 2  # trace tooling just works


def test_replay_serves_wire_responses_without_upstream(recorded_trace):
    proxy = _serve(ProxyServer(port=0, replay_path=recorded_trace))  # no target reachable
    first, second = drive_agent_conversation(proxy.port)
    assert first == WEATHER_TOOL_USE  # byte-identical, zero upstream calls
    assert second == FINAL_ANSWER
    third = urllib.request.Request(
        f"http://127.0.0.1:{proxy.port}/v1/messages", data=b"{}", method="POST"
    )
    try:
        urllib.request.urlopen(third, timeout=10)
        assert False, "expected 410 when the recording runs out"
    except urllib.error.HTTPError as e:
        assert e.code == 410
    proxy.shutdown()


def test_api_key_never_lands_in_the_trace(recorded_trace):
    with open(recorded_trace) as f:
        assert "sk-test" not in f.read()


SSE_STREAM = (
    'data: {"type": "message_start", "message": {"id": "msg_1", "type": "message", '
    '"role": "assistant", "model": "claude-opus-4-8", "usage": {"input_tokens": 4}}}\n\n'
    'data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}\n\n'
    'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Streamed hello."}}\n\n'
    'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 3}}\n\n'
    'data: {"type": "message_stop"}\n\n'
)


class _FakeSSEUpstream(_FakeUpstream):
    """Serves the canned SSE stream to any POST."""


class _FakeSSEHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length", 0)))
        body = SSE_STREAM.encode()
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_sse_passthrough_relays_and_records(tmp_path):
    upstream = _FakeUpstream([])
    upstream.RequestHandlerClass = _FakeSSEHandler
    _serve(upstream)
    path = str(tmp_path / "sse.loom.json")
    proxy = _serve(
        ProxyServer(port=0, target=f"http://127.0.0.1:{upstream.server_address[1]}", save_path=path)
    )
    req = urllib.request.Request(
        f"http://127.0.0.1:{proxy.port}/v1/messages",
        data=json.dumps(
            {"model": "claude-opus-4-8", "stream": True,
             "messages": [{"role": "user", "content": "hi"}]}
        ).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        raw = r.read().decode()
    assert "Streamed hello." in raw  # the client saw the live stream

    with open(path) as f:
        data = json.load(f)
    assert data["output"] == "Streamed hello."  # ...and the trace got the whole message
    assert data["wire"][0]["model"] == "claude-opus-4-8"  # envelope preserved
    proxy.shutdown()
    upstream.shutdown()


def test_replay_synthesizes_sse_for_streaming_clients(recorded_trace):
    proxy = _serve(ProxyServer(port=0, replay_path=recorded_trace))
    req = urllib.request.Request(
        f"http://127.0.0.1:{proxy.port}/v1/messages",
        data=json.dumps({"stream": True, "messages": []}).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        raw = r.read().decode()
    # The recorded tool_use response comes back as a well-formed event stream.
    assert "message_start" in raw and "message_stop" in raw
    assert '"name": "get_weather"' in raw
    from loom.proxy import reconstruct_sse

    round_tripped = reconstruct_sse(raw)
    assert round_tripped["content"][1]["input"] == {"city": "Berlin"}
    proxy.shutdown()


def test_sse_reconstruction():
    raw = "\n".join(
        [
            'data: {"type": "message_start", "message": {"usage": {"input_tokens": 5}}}',
            'data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}',
            'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}}',
            'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo"}}',
            'data: {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "t1", "name": "add"}}',
            'data: {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\\"a\\": 1"}}',
            'data: {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": ", \\"b\\": 2}"}}',
            'data: {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 7}}',
        ]
    )
    msg = reconstruct_sse(raw)
    assert msg["content"][0]["text"] == "Hello"
    assert msg["content"][1]["input"] == {"a": 1, "b": 2}
    assert msg["stop_reason"] == "tool_use"
    assert msg["usage"] == {"input_tokens": 5, "output_tokens": 7}