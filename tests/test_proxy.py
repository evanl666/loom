"""loom proxy: record any Anthropic-API agent's traffic into a loom trace."""

import json
import os
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


def test_reconstruct_sse_tolerates_a_delta_without_its_block_start():
    # A malformed/truncated upstream stream (a delta for an index that never
    # had a content_block_start) must not crash the reconstruction thread.
    raw = "\n".join([
        'data: {"type":"message_start","message":{"id":"m","role":"assistant","usage":{"input_tokens":1}}}',
        'data: {"type":"content_block_delta","index":5,"delta":{"type":"text_delta","text":"orphan"}}',
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}',
        'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}',
    ])
    msg = reconstruct_sse(raw)
    assert msg["content"] == [{"type": "text", "text": "hi"}]  # orphan delta dropped
    assert msg["stop_reason"] == "end_turn"


OPENAI_TOOL_RESPONSE = {
    "object": "chat.completion",
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Berlin"}'},
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 11, "completion_tokens": 6},
}
OPENAI_FINAL = {
    "object": "chat.completion",
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Rainy in Berlin."},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 25, "completion_tokens": 5},
}


def test_openai_dialect_records_a_loom_trace(tmp_path):
    upstream = _serve(_FakeUpstream([OPENAI_TOOL_RESPONSE, OPENAI_FINAL]))
    path = str(tmp_path / "openai.loom.json")
    proxy = _serve(
        ProxyServer(port=0, target=f"http://127.0.0.1:{upstream.server_address[1]}", save_path=path)
    )
    _post(
        proxy.port,
        {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Weather in Berlin?"},
            ],
        },
    )
    _post(
        proxy.port,
        {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Weather in Berlin?"},
                {"role": "assistant", "tool_calls": OPENAI_TOOL_RESPONSE["choices"][0]["message"]["tool_calls"]},
                {"role": "tool", "tool_call_id": "call_1", "content": "rain, 12C"},
            ],
        },
    )
    with open(path) as f:
        data = json.load(f)
    assert data["system"] == "You are helpful."
    assert data["episodes"] == ["Weather in Berlin?"]
    assert [e["kind"] for e in data["log"]] == ["model", "tool:get_weather", "model"]
    assert data["log"][0]["result"]["tool_calls"][0]["input"] == {"city": "Berlin"}
    assert data["log"][0]["result"]["usage"] == {"input_tokens": 11, "output_tokens": 6}
    assert data["output"] == "Rainy in Berlin."
    assert verify_trace(path) == []
    proxy.shutdown()
    upstream.shutdown()


def test_openai_sse_reconstruction_and_synthesis_roundtrip():
    from loom.proxy import reconstruct_openai_sse, synthesize_openai_sse

    raw = "\n".join(
        [
            'data: {"object": "chat.completion.chunk", "model": "gpt-4o", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]}',
            'data: {"object": "chat.completion.chunk", "model": "gpt-4o", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call_9", "function": {"name": "add", "arguments": "{\\"a\\""}}]}, "finish_reason": null}]}',
            'data: {"object": "chat.completion.chunk", "model": "gpt-4o", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": ": 1}"}}]}, "finish_reason": null}]}',
            'data: {"object": "chat.completion.chunk", "model": "gpt-4o", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}',
            "data: [DONE]",
        ]
    )
    msg = reconstruct_openai_sse(raw)
    tc = msg["choices"][0]["message"]["tool_calls"][0]
    assert tc["function"] == {"name": "add", "arguments": '{"a": 1}'}
    assert msg["choices"][0]["finish_reason"] == "tool_calls"

    # And a recorded completion synthesizes back into a parseable stream.
    stream = synthesize_openai_sse(OPENAI_FINAL).decode()
    assert "Rainy in Berlin." in stream and stream.rstrip().endswith("data: [DONE]")
    assert reconstruct_openai_sse(stream)["choices"][0]["message"]["content"] == "Rainy in Berlin."


# --- Google Gemini dialect (contents / systemInstruction / functionDeclarations) ---
GEMINI_TOOL_RESPONSE = {
    "candidates": [{"content": {"role": "model", "parts": [
        {"functionCall": {"name": "get_weather", "args": {"city": "Berlin"}}}]},
        "finishReason": "STOP"}],
    "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 6},
}
GEMINI_FINAL = {
    "candidates": [{"content": {"role": "model", "parts": [{"text": "Rainy in Berlin."}]},
                    "finishReason": "STOP"}],
    "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 4},
}


def _post_gemini(port, payload, stream=False):
    verb = "streamGenerateContent" if stream else "generateContent"
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1beta/models/gemini-2.0-flash:{verb}",
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json", "x-goog-api-key": "gk-test"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def test_gemini_dialect_records_a_loom_trace(tmp_path):
    from loom.multiagent import infer_agents

    upstream = _serve(_FakeUpstream([GEMINI_TOOL_RESPONSE, GEMINI_FINAL]))
    path = str(tmp_path / "gemini.loom.json")
    proxy = _serve(
        ProxyServer(port=0, target=f"http://127.0.0.1:{upstream.server_address[1]}", save_path=path)
    )
    sys_prompt = "You are a Research Specialist. Use get_weather."
    # native Gemini request: NO model in the body (it is in the URL), system in
    # systemInstruction, tools in functionDeclarations, history in contents.
    _post_gemini(proxy.port, {
        "systemInstruction": {"role": "system", "parts": [{"text": sys_prompt}]},
        "tools": [{"functionDeclarations": [{"name": "get_weather", "parameters": {}}]}],
        "contents": [{"role": "user", "parts": [{"text": "Weather in Berlin?"}]}],
    })
    _post_gemini(proxy.port, {
        "systemInstruction": {"role": "system", "parts": [{"text": sys_prompt}]},
        "tools": [{"functionDeclarations": [{"name": "get_weather", "parameters": {}}]}],
        "contents": [
            {"role": "user", "parts": [{"text": "Weather in Berlin?"}]},
            {"role": "model", "parts": [{"functionCall": {"name": "get_weather", "args": {"city": "Berlin"}}}]},
            {"role": "user", "parts": [{"functionResponse": {"name": "get_weather", "response": {"result": "rain, 12C"}}}]},
        ],
    })
    with open(path) as f:
        data = json.load(f)
    assert data["model"] == "gemini-2.0-flash"          # recovered from the URL
    assert data["system"] == sys_prompt
    # the FULL system prompt is kept once per agent (for display + fork editing)
    assert sys_prompt in data["systems"].values()
    assert data["episodes"] == ["Weather in Berlin?"]
    assert [e["kind"] for e in data["log"]] == ["model", "tool:get_weather", "model"]
    assert data["log"][0]["result"]["tool_calls"][0]["input"] == {"city": "Berlin"}
    assert data["log"][1]["result"] == "rain, 12C"      # functionResponse -> tool effect
    assert data["log"][0]["result"]["usage"] == {"input_tokens": 11, "output_tokens": 6}
    assert data["output"] == "Rainy in Berlin."
    # the same wire fingerprint drives multi-agent recovery, dialect-blind
    assert data["log"][0]["meta"]["sys_role"] == "Research Specialist"
    assert data["log"][0]["meta"]["tools"] == ["get_weather"]
    assert verify_trace(path) == []
    proxy.shutdown()
    upstream.shutdown()


def test_gemini_sse_reconstruction_and_synthesis_roundtrip():
    from loom.proxy import reconstruct_gemini_sse, synthesize_gemini_sse, _reconstruct_stream

    raw = "\n".join([
        'data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Rainy "}]}}]}',
        'data: {"candidates":[{"content":{"role":"model","parts":[{"text":"in Berlin."}]}}]}',
        'data: {"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":20,"candidatesTokenCount":4}}',
    ])
    msg = reconstruct_gemini_sse(raw)
    assert msg["candidates"][0]["content"]["parts"][0]["text"] == "Rainy in Berlin."
    assert msg["usageMetadata"]["candidatesTokenCount"] == 4

    # a tool call split across chunks, routed via the streaming path (capital G)
    fc_raw = 'data: {"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"add","args":{"a":1}}}]},"finishReason":"STOP"}]}'
    routed = _reconstruct_stream("/v1beta/models/gemini-2.0-flash:streamGenerateContent", fc_raw)
    assert routed["candidates"][0]["content"]["parts"][0]["functionCall"]["name"] == "add"

    # a recorded response synthesizes back into a parseable Gemini stream
    stream = synthesize_gemini_sse(GEMINI_FINAL).decode()
    assert reconstruct_gemini_sse(stream)["candidates"][0]["content"]["parts"][0]["text"] == "Rainy in Berlin."


def test_real_openai_sdk_through_the_proxy(tmp_path):
    openai = pytest.importorskip("openai")

    upstream = _serve(_FakeUpstream([OPENAI_FINAL]))
    path = str(tmp_path / "sdk.loom.json")
    proxy = _serve(
        ProxyServer(port=0, target=f"http://127.0.0.1:{upstream.server_address[1]}", save_path=path)
    )
    client = openai.OpenAI(base_url=f"http://127.0.0.1:{proxy.port}/v1", api_key="sk-fake")
    reply = client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "Weather?"}]
    )
    assert reply.choices[0].message.content == "Rainy in Berlin."
    with open(path) as f:
        assert json.load(f)["output"] == "Rainy in Berlin."

    # Now replay through the SDK, streaming, with the fake upstream GONE.
    upstream.shutdown()
    replayer = _serve(ProxyServer(port=0, replay_path=path))
    client2 = openai.OpenAI(base_url=f"http://127.0.0.1:{replayer.port}/v1", api_key="sk-fake")
    stream = client2.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "Weather?"}], stream=True
    )
    text = "".join(c.choices[0].delta.content or "" for c in stream if c.choices)
    assert text == "Rainy in Berlin."
    proxy.shutdown()
    replayer.shutdown()


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

def test_malformed_requests_get_clean_http_errors_not_crashes():
    """A misbehaving client (bad content-length, non-JSON body, non-object body)
    must get a 4xx, not crash the handler with a traceback + connection reset."""
    import http.client

    proxy = _serve(ProxyServer(port=0, target="http://127.0.0.1:59999"))
    try:
        # bad content-length
        c = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=5)
        c.putrequest("POST", "/v1/messages")
        c.putheader("content-length", "abc")
        c.endheaders()
        assert c.getresponse().status == 400
        c.close()

        # malformed JSON body
        c = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=5)
        c.request("POST", "/v1/messages", body=b"{not json")
        assert c.getresponse().status == 400
        c.close()

        # valid JSON but not an object
        c = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=5)
        c.request("POST", "/v1/messages", body=b"[1,2,3]")
        assert c.getresponse().status == 400
        c.close()
    finally:
        proxy.shutdown()


def test_upstream_non_json_response_is_a_502_not_a_crash():
    """If upstream returns 200 with a non-JSON body, surface a gateway error."""
    import http.client

    class _HtmlUpstream(ThreadingHTTPServer):
        def __init__(self):
            super().__init__(("127.0.0.1", 0), _HtmlHandler)

    class _HtmlHandler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("content-length", 0)))
            self.send_response(200)
            self.send_header("content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html>edge error</html>")

    upstream = _serve(_HtmlUpstream())
    proxy = _serve(ProxyServer(port=0, target=f"http://127.0.0.1:{upstream.server_address[1]}"))
    try:
        c = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=5)
        c.request("POST", "/v1/messages", body=b'{"stream": false}')
        assert c.getresponse().status == 502
        c.close()
    finally:
        proxy.shutdown()
        upstream.shutdown()


def test_sse_reconstruction_tolerates_non_dict_events():
    """A stream carrying `data: null` or a bare scalar (or non-dict choices/tool
    calls) must be skipped, not crash the reconstruction with AttributeError."""
    from loom.proxy import reconstruct_openai_sse, reconstruct_sse

    adversarial = [
        "data: null",
        "data: 123",
        'data: "hi"',
        "data: [1,2,3]",
        'data: {"choices":[null,123]}',
        'data: {"choices":[{"delta":{"tool_calls":[null,7]}}]}',
    ]
    for raw in adversarial:
        reconstruct_sse(raw)  # must not raise
        reconstruct_openai_sse(raw)  # must not raise


def test_unreachable_upstream_is_a_502_not_a_crash(monkeypatch):
    """When the upstream API can't be reached (network down / wrong target /
    DNS failure), both POST and GET must return 502, not crash the handler and
    reset the client connection."""
    import http.client

    # Neutralize any system/env proxy so urllib really hits the closed port
    # (a local proxy would otherwise answer with its own error).
    for k in list(__import__("os").environ):
        if k.lower().endswith("_proxy"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("no_proxy", "*")
    monkeypatch.setenv("NO_PROXY", "*")

    proxy = _serve(ProxyServer(port=0, target="http://127.0.0.1:1"))  # port 1: refused
    try:
        c = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=5)
        c.request("POST", "/v1/messages", body=b'{"stream": false}')
        assert c.getresponse().status == 502
        c.close()

        c = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=5)
        c.request("GET", "/v1/models")
        assert c.getresponse().status == 502
        c.close()
    finally:
        proxy.shutdown()


def test_upstream_that_drops_the_connection_is_a_502_not_a_crash(monkeypatch):
    """A real upstream can accept the socket then close it without a response
    (http.client.RemoteDisconnected -- a ConnectionError/OSError, NOT a
    urllib URLError). The handler must turn that into a 502, not crash."""
    import http.client
    import os
    import socket
    import threading

    # Neutralize any system/env proxy so the loopback outbound really reaches
    # our dead server (a local proxy would answer with its own error first).
    for k in list(os.environ):
        if k.lower().endswith("_proxy"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("no_proxy", "*")
    monkeypatch.setenv("NO_PROXY", "*")

    # A bare TCP server that accepts a connection and immediately closes it.
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(5)
    dead_port = srv.getsockname()[1]

    def accept_and_drop():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            conn.close()  # RemoteDisconnected on the client (loom proxy) side

    threading.Thread(target=accept_and_drop, daemon=True).start()

    proxy = _serve(ProxyServer(port=0, target=f"http://127.0.0.1:{dead_port}"))
    try:
        c = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=5)
        c.request("POST", "/v1/messages", body=b'{"stream": false}')
        assert c.getresponse().status == 502
        c.close()
    finally:
        proxy.shutdown()
        srv.close()


# --- proxy-level fork for external agents (replay prefix, live+edited tail) ---
class _EchoUpstream(ThreadingHTTPServer):
    """A lenient fake API: echoes the system it received, never runs out."""

    def __init__(self):
        self.seen = []
        super().__init__(("127.0.0.1", 0), _EchoHandler)

    def server_bind(self):
        import socketserver
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = "127.0.0.1", self.server_address[1]


class _EchoHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        self.server.seen.append(req)
        body = json.dumps({"content": [{"type": "text", "text": f"LIVE:{req.get('system')}"}],
                           "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(body)


def _rec(t):
    return {"content": [{"type": "text", "text": t}], "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1}}


def test_proxy_fork_matches_prefix_by_content_then_lives_with_edited_system():
    from loom.proxy import request_fingerprint, request_sys_hash

    def req(system, msg):
        return {"model": "claude", "system": system, "messages": [{"role": "user", "content": msg}]}

    up = _serve(_EchoUpstream())
    proxy = ProxyServer(port=0, target=f"http://127.0.0.1:{up.server_address[1]}", save_path=None)
    # only turn 0 is in the prefix pool, keyed BY CONTENT (order-independent)
    proxy.fork = {"prefix": {request_fingerprint(req("OLD", "turn0")): [_rec("PREFIX0")]},
                  "inject_key": None, "edit_sys_hash": request_sys_hash(req("OLD", "")),
                  "new_system": "NEW", "append": None, "model": "keep", "_injected": False}
    _serve(proxy)

    def post(r):
        rq = urllib.request.Request(
            f"http://127.0.0.1:{proxy.port}/v1/messages", data=json.dumps(r).encode(),
            headers={"content-type": "application/json", "x-api-key": "k"}, method="POST")
        return json.loads(urllib.request.urlopen(rq, timeout=5).read())

    r0 = post(req("OLD", "turn0"))    # matches the prefix by content -> replayed free
    r1 = post(req("OLD", "turn1"))    # not in the prefix -> live, edited agent's system swapped
    r2 = post(req("OLD", "turn2"))
    assert r0["content"][0]["text"] == "PREFIX0"
    assert r1["content"][0]["text"] == "LIVE:NEW" and r2["content"][0]["text"] == "LIVE:NEW"
    assert [s.get("system") for s in up.seen] == ["NEW", "NEW"]   # only the tail hit the API
    proxy.shutdown()
    up.shutdown()


def test_proxy_fork_content_match_is_order_independent():
    """The re-run may make its prefix calls in a DIFFERENT order; content matching
    still serves each the right recorded response (positional replay could not)."""
    from loom.proxy import request_fingerprint

    def req(msg):
        return {"model": "claude", "system": "S", "messages": [{"role": "user", "content": msg}]}

    up = _serve(_EchoUpstream())
    proxy = ProxyServer(port=0, target=f"http://127.0.0.1:{up.server_address[1]}", save_path=None)
    proxy.fork = {"prefix": {request_fingerprint(req("A")): [_rec("RESP-A")],
                             request_fingerprint(req("B")): [_rec("RESP-B")]},
                  "inject_key": None, "edit_sys_hash": None, "new_system": None,
                  "append": None, "model": "keep", "_injected": False}
    _serve(proxy)

    def post(msg):
        rq = urllib.request.Request(
            f"http://127.0.0.1:{proxy.port}/v1/messages", data=json.dumps(req(msg)).encode(),
            headers={"content-type": "application/json", "x-api-key": "k"}, method="POST")
        return json.loads(urllib.request.urlopen(rq, timeout=5).read())

    assert post("B")["content"][0]["text"] == "RESP-B"   # reversed order -> still correct
    assert post("A")["content"][0]["text"] == "RESP-A"
    assert up.seen == []                                  # both served from the recording
    proxy.shutdown()
    up.shutdown()


def test_fork_wire_edits_across_dialects():
    from loom.proxy import _apply_fork_edits, _wire_read_system, request_sys_hash

    a = {"system": "OLD", "messages": [{"role": "user", "content": "hi"}]}
    fk = {"new_system": "NEW", "edit_sys_hash": request_sys_hash(a), "append": "more", "model": "keep"}
    _apply_fork_edits(a, fk, is_inject=True, path="/v1/messages")
    assert a["system"] == "NEW" and a["messages"][-1] == {"role": "user", "content": "more"}

    o = {"messages": [{"role": "system", "content": "OLD"}, {"role": "user", "content": "hi"}]}
    fko = {"new_system": "NEW", "edit_sys_hash": request_sys_hash(o), "model": "keep"}
    _apply_fork_edits(o, fko, is_inject=False, path="/v1/chat/completions")
    assert _wire_read_system(o) == "NEW"

    g = {"systemInstruction": {"parts": [{"text": "OLD"}]}, "contents": [{"role": "user", "parts": [{"text": "hi"}]}]}
    fkg = {"new_system": "NEW", "edit_sys_hash": request_sys_hash(g), "append": "more", "model": "other"}
    _apply_fork_edits(g, fkg, is_inject=True, path="/v1beta/models/gemini:streamGenerateContent")
    assert _wire_read_system(g) == "NEW"
    assert g["contents"][-1]["parts"][0]["text"] == "more"
    assert "model" not in g                               # Gemini's model lives in the URL

    peer = {"system": "PEER", "messages": []}             # a different agent (sys_hash) -> untouched
    _apply_fork_edits(peer, fk, is_inject=False, path="/v1/messages")
    assert peer["system"] == "PEER"


def test_fork_external_reruns_adapter_in_a_subprocess(tmp_path, monkeypatch):
    import time
    from loom.livesession import LiveSession, start_proxy

    (tmp_path / "fk_agent.py").write_text(
        "import json, os, urllib.request\n"
        "def run(prompt):\n"
        "    outs=[]\n"
        "    for i in range(3):\n"
        "        base=os.environ['ANTHROPIC_BASE_URL']\n"
        "        req=urllib.request.Request(base+'/v1/messages',\n"
        "            data=json.dumps({'model':'claude','system':'ORIG','messages':[{'role':'user','content':f'{prompt}{i}'}]}).encode(),\n"
        "            headers={'content-type':'application/json','x-api-key':'k'}, method='POST')\n"
        "        outs.append(json.loads(urllib.request.urlopen(req,timeout=10).read())['content'][0]['text'])\n"
        "    return ' '.join(outs)\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", ""))

    up = _serve(_EchoUpstream())
    target = f"http://127.0.0.1:{up.server_address[1]}"
    proxy = start_proxy(target)
    import fk_agent
    sess = LiveSession(func=fk_agent.run, proxy=proxy, spec="fk_agent:run", target=target)
    sess.ask("go")
    for _ in range(200):
        if not sess.running:
            break
        time.sleep(0.02)
    assert not sess.running and len(proxy.recorder.wire) == 3

    import hashlib
    edit_hash = hashlib.sha1(b"ORIG").hexdigest()[:12]     # the edited agent's fingerprint
    branch = sess.fork_external(1, {"new_system": "EDIT", "edit_sys_hash": edit_hash,
                                    "append": None, "model": "keep"})
    outs = [e["result"]["text"] for e in branch["log"] if e["kind"] == "model"]
    assert outs[0] == "LIVE:ORIG"                          # turn 0 matched the recorded prefix
    assert outs[1:] == ["LIVE:EDIT", "LIVE:EDIT"]          # tail re-ran live with the edit
    up.shutdown()
