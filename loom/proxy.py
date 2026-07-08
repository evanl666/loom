"""loom proxy: record ANY Anthropic-API agent -- no migration, no SDK, no code.

    loom proxy --save session.loom.json
    export ANTHROPIC_BASE_URL=http://localhost:8788
    # ...run Claude Code, a LangGraph app, or any Anthropic-SDK agent as usual

Everything an agent does is visible in its API traffic: tool calls ride in the
responses, tool results ride in the next request. So a plain recording proxy
reconstructs a full loom trace -- `loom timeline`, `loom export`, and
`loom doctor` work on any agent's session, not just agents built on Loom.

Replay mode serves the recorded wire responses back, byte-identical, with no
upstream and no API key:

    loom proxy --replay session.loom.json

Stdlib only, like the rest of the kernel.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .effect import EffectEntry, _key

DEFAULT_TARGET = "https://api.anthropic.com"
_FORWARD_HEADERS = {
    "x-api-key",
    "authorization",
    "anthropic-version",
    "anthropic-beta",
    "accept",
}


def _flatten(content) -> str:
    """Flatten an Anthropic content field (str or block list) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return str(content)


class WireRecorder:
    """Turns observed (request, response) pairs into a standard loom trace."""

    def __init__(self):
        self.log: list[EffectEntry] = []
        self.wire: list[dict] = []  # raw responses, for byte-identical replay
        self.episodes: list[str] = []
        self.model = ""
        self.system = ""
        self.output = ""
        self._tool_names: dict[str, str] = {}  # tool_use_id -> tool name
        self._seen_messages = 0

    def record(self, request: dict, response: dict) -> None:
        self.model = request.get("model", self.model)
        system = _flatten(request.get("system", ""))
        self.system = system or self.system
        self._absorb_request(request)
        self._absorb_response(response)
        self.wire.append(response)

    def _append(self, kind: str, payload, result) -> None:
        self.log.append(
            EffectEntry(seq=len(self.log), kind=kind, key=_key([kind, payload]), result=result)
        )

    def _absorb_request(self, request: dict) -> None:
        """New user text becomes episodes; new tool results become tool effects."""
        messages = request.get("messages", [])
        for m in messages[self._seen_messages :]:
            if m.get("role") != "user":
                continue  # assistant turns are the responses we already recorded
            content = m.get("content", "")
            blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
            for b in blocks:
                if not isinstance(b, dict):
                    b = {"type": "text", "text": str(b)}
                if b.get("type") == "tool_result":
                    name = self._tool_names.get(b.get("tool_use_id", ""), "tool")
                    self._append(
                        f"tool:{name}",
                        {"id": b.get("tool_use_id", "")},
                        _flatten(b.get("content", "")),
                    )
                elif b.get("type") == "text" and b.get("text"):
                    self.episodes.append(b["text"])
        self._seen_messages = len(messages)

    def _absorb_response(self, response: dict) -> None:
        text, tool_calls = "", []
        for b in response.get("content", []):
            if b.get("type") == "text":
                text += b.get("text", "")
            elif b.get("type") == "tool_use":
                self._tool_names[b.get("id", "")] = b.get("name", "tool")
                tool_calls.append(
                    {"id": b.get("id", ""), "name": b.get("name", ""), "input": b.get("input", {})}
                )
        usage = response.get("usage", {}) or {}
        result = {
            "text": text,
            "tool_calls": tool_calls,
            "stop_reason": "tool_use" if tool_calls else "end_turn",
            "usage": {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            },
        }
        self._append("model", {"n": len(self.wire)}, result)
        if text:
            self.output = text

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "recorded_via": "proxy",
            "model": self.model,
            "system": self.system,
            "prompt": self.episodes[0] if self.episodes else "",
            "episodes": self.episodes or [""],
            "output": self.output,
            "truncated": False,
            "paused": False,
            "pending": None,
            "pending_depth": 0,
            "stop_reason": "",
            "log": [e.to_dict() for e in self.log],
            "wire": self.wire,
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def reconstruct_sse(raw: str) -> dict:
    """Rebuild the final message from an Anthropic SSE stream transcript."""
    blocks: dict[int, dict] = {}
    partial: dict[int, str] = {}
    usage: dict = {}
    envelope: dict = {}
    stop_reason = None
    for line in raw.splitlines():
        if not line.startswith("data:"):
            continue
        try:
            event = json.loads(line[5:].strip())
        except json.JSONDecodeError:
            continue
        etype = event.get("type")
        if etype == "message_start":
            message = event.get("message", {}) or {}
            envelope = {k: v for k, v in message.items() if k not in ("content", "usage")}
            usage.update(message.get("usage", {}) or {})
        elif etype == "content_block_start":
            blocks[event["index"]] = dict(event.get("content_block", {}))
            partial[event["index"]] = ""
        elif etype == "content_block_delta":
            delta = event.get("delta", {})
            if delta.get("type") == "text_delta":
                blocks[event["index"]]["text"] = (
                    blocks[event["index"]].get("text", "") + delta.get("text", "")
                )
            elif delta.get("type") == "input_json_delta":
                partial[event["index"]] += delta.get("partial_json", "")
        elif etype == "message_delta":
            stop_reason = event.get("delta", {}).get("stop_reason", stop_reason)
            usage.update(event.get("usage", {}) or {})
    content = []
    for i in sorted(blocks):
        b = blocks[i]
        if b.get("type") == "tool_use" and partial.get(i):
            try:
                b["input"] = json.loads(partial[i])
            except json.JSONDecodeError:
                b["input"] = {}
        content.append(b)
    return {**envelope, "content": content, "stop_reason": stop_reason, "usage": usage}


def synthesize_sse(message: dict) -> bytes:
    """Render a recorded message as an Anthropic SSE stream, for replay clients."""

    def event(etype: str, payload: dict) -> bytes:
        return f"event: {etype}\ndata: {json.dumps({'type': etype, **payload})}\n\n".encode()

    envelope = {k: v for k, v in message.items() if k not in ("content", "stop_reason", "usage")}
    envelope.setdefault("type", "message")
    envelope.setdefault("role", "assistant")
    envelope.setdefault("id", "msg_replay")
    usage = message.get("usage", {}) or {}
    out = [
        event(
            "message_start",
            {"message": {**envelope, "content": [], "stop_reason": None, "usage": usage}},
        )
    ]
    for i, block in enumerate(message.get("content", [])):
        if block.get("type") == "tool_use":
            start = {k: block.get(k) for k in ("type", "id", "name")}
            start["input"] = {}
            out.append(event("content_block_start", {"index": i, "content_block": start}))
            out.append(
                event(
                    "content_block_delta",
                    {
                        "index": i,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": json.dumps(block.get("input", {})),
                        },
                    },
                )
            )
        else:
            out.append(
                event("content_block_start", {"index": i, "content_block": {"type": "text", "text": ""}})
            )
            out.append(
                event(
                    "content_block_delta",
                    {"index": i, "delta": {"type": "text_delta", "text": block.get("text", "")}},
                )
            )
        out.append(event("content_block_stop", {"index": i}))
    out.append(
        event(
            "message_delta",
            {
                "delta": {"stop_reason": message.get("stop_reason"), "stop_sequence": None},
                "usage": {"output_tokens": usage.get("output_tokens", 0)},
            },
        )
    )
    out.append(event("message_stop", {}))
    return b"".join(out)


class ProxyServer(ThreadingHTTPServer):
    """The recording (or replaying) proxy. Bind port 0 to pick a free port."""

    daemon_threads = True

    def __init__(self, port: int = 8788, target: str = DEFAULT_TARGET,
                 save_path: "str | None" = None, replay_path: "str | None" = None):
        self.target = target.rstrip("/")
        self.save_path = save_path
        self.recorder = WireRecorder()
        self.lock = threading.Lock()
        self.replay_wire: "list[dict] | None" = None
        self.replay_index = 0
        if replay_path:
            with open(replay_path) as f:
                data = json.load(f)
            if "wire" not in data:
                raise ValueError(f"{replay_path} has no wire responses (not a proxy trace)")
            self.replay_wire = data["wire"]
        super().__init__(("127.0.0.1", port), _Handler)

    @property
    def port(self) -> int:
        return self.server_address[1]

    def server_bind(self) -> None:
        # HTTPServer.server_bind calls socket.getfqdn(), which can stall for
        # seconds on machines with slow reverse DNS. We know who we are.
        import socketserver

        socketserver.TCPServer.server_bind(self)
        self.server_name = "127.0.0.1"
        self.server_port = self.server_address[1]


class _Handler(BaseHTTPRequestHandler):
    server: ProxyServer

    def log_message(self, *args) -> None:  # silence per-request stderr noise
        pass

    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_sse(self, payload: bytes) -> None:
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", 0))
        request = json.loads(self.rfile.read(length) or b"{}")
        wants_stream = bool(request.get("stream"))

        if self.server.replay_wire is not None:
            with self.server.lock:
                if self.server.replay_index >= len(self.server.replay_wire):
                    self._send_json(410, {"error": "replay exhausted: no more recorded responses"})
                    return
                response = self.server.replay_wire[self.server.replay_index]
                self.server.replay_index += 1
            if wants_stream:
                self._send_sse(synthesize_sse(response))
            else:
                self._send_json(200, response)
            return

        headers = {k: v for k, v in self.headers.items() if k.lower() in _FORWARD_HEADERS}
        headers["content-type"] = "application/json"
        upstream_req = urllib.request.Request(
            self.server.target + self.path,
            data=json.dumps(request).encode(),
            headers=headers,
            method="POST",
        )
        try:
            upstream = urllib.request.urlopen(upstream_req, timeout=600)
        except urllib.error.HTTPError as e:
            body = e.read()
            try:
                self._send_json(e.code, json.loads(body or b"{}"))
            except json.JSONDecodeError:
                self._send_json(e.code, {"error": body.decode("utf-8", "replace")})
            return

        with upstream:
            if upstream.headers.get("content-type", "").startswith("text/event-stream"):
                # Relay the stream as it arrives; reconstruct the message after.
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("cache-control", "no-cache")
                self.end_headers()
                chunks: list[bytes] = []
                while True:
                    chunk = upstream.read(1024)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    self.wfile.write(chunk)
                    self.wfile.flush()
                response = reconstruct_sse(b"".join(chunks).decode("utf-8", "replace"))
            else:
                data = upstream.read()
                response = json.loads(data)
                self._send_json(200, response)

        with self.server.lock:
            self.server.recorder.record(request, response)
            if self.server.save_path:
                self.server.recorder.save(self.server.save_path)

    def do_GET(self) -> None:
        """Pass non-messages endpoints through untouched (models list, health...)."""
        if self.server.replay_wire is not None:
            self._send_json(404, {"error": "replay mode serves recorded POSTs only"})
            return
        headers = {k: v for k, v in self.headers.items() if k.lower() in _FORWARD_HEADERS}
        try:
            with urllib.request.urlopen(
                urllib.request.Request(self.server.target + self.path, headers=headers),
                timeout=60,
            ) as upstream:
                body = upstream.read()
                self.send_response(upstream.status)
                self.send_header(
                    "content-type", upstream.headers.get("content-type", "application/json")
                )
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        except urllib.error.HTTPError as e:
            self._send_json(e.code, {"error": e.reason})
