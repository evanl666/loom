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
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .effect import EffectEntry, _key

DEFAULT_TARGET = "https://api.anthropic.com"
_FORWARD_HEADERS = {
    "x-api-key",
    "authorization",
    "anthropic-version",
    "anthropic-beta",
    "openai-organization",
    "openai-project",
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
        self.shield_events: list[dict] = []  # firewall decisions, in order
        self.episodes: list[str] = []
        self.model = ""
        self.system = ""
        self.output = ""
        self._tool_names: dict[str, str] = {}  # tool_use_id -> tool name
        self._seen_messages = 0

    def record(self, request: dict, response: dict) -> None:
        self.model = request.get("model", self.model)
        if "choices" in response:  # OpenAI chat-completions dialect
            self._absorb_request_openai(request)
            self._absorb_response_openai(response)
        else:  # Anthropic messages dialect
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

    def _absorb_request_openai(self, request: dict) -> None:
        messages = request.get("messages", [])
        for m in messages[self._seen_messages :]:
            role = m.get("role")
            if role == "system":
                self.system = _flatten(m.get("content", "")) or self.system
            elif role == "tool":
                name = self._tool_names.get(m.get("tool_call_id", ""), "tool")
                self._append(
                    f"tool:{name}",
                    {"id": m.get("tool_call_id", "")},
                    _flatten(m.get("content", "")),
                )
            elif role == "user":
                text = _flatten(m.get("content", ""))
                if text:
                    self.episodes.append(text)
        self._seen_messages = len(messages)

    def _absorb_response_openai(self, response: dict) -> None:
        message = (response.get("choices") or [{}])[0].get("message", {}) or {}
        text = _flatten(message.get("content") or "")
        tool_calls = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function", {}) or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            self._tool_names[tc.get("id", "")] = fn.get("name", "tool")
            tool_calls.append({"id": tc.get("id", ""), "name": fn.get("name", ""), "input": args})
        usage = response.get("usage", {}) or {}
        self._append(
            "model",
            {"n": len(self.wire)},
            {
                "text": text,
                "tool_calls": tool_calls,
                "stop_reason": "tool_use" if tool_calls else "end_turn",
                "usage": {
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                },
            },
        )
        if text:
            self.output = text

    def to_dict(self) -> dict:
        from .trace import TRACE_VERSION

        return {
            "version": TRACE_VERSION,
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
            "shield_events": self.shield_events,
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def compact_wirelog(wirelog_path: str, save_path: str) -> "WireRecorder":
    """Rebuild a full trace from an append-only wirelog (crash recovery).

    The proxy appends one JSON line per exchange before answering the client;
    if it dies before the periodic/final compaction, this turns what survived
    into a normal ``.loom.json``. A torn final line (crash mid-write) is
    ignored, like the journal's.
    """
    rec = WireRecorder()
    with open(wirelog_path) as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn tail
            rec.shield_events.extend(entry.get("shield_events", []))
            rec.record(entry["request"], entry["response"])
    rec.save(save_path)
    return rec


def _runtime_dir() -> str:
    """Per-user runtime state (control tokens): ``~/.loom/proxies``.

    ``LOOM_RUNTIME_DIR`` overrides it (tests point this at a tmp dir).
    """
    path = os.environ.get("LOOM_RUNTIME_DIR") or os.path.join(
        os.path.expanduser("~"), ".loom", "proxies"
    )
    os.makedirs(path, exist_ok=True)
    return path


def control_token_for(port: int) -> "str | None":
    """Read the control token a running shielded proxy registered for its port."""
    try:
        with open(os.path.join(_runtime_dir(), f"{port}.token")) as f:
            return f.read().strip() or None
    except OSError:
        return None


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


def reconstruct_openai_sse(raw: str) -> dict:
    """Rebuild the final chat completion from an OpenAI SSE transcript."""
    text = ""
    tool_calls: dict[int, dict] = {}
    finish_reason = None
    usage: dict = {}
    model = ""
    for line in raw.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        model = event.get("model") or model
        if event.get("usage"):
            usage = event["usage"]
        for choice in event.get("choices") or []:
            finish_reason = choice.get("finish_reason") or finish_reason
            delta = choice.get("delta", {}) or {}
            text += delta.get("content") or ""
            for tc in delta.get("tool_calls") or []:
                slot = tool_calls.setdefault(
                    tc.get("index", 0), {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                )
                slot["id"] = tc.get("id") or slot["id"]
                fn = tc.get("function", {}) or {}
                slot["function"]["name"] = fn.get("name") or slot["function"]["name"]
                slot["function"]["arguments"] += fn.get("arguments") or ""
    message: dict = {"role": "assistant", "content": text or None}
    if tool_calls:
        message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
    return {
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason or "stop"}],
        "usage": usage,
    }


def synthesize_openai_sse(response: dict) -> bytes:
    """Render a recorded chat completion as an OpenAI SSE stream, for replay."""
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message", {}) or {}
    base = {"object": "chat.completion.chunk", "model": response.get("model", "")}

    def chunk(delta: dict, finish=None) -> bytes:
        body = {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        return f"data: {json.dumps(body)}\n\n".encode()

    out = [chunk({"role": "assistant"})]
    if message.get("content"):
        out.append(chunk({"content": message["content"]}))
    for i, tc in enumerate(message.get("tool_calls") or []):
        out.append(chunk({"tool_calls": [{"index": i, "id": tc.get("id"), "type": "function",
                                          "function": tc.get("function", {})}]}))
    out.append(chunk({}, finish=choice.get("finish_reason", "stop")))
    if response.get("usage"):
        out.append(
            f"data: {json.dumps({**base, 'choices': [], 'usage': response['usage']})}\n\n".encode()
        )
    out.append(b"data: [DONE]\n\n")
    return b"".join(out)


class ProxyServer(ThreadingHTTPServer):
    """The recording (or replaying) proxy. Bind port 0 to pick a free port."""

    daemon_threads = True

    def __init__(self, port: int = 8788, target: str = DEFAULT_TARGET,
                 save_path: "str | None" = None, replay_path: "str | None" = None,
                 shield=None, scrub: bool = False,
                 save_interval: float = 5.0, eager_saves: int = 20,
                 max_body: int = 64 * 1024 * 1024, upstream_timeout: float = 600.0,
                 auth: str = ""):
        self.target = target.rstrip("/")
        self.save_path = save_path
        self.shield = shield  # loom.shield.Shield, screens tool calls in responses
        self.scrub = scrub  # redact secrets at the persist boundary (storage only)
        self.max_body = max_body  # 413 for anything larger; 0 disables the cap
        self.upstream_timeout = upstream_timeout
        # Optional data-plane auth: with a token set, /v1/* requests must carry
        # it in x-loom-auth. The control plane has its own per-session token;
        # this one guards the DATA plane -- above all replay mode, where the
        # proxy would otherwise serve a recorded conversation to any local
        # process that asks. Only agents that can add a header can use it.
        self.auth = auth
        self.recorder = WireRecorder()
        self.lock = threading.Lock()
        self.replay_wire: "list[dict] | None" = None
        self.replay_index = 0
        # Persistence: every exchange is APPENDED to <save>.wirelog (flushed)
        # before the client gets its answer -- O(1), crash-safe. The readable
        # trace is written through for the first `eager_saves` exchanges (small
        # sessions stay instantly inspectable), then at most every
        # `save_interval` seconds, and finally on finalize(). Rewriting the
        # whole JSON per exchange was O(n^2) and sat on the critical path.
        self.save_interval = save_interval
        self.eager_saves = eager_saves
        self._wirelog = None
        self._last_save = 0.0
        self._finalized = False
        if replay_path:
            with open(replay_path) as f:
                data = json.load(f)
            if "wire" not in data:
                raise ValueError(f"{replay_path} has no wire responses (not a proxy trace)")
            self.replay_wire = data["wire"]
        super().__init__(("127.0.0.1", port), _Handler)
        self.control_token: "str | None" = None
        if shield is not None and replay_path is None:
            # Without a token, any local process -- including a webpage's JS
            # doing fetch() against 127.0.0.1, or the shielded agent itself --
            # could approve its own pending tool calls. The token lives in a
            # user-only file; `loom approve` picks it up automatically.
            self.control_token = uuid.uuid4().hex
            self._token_path = os.path.join(_runtime_dir(), f"{self.port}.token")
            fd = os.open(self._token_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                f.write(self.control_token)

    @property
    def port(self) -> int:
        return self.server_address[1]

    def persist(self, request: dict, response: dict, events: "list[dict]") -> None:
        """Store one exchange durably BEFORE the client sees the response."""
        if self.scrub:
            from .scrub import scrub_obj

            request, _ = scrub_obj(request)
            response, _ = scrub_obj(response)
            events, _ = scrub_obj(events)
        with self.lock:
            self.recorder.shield_events.extend(events)
            self.recorder.record(request, response)
            if not self.save_path:
                return
            if self._wirelog is None:
                self._wirelog = open(self.save_path + ".wirelog", "a")
            self._wirelog.write(
                json.dumps({"request": request, "response": response, "shield_events": events})
                + "\n"
            )
            self._wirelog.flush()
            now = time.time()
            if len(self.recorder.wire) <= self.eager_saves or now - self._last_save >= self.save_interval:
                self.recorder.save(self.save_path)
                self._last_save = now

    def finalize(self) -> None:
        """Write the final trace and clean up runtime files. Idempotent."""
        with self.lock:
            if self._finalized:
                return
            self._finalized = True
            if self.save_path and self.recorder.wire:
                self.recorder.save(self.save_path)
            if self._wirelog is not None:
                self._wirelog.close()
                try:
                    os.remove(self.save_path + ".wirelog")
                except OSError:
                    pass
            if self.control_token:
                try:
                    os.remove(self._token_path)
                except OSError:
                    pass

    def server_close(self) -> None:
        super().server_close()
        self.finalize()

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

    def _control_authorized(self) -> bool:
        token = self.server.control_token
        return bool(token) and self.headers.get("x-loom-token", "") == token

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
        if self.server.max_body and length > self.server.max_body:
            self._send_json(413, {
                "error": f"request body {length} bytes exceeds the proxy cap "
                         f"({self.server.max_body}); raise it with --max-body-mb"
            })
            return
        request = json.loads(self.rfile.read(length) or b"{}")
        wants_stream = bool(request.get("stream"))

        if (self.server.auth and not self.path.startswith("/loom/")
                and self.headers.get("x-loom-auth", "") != self.server.auth):
            self._send_json(401, {"error": "missing or wrong x-loom-auth header"})
            return

        if self.path == "/loom/shield/decide":
            shield = self.server.shield
            if shield is None:
                self._send_json(404, {"error": "no shield active on this proxy"})
            elif not self._control_authorized():
                self._send_json(403, {"error": "missing or wrong x-loom-token header"})
            elif shield.decide_pending(str(request.get("id", "")), request.get("decision") == "approve"):
                self._send_json(200, {"ok": True})
            else:
                self._send_json(404, {"error": f"no pending approval {request.get('id')!r}"})
            return

        if self.server.replay_wire is not None:
            with self.server.lock:
                if self.server.replay_index >= len(self.server.replay_wire):
                    self._send_json(410, {"error": "replay exhausted: no more recorded responses"})
                    return
                response = self.server.replay_wire[self.server.replay_index]
                self.server.replay_index += 1
            if wants_stream:
                synth = synthesize_openai_sse if "choices" in response else synthesize_sse
                self._send_sse(synth(response))
            else:
                self._send_json(200, response)
            return

        # Taint tripwires: results of the last round's tool calls ride in this
        # request -- the earliest moment the wire can see what a tool returned.
        taint_events = (
            self.server.shield.observe_request(request)
            if self.server.shield is not None
            else []
        )

        headers = {k: v for k, v in self.headers.items() if k.lower() in _FORWARD_HEADERS}
        headers["content-type"] = "application/json"
        upstream_req = urllib.request.Request(
            self.server.target + self.path,
            data=json.dumps(request).encode(),
            headers=headers,
            method="POST",
        )
        try:
            upstream = urllib.request.urlopen(upstream_req, timeout=self.server.upstream_timeout)
        except urllib.error.HTTPError as e:
            body = e.read()
            try:
                self._send_json(e.code, json.loads(body or b"{}"))
            except json.JSONDecodeError:
                self._send_json(e.code, {"error": body.decode("utf-8", "replace")})
            return

        shield = self.server.shield
        with upstream:
            streamed = upstream.headers.get("content-type", "").startswith("text/event-stream")
            if streamed and shield is None:
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
                raw = b"".join(chunks).decode("utf-8", "replace")
                if "chat/completions" in self.path:
                    response = reconstruct_openai_sse(raw)
                else:
                    response = reconstruct_sse(raw)
            elif streamed:
                # A shield must see the whole response BEFORE the client does,
                # so buffer the upstream stream and synthesize one afterwards.
                raw = upstream.read().decode("utf-8", "replace")
                if "chat/completions" in self.path:
                    response = reconstruct_openai_sse(raw)
                else:
                    response = reconstruct_sse(raw)
            else:
                response = json.loads(upstream.read())

        events: list = list(taint_events)
        if shield is not None:
            # May block awaiting a human decision on a confirm rule; the
            # client sees (and the trace records) only the screened response.
            response, screen_events = shield.screen(response)
            events.extend(screen_events)

        # Persist BEFORE answering: when the client sees the reply, the
        # exchange is already on disk (wirelog append + throttled trace write).
        self.server.persist(request, response, events)
        if streamed and shield is not None:
            synth = synthesize_openai_sse if "choices" in response else synthesize_sse
            self._send_sse(synth(response))
        elif not streamed:
            self._send_json(200, response)

    def do_GET(self) -> None:
        """Pass non-messages endpoints through untouched (models list, health...)."""
        if self.path == "/loom/shield/pending":
            shield = self.server.shield
            if shield is None:
                self._send_json(404, {"error": "no shield active on this proxy"})
            elif not self._control_authorized():
                self._send_json(403, {"error": "missing or wrong x-loom-token header"})
            else:
                self._send_json(200, {"pending": shield.pending_list()})
            return
        if self.server.auth and self.headers.get("x-loom-auth", "") != self.server.auth:
            self._send_json(401, {"error": "missing or wrong x-loom-auth header"})
            return
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
