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
    "x-goog-api-key",       # Google Gemini auth
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


def _gemini_parts_text(parts) -> str:
    """Concatenate the text of a Gemini content ``parts`` list."""
    if not isinstance(parts, list):
        return ""
    return "".join(p.get("text", "") for p in parts if isinstance(p, dict) and "text" in p)


def _gemini_system(request: dict) -> str:
    """The system prompt from a Gemini request (camelCase or snake_case)."""
    si = request.get("systemInstruction") or request.get("system_instruction") or {}
    if isinstance(si, str):
        return si
    return _gemini_parts_text(si.get("parts")) if isinstance(si, dict) else ""


def _gemini_tools(request: dict) -> list:
    """Tool names declared in a Gemini request (functionDeclarations)."""
    names = []
    for t in request.get("tools") or []:
        if not isinstance(t, dict):
            continue
        decls = t.get("functionDeclarations") or t.get("function_declarations") or []
        for d in decls:
            if isinstance(d, dict) and d.get("name"):
                names.append(d["name"])
    return sorted(names)


def _gemini_result(resp) -> str:
    """A Gemini functionResponse payload as a readable string for a tool effect."""
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        for k in ("result", "output", "content", "text"):
            if isinstance(resp.get(k), str):
                return resp[k]
        return json.dumps(resp)
    return str(resp)


def _model_from_path(path: str) -> str:
    """Gemini puts the model in the URL (``/v1beta/models/gemini-x:generateContent``)."""
    if "/models/" not in path:
        return ""
    seg = path.split("/models/", 1)[1]
    return seg.split(":", 1)[0].split("?", 1)[0].split("/", 1)[0]


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
        self._pending_fp: dict = {}  # agent fingerprint for the model call being recorded
        self.workspace: "dict | None" = None  # cwd/git/argv/os, set by the recorder

    def record(self, request: dict, response: dict) -> None:
        self.model = request.get("model", self.model)
        if "choices" in response:  # OpenAI chat-completions dialect
            self._pending_fp = self._fingerprint(request, dialect="openai")
            self._absorb_request_openai(request)
            self._absorb_response_openai(response)
        elif "candidates" in response:  # Google Gemini dialect
            self._pending_fp = self._fingerprint(request, dialect="gemini")
            self._absorb_request_gemini(request)
            self._absorb_response_gemini(response)
        else:  # Anthropic messages dialect
            system = _flatten(request.get("system", ""))
            self.system = system or self.system
            self._pending_fp = self._fingerprint(request, dialect="anthropic")
            self._absorb_request(request)
            self._absorb_response(response)
        self.wire.append(response)

    def _fingerprint(self, request: dict, *, dialect: str) -> dict:
        """A per-model-call agent fingerprint recovered from the wire request.

        Parent/child agent structure lives in the application, invisible to a
        recording proxy -- but each sub-agent almost always has a distinct
        system prompt and tool set. Capturing (system hash + head, tool names,
        model) per call lets ``infer_agents`` reconstruct which sub-agent made
        each call for ANY framework, without the framework cooperating.

        The three wire dialects (Anthropic, OpenAI, Gemini) carry the same three
        facts in different shapes; we normalize them here so everything
        downstream (infer_agents, the debugger, the firewall) is dialect-blind."""
        import hashlib

        if dialect == "openai":
            system = ""
            for m in request.get("messages", []):
                if m.get("role") == "system":
                    system = _flatten(m.get("content", ""))
                    break
            tools = sorted(
                (t.get("function", {}) or {}).get("name", "")
                for t in (request.get("tools") or [])
            )
            n_msgs = len(request.get("messages") or [])
        elif dialect == "gemini":
            system = _gemini_system(request)
            tools = _gemini_tools(request)
            n_msgs = len(request.get("contents") or [])
        else:
            system = _flatten(request.get("system", ""))
            tools = sorted(t.get("name", "") for t in (request.get("tools") or []))
            n_msgs = len(request.get("messages") or [])
        system = system or ""
        from .multiagent import best_role

        fp = {
            "sys_hash": hashlib.sha1(system.encode()).hexdigest()[:12],
            "sys_head": system[:160],
            "tools": [t for t in tools if t],
            "model": request.get("model") or self.model,
            # how much conversation this call carried -- a FRESH sub-agent starts
            # small (just its task), a shared-conversation PEER (a group chat)
            # starts large. Lets infer_agents tell delegation from peer turns.
            "msgs": n_msgs,
        }
        role = best_role(system)  # scan the FULL prompt for the specific role
        if role:
            fp["sys_role"] = role
        return fp

    def _append(self, kind: str, payload, result, meta: "dict | None" = None) -> None:
        self.log.append(
            EffectEntry(seq=len(self.log), kind=kind, key=_key([kind, payload]),
                        result=result, meta=meta or {})
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
        self._append("model", {"n": len(self.wire)}, result, meta=self._pending_fp)
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
            meta=self._pending_fp,
        )
        if text:
            self.output = text

    def _absorb_request_gemini(self, request: dict) -> None:
        """Gemini keeps the turn history in ``contents`` (role user/model);
        function results ride as ``functionResponse`` parts in a user turn."""
        si = _gemini_system(request)
        if si:
            self.system = si or self.system
        contents = request.get("contents") or []
        for m in contents[self._seen_messages :]:
            if not isinstance(m, dict) or m.get("role") == "model":
                continue  # model turns are the responses we already recorded
            for p in m.get("parts") or []:
                if not isinstance(p, dict):
                    continue
                if "functionResponse" in p:
                    fr = p.get("functionResponse") or {}
                    name = fr.get("name", "tool")
                    resp = fr.get("response", fr)
                    self._append(f"tool:{name}", {"id": name}, _gemini_result(resp))
                elif p.get("text"):
                    self.episodes.append(p["text"])
        self._seen_messages = len(contents)

    def _absorb_response_gemini(self, response: dict) -> None:
        cand = (response.get("candidates") or [{}])[0] or {}
        parts = ((cand.get("content") or {}).get("parts")) or []
        text, tool_calls = "", []
        for p in parts:
            if not isinstance(p, dict):
                continue
            if "text" in p:
                text += p.get("text", "")
            elif "functionCall" in p:
                fc = p.get("functionCall") or {}
                name = fc.get("name", "")
                # Gemini function calls carry no id; the name keys the pairing.
                tool_calls.append({"id": name, "name": name, "input": fc.get("args", {}) or {}})
        usage = response.get("usageMetadata", {}) or {}
        self._append(
            "model",
            {"n": len(self.wire)},
            {
                "text": text,
                "tool_calls": tool_calls,
                "stop_reason": "tool_use" if tool_calls else "end_turn",
                "usage": {
                    "input_tokens": usage.get("promptTokenCount", 0),
                    "output_tokens": usage.get("candidatesTokenCount", 0),
                },
            },
            meta=self._pending_fp,
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
            **({"workspace": self.workspace} if self.workspace else {}),
        }

    def save(self, path: str) -> None:
        from .trace import trace_checksum

        data = self.to_dict()
        data["checksum"] = trace_checksum(data)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


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
        if not isinstance(event, dict):
            continue  # "data: null" / a bare scalar is not an event: skip
        etype = event.get("type")
        if etype == "message_start":
            message = event.get("message", {}) or {}
            envelope = {k: v for k, v in message.items() if k not in ("content", "usage")}
            usage.update(message.get("usage", {}) or {})
        elif etype == "content_block_start" and "index" in event:
            blocks[event["index"]] = dict(event.get("content_block", {}))
            partial[event["index"]] = ""
        elif etype == "content_block_delta":
            idx = event.get("index")
            if idx not in blocks:
                continue  # a delta for a block we never saw start: skip, don't crash
            delta = event.get("delta", {})
            if delta.get("type") == "text_delta":
                blocks[idx]["text"] = blocks[idx].get("text", "") + delta.get("text", "")
            elif delta.get("type") == "input_json_delta":
                partial[idx] += delta.get("partial_json", "")
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
        if not isinstance(event, dict):
            continue  # "data: null" / a bare scalar is not an event: skip
        model = event.get("model") or model
        if event.get("usage"):
            usage = event["usage"]
        for choice in event.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            finish_reason = choice.get("finish_reason") or finish_reason
            delta = choice.get("delta", {}) or {}
            text += delta.get("content") or ""
            for tc in delta.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
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


def reconstruct_gemini_sse(raw: str) -> dict:
    """Rebuild the final GenerateContentResponse from a Gemini SSE transcript."""
    text = ""
    fcalls: list = []
    finish = None
    usage: dict = {}
    model = ""
    for line in raw.splitlines():
        if not line.startswith("data:"):
            continue
        try:
            event = json.loads(line[5:].strip())
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        model = event.get("modelVersion") or model
        if event.get("usageMetadata"):
            usage = event["usageMetadata"]
        for cand in event.get("candidates") or []:
            if not isinstance(cand, dict):
                continue
            finish = cand.get("finishReason") or finish
            for p in ((cand.get("content") or {}).get("parts")) or []:
                if isinstance(p, dict) and "text" in p:
                    text += p.get("text", "")
                elif isinstance(p, dict) and "functionCall" in p:
                    fcalls.append(p)
    parts: list = ([{"text": text}] if text else []) + fcalls
    out = {"candidates": [{"content": {"role": "model", "parts": parts},
                           "finishReason": finish or "STOP", "index": 0}],
           "usageMetadata": usage}
    if model:
        out["modelVersion"] = model
    return out


def synthesize_gemini_sse(response: dict) -> bytes:
    """Render a recorded GenerateContentResponse as a Gemini SSE stream, for replay."""
    cand = (response.get("candidates") or [{}])[0] or {}
    parts = ((cand.get("content") or {}).get("parts")) or []
    out = []
    for p in parts:
        body = {"candidates": [{"content": {"role": "model", "parts": [p]}, "index": 0}]}
        out.append(f"data: {json.dumps(body)}\n\n".encode())
    final: dict = {"candidates": [{"content": {"role": "model", "parts": []},
                                   "finishReason": cand.get("finishReason", "STOP"), "index": 0}]}
    if response.get("usageMetadata"):
        final["usageMetadata"] = response["usageMetadata"]
    out.append(f"data: {json.dumps(final)}\n\n".encode())
    return b"".join(out)


def _reconstruct_stream(path: str, raw: str) -> dict:
    """Pick the SSE reconstructor for the endpoint's dialect."""
    p = path.lower()
    if "chat/completions" in p:
        return reconstruct_openai_sse(raw)
    if "generatecontent" in p:  # Gemini :generateContent / :streamGenerateContent
        return reconstruct_gemini_sse(raw)
    return reconstruct_sse(raw)


def _synth_for(response: dict):
    """Pick the SSE synthesizer matching a recorded response's dialect."""
    if "choices" in response:
        return synthesize_openai_sse
    if "candidates" in response:
        return synthesize_gemini_sse
    return synthesize_sse


class ProxyServer(ThreadingHTTPServer):
    """The recording (or replaying) proxy. Bind port 0 to pick a free port."""

    daemon_threads = True

    def __init__(self, port: int = 8788, target: str = DEFAULT_TARGET,
                 save_path: "str | None" = None, replay_path: "str | None" = None,
                 shield=None, scrub: bool = False,
                 save_interval: float = 5.0, eager_saves: int = 20,
                 max_body: int = 64 * 1024 * 1024, upstream_timeout: float = 600.0,
                 auth: str = "", host: str = "127.0.0.1"):
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
        # Loopback by default. Binding wider (0.0.0.0 for the docker-sandbox
        # topology) exposes the data plane to the network -- pair it with
        # --auth unless the network itself is closed (an internal bridge).
        super().__init__((host, port), _Handler)
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
        try:
            length = int(self.headers.get("content-length", 0))
        except (TypeError, ValueError):
            self._send_json(400, {"error": "invalid content-length header"})
            return
        if length < 0:
            self._send_json(400, {"error": "negative content-length"})
            return
        if self.server.max_body and length > self.server.max_body:
            self._send_json(413, {
                "error": f"request body {length} bytes exceeds the proxy cap "
                         f"({self.server.max_body}); raise it with --max-body-mb"
            })
            return
        try:
            request = json.loads(self.rfile.read(length) or b"{}")
        except (json.JSONDecodeError, ValueError):
            self._send_json(400, {"error": "request body is not valid JSON"})
            return
        if not isinstance(request, dict):
            self._send_json(400, {"error": "request body must be a JSON object"})
            return
        wants_stream = bool(request.get("stream")) or "streamgeneratecontent" in self.path.lower()

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
            elif shield.decide_pending(str(request.get("id", "")),
                                       request.get("decision") == "approve",
                                       who=str(request.get("by", ""))):
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
                self._send_sse(_synth_for(response)(response))
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
        except OSError as e:
            # urlopen can fail with the whole OSError family, not just URLError:
            # connection refused, DNS failure, TimeoutError, and -- seen against
            # real upstreams -- http.client.RemoteDisconnected / ConnectionReset
            # when the server drops the socket without a response. All become a
            # clean 502 rather than crashing the handler and resetting the client.
            reason = getattr(e, "reason", e)
            self._send_json(502, {"error": f"cannot reach upstream {self.server.target}: {reason}"})
            return

        shield = self.server.shield
        relayed = False  # once we start streaming to the client we can't 502
        try:
            with upstream:
                streamed = upstream.headers.get("content-type", "").startswith("text/event-stream")
                if streamed and shield is None:
                    # Relay the stream as it arrives; reconstruct the message after.
                    self.send_response(200)
                    self.send_header("content-type", "text/event-stream")
                    self.send_header("cache-control", "no-cache")
                    self.end_headers()
                    relayed = True
                    chunks: list[bytes] = []
                    while True:
                        chunk = upstream.read(1024)
                        if not chunk:
                            break
                        chunks.append(chunk)
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    raw = b"".join(chunks).decode("utf-8", "replace")
                    response = _reconstruct_stream(self.path, raw)
                elif streamed:
                    # A shield must see the whole response BEFORE the client does,
                    # so buffer the upstream stream and synthesize one afterwards.
                    raw = upstream.read().decode("utf-8", "replace")
                    response = _reconstruct_stream(self.path, raw)
                else:
                    raw_body = upstream.read()
                    try:
                        response = json.loads(raw_body or b"{}")
                    except (json.JSONDecodeError, ValueError):
                        # Upstream returned 200 with a non-JSON body (edge/CDN
                        # error page, wrong endpoint). Surface it as a gateway
                        # error rather than crashing the handler.
                        self._send_json(502, {
                            "error": "upstream returned a non-JSON response",
                            "body": raw_body.decode("utf-8", "replace")[:2000],
                        })
                        return
        except OSError as e:
            # The socket dropped mid-response. If we hadn't started relaying yet
            # the client still gets a clean 502; a torn stream we can only stop.
            if not relayed:
                self._send_json(502, {"error": f"upstream connection dropped: {e}"})
            return

        events: list = list(taint_events)
        if shield is not None:
            # May block awaiting a human decision on a confirm rule; the
            # client sees (and the trace records) only the screened response.
            response, screen_events = shield.screen(response)
            events.extend(screen_events)

        # Gemini names the model in the URL, not the body -- fold it into the
        # recorded request (after forwarding) so the trace keeps the model name.
        if not request.get("model"):
            m = _model_from_path(self.path)
            if m:
                request["model"] = m
        # Persist BEFORE answering: when the client sees the reply, the
        # exchange is already on disk (wirelog append + throttled trace write).
        self.server.persist(request, response, events)
        if streamed and shield is not None:
            self._send_sse(_synth_for(response)(response))
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
        if self.path.split("?", 1)[0] == "/loom/live":
            # The page embeds the control token (its JS needs it to poll state
            # and decide approvals), so serving it unauthenticated would hand
            # the token to any local process -- including the shielded agent
            # itself. Browsers can't send headers on navigation, so the gate
            # is a query token; the CLI prints/opens the tokenized URL.
            token = self.server.control_token
            if token:
                from urllib.parse import parse_qs, urlparse

                supplied = parse_qs(urlparse(self.path).query).get("token", [""])[0]
                if supplied != token:
                    self._send_json(403, {"error": "missing or wrong ?token= (use the "
                                                   "live URL loom printed at startup)"})
                    return
            from .live import live_html

            page = live_html(self.server.port, token or "").encode()
            self.send_response(200)
            self.send_header("content-type", "text/html; charset=utf-8")
            self.send_header("content-length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)
            return
        if self.path == "/loom/live/state":
            # Gated by the control token when there's a shield (so its
            # approve/deny controls aren't exposed unauthenticated).
            if self.server.control_token and not self._control_authorized():
                self._send_json(403, {"error": "missing or wrong x-loom-token header"})
                return
            from .live import live_state

            self._send_json(200, live_state(self.server))
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
        except OSError as e:  # URLError/TimeoutError/RemoteDisconnected/ConnectionReset...
            reason = getattr(e, "reason", e)
            self._send_json(502, {"error": f"cannot reach upstream {self.server.target}: {reason}"})
