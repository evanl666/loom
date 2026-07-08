"""Loom Shield: the agent firewall at the proxy boundary."""

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from loom.proxy import ProxyServer, reconstruct_sse, synthesize_sse
from loom.shield import ALLOW, CONFIRM, DENY, Shield

READ_ENV = {
    "content": [
        {"type": "text", "text": "Let me peek at the secrets."},
        {"type": "tool_use", "id": "tu_1", "name": "Read", "input": {"file_path": "/app/.env"}},
    ],
    "stop_reason": "tool_use",
    "usage": {"input_tokens": 10, "output_tokens": 5},
}
LIST_FILES = {
    "content": [
        {"type": "tool_use", "id": "tu_2", "name": "Bash", "input": {"command": "ls -la"}},
    ],
    "stop_reason": "tool_use",
    "usage": {"input_tokens": 10, "output_tokens": 5},
}
DONE = {
    "content": [{"type": "text", "text": "All done."}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 20, "output_tokens": 4},
}


# ---------------------------------------------------------------- rule engine


def test_classify_matches_name_and_signature():
    s = Shield(deny=["Read(*.env*)", "Bash(*rm -rf*)"], confirm=["WebFetch"])
    assert s.classify("Read", {"file_path": "/app/.env"}) == (DENY, "Read(*.env*)")
    assert s.classify("Read", {"file_path": "/app/main.py"}) == (ALLOW, "")
    assert s.classify("Bash", {"command": "rm -rf /tmp/x"}) == (DENY, "Bash(*rm -rf*)")
    assert s.classify("WebFetch", {"url": "http://x"}) == (CONFIRM, "WebFetch")


def test_precedence_deny_beats_allow_beats_confirm():
    s = Shield(deny=["Bash(*rm*)"], allow=["Read*"], confirm=["*"])
    assert s.classify("Bash", {"command": "rm x"})[0] == DENY
    assert s.classify("Read", {"file_path": "a.py"})[0] == ALLOW  # allow bypasses confirm
    assert s.classify("Write", {"file_path": "a.py"})[0] == CONFIRM


# ------------------------------------------------------------ deny rewriting


def test_deny_rewrites_anthropic_response():
    s = Shield(deny=["Read(*.env*)"])
    out, events = s.screen(READ_ENV)
    kinds = [b["type"] for b in out["content"]]
    assert "tool_use" not in kinds
    assert out["stop_reason"] == "end_turn"
    assert any("Blocked tool call Read" in b.get("text", "") for b in out["content"])
    assert events == [
        {**events[0], "action": "deny", "via": "rule", "tool": "Read", "rule": "Read(*.env*)"}
    ]
    assert READ_ENV["stop_reason"] == "tool_use"  # original untouched


def test_deny_keeps_allowed_siblings_and_tool_use_stop_reason():
    both = {
        "content": [READ_ENV["content"][1], LIST_FILES["content"][0]],
        "stop_reason": "tool_use",
        "usage": {},
    }
    out, events = Shield(deny=["Read(*.env*)"]).screen(both)
    names = [b.get("name") for b in out["content"] if b["type"] == "tool_use"]
    assert names == ["Bash"]  # the allowed call survives
    assert out["stop_reason"] == "tool_use"
    assert len(events) == 1


def test_deny_rewrites_openai_response():
    response = {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "c1", "type": "function",
                         "function": {"name": "read_file", "arguments": '{"path": "/app/.env"}'}}
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
    }
    out, events = Shield(deny=["read_file(*.env*)"]).screen(response)
    message = out["choices"][0]["message"]
    assert "tool_calls" not in message
    assert "Blocked tool call read_file" in message["content"]
    assert out["choices"][0]["finish_reason"] == "stop"
    assert events[0]["action"] == "deny"


def test_clean_response_passes_through_unchanged():
    s = Shield(deny=["Read(*.env*)"])
    out, events = s.screen(DONE)
    assert out is DONE and events == []


# ------------------------------------------------------- confirm / approvals


def test_confirm_approved_lets_the_call_through():
    s = Shield(confirm=["Bash*"], timeout=5)

    def approve_when_pending():
        while not s.pending_list():
            time.sleep(0.01)
        assert s.decide_pending(s.pending_list()[0]["id"], approve=True)

    t = threading.Thread(target=approve_when_pending)
    t.start()
    out, events = s.screen(LIST_FILES)
    t.join()
    assert out == LIST_FILES  # original response, tool call intact
    assert events[0] == {**events[0], "action": "approve", "via": "operator"}
    assert s.pending_list() == []


def test_confirm_denied_blocks_the_call():
    s = Shield(confirm=["Bash*"], timeout=5)

    def deny_when_pending():
        while not s.pending_list():
            time.sleep(0.01)
        s.decide_pending(s.pending_list()[0]["id"], approve=False)

    t = threading.Thread(target=deny_when_pending)
    t.start()
    out, events = s.screen(LIST_FILES)
    t.join()
    assert not any(b["type"] == "tool_use" for b in out["content"])
    assert "denied by the operator" in out["content"][0]["text"]
    assert events[0]["action"] == "deny" and events[0]["via"] == "operator"


def test_confirm_timeout_denies():
    out, events = Shield(confirm=["Bash*"], timeout=0.05).screen(LIST_FILES)
    assert not any(b["type"] == "tool_use" for b in out["content"])
    assert "timed out" in out["content"][0]["text"]
    assert events[0] == {**events[0], "action": "deny", "via": "timeout"}


def test_decide_unknown_pending_is_false():
    assert Shield().decide_pending("nope", approve=True) is False


def test_webhook_receives_pending_approval():
    received = []

    class _Hook(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["content-length"]))))
            self.send_response(200)
            self.send_header("content-length", "0")
            self.end_headers()

    hook = ThreadingHTTPServer(("127.0.0.1", 0), _Hook)
    threading.Thread(target=hook.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{hook.server_address[1]}/inbox"
    Shield(confirm=["Bash*"], timeout=0.3, webhook=url).screen(LIST_FILES)
    hook.shutdown()
    assert received and received[0]["event"] == "loom.shield.confirm"
    assert received[0]["tool"] == "Bash"
    assert "approve?" in received[0]["text"]


# --------------------------------------------------------- through the proxy


class _FakeUpstream(ThreadingHTTPServer):
    def __init__(self, responses, sse=False):
        self.responses = list(responses)
        self.sse = sse
        super().__init__(("127.0.0.1", 0), _FakeHandler)

    def server_bind(self):
        import socketserver

        socketserver.TCPServer.server_bind(self)
        self.server_name = "127.0.0.1"
        self.server_port = self.server_address[1]


class _FakeHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length", 0)))
        response = self.server.responses.pop(0)
        body = synthesize_sse(response) if self.server.sse else json.dumps(response).encode()
        self.send_response(200)
        self.send_header(
            "content-type", "text/event-stream" if self.server.sse else "application/json"
        )
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _serve(server):
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _post(port, payload, path="/v1/messages", raw=False, headers=None):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json", "x-api-key": "sk-test", **(headers or {})},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        body = r.read()
        return body if raw else json.loads(body)


def _shielded_proxy(tmp_path, upstream, shield):
    path = str(tmp_path / "session.loom.json")
    proxy = _serve(
        ProxyServer(
            port=0,
            target=f"http://127.0.0.1:{upstream.server_address[1]}",
            save_path=path,
            shield=shield,
        )
    )
    return proxy, path


def test_proxy_blocks_env_read_and_records_screened_response(tmp_path):
    upstream = _serve(_FakeUpstream([READ_ENV]))
    proxy, path = _shielded_proxy(tmp_path, upstream, Shield(deny=["Read(*.env*)"]))
    got = _post(proxy.port, {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    proxy.shutdown()
    upstream.shutdown()

    assert not any(b["type"] == "tool_use" for b in got["content"])
    assert "Blocked tool call Read" in got["content"][-1]["text"]
    with open(path) as f:
        data = json.load(f)
    assert data["wire"][0] == got  # trace records what the client saw
    assert data["shield_events"][0]["action"] == "deny"
    assert data["shield_events"][0]["tool"] == "Read"
    assert data["log"][0]["result"]["tool_calls"] == []  # no phantom tool effect


def test_proxy_streaming_client_gets_screened_synthesized_sse(tmp_path):
    upstream = _serve(_FakeUpstream([READ_ENV], sse=True))
    proxy, path = _shielded_proxy(tmp_path, upstream, Shield(deny=["Read(*.env*)"]))
    raw = _post(
        proxy.port,
        {"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
        raw=True,
    ).decode()
    proxy.shutdown()
    upstream.shutdown()

    message = reconstruct_sse(raw)
    assert not any(b["type"] == "tool_use" for b in message["content"])
    assert "Blocked tool call Read" in message["content"][-1]["text"]
    assert message["stop_reason"] == "end_turn"


def test_proxy_control_endpoints_drive_a_confirm(tmp_path):
    upstream = _serve(_FakeUpstream([LIST_FILES]))
    proxy, path = _shielded_proxy(tmp_path, upstream, Shield(confirm=["Bash*"], timeout=10))

    result = {}

    def client():
        result["response"] = _post(
            proxy.port, {"model": "m", "messages": [{"role": "user", "content": "ls"}]}
        )

    t = threading.Thread(target=client)
    t.start()

    token = {"x-loom-token": proxy.control_token}
    pending = []
    deadline = time.time() + 5
    while not pending and time.time() < deadline:
        req = urllib.request.Request(
            f"http://127.0.0.1:{proxy.port}/loom/shield/pending", headers=token
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            pending = json.load(r)["pending"]
        time.sleep(0.02)
    assert pending and pending[0]["tool"] == "Bash"

    approved = _post(
        proxy.port, {"id": pending[0]["id"], "decision": "approve"},
        path="/loom/shield/decide", headers=token,
    )
    assert approved == {"ok": True}
    t.join(timeout=5)
    proxy.shutdown()
    upstream.shutdown()

    assert result["response"] == LIST_FILES  # approval let the original through
    with open(path) as f:
        data = json.load(f)
    assert data["shield_events"][0] == {
        **data["shield_events"][0], "action": "approve", "via": "operator", "tool": "Bash",
    }


# ------------------------------------------------- default action / matching


def test_default_deny_blocks_unmatched_calls():
    s = Shield(allow=["get_weather*"], default=DENY)
    assert s.classify("get_weather", {"city": "x"})[0] == ALLOW
    assert s.classify("Bash", {"command": "ls"}) == (DENY, "")
    out, events = s.screen(LIST_FILES)
    assert not any(b["type"] == "tool_use" for b in out["content"])
    assert events[0] == {**events[0], "action": "deny", "via": "default"}
    assert "denies by default" in out["content"][0]["text"]


def test_default_confirm_holds_unmatched_calls():
    out, events = Shield(default=CONFIRM, timeout=0.05).screen(LIST_FILES)
    assert events[0] == {**events[0], "action": "deny", "via": "timeout"}


def test_invalid_default_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        Shield(default="block")


def test_signature_matching_ignores_whitespace_runs():
    s = Shield(deny=["Bash(*rm -rf*)"])
    assert s.classify("Bash", {"command": "rm   -rf /"})[0] == DENY


# -------------------------------------------------------------- LLM as judge


class _FakeJudge:
    """A judge provider: returns scripted texts, counts how often it's asked."""

    def __init__(self, *texts):
        self.texts = list(texts)
        self.calls = 0

    def complete(self, system, messages, tools):
        self.calls += 1

        class R:
            text = self.texts.pop(0)

        return R()


def test_judge_escalates_risky_unmatched_calls_to_confirm():
    judge = _FakeJudge('{"risk": 0.9, "reason": "reads credentials"}')
    s = Shield(judge=judge, timeout=0.05)
    out, events = s.screen(READ_ENV)
    assert not any(b["type"] == "tool_use" for b in out["content"])
    assert events[0]["judge_risk"] == 0.9
    assert events[0]["judge_reason"] == "reads credentials"
    assert events[0] == {**events[0], "action": "deny", "via": "timeout"}


def test_judge_allows_low_risk_calls_with_an_audit_event():
    judge = _FakeJudge('{"risk": 0.1, "reason": "read-only listing"}')
    out, events = Shield(judge=judge).screen(LIST_FILES)
    assert out == LIST_FILES
    assert events[0] == {**events[0], "action": "allow", "via": "judge", "judge_risk": 0.1}


def test_judge_fails_open_on_junk():
    out, events = Shield(judge=_FakeJudge("I cannot answer that")).screen(LIST_FILES)
    assert out == LIST_FILES
    assert events[0]["via"] == "judge-error"


def test_explicit_rules_bypass_the_judge():
    judge = _FakeJudge()
    s = Shield(deny=["Read(*.env*)"], allow=["Bash*"], judge=judge)
    s.screen(READ_ENV)
    s.screen(LIST_FILES)
    assert judge.calls == 0


# --------------------------------------------------------------- trust ratchet


def _ledger(tmp_path):
    from loom.shield import TrustLedger

    return TrustLedger(str(tmp_path / "trust.json"))


def test_trust_ledger_streaks_persist_and_deny_demotes(tmp_path):
    from loom.shield import TrustLedger

    ledger = _ledger(tmp_path)
    ledger.record("Bash", True, {"id": "a1"})
    ledger.record("Bash", True, {"id": "b2"})
    assert ledger.streak("Bash") == 2
    reloaded = TrustLedger(ledger.path)  # survives a restart
    assert reloaded.streak("Bash") == 2
    assert [e["id"] for e in reloaded.data["Bash"]["evidence"]] == ["a1", "b2"]

    reloaded.record("Bash", False, {"id": "c3"})  # one deny resets the streak
    assert reloaded.streak("Bash") == 0

    assert reloaded.demote("Bash") is True
    assert reloaded.demote("NeverSeen") is False


def test_ratchet_auto_approves_after_enough_operator_approvals(tmp_path):
    ledger = _ledger(tmp_path)
    s = Shield(confirm=["Bash*"], trust=ledger, trust_after=2, timeout=5)

    for _ in range(2):  # two human approvals build the streak
        t = threading.Thread(target=_approve_first_pending, args=(s,))
        t.start()
        out, events = s.screen(LIST_FILES)
        t.join()
        assert events[0]["via"] == "operator"

    # third time: no human needed, the ratchet approves with evidence
    out, events = s.screen(LIST_FILES)
    assert out == LIST_FILES
    assert events[0] == {**events[0], "action": "approve", "via": "ratchet", "streak": 2}
    assert s.pending_list() == []


def _approve_first_pending(s):
    while not s.pending_list():
        time.sleep(0.01)
    s.decide_pending(s.pending_list()[0]["id"], approve=True)


def test_timeouts_do_not_move_the_ratchet(tmp_path):
    ledger = _ledger(tmp_path)
    Shield(confirm=["Bash*"], trust=ledger, trust_after=3, timeout=0.05).screen(LIST_FILES)
    assert ledger.streak("Bash") == 0


def test_operator_deny_demotes_the_tool(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.record("Bash", True, {"id": "x"})
    s = Shield(confirm=["Bash*"], trust=ledger, trust_after=5, timeout=5)

    def deny_first():
        while not s.pending_list():
            time.sleep(0.01)
        s.decide_pending(s.pending_list()[0]["id"], approve=False)

    t = threading.Thread(target=deny_first)
    t.start()
    s.screen(LIST_FILES)
    t.join()
    assert ledger.streak("Bash") == 0


def test_control_endpoints_404_without_shield(tmp_path):
    upstream = _serve(_FakeUpstream([DONE]))
    proxy = _serve(
        ProxyServer(port=0, target=f"http://127.0.0.1:{upstream.server_address[1]}")
    )
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{proxy.port}/loom/shield/pending", timeout=5)
        assert False, "expected 404"
    except urllib.error.HTTPError as e:
        assert e.code == 404
    proxy.shutdown()
    upstream.shutdown()
