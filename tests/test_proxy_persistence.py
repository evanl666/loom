"""Proxy durability: wirelog crash safety, finalize, control-token auth, scrub."""

import json
import os
import threading
import urllib.error
import urllib.request

from loom.proxy import ProxyServer, compact_wirelog, control_token_for
from loom.shield import Shield
from tests.test_proxy import FINAL_ANSWER, _FakeUpstream

SECRET = "sk-ant-api03-" + "a1B2" * 8
LEAKY_ANSWER = {
    "content": [{"type": "text", "text": f"your key is {SECRET}"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 5, "output_tokens": 5},
}


def _serve(server):
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _post(port, payload=None, headers=None):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/messages",
        data=json.dumps(payload or {"model": "m", "messages": [{"role": "user", "content": "hi"}]}).encode(),
        headers={"content-type": "application/json", "x-api-key": "sk-test", **(headers or {})},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def _proxy(tmp_path, upstream, **kwargs):
    path = str(tmp_path / "session.loom.json")
    server = _serve(
        ProxyServer(port=0, target=f"http://127.0.0.1:{upstream.server_address[1]}",
                    save_path=path, **kwargs)
    )
    return server, path


# ------------------------------------------------------------------- wirelog


def test_wirelog_is_appended_during_and_removed_by_finalize(tmp_path):
    upstream = _serve(_FakeUpstream([FINAL_ANSWER]))
    proxy, path = _proxy(tmp_path, upstream)
    _post(proxy.port)
    assert os.path.exists(path + ".wirelog")  # durable before/with the response
    proxy.shutdown()
    upstream.shutdown()
    proxy.finalize()

    assert not os.path.exists(path + ".wirelog")  # clean shutdown needs no recovery
    with open(path) as f:
        assert f.read() and json.loads(open(path).read())["wire"][0] == FINAL_ANSWER
    proxy.finalize()  # idempotent


def test_compact_wirelog_recovers_a_crashed_session(tmp_path):
    wirelog = tmp_path / "session.loom.json.wirelog"
    exchange = {
        "request": {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        "response": FINAL_ANSWER,
        "shield_events": [{"action": "deny", "tool": "Read"}],
    }
    with open(wirelog, "w") as f:
        f.write(json.dumps(exchange) + "\n")
        f.write(json.dumps(exchange) + "\n")
        f.write('{"request": {"model":')  # torn tail: crashed mid-write

    out = str(tmp_path / "recovered.loom.json")
    rec = compact_wirelog(str(wirelog), out)
    assert len(rec.wire) == 2  # torn line ignored
    data = json.load(open(out))
    assert data["wire"] == [FINAL_ANSWER, FINAL_ANSWER]
    assert data["shield_events"] == [exchange["shield_events"][0]] * 2
    assert data["output"] == "It is raining in Berlin."


def test_cli_recovers_a_leftover_wirelog_on_next_record(tmp_path, capsys):
    from loom.cli import _recover_wirelog

    save = str(tmp_path / "session.loom.json")
    with open(save + ".wirelog", "w") as f:
        f.write(json.dumps({"request": {"messages": []}, "response": FINAL_ANSWER}) + "\n")
    _recover_wirelog(save)
    recovered = str(tmp_path / "session.recovered.loom.json")
    assert os.path.exists(recovered)
    assert not os.path.exists(save + ".wirelog")
    assert "recovered" in capsys.readouterr().err


# ------------------------------------------------------------- control token


def test_shielded_proxy_registers_a_control_token(tmp_path):
    upstream = _serve(_FakeUpstream([FINAL_ANSWER]))
    proxy, _ = _proxy(tmp_path, upstream, shield=Shield(deny=["Never*"]))
    try:
        assert proxy.control_token
        assert control_token_for(proxy.port) == proxy.control_token

        req = urllib.request.Request(f"http://127.0.0.1:{proxy.port}/loom/shield/pending")
        try:
            urllib.request.urlopen(req, timeout=5)
            assert False, "expected 403 without the token"
        except urllib.error.HTTPError as e:
            assert e.code == 403

        req.add_header("x-loom-token", proxy.control_token)
        with urllib.request.urlopen(req, timeout=5) as r:
            assert json.load(r) == {"pending": []}
    finally:
        proxy.shutdown()
        upstream.shutdown()
    proxy.finalize()
    assert control_token_for(proxy.port) is None  # token file cleaned up


def test_unshielded_proxy_has_no_token(tmp_path):
    upstream = _serve(_FakeUpstream([FINAL_ANSWER]))
    proxy, _ = _proxy(tmp_path, upstream)
    assert proxy.control_token is None
    proxy.shutdown()
    upstream.shutdown()


# --------------------------------------------------------------------- scrub


def test_scrub_redacts_the_trace_but_not_the_client_response(tmp_path):
    upstream = _serve(_FakeUpstream([LEAKY_ANSWER]))
    proxy, path = _proxy(tmp_path, upstream, scrub=True)
    got = _post(proxy.port)
    proxy.shutdown()
    upstream.shutdown()

    assert SECRET in got["content"][0]["text"]  # the agent still works
    assert SECRET not in open(path + ".wirelog").read()  # never on disk, even pre-finalize
    proxy.finalize()
    on_disk = open(path).read()
    assert SECRET not in on_disk and "[scrubbed:anthropic-key]" in on_disk
