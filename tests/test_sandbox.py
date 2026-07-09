"""loom record --sandbox: the proxy is the agent's only network door."""

import json
import shutil
import sys
import threading

import pytest

from loom.cli import main
from loom.sandbox import sandbox_profile
from tests.test_proxy import FINAL_ANSWER, _FakeUpstream

darwin_only = pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("sandbox-exec") is None,
    reason="--sandbox is sandbox-exec (macOS) only",
)

# A child that first tries to BYPASS the proxy (direct call to another local
# port standing in for 'the open internet'), then talks through the proxy.
CHILD = """
import json, os, sys, urllib.request

def hit(url, payload=None):
    data = json.dumps(payload).encode() if payload else None
    req = urllib.request.Request(url, data=data,
        headers={"content-type": "application/json"}, method="POST" if data else "GET")
    return urllib.request.urlopen(req, timeout=3).read()

# 1. the bypass attempt must fail inside the sandbox
try:
    hit("http://127.0.0.1:{forbidden}/")
    print("BYPASS-SUCCEEDED")
    sys.exit(3)
except OSError:
    print("bypass blocked")

# 2. the sanctioned door still works
body = hit(os.environ["ANTHROPIC_BASE_URL"] + "/v1/messages",
           {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
print("proxy ok:", json.loads(body)["content"][0]["text"])
"""


def test_profile_contains_only_the_sanctioned_holes():
    profile = sandbox_profile([8788], allow=["localhost:8080"])
    assert '(deny network*)' in profile
    assert '(allow network* (remote tcp "localhost:8788"))' in profile
    assert '(allow network* (remote tcp "localhost:8080"))' in profile


def test_non_darwin_gets_a_recipe_pointer(monkeypatch):
    from loom.sandbox import wrap_sandboxed

    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(RuntimeError, match="docker recipe"):
        wrap_sandboxed(["echo"], ports=[1])


@darwin_only
def test_sandboxed_record_blocks_bypass_but_proxy_works(tmp_path, capsys):
    upstream = _serve(_FakeUpstream([FINAL_ANSWER]))
    forbidden = _serve(_FakeUpstream([FINAL_ANSWER]))  # stands in for the internet
    child = tmp_path / "child.py"
    child.write_text(CHILD.replace("{forbidden}", str(forbidden.server_address[1])))
    save = str(tmp_path / "session.loom.json")

    code = main([
        "record", "--sandbox", "--save", save,
        "--target", f"http://127.0.0.1:{upstream.server_address[1]}",
        "--", sys.executable, str(child),
    ])
    upstream.shutdown()
    forbidden.shutdown()

    assert code == 0  # child exited cleanly: bypass blocked, proxy reachable
    with open(save) as f:
        data = json.load(f)
    assert data["output"] == "It is raining in Berlin."  # exchange was recorded


def _serve(server):
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
