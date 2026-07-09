"""The sharp record entrypoint: agent shortcuts, --profile, --report."""

import json
import os
import sys
import threading

from loom.cli import _expand_agent_shortcut, main
from tests.test_proxy import FINAL_ANSWER, WEATHER_TOOL_USE, _FakeUpstream

# A child that speaks the Anthropic API through whatever base URL it's given,
# echoing one tool call so the shield has something to act on.
CHILD = """
import json, os, urllib.request
url = os.environ["ANTHROPIC_BASE_URL"] + "/v1/messages"
def call():
    req = urllib.request.Request(url,
        data=json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}]}).encode(),
        headers={"content-type": "application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(req, timeout=10).read())
call(); call()
"""


def _ns(target="https://api.anthropic.com"):
    import argparse

    return argparse.Namespace(target=target)


def test_shortcut_expansion_rules():
    # missing prompt -> helpful error
    _, err = _expand_agent_shortcut(["claude"], _ns())
    assert "needs a prompt" in err
    # explicit flags -> passthrough untouched
    cmd, err = _expand_agent_shortcut(["claude", "-p", "hi"], _ns())
    assert cmd == ["claude", "-p", "hi"] and err == ""
    # an unknown first token -> passthrough
    cmd, err = _expand_agent_shortcut(["python", "a.py"], _ns())
    assert cmd == ["python", "a.py"] and err == ""


def _run_record(tmp_path, extra_args):
    upstream = _FakeUpstream([WEATHER_TOOL_USE, FINAL_ANSWER])
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    child = tmp_path / "child.py"
    child.write_text(CHILD)
    save = str(tmp_path / "session.loom.json")
    code = main([
        "record", "--save", save,
        "--target", f"http://127.0.0.1:{upstream.server_address[1]}",
        *extra_args,
        "--", sys.executable, str(child),
    ])
    upstream.shutdown()
    return code, save


def test_profile_applies_and_report_writes_both_files(tmp_path, capsys):
    # get_weather isn't allowed by ci-safe (deny-by-default) -> it's blocked
    code, save = _run_record(tmp_path, ["--profile", "ci-safe", "--report"])
    assert code == 0
    data = json.load(open(save))
    assert any(ev["action"] == "deny" for ev in data.get("shield_events", []))
    assert "workspace" in data  # recorded by default

    base = save[: -len(".loom.json")]
    assert os.path.exists(base + ".html")
    assert os.path.exists(base + ".incident.md")
    report = open(base + ".incident.md").read()
    assert "# Incident report" in report

    err = capsys.readouterr().err
    assert ".html" in err and ".incident.md" in err


def test_no_workspace_opts_out(tmp_path):
    _, save = _run_record(tmp_path, ["--no-workspace"])
    assert "workspace" not in json.load(open(save))
