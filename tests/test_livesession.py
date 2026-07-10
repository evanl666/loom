"""LiveSession: run an agent live, stream steps, ask follow-ups."""

import json
import threading
import time
import urllib.request

import pytest

from loom import Agent
from loom.debugger import DebugServer, steps_for
from loom.livesession import LiveSession
from loom.providers import ModelResponse, ScriptedProvider


def _server(live):
    srv = DebugServer(port=0, live=live)
    port = srv.httpd.server_address[1]
    threading.Thread(target=srv.httpd.serve_forever, daemon=True).start()
    time.sleep(0.15)
    return srv, f"http://127.0.0.1:{port}"


def _ask(base, prompt):
    req = urllib.request.Request(
        base + "/api/ask", data=json.dumps({"prompt": prompt}).encode(),
        headers={"content-type": "application/json"}, method="POST")
    urllib.request.urlopen(req)
    for _ in range(100):
        snap = json.load(urllib.request.urlopen(base + "/api/live"))
        if not snap["running"]:
            return snap
        time.sleep(0.05)
    raise AssertionError("live turn never finished")


def test_live_session_streams_a_turn_and_continues():
    agent = Agent(model=ScriptedProvider([
        ModelResponse(text="Paris.", stop_reason="end_turn"),
        ModelResponse(text="2.1 million.", stop_reason="end_turn")]), tools=[], name="a")
    srv, base = _server(LiveSession(agent=agent))
    try:
        run = json.load(urllib.request.urlopen(base + "/api/run"))
        assert run["live"] is True and run["steps"] == []
        s1 = _ask(base, "Capital of France?")
        assert s1["turns"] == 1 and s1["output"] == "Paris." and s1["steps"]
        s2 = _ask(base, "Its population?")
        assert s2["turns"] == 2 and "2.1" in s2["output"]
        assert len(s2["steps"]) > len(s1["steps"])
        assert len([s for s in s2["steps"] if s["type"] == "user"]) == 2
    finally:
        srv.httpd.shutdown()


def test_live_session_requires_exactly_one_source():
    with pytest.raises(ValueError):
        LiveSession()
    with pytest.raises(ValueError):
        LiveSession(agent=object(), func=lambda p: p)


def test_steps_for_interleaves_all_user_turns():
    agent = Agent(model=ScriptedProvider([
        ModelResponse(text="one.", stop_reason="end_turn"),
        ModelResponse(text="two.", stop_reason="end_turn")]), tools=[], name="a")
    steps = steps_for(agent.run(["first?", "second?"]).to_dict())
    assert [s["intent"] for s in steps if s["type"] == "user"] == ["first?", "second?"]
