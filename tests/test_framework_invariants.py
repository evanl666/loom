"""Every supported agent framework, through every read/analysis function.

Loom's whole bet is that ONE normalization (wire -> Action + agent tree) lets a
SINGLE set of debugger functions serve any framework. These fixtures are real
recorded traces from 11 frameworks (delegation, peer group-chat, hand-off,
black-box sub-agents, remote agents). The test runs each through the shared
pipeline and asserts the invariants the UI relies on -- so a change that works
for one framework but breaks a seam in another (the recurring failure mode) turns
this red BEFORE it ships, instead of surfacing later in someone's debugger.
"""
import glob
import json
import os

import pytest

from loom.action import actions
from loom.debugger import (context_at, copilot_report, static_data, static_page,
                           steps_for, _run_summary)
from loom.multiagent import infer_agents

FIXTURES = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "fixtures", "frameworks", "*.loom.json")))
IDS = [os.path.basename(f).replace(".loom.json", "") for f in FIXTURES]


def _load(path):
    with open(path) as f:
        return json.load(f)


def _model_calls(step, steps):
    """Mirror of the UI's modelCallNames: tools this model turn decided to call,
    matched by requester OR (hand-off traces w/o requested_at) same step."""
    s = step.get("step")
    if s is None:
        return []
    return [x["tool"] for x in steps if x.get("type") == "call" and x.get("tool")
            and (x.get("requested_at") == s or (x.get("requested_at") is None and x.get("step") == s))]


def test_fixtures_present():
    assert FIXTURES, "no framework fixtures found under tests/fixtures/frameworks/"


@pytest.mark.parametrize("path", FIXTURES, ids=IDS)
def test_framework_reconstructs_and_renders(path):
    data = _load(path)

    # 1) agent tree recovers (at least one agent, every step attributed if multi)
    ia = infer_agents(data)
    assert ia["agents"], "no agents recovered"

    # 2) the step list builds and every visible non-user row can render a label --
    #    a bare MODEL/TOOL badge with no content is the readability regression we
    #    fixed; a model turn is allowed to be text-less ONLY if it called a tool.
    steps = steps_for(data)
    assert steps, "no steps produced"
    for s in steps:
        if s.get("is_delegation") or s.get("type") == "user" or s.get("tool"):
            continue
        text = (s.get("intent") or (s.get("observation") or {}).get("text", "")).strip()
        assert text or _model_calls(s, steps), f"blank row at step {s.get('step')} ({s.get('type')})"

    # 3) the context frame reconstructs at start / middle / end, grows monotonically,
    #    and always opens with a user message (the prompt).
    mx = max((s.get("step", 0) for s in steps), default=0)
    c0, cmid, cend = context_at(data, 0), context_at(data, mx // 2), context_at(data, mx)
    assert cend and cend[0]["role"] == "user"
    assert len(c0) <= len(cmid) <= len(cend), "context frame is not a growing prefix"

    # 4) the frozen static studio inlines everything and never phones home.
    sd = static_data(data)
    assert sd["run"]["steps"] and sd["agents"]["agents"]
    html = static_page(data)
    assert "127.0.0.1" not in html and "localhost" not in html, "static page hits a server"

    # 5) the analysis functions run without throwing on any framework's shape.
    assert "grade" in copilot_report(data) or "summary" in copilot_report(data)
    _run_summary(data)
    list(actions(data))
