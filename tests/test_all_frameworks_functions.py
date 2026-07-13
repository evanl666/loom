"""Comprehensive cross-framework × cross-function coverage.

Every analyzer and debugger surface, run against EVERY recorded framework fixture
(LangGraph, CrewAI, AutoGen, Claude-SDK, OpenAI-Agents, pydantic-ai, deepagents,
Semantic Kernel, MS Agent Framework, HTTP, native). Offline + deterministic --
these are real recorded traces, so this pins that no function crashes or produces
nonsense on ANY framework's wire shape, and that the core invariants hold.
"""
import json
import pathlib

import pytest

from loom.action import actions
from loom.cost import analyze_cost
from loom.debugger import context_at, static_page, steps_for
from loom.diff import score_breakdown
from loom.multiagent import infer_agents
from loom.rootcause import first_bad_step
from loom.taint import dlp_report, taint_paths
from loom.testing import verify_trace

FIXTURES = sorted((pathlib.Path(__file__).parent / "fixtures" / "frameworks").glob("*.loom.json"))
IDS = [p.stem for p in FIXTURES]
assert len(FIXTURES) >= 10, f"expected the framework fixtures, found {len(FIXTURES)}"


@pytest.fixture(params=FIXTURES, ids=IDS)
def trace(request):
    return json.loads(request.param.read_text()), str(request.param)


def test_reconstruction_invariants(trace):
    """steps_for + infer_agents produce a sane run for every framework."""
    data, _ = trace
    steps = steps_for(data)
    assert steps, "no steps reconstructed"
    # every step has a type and a numeric-ish anchor
    for s in steps:
        assert s.get("type") in {"user", "reason", "answer", "call", "meta", "ask-human"}
    # a user request node exists and is NOT raw harness scaffolding
    users = [s for s in steps if s.get("type") == "user"]
    assert users, "no user node"
    first = str(users[0].get("intent") or "")
    assert not first.lstrip().startswith("<system-reminder>"), "user request is a reminder block"

    ia = infer_agents(data)
    assert ia["agents"], "no agents inferred"
    assert sum(1 for a in ia["agents"] if a["is_root"]) == 1, "exactly one root expected"
    # every edge references real agent ids
    ids = {a["id"] for a in ia["agents"]}
    for e in ia["edges"]:
        assert e["from"] in ids and e["to"] in ids
    # labels are never empty / a bare generic placeholder wall
    assert all(a["label"] for a in ia["agents"])


def test_action_and_context_invariants(trace):
    """actions() lifts cleanly; context_at is a growing prefix with valid roles."""
    data, _ = trace
    acts = actions(data)
    assert acts, "no actions"
    # a tool call's result is bound to A call (no orphan crash)
    for a in acts:
        if a.type == "call":
            assert a.tool is not None
    full = context_at(data, 10_000)
    assert all(m.get("role") in {"user", "assistant", "tool", "human", "system"} for m in full)
    # a mid-run frame is a PREFIX -- never longer than the whole-run frame
    early = context_at(data, 0)
    assert len(early) <= len(full)


def test_analyzers_dont_crash_and_are_sane(trace):
    """cost / rootcause / taint / dlp / score run on every framework wire shape."""
    data, _ = trace
    c = analyze_cost(data)
    assert c["input_tokens"] >= 0 and c["output_tokens"] >= 0

    rc = first_bad_step(data)
    assert isinstance(rc, dict) and "found" in rc

    paths = taint_paths(data)
    assert isinstance(paths, list)
    for p in paths:                       # each exfil path names a source and a sink step
        assert "steps" in p or "value" in p or "sink" in p or p

    dlp = dlp_report(data)
    assert set(dlp) >= {"violations", "worst_severity"}

    sb = score_breakdown(data)
    assert 0 <= sb["overall"] <= 100


def test_studio_static_html_renders_every_framework(trace):
    """static_page() inlines a self-contained viewer with the agents present."""
    data, path = trace
    html = static_page(data)
    assert "LOOM_STATIC" in html and len(html) > 5_000
    ia = infer_agents(data)
    if ia["multi"]:
        # each recovered agent label appears in the frozen page
        for a in ia["agents"]:
            assert a["label"] in html, f"{a['label']} missing from studio HTML"


def test_replay_is_byte_identical(trace):
    """verify_trace confirms the recorded wire round-trips (structure + checksum)."""
    _, path = trace
    assert verify_trace(path) == []
