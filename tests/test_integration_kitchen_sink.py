"""One maximally-complex trace, many concurrent concerns. A 5-agent run that at
once SUCCEEDS (research), EXFILTRATES (support reads PII then emails it), FAILS
(ops tool errors), and LOOPS (poller repeats) -- all interleaved on the wire.
Each analyzer must isolate ITS concern without being confused by the others."""
import hashlib

import pytest

from loom.action import actions
from loom.autopsy import autopsy_html
from loom.debugger import context_at, static_page, steps_for
from loom.diff import score_breakdown
from loom.incident import build_report
from loom.multiagent import infer_agents
from loom.rootcause import first_bad_step
from loom.taint import dlp_report, taint_paths

RES = "You are Research. web_search + calculate."
SUP = "You are Support. read the customer then email."
OPS = "You are Ops. run maintenance."
POLL = "You are Poller. poll job status."
COORD = "You are Coordinator. Delegate to everyone."
SSN = "123-45-6789"


def _h(s):
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def _m(seq, sys, tcs=None, text=""):
    return {"seq": seq, "kind": "model", "depth": 0,
            "meta": {"sys_hash": _h(sys), "sys_head": sys, "tools": ["x"]},
            "result": {"text": text, "tool_calls": tcs or [], "stop_reason": "tool_use" if tcs else "end_turn"}}


def _t(seq, name, res, tuid):
    return {"seq": seq, "kind": f"tool:{name}", "depth": 0, "result": res, "meta": {"tuid": tuid}}


@pytest.fixture(scope="module")
def kitchen_sink():
    log = [
        _m(0, COORD, [{"id": "c1", "name": "ask_research", "input": {}}]), _t(1, "ask_research", "ok", "c1"),
        _m(2, RES, [{"id": "r1", "name": "web_search", "input": {"q": "eiffel"}}]), _t(3, "web_search", "330m", "r1"),
        _m(4, SUP, [{"id": "s1", "name": "read_customer", "input": {}}]), _t(5, "read_customer", f"SSN={SSN}", "s1"),
        _m(6, SUP, [{"id": "s2", "name": "send_email", "input": {"body": f"ssn {SSN}"}}]), _t(7, "send_email", "sent", "s2"),
        _m(8, OPS, [{"id": "o1", "name": "run_maintenance", "input": {}}]),
        _t(9, "run_maintenance", "ERROR: disk full", "o1"),
    ]
    seq = 10
    for i in range(12):
        log.append(_m(seq, POLL, [{"id": f"p{i}", "name": "poll", "input": {"id": "job1"}}])); seq += 1
        log.append(_t(seq, "poll", "still running", f"p{i}")); seq += 1
    log.append(_m(seq, COORD, text="partial completion"))
    return {"recorded_via": "proxy", "episodes": ["do everything"], "output": "partial completion",
            "systems": {_h(x): x for x in [RES, SUP, OPS, POLL, COORD]},
            "tools": {"send_email": ["network", "user_communication"]}, "model": "m", "log": log}


def test_all_five_agents_recovered_with_correct_concerns(kitchen_sink):
    ia = infer_agents(kitchen_sink)
    labels = sorted(a["label"] for a in ia["agents"])
    assert labels == ["Coordinator", "Ops", "Poller", "Research", "Support"]


def test_taint_isolates_the_exfil_agent_only(kitchen_sink):
    """Only Support's read->email is an exfil path; the loop/failure aren't."""
    paths = taint_paths(kitchen_sink)
    assert len(paths) == 1
    assert paths[0]["source"]["tool"] == "read_customer" and paths[0]["sink"]["tool"] == "send_email"
    assert dlp_report(kitchen_sink)["worst_severity"] == "critical"


def test_rootcause_picks_the_failure_over_the_risk_and_loop(kitchen_sink):
    """A hard FAILURE (the ops tool error) outranks the exfil risk and the loop as
    the first bad step -- rootcause must not report the loop or the exfil instead."""
    rc = first_bad_step(kitchen_sink)
    assert rc["found"] and rc["kind"] == "failure"
    assert rc["tool"] == "run_maintenance" and rc["step"] == 9
    assert any("error" in s.lower() for s in rc["signals"])


def test_every_step_is_attributed_and_frames_are_prefixes(kitchen_sink):
    steps = steps_for(kitchen_sink)
    assert len(steps) >= 36
    # every model/call step has an agent (no orphan in a 5-agent interleave)
    for s in steps:
        if s.get("type") in ("reason", "answer", "call"):
            assert s.get("agent"), f"unattributed step {s.get('step')}"
    assert len(context_at(kitchen_sink, 0)) <= len(context_at(kitchen_sink, 10_000))


def test_all_reports_survive_the_kitchen_sink(kitchen_sink):
    actions(kitchen_sink)
    autopsy_html(kitchen_sink)
    build_report(kitchen_sink, "x.loom.json")
    assert 0 <= score_breakdown(kitchen_sink)["overall"] <= 100
    assert "LOOM_STATIC" in static_page(kitchen_sink)
