"""Scale: every function stays fast + correct on a LARGE deep multi-agent trace.

Production traces are big (hundreds of turns). This generates a 500-entry, 5-agent
run and asserts each analyzer/debugger surface completes well under a generous
budget -- a guard against an O(n^2) regression sneaking back in (Studio has had
two) -- and still reconstructs a sane run.
"""
import hashlib
import time

import pytest

from loom.action import actions
from loom.cost import analyze_cost
from loom.debugger import context_at, static_page, steps_for
from loom.diff import score_breakdown
from loom.multiagent import infer_agents
from loom.rootcause import first_bad_step
from loom.taint import dlp_report, taint_paths

AGENTS = ["Coordinator", "Research Lead", "Data Analyst", "Support Lead", "Auditor"]


def _big_trace(rounds=50):
    def sysh(s):
        return hashlib.sha1(s.encode()).hexdigest()[:12]

    log, seq, systems, tuid = [], 0, {}, 0

    def add(kind, **kw):
        nonlocal seq
        log.append({"seq": seq, "kind": kind, **kw})
        seq += 1

    for rnd in range(rounds):
        for ag in AGENTS:
            s = f"You are the {ag}. Do your job."
            systems[sysh(s)] = s
            tuid += 1
            tc = [{"id": f"t{tuid}", "name": "work",
                   "input": {"round": rnd, "agent": ag, "payload": "x" * 200}}]
            add("model", depth=0,
                meta={"sys_hash": sysh(s), "sys_head": s, "tools": ["work", "ask_next"], "msgs": rnd * 2 + 1},
                result={"text": f"{ag} thinking round {rnd}", "tool_calls": tc, "stop_reason": "tool_use"})
            add("tool:work", depth=0, result=f"result-{ag}-{rnd}-" + "y" * 300, meta={"tuid": f"t{tuid}"})
    add("model", depth=0, meta={"sys_hash": sysh(f"You are the {AGENTS[0]}. Do your job."), "tools": ["work"]},
        result={"text": "FINAL SUMMARY", "stop_reason": "end_turn"})
    return {"recorded_via": "proxy", "model": "m", "episodes": ["run the big job"],
            "output": "FINAL SUMMARY", "systems": systems, "log": log, "tools": {}}


BUDGET_S = 5.0   # generous: purely an O(n^2) tripwire, not a benchmark


@pytest.fixture(scope="module")
def big():
    return _big_trace(50)


def _timed(fn, data):
    t = time.perf_counter()
    out = fn(data)
    return out, time.perf_counter() - t


@pytest.mark.parametrize("fn", [
    steps_for, infer_agents, actions, analyze_cost, first_bad_step,
    taint_paths, dlp_report, score_breakdown, static_page,
    lambda d: context_at(d, 10_000),
])
def test_function_stays_under_budget_on_a_big_trace(big, fn):
    _out, secs = _timed(fn, big)
    assert secs < BUDGET_S, f"{getattr(fn, '__name__', fn)} took {secs:.2f}s on 500 entries (O(n^2)?)"


def test_big_trace_still_reconstructs_correctly(big):
    steps = steps_for(big)
    assert len(steps) >= 500                       # ~one per log entry + user node
    ia = infer_agents(big)
    assert ia["multi"] and len(ia["agents"]) >= 5
    assert sum(1 for a in ia["agents"] if a["is_root"]) == 1
    # the whole-run context frame is built without exploding, and is a prefix
    assert len(context_at(big, 0)) <= len(context_at(big, 10_000))
    # Studio inlines it ONCE (not O(n^2) per-step): a 500-turn run must stay well
    # under a few MB (the pre-fix cumulative-per-step inline was ~23MB at 300).
    assert len(static_page(big)) < 4_000_000
