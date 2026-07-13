"""Property-based invariants over MANY randomly-generated traces (seeded, so it's
deterministic). Universal properties that must hold for ANY valid trace -- the
'cover all cases' backstop for hand-written examples."""
import hashlib
import random

import pytest

from loom.action import actions
from loom.cost import analyze_cost
from loom.debugger import context_at, static_page, steps_for
from loom.diff import score_breakdown
from loom.multiagent import infer_agents
from loom.taint import taint_paths

VALID_TYPES = {"user", "reason", "answer", "call", "meta", "ask-human"}


def _h(s):
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def _random_trace(rng):
    n_agents = rng.randint(1, 5)
    syss = [f"You are Agent{i}. Role {rng.choice(['research', 'support', 'ops', 'math'])}." for i in range(n_agents)]
    tools = rng.sample(["read_customer", "send_email", "web_search", "calculate", "poll", "run_sql", "write_file"],
                       rng.randint(1, 5))
    log, seq, tuid = [], 0, 0
    for _ in range(rng.randint(1, 20)):
        sys = rng.choice(syss)
        meta = {"sys_hash": _h(sys), "sys_head": sys, "tools": tools, "msgs": rng.randint(1, 10)}
        if rng.random() < 0.7:
            tuid += 1
            tool = rng.choice(tools)
            log.append({"seq": seq, "kind": "model", "depth": 0, "meta": meta,
                        "result": {"tool_calls": [{"id": f"t{tuid}", "name": tool, "input": {"x": rng.randint(0, 999)}}],
                                   "stop_reason": "tool_use"}}); seq += 1
            res = rng.choice(["ok", f"SSN={rng.randint(100, 999)}-45-6789", "ERROR: boom", "",
                              f"result {rng.randint(0, 99)}", "日本語 🔥\x00 <script>"])
            log.append({"seq": seq, "kind": f"tool:{tool}", "depth": 0, "result": res,
                        "meta": {"tuid": f"t{tuid}"}}); seq += 1
        else:
            log.append({"seq": seq, "kind": "model", "depth": 0, "meta": meta,
                        "result": {"text": rng.choice(["done", "", "partial 42"]), "stop_reason": "end_turn"}}); seq += 1
    return {"recorded_via": "proxy", "episodes": ["go"], "output": "x",
            "systems": {_h(s): s for s in syss}, "tools": {}, "model": "m", "log": log}


@pytest.mark.parametrize("seed", range(6))
def test_universal_invariants_on_random_traces(seed):
    """~500 random traces per run: nothing crashes, and the structural invariants
    hold for every one."""
    rng = random.Random(seed)
    for _ in range(80):
        t = _random_trace(rng)
        steps = steps_for(t)
        ia = infer_agents(t)
        actions(t)
        analyze_cost(t)
        taint_paths(t)
        # at most one root
        assert sum(1 for a in ia["agents"] if a["is_root"]) <= 1
        # every edge references a real agent id
        ids = {a["id"] for a in ia["agents"]}
        assert all(e["from"] in ids and e["to"] in ids for e in ia["edges"])
        # every step has a valid type
        assert all(s.get("type") in VALID_TYPES for s in steps)
        # score is in range
        assert 0 <= score_breakdown(t)["overall"] <= 100


@pytest.mark.parametrize("seed", range(4))
def test_context_frame_is_always_a_growing_prefix(seed):
    """context_at(a) must be a prefix of context_at(b) for a < b -- the debugger's
    'stack & variables' only ever grow as you step forward, never reshuffle."""
    rng = random.Random(100 + seed)
    for _ in range(50):
        t = _random_trace(rng)
        n = len(t["log"])
        frames = [[m["role"] for m in context_at(t, s)] for s in range(0, n + 1, max(1, n // 5))]
        for a, b in zip(frames, frames[1:]):
            assert a == b[:len(a)], f"context frame not a growing prefix: {a} vs {b}"


def test_static_page_never_crashes_on_random_traces():
    rng = random.Random(7)
    for _ in range(40):
        html = static_page(_random_trace(rng))
        assert "LOOM_STATIC" in html
