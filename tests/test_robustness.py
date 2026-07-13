"""Analyzers tolerate hand-edited / third-party / corrupted traces."""

import pytest

from loom.action import actions, effect_dicts
from loom.autopsy import autopsy_html
from loom.cost import analyze_cost
from loom.diagnose import diagnose
from loom.diff import score_breakdown
from loom.export import trace_to_html
from loom.incident import build_report
from loom.insight import provenance, side_effect_map
from loom.movie import movie_html
from loom.packs import install_builtin
from loom.providers.base import ModelResponse
from loom.taint import dlp_report, taint_paths

install_builtin()

MALFORMED = [
    {"log": "not a list"},
    {"log": ["a string entry", 42, None]},
    {"log": [{"seq": 0, "kind": "model", "key": "k", "result": "not a dict"}]},
    {"log": [{"seq": 0, "kind": "model", "key": "k",
              "result": {"text": "", "tool_calls": None, "usage": None}}]},
    {"log": [], "shield_events": None},
    {"log": [{"seq": 0, "kind": "tool:x", "key": "k", "result": None}]},
    {},  # no keys at all
]

ANALYZERS = [actions, analyze_cost, score_breakdown, movie_html, taint_paths,
             dlp_report, diagnose, side_effect_map, provenance, trace_to_html,
             autopsy_html]


@pytest.mark.parametrize("data", MALFORMED)
def test_analyzers_never_crash_on_malformed_traces(data):
    for fn in ANALYZERS:
        fn(data)                      # must not raise
    build_report(data, "x.loom.json")  # nor the incident report


def test_model_response_from_dict_tolerates_null_and_non_dict():
    assert ModelResponse.from_dict({"tool_calls": None}).tool_calls == []
    assert ModelResponse.from_dict({"usage": None}).usage == {}
    assert ModelResponse.from_dict("not a dict").text == ""
    # a non-dict tool_call entry is skipped, not crashed on
    assert ModelResponse.from_dict({"tool_calls": ["junk", {"id": "t", "name": "X"}]}).tool_calls[0].name == "X"


def test_effect_dicts_normalizes_the_log():
    assert effect_dicts({"log": "oops"}) == []
    assert effect_dicts({"log": ["x", {"seq": 0, "kind": "model"}, 5]}) == [{"seq": 0, "kind": "model"}]
    assert effect_dicts({}) == []


def test_effectentry_from_dict_tolerates_missing_keys():
    """The trace/journal load chokepoint must not crash on a corrupted effect
    entry missing seq/kind/key/result -- it should degrade, so analyzers can
    still process the rest of the trace."""
    from loom.effect import EffectEntry

    for d in [{}, {"seq": 3}, {"kind": "model"}, {"result": {"text": "hi"}}]:
        e = EffectEntry.from_dict(d)  # must not raise
        assert isinstance(e.seq, int) and isinstance(e.kind, str)


# --- complex / adversarial shapes (not just missing keys) --------------------
def _m(seq, sys, tools, tcs=None, text=""):
    import hashlib
    return {"seq": seq, "kind": "model", "depth": 0,
            "meta": {"sys_hash": hashlib.sha1(sys.encode()).hexdigest()[:12],
                     "sys_head": sys, "tools": tools, "msgs": seq + 1},
            "result": {"text": text, "tool_calls": tcs or [],
                       "stop_reason": "tool_use" if tcs else "end_turn"}}


def _parallel_same_name_out_of_order(n=10):
    """n parallel Read calls whose results are recorded in REVERSE order -- the
    tuid-attribution path at scale."""
    calls = [{"id": f"t{i}", "name": "Read", "input": {"file": f"f{i}.py"}} for i in range(n)]
    log = [_m(0, "agent", ["Read"], tcs=calls)]
    for i in reversed(range(n)):   # results out of call order
        log.append({"seq": len(log), "kind": "tool:Read", "depth": 0,
                    "result": f"content-{i}", "meta": {"tuid": f"t{i}"}})
    log.append(_m(len(log), "agent", ["Read"], text="done"))
    return {"recorded_via": "proxy", "log": log, "episodes": ["do it"], "output": "done",
            "systems": {}, "model": "m"}


def _self_delegation():
    """An agent whose tool call names ITSELF (circular) must not infinite-loop."""
    return {"recorded_via": "proxy", "episodes": ["x"], "output": "y", "model": "m", "log": [
        _m(0, "You are the Coordinator.", ["ask_coordinator"],
           tcs=[{"id": "c1", "name": "ask_coordinator", "input": {}}]),
        {"seq": 1, "kind": "tool:ask_coordinator", "depth": 0, "result": "looped"},
        _m(2, "You are the Coordinator.", ["ask_coordinator"], text="done"),
    ]}


def _deep_nesting(depth=8):
    """A chain of `depth` distinct sub-agents each delegating one level deeper."""
    log = []
    for i in range(depth):
        log.append(_m(len(log), f"You are agent level {i}.", [f"ask_{i+1}"],
                      tcs=[{"id": f"d{i}", "name": f"ask_{i+1}", "input": {}}]))
        log.append({"seq": len(log), "kind": f"tool:ask_{i+1}", "depth": 0, "result": "ok"})
    log.append(_m(len(log), f"You are agent level {depth}.", [], text="bottom"))
    return {"recorded_via": "proxy", "episodes": ["go"], "output": "bottom", "systems": {}, "log": log, "model": "m"}


def _unicode_and_control():
    weird = "日本語 🔥\x00\x07\x1b[31m <script>alert(1)</script> " + "🎉" * 500
    return {"recorded_via": "proxy", "episodes": [weird], "output": weird, "model": "m", "systems": {}, "log": [
        _m(0, "agent", ["t"], tcs=[{"id": "u", "name": "t", "input": {"q": weird}}]),
        {"seq": 1, "kind": "tool:t", "depth": 0, "result": weird, "meta": {"tuid": "u"}},
        _m(2, "agent", ["t"], text=weird),
    ]}


def _orphan_tool_result():
    """A tool_result whose tool_use_id matches no call must not crash attribution."""
    return {"recorded_via": "proxy", "episodes": ["x"], "output": "d", "model": "m", "systems": {}, "log": [
        _m(0, "agent", ["Read"], tcs=[{"id": "real", "name": "Read", "input": {}}]),
        {"seq": 1, "kind": "tool:Read", "depth": 0, "result": "r", "meta": {"tuid": "GHOST"}},
        _m(2, "agent", ["Read"], text="d"),
    ]}


COMPLEX = [_parallel_same_name_out_of_order(), _self_delegation(), _deep_nesting(),
           _unicode_and_control(), _orphan_tool_result()]


@pytest.mark.parametrize("data", COMPLEX)
def test_debugger_surfaces_survive_complex_adversarial_traces(data):
    """steps_for / infer_agents / context_at / static_page + every analyzer must
    handle complex adversarial shapes without raising or hanging."""
    from loom.debugger import context_at, static_page, steps_for
    from loom.multiagent import infer_agents

    steps = steps_for(data)
    assert isinstance(steps, list)
    ia = infer_agents(data)
    assert isinstance(ia["agents"], list)
    context_at(data, 0)
    context_at(data, 10_000)
    static_page(data)                    # must produce a page, not crash
    for fn in ANALYZERS:
        fn(data)


def test_parallel_same_name_results_bind_by_id_at_scale():
    """The 张冠李戴 fix holds at scale: 10 Reads, results reversed -> each binds to
    its own file by tuid."""
    from loom.action import actions
    data = _parallel_same_name_out_of_order(10)
    by_file = {a.input["file"]: a.observation.text
               for a in actions(data) if a.type == "call"}
    assert by_file == {f"f{i}.py": f"content-{i}" for i in range(10)}
