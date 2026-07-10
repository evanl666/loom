"""Loop detection + conditional breakpoints."""

from loom.breakpoint import find_all_breaks, find_break
from loom.loops import detect_loops


def _looping_trace():
    log = []
    seq = ["search", "read", "search", "read", "search"]
    for i, n in enumerate(seq):
        log.append({"seq": i * 2, "kind": "model", "result": {
            "tool_calls": [{"id": str(i), "name": n, "input": {"q": "x"}}], "stop_reason": "tool_use"}})
        log.append({"seq": i * 2 + 1, "kind": "tool:" + n, "result": "..."})
    return {"log": log, "prompt": "p", "output": "o",
            "tools": {"search": ["network"], "read": ["read"]}}


def test_detects_repeat_and_cycle():
    r = detect_loops(_looping_trace())
    assert r["looping"]
    kinds = {f["kind"] for f in r["findings"]}
    assert "repeat" in kinds  # search called 3x
    assert "cycle" in kinds   # read<->search oscillation


def test_no_loop_on_progressing_run():
    log = []
    for i, n in enumerate(["a", "b", "c"]):
        log.append({"seq": i, "kind": "model", "result": {
            "tool_calls": [{"id": str(i), "name": n, "input": {}}], "stop_reason": "tool_use"}})
    assert not detect_loops({"log": log, "prompt": "p", "output": "o"})["looping"]


def test_breakpoint_conditions():
    t = _looping_trace()
    assert find_break(t, "cap:network")["hit"]
    assert find_break(t, "tool:search")["step"] == 1
    # sequence: read AFTER a network call
    r = find_break(t, "tool:read after cap:network")
    assert r["hit"] and r["step"] == 3
    # never-tripped
    assert not find_break(t, "cap:money_movement")["hit"]
    assert len(find_all_breaks(t, "tool:search")) == 3
