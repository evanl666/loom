"""Behavioural assertion language over a recorded run."""

from loom.assertions import check_assertions


def _trace(output="Refunded order 42", *, blocked=False):
    log = [
        {"seq": 0, "kind": "model", "result": {"tool_calls": [
            {"id": "1", "name": "get_order", "input": {}}], "stop_reason": "tool_use"}},
        {"seq": 1, "kind": "tool:get_order", "result": "order 42"},
        {"seq": 2, "kind": "model", "result": {"text": output, "stop_reason": "end_turn"}},
    ]
    tr = {"log": log, "prompt": "refund", "output": output, "tools": {}}
    if blocked:
        # a firewall deny is recorded as a shield_event, surfaced as a
        # policy.blocked Action by actions()
        tr["shield_events"] = [{"action": "deny", "tool": "issue_refund",
                                "rule": "cap:money_movement", "input": {}}]
    return tr


def test_passing_assertions():
    r = check_assertions(_trace(), [
        "output contains order 42",
        "output matches [Rr]efunded",
        "calls get_*",
        "never issue_refund",
        "no blocked",
        "no risk",
        "steps < 10",
        "answers",
    ])
    assert r["all_pass"], [x for x in r["results"] if not x.get("ok")]
    assert r["passed"] == r["total"] == 8


def test_failing_and_unknown_assertions():
    r = check_assertions(_trace(), [
        "output contains not-present",   # fail
        "calls issue_refund",            # fail (never called)
        "steps > 100",                   # fail
        "gibberish assertion",           # error
    ])
    assert r["passed"] == 0
    assert not r["all_pass"]
    assert any("error" in x for x in r["results"])


def test_blocked_detection():
    tr = _trace(blocked=True)
    r = check_assertions(tr, ["blocked issue_refund", "no blocked"])
    ok = {x["expr"]: x.get("ok") for x in r["results"]}
    assert ok["blocked issue_refund"] is True
    assert ok["no blocked"] is False


def test_comments_and_blanks_ignored():
    r = check_assertions(_trace(), "# a comment\n\noutput contains order 42\n")
    assert r["total"] == 1 and r["all_pass"]
