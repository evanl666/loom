"""Intent firewall: flag consequential actions that don't serve the request."""

from loom.intent import intent_scan
from loom.providers import ModelResponse


class _Judge:
    """A mock judge: refunds/emails are off-mission, everything else aligned."""
    model = "mock"

    def complete(self, system, messages, tools):
        content = messages[0]["content"]
        if "issue_refund" in content or "send_email" in content:
            return ModelResponse(text='{"aligned": false, "score": 0.05, "reason": "off-mission"}')
        return ModelResponse(text='{"aligned": true, "score": 0.95, "reason": "ok"}')


def _trace():
    return {"log": [
        {"seq": 0, "kind": "model", "result": {"tool_calls": [{"id": "1", "name": "get_order", "input": {"id": "1"}}], "stop_reason": "tool_use"}},
        {"seq": 1, "kind": "tool:get_order", "result": "{}"},
        {"seq": 2, "kind": "model", "result": {"tool_calls": [{"id": "2", "name": "issue_refund", "input": {"amount": 9}}], "stop_reason": "tool_use"}},
        {"seq": 3, "kind": "tool:issue_refund", "result": "REFUNDED"}],
        "prompt": "Summarize the order.", "episodes": ["Summarize the order."], "output": "done",
        "tools": {"issue_refund": ["money_movement"], "get_order": ["read"]}}


def test_intent_flags_off_mission_money_but_not_the_read():
    findings = intent_scan(_trace(), judge=_Judge())
    tools = {f["tool"] for f in findings}
    assert "issue_refund" in tools  # money movement the user didn't ask for
    assert "get_order" not in tools  # a read that serves "summarize" is fine
    assert findings[0]["score"] < 0.5


def test_intent_no_findings_when_all_aligned():
    # a trace whose only consequential call the mock judge deems aligned
    trace = {"log": [
        {"seq": 0, "kind": "model", "result": {"tool_calls": [{"id": "1", "name": "write_report", "input": {}}], "stop_reason": "tool_use"}},
        {"seq": 1, "kind": "tool:write_report", "result": "ok"}],
        "prompt": "Write the report.", "episodes": ["Write the report."], "output": "done",
        "tools": {"write_report": ["write"]}}
    assert intent_scan(trace, judge=_Judge()) == []
