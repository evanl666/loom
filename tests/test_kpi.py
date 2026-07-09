"""loom kpi: platform-team KPI aggregation over a corpus."""

from loom import Agent, tool
from loom.cli import main
from loom.kpi import compute_kpis, kpi_html, kpi_text
from loom.providers import ModelResponse, ScriptedProvider, ToolCall


@tool
def get_customer(id: int) -> str:
    "lookup"
    return "Jane"


@tool
def issue_refund(amount: int, order_id: str) -> str:
    "refund"
    return "ok"


@tool
def Read(file_path: str) -> str:
    "read"
    return "data"


def _corpus(tmp_path):
    Agent(model=ScriptedProvider([
        ModelResponse(tool_calls=[ToolCall("t", "get_customer", {"id": 1})], stop_reason="tool_use"),
        ModelResponse(tool_calls=[ToolCall("t2", "issue_refund", {"amount": 50, "order_id": "A"})],
                      stop_reason="tool_use"),
        ModelResponse(text="done", usage={"input_tokens": 100, "output_tokens": 20}),
    ]), tools=[get_customer, issue_refund, Read]).run("go").save(str(tmp_path / "r1.loom.json"))

    Agent(model=ScriptedProvider([
        ModelResponse(tool_calls=[ToolCall("t", "Read", {"file_path": "a"})], stop_reason="tool_use"),
        ModelResponse(text="done", usage={"input_tokens": 50000, "output_tokens": 2000}),
    ]), tools=[get_customer, issue_refund, Read]).run("go").save(str(tmp_path / "r2.loom.json"))

    r = Agent(model=ScriptedProvider([
        ModelResponse(tool_calls=[ToolCall("t", "Read", {"file_path": "a"})], stop_reason="tool_use"),
        ModelResponse(text="ERROR"),
    ]), tools=[get_customer, issue_refund, Read]).run("go")
    import json
    d = r.to_dict()
    d["stop_reason"] = "budget"
    (tmp_path / "r3.loom.json").write_text(json.dumps(d))
    return str(tmp_path)


def test_compute_kpis_aggregates_the_corpus(tmp_path):
    k = compute_kpis([_corpus(tmp_path)] and __import__("glob").glob(str(tmp_path / "*.loom.json")))
    assert k["runs"] == 3
    assert k["failure_rate"] == 33 and k["failed"] == 1
    assert k["risky_calls"] == 2
    caps = {c["capability"]: c for c in k["capabilities"]}
    assert caps["pii_access"]["runs"] == 1 and caps["money_movement"]["actions"] == 1
    assert k["tokens"]["p95"] >= 50000            # the expensive run shows in the tail
    tools = {t["tool"] for t in k["top_risky_tools"]}
    assert {"get_customer", "issue_refund"} <= tools


def test_kpi_renderers(tmp_path):
    import glob
    _corpus(tmp_path)
    k = compute_kpis(glob.glob(str(tmp_path / "*.loom.json")))
    assert "failure rate" in kpi_text(k) and "capability exposure" in kpi_text(k)
    html = kpi_html(k)
    assert "Loom agent KPIs" in html and "pii_access" in html and "tile" in html


def test_cli_kpi_html(tmp_path, capsys):
    _corpus(tmp_path)
    out = tmp_path / "kpi.html"
    assert main(["kpi", str(tmp_path), "--html", str(out)]) == 0
    assert "money_movement" in out.read_text()
    assert "KPI dashboard ->" in capsys.readouterr().out


def test_tool_trust_ranks_by_risk_and_undo(tmp_path):
    import glob
    from loom.kpi import tool_trust

    _corpus(tmp_path)
    rows = {r["tool"]: r for r in tool_trust(glob.glob(str(tmp_path / "*.loom.json")))}
    # a plain Read is fully trusted; a risky tool scores lower
    assert rows["Read"]["trust"] == 100
    assert rows["get_customer"]["trust"] < 100 and rows["get_customer"]["risky_rate"] == 100
    # undo support lifts a risky tool above one without it
    assert rows["issue_refund"]["undo_support"] == 100
    assert rows["issue_refund"]["trust"] > rows["get_customer"]["trust"]


def test_cli_tools_trust(tmp_path, capsys):
    from loom.cli import main

    _corpus(tmp_path)
    assert main(["tools", "--trust", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "trust" in out and "get_customer" in out and "Read" in out
