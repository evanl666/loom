"""loom scan: flag ungated dangerous tools + unknown-capability tools."""

from loom import Agent, tool
from loom.providers import ModelResponse, ScriptedProvider, ToolCall
from loom.scan import scan


@tool(capabilities={"money_movement", "external_side_effect"})
def wire(to: str, amount: float) -> str:
    "Wire money."
    return "sent"


@tool
def mystery(x: str) -> str:
    "An unclassifiable tool."
    return "ok"


def _run():
    prov = ScriptedProvider([
        ModelResponse(tool_calls=[ToolCall("t1", "wire", {"to": "x", "amount": 9})], stop_reason="tool_use"),
        ModelResponse(tool_calls=[ToolCall("t2", "mystery", {"x": "y"})], stop_reason="tool_use"),
        ModelResponse(text="done", stop_reason="end_turn"),
    ])
    return Agent(model=prov, tools=[wire, mystery]).run("do it").to_dict()


def test_scan_flags_ungated_money_movement_and_unknown_tool():
    report = scan(_run())
    issues = " | ".join(f["issue"] for f in report["findings"])
    assert "money_movement tool ran with no firewall rule" in issues
    assert "unknown capability" in issues
    assert report["high"] >= 1
    assert report["grade"] in ("D", "F")  # a high finding drops the grade
    names = {t["name"] for t in report["tools"]}
    assert {"wire", "mystery"} <= names
