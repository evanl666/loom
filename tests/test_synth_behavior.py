"""Policy synthesis + behavior unit-test generator."""

from loom import Agent, tool
from loom.behavior import behavior_spec, to_pytest
from loom.providers import ModelResponse, ScriptedProvider, ToolCall
from loom.synth import synthesize_policy


@tool(capabilities={"money_movement"})
def refund(x: int) -> str:
    "Refund."
    return "ok"


@tool
def get_data(x: int) -> str:
    "Read."
    return "data"


def _run():
    prov = ScriptedProvider([
        ModelResponse(tool_calls=[ToolCall("t", "get_data", {"x": 1})], stop_reason="tool_use"),
        ModelResponse(tool_calls=[ToolCall("t2", "refund", {"x": 1})], stop_reason="tool_use"),
        ModelResponse(text="done", stop_reason="end_turn"),
    ])
    return Agent(model=prov, tools=[refund, get_data]).run("do it").to_dict()


def test_synthesize_least_privilege_denies_money_allows_read():
    doc = synthesize_policy([_run()], goal="least-privilege")
    assert doc["default"] == "confirm"
    assert "refund*" in doc["deny"]      # money_movement -> deny
    assert "get_data*" in doc["allow"]   # read -> allow


def test_behavior_spec_and_generated_pytest(tmp_path):
    data = _run()
    trace = tmp_path / "r.loom.json"
    import json
    trace.write_text(json.dumps(data))
    spec = behavior_spec(data)
    assert spec["tools_used"] == ["get_data", "refund"]
    assert ["get_data", "refund"] in spec["required_order"]
    # the generated pytest is valid Python and its contract holds on its source
    src = to_pytest(spec, trace_path=str(trace))
    testfile = tmp_path / "test_contract.py"
    testfile.write_text(src)
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "-m", "pytest", str(testfile), "-q"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
