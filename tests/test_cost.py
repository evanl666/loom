"""loom cost: token-burn root-cause analysis."""

from loom import Agent, tool
from loom.cli import main
from loom.cost import analyze_cost, describe_cost
from loom.providers import ModelResponse, ScriptedProvider, ToolCall


@tool
def search(q: str) -> str:
    "search"
    # one giant result (q='0'), the rest tiny -> a single dominating result
    return "X" * 40000 if q == "0" else "ok"


def _burn_run():
    resp = [ModelResponse(tool_calls=[ToolCall(f"t{i}", "search", {"q": str(i)})],
                          stop_reason="tool_use",
                          usage={"input_tokens": 1000 * (i + 1) * (i + 1), "output_tokens": 50})
            for i in range(6)]
    resp.append(ModelResponse(text="done", usage={"input_tokens": 50000, "output_tokens": 100}))
    return Agent(model=ScriptedProvider(resp), tools=[search]).run("go")


def test_detects_bloat_looping_and_overfetch():
    c = analyze_cost(_burn_run().to_dict())
    patterns = {f["pattern"] for f in c["findings"]}
    assert "context bloat" in patterns
    assert "looping" in patterns
    assert "tool-result explosion" in patterns
    assert c["total_tokens"] > 100000
    assert ("search", 6) in c["top_tools"]
    assert "fix:" in describe_cost(c)


def test_clean_cheap_run_has_no_findings():
    run = Agent(model=ScriptedProvider([
        ModelResponse(text="42", usage={"input_tokens": 100, "output_tokens": 5})])).run("q")
    c = analyze_cost(run.to_dict())
    assert c["findings"] == []
    assert "no burn pattern" in describe_cost(c)


def test_cli_cost_gate(tmp_path, capsys):
    path = str(tmp_path / "r.loom.json")
    _burn_run().save(path)
    assert main(["cost", path, "--gate"]) == 1
    out = capsys.readouterr().out
    assert "burn patterns" in out and "context bloat" in out
