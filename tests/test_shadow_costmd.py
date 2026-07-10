"""Shadow deployment eval + cost PR markdown."""

from loom import Agent, tool
from loom.cost import cost_markdown
from loom.providers import ModelResponse, ScriptedProvider, ToolCall
from loom.shadow import shadow_eval


@tool(capabilities={"money_movement"})
def refund(x: int) -> str:
    "Refund."
    return "ok"


def test_shadow_flags_breakage(tmp_path):
    prov = ScriptedProvider([
        ModelResponse(tool_calls=[ToolCall("t", "refund", {"x": 1})], stop_reason="tool_use"),
        ModelResponse(text="done", stop_reason="end_turn"),
    ])
    Agent(model=prov, tools=[refund]).run("refund").save(str(tmp_path / "r.loom.json"))
    pol = tmp_path / "p.yml"
    pol.write_text("default: allow\ndeny:\n  - refund*\n")

    r = shadow_eval([str(tmp_path)], str(pol))
    assert r["would_deny_runs"] == 1
    assert r["breakages"]  # the completed refund run would break
    assert not r["safe"] and "HOLD" in r["verdict"]


def test_cost_markdown_is_pr_ready():
    md = cost_markdown({"log": [
        {"seq": 0, "kind": "model", "result": {"usage": {"input_tokens": 100, "output_tokens": 20}, "tool_calls": []}}],
        "prompt": "p", "output": "o"})
    assert md.startswith("### 💸 Loom cost report")
    assert "tokens" in md
