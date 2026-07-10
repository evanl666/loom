"""Incident autopilot: one trace -> the full fix bundle."""

from loom import Agent, tool
from loom.autopilot import run_autopilot
from loom.providers import ModelResponse, ScriptedProvider, ToolCall


@tool(capabilities={"money_movement"})
def wire(x: int) -> str:
    "Wire money."
    return "sent"


def test_autopilot_produces_the_full_bundle(tmp_path):
    prov = ScriptedProvider([
        ModelResponse(tool_calls=[ToolCall("t", "wire", {"x": 1})], stop_reason="tool_use"),
        ModelResponse(text="done", stop_reason="end_turn"),
    ])
    trace = tmp_path / "incident.loom.json"
    Agent(model=prov, tools=[wire]).run("do it").save(str(trace))

    out = tmp_path / "fix"
    manifest = run_autopilot(str(trace), str(out))

    for f in ("autopsy.html", "movie.html", "incident.md", "diagnosis.md",
              "policy-patch.yml", "PR-BODY.md"):
        assert (out / f).exists(), f"missing {f}"
    # the patch denies the money-moving tool that caused the incident
    assert "wire*" in manifest["deny"]
    assert "wire*" in (out / "policy-patch.yml").read_text()
    assert (out / "PR-BODY.md").read_text().startswith("# 🤖 Loom autopilot")
