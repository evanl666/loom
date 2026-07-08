"""loom why: a debugger agent that answers questions about a trace, citing seqs."""

import json

from loom.agent import Agent
from loom.providers.base import ModelResponse, ToolCall
from loom.providers.scripted import ScriptedProvider
from loom.tools import tool
from loom.why import build_why_agent, build_why_tools, why


@tool
def get_weather(city: str) -> str:
    "Weather for a city."
    return f"raining in {city}"


def _make_trace(tmp_path):
    provider = ScriptedProvider(
        [
            ModelResponse(
                text="Checking.",
                tool_calls=[ToolCall(id="t1", name="get_weather", input={"city": "Berlin"})],
                stop_reason="tool_use",
                usage={"input_tokens": 12, "output_tokens": 8},
            ),
            ModelResponse(text="It is raining in Berlin.",
                          usage={"input_tokens": 30, "output_tokens": 9}),
        ]
    )
    run = Agent(model=provider, tools=[get_weather]).run("weather in Berlin?")
    path = str(tmp_path / "run.loom.json")
    run.save(path)
    return path


def test_why_tools_read_the_trace(tmp_path):
    path = _make_trace(tmp_path)
    tools = {t.name: t for t in build_why_tools(path)}

    facts = json.loads(tools["conversation"].fn())
    assert facts["episodes"] == ["weather in Berlin?"]
    assert facts["output"] == "It is raining in Berlin."

    timeline = tools["timeline"].fn()
    assert "[0] model" in timeline and "get_weather" in timeline

    assert '"seq": 0' in tools["effect"].fn(seq=0)
    assert "no effect with seq 99" in tools["effect"].fn(seq=99)

    cost = tools["cost"].fn()
    assert "[0] in=12 out=8" in cost and "in=30" in cost

    assert tools["shield_log"].fn() == "(no shield events)"
    assert "context" in tools["checkup"].fn().lower() or tools["checkup"].fn()


def test_why_agent_investigates_and_cites_seqs(tmp_path):
    path = _make_trace(tmp_path)
    debugger = ScriptedProvider(
        [
            ModelResponse(
                text="Let me look at the timeline.",
                tool_calls=[ToolCall(id="d1", name="timeline", input={})],
                stop_reason="tool_use",
            ),
            ModelResponse(
                text="At seq 0 the model called get_weather(Berlin); the tool result "
                     "at seq 1 said raining, so the final answer follows."
            ),
        ]
    )
    run = why(path, "why did it say raining?", provider=debugger)
    assert "seq 0" in run.output
    # the diagnosis is itself a recorded loom run: replayable, saveable
    diagnosis = str(tmp_path / "diagnosis.loom.json")
    run.save(diagnosis)
    assert json.load(open(diagnosis))["log"][0]["kind"] == "model"


def test_build_why_agent_wires_system_and_tools(tmp_path):
    agent = build_why_agent(_make_trace(tmp_path), provider=ScriptedProvider([]))
    assert "cite the seq numbers" in agent.system
    assert set(agent.tools) >= {"conversation", "timeline", "effect", "cost"}
