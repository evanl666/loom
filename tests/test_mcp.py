"""MCP integration: server tools flow through the Effect boundary.

Spawns the fixture server (tests/fixtures/mcp_math_server.py) over stdio.
Skipped entirely when the optional 'mcp' package is not installed.
"""

import os
import sys

import pytest

pytest.importorskip("mcp")

from loom import Agent, Run  # noqa: E402
from loom.mcp import MCPServer  # noqa: E402
from loom.providers import ModelResponse, ScriptedProvider, ToolCall  # noqa: E402

SERVER = os.path.join(os.path.dirname(__file__), "fixtures", "mcp_math_server.py")


@pytest.fixture(scope="module")
def server():
    s = MCPServer(sys.executable, [SERVER])
    yield s
    s.close()


def test_discovers_tools_with_schemas(server):
    tools = {t.name: t for t in server.tools()}
    assert {"add", "shout"} <= set(tools)
    assert tools["add"].input_schema["required"] == ["a", "b"]
    assert "Add two numbers" in tools["add"].description


def test_direct_call(server):
    assert server.call("add", a=2, b=3) == "5"
    assert server.call("shout", text="quiet") == "QUIET"


def test_mcp_calls_are_recorded_effects(server):
    provider = ScriptedProvider(
        [
            ModelResponse(
                tool_calls=[ToolCall("t1", "add", {"a": 20, "b": 22})],
                stop_reason="tool_use",
            ),
            ModelResponse(text="The answer is 42.", stop_reason="end_turn"),
        ]
    )
    agent = Agent(model=provider, tools=server.tools())
    run = agent.run("What is 20 + 22?")
    assert run.output == "The answer is 42."
    tool_effects = [e for e in run.log if e.kind == "tool:add"]
    assert len(tool_effects) == 1
    assert tool_effects[0].result == "42"


def test_replay_needs_no_server(server, tmp_path):
    provider = ScriptedProvider(
        [
            ModelResponse(
                tool_calls=[ToolCall("t1", "add", {"a": 1, "b": 2})],
                stop_reason="tool_use",
            ),
            ModelResponse(text="3", stop_reason="end_turn"),
        ]
    )
    agent = Agent(model=provider, tools=server.tools())
    run = agent.run("1 + 2?")
    path = str(tmp_path / "mcp.loom.json")
    run.save(path)

    # Replay against an agent whose 'add' tool would explode if ever executed:
    # the recorded result is served from the log, no MCP server required.
    from loom import Tool

    def boom(**kwargs):
        raise AssertionError("replay must not execute the tool")

    stub = Tool(name="add", description="stub", fn=boom, input_schema={"type": "object"})
    offline_agent = Agent(model=ScriptedProvider([]), tools=[stub])
    loaded = Run.load(path, agent=offline_agent)
    # The stub's schema differs from the real server tool's, so this is a
    # changed config -- strict replay would (rightly) flag it. strict=False
    # is the documented way to walk a trace without the original tools.
    assert loaded.replay(strict=False).output == "3"
