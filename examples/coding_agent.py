"""A coding agent, debugged: file-edit state diffs and the git undo plan.

Offline -- a ScriptedProvider plays the model. Swap it for a real model (or
record Claude Code itself with `loom record claude "..." --safe`).
"""

from _shared import show

from loom import Agent, tool
from loom.providers import ModelResponse, ScriptedProvider, ToolCall


@tool
def Read(file_path: str) -> str:
    "Read a file."
    return "def add(a, b):\n    return a - b   # BUG\n"


@tool
def Edit(file_path: str, old: str, new: str) -> str:
    "Replace text in a file."
    return "edited"


model = ScriptedProvider([
    ModelResponse(text="Let me look at the failing function first.",
                  tool_calls=[ToolCall("t1", "Read", {"file_path": "src/math_utils.py"})],
                  stop_reason="tool_use"),
    ModelResponse(text="Found it: add() subtracts. Fixing the operator.",
                  tool_calls=[ToolCall("t2", "Edit", {
                      "file_path": "src/math_utils.py",
                      "old": "return a - b", "new": "return a + b"})],
                  stop_reason="tool_use"),
    ModelResponse(text="Fixed: add() now returns a + b."),
])

run = Agent(model=model, tools=[Read, Edit], name="coder").run(
    "tests fail: add(2, 2) returns 0 -- fix it")
show(run, "coding_agent.loom.json")
