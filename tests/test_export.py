"""HTML export renders a complete, escaped, self-contained page."""

from loom import Agent, tool, trace_to_html
from loom.providers import ModelResponse, ScriptedProvider, ToolCall


@tool
def add(a: int, b: int) -> int:
    "Add two numbers."
    return a + b


def make_run():
    provider = ScriptedProvider(
        [
            ModelResponse(
                tool_calls=[ToolCall("t1", "add", {"a": 2, "b": 3})],
                stop_reason="tool_use",
                usage={"input_tokens": 10, "output_tokens": 4},
            ),
            ModelResponse(
                text="The answer is <b>5</b> & done.",  # exercises escaping
                stop_reason="end_turn",
                usage={"input_tokens": 20, "output_tokens": 6},
            ),
        ]
    )
    return Agent(model=provider, tools=[add]).run("What is 2 + 3?")


def test_export_contains_the_essentials():
    html_page = trace_to_html(make_run().to_dict())
    assert html_page.startswith("<!DOCTYPE html>")
    assert "What is 2 + 3?" in html_page
    assert "tool:add" in html_page
    assert ">30<" in html_page  # input tokens total
    # Model text is escaped, not injected as markup.
    assert "<b>5</b>" not in html_page
    assert "&lt;b&gt;5&lt;/b&gt; &amp; done." in html_page


def test_export_marks_paused_runs():
    from loom import ask_human

    provider = ScriptedProvider(
        [
            ModelResponse(
                tool_calls=[ToolCall("h1", "ask_human", {"question": "Proceed?"})],
                stop_reason="tool_use",
            )
        ]
    )
    run = Agent(model=provider, tools=[ask_human()]).run("Do the thing.")
    page = trace_to_html(run.to_dict())
    assert "paused" in page
    assert "Proceed?" in page


def test_studio_workspace_panel_and_dirty_banner():
    from loom.export import trace_to_html

    data = {
        "model": "m", "episodes": ["fix"], "output": "done", "log": [],
        "workspace": {
            "os": "Linux", "cwd": "/repo",
            "git": {"commit": "abc1234567", "branch": "main", "dirty": True},
            "changes": {
                "stat": "2 files changed", "dirty_hash": "beef",
                "files": [{"status": "M", "path": "app.py", "pre_existing": False},
                          {"status": "A", "path": "new.py", "pre_existing": True}],
                "diff": "--- a/app.py\n+++ b/app.py\n+new\n",
            },
        },
    }
    html = trace_to_html(data)
    assert "dirty working tree" in html          # top banner
    assert ">Workspace</h2>" in html             # the panel
    assert "app.py" in html and "new.py" in html
    assert "was dirty" in html                   # pre_existing marker
    assert "view patch" in html and "+new" in html  # the embedded diff
