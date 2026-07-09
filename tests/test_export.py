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


def test_studio_action_debugger_panel():
    """The main view is an action timeline: why / input / output / state diff /
    risk / policy decision, world-neutral."""

    @tool
    def Edit(file_path: str, old: str, new: str) -> str:
        "Edit a file."
        return "edited"

    provider = ScriptedProvider([
        ModelResponse(text="Fixing the bug in app.py.",
                      tool_calls=[ToolCall("t1", "Edit",
                                           {"file_path": "src/app.py", "old": "a", "new": "b"})],
                      stop_reason="tool_use"),
        ModelResponse(text="done"),
    ])
    run = Agent(model=provider, tools=[Edit]).run("fix the bug")
    data = run.to_dict()
    data["workspace"] = {"changes": {"files": [
        {"path": "src/app.py", "status": "M", "pre_existing": False}]}}
    data["shield_events"] = [
        {"tool": "Edit", "input": {"file_path": "src/app.py"}, "action": "approve",
         "rule": "cap:write", "via": "operator", "by": "evan"},
    ]
    page = trace_to_html(data)
    assert "Actions — what it did, why, what changed" in page
    assert "Fixing the bug in app.py." in page            # why (intent)
    assert "fs-write" in page                             # risk badge
    assert "Δ wrote src/app.py" in page                   # state diff (Coding Pack)
    assert "🛡 approve · cap:write" in page               # policy decision
    assert 'data-seq=' in page                            # wired to the scrubber
    assert ">Raw effects</h2>" in page                    # effect log demoted


def test_studio_impact_map_bipartite_worlds():
    from loom.packs import install_builtin
    install_builtin()

    @tool
    def Edit(file_path: str, new: str) -> str:
        "edit"
        return "edited"

    @tool
    def run_sql(query: str) -> str:
        "sql"
        return "1 row inserted"

    @tool
    def WebFetch(url: str) -> str:
        "fetch"
        return "data"

    provider = ScriptedProvider([
        ModelResponse(tool_calls=[ToolCall("t1", "Edit", {"file_path": "a.py", "new": "x"})],
                      stop_reason="tool_use"),
        ModelResponse(tool_calls=[ToolCall("t2", "run_sql", {"query": "INSERT INTO t VALUES (1)"})],
                      stop_reason="tool_use"),
        ModelResponse(tool_calls=[ToolCall("t3", "WebFetch", {"url": "http://x"})],
                      stop_reason="tool_use"),
        ModelResponse(text="done"),
    ])
    page = trace_to_html(Agent(model=provider, tools=[Edit, run_sql, WebFetch]).run("go").to_dict())
    assert "Impact map — what it touched in the world" in page
    assert "<svg viewBox" in page
    assert "📄 files" in page and "🗄 database" in page and "↗ network" in page
    assert "fs-write" in page and "db-write" in page   # edge risk labels


def test_studio_impact_map_absent_when_nothing_touched():
    # A pure Q&A run (no tool calls) has no world to map.
    provider = ScriptedProvider([ModelResponse(text="42")])
    page = trace_to_html(Agent(model=provider).run("what is 6*7?").to_dict())
    assert "Impact map" not in page


def test_studio_shows_blocked_actions():
    page = trace_to_html({
        "model": "m", "episodes": ["go"], "output": "x",
        "log": [{"seq": 0, "kind": "model", "key": "k",
                 "result": {"text": "reading env", "tool_calls": [], "stop_reason": "end_turn",
                            "usage": {}}}],
        "shield_events": [{"tool": "Read", "input": {"file_path": "/app/.env"},
                           "action": "deny", "rule": "Read(*.env*)", "via": "rule"}],
    })
    assert "🛡 blocked · Read(*.env*)" in page
    assert "secret-read" in page


def test_studio_action_buttons_use_the_trace_path():
    from loom.export import trace_to_html

    html = trace_to_html({"model": "m", "episodes": ["fix"], "output": "x", "log": []},
                         path="runs/deploy.loom.json")
    assert 'class="actions"' in html
    assert "loom incident runs/deploy.loom.json" in html
    assert "loom heal runs/deploy.loom.json" in html
    assert "loom proxy --replay runs/deploy.loom.json" in html
    assert 'data-cmd=' in html and "navigator.clipboard.writeText(b.dataset.cmd)" in html
