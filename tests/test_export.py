"""`trace_to_html` / `loom studio` now render the ONE debugger UI as a
self-contained static file (see loom.debugger.static_page). The old bespoke
Studio renderer was retired; its reusable analyzer panels (_impact_map,
_data_flow) live on for the incident/autopsy bundles and are covered there."""

import json

from loom import Agent, tool, trace_to_html
from loom.providers import ModelResponse, ScriptedProvider, ToolCall


@tool
def add(a: int, b: int) -> int:
    "Add two numbers."
    return a + b


def make_run():
    provider = ScriptedProvider([
        ModelResponse(tool_calls=[ToolCall("t1", "add", {"a": 2, "b": 3})],
                      stop_reason="tool_use",
                      usage={"input_tokens": 10, "output_tokens": 4}),
        ModelResponse(text="The answer is <b>5</b> & done.",
                      stop_reason="end_turn",
                      usage={"input_tokens": 20, "output_tokens": 6}),
    ])
    return Agent(model=provider, tools=[add]).run("What is 2 + 3?")


def _inlined(page: str) -> dict:
    blob = page.split("window.LOOM_STATIC=", 1)[1].split(";</script>", 1)[0]
    return json.loads(blob.replace("<\\/", "</"))


def test_export_is_the_self_contained_debugger_ui():
    page = trace_to_html(make_run().to_dict())
    assert page.startswith("<!DOCTYPE html>")
    assert "window.LOOM_STATIC=" in page      # data inlined
    assert "function renderTree" in page       # the debugger's UI, frozen
    assert "127.0.0.1" not in page             # self-contained, no server


def test_export_inlines_the_run():
    data = _inlined(trace_to_html(make_run().to_dict()))
    steps = data["run"]["steps"]
    assert any(s.get("tool") == "add" for s in steps)
    assert data["run"]["prompt"] == "What is 2 + 3?"
    # the user's prompt shows in the page
    assert "What is 2 + 3?" in trace_to_html(make_run().to_dict())


def test_export_cannot_break_out_of_the_inline_blob():
    # a </script> in trace content must be escaped so it can't close the tag
    data = {"model": "m", "episodes": ["</script><b>x"], "output": "</script>",
            "tools": {}, "log": [{"seq": 0, "kind": "model",
            "result": {"text": "</script><img src=x>", "stop_reason": "end_turn"}}]}
    page = trace_to_html(data)
    assert "</script><b>x" not in page
    assert "</script><img" not in page
    _inlined(page)  # still valid JSON


def test_export_carries_the_scrub_banner():
    unsafe = trace_to_html(make_run().to_dict())
    assert "Not scrubbed" in unsafe and "loom share" in unsafe
    d = make_run().to_dict()
    d["scrubbed"] = True
    assert "safe to share" in trace_to_html(d)


def test_export_survives_a_paused_run():
    from loom import ask_human

    provider = ScriptedProvider([
        ModelResponse(tool_calls=[ToolCall("h1", "ask_human", {"question": "Proceed?"})],
                      stop_reason="tool_use")])
    run = Agent(model=provider, tools=[ask_human()]).run("Do the thing.")
    page = trace_to_html(run.to_dict())
    assert page.startswith("<!DOCTYPE html>") and "Proceed?" in page


def test_studio_cli_renders_and_exits_zero(tmp_path, monkeypatch):
    import webbrowser

    from loom.cli import main

    monkeypatch.setattr(webbrowser, "open", lambda *a, **k: True)
    trace = tmp_path / "s.loom.json"
    make_run().save(str(trace))
    out = tmp_path / "s.html"
    assert main(["studio", str(trace), "-o", str(out)]) == 0
    assert out.exists() and out.read_text().startswith("<!DOCTYPE html>")
    assert "window.LOOM_STATIC=" in out.read_text()
