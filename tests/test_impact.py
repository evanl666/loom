"""Impact analysis: replay the corpus against a changed config, free."""

from loom import Agent, tool
from loom.impact import assess, assess_trace, report
from loom.providers import ModelResponse, RuleProvider, ToolCall


@tool
def lookup(city: str) -> str:
    "Look up a city."
    return f"{city}: ok"


def build_agent(system: str = "You are helpful.") -> Agent:
    def wants_tool(messages):
        if not any(m["role"] == "tool" for m in messages):
            return ModelResponse(
                tool_calls=[ToolCall("t1", "lookup", {"city": "Berlin"})],
                stop_reason="tool_use",
            )
        return None

    def answer(messages):
        return ModelResponse(text="Berlin looks fine.", stop_reason="end_turn")

    return Agent(model=RuleProvider(rules=[wants_tool, answer]), tools=[lookup], system=system)


def record_trace(tmp_path, name="a.loom.json", system="You are helpful."):
    run = build_agent(system).run("How is Berlin?")
    path = str(tmp_path / name)
    run.save(path)
    return path


def test_unchanged_config_reports_unchanged(tmp_path):
    path = record_trace(tmp_path)
    impact = assess_trace(path, build_agent())
    assert impact.verdict == "unchanged"
    assert not impact.changed


def test_system_prompt_change_is_inputs_differ_at_seq_zero(tmp_path):
    path = record_trace(tmp_path)
    impact = assess_trace(path, build_agent(system="You are EXTREMELY helpful."))
    assert impact.verdict == "inputs-differ"
    assert impact.first_seq == 0  # the very first model call sees the new system
    assert impact.changed


def test_structural_change_is_detected(tmp_path):
    path = record_trace(tmp_path)
    # Turning compaction on makes the harness want a "compact" effect where the
    # recording has none: the new config walks a structurally different path.
    # (Swapping the MODEL is invisible to dry mode by design -- replay never
    # consults the provider; use --live to see output changes.)
    compacting = build_agent()
    compacting.compact_after = 1
    compacting.compact_keep = 0
    impact = assess_trace(path, compacting)
    assert impact.verdict == "structure-differs"


def test_live_mode_shows_output_change(tmp_path):
    path = record_trace(tmp_path)

    def contrarian(messages):
        return ModelResponse(text="Berlin is overrated.", stop_reason="end_turn")

    changed = Agent(
        model=RuleProvider(rules=[contrarian]),
        tools=[lookup],
        system="You are helpful.",
    )
    impact = assess_trace(path, changed, live=True)
    assert impact.verdict == "outputs-differ"
    assert "overrated" in impact.detail


def test_report_counts_affected(tmp_path):
    p1 = record_trace(tmp_path, "a.loom.json")
    p2 = record_trace(tmp_path, "b.loom.json")
    impacts = assess([p1, p2], build_agent(system="New instructions."))
    text = report(impacts)
    assert "2 of 2 recorded run(s) affected" in text


def test_cli_missing_trace_is_exit_2_not_impact(tmp_path, monkeypatch, capsys):
    from loom.cli import main

    agent_mod = tmp_path / "myagent.py"
    agent_mod.write_text("from tests.test_impact import build_agent\nsame = build_agent()\n")
    monkeypatch.chdir(tmp_path)
    code = main(["impact", str(tmp_path / "nope.loom.json"), "--agent", "myagent:same"])
    assert code == 2  # unreadable corpus is an error, not "affected"
    assert "could not read" in capsys.readouterr().err


def test_cli_impact_exit_codes(tmp_path, monkeypatch, capsys):
    from loom.cli import main

    record_trace(tmp_path, "a.loom.json")
    agent_mod = tmp_path / "myagent.py"
    agent_mod.write_text(
        "from tests.test_impact import build_agent\n"
        "same = build_agent()\n"
        "changed = build_agent(system='Answer in French.')\n"
    )
    monkeypatch.chdir(tmp_path)

    assert main(["impact", str(tmp_path), "--agent", "myagent:same"]) == 0
    assert main(["impact", str(tmp_path), "--agent", "myagent:changed"]) == 1
    out = capsys.readouterr().out
    assert "unchanged" in out and "inputs-differ" in out