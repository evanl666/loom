"""Impact analysis: replay the corpus against a changed config, free."""

import json

from loom import Agent, tool
from loom.impact import assess, assess_trace, cost_delta, cost_delta_files, report, to_json
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


def test_dry_mode_sizes_model_inputs(tmp_path):
    path = record_trace(tmp_path)
    small = assess_trace(path, build_agent())
    big = assess_trace(path, build_agent(system="You are helpful. " * 200))
    assert small.est_input_tokens and small.est_input_tokens > 0
    assert big.est_input_tokens > small.est_input_tokens  # bigger prompt = more expensive
    assert small.verdict == "unchanged" and big.verdict == "inputs-differ"


def test_structure_differs_is_not_sized(tmp_path):
    path = record_trace(tmp_path)
    compacting = build_agent()
    compacting.compact_after = 1
    compacting.compact_keep = 0
    assert assess_trace(path, compacting).est_input_tokens is None


def test_live_mode_reports_actual_input_tokens(tmp_path):
    path = record_trace(tmp_path)
    impact = assess_trace(path, build_agent(), live=True)
    # RuleProvider reports no usage, so live spend is honestly None here --
    # the point is the field is filled from actual cost, not the estimator.
    from loom import Run

    assert impact.est_input_tokens == (Run.load(path, agent=build_agent()).cost()["input_tokens"] or None)


def test_cost_delta_compares_common_sized_runs():
    base = [
        {"path": "a", "est_input_tokens": 1000},
        {"path": "b", "est_input_tokens": None},  # unsized on base: excluded
        {"path": "gone", "est_input_tokens": 50},  # absent on head: excluded
    ]
    head = [
        {"path": "a", "est_input_tokens": 1120},
        {"path": "b", "est_input_tokens": 400},
        {"path": "new", "est_input_tokens": 70},  # absent on base: excluded
    ]
    line = cost_delta(base, head)
    assert "12.0% more expensive" in line
    assert "~1,000 -> ~1,120" in line and "1 recorded run(s)" in line
    assert cost_delta(base, []) == ""
    assert "unchanged" in cost_delta(head, head)


def test_cli_json_report_and_cost_delta_files(tmp_path, monkeypatch):
    from loom.cli import main

    record_trace(tmp_path, "a.loom.json")
    agent_mod = tmp_path / "myagent_cost.py"
    agent_mod.write_text(
        "from tests.test_impact import build_agent\n"
        "same = build_agent()\n"
        "pricier = build_agent(system='You are helpful. ' * 200)\n"
    )
    monkeypatch.chdir(tmp_path)

    assert main(["impact", str(tmp_path), "--agent", "myagent_cost:same",
                 "--json", "base.json"]) == 0
    assert main(["impact", str(tmp_path), "--agent", "myagent_cost:pricier",
                 "--json", "head.json"]) == 1
    with open("base.json") as f:
        base = json.load(f)
    assert base["total"] == 1 and base["affected"] == 0
    assert base["est_input_tokens"] > 0
    assert base["impacts"][0]["verdict"] == "unchanged"

    line = cost_delta_files("base.json", "head.json")
    assert "more expensive" in line


def test_report_includes_input_volume_line(tmp_path):
    path = record_trace(tmp_path)
    text = report(assess([path], build_agent()))
    assert "input volume under this config" in text
    data = to_json(assess([path], build_agent()))
    assert data["est_input_tokens"] == data["impacts"][0]["est_input_tokens"]


def test_cli_missing_trace_is_exit_2_not_impact(tmp_path, monkeypatch, capsys):
    from loom.cli import main

    # unique module name: sys.modules caches by name across tests in-process
    agent_mod = tmp_path / "myagent_missing.py"
    agent_mod.write_text("from tests.test_impact import build_agent\nsame = build_agent()\n")
    monkeypatch.chdir(tmp_path)
    code = main(["impact", str(tmp_path / "nope.loom.json"), "--agent", "myagent_missing:same"])
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

def test_to_json_carries_the_tool_inventory(tmp_path):
    from loom.impact import assess, to_json, tools_delta

    agent = build_agent()
    run = agent.run("What is 2+2?")
    path = str(tmp_path / "run.loom.json")
    run.save(path)

    head = to_json(assess([path], agent), agent=agent)
    assert head["agent_tools"] == sorted(agent.tools)
    assert to_json([], agent=None)["agent_tools"] is None

    base = dict(head, agent_tools=[t for t in head["agent_tools"]][:0])
    line = tools_delta(base, head)
    assert "grants the agent new tool(s)" in line

    # inventories equal -> no line; missing inventory -> no line (old loom)
    assert tools_delta(head, head) == ""
    assert tools_delta({"agent_tools": None}, head) == ""


def test_tools_delta_flags_dangerous_grants():
    from loom.impact import tools_delta

    base = {"agent_tools": ["Read", "Glob"]}
    head = {"agent_tools": ["Read", "Glob", "Bash", "summarize"]}
    line = tools_delta(base, head)
    assert "DANGEROUS" in line and "Bash" in line and "code-exec" in line
    assert "summarize" in line  # the safe one still mentioned
    # a purely safe addition doesn't cry wolf
    safe = tools_delta({"agent_tools": ["Read"]}, {"agent_tools": ["Read", "summarize"]})
    assert "DANGEROUS" not in safe and "summarize" in safe
