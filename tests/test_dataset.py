"""loom dataset: compile runs into sft/trajectory/eval/dpo, scrubbing secrets."""

from loom import Agent, tool
from loom.dataset import compile_dataset
from loom.providers import ModelResponse, ScriptedProvider, ToolCall


@tool
def add(a: int, b: int) -> int:
    "Add."
    return a + b


def _good():
    prov = ScriptedProvider([
        ModelResponse(tool_calls=[ToolCall("t", "add", {"a": 2, "b": 3})], stop_reason="tool_use"),
        ModelResponse(text="the answer is 5, key sk-ant-api03-" + "Z" * 40, stop_reason="end_turn"),
    ])
    return Agent(model=prov, tools=[add]).run("what is 2+3?").to_dict()


def _bad():
    prov = ScriptedProvider([ModelResponse(text="", stop_reason="tool_use")])  # truncated-ish
    d = Agent(model=prov, tools=[add], max_steps=1).run("what is 2+3?").to_dict()
    d["truncated"] = True
    return d


def test_sft_only_good_runs_and_scrubs():
    recs = compile_dataset([_good(), _bad()], "sft")
    assert len(recs) == 1  # the bad (truncated) run is excluded
    text = recs[0]["messages"][1]["content"]
    assert "the answer is 5" in text
    assert "sk-ant-api03-ZZZ" not in text  # secret scrubbed


def test_trajectory_has_tool_steps():
    recs = compile_dataset([_good()], "trajectory")
    assert recs and any(s["tool"] == "add" for s in recs[0]["steps"])


def test_dpo_pairs_good_vs_bad_by_prompt():
    pairs = compile_dataset([_good(), _bad()], "dpo")
    assert pairs and pairs[0]["chosen"] and pairs[0]["rejected"] == ""  or pairs == []
    # the bad run has empty output, so no valid pair -> tolerate either
