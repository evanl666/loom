"""Sweep: batch counterfactual forks with side-by-side comparison."""

import pytest

from loom import Agent
from loom.providers import ModelResponse, RuleProvider


def _last_user(messages):
    for m in reversed(messages):
        if m["role"] == "user":
            return m["content"].lower()
    return ""


def build_agent():
    # Answer depends on the question -- so context edits change the branch.
    def celsius(messages):
        if "celsius" in _last_user(messages):
            return ModelResponse(text="20 degrees celsius.", stop_reason="end_turn")
        return None

    def paris(messages):
        if "paris" in _last_user(messages):
            return ModelResponse(text="15 degrees in Paris.", stop_reason="end_turn")
        return None

    def default(messages):
        return ModelResponse(text="68 degrees fahrenheit.", stop_reason="end_turn")

    return Agent(model=RuleProvider(rules=[celsius, paris, default]))


def rewrite(new_text):
    def edit(ctx):
        ctx.items[0].content = new_text

    return edit


def test_sweep_produces_distinct_branches():
    agent = build_agent()
    run = agent.run("Temperature in celsius?")
    sweep = run.sweep(
        at=0,
        variants=[None, rewrite("Temperature?"), rewrite("Temperature in Paris?")],
        labels=["control", "plain", "paris"],
    )
    outputs = {label: r.output for label, r in sweep}
    assert outputs["control"] == run.output  # no-edit control reproduces the base
    assert "fahrenheit" in outputs["plain"]
    assert "Paris" in outputs["paris"]


def test_sweep_compare_rows():
    agent = build_agent()
    run = agent.run("Temperature in celsius?")
    sweep = run.sweep(at=0, variants=[None, rewrite("Temperature?")])
    rows = sweep.compare()
    assert [r["label"] for r in rows] == ["base", "v0", "v1"]  # default labels
    assert rows[1]["diverged_at"] is None  # control branch is identical
    assert rows[2]["diverged_at"] == 0  # edited branch diverges at the fork step
    assert rows[2]["output"] != run.output


def test_sweep_validates_inputs():
    agent = build_agent()
    run = agent.run("Temperature in celsius?")
    with pytest.raises(IndexError):
        run.sweep(at=5, variants=[None])
    with pytest.raises(ValueError):
        run.sweep(at=0, variants=[None, None], labels=["only-one"])
