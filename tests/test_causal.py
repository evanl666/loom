"""Causal why: counterfactual forks prove which turn caused an action.

Uses a context-sensitive provider (deterministic, no API key): it deploys unless
a 'disregard' correction is in context -- so neutralizing the causal turn removes
the action, which causal_why should detect.
"""
import json

from loom import Agent, tool
from loom.insight import causal_why
from loom.providers import ModelResponse, ToolCall


@tool
def read_flag() -> str:
    "Read the deploy flag."
    return "DEPLOY=true"


@tool
def deploy() -> str:
    "Deploy to production."
    return "deployed"


class _CtxProvider:
    """Reads flag on turn 0; deploys on turn 1 UNLESS a correction is present."""
    model = "ctx"

    def complete(self, system, messages, tools):
        corrected = "disregard" in str(messages).lower()
        n_assistant = sum(1 for m in messages if m.get("role") == "assistant")
        if n_assistant == 0:
            return ModelResponse(tool_calls=[ToolCall("1", "read_flag", {})], stop_reason="tool_use")
        if n_assistant == 1 and not corrected:
            return ModelResponse(tool_calls=[ToolCall("2", "deploy", {})], stop_reason="tool_use")
        return ModelResponse(text="done", stop_reason="end_turn")


def test_causal_why_finds_the_cause_turn():
    agent = Agent(model=_CtxProvider(), tools=[read_flag, deploy])
    run = agent.run("check the flag and act on it")
    # find the deploy step
    from loom.action import actions
    deploy_step = next(a.step for a in actions(run.to_dict())
                       if a.type == "call" and a.tool == "deploy")

    result = causal_why(run, deploy_step)
    assert result["tool"] == "deploy"
    # neutralizing turn 0 (the flag read) removes the deploy -> it's the cause
    assert result["earliest_cause_turn"] == 0
    assert result["confidence"] > 0
