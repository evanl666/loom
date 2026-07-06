"""Forking rewinds to a turn, edits context, and takes a different branch.

Uses a context-sensitive RuleProvider so that editing the context genuinely
changes what the model does downstream -- the whole point of a fork.
"""

from loom import Agent
from loom.providers import ModelResponse, RuleProvider


def _last_user_text(messages):
    for m in reversed(messages):
        if m["role"] == "user":
            return m["content"]
    return ""


def build_agent():
    # If the prompt mentions "celsius", answer in C; otherwise answer in F.
    def rule_celsius(messages):
        if "celsius" in _last_user_text(messages).lower():
            return ModelResponse(text="It is 20 degrees celsius.", stop_reason="end_turn")
        return None

    def rule_default(messages):
        return ModelResponse(text="It is 68 degrees fahrenheit.", stop_reason="end_turn")

    provider = RuleProvider(rules=[rule_celsius, rule_default])
    return Agent(model=provider)


def test_fork_changes_branch_via_context_edit():
    agent = build_agent()
    run = agent.run("What is the temperature in celsius?")
    assert "celsius" in run.output

    # Rewind to turn 0 and rewrite the user's question -- a different branch.
    def drop_celsius(ctx):
        ctx.items[0].content = "What is the temperature?"

    forked = run.fork(at=0, edit=drop_celsius)
    assert "fahrenheit" in forked.output
    assert forked.output != run.output


def test_fork_without_edit_reproduces_original():
    agent = build_agent()
    run = agent.run("What is the temperature in celsius?")
    forked = run.fork(at=0)
    assert forked.output == run.output


def test_fork_out_of_range_raises():
    agent = build_agent()
    run = agent.run("hello")
    try:
        run.fork(at=5)
        assert False, "expected IndexError"
    except IndexError:
        pass
