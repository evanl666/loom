"""Policy at the boundary: deny, confirm, dry-run, and hard token budgets."""

from loom import Agent, Policy, tool
from loom.providers import ModelResponse, ScriptedProvider, ToolCall

CALLS = {"read": 0, "delete": 0}


@tool
def read_data() -> str:
    "Read some data."
    CALLS["read"] += 1
    return "data contents"


@tool
def delete_data() -> str:
    "Delete the data. Destructive."
    CALLS["delete"] += 1
    return "deleted"


def provider_calling_both():
    return ScriptedProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall("t1", "read_data", {}),
                    ToolCall("t2", "delete_data", {}),
                ],
                stop_reason="tool_use",
            ),
            ModelResponse(text="done", stop_reason="end_turn"),
        ]
    )


def reset():
    CALLS["read"] = 0
    CALLS["delete"] = 0


def test_deny_blocks_without_executing():
    reset()
    agent = Agent(
        model=provider_calling_both(),
        tools=[read_data, delete_data],
        policy=Policy(deny=["delete_*"]),
    )
    run = agent.run("clean up")
    assert CALLS["delete"] == 0  # never executed
    assert CALLS["read"] == 1
    blocked = [e for e in run.log if e.kind == "tool:delete_data"][0]
    assert blocked.result.startswith("BLOCKED:")
    assert run.intents() == [
        {"tool": "read_data", "status": "executed", "seq": 1},
        {"tool": "delete_data", "status": "blocked", "seq": 2},
    ]


def test_dry_run_stubs_non_allowlisted():
    reset()
    agent = Agent(
        model=provider_calling_both(),
        tools=[read_data, delete_data],
        policy=Policy(allow=["read_*"], dry_run=True),
    )
    run = agent.run("clean up")
    assert CALLS["read"] == 1  # allowlisted reads still run
    assert CALLS["delete"] == 0  # everything else is stubbed
    statuses = {i["tool"]: i["status"] for i in run.intents()}
    assert statuses == {"read_data": "executed", "delete_data": "stubbed"}


def test_confirm_with_handler_approve_and_reject():
    reset()
    agent = Agent(
        model=provider_calling_both(),
        tools=[read_data, delete_data],
        policy=Policy(confirm=["delete_*"]),
        on_human=lambda q: "yes",
    )
    agent.run("clean up")
    assert CALLS["delete"] == 1  # approved -> executed

    reset()
    agent = Agent(
        model=provider_calling_both(),
        tools=[read_data, delete_data],
        policy=Policy(confirm=["delete_*"]),
        on_human=lambda q: "no",
    )
    run = agent.run("clean up")
    assert CALLS["delete"] == 0  # rejected -> never executed
    assert any(
        e.kind == "tool:delete_data" and "rejected" in e.result for e in run.log
    )


def test_confirm_without_handler_pauses_then_resumes():
    reset()
    agent = Agent(
        model=provider_calling_both(),
        tools=[read_data, delete_data],
        policy=Policy(confirm=["delete_*"]),
    )
    paused = agent.run("clean up")
    assert paused.paused
    assert "delete_data" in paused.pending
    assert CALLS["delete"] == 0

    done = paused.resume("yes, approved")
    assert done.output == "done"
    assert CALLS["delete"] == 1  # executed exactly once, after approval


def test_budget_stops_run_and_proceed_continues():
    responses = [
        ModelResponse(
            tool_calls=[ToolCall("t1", "read_data", {})],
            stop_reason="tool_use",
            usage={"input_tokens": 600, "output_tokens": 0},
        ),
        ModelResponse(
            text="finished", stop_reason="end_turn", usage={"input_tokens": 100, "output_tokens": 5}
        ),
    ]
    reset()
    agent = Agent(
        model=ScriptedProvider(list(responses)),
        tools=[read_data],
        policy=Policy(budget_tokens=500),
    )
    run = agent.run("go")
    assert run.stop_reason == "budget"
    assert run.truncated
    assert run.num_turns == 1  # stopped right after the over-budget model call
    assert CALLS["read"] == 0  # stopped BEFORE executing the requested tool

    # Raise the budget; proceed() replays the prefix and finishes live.
    agent.policy.budget_tokens = 10_000
    agent.provider = ScriptedProvider(responses[1:])  # only the tail runs live
    done = run.proceed()
    assert done.output == "finished"
    assert done.stop_reason == ""
    assert CALLS["read"] == 1  # executed exactly once, during proceed()
