"""Human-in-the-loop: the human answer is an effect; runs pause and resume."""

from loom import Agent, Run, ask_human
from loom.providers import ModelResponse, ScriptedProvider, ToolCall


def build_provider():
    return ScriptedProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall("h1", "ask_human", {"question": "Approve the refund?"})
                ],
                stop_reason="tool_use",
            ),
            ModelResponse(text="Done, refund processed.", stop_reason="end_turn"),
        ]
    )


def test_run_pauses_without_handler():
    agent = Agent(model=build_provider(), tools=[ask_human()])
    run = agent.run("Refund order A123.")
    assert run.paused
    assert run.pending == "Approve the refund?"
    assert run.output == ""


def test_resume_continues_and_records_answer():
    agent = Agent(model=build_provider(), tools=[ask_human()])
    paused = agent.run("Refund order A123.")
    done = paused.resume("yes, approved")

    assert not done.paused
    assert done.output == "Done, refund processed."
    human = [e for e in done.log if e.kind == "human"]
    assert len(human) == 1
    assert human[0].result == "yes, approved"  # the decision is in the trace

    # The resumed run replays deterministically -- human answer included.
    assert done.replay().output == done.output


def test_paused_run_survives_save_load(tmp_path):
    agent = Agent(model=build_provider(), tools=[ask_human()])
    paused = agent.run("Refund order A123.")
    path = str(tmp_path / "paused.loom.json")
    paused.save(path)

    loaded = Run.load(path, agent=agent)
    assert loaded.paused
    assert loaded.pending == "Approve the refund?"
    done = loaded.resume("approved")
    assert done.output == "Done, refund processed."


def test_on_human_handler_answers_inline():
    seen = []

    def handler(question):
        seen.append(question)
        return "yes"

    agent = Agent(model=build_provider(), tools=[ask_human()], on_human=handler)
    run = agent.run("Refund order A123.")
    assert not run.paused
    assert run.output == "Done, refund processed."
    assert seen == ["Approve the refund?"]
    assert [e.result for e in run.log if e.kind == "human"] == ["yes"]


def test_resume_requires_paused():
    agent = Agent(model=ScriptedProvider([ModelResponse(text="hi")]))
    run = agent.run("hello")
    try:
        run.resume("answer")
        assert False, "expected ValueError"
    except ValueError:
        pass
