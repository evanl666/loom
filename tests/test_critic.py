"""Self-correction: the critic gate and deliberate mode, fully replayable."""

import pytest

from loom import Agent
from loom.providers import ModelResponse, ScriptedProvider


def scripted(*texts):
    return ScriptedProvider([ModelResponse(text=t, stop_reason="end_turn") for t in texts])


# -- critic gate -------------------------------------------------------------


def test_low_score_rewinds_the_turn():
    agent = Agent(
        model=scripted("The capital of France is Lyon.", "The capital of France is Paris."),
        critic=scripted(
            '{"score": 0.2, "critique": "Lyon is not the capital."}',
            '{"score": 0.95, "critique": "Correct."}',
        ),
    )
    run = agent.run("Capital of France?")
    assert run.output == "The capital of France is Paris."
    assert [e.kind for e in run.log] == ["model", "critic", "model", "critic"]
    # The critique the model saw is in context, and the verdicts are recorded.
    assert any(i.source == "critique" for i in run.context.items)
    assert run.log[1].result["score"] == 0.2

    replay = run.replay()  # the whole self-correction replays, zero API calls
    assert replay.output == run.output


def test_good_answer_passes_first_time():
    agent = Agent(
        model=scripted("Paris."),
        critic=scripted('{"score": 0.9, "critique": "Good."}'),
    )
    run = agent.run("Capital of France?")
    assert run.output == "Paris."
    assert [e.kind for e in run.log] == ["model", "critic"]


def test_critic_retries_are_bounded():
    agent = Agent(
        model=scripted("bad v1", "bad v2", "bad v3"),
        critic=scripted(*['{"score": 0.1, "critique": "Still wrong."}'] * 3),
        critic_retries=1,
    )
    run = agent.run("Hard question?")
    # One rewind allowed: attempt, verdict, retry, verdict -- then it ships.
    assert run.output == "bad v2"
    assert [e.kind for e in run.log] == ["model", "critic", "model", "critic"]


def test_unparseable_critic_fails_open():
    agent = Agent(
        model=scripted("Some answer."),
        critic=scripted("I refuse to emit JSON."),
    )
    run = agent.run("Q?")
    assert run.output == "Some answer."  # no retry, answer passes
    assert run.log[-1].result["score"] == 1.0


# -- deliberate --------------------------------------------------------------


def test_deliberate_picks_the_better_candidate():
    agent = Agent(
        model=scripted("Answer A (mediocre).", "Answer B (excellent)."),
        critic=scripted('{"best": 1, "why": "B is more precise."}',
                        '{"score": 0.9, "critique": "Good."}'),
        deliberate=2,
    )
    run = agent.run("Q?")
    assert run.output == "Answer B (excellent)."
    kinds = [e.kind for e in run.log]
    assert kinds == ["model", "sample", "choose", "critic"]
    assert run.num_turns == 1  # samples are not turns; fork semantics intact
    # The chosen text also lands in context for any following episode.
    assert run.context.items[-1].content == "Answer B (excellent)."

    replay = run.replay()
    assert replay.output == run.output
    assert replay.num_turns == 1


def test_deliberate_requires_a_critic():
    with pytest.raises(ValueError, match="needs a critic"):
        Agent(model=scripted("x"), deliberate=2)


def test_deliberate_keeps_first_on_junk_choice():
    agent = Agent(
        model=scripted("Answer A.", "Answer B."),
        critic=scripted("no json here", '{"score": 1.0, "critique": "ok"}'),
        deliberate=2,
    )
    run = agent.run("Q?")
    assert run.output == "Answer A."