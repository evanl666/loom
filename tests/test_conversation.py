"""Conversations: run.ask() continues with full context, in one growing trace."""

from loom import Agent, Run
from loom.providers import ModelResponse, RuleProvider


def _users(messages):
    return [m["content"] for m in messages if m["role"] == "user"]


def build_agent():
    # The refund rule needs the ORDER ID from an EARLIER episode -- so a correct
    # answer proves that context carried across ask().
    def refund(messages):
        users = _users(messages)
        if "refund" in users[-1].lower():
            past = " ".join(users[:-1])
            if "A123" in past:
                return ModelResponse(text="Refund for order A123 initiated.", stop_reason="end_turn")
            return ModelResponse(text="Which order do you mean?", stop_reason="end_turn")
        return None

    def status(messages):
        if "A123" in _users(messages)[-1]:
            return ModelResponse(text="Order A123 shipped yesterday.", stop_reason="end_turn")
        return None

    def default(messages):
        return ModelResponse(text="How can I help?", stop_reason="end_turn")

    return Agent(model=RuleProvider(rules=[refund, status, default]))


def test_ask_carries_context_across_episodes():
    agent = build_agent()
    run1 = agent.run("Where is order A123?")
    assert "shipped" in run1.output

    run2 = run1.ask("I want a refund.")
    assert run2.output == "Refund for order A123 initiated."  # knows A123 from run1
    assert run2.episodes == ["Where is order A123?", "I want a refund."]
    assert run2.num_turns == 2  # both episodes live in one trace


def test_fresh_run_lacks_that_context():
    agent = build_agent()
    cold = agent.run("I want a refund.")
    assert cold.output == "Which order do you mean?"  # no prior episode -> no A123


def test_conversation_replays_and_forks():
    agent = build_agent()
    convo = agent.run("Where is order A123?").ask("I want a refund.")

    assert convo.replay().output == convo.output  # whole conversation replays free

    # Fork at the second episode's turn: rewrite the follow-up question.
    def rewrite(ctx):
        ctx.items[-1].content = "Where is order A123 again?"

    branch = convo.fork(at=1, edit=rewrite)
    assert "shipped" in branch.output
    assert branch.output != convo.output


def test_conversation_save_load_roundtrip(tmp_path):
    agent = build_agent()
    convo = agent.run("Where is order A123?").ask("I want a refund.")
    path = str(tmp_path / "convo.loom.json")
    convo.save(path)

    loaded = Run.load(path, agent=agent)
    assert loaded.episodes == convo.episodes
    assert loaded.replay().output == convo.output
