"""Trace memory (learn from past runs) and context compaction (long horizons)."""

from loom import Agent, TraceMemory, tool
from loom.providers import ModelResponse, ScriptedProvider, ToolCall


def one_shot(text):
    return ScriptedProvider([ModelResponse(text=text, stop_reason="end_turn")])


# -- memory ------------------------------------------------------------------


def test_recall_ranks_similar_runs(tmp_path):
    mem = TraceMemory(str(tmp_path))
    mem.add(Agent(model=one_shot("Order A123 was refunded.")).run("Refund order A123 please"))
    mem.add(Agent(model=one_shot("It is sunny in Tokyo.")).run("Weather in Tokyo today"))

    hits = mem.recall("How do I refund order A123?")
    assert hits and "refund" in hits[0]["episodes"][0].lower()
    assert "refund" in mem.recall_text("refund order A123").lower()
    assert mem.recall_text("completely unrelated zebra query") == ""


def test_agent_recalls_experience_as_effect(tmp_path):
    mem = TraceMemory(str(tmp_path))
    mem.add(Agent(model=one_shot("Resolved by resetting the router.")).run(
        "Customer internet outage complaint"
    ))

    agent = Agent(model=one_shot("Try resetting the router first."), memory=mem)
    run = agent.run("Another customer internet outage")

    assert [e.kind for e in run.log][0] == "memory"  # recall recorded as an effect
    sources = [p["source"] for p in run.context.provenance()]
    assert "memory" in sources  # experience was injected into context
    # Replay serves the recorded recall -- even if the store changed since.
    assert run.replay().output == run.output


def test_auto_store_accumulates_experience(tmp_path):
    mem = TraceMemory(str(tmp_path), auto_store=True)
    agent = Agent(model=one_shot("Done."), memory=mem)
    agent.run("First task about database migration")
    assert len(mem.recall("database migration")) == 1  # stored automatically


# -- compaction --------------------------------------------------------------

BIG = "orderdata " * 300  # ~750 tokens of tool output


@tool
def fetch_history() -> str:
    "Fetch a large history blob."
    return BIG


def test_compaction_summarizes_and_replays():
    provider = ScriptedProvider(
        [
            ModelResponse(
                tool_calls=[ToolCall("t1", "fetch_history", {})], stop_reason="tool_use"
            ),
            # consumed by the summarizer once the threshold trips:
            ModelResponse(text="SUMMARY: customer has 42 old orders.", stop_reason="end_turn"),
            ModelResponse(text="Final: archived 42 orders.", stop_reason="end_turn"),
        ]
    )
    # compact_keep=1: the huge tool result is old history, not the kept tail.
    agent = Agent(
        model=provider, tools=[fetch_history], compact_after=300, compact_keep=1
    )
    run = agent.run("Archive old orders.")

    kinds = [e.kind for e in run.log]
    assert "compact" in kinds  # compaction happened and is in the trace
    assert run.output == "Final: archived 42 orders."
    # The big blob is gone from context; the pinned summary is present.
    sources = [p["source"] for p in run.context.provenance()]
    assert "compaction" in sources
    assert run.context.total_tokens() < 400

    # Deterministic replay, including the compaction step.
    replayed = run.replay()
    assert replayed.output == run.output
    assert "compaction" in [p["source"] for p in replayed.context.provenance()]
