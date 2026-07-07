"""Context-rot detection (checkup) and the self-healing loop (heal)."""

from loom import Agent, tool
from loom.providers import ModelResponse, RuleProvider, ScriptedProvider, ToolCall

POISON = "POISONMARKER junkdata noisepayload " * 60  # ~500 tokens of rot


# -- checkup ---------------------------------------------------------------


def run_with_tool_result(result_text, answer_text):
    @tool
    def fetch() -> str:
        "Fetch some data."
        return result_text

    provider = ScriptedProvider(
        [
            ModelResponse(
                tool_calls=[ToolCall("t1", "fetch", {})], stop_reason="tool_use"
            ),
            ModelResponse(text=answer_text, stop_reason="end_turn"),
        ]
    )
    return Agent(model=provider, tools=[fetch]).run("go")


def test_checkup_flags_oversized_tool_result():
    run = run_with_tool_result(POISON, "short answer")
    report = run.checkup()
    kinds = [f.kind for f in report.findings]
    assert "oversized" in kinds


def test_checkup_flags_unused_tool_result():
    run = run_with_tool_result(
        "zephyrblue quantumfrog anomalous telemetry payload", "The weather is fine."
    )
    report = run.checkup()
    assert any(f.kind == "unused" for f in report.findings)


def test_checkup_passes_used_small_result():
    run = run_with_tool_result(
        "order A123 shipped yesterday evening", "Your order A123 shipped yesterday."
    )
    report = run.checkup()
    # Small and referenced ("shipped", "yesterday" appear in the answer) -> clean.
    assert report.ok, report.summary()


def test_checkup_flags_duplicates():
    dup = "identical bulky knowledge chunk repeated verbatim in context"

    @tool
    def fetch(n: int) -> str:
        "Fetch chunk n."
        return dup

    provider = ScriptedProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall("t1", "fetch", {"n": 1}),
                    ToolCall("t2", "fetch", {"n": 2}),
                ],
                stop_reason="tool_use",
            ),
            ModelResponse(text="identical bulky chunk noted verbatim", stop_reason="end_turn"),
        ]
    )
    run = Agent(model=provider, tools=[fetch]).run("go")
    assert any(f.kind == "duplicate" for f in run.checkup().findings)


def test_experiments_dedupe_same_item():
    # POISON is both oversized and unused -> one experiment, not two.
    run = run_with_tool_result(POISON, "unrelated answer text entirely")
    labels, variants = run.checkup().experiments()
    assert len(labels) == len(variants) == 1


# -- heal ------------------------------------------------------------------


def build_poisoned_agent():
    """An agent whose final answer is corrupted whenever POISON is in context."""

    @tool
    def fetch_context() -> str:
        "Fetch background data."
        return POISON

    def wants_tool(messages):
        if not any(m["role"] == "tool" for m in messages):
            return ModelResponse(
                tool_calls=[ToolCall("t1", "fetch_context", {})], stop_reason="tool_use"
            )
        return None

    def poisoned(messages):
        if any("POISONMARKER" in str(m.get("content", "")) for m in messages):
            return ModelResponse(text="ERROR: reasoning corrupted by junk", stop_reason="end_turn")
        return None

    def clean(messages):
        return ModelResponse(text="Clean answer: 42", stop_reason="end_turn")

    return Agent(
        model=RuleProvider(rules=[wants_tool, poisoned, clean]),
        tools=[fetch_context],
    )


def test_heal_fixes_a_poisoned_run():
    run = build_poisoned_agent().run("What is the answer?")
    assert "ERROR" in run.output  # the run is genuinely broken

    healed = run.heal(check=lambda text: "ERROR" not in text)
    assert healed is not None
    assert healed.output == "Clean answer: 42"
    assert healed.healed_by.startswith("redact-")  # checkup named the culprit
    # The original timeline is untouched.
    assert "ERROR" in run.output


def test_heal_returns_self_when_already_passing():
    agent = Agent(model=ScriptedProvider([ModelResponse(text="fine", stop_reason="end_turn")]))
    run = agent.run("hi")
    assert run.heal(check=lambda t: "fine" in t) is run


def test_heal_returns_none_when_nothing_works():
    run = build_poisoned_agent().run("What is the answer?")
    # An impossible check: no experiment can satisfy it.
    assert run.heal(check=lambda t: "unicorn" in t) is None


# -- heal-to-test ------------------------------------------------------------


def test_heal_saves_regression_trace(tmp_path):
    from loom import Run
    from loom.testing import verify_replay, verify_trace

    run = build_poisoned_agent().run("What is the answer?")
    check = lambda text: "ERROR" not in text  # noqa: E731

    healed = run.heal(check, regression_dir=str(tmp_path))
    assert healed is not None
    assert healed.regression_path is not None
    assert healed.regression_path.endswith(".loom.json")

    # The saved golden trace is structurally sound and replays byte-identically
    # against a fresh agent -- a ready-made regression test.
    assert verify_trace(healed.regression_path) == []
    verify_replay(healed.regression_path, agent=build_poisoned_agent())

    # The repair provenance survives the round-trip.
    loaded = Run.load(healed.regression_path, agent=build_poisoned_agent())
    assert loaded.healed_by == healed.healed_by
    assert loaded.output == "Clean answer: 42"


def test_heal_regression_save_is_idempotent(tmp_path):
    run = build_poisoned_agent().run("What is the answer?")
    check = lambda text: "ERROR" not in text  # noqa: E731

    first = run.heal(check, regression_dir=str(tmp_path))
    second = run.heal(check, regression_dir=str(tmp_path))
    # Same repair -> same content-addressed file, no duplicate accumulation.
    assert first.regression_path == second.regression_path
    assert len(list(tmp_path.glob("*.loom.json"))) == 1


def test_heal_already_passing_saves_nothing(tmp_path):
    from loom.providers import ModelResponse, ScriptedProvider

    agent = Agent(model=ScriptedProvider([ModelResponse(text="fine", stop_reason="end_turn")]))
    run = agent.run("hi")
    assert run.heal(check=lambda t: "fine" in t, regression_dir=str(tmp_path)) is run
    assert list(tmp_path.glob("*.loom.json")) == []
