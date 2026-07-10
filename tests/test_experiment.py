"""Experiment A/B: run a task under N variants, scored + ranked."""

from loom import Agent, tool
from loom.experiment import contains_check, run_experiment
from loom.providers import ModelResponse


@tool
def noop() -> str:
    "noop"
    return "ok"


class _Fixed:
    """Stateless provider (reusable across variants) that always answers the same."""
    model = "fixed"

    def complete(self, system, messages, tools):
        return ModelResponse(text="done: REFUNDED", stop_reason="end_turn")


def test_experiment_runs_and_ranks_variants():
    agent = Agent(model=_Fixed(), tools=[noop], system="base")
    results = run_experiment(agent, "do it",
                             systems=["careful prompt", "fast prompt"],
                             check=contains_check("REFUNDED"))
    assert len(results) == 2
    assert all(r["success"] for r in results)  # both outputs contain REFUNDED
    assert all("REFUNDED" in r["output"] for r in results)
    # cross-product with models
    m = run_experiment(agent, "do it", systems=["a", "b"], models=None)
    assert len(m) == 2


def test_experiment_check_marks_failure():
    agent = Agent(model=_Fixed(), tools=[noop])
    results = run_experiment(agent, "do it", check=contains_check("NOT_PRESENT"))
    assert results[0]["success"] is False
