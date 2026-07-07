"""Durable runs: crash mid-run, recover, finish -- nothing paid for is lost.

    python examples/09_durable_runs.py

With journal=..., every effect hits disk the moment it is recorded. We simulate
a process crash after the expensive tool ran, then recover: the journaled
prefix replays for free and only the unfinished tail runs live. The expensive
tool executes exactly once across both attempts.
"""

from loom import Agent, Run, tool
from loom.providers import ModelResponse, ScriptedProvider, ToolCall

JOURNAL = "task.journal.jsonl"
CALLS = {"expensive": 0}


@tool
def expensive_migration() -> str:
    "A slow, costly, side-effectful operation."
    CALLS["expensive"] += 1
    return "migrated 1M rows"


def script():
    return [
        ModelResponse(
            text="Starting the migration.",
            tool_calls=[ToolCall("t1", "expensive_migration", {})],
            stop_reason="tool_use",
        ),
        ModelResponse(text="Migration finished: 1M rows moved.", stop_reason="end_turn"),
    ]


class CrashAfter:
    """A provider that dies after N calls -- our simulated power loss."""

    def __init__(self, inner, n):
        self.inner, self.model, self.name, self.calls, self.n = inner, inner.model, "crash", 0, n

    def complete(self, system, messages, tools):
        if self.calls >= self.n:
            raise ConnectionError("simulated crash (power loss / OOM / deploy)")
        self.calls += 1
        return self.inner.complete(system, messages, tools)


print("=" * 64)
print("1. Run with a journal -- and CRASH mid-run")
print("=" * 64)
agent = Agent(
    model=CrashAfter(ScriptedProvider(script()), n=1),
    tools=[expensive_migration],
    journal=JOURNAL,
)
try:
    agent.run("Migrate the database.")
except ConnectionError as e:
    print(f"CRASHED: {e}")
print(f"expensive tool ran {CALLS['expensive']} time(s) before the crash")

print()
print("=" * 64)
print("2. RECOVER from the journal -- prefix replays, tail runs live")
print("=" * 64)
healthy = Agent(model=ScriptedProvider(script()[1:]), tools=[expensive_migration],
                journal=JOURNAL)
run = Run.recover(JOURNAL, agent=healthy)
print("output:", run.output)
print(f"expensive tool total executions: {CALLS['expensive']}  <- exactly once!")

print()
print("=" * 64)
print("3. The recovered run is a normal Run -- trace intact")
print("=" * 64)
run.print_timeline()
