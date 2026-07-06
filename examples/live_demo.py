"""Live end-to-end exercise of every Loom feature. Offline, deterministic.

    python examples/live_demo.py
"""

from loom import Agent, Run, tool
from loom.providers import ModelResponse, RuleProvider, ScriptedProvider, ToolCall


def hr(title):
    print("\n" + "=" * 64 + f"\n{title}\n" + "=" * 64)


# ----------------------------------------------------------------------
# A counting wrapper so we can PROVE replay makes zero live model calls.
# ----------------------------------------------------------------------
class Counting:
    def __init__(self, inner):
        self.inner, self.model, self.name, self.calls = inner, inner.model, "counting", 0

    def complete(self, system, messages, tools):
        self.calls += 1
        return self.inner.complete(system, messages, tools)


# ----------------------------------------------------------------------
# PART A — a real multi-turn, tool-using agent
# ----------------------------------------------------------------------
@tool
def lookup_order(order_id: str) -> str:
    "Look up the status of an order by id."
    return f"Order {order_id}: shipped, arriving tomorrow."


provider = Counting(
    ScriptedProvider(
        [
            ModelResponse(
                text="Let me check that order.",
                tool_calls=[ToolCall("c1", "lookup_order", {"order_id": "A123"})],
                stop_reason="tool_use",
                usage={"input_tokens": 42, "output_tokens": 12},
            ),
            ModelResponse(
                text="Your order A123 has shipped and arrives tomorrow.",
                stop_reason="end_turn",
                usage={"input_tokens": 61, "output_tokens": 14},
            ),
        ]
    )
)
agent = Agent(model=provider, tools=[lookup_order], system="You are a support agent.")

hr("1. SEND A QUERY  ->  get a response")
run = agent.run("Where is my order A123?")
print("QUERY :", run.prompt)
print("ANSWER:", run.output)
print("live model calls so far:", provider.calls)

hr("2. TRACE / TIMELINE")
run.print_timeline()

hr("3. COST ACCOUNTING")
print(run.cost())

hr("4. CONTEXT PROVENANCE (where every item came from)")
for p in run.context.provenance():
    print(f"  {p['source']:<16} role={p['role']:<10} tokens={p['tokens']}")

hr("5. REPLAY  (deterministic, ZERO live model calls)")
before = provider.calls
replay = run.replay()
print("replayed answer:", replay.output)
print("identical to original:", replay.output == run.output)
print("extra live model calls during replay:", provider.calls - before, "(expected 0)")

hr("6. SAVE  ->  LOAD  ->  REPLAY from disk")
run.save("support.loom.json")
loaded = Run.load("support.loom.json", agent=agent)
print("loaded turns:", loaded.num_turns, "| output:", loaded.replay().output)


# ----------------------------------------------------------------------
# PART B — fork (change context, new branch) and bisect
# ----------------------------------------------------------------------
def last_user(messages):
    for m in reversed(messages):
        if m["role"] == "user":
            return m["content"].lower()
    return ""


def rewrite_first_user(new_text):
    "Return an edit() that rewrites the first user message at a fork point."

    def edit(ctx):
        ctx.items[0].content = new_text
        ctx.items[0].tokens = max(1, len(new_text) // 4)

    return edit


branch_provider = RuleProvider(
    rules=[
        lambda m: ModelResponse(text="Refund approved. ERROR: no reason on file.", stop_reason="end_turn")
        if "refund" in last_user(m)
        else None,
        lambda m: ModelResponse(text="Happy to help with your order.", stop_reason="end_turn"),
    ]
)
b_agent = Agent(model=branch_provider)

hr("7. FORK  (rewind, edit context, take a different branch)")
orig = b_agent.run("I want a refund please.")
print("original branch:", orig.output)
forked = orig.fork(at=0, edit=rewrite_first_user("Where is my order?"))
print("forked branch  :", forked.output)

hr("8. BISECT  (find the first turn that looks wrong)")
bad = orig.bisect(lambda text: "ERROR" not in text)
print("original run: first bad turn =", bad)
ok = forked.bisect(lambda text: "ERROR" not in text)
print("forked run  : first bad turn =", ok, "(-1 means all clean)")

hr("ALL FEATURES EXERCISED LIVE ✓")
