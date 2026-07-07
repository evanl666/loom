"""Sweep + diff: test five fixes at once, then diff two runs. Offline, no API key.

    python examples/05_sweep_diff.py

Sweep is the batch version of fork: every branch replays the shared prefix for
free and only pays for its divergent tail. Diff pinpoints the first step where
two runs went different ways -- and whether inputs or outputs are to blame.
"""

from loom import Agent
from loom.providers import ModelResponse, RuleProvider


def last_user(messages):
    for m in reversed(messages):
        if m["role"] == "user":
            return m["content"].lower()
    return ""


# A deterministic "model" whose answer depends on the question content.
def rule_refund(messages):
    if "refund" in last_user(messages):
        return ModelResponse(text="Refund approved. ERROR: no reason on file.", stop_reason="end_turn")
    return None


def rule_status(messages):
    if "status" in last_user(messages):
        return ModelResponse(text="Your order shipped yesterday.", stop_reason="end_turn")
    return None


def rule_default(messages):
    return ModelResponse(text="Happy to help with your order.", stop_reason="end_turn")


agent = Agent(model=RuleProvider(rules=[rule_refund, rule_status, rule_default]))


def rewrite(new_text):
    def edit(ctx):
        ctx.items[0].content = new_text

    return edit


print("=" * 64)
print("1. BASE RUN (contains an ERROR)")
print("=" * 64)
run = agent.run("I want a refund please.")
print("output:", run.output)

print()
print("=" * 64)
print("2. SWEEP: test three counterfactuals at once")
print("=" * 64)
sweep = run.sweep(
    at=0,
    variants=[
        None,                                  # control: no edit
        rewrite("What is my order status?"),   # hypothesis 1
        rewrite("Hello!"),                     # hypothesis 2
    ],
    labels=["control", "ask-status", "greet"],
)
sweep.print_compare()

print()
print("=" * 64)
print("3. DIFF: where exactly did a branch diverge, and why?")
print("=" * 64)
branch = dict(iter(sweep))["ask-status"]
print(run.diff(branch).summary())
