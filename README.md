# Loom

**The agent harness you can read, replay, and rewind.**

Every other agent framework asks you to trust a black box. Loom's entire kernel
is a few hundred lines you can read in an afternoon — and because every
nondeterministic step flows through a single **Effect boundary**, any agent run
becomes reproducible, forkable, and debuggable.

One primitive. Five superpowers:

| Superpower | What it means |
|---|---|
| **Replay** | Re-run any recorded run with **zero API calls** — identical output. |
| **Fork** | Rewind to any turn, edit the context, and take a different branch. |
| **Bisect** | Walk the recorded turns to find exactly where a run went wrong. |
| **Free CI tests** | Record once; replay in CI forever without burning tokens. |
| **Cost accounting** | Every model call is metered at the boundary. |

```
pip install loom-agent            # zero dependencies
pip install "loom-agent[anthropic]"   # + live Claude models
```

## Quickstart (works offline, no API key)

```python
from loom import Agent, tool
from loom.providers import ModelResponse, ScriptedProvider, ToolCall

@tool
def add(a: int, b: int) -> int:
    "Add two numbers."
    return a + b

# A deterministic offline "model" so the example runs with no key.
provider = ScriptedProvider([
    ModelResponse(tool_calls=[ToolCall("t1", "add", {"a": 2, "b": 3})], stop_reason="tool_use"),
    ModelResponse(text="The answer is 5.", stop_reason="end_turn"),
])

agent = Agent(model=provider, tools=[add])
run = agent.run("What is 2 + 3?")

print(run.output)          # -> The answer is 5.
run.print_timeline()       # step-by-step trace
```

## Use a real model

```python
from loom import Agent, tool

@tool
def get_weather(city: str) -> str:
    "Get the current weather for a city."
    return f"It's sunny in {city}."

agent = Agent(model="claude-opus-4-8", tools=[get_weather])  # needs ANTHROPIC_API_KEY
run = agent.run("What's the weather in Tokyo?")
print(run.output)
```

## Time travel

```python
run = agent.run("Plan a 3-day trip to Rome.")

# Save the trace (git-friendly JSON) and replay it later for free.
run.save("trip.loom.json")
replay = run.replay()                 # zero API calls, identical output

# Rewind to turn 1, change the context, take a different branch.
branch = run.fork(at=1, edit=lambda ctx: ctx.add_user("Actually, make it Paris."))

# Find the first turn whose output looks wrong.
bad_turn = run.bisect(lambda text: "error" not in text.lower())
```

## Sweep: cheap counterfactuals

`sweep` is the batch version of `fork`: test N hypotheses from the same rewind
point in one call. Every branch replays the shared prefix **for free** — you
only pay for each divergent tail. Ten variants of a 20-turn run forked at turn
18 cost 10×2 turns, not 10×20.

```python
sweep = run.sweep(at=3, variants=[
    None,                                   # control (no edit)
    lambda ctx: ctx.items.pop(2),           # hypothesis: drop the stale item
    lambda ctx: setattr(ctx, "budget", 2000),  # hypothesis: tighten the budget
], labels=["control", "drop-stale", "tight-budget"])

sweep.print_compare()
#   base         turns=5  live_tokens=0     diverged_at=-  ...ERROR...
#   control      turns=5  live_tokens=812   diverged_at=-  ...ERROR...
#   drop-stale   turns=4  live_tokens=655   diverged_at=6  The answer is 42.   <- fixed!
#   tight-budget turns=5  live_tokens=790   diverged_at=6  ...ERROR...
```

## Diff: "it worked yesterday"

`loom diff` compares two runs **at the effect level** and tells you not just
*where* they diverged but *why* — because every recorded step carries a hash of
its inputs:

- `kinds-differ` — control flow diverged (a different action was taken)
- `inputs-differ` — same action, but the context drifted
- `results-differ` — same action, same inputs, different outcome

```python
d = run.diff(other_run)
print(d.summary())
# identical prefix: 5 step(s)
# first divergence:
#   step 5 [inputs-differ]
#     a model: calls search({"q": "order status"})
#     b model: I don't have access to orders.
```

Record a fixture suite, re-run against a new model or prompt, diff — that's
regression testing for agents.

## Why the Effect boundary?

The kernel routes **every** model call, tool call, and side effect through one
function, `Recorder.run(...)`. In record mode it executes and logs the result;
in replay mode it returns the logged result without executing. That single
chokepoint is the whole trick — replay, fork, bisect, and cost metering all fall
out of it for free. Read [`loom/effect.py`](loom/effect.py) — it's ~120 lines.

## Bring your own model

A provider is anything with one method:

```python
class MyProvider:
    name = "mine"
    model = "my-model"
    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> ModelResponse:
        ...
```

Ships with:

- `ScriptedProvider`, `RuleProvider` — offline, no deps (used in all examples)
- `AnthropicProvider` — `pip install "loom-agent[anthropic]"`, needs `ANTHROPIC_API_KEY`
- `OpenAIProvider` — `pip install "loom-agent[openai]"`; works with OpenAI **and**
  any OpenAI-compatible server via `base_url` (vLLM, Ollama, LM Studio, Together,
  Groq, OpenRouter, …):

```python
from loom import Agent
from loom.providers import OpenAIProvider

# OpenAI
agent = Agent(provider=OpenAIProvider("gpt-4o"))
# A local model (Ollama / vLLM) — same code, different base_url
agent = Agent(provider=OpenAIProvider("llama3.1", base_url="http://localhost:11434/v1", api_key="x"))
```

## Subagents

Any agent can be exposed as a tool for another agent. The child runs with its own
**isolated context**, and its steps **nest into the same trace** — so replay,
fork, and bisect keep working across delegation.

```python
researcher = Agent(model=..., tools=[search], name="researcher")
lead = Agent(model=..., tools=[researcher.as_tool()])

run = lead.run("Summarize the latest on X.")
run.print_timeline()      # the researcher's turns show up indented under the lead
run.replay()              # deterministic through the delegation, zero API calls
```

The parent only ever sees the delegated *result*, not the child's internal steps
— context stays clean. See [`examples/04_subagents.py`](examples/04_subagents.py).

## CLI

```
loom run "What is 2 + 3?" --model claude-opus-4-8   # run an agent
loom timeline trip.loom.json                        # inspect a saved trace
loom replay trip.loom.json                          # replay offline
loom diff yesterday.loom.json today.loom.json       # where + why two runs diverged
```

## Status

`v0.1` — alpha. The kernel and time-travel are complete and tested. See
[ROADMAP](#roadmap).

### Roadmap
- ~~Subagents (isolated context, nested traces)~~ ✅ shipped
- ~~OpenAI-compatible provider~~ ✅ shipped
- ~~Sweep (batch counterfactual forks)~~ ✅ shipped
- ~~Trace diff (`loom diff`)~~ ✅ shipped
- Streaming
- Provenance-based context-rot warnings
- Human-in-the-loop as an effect (pause / approve / resume)

## License

MIT
