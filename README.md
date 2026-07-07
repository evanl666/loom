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

## Conversations

`run.ask()` continues a conversation with full context — as one growing trace.
The recorded history replays for free; only the new exchange runs live.

```python
run1 = agent.run("Where is order A123?")
run2 = run1.ask("Can I get a refund?")     # knows about A123
run3 = run2.ask("How long will it take?")  # knows everything so far

run3.print_timeline()   # the whole conversation, one trace
run3.replay()           # replays end-to-end, zero API calls
run3.fork(at=1, ...)    # rewind the conversation itself: "what if the user had asked X?"
```

## Human-in-the-loop

A human's answer is nondeterminism like any other — so Loom records it as an
effect. Add the built-in `ask_human()` tool and you get pausable, auditable
approval flows with no extra machinery:

```python
from loom import Agent, Run, ask_human

agent = Agent(model=..., tools=[ask_human()])
run = agent.run("Refund $500 on order A123.")

run.paused              # True -- the agent asked for approval
run.pending             # "Approve $500 refund for A123?"
run.save("pending.loom.json")               # answer it tomorrow

loaded = Run.load("pending.loom.json", agent=agent)
done = loaded.resume("yes, approved")       # continues from exactly where it paused
done.replay()           # the human decision is in the trace -- fully auditable
```

For interactive use, pass a handler instead: `Agent(..., on_human=input)`.

## Streaming, parallel tools, async

```python
# Stream tokens as they arrive (recorded effect is still the full response;
# replays return instantly without re-streaming).
provider = AnthropicProvider("claude-opus-4-8", on_token=print)

# Run one turn's tool calls concurrently (opt-in). Results are recorded in
# call order, so the trace stays deterministic and replayable.
agent = Agent(model=..., tools=[fetch_a, fetch_b], parallel_tools=True)

# Embed in async apps (FastAPI etc.).
run = await agent.arun("...")
```

## Visual traces

`loom export` renders any saved trace to a single self-contained HTML page —
no external assets, safe to attach to a bug report or email to a teammate:

```
loom export run.loom.json        # writes run.loom.html
```

## Policy: control the agent before it acts

Every tool call flows through one chokepoint, so one policy gates them all:

```python
agent = Agent(model=..., tools=[...], policy=Policy(
    allow=["read_*", "search_*"],    # run freely
    confirm=["delete_*", "send_*"],  # pause for human approval (reuses resume())
    deny=["drop_db"],                # blocked outright, never executed
    budget_tokens=50_000,            # hard spend cap; run stops resumably
))

run = agent.run("clean up old data")
run.intents()      # [{"tool": "delete_orders", "status": "blocked"}, ...]
run.proceed()      # continue a budget-stopped run after raising the cap
```

`Policy(dry_run=True)` stubs every non-allowlisted tool with a
"would call ..." marker — audit what an agent *would* do before granting real
access. Approvals are recorded human effects, so approved runs replay
deterministically and every decision is auditable in the trace.

## Effect cache: iterate without paying twice

```python
cache = EffectCache("dev-cache.jsonl")     # persistent (or EffectCache() in-memory)
agent = Agent(model=..., cache=cache)
agent.run("same prompt")    # pays for the model call
agent.run("same prompt")    # zero API calls -- served by input hash
```

Only `model` effects are cached by default (tools have side effects); opt in
with `kinds=("model", "tool:*")`.

## Model A/B: rerun and diff

```python
run_b = run.rerun(model="claude-haiku-4-5")   # same conversation, same tools
print(run.diff(run_b).summary())              # where and why the models diverged
```

## Durable runs (crash recovery)

With a journal, every effect hits disk the moment it's recorded — one JSON
line per effect, flushed immediately. If the process dies mid-run (crash,
kill, deploy), nothing you paid for is lost:

```python
agent = Agent(model=..., tools=[...], journal="task.jsonl")
agent.run("Migrate the database.")     # 💥 process dies at turn 17

# later, any process:
run = Run.recover("task.jsonl", agent=agent)
```

The journaled prefix replays for free; only the unfinished tail runs live.
Model calls and tool side effects that already happened are **never
re-executed** — the same exactly-once guarantee replay gives, extended across
process death. Recovery is idempotent: recovering a finished run just replays
it. A torn final line (crash mid-write) is detected and ignored.

## Context-rot detection — and self-healing

Context rot (stale, bloated, unused context) is the leading cause of agent
failures. Loom can diagnose it after the fact — and *test the repairs*:

```python
report = run.checkup()
print(report.summary())
# 2 finding(s) in 688 tokens of context:
#   [high] oversized: tool:fetch result is 675 tokens (98% of context)
#   [warn] unused: tool:fetch result never referenced by any later answer

healed = run.heal(check=lambda text: "ERROR" not in text)
healed.output      # "The answer is 42."     <- fixed
healed.healed_by   # "redact-oversized-0"    <- and it names the culprit
```

`heal()` is the loop nobody else can run: **checkup** flags suspects →
each one becomes a **fork** that redacts it → only the divergent tail re-runs
→ the first branch that passes your check wins. Diagnosis to *verified* fix,
automatically. Also available for any saved trace: `loom doctor run.loom.json`.

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
loom export trip.loom.json                          # self-contained HTML trace viewer
loom doctor trip.loom.json                          # check a trace for context rot
```

## FAQ

**Is Loom a harness or a debugging plugin?**

A harness — you build your agent *on* Loom, and the debugging superpowers come
built in. They can't be bolted onto another framework: replay/fork/sweep work
because *every* nondeterministic step flows through the Effect boundary and gets
recorded. An agent built elsewhere never passed through that chokepoint, so
there is nothing to replay. Think Git, not a browser extension: Git can diff
and bisect your history because your commits live in it from day one.

**Can I use Loom to debug my existing LangGraph / CrewAI / OpenAI-SDK agent?**

Not in place — but migrating is deliberately cheap. Loom's `Agent` is a thin
loop and tools are plain decorated functions, so porting an agent is usually a
dozen lines: bring your system prompt, re-declare each tool with `@tool`, pick
a provider. From then on every run is recorded, replayable, and diffable.

**Do I pay for replays?**

No. Replay serves every model and tool result from the recorded log — zero API
calls, zero tokens. That's also why forks and sweeps are cheap: the shared
prefix replays free and you only pay for the divergent tail.

**Is a trace tied to one vendor?**

The trace format is vendor-neutral JSON (`ModelResponse`, tool results, input
hashes). Providers translate at the edge; the kernel and the traces never
import an SDK.

## Status

`v0.5` — alpha. Kernel, time-travel (replay/fork/bisect), sweep, diff,
subagents, conversations, human-in-the-loop, streaming, parallel tools, HTML
export, context-rot checkup/heal, and durable runs are complete and tested.
See [Roadmap](#roadmap).

### Roadmap
- ~~Subagents (isolated context, nested traces)~~ ✅ shipped
- ~~OpenAI-compatible provider~~ ✅ shipped
- ~~Sweep (batch counterfactual forks)~~ ✅ shipped
- ~~Trace diff (`loom diff`)~~ ✅ shipped
- ~~Conversations (`run.ask`)~~ ✅ shipped
- ~~Human-in-the-loop as an effect (pause / resume)~~ ✅ shipped
- ~~Streaming, parallel tools, `arun`~~ ✅ shipped
- ~~HTML trace export~~ ✅ shipped
- ~~Context-rot checkup + self-healing (`run.heal`)~~ ✅ shipped
- ~~Durable runs (write-ahead journal + `Run.recover`)~~ ✅ shipped
- ~~Policy at the boundary (deny/confirm/dry-run/budget) + `intents()`~~ ✅ shipped
- ~~Effect-level caching~~ ✅ shipped
- ~~Model A/B (`run.rerun`) + edits persisted as effects~~ ✅ shipped
- `loom test` (fixture regression suite) & `loom watch` (live journal tail)
- Trace memory (agents that learn from their own history) & context compaction
- PyPI release

## License

MIT
