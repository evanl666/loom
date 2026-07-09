# Loom

[![PyPI](https://img.shields.io/pypi/v/loom-harness)](https://pypi.org/project/loom-harness/)
[![CI](https://github.com/evanl666/loom/actions/workflows/ci.yml/badge.svg)](https://github.com/evanl666/loom/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/loom-harness)](https://pypi.org/project/loom-harness/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**The black box recorder for AI agents.**

Record Claude Code, Codex, LangGraph — any agent that speaks the Anthropic or
OpenAI API — with **one command and zero code changes**. Replay failures with
zero API calls. Diff prompt changes before merging. Turn every agent bug into
a regression test.

```
pip install loom-harness                       # zero dependencies

loom record -- claude -p "fix the failing test"   # record a real Claude Code session
loom studio session.loom.json                     # time-travel through it in your browser
loom proxy --replay session.loom.json             # serve it back -- zero API calls, fake key works
```

![Loom demo: record, read, replay, rewind](docs/demo.gif)

Everything an agent does is visible in its API traffic — so recording that
traffic gives you the whole story, without touching the agent:

| You get | What it means |
|---|---|
| **Replay** | Re-serve any recorded session with **zero API calls** — byte-identical. |
| **Time travel** | `loom studio` — scrub through the run, see the context at every step. |
| **Cost accounting** | Every model call metered: the exact turn your $23.40 went off the rails. |
| **Diff** | Two runs compared at the effect level: where they diverged, and *why*. |
| **Free CI** | Record once; the [GitHub Action](#the-github-action) flags PRs that change agent behavior — no tokens burned. |
| **Firewall** | [Shield](#loom-shield-dont-dangerously-skip-permissions): `--deny 'Read(*.env*)'` blocks a tool call before the agent ever sees it. |
| **Ask the trace** | `loom why trace.json "why did it read my .env?"` — a debugger agent investigates and answers, citing exact steps. |

> The package installs as `loom-harness`, imports as `loom` (like `beautifulsoup4` / `bs4`).

## Record any agent — no migration

`loom record` black-boxes a single session: it starts a recording proxy on a
free port, points `ANTHROPIC_BASE_URL` (or `OPENAI_BASE_URL` with
`--target https://api.openai.com`) at it, runs your command unchanged, and
writes the trace:

```
$ loom record -- claude -p "what does loom/effect.py do?"
loom record: proxying https://api.anthropic.com on http://127.0.0.1:52104
...your agent runs exactly as normal...

recorded 3 step(s), 1841 tokens -> session.loom.json
  replay it:  loom replay session.loom.json
  inspect it: loom studio session.loom.json
```

For anything longer-lived than one command, run the proxy yourself:

```
loom proxy --save session.loom.json
export ANTHROPIC_BASE_URL=http://127.0.0.1:8788      # Claude Code & friends
# or, for OpenAI-API agents:
loom proxy --save session.loom.json --target https://api.openai.com
export OPENAI_BASE_URL=http://127.0.0.1:8788/v1
# ...run your agent exactly as before
```

Verified end-to-end: a real Claude Code session recorded through the proxy
(its internal calls included, every token accounted), then **replayed offline
with a fake API key**. Streaming works both ways — SSE is relayed live while
the trace gets the complete message, and replays synthesize a well-formed
stream for streaming clients.

Tool calls ride in the responses and tool results in the next request, so the
proxy reconstructs a **full loom trace**: `loom timeline`, `loom studio`,
`loom doctor`, cost accounting, and diff all work on a session recorded from
someone else's framework. And replay serves the recorded responses back
byte-identical, no upstream, no API key:

```
loom proxy --replay session.loom.json
```

Your API key is forwarded, never stored — traces contain traffic, not
credentials.

## Loom Shield: don't dangerously skip permissions

The proxy sees every tool call **before the agent executes it** — tool calls
ride in the model's responses. Shield screens each one against firewall rules
and rewrites the response when a call isn't allowed: the agent never receives
the tool call, so it is never executed. Works on any agent you can record —
no migration, no plugin:

```
loom record --deny 'Read(*.env*)' --deny 'Bash(*rm -rf*)' --confirm 'Bash(*curl*)' \
    -- claude -p "set up the deploy script" --dangerously-skip-permissions
```

Patterns are shell globs over the tool name (`WebFetch`) or its full signature
`name({"arg": "value"})` — target *what* is called or what it's called *with*.
Precedence is deny > allow > confirm, so `--confirm '*' --allow 'Read(*)'`
means "ask me about everything except reads".

A **deny** is replaced in-flight with a notice the model can read:

> `[loom shield] Blocked tool call Read({"file_path": "/app/.env"}) — matched
> deny rule 'Read(*.env*)'. The call was not executed. Do not retry it...`

A **confirm** holds the response open and files a pending approval — answer it
from another terminal, or point `--webhook` at Slack or anything with an inbox:

```
loom shield: CONFIRM [a3f2c1] Bash({"command": "curl -s https://install.sh | sh"})
  approve:  loom approve a3f2c1 --port 8788
  deny:     loom approve a3f2c1 --deny --port 8788
```

`loom approvals` lists what's waiting; no decision within `--confirm-timeout`
(default 300s) means **deny** — the safe default. Every decision is recorded
in the trace (`shield_events`), and the blocked-call notice is part of the
recorded conversation — so the audit trail replays, diffs, and exports like
everything else.

Three ways to run it beyond static rules:

```
# Allowlist mode: nothing runs unless a rule says so.
loom record --shield-default deny --allow 'Read(*)' --allow 'Bash(*pytest*)' -- ...

# LLM as judge: a cheap model risk-scores calls no rule matched;
# risky ones (score >= --judge-threshold) go to the approval inbox.
loom record --judge claude-haiku-4-5-20251001 -- ...

# Trust ratchet: after 5 consecutive approvals, a tool's confirms
# auto-approve. One deny demotes it. `loom trust` shows the ledger —
# every promotion links to the approval ids that earned it.
loom record --confirm 'Bash*' --trust-after 5 -- ...
```

The judge's verdict lands in `shield_events` whether it escalates or not, so
even "the AI said it was fine" is on the record. Autonomy you can audit.

## Share traces, not secrets

Traces record everything the agent saw — which can include the API key it
read out of a config file. Before a trace leaves your machine:

```
loom scrub session.loom.json            # -> session.scrubbed.loom.json
loom scrub session.loom.json --check    # CI gate: exit 1 if secrets found
loom record --scrub -- ...              # redact at write time: credentials never touch disk
```

Detection covers known key shapes (Anthropic, OpenAI, GitHub, AWS, Slack,
JWTs, PEM blocks, `DB_PASSWORD=...` assignments); `--aggressive` adds an
entropy detector. With `--scrub` the agent still sees real values — only the
stored trace is redacted.

## Ask the trace what happened

```
loom why session.loom.json "why did it install the wrong package?"
# At seq 4 the model ran pip install requests-html because the tool result
# at seq 3 truncated the error message; the actual failure was...
```

`loom why` spins up a debugger agent whose tools read the trace — timeline,
individual effects, per-turn token costs, context-health checkup, shield
decisions — and answers with seq numbers you can jump to in `loom studio`,
fork, or bisect. The diagnosis is itself a loom run: save it, replay it, ask
why about the why.

## Loom is also a harness

Recording is the front door. Underneath is a full agent framework whose kernel
is a few hundred lines you can read in an afternoon — every nondeterministic
step (model calls, tools, human input, even the clock) flows through a single
**Effect boundary**, so agents *built on* Loom get superpowers traces alone
can't give you: **fork** a run at any turn and take a different branch,
**sweep** counterfactuals in batch, **bisect** to the turn that went wrong,
and **heal** — automatically find and verify the context repair that fixes a
failed run.

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

## Structured output

Give the agent a type; get a validated object back. The schema rides in the
system prompt, and the final answer is parsed **at the Effect boundary** — a
failed parse feeds the error back to the model and retries, and every retry is
an ordinary recorded effect, so validated runs replay deterministically:

```python
from dataclasses import dataclass

@dataclass
class Weather:
    city: str
    temp_c: float
    rain: bool

agent = Agent(model="claude-opus-4-8", output_type=Weather)  # or TypedDict / pydantic
run = agent.run("Weather in Tokyo?")
run.parsed          # Weather(city='Tokyo', temp_c=21.0, rain=False)
run.parsed.temp_c   # a real float, validated -- not a string you hope is a number
```

Retries exhausted (`output_retries`, default 2) sets
`run.stop_reason == "invalid_output"` instead of raising — inspect the trace to
see exactly what the model kept saying.

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

## Loom Studio: visual time travel

`loom studio` opens any saved trace as a self-contained HTML time-travel
viewer — scrub (or press play) to watch the run unfold, see the exact context
the model saw at every step, hover the cost strip to find the expensive turns,
and copy a ready-made fork snippet from any effect. No external assets, safe
to attach to a bug report or email to a teammate:

```
loom studio run.loom.json        # writes run.loom.html and opens it
loom export run.loom.json        # just write the file
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

**What's guaranteed — and what isn't.** The journal is two-phase: an `intent`
line is flushed *before* a side effect executes, the `effect` line lands after
its result is known. So every recorded effect replays exactly once — but a
crash can still land in the window between a tool starting and its result
hitting disk, and no journal on earth can know from the log alone whether that
tool ran. Loom doesn't pretend otherwise: recovery finds the dangling intent
and **raises `UnfinishedEffect`** instead of silently re-executing, telling
you exactly which tool was in flight. You check the outside world, then
`Run.recover(..., on_unfinished="retry")` to accept the re-execution.
Harness-internal effects (model calls, memory recalls, compaction) retry
silently — a repeated model call costs tokens, not correctness. For tools that
are genuinely destructive and non-idempotent (payments, sends, deletes), give
the tool itself an idempotency key; Loom makes the ambiguity *visible and
exactly as wide as one effect*, but only the outside system can close it.

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

And every repair can grow your test suite — pass `regression_dir` and the
healed branch is saved as a golden trace, ready for `loom test` and
`verify_replay`:

```python
healed = run.heal(check, regression_dir="tests/regressions/")
healed.regression_path   # tests/regressions/healed-3fa1b2c4d5.loom.json
```

Every bug becomes a test, automatically.

## Trace memory: agents that learn from their own history

Every run leaves a complete trace — so a directory of traces is recallable
experience. Before a run starts, the most similar past runs (with their
outcomes) are injected into context, recorded as a `"memory"` effect so
replays reproduce exactly what was recalled:

```python
memory = TraceMemory("runs/", auto_store=True)   # completed runs become experience
agent = Agent(model=..., tools=[...], memory=memory)
agent.run("Migrate the staging database.")       # walks in knowing what worked last time
```

## Compaction: long-horizon runs that don't rot

When history outgrows a threshold, it's summarized into one pinned item — and
the summarization is itself a recorded effect, so compacted runs replay
deterministically:

```python
agent = Agent(model=..., compact_after=8000, compact_keep=4)
```

## Self-correction: a critic at the boundary

Give the agent a (cheaper) reviewer. Every final answer is scored as a
recorded `"critic"` effect — a low score rewinds the turn with the critique in
context, and the model tries again. The failed attempt, the verdict, and the
retry are all in the trace: **self-correction you can replay and audit**.

```python
agent = Agent(model="claude-opus-4-8", critic="claude-haiku-4-5", critic_threshold=0.6)
run = agent.run("Capital of France?")
run.print_timeline()
#  [0] model   The capital of France is Lyon.
#  [1] critic  {"score": 0.2, "critique": "Lyon is not the capital."}
#  [2] model   The capital of France is Paris.      <- caught by its own reviewer
#  [3] critic  {"score": 0.95, "critique": "Correct."}
```

And when the answer really matters, deliberate: sample N candidates and let
the critic pick. Samples are `"sample"` effects, not turns — fork and bisect
semantics stay intact:

```python
agent = Agent(model="claude-opus-4-8", critic="claude-haiku-4-5", deliberate=3)
```

Spend compute exactly where you need confidence — and replay the whole
deliberation later for free.

## Skills: the toolbox grows itself

Your trace lake is full of tool sequences that demonstrably worked. Mine them
into **skills** — macro-tools the agent can call in one step next time:

```python
from loom.skills import mine, save

runs = [Run.load(p, agent=agent) for p in glob("runs/*.loom.json")]
skills = mine(runs)          # sequences seen in >= 2 successful runs
skills[0].name               # "skill_geocode_then_forecast"
skills[0].params             # ["city", "coords"]  <- learned by comparing runs

agent2 = Agent(model=..., tools=[*tools, *[s.as_tool(tools) for s in skills]])
```

Parameterization is learned by comparison: argument values that **varied**
across the mined runs become parameters, values that never changed are baked
in. Every skill carries its provenance (`support` = how many recorded runs
prove it) — the agent's habits have receipts.

## The clock is an effect too

```python
agent = Agent(model=..., clock=True)   # the model knows today's date
run = agent.run("What day is it tomorrow?")
run.replay()                           # ...and the replay sees the ORIGINAL date
```

`loom.now()` and `loom.random()` complete the promise: at harness level they
are recorded effects (replays serve the recorded value); inside a tool they
return real values on purpose — a tool either runs live (fresh time is
correct) or not at all (its recorded result already embeds the time it saw).

## Impact: change your prompt without fear

Every team has the same fear: touch the system prompt and something,
somewhere, silently breaks. `loom impact` is snapshot testing for agents —
replay your recorded corpus against the changed configuration and see exactly
which runs are affected and where, **before paying for a single API call**:

```
$ loom impact fixtures/ --agent myproject.agents:support_agent
inputs-differ    fixtures/refund.loom.json (first at seq 0)
    3 effect(s) see different inputs, starting with 'model'
unchanged        fixtures/greeting.loom.json
    every recorded effect gets identical inputs

1 of 2 recorded run(s) affected
```

Dry mode (free) recomputes every effect's input hash under the new config and
reports the first divergence. Add `--live` to re-run affected conversations
and see **how** the outputs change, not just where. Exit code 1 when anything
is affected — drop it straight into CI. Python API: `loom.impact.assess`.

### The GitHub Action

Lock recorded behavior into every PR — the impact report lands as a comment
and the check fails when a prompt/config change touches recorded runs:

```yaml
jobs:
  agent-ci:
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - uses: evanl666/loom@main
        with:
          traces: tests/agent_traces
          agent: myapp.agents:build_agent
```

> ### ❌ Loom: this change affects recorded agent runs
> ```
> inputs-differ    tests/agent_traces/refund.loom.json (first at seq 0)
>     3 effect(s) see different inputs, starting with 'model'
> 1 of 2 recorded run(s) affected
> ```
>
> 💸 this change makes your agent 12.0% more expensive in input tokens
> (~12,380 -> ~13,860 across 2 recorded run(s))

The cost line is a real regression check, still with **zero API calls**: the
action sizes every recorded conversation's model inputs under the PR branch
*and* under the base branch (same ~4 chars/token estimator on both sides, so
the delta is apples-to-apples) and reports the difference. Prompt bloat shows
up in review, priced, before it ships.

Dry mode costs nothing (no API calls). Add `live: 'true'` to also show *how*
outputs change. This repo dogfoods the action on its own demo traces
(`.github/workflows/agent-ci.yml`).

## Agent CI: `loom test` and `loom watch`

```
loom test fixtures/            # verify a suite of saved traces (exit 1 on failure)
loom watch task.jsonl          # follow a running agent's journal live (tail -f)
```

For full behavioral regression in your test suite (zero API calls):

```python
from loom import verify_replay
def test_agent_fixtures():
    for path in glob("fixtures/*.loom.json"):
        verify_replay(path, agent=build_agent())
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
- `AnthropicProvider` — `pip install "loom-harness[anthropic]"`, needs `ANTHROPIC_API_KEY`
- `OpenAIProvider` — `pip install "loom-harness[openai]"`; works with OpenAI **and**
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

## MCP: bring your tool ecosystem

Any [Model Context Protocol](https://modelcontextprotocol.io) server plugs in
as ordinary tools (`pip install "loom-harness[mcp]"`):

```python
from loom.mcp import MCPServer

with MCPServer("npx", ["-y", "@modelcontextprotocol/server-filesystem", "."]) as fs:
    agent = Agent(model="claude-opus-4-8", tools=fs.tools())
    run = agent.run("What's in this directory?")
    run.save("fs.loom.json")
```

Because MCP calls cross the same Effect boundary as everything else, they are
recorded like any tool call — which means **a trace recorded against a live
MCP server replays with the server gone**. Your CI verifies filesystem,
database, or browser-driving agent behavior with zero MCP processes running.

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
loom record -- claude -p "..."                      # black-box a real agent session
loom record --deny 'Read(*.env*)' -- claude ...     # ...with the Shield firewall active
loom proxy --save session.loom.json                 # long-lived recording proxy (or --replay)
loom approvals && loom approve a3f2c1               # the Shield confirm inbox
loom studio trip.loom.json                          # open the time-travel viewer
loom run "What is 2 + 3?" --model claude-opus-4-8   # run an agent
loom timeline trip.loom.json                        # inspect a saved trace
loom replay trip.loom.json                          # replay offline
loom diff yesterday.loom.json today.loom.json       # where + why two runs diverged
loom export trip.loom.json                          # self-contained HTML trace viewer
loom doctor trip.loom.json                          # check a trace for context rot
loom heal run.loom.json --agent app:agent --forbid ERROR --save-regression tests/regressions
loom impact fixtures/ --agent app:agent             # which recorded runs a change affects
loom skills mine runs/ --save skills.json           # crystallize proven tool sequences
loom test fixtures/                                 # verify saved traces (CI)
loom watch task.jsonl                               # follow a running agent's journal
```

## FAQ

**Can I use Loom with my existing Claude Code / LangGraph / CrewAI / OpenAI-SDK agent?**

Yes — that's the front door. `loom record -- <your command>` (or `loom proxy`)
records any agent that speaks the Anthropic or OpenAI API with zero code
changes, and everything that works on a trace works on that recording: replay,
Studio, timeline, diff, doctor, cost accounting.

**Then why would I build on the harness?**

Recording captures what *did* happen; the harness can also run what *didn't*.
Fork, sweep, heal, live rerun, and resume all need Loom to re-execute the
divergent tail of a run — that requires your agent's loop and tools to live
inside the Effect boundary. A recorded trace of someone else's agent has no
loop to hand control back to. Think Git: `git log` works on any history you
import, but branching needs your work to happen in the repo. Migrating is
deliberately cheap — tools are plain decorated functions, the `Agent` is a
thin loop, so porting is usually a dozen lines.

**Do I pay for replays?**

No. Replay serves every model and tool result from the recorded log — zero API
calls, zero tokens. That's also why forks and sweeps are cheap: the shared
prefix replays free and you only pay for the divergent tail.

**Is a trace tied to one vendor?**

The trace format is vendor-neutral JSON (`ModelResponse`, tool results, input
hashes). Providers translate at the edge; the kernel and the traces never
import an SDK.

## Status

`v0.10` — alpha, on PyPI as
[`loom-harness`](https://pypi.org/project/loom-harness/). Recording proxy
(`loom record` / `loom proxy`, Anthropic + OpenAI dialects, streaming both
ways), Shield firewall + approval inbox, Studio time-travel viewer, GitHub
Action, kernel, time-travel
(replay/fork/bisect), sweep, diff, subagents, conversations, human-in-the-loop,
streaming, parallel tools, context-rot checkup/heal, durable runs, policy,
effect cache, trace memory, compaction, structured output, critic/deliberate,
skills, impact analysis, and MCP are complete and tested. See
[Roadmap](#roadmap).

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
- ~~`loom test` & `loom watch`~~ ✅ shipped
- ~~Trace memory + context compaction~~ ✅ shipped
- ~~PyPI release (`pip install loom-harness`)~~ ✅ shipped
- ~~Structured output (`output_type=`, validation-retry at the boundary)~~ ✅ shipped
- ~~Impact analysis (`loom impact` — snapshot testing for config changes)~~ ✅ shipped
- ~~Heal-to-test (`heal(regression_dir=)` — every bug becomes a test)~~ ✅ shipped
- ~~MCP servers as tools (`loom-harness[mcp]`)~~ ✅ shipped
- ~~Clock & randomness as effects (`loom.now`, `loom.random`, `Agent(clock=True)`)~~ ✅ shipped
- ~~Critic gate + deliberate mode (replayable self-correction)~~ ✅ shipped
- ~~Skill crystallization (`loom.skills.mine` — proven sequences become tools)~~ ✅ shipped
- ~~`loom proxy` / `loom record` — record any Anthropic- or OpenAI-API agent (streaming included), replay offline~~ ✅ shipped
- ~~Loom CI GitHub Action — impact reports as PR comments~~ ✅ shipped
- ~~Loom Studio — time-travel debugger UI (`loom studio`)~~ ✅ shipped
- ~~Loom Shield — an agent firewall on the proxy: deny/confirm patterns for any agent, no migration~~ ✅ shipped
- `loom fuzz` — chaos engineering for agents (fault injection at any effect)

## License

MIT
