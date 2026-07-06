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

Ships with `ScriptedProvider` and `RuleProvider` (offline, no deps) and
`AnthropicProvider` (optional). OpenAI-compatible and local providers are ~30
lines each.

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
```

## Status

`v0.1` — alpha. The kernel and time-travel are complete and tested. See
[ROADMAP](#roadmap).

### Roadmap
- ~~Subagents (isolated context, nested traces)~~ ✅ shipped
- Streaming
- OpenAI-compatible provider
- Provenance-based context-rot warnings
- `loom bisect` binary search over turns

## License

MIT
