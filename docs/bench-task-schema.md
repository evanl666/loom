# The loom bench task format

A bench task defines one prompt and what counts as success, so the same task
can run through several agents and be scored the same way. YAML or JSON (same
reader as policy files).

```
loom bench task.yaml \
    --agent "claude:claude -p {prompt}" \
    --agent "codex:codex exec {prompt}" \
    --profile claude-code-safe \
    --reset git --studio
```

## Fields

| field | type | meaning |
|---|---|---|
| `prompt` | string (required) | the task given to each agent |
| `success` | map | how to judge a run passed (see below) |

### `success` — pick one

| key | type | passes when |
|---|---|---|
| `contains` | string | the agent's final output contains this substring |
| `absent` | string | the final output does **not** contain this |
| `command` | string | this shell command exits 0 after the agent finishes (run in the agent's workspace) — e.g. `pytest -q` |

Omit `success` entirely to score "ran to completion" only.

```yaml
prompt: "Fix the failing tests in this repository."
success:
  command: "pytest -q"     # the real oracle: did the tests go green?
```

## Agents and isolation

`--agent name:command` — `{prompt}` is substituted into the command, or the
prompt is appended as the last argument. The API dialect is inferred per agent
(`codex`/`openai` → OpenAI, else the `--target` default), so a mixed
Claude-vs-Codex comparison just works.

`--reset` isolates agents so one's file edits don't pollute the next — the
difference between a demo and a credible benchmark:

| mode | what it does |
|---|---|
| `none` | (default) all agents share the working directory |
| `git` | hard-reset to HEAD + `git clean -fd` between agents (refuses a dirty tree unless `--force`) |
| `copy` | each agent runs in its own copy of the repo |

## Output

A table scoring each agent on pass / tokens / steps / tools / blocked, with
the cheapest passing one called out. Every cell is backed by a replayable
trace in `--outdir`; `--studio` also exports each to HTML so you can open the
loser and see exactly where it went wrong.
