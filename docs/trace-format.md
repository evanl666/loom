# The loom trace format

A trace is one JSON document (conventionally `*.loom.json`): a complete,
self-contained recording of an agent run. Everything loom does — replay,
fork, diff, impact, studio, search — reads this one format. It is designed
to be **git-friendly** (indented, stable key order per writer, meaningful
line diffs) and **forward-readable** (readers ignore unknown fields).

Current version: **2**.

## Top-level fields

| field | type | meaning |
|---|---|---|
| `version` | int | trace format version (see below). Absent ⇒ 1. |
| `checksum` | string | `sha256:<hex>` over the canonical JSON of every other field. Tamper-*evident*: loaders warn on mismatch, never fail. `loom migrate` re-stamps a deliberate edit. |
| `model` | string | model name the agent was configured with |
| `system` | string | system prompt at record time |
| `prompt` | string | first user message (kept for compatibility; prefer `episodes`) |
| `episodes` | list[string] | every user message, in order — the run's script |
| `output` | string | the run's final answer |
| `stop_reason` | string | `""`/`end_turn`, `budget`, `max_turns`, `invalid_output`, ... |
| `truncated` | bool | run ended before a final answer |
| `paused` / `pending` / `pending_depth` | bool / string / int | human-in-the-loop pause state (`Run.resume` continues it) |
| `log` | list[EffectEntry] | **the recording** — see below |
| `healed_by` | string? | name of the repair when the trace was produced by `heal()` |
| `recorded_via` | string? | `"proxy"` for wire recordings (`loom record` / `loom proxy`) |
| `wire` | list? | proxy traces only: the raw API responses, replayable by `loom proxy --replay` |
| `shield_events` | list? | firewall decisions (deny/approve/tainted), each naming the rule and route (`via`) |

## EffectEntry

Every nondeterministic step, in execution order:

| field | type | meaning |
|---|---|---|
| `seq` | int | position in the log, 0-based, dense |
| `kind` | string | `model`, `tool:<name>`, `human`, `memory`, `compact`, `critic`, `sample`, `choose`, `edit`, `time` |
| `key` | string | sha over `[kind, payload]` where payload is everything the effect's executor received. For `model` (v2): `{system, messages, tools}`. Strict replay recomputes and compares it. The literal `"resumed"` marks an answer injected by `Run.resume()` — a sentinel, exempt from verification. |
| `result` | any | the recorded outcome, JSON-shaped. For `model`: `{text, tool_calls, stop_reason, usage}` |
| `depth` | int | subagent nesting (0 = top level; forks rewind at depth 0 only) |

## Version history

| version | change |
|---|---|
| 1 | original format; `model` keys hashed `{system, messages}` only |
| 2 | tool schemas joined the `model` key hash — adding a tool or editing a schema now fails strict replay and shows in `loom impact` |

## Compatibility policy

- **Fields are only added, never repurposed, within a version.** Readers must
  ignore unknown fields; writing loom always emits the current version.
- **The version bumps only when the meaning of existing data changes** (so
  far: key computation). Loading a trace from another version **warns** with
  what to expect — it never fails and never silently lies.
- `loom migrate <trace> --agent module:attr` recomputes harness-trace keys
  under the current semantics (the same-config rule applies: migration needs
  the recording agent). Proxy traces migrate without an agent — their key
  semantics have not changed.
- Large binary artifacts do not belong in traces; keep tool results textual
  and reference files by path/hash. (Automatic externalization of oversized
  results is deliberately not implemented yet — it would touch every reader;
  `loom doctor` flags oversized results instead.)

## Integrity, honestly

The checksum catches accidents (hand-edited fixtures, truncated copies,
merge damage), not adversaries — anyone can edit content *and* re-stamp.
Cryptographic signing (keys, provenance chains) is out of scope for the
file format; put signed traces in signed storage.
