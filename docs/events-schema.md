# Loom event schema (observability export)

`loom export --jsonl` emits one JSON object per effect; `--otel` wraps each as
an OpenTelemetry-style log record. These field names are a **stable contract** —
they change only with a documented version bump, so a Datadog/Splunk/Grafana
pipeline built on them keeps working.

## JSONL event (flat)

Every event carries:

| field | type | meaning |
|---|---|---|
| `run` | string | stable id for the run (hash of checksum + filename) |
| `kind` | string | `model`, `tool:<name>`, `shield`, or a harness kind |
| `seq` | int? | effect position in the trace (absent on `shield` events) |
| `model` | string | the model the run used |
| `prompt` | string | first user message, truncated to 120 chars |

Model events add:

| field | type | meaning |
|---|---|---|
| `input_tokens` / `output_tokens` | int | usage for this call |
| `tool_calls` | list[string] | tool names the model requested |

Tool events (`kind: "tool:<name>"`) add:

| field | type | meaning |
|---|---|---|
| `tool` | string | the tool name |
| `capabilities` | list[string] | inferred capability contract (read/write/exec/network/secret/destructive/idempotent) |
| `error` | bool | the tool returned an `ERROR:` result |
| `blocked` | bool | the tool returned a `BLOCKED:` result |

Shield events (`kind: "shield"`) add:

| field | type | meaning |
|---|---|---|
| `tool` | string | the tool the decision was about |
| `action` | string | `deny` / `approve` / `tainted` |
| `rule` | string | the rule that decided |
| `via` | string | `rule` / `sequence` / `operator` / `timeout` / `judge` / `ratchet` / `default` |
| `by` | string? | operator identity, when a human decided |
| `risk` | list[string] | risk categories of the call |

## OTel log record (`--otel`)

Each event becomes:

```json
{
  "resource": {"service.name": "loom-agent", "loom.run": "<run id>"},
  "name": "loom.<kind>",
  "attributes": { "loom.seq": 0, "loom.tokens.input": 10, ... }
}
```

Attribute keys are namespaced `loom.*`; token usage follows the semantic
convention shape `loom.tokens.input` / `loom.tokens.output`.

## Versioning

This schema is **v1**. Fields are only ever added within a version; a rename
or a meaning change bumps the version and is noted here. Consumers should
ignore unknown fields.
