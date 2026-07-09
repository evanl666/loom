# The Action schema — Loom's stable semantic layer

The Action schema is the domain-neutral vocabulary the Action Debugger, the
event export, and the packs all speak. A coding agent, a browser agent, and a
SQL agent are described by the same five types:

| Type | What it answers |
|---|---|
| `Action` | one thing the agent did — reasoned, called a tool, answered |
| `Observation` | what came back (result text, error flag, token usage) |
| `StateDiff` | how the **outside world** changed (files, rows, DOM, fields) |
| `PolicyDecision` | what the firewall decided — allow / deny / confirm, and why |
| `ReplayPoint` | a handle to replay or fork from this step |

```python
from loom import actions            # or: run.actions()
for a in actions(trace_dict):
    a.type          # "reason" | "call" | "answer" | "ask-human" | "meta"
    a.tool          # tool name for calls
    a.intent        # WHY: the model text that requested this action
    a.capabilities  # ["exec", "database_write", ...]
    a.risk          # top risk category ("secret-read", "money-movement", ...)
    a.observation   # Observation(text, error, tokens)
    a.state_diff    # StateDiff(kind, summary, detail) -- filled by a pack
    a.policy        # PolicyDecision(action, rule, via, by)
    a.replay        # ReplayPoint(step, turn, forkable)
```

## Capability vocabulary

Infrastructure: `read` `write` `exec` `network` `secret` `destructive`
`idempotent`. Business: `pii_access` `database_write` `browser_submit`
`user_communication` `money_movement` `external_side_effect`.

Firewall rules match them with `cap:` patterns — `--confirm
'cap:money_movement'` gates every refund tool regardless of its name.

## OTel / JSONL event fields (stable)

`loom export <trace|dir> --jsonl -` emits one flat JSON event per effect,
per firewall decision, and — the semantic layer — per **Action**
(`"kind": "action"`). `--otel` wraps the same events as OTel-style log
records with namespaced attributes.

**Stability policy: attribute names are only ever ADDED, never renamed or
removed**, so a dashboard keyed on them does not break across Loom versions.

| Flat key (JSONL) | OTel attribute | Meaning |
|---|---|---|
| `action_type` | `loom.action.type` | reason / call / answer / ask-human / meta |
| `tool` | `loom.tool` | tool name |
| `capabilities` | `loom.capability` | capability list |
| `risk` | `loom.risk` | top risk category |
| `policy_action` | `loom.policy.action` | allow / approve / deny |
| `policy_rule` | `loom.policy.rule` | the rule that fired |
| `policy_via` | `loom.policy.via` | rule / sequence / judge / operator |
| `state_diff_kind` | `loom.state_diff.kind` | file / database / dom / record / field |
| `state_diff_summary` | `loom.state_diff.summary` | one-line world change |
| `model` | `loom.agent.id` | the agent identity (model id) |
| `input_tokens` / `output_tokens` | `loom.tokens.input` / `.output` | usage |
| `run` | `loom.run` (resource) | stable run id |
| `seq` | `loom.seq` | step within the run |

Effect-level events (`kind: model / tool:* / shield`) keep their existing
fields (documented in `events-schema.md`); action events add the semantic
layer on top rather than replacing them.

## Packs fill the StateDiff

The base builder computes everything derivable from the trace itself; only a
domain pack knows how to read its own world. Built-ins: `coding` (file diffs,
git undo), `sql`, `browser`, `support` — see `loom/packs/`. Register your own
with `loom.packs.register(...)`; the most recently registered pack wins when
several could claim an action.
