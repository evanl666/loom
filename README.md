# 🧵 Loom

[![PyPI](https://img.shields.io/pypi/v/loom-harness)](https://pypi.org/project/loom-harness/)
[![CI](https://github.com/evanl666/loom/actions/workflows/ci.yml/badge.svg)](https://github.com/evanl666/loom/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/loom-harness)](https://pypi.org/project/loom-harness/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

<p align="right"><b>English</b> · <a href="README.zh-CN.md">中文</a></p>

### The black box, firewall & debugger for AI agents.

Your agent ran — touched files, called tools, spent tokens — and you have no idea
what it did or why. **Loom records every action, replays it byte-for-byte for $0,
firewalls dangerous calls before they run, and lets you step through the whole run
like a debugger.** Works with any Claude/OpenAI-API agent — Claude Code, LangGraph,
CrewAI, your own.

```bash
pip install loom-harness          # zero dependencies
loom record claude "fix the failing test" --safe
```
```
recorded 17 steps · 42k tokens → session.loom.json
🛡  firewall blocked 1 risky call:  Read(".env")
🔬 loom debug session.loom.json   # step through it, fork any turn live
```

---

## Why Loom

- 🎥 **Record any agent** — proxy Claude Code / Codex / Cursor / your own, one command, zero code changes.
- ⏪ **Replay for $0** — every call recorded at one boundary → **byte-identical, offline**. Deterministic CI for a stochastic agent.
- 🔬 **Step-debug it** — walk each step, see the *exact context the model saw*, then **edit a turn and re-run it live**.
- 🕸 **Any multi-agent framework** — LangGraph · CrewAI · AutoGen · OpenAI-Agents · Claude-SDK, recovered into one **agent tree** from the wire, zero code changes.
- 🛡 **Firewall it** — deny / confirm dangerous calls *before they run*, by capability (`cap:money_movement`) or sequence (`after Read(.env): deny network`).
- 🕵 **Catch exfiltration** — a secret flowing to an egress, even **base64-encoded or paraphrased**, confirmed by an LLM judge.
- ↩ **Undo the world** — revert the files an agent changed, or snapshot & restore a whole workspace + database.

---

## The debugger

`loom debug run.loom.json` (or `loom live` to watch it run) opens a step-debugger in your browser:

- **Step** through every action — the model's reasoning, the tool call + args, the world-diff (file / SQL row / DOM), risk, tokens.
- **Context frame** — the exact conversation the model saw at each step: the debugger's *stack & variables*.
- **Fork & re-run live** — inject a message or switch the model at any turn; only the divergent tail costs a call, and the branch appears beside the original.
- **Multi-agent tree** — a supervisor/sub-agent system (yours or a third-party framework) recovered from the wire and shown as a collapsible tree, laned by agent.
- **Ask & assert** — send the live agent a new message, or check plain-English expectations (`never issue_refund`, `output contains …`) as a CI gate.

`loom studio <trace>` freezes the whole UI into **one shareable HTML file** (no server, no agent).

---

## Debug a live agent

```bash
loom live --agent app:agent        # watch it run, send follow-ups, fork any turn
```

Behind a **gRPC / HTTP endpoint**? Point your server at the recording proxy and drive it from the same debugger — no code, just your `grpcurl`:

```bash
loom live --proxy-port 9000 \
  --trigger 'grpcurl -d "{\"prompt\": $LOOM_PROMPT_JSON}" -plaintext :50051 agent.Agent/Run'
# then start your server with ANTHROPIC_BASE_URL=http://127.0.0.1:9000
```

Loom reconstructs the agent's **full internal hierarchy** even though it's behind an endpoint.

---

## Use it as a Python harness

```python
from loom import Agent, tool, Policy

@tool
def search(q: str) -> str:
    "Search the docs."
    return db.search(q)

agent = Agent(model="claude-opus-4-8", tools=[search],
              policy=Policy(deny=["issue_refund*"], budget_tokens=50_000))  # in-loop firewall
run = agent.run("What changed in the API last week?")

run.replay()        # byte-identical, no API calls
run.fork(at=3)      # rewind to turn 3, continue live on a new branch
```

One **effect boundary** records every model + tool call — so replay, fork, free CI,
human-in-the-loop, the firewall, and every analyzer fall out of the same primitive.
The kernel is **zero-dependency**.

---

## A few more commands

| | |
|---|---|
| `loom replay <trace>` | re-run byte-identical, $0, offline |
| `loom taint` · `loom dlp --judge` | exfiltration lineage · semantic DLP |
| `loom redteam run --generate <m>` | AI red-teamer — invents attacks for *your* tool surface |
| `loom mcp gateway -- <server>` | firewall + record any MCP server |
| `loom undo <trace>` | revert the files the agent changed |

Run `loom --help` for the full set — record, replay, debug, live, studio, firewall,
taint, dlp, redteam, mcp, undo, cost, rootcause, experiment, and more.

---

## Install

```bash
pip install loom-harness                # kernel + CLI, zero deps
pip install "loom-harness[anthropic]"   # + live Claude
pip install "loom-harness[mcp]"         # + MCP gateway
```

Python 3.10–3.13 · MIT · `import loom`

> Loom shrinks an agent's blast radius and makes its behavior inspectable — it is
> **not** a guarantee a model can't misbehave. See the [threat model](docs/threat-model.md).
