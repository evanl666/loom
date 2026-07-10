# Loom

[![PyPI](https://img.shields.io/pypi/v/loom-harness)](https://pypi.org/project/loom-harness/)
[![CI](https://github.com/evanl666/loom/actions/workflows/ci.yml/badge.svg)](https://github.com/evanl666/loom/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/loom-harness)](https://pypi.org/project/loom-harness/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

### The black-box recorder, firewall, and step-debugger for AI agents.

Your agent called a tool, touched a file, sent a request — and you have no idea
what it did or why. Loom records every action, lets you **replay it byte-for-byte
for free**, **firewalls dangerous calls before they run**, and **steps through
the run like a debugger** — for any Claude/OpenAI-API agent, from Claude Code to
your own.

```bash
pip install loom-harness          # zero dependencies

loom record claude "fix the failing test" --safe
```
```
recorded 17 steps · 42k tokens → session.loom.json
🛡️  firewall blocked 1 risky call:  Read(".env")
▶  loom replay session.loom.json     # re-run byte-identical, $0, no network
🔬 loom debug  session.loom.json     # step through it, fork any turn live
```

---

## Why Loom

| | |
|---|---|
| 🎥 **Record anything** | Proxy any Claude/OpenAI agent — Claude Code, Codex, Cursor, your own — one command, zero code changes. |
| ⏪ **Replay for free** | Every model + tool call is recorded at one boundary, so replay is **byte-identical and costs $0** — deterministic CI for a stochastic agent. |
| 🔬 **Step-debug it** | An interactive debugger: step forward/back, inspect each step's reasoning, tool code, world-diff, and the exact context the model saw — then **edit a turn and re-run it live**. |
| 🛡️ **Firewall it** | Deny / confirm / approve dangerous tool calls *before they execute* — by name, by capability (`cap:money_movement`), or by sequence (`after Read(.env): deny network`). |
| 🕵️ **Prove exfiltration** | Value-lineage taint shows a secret flowing from a read to an egress — even **base64-encoded or paraphrased**, confirmed by an LLM judge. |
| ↩️ **Undo the world** | Revert the files an agent changed, or snapshot & restore a whole workspace + database — **world-state time travel**, not just a transcript rewind. |

---

## 60-second tour

```bash
# 1. record any agent (or use the Python harness — see below)
loom record claude "add pagination to the users endpoint" --safe

# 2. replay it — byte-identical, zero API cost, offline
loom replay session.loom.json

# 3. step-debug it: walk each action, see the diff + context, fork any turn LIVE
loom debug session.loom.json --agent app:agent

# 4. see where the data went (secret → egress lineage, incl. encoded)
loom taint session.loom.json
loom dlp   session.loom.json --judge claude-haiku-4-5   # semantic DLP

# 5. undo what it did to your files
loom undo session.loom.json
```

## Use it as a harness (Python)

```python
from loom import Agent, tool, Policy

@tool
def search(q: str) -> str:
    "Search the docs."
    return db.search(q)

agent = Agent(
    model="claude-opus-4-8",
    tools=[search],
    policy=Policy(deny=["issue_refund*"], budget_tokens=50_000),  # in-loop firewall
)
run = agent.run("What changed in the API last week?")

run.replay()          # byte-identical, no API calls
run.fork(at=3)        # rewind to turn 3, continue live on a new branch
run.save("run.loom.json")
```

One **effect boundary** records every model and tool call, so replay, fork,
bisect, free CI tests, structured output, human-in-the-loop, subagents, caching,
and journaled crash-recovery all fall out of the same primitive.

---

## The interactive debugger

`loom debug run.loom.json --agent app:agent` opens a step-debugger in your browser:

- **Step** forward / back (`←` `→`), click any action, jump to first/last.
- **Inspect** each step: the model's reasoning, the tool call + arguments, the
  result, the **world-diff** (a file diff for coding, a row diff for SQL, a DOM
  diff for a browser agent), risk, capabilities, firewall decision, and tokens.
- **Context frame:** see the exact conversation the model saw at that step — the
  debugger's "stack & variables."
- **Edit & re-run live:** at any turn, inject a message into the model's context
  or switch the model, hit **Fork & Run** — only the divergent tail costs a call,
  and the new branch appears beside the original with the first divergence marked.
- **Timeline & play:** a scrubber colored by risk and sized by token cost —
  click to jump, or hit ▶ to watch the run animate.
- **Branch compare & walk:** fork three ways, then **diff any two branches**
  side-by-side (winner called on score/tokens) and step through each one.
- **Assert & explain:** check plain-English expectations (`never issue_refund`,
  `output contains …`) against the run, ask the copilot to **explain any step**,
  and drive it all from a **⌘K command palette**.
- **Multi-agent aware:** for a supervisor/sub-agent system — your own, or a
  third-party framework (LangGraph, CrewAI, the Claude Agent SDK) recorded via
  the proxy — Loom **recovers the agent hierarchy from the wire** and shows it
  as a collapsible **tree**, each step laned and colored by which agent ran it.

---

## Guard MCP servers

Loom is a **firewall + black-box recorder for MCP**, too:

```bash
# see what a server can do before you trust it — with a trust score
loom mcp manifest -- npx -y @modelcontextprotocol/server-filesystem .

# re-serve it firewalled: a drop-in guarded endpoint for Claude Desktop / Cursor
loom mcp gateway --deny write_file* --save traffic.loom.json \
  -- npx -y @modelcontextprotocol/server-filesystem .
```

Every `tools/call` is screened by your policy before it reaches the server and
recorded as a replayable, taint-able trace.

---

## Command cheat-sheet

| | |
|---|---|
| `loom record <agent> "<task>"` | record any Claude/OpenAI agent through a proxy |
| `loom replay <trace>` | re-run byte-identical, $0, offline |
| `loom debug <trace> --agent m:a` | **interactive step-debugger** + live fork |
| `loom live --agent m:a` | **run an agent live** in the debugger: watch steps stream, ask follow-ups |
| `loom studio <trace>` | self-contained visual report |
| `loom rootcause` / `loom loops` | the **first bad step** + cascade · repeated/oscillating loops |
| `loom whatif --step N --result X` | **fault injection**: re-run with a tool result overridden |
| `loom experiment "task" --system … --model …` | **A/B** prompts + models, scored & ranked |
| `loom intent <trace> --judge` | **intent firewall**: flag actions that don't serve the request |
| `loom assert <trace> -e "never issue_refund*"` | **behavioural assertions** as a CI gate (the debugger's assert bar) |
| `loom canary run --agent m:a` | **honeytokens**: bait the agent, catch exfiltration |
| `loom taint` / `loom dlp --judge` | exfiltration lineage · **semantic DLP** |
| `loom scan` / `loom sbom` | supply-chain posture · CycloneDX **bill of materials** |
| `loom memory forensics/audit` | catch **memory poisoning** (+ `MemoryFirewall` at runtime) |
| `loom snapshot` / `loom world` | **world-state** time travel · git-style world branches |
| `loom tools --verify` | **trust-but-verify**: declared vs observed capabilities |
| `loom why --causal` | prove an action's cause by **counterfactual fork** |
| `loom autopilot <trace>` | incident → autopsy + movie + policy patch + PR |
| `loom cost --fix` / `--md` | token-burn RCA + patches · PR comment |
| `loom policy rollout / synthesize` | gated canary → enforce · **auto-generate** least-privilege |
| `loom mcp gateway / audit -- <srv>` | firewall + record an MCP server · npm-audit for MCP |
| `RemoteAgent(name, call=…)` | record a **black-box remote (HTTP/gRPC) agent** call as one replayable, firewallable Action |
| `loom shadow` / `loom behavior` | offline policy canary · behavior unit tests |
| `loom fuzz` / `loom dataset from` | hostile-trace CI guard · SFT/DPO/eval data |

Run `loom --help` for the full set.

---

## How it works

Every nondeterministic action an agent takes — a model call, a tool call — flows
through a single **effect boundary**. Record mode logs the result; replay mode
serves it. From that one primitive:

- **replay** is byte-identical and free (no network, no tokens),
- **fork / bisect** rewind to any turn and continue live,
- **CI tests** run a stochastic agent deterministically,
- the **firewall** sits exactly where every tool call must pass,
- and every analyzer — taint, cost, incident, scan — reads the same log.

The kernel is **zero-dependency**. `[anthropic]`, `[openai]`, and `[mcp]` extras
add live providers and the MCP gateway.

## Install

```bash
pip install loom-harness                # kernel + CLI, zero deps
pip install "loom-harness[anthropic]"   # + live Claude
pip install "loom-harness[mcp]"         # + MCP gateway
```

Python 3.10–3.13 · MIT license · `import loom`

## Links

- **Docs & examples:** [`examples/`](examples/) · [`docs/`](docs/)
- **Packs** (coding · SQL · browser · support): [`loom/packs/`](loom/packs/)
- **Threat model:** [`docs/threat-model.md`](docs/threat-model.md) — what Loom
  does and does not stop.

> Loom reduces the blast radius of an agent and makes its behavior inspectable.
> It is not a guarantee that a model can't misbehave — see the threat model.
