# The flight recorder demo

Five acts, fully offline, no API key. One command:

```
bash demo.sh
```

## The story

**Act 1 — the crash.** A deploy bot reads a 14-months-stale `deploy.toml`,
trusts it, health-checks a database host that no longer exists, and aborts
the deploy. Every model call and tool result is recorded into
`flight.loom.json` — the flight recording.

**Act 2 — read the black box.** `loom replay` reproduces the run with zero
API calls. `loom timeline` shows every step. `loom doctor` finds the smoking
gun: one tool result is **94% of the context**, oversized and stale.
`loom studio` opens the interactive viewer.

**Act 3 — the fix, verified.** `loom heal` forks the recorded run, redacts
the suspect context item, and re-runs only the tail: the bot asks service
discovery instead, finds the real host, and the deploy goes **GREEN**. The
winning branch is saved as a golden regression trace.

**Act 4 — agent CI.** Someone opens an "innocent" PR adding one sentence to
the system prompt (`agent_v2.py`). `loom impact` replays the regression
corpus against the new config — offline, free — and fails the check: this
recorded run would behave differently. The bug from Act 1 can't sneak back.

**Act 5 — the firewall.** The same recorder, now with rules: a model that
goes for `/app/.env` gets its tool call rewritten out of the response before
the client ever sees it. The block is part of the recording
(`loom search artifacts 'shield:deny'` finds it later).

## Why this is hard anywhere else

The recording is not a log — it's an **effect trace**: every nondeterministic
step with its inputs hashed and its result stored. That's what makes replay
byte-identical, forks cheap (the prefix replays for free), impact analysis
free (recompute input hashes, never call a model), and the firewall auditable
(decisions live in the same trace). Point the same proxy at a real agent —
`loom record -- claude -p "..."` — and everything in this demo works on it.
