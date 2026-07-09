# Docker sandbox: the proxy is the only door

The Linux/anywhere counterpart of `loom record --sandbox` (macOS). Two
networks; the agent lives on an **internal** one with no route out; the
proxy is dual-homed and is therefore the only path to the world:

```
        ┌────────────── jail (internal) ─────────────┐
        │  agent ── ANTHROPIC_BASE_URL ──▶ proxy ────┼──▶ egress ──▶ api.anthropic.com
        │  (any other destination: unreachable)      │
        └────────────────────────────────────────────┘
```

- Shield rules on the proxy (`--deny`, `--confirm`, `--rule`) are now
  **impossible to bypass** — there is no other wire.
- The trace in `./traces/` is the complete account of the model's traffic.
- Add `--auth <token>` to the proxy command and set the matching
  `x-loom-auth` header if the jail network has tenants you don't trust.

Run it:

```
ANTHROPIC_API_KEY=sk-ant-... docker compose up
```

Swap `your-agent-image` for anything that speaks the Anthropic API (or
OpenAI — give the proxy `--target https://api.openai.com` and set
`OPENAI_BASE_URL` instead). The agent needs no code changes.

The `ci/` directory is this topology under test: loom's CI stands it up on
every push and asserts that the agent's direct egress fails while the same
request through the proxy succeeds and is recorded.
