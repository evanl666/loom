# Loom threat model — what it stops, what it can't

Security tooling earns trust by naming its limits. This is the honest map.

## The pieces

| Layer | Mechanism | Guards against |
|---|---|---|
| **Recorder (proxy)** | every model exchange flows through a local reverse proxy | "what did the agent actually do?" — evidence, replay, audit |
| **Shield (firewall)** | screens tool calls **in the model's response, before the client executes them** | dangerous calls the model *asks for* |
| **Sequence rules / taint** | temporal tripwires ("after a secret read, deny egress") | multi-step incident chains |
| **Approvals** | confirm-holds, approver policy, chains, HMAC-signed decisions | unauthorized/unaudited high-risk actions |
| **Sandbox / container** | OS-level network (macOS) or fs+network (Docker) confinement | the *process* bypassing the proxy |
| **Scrub / share / taint** | secret detection at persist time; value lineage | traces leaking what the agent saw |

## What Loom stops

1. **A tool call the model requests** that matches a deny rule — the client never sees it; the block is recorded.
2. **Egress after a sensitive read** (sequence rules), including value-verbatim exfiltration (`loom taint` proves the chain).
3. **Unaudited approval of high-risk actions** — approver policy limits who may approve; chains need N identities; signatures make the record tamper-proof; break-glass is possible but loudly flagged.
4. **The agent process bypassing the proxy** — *only* when sandboxed (`--sandbox` macOS network confinement, `--container` Docker). The proxy alone is a camera, not a wall.
5. **Traces leaking secrets** — `--scrub` redacts at persist; `loom share` refuses to emit if secrets survive.

## What Loom does NOT stop

1. **Prompt injection changing the agent's goals.** Shield gates *actions*, not thoughts. A hijacked agent that acts within policy stays within policy. Mitigation: tight capability policies + sequence rules narrow what a hijacked agent can do.
2. **Exfiltration via the reply text.** The model can paraphrase a secret into its ANSWER; the proxy sees it (recorded evidence) but the human reading the reply carries it out of scope. `loom taint` catches verbatim propagation only, and says so.
3. **Un-sandboxed process behavior.** Without `--sandbox`/`--container`, the agent process can open sockets and write files directly — the recording is then an account of API traffic, not a guarantee of confinement.
4. **Filesystem damage under `--sandbox`** (macOS mode confines the *network* only). Use `--container` for fs isolation; even then, mounted paths are writable by design.
5. **A malicious tool implementation.** Loom classifies calls by name/arguments/declared capabilities. A tool named `read_notes` that secretly sends email defeats name-based rules — declare capabilities honestly, certify packs (`loom packs lint`), and sandbox.
6. **Approval spoofing without keys.** Identities (`--as NAME`) are labels unless you enable HMAC signing (`--sign-approvals-key-env`) and verify (`loom trace verify-approvals`). OS-level identity/OIDC is a deployment concern.
7. **A compromised host.** Loom runs at user level; a root attacker owns the recorder too.

## Recommended deployment modes

- **Local development:** `loom record claude "..." --safe` — profile + scrub + report. Threats: agent misbehavior, accidental secrets in traces.
- **Untrusted/high-risk task:** add `--sandbox` (macOS) or `--container` — the proxy becomes the only door; sequence rules cut egress after secret contact.
- **Production business agents (support/data/payments):** capability policies (`cap:money_movement` etc.), approver chains + signed decisions, `loom serve` on a trusted network, retention with legal hold, `loom alert` on the fleet.
- **CI:** `ci-safe`/`github-actions-safe` profiles (non-interactive), `loom policy simulate --fail-on-deny`, the Action's `fail-on-new-risk` + `diagnose`.

## Reporting

Found a hole in these claims? Open a GitHub issue (or a private security advisory on the repo) — claims this document makes are treated as bugs when wrong.
