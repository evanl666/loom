# The loom policy format

A policy is the firewall as a versionable document, instead of a wall of
`--deny`/`--confirm` flags. It applies with `--policy loom-policy.yml` or by
name with `--profile <name>`; either resolves to the same `Shield`.

Files are YAML or JSON. YAML is read by a small **bounded** parser (nested
maps, scalar lists, scalars — the policy schema, nothing more) so the
zero-dependency install works; `pip install "loom-harness[yaml]"` uses PyYAML
for the full language.

## Fields

| field | type | meaning |
|---|---|---|
| `default` | `allow` \| `confirm` \| `deny` | action when no rule matches (default `allow`) |
| `allow` | list[glob] | tool calls that flow through untouched |
| `confirm` | list[glob] | tool calls held for human approval |
| `deny` | list[glob] | tool calls blocked before the agent sees them |
| `sequence` | list[rule] | temporal tripwires (see below) |
| `profile` | string | (files only) start from this built-in profile, then extend |
| `profiles` | map | (files only) define named profiles; select with `profile:` |

Precedence is **deny > allow > confirm** — an `allow` rule bypasses `confirm`,
so `confirm: ['*']` + `allow: ['Read(*)']` means "ask about everything except
reads". A `default` other than `allow` makes the policy an allowlist.

### Patterns

Shell globs matched against the tool **name** (`WebFetch`) or its full
**signature** `name({"arg": "value"})` — so a rule can target *what* is called
or what it's called *with*. Whitespace in signatures is normalized, so
`Bash(*rm -rf*)` matches `rm   -rf`.

> **Footgun** (caught by `loom policy lint`): `deny: rm -rf` targets a *tool
> named* `rm -rf`, which never exists — it never fires. You want
> `Bash(*rm -rf*)`. Likewise `Read(.env)` matches only that exact argument;
> you want `Read(*.env*)`.

### Sequence rules

```
after <call-glob>: <action> <glob>, <action> <glob>, ...
taint <text-glob>: <action> <glob>, ...
```

`after` arms when a matching call is allowed through; `taint` arms when a tool
*result* matches. Once armed, the consequences apply to every later call in
the session. A tripped `deny` beats a static `allow`; a tripped `confirm`
never auto-approves via the trust ratchet.

## Testing a policy (policy-as-code in CI)

`loom policy test cases.json --policy loom-policy.yml` classifies a list of
calls against the policy and exits non-zero on any mismatch:

```json
[
  {"name": "Read", "input": {"file_path": ".env"}, "expect": "deny",
   "why": "never read env files"},
  {"name": "Bash", "input": {"command": "pytest -q"}, "expect": "allow"}
]
```

`loom policy lint` catches misconfigurations before they ship;
`loom policy explain session.loom.json --policy ...` shows what the policy
would do to a recorded run.

## Built-in profiles

`claude-code-safe`, `ci-safe` (deny-by-default, no prompts), `prod-data-safe`.
`loom policy init <name>` scaffolds an editable file from one.
