"""Policy-as-code: named safety profiles and ``loom-policy.yml``.

Writing a wall of ``--deny``/``--confirm`` globs on every run doesn't scale.
A policy is a named, versionable document instead -- ship a built-in profile
or point at your own file:

    loom record claude "fix the test" --profile claude-code-safe
    loom record claude "fix the test" --policy loom-policy.yml

A policy resolves to the same Shield the flags build, so everything the
firewall can do (deny/allow/confirm precedence, sequence tripwires, the
default action) is expressible. Files are YAML (or JSON); YAML is read with
a tiny bounded parser so the zero-dependency install still works, and with
PyYAML if it happens to be installed.
"""

from __future__ import annotations

import json

# Built-in profiles. Each is exactly the keyword arguments Shield takes, so a
# profile is just a policy that ships with loom. Kept deliberately small and
# readable -- they are the thing users copy and adapt.
PROFILES: "dict[str, dict]" = {
    "claude-code-safe": {
        "description": "Sane defaults for a coding agent run with permissions skipped: "
                       "reads and test runs flow, network/installs/pushes ask, secrets "
                       "and destructive shell are blocked, egress after a secret read is cut.",
        "default": "confirm",
        "allow": [
            "Read(*)", "Glob(*)", "Grep(*)", "LS(*)", "WebSearch(*)",
            "Bash(*pytest*)", "Bash(*npm test*)", "Bash(*npm run test*)",
            "Bash(*go test*)", "Bash(*cargo test*)", "Bash(*ls *)", "Bash(*git status*)",
            "Bash(*git diff*)", "Edit(*)", "Write(*)",
        ],
        "confirm": [
            "Bash(*curl*)", "Bash(*wget*)", "Bash(*git push*)",
            "Bash(*npm install*)", "Bash(*pip install*)", "WebFetch(*)",
        ],
        "deny": [
            "Read(*.env*)", "Read(*/.ssh/*)", "Read(*/.aws/*)", "Read(*secrets*)",
            "Bash(*rm -rf*)", "Bash(*curl* | *sh*)", "Bash(*:(){*", "Bash(*mkfs*)",
        ],
        "sequence": [
            "after Read(*.env*): deny WebFetch*, deny Bash(*curl*), deny Bash(*wget*)",
            "taint sk-ant-*: confirm *",
            "taint sk-proj-*: confirm *",
        ],
    },
    "ci-safe": {
        "description": "Non-interactive: nothing waits for a human (no confirms), "
                       "read-only plus tests allowed, everything else denied.",
        "default": "deny",
        "allow": [
            "Read(*)", "Glob(*)", "Grep(*)", "LS(*)",
            "Bash(*pytest*)", "Bash(*npm test*)", "Bash(*go test*)",
        ],
        "deny": ["Read(*.env*)", "Read(*/.ssh/*)"],
        "sequence": ["after Read(*.env*): deny *"],
    },
    "prod-data-safe": {
        "description": "For agents near real data: reads ask, any write/delete/egress "
                       "is denied, a credential sighting locks everything to confirm.",
        "default": "confirm",
        "allow": ["Read(*)", "Glob(*)", "Grep(*)", "WebSearch(*)"],
        "deny": [
            "Bash(*rm*)", "Bash(*DROP *)", "Bash(*DELETE *)", "Bash(*curl*)",
            "Bash(*wget*)", "Write(*)", "Edit(*)", "Read(*.env*)", "Read(*/.ssh/*)",
        ],
        "sequence": ["taint sk-*: confirm *", "taint AKIA*: confirm *"],
    },
    "prod-db-safe": {
        "description": "Database work against production: SELECTs flow, anything that "
                       "mutates schema or rows asks, drops/truncates/deletes are blocked.",
        "default": "confirm",
        "allow": ["Read(*)", "Glob(*)", "Grep(*)", "Bash(*SELECT *)", "Bash(*EXPLAIN *)",
                  "Bash(*\\\\d*)", "Bash(*SHOW *)"],
        "confirm": ["Bash(*INSERT *)", "Bash(*UPDATE *)", "Bash(*ALTER *)",
                    "Bash(*CREATE *)", "Bash(*migrate*)"],
        "deny": ["Bash(*DROP *)", "Bash(*TRUNCATE *)", "Bash(*DELETE FROM*)",
                 "Read(*.env*)", "cap:destructive"],
        "sequence": ["taint password*: confirm *", "taint sk-*: confirm *"],
    },
    "github-actions-safe": {
        "description": "Running inside CI: fully non-interactive (nothing waits for a "
                       "human), read/build/test allowed, secrets and egress denied.",
        "default": "deny",
        "allow": ["Read(*)", "Glob(*)", "Grep(*)", "LS(*)", "Write(*)", "Edit(*)",
                  "Bash(*pytest*)", "Bash(*npm test*)", "Bash(*npm run build*)",
                  "Bash(*go test*)", "Bash(*cargo test*)", "Bash(*make *)",
                  "Bash(*git status*)", "Bash(*git diff*)", "Bash(*git log*)"],
        "deny": ["Read(*.env*)", "Read(*/.ssh/*)", "cap:secret",
                 "Bash(*curl*)", "Bash(*wget*)", "WebFetch(*)"],
        "sequence": ["after Read(*secret*): deny *"],
    },
    "k8s-safe": {
        "description": "Cluster operations: get/describe/logs flow, apply/scale ask, "
                       "delete/drain and anything against kube-system is blocked.",
        "default": "confirm",
        "allow": ["Read(*)", "Glob(*)", "Grep(*)", "Bash(*kubectl get*)",
                  "Bash(*kubectl describe*)", "Bash(*kubectl logs*)",
                  "Bash(*kubectl top*)", "Bash(*helm list*)", "Bash(*helm status*)"],
        "confirm": ["Bash(*kubectl apply*)", "Bash(*kubectl scale*)",
                    "Bash(*kubectl rollout*)", "Bash(*helm upgrade*)", "Bash(*helm install*)"],
        "deny": ["Bash(*kubectl delete*)", "Bash(*kubectl drain*)",
                 "Bash(*kube-system*)", "Bash(*kubectl exec*)", "Read(*/.kube/config*)"],
        "sequence": ["taint token*: confirm *"],
    },
    "customer-data-safe": {
        "description": "Near PII: aggregate reads ask, exports/joins to the outside are "
                       "blocked, any credential or a data sighting cuts egress.",
        "default": "confirm",
        "allow": ["Glob(*)", "Grep(*)"],
        "deny": ["cap:network", "Bash(*COPY *)", "Bash(*\\\\copy*)", "Bash(*mysqldump*)",
                 "Bash(*pg_dump*)", "Write(*export*)", "Read(*.env*)"],
        "sequence": ["taint *@*.*: deny cap:network", "taint sk-*: confirm *"],
    },
}

_SHIELD_KEYS = ("default", "allow", "confirm", "deny", "sequence")


def profile_names() -> "list[str]":
    return sorted(PROFILES)


def to_shield_kwargs(doc: dict) -> dict:
    """Extract the Shield-constructor keys from a policy document."""
    kwargs = {k: doc[k] for k in _SHIELD_KEYS if k in doc}
    for listkey in ("allow", "confirm", "deny", "sequence"):
        kwargs.setdefault(listkey, [])
    # A policy file names it `require_approver`; the Shield kwarg is `approvers`.
    if doc.get("require_approver"):
        kwargs["approvers"] = doc["require_approver"]
    if doc.get("break_glass"):
        kwargs["break_glass"] = doc["break_glass"]
    return kwargs


def resolve(profile: str = "", policy_path: str = "") -> dict:
    """Resolve a --profile name and/or a --policy file into one document.

    A file may itself select a profile (``profile: claude-code-safe``) and add
    to it; explicit lists in the file extend the profile's lists, and a
    ``default`` in the file overrides. Returns a policy document (dict).
    """
    doc: dict = {}
    if profile:
        if profile not in PROFILES:
            raise ValueError(
                f"unknown profile {profile!r}; built-in: {', '.join(profile_names())}"
            )
        doc = _clone(PROFILES[profile])

    if policy_path:
        loaded = load_document(policy_path)
        named = loaded.get("profile")
        if named:
            if named not in PROFILES:
                raise ValueError(f"policy file selects unknown profile {named!r}")
            base = _clone(PROFILES[named])
            doc = _merge(base, doc) if doc else base
        doc = _merge(doc, loaded)
    return doc


def load_document(path: str) -> dict:
    """Load a policy file. A top-level ``profiles:`` map is supported (returns
    the sole entry, or requires the file to also name a ``profile:``)."""
    with open(path) as f:
        text = f.read()
    data = _parse(text, path)
    if not isinstance(data, dict):
        raise ValueError(
            f"{path} is not a valid policy (expected a mapping of settings like "
            f"deny/allow/confirm/default, got {type(data).__name__})"
        )
    if "profiles" in data and set(data) <= {"profiles", "profile", "version"}:
        profiles = data["profiles"]
        chosen = data.get("profile")
        if chosen:
            if chosen not in profiles:
                raise ValueError(f"{path}: profile {chosen!r} not in this file")
            return profiles[chosen]
        if len(profiles) == 1:
            return next(iter(profiles.values()))
        raise ValueError(
            f"{path} defines {len(profiles)} profiles; pick one with `profile: <name>` "
            f"at the top level (or --profile on the command line)"
        )
    return data


def lint(doc: dict) -> "list[str]":
    """Catch the misconfigurations that make a policy quietly not work.

    The classic footgun: ``deny: rm -rf`` looks like it blocks ``rm -rf`` but
    actually targets a TOOL NAMED 'rm -rf', which never exists -- so it never
    fires. We flag command-shaped patterns, wildcard-less signatures, rules
    shadowed by a broader deny, and an empty policy.
    """
    warnings: list[str] = []
    kw = to_shield_kwargs(doc)
    all_patterns = [(a, p) for a in ("deny", "allow", "confirm") for p in kw.get(a, [])]
    if not all_patterns and not kw.get("sequence") and doc.get("default", "allow") == "allow":
        warnings.append("policy is empty and defaults to allow -- it blocks nothing")

    for action, p in all_patterns:
        has_sig = "(" in p and p.endswith(")")
        # A pattern with a space but no signature parens targets a tool *named*
        # that string -- almost always a mistake for a Bash command.
        if " " in p and not has_sig:
            cmd = p.strip("*")
            warnings.append(
                f"{action} '{p}': matches a TOOL NAMED '{p}', not a command. "
                f"For a shell command use a signature glob, e.g. `Bash(*{cmd}*)`."
            )
        # A signature with no wildcard only matches that exact argument string.
        elif has_sig:
            inside = p[p.index("(") + 1: -1]
            if inside and "*" not in inside and "?" not in inside:
                warnings.append(
                    f"{action} '{p}': the argument has no wildcard, so it matches only "
                    f"that exact value. Did you mean `{p[:p.index('(')]}(*{inside}*)`?"
                )

    # An allow shadowed by a broader deny never takes effect (deny > allow).
    from fnmatch import fnmatchcase as fnmatch

    for a in kw.get("allow", []):
        for d in kw.get("deny", []):
            if fnmatch(a, d) or a == d:
                warnings.append(f"allow '{a}' is shadowed by deny '{d}' (deny wins)")
    return warnings


def _clone(d: dict) -> dict:
    return json.loads(json.dumps(d))


def _merge(base: dict, extra: dict) -> dict:
    """Extend base's lists with extra's, override scalars."""
    out = _clone(base)
    for k, v in extra.items():
        if isinstance(v, list) and isinstance(out.get(k), list):
            out[k] = out[k] + [x for x in v if x not in out[k]]
        else:
            out[k] = v
    return out


def _parse(text: str, path: str) -> dict:
    try:
        import yaml  # optional; handles the full language when present

        return yaml.safe_load(text) or {}
    except ImportError:
        pass
    if text.lstrip().startswith("{"):
        return json.loads(text)
    return _mini_yaml(text, path)


def _unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] in "\"'" and s[-1] == s[0]:
        return s[1:-1]  # quoted: taken verbatim, may contain '#' or ':'
    # unquoted: an inline ' # comment' ends the value
    hashpos = s.find(" #")
    if hashpos != -1:
        s = s[:hashpos].rstrip()
    return s


def _mini_yaml(text: str, path: str) -> dict:
    """A bounded YAML reader for the policy schema: nested mappings, lists of
    scalar strings, and scalar values. Not a general YAML parser -- it rejects
    what it doesn't understand rather than guessing.

    A key with an empty value opens a block whose kind (list vs mapping) is
    decided by its first child line; ``_Block`` holds it until then.
    """
    root: dict = {}
    stack: "list[tuple[int, object]]" = [(-1, root)]

    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        content = raw.strip()

        while indent <= stack[-1][0]:
            stack.pop()
        container = _materialize(stack[-1][1])

        if content.startswith("- "):
            if not isinstance(container, (list, _Block)):
                raise ValueError(f"{path}: list item outside a list: {content!r}")
            container.append(_unquote(content[2:]))
        elif ":" in content:
            if not isinstance(container, (dict, _Block)):
                raise ValueError(f"{path}: mapping key inside a list: {content!r}")
            key, _, value = content.partition(":")
            key = key.strip()
            if value.strip() == "":
                block = _Block()
                container[key] = block
                stack.append((indent, block))
            else:
                container[key] = _unquote(value)
        else:
            raise ValueError(f"{path}: cannot parse line: {content!r}")

    return _finalize(root)


class _Block:
    """A key's not-yet-typed child block: becomes a list or a dict on first use."""

    def __init__(self):
        self.value: "list | dict | None" = None

    def append(self, item) -> None:
        if self.value is None:
            self.value = []
        self.value.append(item)

    def __setitem__(self, k, v) -> None:
        if self.value is None:
            self.value = {}
        self.value[k] = v


def _materialize(node):
    return node.value if isinstance(node, _Block) and node.value is not None else node


def _finalize(node):
    if isinstance(node, _Block):
        node = node.value if node.value is not None else {}
    if isinstance(node, dict):
        return {k: _finalize(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_finalize(v) for v in node]
    return node


# ---------------------------------------------------------------- simulation

def simulate(shield, paths: "list[str]") -> dict:
    """Replay a corpus of traces through ``shield`` and report the blast radius.

    Returns a structured result (rendered as text/HTML/Markdown by the CLI):
    per-run deny/confirm verdicts, per-rule hit counts with an example, and a
    per-capability breakdown -- the rollout review a security team needs
    before a deny rule goes live.
    """
    import json
    import os

    total_runs = total_calls = 0
    denied_runs: list[dict] = []
    confirm_runs: list[str] = []
    rule_hits: dict = {}          # (action, rule) -> {count, example}
    cap_hits: dict = {}           # capability -> {deny, confirm}
    from .capabilities import capabilities as _caps

    for p in paths:
        try:
            with open(p) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        total_runs += 1
        denies = confirms = 0
        for e in data.get("log", []):
            if e.get("kind") != "model" or not isinstance(e.get("result"), dict):
                continue
            for tc in e["result"].get("tool_calls") or []:
                total_calls += 1
                name, tinput = tc.get("name", ""), tc.get("input", {})
                action, rule = shield.classify(name, tinput)
                if action not in ("deny", "confirm"):
                    continue
                sig = f"{name}({json.dumps(tinput, sort_keys=True, default=str)})"
                hit = rule_hits.setdefault((action, rule or "(policy default)"),
                                           {"count": 0, "example": sig[:90]})
                hit["count"] += 1
                for c in _caps(name, tinput):
                    cap_hits.setdefault(c, {"deny": 0, "confirm": 0})[action] += 1
                if action == "deny":
                    denies += 1
                else:
                    confirms += 1
        completed = (bool(data.get("output")) and not data.get("truncated")
                     and not data.get("paused"))
        if denies:
            denied_runs.append({"path": p, "name": os.path.basename(p),
                                "completed": completed})
        elif confirms:
            confirm_runs.append(p)

    return {
        "runs": total_runs,
        "calls": total_calls,
        "denied": denied_runs,
        "confirm_only": confirm_runs,
        "untouched": total_runs - len(denied_runs) - len(confirm_runs),
        "false_positives": [d for d in denied_runs if d["completed"]],
        "rule_hits": [
            {"action": a, "rule": r, "count": h["count"], "example": h["example"]}
            for (a, r), h in sorted(rule_hits.items(), key=lambda kv: -kv[1]["count"])
        ],
        "capabilities": [
            {"capability": c, "deny": v["deny"], "confirm": v["confirm"]}
            for c, v in sorted(cap_hits.items(), key=lambda kv: -(kv[1]["deny"] + kv[1]["confirm"]))
        ],
    }


def _pct(n: int, total: int) -> str:
    return f"{100 * n // total}%" if total else "0%"


def simulate_text(r: dict) -> str:
    runs = r["runs"]
    lines = [f"simulated policy over {runs} run(s), {r['calls']} tool call(s):", ""]
    lines.append(f"  would DENY in    {len(r['denied']):>4} run(s)  ({_pct(len(r['denied']), runs)})")
    lines.append(f"  would CONFIRM in {len(r['confirm_only']):>4} run(s)  ({_pct(len(r['confirm_only']), runs)})")
    lines.append(f"  untouched        {r['untouched']:>4} run(s)  ({_pct(r['untouched'], runs)})")
    if r["false_positives"]:
        lines.append(f"\n  ⚠️  {len(r['false_positives'])} of the denied runs had completed "
                     "successfully -- candidate false positives:")
        for d in r["false_positives"][:5]:
            lines.append(f"      {d['path']}")
    if r["rule_hits"]:
        lines.append("\n  per-rule hits:")
        for h in r["rule_hits"]:
            mark = "🚫" if h["action"] == "deny" else "⏸️ "
            lines.append(f"    {mark} {h['action']:8} {h['rule']:<42} x{h['count']}  e.g. {h['example']}")
    return "\n".join(lines)


def simulate_markdown(r: dict, title: str = "Loom policy simulation") -> str:
    runs = r["runs"]
    md = [f"### 🛡️ {title}", "",
          f"Simulated over **{runs} run(s)**, {r['calls']} tool call(s).", "",
          "| verdict | runs | share |", "|---|---:|---:|",
          f"| 🚫 would **deny** | {len(r['denied'])} | {_pct(len(r['denied']), runs)} |",
          f"| ⏸️ would **confirm** | {len(r['confirm_only'])} | {_pct(len(r['confirm_only']), runs)} |",
          f"| ✅ untouched | {r['untouched']} | {_pct(r['untouched'], runs)} |"]
    if r["false_positives"]:
        md += ["", f"⚠️ **{len(r['false_positives'])} candidate false positive(s)** — "
               "runs that completed successfully but would now be denied:"]
        md += [f"- `{d['name']}`" for d in r["false_positives"][:5]]
    if r["rule_hits"]:
        md += ["", "<details><summary>Per-rule hits</summary>", "",
               "| rule | action | hits | example |", "|---|---|---:|---|"]
        for h in r["rule_hits"]:
            md.append(f"| `{h['rule']}` | {h['action']} | {h['count']} | `{h['example']}` |")
        md += ["", "</details>"]
    return "\n".join(md)


def simulate_html(r: dict, title: str = "Loom policy simulation") -> str:
    from .lake import _esc
    runs = r["runs"] or 1

    def bar(n, cls):
        return (f'<div class="simrow"><span class="siglabel">{cls}</span>'
                f'<span class="simbar {cls}" style="width:{max(100*n//runs,1)}%"></span>'
                f'<span class="simval">{n} ({_pct(n, r["runs"])})</span></div>')

    fp = ""
    if r["false_positives"]:
        items = "".join(f"<li>{_esc(d['name'])}</li>" for d in r["false_positives"][:10])
        fp = (f'<div class="warn"><b>{len(r["false_positives"])} candidate false '
              f'positive(s)</b> — completed successfully but would be denied:<ul>{items}</ul></div>')
    rule_rows = "".join(
        f'<tr><td><code>{_esc(h["rule"])}</code></td><td>{h["action"]}</td>'
        f'<td class="num">{h["count"]}</td><td><code>{_esc(h["example"])}</code></td></tr>'
        for h in r["rule_hits"])
    cap_rows = "".join(
        f'<tr><td><code>{_esc(c["capability"])}</code></td>'
        f'<td class="num">{c["deny"]}</td><td class="num">{c["confirm"]}</td></tr>'
        for c in r["capabilities"])
    css = """body{font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
    max-width:820px;margin:0 auto;padding:32px;color:#0b0b0b;background:#f9f9f7}
    @media(prefers-color-scheme:dark){body{color:#fff;background:#0d0d0d}
    table,.warn{background:#1a1a19!important;border-color:#2c2c2a!important}}
    h1{font-size:20px}.sub{color:#898781;margin-bottom:20px}
    .simrow{display:grid;grid-template-columns:90px 1fr 130px;gap:10px;align-items:center;margin:6px 0}
    .siglabel{color:#52514e;font-size:13px}.simbar{height:16px;border-radius:0 4px 4px 0;min-width:2px}
    .simbar.deny{background:#e5484d}.simbar.confirm{background:#fab219}.simbar.untouched{background:#1baf7a}
    .simval{text-align:right;font-variant-numeric:tabular-nums}
    .warn{border:1px solid #fab219;border-radius:10px;padding:12px 16px;margin:16px 0;background:#fff}
    table{width:100%;border-collapse:collapse;margin:14px 0;background:#fff;border:1px solid #e1e0d9;border-radius:10px}
    th{text-align:left;font-size:11px;color:#898781;text-transform:uppercase;padding:8px 12px;border-bottom:1px solid #e1e0d9}
    td{padding:7px 12px;border-bottom:1px solid #e1e0d9}td.num{text-align:right;font-variant-numeric:tabular-nums}
    h2{font-size:13px;color:#52514e;margin:22px 0 6px}code{font-size:12px}"""
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(title)}</title><style>{css}</style></head><body>
<h1>🛡️ {_esc(title)}</h1>
<p class="sub">Rollout blast radius over {r['runs']} run(s), {r['calls']} tool call(s).</p>
{bar(len(r['denied']), 'deny')}{bar(len(r['confirm_only']), 'confirm')}{bar(r['untouched'], 'untouched')}
{fp}
<h2>Per-rule hits</h2><table><tr><th>rule</th><th>action</th><th>hits</th><th>example</th></tr>{rule_rows or '<tr><td colspan=4>none</td></tr>'}</table>
<h2>By capability</h2><table><tr><th>capability</th><th>deny</th><th>confirm</th></tr>{cap_rows or '<tr><td colspan=3>none</td></tr>'}</table>
</body></html>"""


def simulate_diff(shield_old, shield_new, paths: "list[str]") -> dict:
    """The rollout diff between two policy versions over one corpus.

    Answers "what changes if I ship the new policy?": which runs become
    newly denied / newly confirmed, which are released (previously denied,
    now clean), and which rules drive the change.
    """
    old = simulate(shield_old, paths)
    new = simulate(shield_new, paths)

    def verdicts(r):
        v = {}
        for d in r["denied"]:
            v[d["path"]] = "deny"
        for p in r["confirm_only"]:
            v[p] = "confirm"
        return v

    vo, vn = verdicts(old), verdicts(new)
    every = sorted(set(vo) | set(vn))
    newly_denied = [p for p in every if vn.get(p) == "deny" and vo.get(p) != "deny"]
    newly_confirmed = [p for p in every
                       if vn.get(p) == "confirm" and vo.get(p) not in ("deny", "confirm")]
    released = [p for p in every if vo.get(p) == "deny" and vn.get(p) != "deny"]

    old_rules = {(h["action"], h["rule"]): h["count"] for h in old["rule_hits"]}
    rule_changes = []
    for h in new["rule_hits"]:
        delta = h["count"] - old_rules.pop((h["action"], h["rule"]), 0)
        if delta:
            rule_changes.append({**h, "delta": delta})
    for (action, rule), count in old_rules.items():  # rules that stopped firing
        rule_changes.append({"action": action, "rule": rule, "count": 0,
                             "example": "", "delta": -count})

    return {
        "runs": new["runs"],
        "newly_denied": newly_denied,
        "newly_confirmed": newly_confirmed,
        "released": released,
        "rule_changes": sorted(rule_changes, key=lambda r: -abs(r["delta"])),
        "old": {"denied": len(old["denied"]), "confirm": len(old["confirm_only"])},
        "new": {"denied": len(new["denied"]), "confirm": len(new["confirm_only"])},
    }


def simulate_diff_text(d: dict) -> str:
    import os

    lines = [f"policy diff over {d['runs']} run(s): "
             f"denied {d['old']['denied']} → {d['new']['denied']}, "
             f"confirmed {d['old']['confirm']} → {d['new']['confirm']}"]
    for label, paths, mark in (("newly DENIED", d["newly_denied"], "🚫"),
                               ("newly confirmed", d["newly_confirmed"], "⏸️"),
                               ("released (no longer denied)", d["released"], "✅")):
        if paths:
            lines.append(f"\n  {mark} {label}: {len(paths)} run(s)")
            lines += [f"      {os.path.basename(p)}" for p in paths[:5]]
    if d["rule_changes"]:
        lines.append("\n  rule-hit changes:")
        for r in d["rule_changes"][:10]:
            sign = "+" if r["delta"] > 0 else ""
            lines.append(f"    {sign}{r['delta']:>3}  {r['action']:8} {r['rule']}")
    if not (d["newly_denied"] or d["newly_confirmed"] or d["released"]):
        lines.append("  no run changes verdict under the new policy")
    return "\n".join(lines)
