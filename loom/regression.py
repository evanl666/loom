"""``loom regression from``: turn a bad agent run into a test that stays red.

An incident report tells you what went wrong once. A *regression* makes sure
it can't happen again silently. From a single trace this generates everything
a repo needs to guard the behavior:

  traces/<name>.loom.json   the run, SCRUBBED, as a golden fixture
  regression-policy.yml     a firewall policy derived from the risky actions
  regression-cases.json     policy-test cases asserting those calls are gated
  test_<name>.py            a pytest: the fixture stays valid AND the policy
                            still catches the risky calls
  ci.yml                    a GitHub Actions snippet wiring it into review
  README.md                 what this guards and how to run it

"Every agent bug becomes a test." Offline; ``--open-pr`` optionally opens the
PR with the GitHub CLI.
"""

from __future__ import annotations

import json
import os
import re


def _slug(path: str) -> str:
    base = os.path.basename(path)
    base = base[: -len(".loom.json")] if base.endswith(".loom.json") else base
    return re.sub(r"[^A-Za-z0-9_]+", "_", base).strip("_") or "run"


def _risky_calls(data: dict) -> "list[tuple[str, dict, list[str]]]":
    """(tool, input, risk-categories) for every risky tool call in the trace."""
    from .risk import classify_all

    out = []
    seen = set()
    for e in data.get("log", []):
        if e.get("kind") != "model" or not isinstance(e.get("result"), dict):
            continue
        for tc in e["result"].get("tool_calls") or []:
            name, tinput = tc.get("name", ""), tc.get("input", {})
            cats = classify_all(name, tinput)
            key = (name, json.dumps(tinput, sort_keys=True, default=str))
            if cats and key not in seen:
                seen.add(key)
                out.append((name, tinput, cats))
    return out


def build_regression(trace_path: str, outdir: str) -> dict:
    """Generate the regression bundle from a trace. Returns the written paths."""
    from .risk import recommended_rule
    from .scrub import scrub_trace
    from .trace import trace_checksum

    with open(trace_path) as f:
        data = json.load(f)

    slug = _slug(trace_path)
    os.makedirs(os.path.join(outdir, "traces"), exist_ok=True)

    # 1. scrubbed golden fixture
    clean, _found = scrub_trace(data)
    clean["scrubbed"] = True
    if "checksum" in clean:
        clean["checksum"] = trace_checksum(clean)
    fixture_rel = f"traces/{slug}.loom.json"
    fixture_path = os.path.join(outdir, fixture_rel)
    with open(fixture_path, "w") as f:
        json.dump(clean, f, indent=2)

    # 2. a firewall policy that would have gated the risky calls
    risky = _risky_calls(data)
    rules: dict[str, list[str]] = {"deny": [], "confirm": []}
    cases = []
    for name, tinput, cats in risky:
        rule = recommended_rule(cats[0])
        # recommended_rule returns human text; derive a concrete pattern.
        pattern = _pattern_for(name, cats[0])
        action = "deny" if cats[0] in ("secret-read", "fs-destructive") else "confirm"
        if pattern not in rules[action]:
            rules[action].append(pattern)
        cases.append({"name": name, "input": tinput, "expect": action,
                      "why": f"regression: this run's {cats[0]} action must stay gated"})

    policy = {"default": "allow"}
    for k in ("deny", "confirm"):
        if rules[k]:
            policy[k] = rules[k]
    policy_rel = "regression-policy.yml"
    with open(os.path.join(outdir, policy_rel), "w") as f:
        f.write(_policy_yaml(policy))

    cases_rel = "regression-cases.json"
    with open(os.path.join(outdir, cases_rel), "w") as f:
        json.dump(cases, f, indent=2)

    # 3. the pytest
    test_rel = f"test_{slug}.py"
    with open(os.path.join(outdir, test_rel), "w") as f:
        f.write(_test_py(slug, fixture_rel, policy_rel, cases_rel, bool(cases)))

    # 4. CI + README snippets
    with open(os.path.join(outdir, "ci.yml"), "w") as f:
        f.write(_ci_yaml(fixture_rel))
    with open(os.path.join(outdir, "README.md"), "w") as f:
        f.write(_readme(slug, fixture_rel, risky))

    return {"outdir": outdir, "fixture": fixture_rel, "policy": policy_rel,
            "cases": cases_rel, "test": test_rel, "risky": len(risky),
            "files": [fixture_rel, policy_rel, cases_rel, test_rel, "ci.yml", "README.md"]}


def _pattern_for(name: str, category: str) -> str:
    if category == "secret-read":
        return f"{name}(*.env*)"
    if category == "fs-destructive":
        return "Bash(*rm -rf*)"
    if category.startswith("network"):
        return f"{name}*"
    # capability-based for business categories
    capmap = {"money-movement": "cap:money_movement", "pii-access": "cap:pii_access",
              "user-comm": "cap:user_communication", "db-write": "cap:database_write",
              "browser-submit": "cap:browser_submit", "code-exec": f"{name}(*)"}
    return capmap.get(category, f"{name}*")


def _policy_yaml(policy: dict) -> str:
    lines = [f"# Regression policy generated by `loom regression from`.",
             f"default: {policy.get('default', 'allow')}"]
    for section in ("deny", "confirm"):
        if policy.get(section):
            lines.append(f"{section}:")
            lines += [f'  - "{p}"' if ":" in p else f"  - {p}" for p in policy[section]]
    return "\n".join(lines) + "\n"


def _test_py(slug, fixture, policy, cases, has_cases) -> str:
    body = f'''"""Regression guard generated by `loom regression from`.

If a change reintroduces the behavior from this recorded run, this test goes
red. Fixture: {fixture}
"""

import json
import os

from loom.testing import verify_trace

HERE = os.path.dirname(__file__)


def test_fixture_stays_valid():
    problems = verify_trace(os.path.join(HERE, "{fixture}"))
    assert not problems, problems
'''
    if has_cases:
        body += f'''

def test_policy_still_gates_the_risky_calls():
    from loom.policy_file import resolve, to_shield_kwargs
    from loom.shield import Shield

    shield = Shield(**to_shield_kwargs(resolve(policy_path=os.path.join(HERE, "{policy}"))))
    cases = json.load(open(os.path.join(HERE, "{cases}")))
    for c in cases:
        action, _ = shield.classify(c["name"], c.get("input", {{}}))
        assert action == c["expect"], (
            f"{{c['name']}} is no longer {{c['expect']}}ed -- {{c['why']}}")
'''
    return body


def _ci_yaml(fixture) -> str:
    return f'''# Add to .github/workflows/ -- runs the regression guard on every PR.
name: Loom regression
on: [pull_request]
jobs:
  guard:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {{ python-version: "3.12" }}
      - run: pip install loom-harness pytest
      - run: loom test {fixture}          # the fixture stays replayable
      - run: pytest test_*.py             # the policy still gates the risky calls
'''


def _readme(slug, fixture, risky) -> str:
    risks = "\n".join(
        f"- `{name}` — {', '.join(cats)}" for name, _inp, cats in risky) or "- (none flagged)"
    return f'''# Regression guard: {slug}

Generated by `loom regression from` from a recorded run. It guards against a
change silently reintroducing this behavior.

**Risky actions this run took:**
{risks}

**What's here**
- `{fixture}` — the run, scrubbed, as a golden fixture.
- `regression-policy.yml` / `regression-cases.json` — a firewall policy and the
  cases asserting the risky calls stay gated.
- `test_{slug}.py` — a pytest: the fixture stays valid and the policy still
  catches the risky calls.
- `ci.yml` — wire it into pull requests.

**Run it**
```
loom test {fixture}
pytest test_{slug}.py
```
'''


def open_pr(outdir: str, slug: str) -> "tuple[bool, str]":
    """Create a branch + PR with the generated bundle via the GitHub CLI."""
    import subprocess

    branch = f"loom/regression-{slug}"
    def run(*args):
        return subprocess.run(args, cwd=outdir, capture_output=True, text=True)

    if run("git", "rev-parse", "--is-inside-work-tree").returncode != 0:
        return False, "not inside a git repository"
    if run("git", "checkout", "-b", branch).returncode != 0:
        run("git", "checkout", branch)
    run("git", "add", "-A")
    if run("git", "commit", "-m", f"loom regression guard: {slug}").returncode != 0:
        return False, "nothing to commit"
    if run("git", "push", "-u", "origin", branch).returncode != 0:
        return False, "git push failed (check remote/auth)"
    pr = run("gh", "pr", "create", "--fill", "--head", branch)
    if pr.returncode != 0:
        return False, f"gh pr create failed: {pr.stderr.strip()[:120]}"
    return True, pr.stdout.strip()
