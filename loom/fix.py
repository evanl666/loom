"""``loom fix from``: a bad run becomes a fix PR, not just a test.

``loom regression from`` guards against recurrence; ``loom fix from`` goes the
rest of the way -- it bundles the guard WITH the diagnosis and the concrete
fix, ready to review as one PR:

  everything from `loom regression from`   (scrubbed fixture, policy, pytest, CI)
  FIX.md          root cause, the fix by category, how it was verified
  pr-body.md      a paste-ready (or --open-pr'd) PR description

    loom fix from failed.loom.json
    loom fix from failed.loom.json --open-pr
"""

from __future__ import annotations

import json
import os


def _prompt_patch(d: dict) -> str:
    """A suggested prompt/config patch for failure modes that live in config."""
    return {
        "max-steps": ("Add to the system prompt: \"If you find yourself repeating an "
                      "action, stop and produce your best final answer.\" And/or raise "
                      "Agent(max_steps=...)."),
        "budget-stop": "Raise Agent(budget=...) or set Agent(compact_after=...) to trim context.",
        "invalid-output": ("Show the model ONE example of the exact output JSON in the "
                           "system prompt; raise output_retries if the schema is strict."),
        "context-rot": ("Set Agent(compact_after=...) so oversized history is summarized "
                        "before it degrades attention."),
    }.get(d["failure"], "")


def build_fix(trace_path: str, outdir: str) -> dict:
    """Generate the full fix bundle. Returns paths + the diagnosis."""
    from .diagnose import diagnose
    from .regression import build_regression

    with open(trace_path) as f:
        data = json.load(f)
    d = diagnose(data, os.path.basename(trace_path))
    result = build_regression(trace_path, outdir)

    patch = _prompt_patch(d)
    fix_lines = [
        f"# Fix: {d['failure']}", "",
        f"**Severity:** {d['severity']}  ·  **Fix category:** {d['fix_category']}", "",
        "## Root cause", d["diagnosis"], "",
    ]
    if d["symptoms"]:
        fix_lines += ["**Symptoms**", *[f"- {s}" for s in d["symptoms"]], ""]
    fix_lines += ["## The fix", f"- {d['suggestion']}"]
    if result["risky"]:
        fix_lines.append(f"- firewall policy added (`{result['policy']}`): the "
                         f"{result['risky']} risky action(s) this run took stay gated")
    if patch:
        fix_lines.append(f"- prompt/config patch: {patch}")
    fix_lines += ["", "## Verify",
                  *[f"- `{c}`" for c in d["verify"]],
                  f"- `pytest {result['test']}`  (the regression guard stays green)", ""]
    fix_md = "\n".join(fix_lines)
    with open(os.path.join(outdir, "FIX.md"), "w") as f:
        f.write(fix_md)

    pr_body = "\n".join([
        f"## 🤖 Agent fix: {d['failure']} ({d['severity']})", "",
        f"**Root cause** ({d['fix_category']}): {d['diagnosis']}", "",
        f"**Fix:** {d['suggestion']}"
        + (f"\n**Prompt/config patch:** {patch}" if patch else ""), "",
        "**This PR adds the regression guard** -- if a change reintroduces the "
        "behavior, the generated test goes red:",
        f"- `{result['fixture']}` (scrubbed golden fixture)",
        f"- `{result['policy']}` + `{result['cases']}` (the risky calls stay gated)",
        f"- `{result['test']}` + `ci.yml`", "",
        "**Verified with:**",
        *[f"- `{c}`" for c in d["verify"]],
    ])
    with open(os.path.join(outdir, "pr-body.md"), "w") as f:
        f.write(pr_body + "\n")

    result["diagnosis"] = d
    result["files"] = result["files"] + ["FIX.md", "pr-body.md"]
    return result


def open_fix_pr(outdir: str, slug: str) -> "tuple[bool, str]":
    """Branch + commit + PR with the fix bundle, PR body from pr-body.md."""
    import subprocess

    branch = f"loom/fix-{slug}"

    def run(*args):
        return subprocess.run(args, cwd=outdir, capture_output=True, text=True)

    if run("git", "rev-parse", "--is-inside-work-tree").returncode != 0:
        return False, "not inside a git repository"
    if run("git", "checkout", "-b", branch).returncode != 0:
        run("git", "checkout", branch)
    run("git", "add", "-A")
    if run("git", "commit", "-m", f"loom fix: {slug}").returncode != 0:
        return False, "nothing to commit"
    if run("git", "push", "-u", "origin", branch).returncode != 0:
        return False, "git push failed (check remote/auth)"
    pr = run("gh", "pr", "create", "--title", f"Agent fix: {slug}",
             "--body-file", "pr-body.md", "--head", branch)
    if pr.returncode != 0:
        return False, f"gh pr create failed: {pr.stderr.strip()[:120]}"
    return True, pr.stdout.strip()
