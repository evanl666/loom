"""``loom bench``: run the same task through several agents, compare the traces.

    loom bench tasks/fix-tests.yaml \\
        --agent "claude:claude -p {prompt}" \\
        --agent "codex:codex exec {prompt}" \\
        --profile claude-code-safe

Each agent runs behind the recording proxy (and firewall, if a profile is
given), exactly like ``loom record``. The task file names the prompt and what
counts as success; the report scores each agent on pass/cost/steps/tools/
blocked -- all read from the resulting trace. "SWE-bench for your own repo,
with a replayable trace behind every cell."

Because every agent is just a command talking to the API through the proxy,
this works for anything that speaks Anthropic or OpenAI -- and is testable
offline with a scripted stand-in.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import threading


def load_task(path: str) -> dict:
    """Load a bench task: {prompt, success: {contains/absent/command}}."""
    from .policy_file import _parse

    with open(path) as f:
        task = _parse(f.read(), path)
    if not task.get("prompt"):
        raise ValueError(f"{path}: a bench task needs a 'prompt'")
    return task


def _oracle(task: dict, output: str, workdir: str) -> "tuple[bool, str]":
    """Did the agent succeed? Returns (passed, how it was judged)."""
    success = task.get("success") or {}
    if "contains" in success:
        needle = success["contains"]
        return (needle in output), f"output contains {needle!r}"
    if "absent" in success:
        needle = success["absent"]
        return (needle not in output), f"output lacks {needle!r}"
    if "command" in success:
        cmd = success["command"]
        try:
            code = subprocess.call(shlex.split(cmd), cwd=workdir,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            return False, f"could not run {cmd!r}"
        return (code == 0), f"`{cmd}` exit {code}"
    return True, "no success check (ran to completion)"


def score(trace: dict, passed: bool) -> dict:
    """Everything the comparison table needs, from one trace."""
    log = trace.get("log") or []
    tokens = tools = steps = 0
    toolset: set = set()
    for e in log:
        if e.get("kind") == "model" and isinstance(e.get("result"), dict):
            steps += 1
            usage = e["result"].get("usage") or {}
            tokens += (usage.get("input_tokens", 0) or 0) + (usage.get("output_tokens", 0) or 0)
            for tc in e["result"].get("tool_calls") or []:
                toolset.add(tc.get("name", "?"))
    events = trace.get("shield_events") or []
    blocked = sum(1 for ev in events if ev.get("action") == "deny")
    return {
        "passed": passed, "tokens": tokens, "steps": steps,
        "tools": len(toolset), "blocked": blocked,
    }


def run_agent(name: str, command: str, task: dict, target: str,
              shield=None, outdir: str = ".", recorder_factory=None) -> dict:
    """Record one agent against the task. Returns its scored result."""
    from .proxy import ProxyServer

    prompt = task["prompt"]
    if "{prompt}" in command:
        argv = [p.replace("{prompt}", prompt) for p in shlex.split(command)]
    else:
        argv = shlex.split(command) + [prompt]

    save = os.path.join(outdir, f"{name}.loom.json")
    server = ProxyServer(port=0, target=target, save_path=save, shield=shield)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    env = dict(os.environ)
    base = f"http://127.0.0.1:{server.port}"
    env["OPENAI_BASE_URL" if "openai" in target else "ANTHROPIC_BASE_URL"] = (
        base + "/v1" if "openai" in target else base
    )

    result: dict = {"name": name, "error": ""}
    try:
        code = subprocess.call(argv, env=env)
        result["exit_code"] = code
    except OSError as e:
        result["error"] = f"could not run {argv[0]!r}: {e}"
    finally:
        server.shutdown()
        server.finalize()

    try:
        with open(save) as f:
            trace = json.load(f)
    except (OSError, json.JSONDecodeError):
        result["error"] = result["error"] or "no traffic recorded"
        return {**result, "passed": False, "tokens": 0, "steps": 0, "tools": 0, "blocked": 0}

    output = str(trace.get("output", ""))
    passed, how = _oracle(task, output, os.getcwd())
    result["how"] = how
    result["trace"] = save
    return {**result, **score(trace, passed)}


def report(task_path: str, results: "list[dict]") -> str:
    """A comparison table, one row per agent."""
    lines = [f"Task: {task_path}", ""]
    header = f"{'agent':<14} {'pass':<5} {'tokens':>8} {'steps':>6} {'tools':>6} {'blocked':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in results:
        if r.get("error"):
            lines.append(f"{r['name']:<14} error: {r['error']}")
            continue
        mark = "✅" if r["passed"] else "❌"
        lines.append(
            f"{r['name']:<14} {mark:<5} {r['tokens']:>8,} {r['steps']:>6} "
            f"{r['tools']:>6} {r['blocked']:>8}"
        )
    ok = [r for r in results if r.get("passed")]
    if ok:
        cheapest = min(ok, key=lambda r: r["tokens"])
        lines += ["", f"cheapest passing: {cheapest['name']} ({cheapest['tokens']:,} tokens)"]
    return "\n".join(lines)
