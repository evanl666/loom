"""``loom loops``: find where an agent got stuck repeating itself.

"Why is it looping?" is a top agent-debugging question. This detects the shapes
a stuck agent makes -- the same tool call repeated, or an A→B→A→B oscillation --
names the cycle, says where it started, and suggests a fix:

    loom loops session.loom.json
    loom loops session.loom.json --gate     # exit 1 if a loop is found
"""

from __future__ import annotations

import json
from typing import Any


def _sig(tool: str, tool_input: Any) -> str:
    return f"{tool}({json.dumps(tool_input, sort_keys=True, default=str)})"


def detect_loops(data: Any) -> dict:
    """Repeated / oscillating tool-call patterns in a run."""
    from .action import actions

    calls = [(a.step, _sig(a.tool, a.input)) for a in actions(data)
             if a.type == "call" and a.tool]
    sigs = [s for _, s in calls]
    findings: list[dict] = []

    # 1. a call repeated many times (anywhere)
    counts: dict[str, list] = {}
    for step, s in calls:
        counts.setdefault(s, []).append(step)
    for s, steps in counts.items():
        if len(steps) >= 3:
            findings.append({"kind": "repeat", "pattern": s[:80], "times": len(steps),
                             "steps": steps, "started": steps[0]})

    # 2. an oscillating cycle: the tail is k>=2 repetitions of an L-cycle
    n = len(sigs)
    best = None
    for L in range(1, n // 2 + 1):
        reps = 1
        i = n - L
        while i - L >= 0 and sigs[i - L:i] == sigs[i:i + L] and sigs[i:i + L] == sigs[n - L:n]:
            reps += 1
            i -= L
        if reps >= 2 and (best is None or L * reps > best[0] * best[1]):
            best = (L, reps, sigs[n - L:n])
    if best and best[0] >= 2:  # a >=2-step cycle (single-step repeat is case 1)
        L, reps, cyc = best
        findings.append({"kind": "cycle", "length": L, "repeats": reps,
                         "pattern": " → ".join(c.split("(")[0] for c in cyc),
                         "started": calls[n - L * reps][0]})

    looping = bool(findings)
    tip = ("add a stop condition to the prompt, lower max_steps, cache identical "
           "calls (Agent(cache=...)), or fix the tool so the model stops retrying"
           if looping else "")
    return {"looping": looping, "calls": len(calls), "findings": findings, "fix": tip}


def describe_loops(r: dict) -> str:
    if not r["looping"]:
        return f"no loops ({r['calls']} tool call(s), all making progress)"
    lines = [f"⟳ loop detected across {r['calls']} tool call(s):"]
    for f in r["findings"]:
        if f["kind"] == "repeat":
            lines.append(f"  🔁 {f['pattern']} called {f['times']}× "
                         f"(steps {f['steps'][0]}…{f['steps'][-1]})")
        else:
            lines.append(f"  🔄 {f['length']}-step cycle ×{f['repeats']}: {f['pattern']} "
                         f"(from step {f['started']})")
    lines.append(f"  fix: {r['fix']}")
    return "\n".join(lines)
