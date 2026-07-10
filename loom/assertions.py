"""Behavioural assertions over a recorded run -- turn a debugging session into
a repeatable check.

While you debug, you form expectations ("it should never call the refund tool",
"the answer must mention the order id"). An assertion bar lets you write those
down in plain English and get pass/fail against the trace *right now* -- and,
because it's the same expression language a policy or a CI test would use, the
expectation you eyeballed in the debugger can graduate straight into a test.

One assertion per line. The grammar is deliberately small and forgiving:

    output contains <text>      the final answer includes <text>  (case-insensitive)
    output matches <regex>      the final answer matches <regex>
    calls <tool>                some tool call matches <tool>      (glob ok: issue_*)
    never <tool>                no tool call matches <tool>
    no risk                     no action was flagged risky
    no blocked                  nothing was blocked by the firewall
    blocked <tool>              <tool> was blocked at least once
    steps < N        (or <=, =) how many actions the run took
    tokens < N                  total token budget bound
    answers                     the run produced a final answer at all
    judge: <expectation>        a SEMANTIC expectation an LLM judges against the
                                transcript ("the agent verified the order before
                                refunding") -- needs a judge model (--judge / the
                                debugger's copilot model)

Unknown lines are reported as errors, never silently passed.
"""

from __future__ import annotations

import re
from fnmatch import fnmatchcase as _glob


def _tool_calls(acts) -> list:
    return [a for a in acts if a.type == "call"]


def _eval_one(expr: str, data: dict, acts, output: str, judge=None) -> dict:
    """Evaluate a single assertion line -> {expr, ok, detail} or {expr, error}."""
    raw = expr.strip()
    low = raw.lower()

    def ok(cond: bool, detail: str = "") -> dict:
        return {"expr": raw, "ok": bool(cond), "detail": detail}

    # -- semantic expectation, judged by an LLM --
    m = re.match(r"judge\s*:\s*(.+)", raw, re.I)
    if m:
        if judge is None:
            return {"expr": raw, "error": "semantic assertion needs a judge model "
                                          "(--judge MODEL, or the debugger's copilot)"}
        from .judge import llm_judge

        v = llm_judge(judge, m.group(1).strip(), data)
        if "error" in v:
            return {"expr": raw, "error": v["error"]}
        return ok(v["ok"], v["reason"])

    # -- output checks --
    m = re.match(r"output\s+(?:contains|has|includes)\s+(.+)", low)
    if m:
        needle = raw[raw.lower().index(m.group(1)):].strip().strip("'\"")
        return ok(needle.lower() in output.lower(), f"looked for {needle!r}")
    m = re.match(r"output\s+matches\s+(.+)", raw, re.I)
    if m:
        pat = m.group(1).strip().strip("'\"")
        try:
            return ok(re.search(pat, output, re.I) is not None, f"/{pat}/")
        except re.error as e:
            return {"expr": raw, "error": f"bad regex: {e}"}

    # -- tool-call presence / absence --
    m = re.match(r"(?:calls?|uses?)\s+(.+)", low)
    if m:
        pat = m.group(1).strip()
        hit = [a.tool for a in _tool_calls(acts) if _glob(a.tool.lower(), pat) or a.tool.lower() == pat]
        return ok(bool(hit), f"matched {sorted(set(hit))}" if hit else "no matching call")
    m = re.match(r"(?:never|no)\s+(?:calls?\s+|uses?\s+)?(.+)", low)
    if m and m.group(1) not in ("risk", "blocked", "risky"):
        pat = m.group(1).strip()
        hit = [a.tool for a in _tool_calls(acts) if _glob(a.tool.lower(), pat) or a.tool.lower() == pat]
        return ok(not hit, f"unexpected {sorted(set(hit))}" if hit else "never called")

    # -- risk / firewall --
    if low in ("no risk", "no risky", "not risky"):
        risky = [a.tool for a in acts if getattr(a, "risky", False) or a.risk]
        return ok(not risky, f"risky: {sorted(set(risky))}" if risky else "clean")
    if low in ("no blocked", "not blocked", "nothing blocked"):
        blk = [a.tool for a in acts if a.policy is not None and a.policy.blocked]
        return ok(not blk, f"blocked: {sorted(set(blk))}" if blk else "none blocked")
    m = re.match(r"blocked\s+(.+)", low)
    if m:
        pat = m.group(1).strip()
        blk = [a.tool for a in acts if a.policy is not None and a.policy.blocked
               and (_glob(a.tool.lower(), pat) or a.tool.lower() == pat)]
        return ok(bool(blk), f"blocked {sorted(set(blk))}" if blk else "not blocked")

    # -- counts --
    m = re.match(r"(steps|tokens)\s*(<=|>=|<|>|=|==)\s*(\d[\d_]*)", low)
    if m:
        which, op, n = m.group(1), m.group(2), int(m.group(3).replace("_", ""))
        if which == "steps":
            val = len(acts)
        else:
            from .cost import analyze_cost
            val = analyze_cost(data)["total_tokens"]
        cmp = {"<": val < n, "<=": val <= n, ">": val > n, ">=": val >= n,
               "=": val == n, "==": val == n}[op]
        return ok(cmp, f"{which}={val}")

    if low in ("answers", "has answer", "answered"):
        return ok(bool(output.strip()), "produced output" if output.strip() else "no output")

    return {"expr": raw, "error": "unrecognized assertion"}


def check_assertions(data: dict, exprs: "list[str] | str", judge=None) -> dict:
    """Evaluate assertions against a trace. Returns {results:[...], passed, total}.

    ``exprs`` is a list of lines or a single newline-separated string. Blank
    lines and ``#`` comments are ignored. ``judge`` (a model name or provider)
    enables semantic ``judge:`` lines."""
    from .action import actions

    if isinstance(exprs, str):
        exprs = exprs.splitlines()
    lines = [e for e in (x.strip() for x in exprs) if e and not e.startswith("#")]
    acts = actions(data)
    output = str(data.get("output", "") or "")
    results = [_eval_one(e, data, acts, output, judge=judge) for e in lines]
    passed = sum(1 for r in results if r.get("ok"))
    return {"results": results, "passed": passed, "total": len(results),
            "all_pass": bool(results) and all(r.get("ok") for r in results)}
