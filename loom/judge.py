"""One shared LLM-judgment primitive for every "is this expectation met?" surface.

String rules (``output contains X``) can't express *semantic* expectations --
"the agent verified the order before refunding", "the reply is polite". This
module turns such an expectation + a compact view of the run into a strict
pass/fail verdict with a reason, using whatever judge model the caller supplies.

Used by:
  * assertions.py  -- ``judge: <expectation>`` lines (assert bar + ``loom assert --judge``)
  * experiment.py  -- ``judge_check(model, criteria)`` as a ranking check
  * debugger.py    -- the assert drawer judges with the copilot model automatically

Deliberately conservative: a judge error or malformed reply is reported as an
error, never silently passed -- an eval that can quietly succeed is worse than
no eval.
"""

from __future__ import annotations

import json
import re
from typing import Any


def _resolve(model: Any):
    if isinstance(model, str):
        from .agent import _resolve_provider

        return _resolve_provider(model, None)
    return model


def run_summary(data: dict, max_chars: int = 4000) -> str:
    """A compact, judge-readable view of a run: prompt, each action, output."""
    from .action import actions

    head = f"USER REQUEST: {str(data.get('prompt', ''))[:500]}"
    tail = f"FINAL OUTPUT: {str(data.get('output', ''))[:800]}"
    steps: list[str] = []
    for a in actions(data):
        if a.type == "call":
            obs = (a.observation.text or "")[:200] if a.observation else ""
            steps.append(f"[{a.step}] tool {a.tool}({json.dumps(a.input, default=str)[:200]})"
                         f" -> {obs}")
        elif a.type == "answer" and a.intent:
            steps.append(f"[{a.step}] agent answered: {a.intent[:300]}")
        elif a.type == "reason" and a.intent:
            steps.append(f"[{a.step}] agent reasoning: {a.intent[:200]}")

    # The request and the FINAL OUTPUT are what a judge most needs, so keep them
    # whole and elide the MIDDLE of the step list to fit -- never chop the output
    # off the end (the old bug: a long run's answer got truncated away).
    budget = max(200, max_chars - len(head) - len(tail) - 60)
    body = "\n".join(steps)
    if len(body) > budget:
        front: list[str] = []
        back: list[str] = []
        flen = blen = 0
        i, j = 0, len(steps) - 1
        while i <= j:
            if flen <= blen:
                if flen + len(steps[i]) + 1 > budget // 2:
                    break
                front.append(steps[i]); flen += len(steps[i]) + 1; i += 1
            else:
                if blen + len(steps[j]) + 1 > budget // 2:
                    break
                back.insert(0, steps[j]); blen += len(steps[j]) + 1; j -= 1
        omitted = len(steps) - len(front) - len(back)
        body = "\n".join(front + [f"... [{omitted} step(s) elided] ..."] + back)
    return "\n".join([head, body, tail])


def judge_text(model: Any, question: str, text: str, max_chars: int = 2500) -> dict:
    """Judge one yes/no ``question`` about a raw ``text`` snippet (not a whole
    run). Returns {"ok": bool, "reason": str} or {"error": str} -- never raises.

    Used for per-snippet checks like "does this untrusted content contain an
    injection?", where a whole-run summary would be the wrong granularity."""
    try:
        provider = _resolve(model)
        system = (
            "You answer a single yes/no question about a piece of text. "
            'Reply with ONLY JSON: {"yes": true|false, "reason": "<one short sentence>"}. '
            "Be precise; when genuinely unsure, answer false and say why."
        )
        user = f"QUESTION: {question}\n\nTEXT:\n{text[:max_chars]}"
        resp = provider.complete(system, [{"role": "user", "content": user}], [])
        m = re.search(r"\{.*\}", resp.text or "", re.S)
        if not m:
            return {"error": f"judge gave no JSON: {(resp.text or '')[:80]!r}"}
        v = json.loads(m.group(0))
        if not isinstance(v, dict) or not isinstance(v.get("yes"), bool):
            return {"error": f"judge verdict malformed: {(resp.text or '')[:80]!r}"}
        return {"ok": v["yes"], "reason": str(v.get("reason", ""))[:200]}
    except Exception as e:  # noqa: BLE001
        return {"error": f"judge error: {type(e).__name__}: {e}"}


def llm_judge(model: Any, expectation: str, data: dict) -> dict:
    """Judge one semantic expectation against a run.

    Returns {"ok": bool, "reason": str} or {"error": str} -- never raises."""
    try:
        provider = _resolve(model)
        system = (
            "You are a strict evaluator of an AI agent's recorded run. Given the "
            "transcript and ONE expectation, decide if the run satisfies it.\n"
            'Reply with ONLY a JSON object: {"pass": true|false, "reason": "<one short sentence>"}\n'
            "Judge only what the transcript shows; when genuinely ambiguous, fail it "
            "and say why."
        )
        user = f"EXPECTATION: {expectation}\n\nTRANSCRIPT:\n{run_summary(data)}"
        resp = provider.complete(system, [{"role": "user", "content": user}], [])
        text = resp.text or ""
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return {"error": f"judge gave no JSON verdict: {text[:80]!r}"}
        verdict = json.loads(m.group(0))
        if not isinstance(verdict, dict) or not isinstance(verdict.get("pass"), bool):
            return {"error": f"judge verdict malformed: {text[:80]!r}"}
        return {"ok": verdict["pass"], "reason": str(verdict.get("reason", ""))[:200]}
    except Exception as e:  # noqa: BLE001 -- a judge failure is a result, not a crash
        return {"error": f"judge error: {type(e).__name__}: {e}"}
