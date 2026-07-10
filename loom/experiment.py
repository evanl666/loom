"""``loom experiment``: A/B test prompts and models on one task, scored.

"Which of these 3 system prompts / 2 models is best?" -- run them all on the
same task, and rank by success, behavior score, and cost. Every branch is a
recorded trace, so you can then `loom debug` the winner (or the loser):

    loom experiment --agent app:agent "refund order 12345" \\
        --system "You are careful." --system "You are fast." \\
        --model claude-haiku-4-5 --model claude-sonnet-5 \\
        --contains REFUNDED

Runs the cross-product of system prompts x models, scores each, and prints a
ranked table. --contains / --absent define success.
"""

from __future__ import annotations

from typing import Any, Callable


def run_experiment(agent: Any, prompt: str, systems: "list[str] | None" = None,
                   models: "list[str] | None" = None,
                   check: "Callable[[str], bool] | None" = None,
                   save_dir: str = "", judge: Any = None, criteria: str = "") -> "list[dict]":
    """Run the task under each (system x model) variant; return ranked results.

    ``check`` is a plain output predicate; ``judge`` + ``criteria`` instead give
    a SEMANTIC success signal -- an LLM judges each variant's full transcript
    against the criteria ("the agent verified the order before refunding"), so
    variants are ranked by meaning, not string matching."""
    import os

    from .agent import Agent
    from .cost import analyze_cost
    from .diff import score_breakdown

    sys_variants = systems or [None]
    model_variants = models or [None]
    tools = list(agent.tools.values())

    results: list[dict] = []
    idx = 0
    for s in sys_variants:
        for m in model_variants:
            idx += 1
            sys_p = agent.system if s is None else s
            model_arg = m if m else agent.provider  # keep the base provider if no override
            a = Agent(model=model_arg, tools=tools, system=sys_p, max_steps=agent.max_steps)
            run = a.run(prompt)
            data = run.to_dict()
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                run.save(os.path.join(save_dir, f"variant{idx}.loom.json"))
            label = _label(s, m, idx)
            success, reason = (bool(check(run.output)) if check else None), ""
            if judge is not None and criteria:
                from .judge import llm_judge

                v = llm_judge(judge, criteria, data)
                if "error" in v:
                    success, reason = None, v["error"]
                else:
                    success, reason = v["ok"], v["reason"]
            results.append({
                "variant": label,
                "system": (sys_p[:60] + "…") if sys_p and len(sys_p) > 60 else (sys_p or "(default)"),
                "model": m or getattr(agent, "model", "(base)"),
                "output": run.output,
                "score": score_breakdown(data)["overall"],
                "tokens": analyze_cost(data)["total_tokens"],
                "success": success,
                **({"judge_reason": reason} if reason else {}),
                "ok": not run.truncated and not run.stop_reason,
            })

    def _rank(r: dict):
        succ = 2 if r["success"] else (1 if r["success"] is None else 0)
        return (succ, r["score"], -r["tokens"])
    results.sort(key=_rank, reverse=True)
    return results


def _label(system: "str | None", model: "str | None", i: int) -> str:
    parts = []
    if system is not None:
        parts.append(f"sys:{system[:18]}…" if len(system) > 18 else f"sys:{system}")
    if model is not None:
        parts.append(model.split("-2025")[0])
    return " · ".join(parts) or f"variant{i}"


def contains_check(needle: str) -> "Callable[[str], bool]":
    return lambda out: needle.lower() in out.lower()


def absent_check(needle: str) -> "Callable[[str], bool]":
    return lambda out: needle.lower() not in out.lower()


def describe_experiment(results: "list[dict]") -> str:
    lines = [f"experiment — {len(results)} variant(s), ranked best first", "",
             f"  {'variant':<28} {'score':>5} {'tokens':>7} {'ok':>4} {'pass':>5}",
             "  " + "-" * 56]
    for i, r in enumerate(results):
        win = "🏆" if i == 0 else "  "
        p = "✓" if r["success"] else ("✗" if r["success"] is False else "—")
        lines.append(f"{win}{r['variant']:<28} {r['score']:>5} {r['tokens']:>7,} "
                     f"{'✓' if r['ok'] else '✗':>4} {p:>5}")
    if results:
        w = results[0]
        lines += ["", f"  winner: {w['variant']} (score {w['score']}, {w['tokens']:,} tokens)",
                  f"  output: {w['output'].strip()[:100]}"]
    return "\n".join(lines)
