"""``loom cost``: root-cause analysis for a run's token burn.

The first production problem most teams hit isn't security -- it's a run that
quietly costs 10x what it should. This attributes the tokens and names the
pattern behind a blow-up:

  context bloat        input tokens climb every turn (history never trimmed)
  looping              the same tool called over and over
  retrieval overfetch  a read/search returns a huge result
  tool-result explosion  one tool result dominates the context

    loom cost session.loom.json

Offline; built on the recorded usage + the context-health analyzer.
"""

from __future__ import annotations


def analyze_cost(data: dict) -> dict:
    """Attribute token spend and detect burn patterns."""
    from .providers.base import ModelResponse

    turns = []  # per top-level model call: input/output tokens
    tool_counts: dict[str, int] = {}
    tool_result_tokens: list[tuple[int, str, int]] = []  # (seq, kind, est tokens)
    from .action import effect_dicts
    for e in effect_dicts(data):
        kind = e.get("kind", "")
        if kind == "model" and isinstance(e.get("result"), dict):
            u = e["result"].get("usage") or {}
            turns.append({"seq": e.get("seq"),
                          "input": u.get("input_tokens", 0) or 0,
                          "output": u.get("output_tokens", 0) or 0})
            for tc in ModelResponse.from_dict(e["result"]).tool_calls:
                tool_counts[tc.name] = tool_counts.get(tc.name, 0) + 1
        elif kind.startswith("tool:"):
            text = e["result"] if isinstance(e["result"], str) else str(e.get("result"))
            tool_result_tokens.append((e.get("seq"), kind, max(1, len(text) // 4)))

    total_in = sum(t["input"] for t in turns)
    total_out = sum(t["output"] for t in turns)
    findings = []

    # context bloat: input tokens grow monotonically and materially across turns
    inputs = [t["input"] for t in turns if t["input"]]
    if len(inputs) >= 3 and inputs[-1] > 2 * inputs[0] and inputs[-1] > 4000:
        growth = inputs[-1] - inputs[0]
        findings.append({
            "pattern": "context bloat", "severity": "high",
            "detail": f"input grew {inputs[0]:,} → {inputs[-1]:,} tokens over "
                      f"{len(inputs)} turns (+{growth:,})",
            "fix": "set Agent(compact_after=...) to summarize history, or trim "
                   "old tool results from context"})

    # looping: one tool called many times
    for name, n in sorted(tool_counts.items(), key=lambda kv: -kv[1]):
        if n >= 5:
            findings.append({
                "pattern": "looping", "severity": "high",
                "detail": f"{name} called {n} times -- the agent may be stuck",
                "fix": "add a stop condition to the prompt, lower max_steps, or "
                       "fix the tool so the model doesn't retry it"})
            break

    # retrieval overfetch / tool-result explosion: a single result dominates
    if tool_result_tokens:
        biggest = max(tool_result_tokens, key=lambda r: r[2])
        result_total = sum(r[2] for r in tool_result_tokens) or 1
        if biggest[2] >= 2000 and biggest[2] / result_total >= 0.4:
            findings.append({
                "pattern": "tool-result explosion", "severity": "medium",
                "detail": f"{biggest[1]} at step {biggest[0]} is ~{biggest[2]:,} "
                          f"tokens ({100 * biggest[2] // result_total}% of all tool output)",
                "fix": "have the tool paginate/summarize, or set a result size cap; "
                       "loom heal can verify a redaction"})

    return {
        "input_tokens": total_in,
        "output_tokens": total_out,
        "total_tokens": total_in + total_out,
        "turns": len(turns),
        "per_turn": turns,
        "top_tools": sorted(tool_counts.items(), key=lambda kv: -kv[1])[:5],
        "findings": findings,
    }


def cost_patches(data: dict) -> "list[dict]":
    """Concrete, copy-pasteable remedies derived from the burn analysis.

    Turns the burn *findings* into specific config/CLI patches with computed
    parameters -- the "cost surgeon": not "reduce context" but
    ``Agent(compact_after=3)``, not "the result is big" but a threshold.
    """
    c = analyze_cost(data)
    turns = c["per_turn"]
    patches: list[dict] = []
    for f in c["findings"]:
        if f["pattern"] == "context bloat":
            # compact once the growing history is a few turns deep
            inputs = [t["input"] for t in turns if t["input"]]
            after = max(2, next((i for i, v in enumerate(inputs) if v > 2 * inputs[0]),
                                len(inputs) // 2) or 2)
            patches.append({
                "title": "cap context growth",
                "patch": f"Agent(..., compact_after={after}, compact_keep=4)",
                "why": f"{f['detail']}; compaction summarizes history so input stops climbing"})
        elif f["pattern"] == "looping":
            tool = c["top_tools"][0][0] if c["top_tools"] else "the tool"
            patches.append({
                "title": "stop the loop",
                "patch": "Agent(..., cache=EffectCache('dev-cache.jsonl'), "
                         "kinds=('model','tool:*'))  # if calls are identical",
                "why": f"{f['detail']}; cache identical {tool} calls, or lower max_steps / "
                       "add a stop condition"})
        elif f["pattern"] == "tool-result explosion":
            patches.append({
                "title": "shrink the giant result",
                "patch": "loom artifacts externalize <trace> --threshold 8kb",
                "why": f"{f['detail']}; externalize big tool results out of the context window"})
    return patches


def cost_markdown(data: dict) -> str:
    """A PR-comment cost report: totals, burn findings, and copy-paste patches."""
    c = analyze_cost(data)
    patches = cost_patches(data)
    lines = ["### 💸 Loom cost report",
             f"**{c['total_tokens']:,} tokens** ({c['input_tokens']:,} in / "
             f"{c['output_tokens']:,} out) over {c['turns']} turn(s)"]
    if c["findings"]:
        lines.append("\n| pattern | severity | detail |\n|---|---|---|")
        for f in c["findings"]:
            lines.append(f"| {f['pattern']} | {f['severity']} | {f['detail']} |")
    if patches:
        lines.append("\n**Suggested patches:**")
        for p in patches:
            lines.append(f"- **{p['title']}** — `{p['patch']}`  \n  {p['why']}")
    if not c["findings"]:
        lines.append("\n✅ no burn patterns detected.")
    return "\n".join(lines)


def describe_patches(patches: "list[dict]") -> str:
    if not patches:
        return "no cost patches -- nothing burning."
    lines = [f"{len(patches)} cost patch(es):"]
    for p in patches:
        lines += [f"\n  ▸ {p['title']}", f"      {p['patch']}", f"      why: {p['why']}"]
    return "\n".join(lines)


def describe_cost(c: dict) -> str:
    lines = [f"cost: {c['total_tokens']:,} tokens "
             f"({c['input_tokens']:,} in / {c['output_tokens']:,} out) "
             f"over {c['turns']} turn(s)"]
    if c["top_tools"]:
        lines.append("  tool calls: "
                     + ", ".join(f"{n}×{t}" for t, n in c["top_tools"]))
    if c["findings"]:
        lines.append("\n  burn patterns:")
        for f in c["findings"]:
            lines.append(f"    ⚠ [{f['severity']}] {f['pattern']}: {f['detail']}")
            lines.append(f"        fix: {f['fix']}")
    else:
        lines.append("  no burn pattern detected")
    return "\n".join(lines)
