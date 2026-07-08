"""``loom why``: ask a trace what happened -- and get answers that cite seqs.

The debugger is itself a loom agent: it reads the saved trace through tools
(timeline, individual effects, the conversation, a context-health checkup,
shield decisions) and answers questions like "why did the run go off the
rails after turn 3?" with references to specific effect seqs you can then
inspect, fork, or bisect.

    loom why session.loom.json "why did it read the .env file?"

Because the debugger runs on loom, its OWN run is recorded too (``--save``)
-- you can replay the diagnosis, or ask why about the why.
"""

from __future__ import annotations

import json

from .effect import EffectEntry
from .tools import tool

_SYSTEM = (
    "You are a debugger for recorded AI-agent runs (loom traces). A trace is an "
    "ordered log of effects, each with a seq number: 'model' effects are what "
    "the model said (text and tool_calls), 'tool:<name>' effects are tool "
    "results, and other kinds are harness events (human, memory, compact, "
    "critic...). Investigate with your tools before answering; do not guess. "
    "ALWAYS cite the seq numbers your conclusions rest on, e.g. 'at seq 4 the "
    "model...'. Be concrete and brief. If the evidence is inconclusive, say "
    "what to look at next (which seq to inspect, where to fork)."
)

_SNIPPET = 200  # chars of a result shown per timeline row


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _summarize_effect(entry: dict) -> str:
    result = entry.get("result")
    if isinstance(result, dict):  # a model effect
        calls = result.get("tool_calls") or []
        if calls:
            sig = ", ".join(
                f"{c.get('name')}({json.dumps(c.get('input', {}), sort_keys=True, default=str)})"
                for c in calls
            )
            return f"calls {sig}"[:_SNIPPET]
        return (result.get("text") or "")[:_SNIPPET]
    return str(result)[:_SNIPPET]


def build_why_tools(path: str) -> list:
    """The trace-reading tools the debugger agent investigates with."""
    data = _load(path)
    log: list[dict] = data.get("log", [])

    @tool
    def conversation() -> str:
        "The user's messages, the run's final output, and basic facts."
        return json.dumps(
            {
                "episodes": data.get("episodes", []),
                "output": data.get("output", ""),
                "model": data.get("model", ""),
                "stop_reason": data.get("stop_reason", ""),
                "num_effects": len(log),
                "recorded_via": data.get("recorded_via", "harness"),
            }
        )

    @tool
    def timeline() -> str:
        "Every effect in order: seq, kind, one-line summary. Start here."
        rows = [f"[{e.get('seq')}] {e.get('kind')}  {_summarize_effect(e)}" for e in log]
        return "\n".join(rows) or "(empty trace)"

    @tool
    def effect(seq: int) -> str:
        "The full recorded entry at one seq (inputs key, complete result)."
        for e in log:
            if e.get("seq") == seq:
                return json.dumps(e, default=str)[:8000]
        return f"no effect with seq {seq} (valid: 0..{len(log) - 1})"

    @tool
    def checkup() -> str:
        "Context-health findings: oversized, unused, duplicate context items."
        from .health import analyze

        entries = [EffectEntry.from_dict(e) for e in log]
        return analyze(data.get("episodes", []), entries).summary()

    @tool
    def shield_log() -> str:
        "Firewall decisions made during the run (blocked/approved tool calls)."
        events = data.get("shield_events", [])
        return json.dumps(events) if events else "(no shield events)"

    @tool
    def cost() -> str:
        "Token usage per model effect: find the expensive turns."
        rows = []
        for e in log:
            if e.get("kind") == "model" and isinstance(e.get("result"), dict):
                usage = e["result"].get("usage", {}) or {}
                rows.append(
                    f"[{e.get('seq')}] in={usage.get('input_tokens', 0)} "
                    f"out={usage.get('output_tokens', 0)}"
                )
        return "\n".join(rows) or "(no model effects)"

    return [conversation, timeline, effect, checkup, shield_log, cost]


def build_why_agent(path: str, model: str = "claude-opus-4-8", provider=None):
    """A loom agent wired to investigate the trace at ``path``."""
    from .agent import Agent

    return Agent(
        model=provider if provider is not None else model,
        tools=build_why_tools(path),
        system=_SYSTEM,
    )


def why(path: str, question: str, model: str = "claude-opus-4-8", provider=None):
    """Ask one question about a trace. Returns the debugger's Run."""
    return build_why_agent(path, model=model, provider=provider).run(question)
