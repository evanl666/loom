"""The generic Action schema -- the vocabulary of the Action Debugger.

Loom's effect log records low-level events (model calls, tool calls, human
answers). This module lifts that log into the domain-neutral vocabulary a
*debugger* needs, so a coding agent, a browser agent, and a SQL agent are all
described the same way:

  Action          one thing the agent did (reasoned, called a tool, answered)
  Observation     what came back (result text, error flag, tokens)
  StateDiff       how the outside world changed (files, rows, DOM, fields)
  PolicyDecision  what the firewall decided (allow / deny / confirm, and why)
  ReplayPoint     a handle to replay or fork from this step

The base builder (``actions(trace)``) fills everything derivable from the
trace itself -- intent, inputs, capabilities, risk, observations, firewall
decisions, replay points. Packs (``loom.packs``) enrich each Action with a
domain-specific ``StateDiff`` (a file diff for coding, a row diff for SQL, a
DOM diff for a browser agent), because only a pack knows how to read its own
world. That split -- generic timeline here, world-diff in the pack -- is what
lets Loom debug agents it didn't build.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .effect import EffectEntry
from .providers.base import ModelResponse


@dataclass
class Observation:
    """What an action returned."""

    text: str = ""          # human-readable result
    error: bool = False     # did it fail (tool error) or get blocked
    tokens: dict = field(default_factory=dict)  # usage, for model actions
    raw: Any = None         # the raw recorded result payload

    def to_dict(self) -> dict:
        d = {"text": self.text, "error": self.error}
        if self.tokens:
            d["tokens"] = self.tokens
        return d


@dataclass
class StateDiff:
    """How the outside world changed because of an action.

    ``kind`` names the world ("file", "database", "dom", "field", "none");
    ``summary`` is a one-line human description; ``detail`` is pack-specific
    (a unified diff, row counts, a screenshot ref). Filled by a Pack -- the
    base builder leaves it None because only a pack can read its own world.
    """

    kind: str = "none"
    summary: str = ""
    detail: Any = None

    def to_dict(self) -> dict:
        d = {"kind": self.kind, "summary": self.summary}
        if self.detail is not None:
            d["detail"] = self.detail
        return d


@dataclass
class PolicyDecision:
    """What the firewall decided about an action."""

    action: str = ""    # "allow" | "deny" | "confirm"/"approve" | ""
    rule: str = ""
    via: str = ""       # "rule" | "sequence" | "judge" | "operator" | "default"
    by: str = ""        # operator identity, when a human decided

    @property
    def blocked(self) -> bool:
        return self.action == "deny"

    def to_dict(self) -> dict:
        d = {"action": self.action, "rule": self.rule, "via": self.via}
        if self.by:
            d["by"] = self.by
        return d


@dataclass
class ReplayPoint:
    """A handle to replay or fork the run from this step.

    ``turn`` is the top-level turn a ``run.fork(turn)`` would rewind to;
    ``forkable`` is True only on top-level turn boundaries (model calls at
    depth 0), which are the points fork can restart from.
    """

    step: int
    turn: int
    forkable: bool = False

    def to_dict(self) -> dict:
        return {"step": self.step, "turn": self.turn, "forkable": self.forkable}


@dataclass
class Action:
    """One thing the agent did, described in world-neutral terms."""

    step: int
    depth: int
    type: str               # "reason" | "call" | "answer" | "ask-human" | "meta"
    tool: str = ""          # tool name for calls, "" otherwise
    intent: str = ""        # WHY: the model's text/reasoning around this action
    input: Any = None       # tool input / prompt
    capabilities: list[str] = field(default_factory=list)
    risk: str = ""          # top risk category, "" if none
    observation: "Observation | None" = None
    state_diff: "StateDiff | None" = None   # filled by a Pack
    policy: "PolicyDecision | None" = None
    replay: "ReplayPoint | None" = None

    @property
    def risky(self) -> bool:
        from .risk import DANGEROUS

        return self.risk in DANGEROUS

    def to_dict(self) -> dict:
        d: dict = {"step": self.step, "depth": self.depth, "type": self.type}
        if self.tool:
            d["tool"] = self.tool
        if self.intent:
            d["intent"] = self.intent
        if self.input is not None:
            d["input"] = self.input
        if self.capabilities:
            d["capabilities"] = self.capabilities
        if self.risk:
            d["risk"] = self.risk
        for k in ("observation", "state_diff", "policy", "replay"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v.to_dict()
        return d


def _as_trace(source: Any) -> dict:
    """Accept a trace dict, a Run, or anything with .to_dict()."""
    if isinstance(source, dict):
        data = source
    elif hasattr(source, "to_dict"):
        data = source.to_dict()
        # Run.to_dict() omits proxy-added keys; pull them off the object.
        for k in ("shield_events", "workspace"):
            if k not in data:
                extra = getattr(source, k, None)
                rec = getattr(source, "recorder", None)
                extra = extra if extra is not None else getattr(rec, k, None)
                if extra:
                    data[k] = extra
    else:
        raise TypeError(f"cannot read a trace from {type(source).__name__}")
    return data


def _result_text(result: Any) -> str:
    import json

    if isinstance(result, str):
        return result
    try:
        return json.dumps(result)
    except (TypeError, ValueError):
        return str(result)


def actions(source: Any) -> list[Action]:
    """Lift a trace (dict / Run) into the generic Action timeline.

    Executed tool calls are paired with the model's reasoning that requested
    them; firewall decisions are matched from ``shield_events``; blocked calls
    (denied before they ran) appear as their own ``policy.blocked`` Actions.
    """
    from .capabilities import capabilities as _caps
    from .risk import classify as _classify

    data = _as_trace(source)
    log = [EffectEntry.from_dict(e) if isinstance(e, dict) else e for e in data.get("log", [])]
    shield_events = list(data.get("shield_events", []))

    # Turn boundaries: top-level model calls are the fork points.
    turn_of_seq: dict[int, int] = {}
    turn = 0
    for e in log:
        if e.kind == "model" and e.depth == 0:
            turn_of_seq[e.seq] = turn
            turn += 1

    # Firewall decisions that let a call through, queued per tool (in order),
    # so each executed call claims the next matching allow/approve; denies are
    # emitted separately as blocked Actions.
    allow_q: dict[str, list[dict]] = {}
    denies: list[dict] = []
    for ev in shield_events:
        if ev.get("action") == "deny":
            denies.append(ev)
        elif ev.get("action") in ("allow", "approve"):
            allow_q.setdefault(ev.get("tool", ""), []).append(ev)

    def _policy_for(tool: str) -> "PolicyDecision | None":
        q = allow_q.get(tool)
        if not q:
            return None
        ev = q.pop(0)
        return PolicyDecision(ev.get("action", ""), ev.get("rule", ""),
                              ev.get("via", ""), ev.get("by", ""))

    out: list[Action] = []
    # Pending tool_calls awaiting their execution effect, per depth (subagents
    # run at depth+1 and their calls must not be paired with a parent's tool).
    pending: dict[int, list[tuple[str, str, Any]]] = {}  # depth -> [(name, intent, input)]
    cur_turn = -1

    for e in log:
        if e.seq in turn_of_seq:
            cur_turn = turn_of_seq[e.seq]
        replay = ReplayPoint(step=e.seq, turn=max(cur_turn, 0), forkable=e.seq in turn_of_seq)

        if e.kind == "model":
            resp = ModelResponse.from_dict(e.result)
            intent = resp.text or ""
            if resp.tool_calls:
                for tc in resp.tool_calls:
                    pending.setdefault(e.depth, []).append((tc.name, intent, tc.input))
                if intent.strip():  # a "thought" preceding the calls
                    out.append(Action(step=e.seq, depth=e.depth, type="reason",
                                       intent=intent, replay=replay,
                                       observation=Observation(tokens=resp.usage)))
            else:  # a final text answer
                out.append(Action(step=e.seq, depth=e.depth, type="answer", intent=intent,
                                   replay=replay,
                                   observation=Observation(text=intent, tokens=resp.usage)))
        elif e.kind.startswith("tool:"):
            name = e.kind[5:]
            queue = pending.get(e.depth) or []
            match = next((i for i, (n, _, _) in enumerate(queue) if n == name), None)
            if match is None:
                intent, tool_input = "", None
            else:
                _, intent, tool_input = queue.pop(match)
            text = _result_text(e.result)
            err = isinstance(e.result, str) and e.result.startswith(("ERROR", "BLOCKED", "DRY-RUN"))
            out.append(Action(
                step=e.seq, depth=e.depth, type="call", tool=name, intent=intent,
                input=tool_input,
                capabilities=sorted(_caps(name, tool_input or {})),
                risk=_classify(name, tool_input or {}),
                observation=Observation(text=text, error=err, raw=e.result),
                policy=_policy_for(name), replay=replay,
            ))
        elif e.kind == "human":
            out.append(Action(step=e.seq, depth=e.depth, type="ask-human",
                               observation=Observation(text=_result_text(e.result)),
                               replay=replay))
        else:  # memory, compaction, sample, critic... harness-internal steps
            out.append(Action(step=e.seq, depth=e.depth, type="meta", tool=e.kind,
                               observation=Observation(text=_result_text(e.result)),
                               replay=replay))

    # Calls the firewall blocked before they ran (no execution effect exists).
    for ev in denies:
        name = ev.get("tool", "")
        tool_input = ev.get("input", {})
        out.append(Action(
            step=-1, depth=0, type="call", tool=name, input=tool_input,
            capabilities=sorted(_caps(name, tool_input or {})),
            risk=_classify(name, tool_input or {}),
            observation=Observation(text="(blocked -- not executed)", error=True),
            policy=PolicyDecision("deny", ev.get("rule", ""), ev.get("via", "rule"),
                                  ev.get("by", "")),
        ))
    return out
