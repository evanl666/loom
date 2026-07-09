"""Impact analysis: which recorded runs does a config change affect?

Snapshot testing for agents. You changed the system prompt, swapped a tool,
or reordered instructions -- before paying for a single API call, replay your
trace corpus against the new configuration and see exactly which runs are
touched and where:

    impacts = assess(["fixtures/a.loom.json", ...], agent=new_agent)
    print(report(impacts))

Dry mode (default) is free: it walks each trace in replay, recomputing every
effect's input key under the new config and comparing it with the recorded
key. The first mismatch is the first turn your change reaches. Live mode
(``live=True``) actually re-runs each conversation and diffs the outputs --
that is what it costs to know how behavior changes, not just where.

Dry mode also *sizes* the recomputed model inputs (system + rebuilt messages,
~4 chars/token): ``Impact.est_input_tokens`` is what the same conversation
would cost in input tokens under this configuration. Measure it on two
branches and the difference is a cost regression -- "this PR makes your agent
12% more expensive" -- with zero API calls (the GitHub Action does exactly
this against the PR's base branch).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .effect import Recorder, ReplayExhausted, ReplayMismatch, _key


class _KeyCheckRecorder(Recorder):
    """A strict replayer that also notes every effect whose inputs changed."""

    def __init__(self, log):
        super().__init__(log=log, replay_until=len(log), allow_live=False)
        self.mismatches: "list[tuple[int, str]]" = []  # (seq, kind)
        self.model_input_chars = 0

    def run(self, kind, payload, fn, encode=lambda x: x, decode=lambda x: x):
        seq = self._cursor
        if seq < self.replay_until:
            entry = self.log[seq]
            if entry.kind == kind and _key([kind, payload]) != entry.key:
                self.mismatches.append((seq, kind))
            if kind == "model":
                try:
                    self.model_input_chars += len(json.dumps(payload, sort_keys=True, default=str))
                except (TypeError, ValueError):
                    pass
        return super().run(kind, payload, fn, encode=encode, decode=decode)


@dataclass
class Impact:
    """How one recorded trace is affected by the new configuration."""

    path: str
    verdict: str  # "unchanged" | "inputs-differ" | "structure-differs" | "outputs-differ"
    detail: str
    first_seq: "int | None" = None
    # What the recorded conversation costs in input tokens under THIS config
    # (estimated at ~4 chars/token; actual usage in live mode). None when the
    # replay diverges structurally -- the conversation can't be sized then.
    est_input_tokens: "int | None" = None

    @property
    def changed(self) -> bool:
        return self.verdict != "unchanged"

    def describe(self) -> str:
        where = f" (first at seq {self.first_seq})" if self.first_seq is not None else ""
        return f"{self.verdict:<16} {self.path}{where}\n    {self.detail}"

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "verdict": self.verdict,
            "detail": self.detail,
            "first_seq": self.first_seq,
            "est_input_tokens": self.est_input_tokens,
        }


def assess_trace(path: str, agent, live: bool = False) -> Impact:
    """Assess one saved trace against ``agent``'s current configuration."""
    from .trace import Run

    original = Run.load(path, agent=agent)
    if live:
        return _assess_live(path, original, agent)

    rec = _KeyCheckRecorder(original.log)
    try:
        agent.run(original.episodes, recorder=rec)
    except ReplayMismatch as e:
        return Impact(path, "structure-differs", str(e), first_seq=rec.cursor)
    except ReplayExhausted as e:
        return Impact(path, "structure-differs", str(e), first_seq=rec.cursor)
    if rec.cursor < len(original.log):
        return Impact(
            path,
            "structure-differs",
            f"new config finishes after {rec.cursor} of {len(original.log)} recorded effects",
            first_seq=rec.cursor,
        )
    est = rec.model_input_chars // 4 or None
    if rec.mismatches:
        seq, kind = rec.mismatches[0]
        return Impact(
            path,
            "inputs-differ",
            f"{len(rec.mismatches)} effect(s) see different inputs, starting with {kind!r}",
            first_seq=seq,
            est_input_tokens=est,
        )
    return Impact(
        path, "unchanged", "every recorded effect gets identical inputs", est_input_tokens=est
    )


def _assess_live(path: str, original, agent) -> Impact:
    from .diff import diff_logs

    fresh = agent.run(original.episodes)
    diff = diff_logs(original.log, fresh.log)
    spent = fresh.cost()["input_tokens"] or None  # live mode: actual, not estimated
    if fresh.output == original.output and diff.identical:
        return Impact(path, "unchanged", "re-run produced an identical trace", est_input_tokens=spent)
    if fresh.output == original.output:
        return Impact(
            path,
            "inputs-differ",
            "same final output via a different path",
            first_seq=diff.first_divergence,
            est_input_tokens=spent,
        )
    return Impact(
        path,
        "outputs-differ",
        f"output changed:\n      was: {original.output[:80]!r}\n      now: {fresh.output[:80]!r}",
        first_seq=diff.first_divergence,
        est_input_tokens=spent,
    )


def assess(paths: list[str], agent, live: bool = False) -> list[Impact]:
    """Assess a corpus of saved traces. See ``assess_trace``."""
    return [assess_trace(p, agent, live=live) for p in paths]


def report(impacts: list[Impact]) -> str:
    """Human-readable impact report."""
    changed = [i for i in impacts if i.changed]
    lines = [i.describe() for i in impacts]
    lines.append("")
    lines.append(f"{len(changed)} of {len(impacts)} recorded run(s) affected")
    sized = [i.est_input_tokens for i in impacts if i.est_input_tokens]
    if sized:
        lines.append(
            f"input volume under this config: ~{sum(sized):,} tokens across {len(sized)} run(s)"
        )
    return "\n".join(lines)


def to_json(impacts: list[Impact], agent=None) -> dict:
    """Machine-readable report, one branch's half of a cross-branch comparison."""
    import hashlib
    import json as _json

    sized = [i.est_input_tokens for i in impacts if i.est_input_tokens]
    system = getattr(agent, "system", "") if agent is not None else None
    # A per-tool schema hash so a *changed* (not just added) tool is detectable
    # across branches -- the risk-blame needs to point at what actually moved.
    tool_schemas = None
    if agent is not None:
        tool_schemas = {}
        for name, t in agent.tools.items():
            try:
                schema = t.schema() if hasattr(t, "schema") else {}
            except Exception:
                schema = {}
            tool_schemas[name] = hashlib.sha256(
                _json.dumps(schema, sort_keys=True, default=str).encode()
            ).hexdigest()[:12]
    return {
        "impacts": [i.to_dict() for i in impacts],
        "total": len(impacts),
        "affected": sum(1 for i in impacts if i.changed),
        "est_input_tokens": sum(sized) if sized else None,
        # The agent's tool inventory: comparing this across branches shows a
        # PR granting (or revoking) capabilities -- a risk signal no diff of
        # recorded runs can carry, since dry replay never consults the model.
        "agent_tools": sorted(agent.tools) if agent is not None else None,
        "tool_schemas": tool_schemas,
        "system_hash": hashlib.sha256((system or "").encode()).hexdigest()[:12]
        if system is not None else None,
    }


def blame(base: "dict", head: "dict") -> "list[str]":
    """Why did this change get riskier? Attribute each new risk to its cause.

    Reads two ``to_json()`` documents and points at the specific config move --
    a granted dangerous tool, an edited tool schema, a changed system prompt --
    so a reviewer sees the line, not just the danger.
    """
    from .risk import DANGEROUS, categories_for_names

    lines: list[str] = []
    b_tools, h_tools = base.get("agent_tools"), head.get("agent_tools")
    if b_tools is not None and h_tools is not None:
        label = {"network-egress": "network egress", "secret-read": "secret reads",
                 "code-exec": "shell execution", "fs-destructive": "destructive filesystem",
                 "fs-write": "file writes"}
        for t in sorted(set(h_tools) - set(b_tools)):
            cats = categories_for_names([t]) & DANGEROUS
            if cats:
                lines.append(f"grants **{', '.join(label.get(c, c) for c in sorted(cats))}** "
                             f"— you added the `{t}` tool")

    # A tool whose schema changed (same name, different shape).
    bs, hs = base.get("tool_schemas") or {}, head.get("tool_schemas") or {}
    changed = sorted(n for n in set(bs) & set(hs) if bs[n] != hs[n])
    for n in changed:
        lines.append(f"the schema for the `{n}` tool changed")

    if (base.get("system_hash") and head.get("system_hash")
            and base["system_hash"] != head["system_hash"]):
        lines.append("the system prompt changed")
    return lines


def tools_delta(base: "dict", head: "dict") -> str:
    """One-line capability verdict between two ``to_json()`` documents.

    A newly-granted tool that carries a dangerous capability (shell, network,
    secret read, destructive fs) is called out specifically -- that is a risk
    signal a diff of recorded outputs cannot see.
    """
    from .risk import DANGEROUS, categories_for_names

    b, h = base.get("agent_tools"), head.get("agent_tools")
    if b is None or h is None:
        return ""
    added, removed = sorted(set(h) - set(b)), sorted(set(b) - set(h))
    parts = []
    if added:
        dangerous = sorted(t for t in added if categories_for_names([t]) & DANGEROUS)
        if dangerous:
            cats = sorted({c for t in dangerous for c in categories_for_names([t])})
            parts.append("⚠️ grants DANGEROUS tool(s) [" + ", ".join(cats) + "]: "
                         + ", ".join(dangerous))
            safe = [t for t in added if t not in dangerous]
            if safe:
                parts.append("also new: " + ", ".join(safe))
        else:
            parts.append("grants the agent new tool(s): " + ", ".join(added))
    if removed:
        parts.append("removes tool(s): " + ", ".join(removed))
    return "this change " + "; ".join(parts) if parts else ""


# Capability weights for the security score: how much surface each dangerous
# capability opens. Deliberately simple and documented -- a reviewer should be
# able to recompute the number in their head.
_SCORE_WEIGHTS = {
    "fs-destructive": 18,
    "money-movement": 18,   # moving money is as grave as destroying data
    "secret-read": 15,
    "pii-access": 12,
    "network-egress": 12,
    "user-comm": 10,        # a message to a real user can't be unsent
    "code-exec": 10,
    "db-write": 8,
    "fs-write": 5,
    "browser-submit": 5,
}


def security_score(doc: "dict") -> "int | None":
    """A 0-100 agent security score from the tool inventory's capabilities.

    100 = no dangerous capability exposed; each capability class present in
    the inventory deducts its weight (once -- it measures surface, not count).
    Codecov-style: the absolute number matters less than the delta on a PR.
    """
    tools = doc.get("agent_tools")
    if tools is None:
        return None
    from .risk import categories_for_names

    present = set()
    for t in tools:
        present |= categories_for_names([t])
    return max(0, 100 - sum(_SCORE_WEIGHTS.get(c, 0) for c in present))


def score_line(base: "dict", head: "dict") -> str:
    """'Security score: 82 → 61' when the score moved; '' otherwise."""
    b, h = security_score(base), security_score(head)
    if b is None or h is None or b == h:
        return ""
    arrow = "⬇" if h < b else "⬆"
    return f"Security score: {b} → {h} {arrow}"


_RISK_LABELS = {
    "network-egress": "network egress", "secret-read": "secret reads",
    "code-exec": "shell execution", "fs-destructive": "destructive filesystem",
    "fs-write": "file writes", "money-movement": "money movement",
    "pii-access": "PII access", "user-comm": "user communication",
    "db-write": "database writes", "browser-submit": "browser submissions",
}


def risk_delta(base: "dict", head: "dict") -> str:
    """Per-category risk verdict between two ``to_json()`` docs.

    Groups newly-granted tools by risk category so a PR reviewer sees the
    capability change in safety terms -- "new network egress", "new shell
    execution" -- not just a tool name. Removed categories are noted too.
    """
    from .risk import DANGEROUS, categories_for_names

    b, h = base.get("agent_tools"), head.get("agent_tools")
    if b is None or h is None:
        return ""
    added, removed = set(h) - set(b), set(b) - set(h)
    label = _RISK_LABELS
    gained: dict = {}
    for t in added:
        for c in categories_for_names([t]) & DANGEROUS:
            gained.setdefault(c, []).append(t)
    lost = {c for t in removed for c in categories_for_names([t]) & DANGEROUS}
    lost -= set(gained)
    parts = [f"+{label.get(c, c)} ({', '.join(sorted(ts))})"
             for c, ts in sorted(gained.items())]
    parts += [f"-{label.get(c, c)}" for c in sorted(lost)]
    return "risk delta: " + ", ".join(parts) if parts else ""


def cost_delta(base: "list[dict]", head: "list[dict]") -> str:
    """One-line cost verdict between two ``to_json()['impacts']`` lists.

    Only runs sized on BOTH sides are compared (same estimator on both, so the
    delta is apples-to-apples even though each number is an estimate). Empty
    string when there is nothing comparable.
    """
    base_by = {i.get("path"): i.get("est_input_tokens") for i in base}
    pairs = [
        (base_by[i["path"]], i["est_input_tokens"])
        for i in head
        if i.get("est_input_tokens") and base_by.get(i.get("path"))
    ]
    if not pairs:
        return ""
    b = sum(x for x, _ in pairs)
    h = sum(y for _, y in pairs)
    runs = f"{len(pairs)} recorded run(s)"
    pct = (h - b) / b * 100.0
    if abs(pct) < 0.05:
        return f"input cost unchanged: ~{h:,} input tokens across {runs}"
    direction = "more" if pct > 0 else "less"
    return (
        f"this change makes your agent {abs(pct):.1f}% {direction} expensive in input tokens "
        f"(~{b:,} -> ~{h:,} across {runs})"
    )


def cost_delta_files(base_path: str, head_path: str) -> str:
    """``cost_delta`` over two files written by ``loom impact --json``."""
    with open(base_path) as f:
        base = json.load(f)
    with open(head_path) as f:
        head = json.load(f)
    return cost_delta(base.get("impacts", []), head.get("impacts", []))
