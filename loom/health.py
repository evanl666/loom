"""Context-rot detection -- and repairs you can actually test.

Context rot (stale, bloated, unused, or duplicated context) is the leading
cause of agent failures. Because Loom records every effect, a finished run can
be examined after the fact:

    report = run.checkup()          # findings: oversized / unused / duplicate
    print(report.summary())

And because runs can fork, every finding doubles as an *experiment*: redact the
suspect item, re-run only the divergent tail, and see whether the answer
improves. ``run.heal(check)`` automates that loop -- diagnosis to verified fix.

Findings are heuristics on the recorded trace; they point at suspects, and the
fork is what convicts them.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable

from .context import estimate_tokens
from .effect import EffectEntry
from .providers.base import ModelResponse

# Words worth tracking for usage analysis: long enough to be distinctive.
_WORD = re.compile(r"[A-Za-z0-9]{5,}")

# An item is "oversized" when it is both large and a big share of the context.
OVERSIZED_MIN_TOKENS = 200
OVERSIZED_MIN_SHARE = 0.25

# Two items are "duplicates" when their word sets overlap this much.
DUPLICATE_JACCARD = 0.9


def _words(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text)}


@dataclass
class Finding:
    """One suspected piece of context rot."""

    kind: str  # "oversized" | "unused" | "duplicate"
    severity: str  # "warn" | "high"
    message: str
    content: str  # the offending item's content
    tokens: int
    seq: "int | None" = None  # effect seq when the item came from the log

    def describe(self) -> str:
        return f"[{self.severity}] {self.kind}: {self.message}"


@dataclass
class HealthReport:
    """The result of a context checkup."""

    findings: list[Finding]
    total_tokens: int

    @property
    def ok(self) -> bool:
        return not self.findings

    def summary(self) -> str:
        if self.ok:
            return f"context looks healthy ({self.total_tokens} tokens, no findings)"
        lines = [f"{len(self.findings)} finding(s) in {self.total_tokens} tokens of context:"]
        for f in self.findings:
            snippet = (f.content[:60] + "...") if len(f.content) > 60 else f.content
            lines.append(f"  {f.describe()}")
            lines.append(f"      {snippet!r}")
        return "\n".join(lines)

    def experiments(self) -> "tuple[list[str], list[Callable]]":
        """Sweep-ready repairs: one context edit per distinct finding.

        Each edit REDACTS the suspect item (replaces its content) rather than
        dropping it, so tool_use/tool_result pairing stays valid for real APIs.
        Feed the result to ``Run.sweep`` or let ``Run.heal`` drive it.
        """
        labels: list[str] = []
        variants: list[Callable] = []
        seen_prefixes: set[str] = set()
        for i, f in enumerate(self.findings):
            prefix = f.content[:48]
            if prefix in seen_prefixes:
                continue  # two findings about the same item -> one experiment
            seen_prefixes.add(prefix)
            labels.append(f"redact-{f.kind}-{i}")
            variants.append(_redact_edit(prefix))
        return labels, variants


def _redact_edit(prefix: str) -> Callable:
    """An edit that redacts every context item starting with ``prefix``."""

    def edit(ctx) -> None:
        for item in ctx.items:
            if item.content.startswith(prefix):
                item.content = "[redacted: flagged by checkup]"
                item.tokens = 8

    return edit


def analyze(episodes: list[str], log: list[EffectEntry]) -> HealthReport:
    """Inspect a recorded run's context for rot. Works on live or loaded traces."""
    # Flatten the run into (kind, text, tokens, seq) items in order.
    items: list[tuple] = [("user", ep, estimate_tokens(ep), None) for ep in episodes]
    for e in log:
        if e.kind == "model":
            text = ModelResponse.from_dict(e.result).text
            items.append(("assistant", text, estimate_tokens(text), e.seq))
        else:  # tool:* and human results
            text = e.result if isinstance(e.result, str) else json.dumps(e.result)
            items.append((e.kind, text, estimate_tokens(text), e.seq))

    total = sum(t for _, _, t, _ in items) or 1
    findings: list[Finding] = []

    # 1. Oversized tool results: one item hogging the window degrades attention.
    for kind, text, tokens, seq in items:
        if not kind.startswith("tool:"):
            continue
        if tokens >= OVERSIZED_MIN_TOKENS and tokens / total >= OVERSIZED_MIN_SHARE:
            findings.append(
                Finding(
                    "oversized",
                    "high",
                    f"{kind} result is {tokens} tokens ({tokens * 100 // total}% of context)",
                    text,
                    tokens,
                    seq,
                )
            )

    # 2. Unused tool results: paid for, never referenced by any later answer.
    for idx, (kind, text, tokens, seq) in enumerate(items):
        if not kind.startswith("tool:"):
            continue
        distinctive = _words(text)
        if len(distinctive) < 3:
            continue  # too little signal to judge
        later_answers = " ".join(t for k, t, _, _ in items[idx + 1 :] if k == "assistant")
        if not distinctive & _words(later_answers):
            findings.append(
                Finding(
                    "unused",
                    "warn",
                    f"{kind} result ({tokens} tokens) never referenced by any later answer",
                    text,
                    tokens,
                    seq,
                )
            )

    # 3. Near-duplicate items: the same content paid for twice.
    seen: list[tuple[str, set[str]]] = []
    for kind, text, tokens, seq in items:
        if kind == "assistant":
            continue
        ws = _words(text)
        if len(ws) < 5:
            continue
        for earlier_kind, earlier_ws in seen:
            jaccard = len(ws & earlier_ws) / len(ws | earlier_ws)
            if jaccard >= DUPLICATE_JACCARD:
                findings.append(
                    Finding(
                        "duplicate",
                        "warn",
                        f"{kind} content nearly identical to earlier {earlier_kind}",
                        text,
                        tokens,
                        seq,
                    )
                )
                break
        seen.append((kind, ws))

    return HealthReport(findings=findings, total_tokens=total)
