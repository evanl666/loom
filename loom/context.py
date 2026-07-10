"""Context: the model's working memory, with provenance and a token budget.

Every piece of context carries where it came from (``source``) so a run can be
inspected for context rot -- the single biggest cause of agent failures. When a
token budget is set, the oldest unpinned items are trimmed first, and tool-result
pairs are never left dangling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .providers.base import ModelResponse, ToolCall


def estimate_tokens(text: str) -> int:
    """Cheap, dependency-free token estimate (~4 chars/token)."""
    return max(1, len(text) // 4)


@dataclass
class Item:
    """One message in the context window, plus provenance metadata."""

    role: str  # "user" | "assistant" | "tool"
    content: str
    source: str  # "user" | "model" | "tool:<name>"
    pinned: bool = False
    tokens: int = 0
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str = ""
    name: str = ""

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "source": self.source,
            "pinned": self.pinned,
            "tokens": self.tokens,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "tool_call_id": self.tool_call_id,
            "name": self.name,
        }

    @staticmethod
    def from_dict(d: dict) -> "Item":
        return Item(
            role=d.get("role", "user"),
            content=d.get("content", ""),
            source=d.get("source", ""),
            pinned=d.get("pinned", False),
            tokens=d.get("tokens", 0),
            tool_calls=[ToolCall.from_dict(t) for t in d.get("tool_calls", [])],
            tool_call_id=d.get("tool_call_id", ""),
            name=d.get("name", ""),
        )


class Context:
    """An ordered list of provenance-tracked items plus a system prompt."""

    def __init__(self, system: str = "", budget: "int | None" = None):
        self.system = system
        self.budget = budget
        self.items: list[Item] = []

    # -- mutation ---------------------------------------------------------

    def add_user(self, text: str, source: str = "user", pinned: bool = False) -> None:
        self.items.append(
            Item("user", text, source, pinned=pinned, tokens=estimate_tokens(text))
        )

    def add_assistant(self, resp: ModelResponse) -> None:
        self.items.append(
            Item(
                "assistant",
                resp.text,
                "model",
                tokens=estimate_tokens(resp.text),
                tool_calls=list(resp.tool_calls),
            )
        )

    def add_tool_result(self, tool_call_id: str, name: str, content: str) -> None:
        content = str(content)
        self.items.append(
            Item(
                "tool",
                content,
                f"tool:{name}",
                tokens=estimate_tokens(content),
                tool_call_id=tool_call_id,
                name=name,
            )
        )

    # -- views ------------------------------------------------------------

    def messages(self) -> list[dict]:
        """Neutral message list for a provider, after budget trimming."""
        items = self._trim()
        out: list[dict] = []
        for it in items:
            if it.role == "user":
                out.append({"role": "user", "content": it.content})
            elif it.role == "assistant":
                msg: dict = {"role": "assistant", "content": it.content}
                if it.tool_calls:
                    msg["tool_calls"] = it.tool_calls
                out.append(msg)
            elif it.role == "tool":
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": it.tool_call_id,
                        "name": it.name,
                        "content": it.content,
                    }
                )
        return out

    def provenance(self) -> list[dict]:
        """Where every item came from and how many tokens it costs -- for audits."""
        return [
            {"role": it.role, "source": it.source, "tokens": it.tokens, "pinned": it.pinned}
            for it in self.items
        ]

    def total_tokens(self) -> int:
        return sum(it.tokens for it in self.items)

    # -- budget -----------------------------------------------------------

    def _trim(self) -> list[Item]:
        """Drop the oldest unpinned items until under budget, keeping tool pairs sane."""
        if self.budget is None or self.total_tokens() <= self.budget:
            return self._prune_orphan_tools(list(self.items))

        items = list(self.items)
        # Walk from the front dropping unpinned items, then prune any tool result
        # left without its preceding tool_use -- dropping the assistant that made a
        # call orphans its result, and a pinned item ahead of the pair means the
        # orphan is stranded mid-list, not at the front (providers reject that).
        while items and sum(i.tokens for i in items) > self.budget:
            drop_idx = next((i for i, it in enumerate(items) if not it.pinned), None)
            if drop_idx is None:
                break  # everything left is pinned
            del items[drop_idx]
            items = self._prune_orphan_tools(items)
        return items

    @staticmethod
    def _prune_orphan_tools(items: list[Item]) -> list[Item]:
        """Drop tool results whose matching tool_use assistant is no longer present."""
        out: list[Item] = []
        live: set[str] = set()  # tool_call ids offered by the most recent assistant
        for it in items:
            if it.role == "assistant":
                live = {tc.id for tc in it.tool_calls}
                out.append(it)
            elif it.role == "tool":
                if it.tool_call_id in live:
                    out.append(it)  # else: orphan, drop it
            else:  # user turn closes the open tool_use window
                live = set()
                out.append(it)
        return out
