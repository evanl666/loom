"""Packs: teach Loom about a *domain* of agent without changing the core.

Loom's core is domain-neutral: the Effect boundary captures actions, the
Action schema describes them, Shield gates them. What the core can't know is
how a *specific world* works -- how to diff a database, screenshot a browser,
or undo a CRM write. That knowledge lives in a **Pack**.

A Pack is a small plug-in with optional hooks; implement only what your domain
needs:

    class SqlPack(Pack):
        name = "sql"
        def owns(self, action):        # which actions are mine?
            return "database_write" in action.capabilities
        def capabilities(self, name, tool_input):   # domain risk hints
            return {"database_write"} if "insert" in name.lower() else set()
        def state_diff(self, action, trace):         # how the world changed
            return StateDiff("database", f"+{action.observation.raw['rows']} rows")
        def undo(self, action, trace):               # how to reverse it
            return UndoPlan("compensate", "DELETE the inserted rows", [...])

Register it (``register(SqlPack())``) and every Action the debugger builds is
enriched: domain capabilities merged in, a StateDiff attached, an undo plan
available. The Coding Pack ships built-in; Browser / SQL / Support packs plug
in the same way -- which is what makes Loom a debugger for agents it didn't
build.

The four hooks map to the four things a debugger must do per domain:

    capture       turn a raw domain event into an Action (custom runtimes)
    state_diff    compute how the outside world changed
    undo          reverse an action, or describe a compensating one
    replay_hint   how to restore this domain's state before re-running
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..action import Action, StateDiff

# A throwaway action for hooks (like the default restore) that need to reuse a
# pack's advisory replay_hint but have no specific action in hand.
_STUB_ACTION = Action(step=-1, depth=0, type="call")


@dataclass
class UndoPlan:
    """How to reverse an action (or compensate for it if it's irreversible).

    ``kind`` is "revert" (restore prior state), "compensate" (a new action
    that offsets it, e.g. a refund reversing a charge), or "noop" (nothing to
    undo). ``commands`` are the concrete steps; ``apply`` optionally executes
    them. ``reversible`` is False when the effect genuinely cannot be taken
    back (an email already sent) and only a compensating action is possible.
    """

    kind: str
    summary: str
    commands: list[str] = field(default_factory=list)
    reversible: bool = True
    apply: Any = None  # optional callable() -> str

    def to_dict(self) -> dict:
        return {"kind": self.kind, "summary": self.summary,
                "commands": self.commands, "reversible": self.reversible}


@dataclass
class RestorePlan:
    """How to put a domain's WORLD back before replaying/forking a run.

    Replaying a trace rewinds the model and tool log deterministically, but not
    the outside world -- the files, database rows, or browser session the agent
    changed. A pack describes (and, where it can, executes) the restore:
    ``executable`` is True only when ``commands`` will actually reproduce the
    snapshot; otherwise the plan is advisory (a human runs the steps).
    """

    kind: str            # "git" | "manual" | "noop"
    summary: str
    commands: list[str] = field(default_factory=list)
    executable: bool = False

    def to_dict(self) -> dict:
        return {"kind": self.kind, "summary": self.summary,
                "commands": self.commands, "executable": self.executable}


class Pack:
    """Base class for a domain pack. Every hook is optional (safe defaults)."""

    name: str = "pack"

    def snapshot(self, trace: dict) -> "dict | None":
        """Capture the world-state reference this domain needs to restore later
        (e.g. the git commit a coding run started from). None when the trace
        didn't record enough to pin the world."""
        return None

    def restore(self, snapshot: "dict | None") -> "RestorePlan | None":
        """How to put the world back to ``snapshot`` before replaying. Default:
        fall back to the advisory ``replay_hint`` -- honest that the restore is
        manual for domains Loom doesn't itself capture."""
        hint = self.replay_hint(_STUB_ACTION)
        return RestorePlan("manual", hint, executable=False) if hint else None

    def owns(self, action: Action) -> bool:
        """Does this pack handle ``action``? Default: no."""
        return False

    def capture(self, raw: Any) -> "Action | None":
        """Turn a raw domain event into an Action (for non-wire runtimes).

        Wire agents (Anthropic/OpenAI) are captured by the proxy/effect
        boundary already; override this only for a custom runtime (a browser
        driver, an RPA tool) that doesn't speak a recorded model API."""
        return None

    def capabilities(self, name: str, tool_input: Any) -> "set[str]":
        """Extra capability hints this domain knows (merged with the core)."""
        return set()

    def state_diff(self, action: Action, trace: dict) -> "StateDiff | None":
        """How the outside world changed because of ``action`` (or None)."""
        return None

    def undo(self, action: Action, trace: dict) -> "UndoPlan | None":
        """How to reverse ``action`` (or compensate for it), or None."""
        return None

    def replay_hint(self, action: Action) -> str:
        """How to restore this domain's state before replaying (advisory)."""
        return ""

    def debugger_panels(self, action: Action, trace: dict) -> "list[dict]":
        """Extra panels this domain contributes to the `loom debug` UI for
        ``action`` -- e.g. a SQL pack's query plan, a browser pack's screenshot,
        a support pack's customer-impact card. Each panel is
        {"title": str, "text"|"code": str}. Default: none, so the debugger is a
        platform packs can extend, not a fixed UI."""
        return []

    def safe_runtime(self) -> str:
        """How to run this domain SAFELY while debugging -- a sandboxed world
        where the agent's actions can't cause real harm (a dry-run DB, a
        no-submit browser, a fake customer tenant). Advisory text; empty when
        the pack has no specific guidance."""
        return ""


# -- registry --------------------------------------------------------------

_REGISTRY: "list[Pack]" = []


def register(pack: Pack) -> None:
    """Register a pack (idempotent by name -- re-registering replaces)."""
    global _REGISTRY
    _REGISTRY = [p for p in _REGISTRY if p.name != pack.name] + [pack]


def unregister(name: str) -> None:
    global _REGISTRY
    _REGISTRY = [p for p in _REGISTRY if p.name != name]


def packs() -> "list[Pack]":
    """All registered packs (Coding Pack first, since it's built in)."""
    return list(_REGISTRY)


def pack_for(action: Action) -> "Pack | None":
    """The owning pack -- most recently registered wins, so an opt-in domain
    pack (sql, browser...) takes precedence over the built-in Coding Pack for
    actions both could claim (e.g. a DROP TABLE looks destructive to both)."""
    for p in reversed(_REGISTRY):
        if p.owns(action):
            return p
    return None


def enrich(action_list: "list[Action]", trace: dict) -> "list[Action]":
    """Run every registered pack over the Actions, in place.

    Merges each owning pack's domain capabilities and attaches its StateDiff.
    Returns the same list for chaining. Safe to call with no packs registered
    (it's a no-op), so the core never depends on any pack existing.
    """
    if not _REGISTRY:
        return action_list
    for a in action_list:
        # Most recently registered first (same precedence as pack_for), so a
        # domain pack's StateDiff wins over the built-in coding fallback.
        for p in reversed(_REGISTRY):
            if not p.owns(a):
                continue
            extra = p.capabilities(a.tool, a.input or {})
            if extra:
                a.capabilities = sorted(set(a.capabilities) | extra)
            if a.state_diff is None:
                sd = p.state_diff(a, trace)
                if sd is not None:
                    a.state_diff = sd
    return action_list


def undo_plan(action: Action, trace: dict) -> "UndoPlan | None":
    """The undo plan from whichever pack owns ``action`` (or None)."""
    p = pack_for(action)
    return p.undo(action, trace) if p else None


def restore_plans(action_list: "list[Action]", trace: dict) -> "list[tuple[str, RestorePlan]]":
    """(pack_name, RestorePlan) for every domain a run touched -- how to put
    each world back before replaying/forking. One plan per pack, in the order
    the domains first appear. Used by ``loom fork``."""
    seen: set = set()
    out: list[tuple[str, RestorePlan]] = []
    for a in action_list:
        p = pack_for(a)
        if p is None or p.name in seen:
            continue
        seen.add(p.name)
        plan = p.restore(p.snapshot(trace))
        if plan is not None:
            out.append((p.name, plan))
    return out


def install_builtin() -> None:
    """Register every built-in domain pack (coding, sql, browser, support).

    Pack-aware surfaces (``loom undo --plan``, ``loom fork``, ``Run.undo_plans``,
    the incident report) call this so plans and hints cover every domain out of
    the box; library users composing their own registry simply don't.

    Registers explicit instances (idempotent by name) rather than relying on
    import side effects -- an import happens once per process, so a registry
    cleared later (tests, custom setups) could never get the packs back."""
    from . import browser, coding, sql, support

    register(coding.CodingPack())
    register(sql.SqlPack())
    register(browser.BrowserPack())
    register(support.SupportPack())
