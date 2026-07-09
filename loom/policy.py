"""Policy at the Effect boundary: decide what an agent may do BEFORE it does it.

Every tool call flows through one chokepoint, so one policy object controls
them all -- no per-tool wrappers:

    agent = Agent(model=..., tools=[...], policy=Policy(
        allow=["read_*", "search_*"],   # run freely
        confirm=["delete_*", "send_*"], # pause for human approval first
        deny=["drop_db"],               # blocked outright, never executed
        budget_tokens=50_000,           # hard spend cap; exceeding stops the run
    ))

    run = agent.run("clean up old data", ...)
    run.intents()   # what the agent did / tried to do, with statuses

``dry_run=True`` stubs every non-allowlisted tool with a "would call ..."
marker instead of executing it -- see what an agent WOULD do before letting it.
Approvals reuse the human-in-the-loop effect: with no ``on_human`` handler the
run pauses and ``resume("yes")`` continues it; every decision is recorded in
the trace, so approved runs replay deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatchcase as fnmatch

ALLOW = "allow"
CONFIRM = "confirm"
DENY = "deny"
STUB = "stub"  # dry-run: record the intent, execute nothing


def _matches(name: str, patterns: list[str]) -> bool:
    return any(fnmatch(name, p) for p in patterns)


@dataclass
class Policy:
    """Tool-call rules plus a hard token budget.

    Precedence: deny > confirm > allow. In ``dry_run`` mode, allowlisted tools
    still execute (typically read-only ones the agent needs to plan) and
    everything else is stubbed. Tools matching nothing fall back to ``default``.
    """

    allow: list[str] = field(default_factory=list)
    confirm: list[str] = field(default_factory=list)
    deny: list[str] = field(default_factory=list)
    default: str = ALLOW
    dry_run: bool = False
    budget_tokens: "int | None" = None

    def decide(self, tool_name: str) -> str:
        if _matches(tool_name, self.deny):
            return DENY
        if _matches(tool_name, self.confirm):
            return CONFIRM
        if _matches(tool_name, self.allow):
            return ALLOW
        if self.dry_run:
            return STUB
        return self.default


def affirmative(answer: str) -> bool:
    """Interpret a human approval answer. Conservative: unclear means no."""
    a = str(answer).strip().lower()
    return a.startswith(("y", "approve")) or a in {"ok", "true", "1", "approved"}
