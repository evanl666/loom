"""The Browser Pack -- debug agents that click, type, and submit.

Opt-in (``import loom.packs.browser`` registers it). A browser agent's side
effects are the most visible of any domain -- a submitted form is an email
sent, an order placed, a record created somewhere you don't control. What it
teaches Loom:

  owns        click / fill / type / submit / navigate / screenshot tools
  capabilities  submits and clicks are ``browser_submit`` (an external side
              effect); navigation is ``network``; screenshots/DOM reads read
  state_diff  "navigated to <url>", "submitted <form/selector>" -- and when
              the recording captured DOM snapshots (``dom_before`` /
              ``dom_after`` keys in the tool result), a real before/after
              size delta with both snapshots in the detail
  undo        navigation is trivially reversible (go back); a submitted form
              is NOT -- the plan says so instead of pretending

The pack reads recorded traffic only; it does not drive a browser.
"""

from __future__ import annotations

from fnmatch import fnmatchcase as fnmatch

from ..action import Action, StateDiff
from . import Pack, UndoPlan, register

_SUBMIT = ["click*", "*_click", "submit*", "*form_submit*", "press*", "tap*"]
_FILL = ["fill*", "type*", "*_type", "select_option*", "check*", "upload*"]
_NAV = ["navigate*", "goto*", "go_to*", "open_url*", "*_navigate", "back", "forward",
        "reload"]
_READ = ["screenshot*", "*_screenshot", "get_dom*", "*dom_snapshot*", "get_text*",
         "read_page*", "*accessibility_tree*"]
_ALL = _SUBMIT + _FILL + _NAV + _READ


def _matches(name: str, globs: "list[str]") -> bool:
    return any(fnmatch(name.lower(), g) for g in globs)


def _target(action: Action) -> str:
    """The url/selector/element the action aimed at, best-effort."""
    if isinstance(action.input, dict):
        for k in ("url", "selector", "element", "target", "text", "name"):
            v = action.input.get(k)
            if isinstance(v, str) and v:
                return v
    return ""


class BrowserPack(Pack):
    name = "browser"

    def owns(self, action: Action) -> bool:
        return action.type == "call" and _matches(action.tool, _ALL)

    def capabilities(self, name: str, tool_input) -> "set[str]":
        if _matches(name, _SUBMIT):
            return {"browser_submit"}
        if _matches(name, _NAV):
            return {"network"}
        if _matches(name, _FILL):
            return {"write"}
        if _matches(name, _READ):
            return {"read"}
        return set()

    def state_diff(self, action: Action, trace: dict) -> "StateDiff | None":
        raw = action.observation.raw if action.observation is not None else None
        detail = None
        summary_extra = ""
        if isinstance(raw, dict) and ("dom_before" in raw or "dom_after" in raw):
            before, after = str(raw.get("dom_before", "")), str(raw.get("dom_after", ""))
            detail = {"dom_before": before[:20_000], "dom_after": after[:20_000]}
            summary_extra = f" (DOM {len(before)} -> {len(after)} chars)"
        target = _target(action)
        if _matches(action.tool, _NAV):
            return StateDiff("dom", f"navigated to {target or '?'}" + summary_extra, detail)
        if _matches(action.tool, _SUBMIT):
            return StateDiff("dom", f"submitted {target or 'a form'}" + summary_extra, detail)
        if _matches(action.tool, _FILL):
            return StateDiff("dom", f"filled {target or 'a field'}" + summary_extra, detail)
        return None

    def undo(self, action: Action, trace: dict) -> "UndoPlan | None":
        if _matches(action.tool, _NAV):
            return UndoPlan("revert", "navigate back", ["browser.back()"])
        if _matches(action.tool, _FILL):
            return UndoPlan("revert", "clear the field / reload the page",
                            ["reload the page (unsubmitted input is discarded)"])
        if _matches(action.tool, _SUBMIT):
            return UndoPlan(
                "noop", "a submitted form cannot be unsubmitted -- compensate on the "
                        "receiving system (cancel the order, delete the record)",
                reversible=False)
        return None

    def replay_hint(self, action: Action) -> str:
        return "restore the browser session (same page, cookies, form state), then replay"


register(BrowserPack())
