"""Loom Shield: an agent firewall at the proxy -- for agents you didn't build.

The recording proxy already sees every action an agent is about to take: tool
calls ride in the model's responses, before the client executes them. Shield
screens each one against firewall rules and rewrites the response when a call
is not allowed -- the agent never sees the tool call it wasn't permitted to
make, so it is never executed:

    loom proxy --deny 'Read(*.env*)' --confirm 'Bash(*)'
    export ANTHROPIC_BASE_URL=http://127.0.0.1:8788
    # ...run Claude Code as usual; don't dangerously skip permissions

Patterns are shell globs matched against the tool name (``Bash``) and against
its full signature ``name({"arg": "value"})`` -- so rules can target *what* is
called (``WebFetch``) or what it's called *with* (``Bash(*rm -rf*)``).

Precedence: **deny > allow > confirm** (anything unmatched is allowed). Note
this differs from ``Policy`` on purpose: at a firewall you want
``--confirm '*' --allow 'Read(*)'`` to mean "ask me about everything except
reads", so an allow rule bypasses confirm.

A ``confirm`` match holds the response open and files a pending approval --
answer it from another terminal (``loom approve <id>``), the proxy's control
endpoint, or a webhook-driven inbox. No decision within ``timeout`` seconds
means deny: the safe default. Every decision lands in the trace under
``shield_events``, and the blocked-call notice is part of the recorded model
response -- the audit trail replays like everything else.

Stdlib only, like the rest of the kernel.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from fnmatch import fnmatch

ALLOW = "allow"
CONFIRM = "confirm"
DENY = "deny"


def _signature(name: str, tool_input) -> str:
    try:
        args = json.dumps(tool_input, sort_keys=True, default=str)
    except (TypeError, ValueError):
        args = str(tool_input)
    return f"{name}({args})"


def _blocked_text(name: str, tool_input, reason: str) -> str:
    return (
        f"[loom shield] Blocked tool call {_signature(name, tool_input)} -- {reason} "
        "The call was not executed. Do not retry it; continue without it or explain "
        "to the user that the action was blocked by policy."
    )


@dataclass
class PendingApproval:
    """A confirm-rule hit waiting for a human decision."""

    id: str
    tool: str
    input: dict
    rule: str
    created: float = field(default_factory=time.time)
    event: threading.Event = field(default_factory=threading.Event)
    decision: str = ""  # "" until decided, then "approve" or "deny"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "tool": self.tool,
            "input": self.input,
            "rule": self.rule,
            "age_s": round(time.time() - self.created, 1),
        }


class Shield:
    """Screens wire responses; rewrites what the rules don't allow through."""

    def __init__(
        self,
        deny: "list[str] | tuple" = (),
        confirm: "list[str] | tuple" = (),
        allow: "list[str] | tuple" = (),
        timeout: float = 300.0,
        webhook: str = "",
        notify=None,
    ):
        self.deny = list(deny)
        self.confirm = list(confirm)
        self.allow = list(allow)
        self.timeout = timeout
        self.webhook = webhook
        self.notify = notify  # callable(PendingApproval) -> None, e.g. a console printer
        self.pending: dict[str, PendingApproval] = {}
        self.lock = threading.Lock()

    # -- rules ------------------------------------------------------------

    def classify(self, name: str, tool_input) -> "tuple[str, str]":
        """First matching action wins, checked deny > allow > confirm."""
        sig = _signature(name, tool_input)
        for patterns, action in ((self.deny, DENY), (self.allow, ALLOW), (self.confirm, CONFIRM)):
            for p in patterns:
                if fnmatch(name, p) or fnmatch(sig, p):
                    return action, p
        return ALLOW, ""

    # -- approval inbox ----------------------------------------------------

    def pending_list(self) -> "list[dict]":
        with self.lock:
            return [p.to_dict() for p in self.pending.values() if not p.event.is_set()]

    def decide_pending(self, pending_id: str, approve: bool) -> bool:
        with self.lock:
            p = self.pending.get(pending_id)
            if p is None or p.event.is_set():
                return False
            p.decision = "approve" if approve else "deny"
            p.event.set()
            return True

    def _post_webhook(self, p: PendingApproval) -> None:
        import urllib.request

        payload = {
            "event": "loom.shield.confirm",
            "id": p.id,
            "tool": p.tool,
            "input": p.input,
            "rule": p.rule,
            "text": f"loom shield: approve? [{p.id}] {_signature(p.tool, p.input)}",
        }
        req = urllib.request.Request(
            self.webhook,
            data=json.dumps(payload).encode(),
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=10).close()
        except OSError:
            pass  # the inbox is best-effort; the console + control endpoint remain

    def _await_approval(self, name: str, tool_input, rule: str) -> "tuple[bool, str]":
        """File a pending approval and block until decided or timed out."""
        p = PendingApproval(id=uuid.uuid4().hex[:6], tool=name, input=tool_input, rule=rule)
        with self.lock:
            self.pending[p.id] = p
        if self.notify is not None:
            try:
                self.notify(p)
            except Exception:
                pass
        if self.webhook:
            threading.Thread(target=self._post_webhook, args=(p,), daemon=True).start()
        decided = p.event.wait(self.timeout)
        with self.lock:
            self.pending.pop(p.id, None)
        if not decided:
            return False, "timeout"
        return p.decision == "approve", "operator"

    # -- screening ---------------------------------------------------------

    def screen(self, response: dict) -> "tuple[dict, list[dict]]":
        """Return (response the client may see, shield events). May block on confirm."""
        if "choices" in response:
            return self._screen_openai(response)
        return self._screen_anthropic(response)

    def _judge(self, name: str, tool_input) -> "tuple[bool, dict | None]":
        """Decide one tool call. Returns (allowed, event-or-None)."""
        action, rule = self.classify(name, tool_input)
        base = {"ts": time.time(), "tool": name, "input": tool_input, "rule": rule}
        if action == DENY:
            return False, {**base, "action": "deny", "via": "rule"}
        if action == CONFIRM:
            approved, via = self._await_approval(name, tool_input, rule)
            return approved, {**base, "action": "approve" if approved else "deny", "via": via}
        return True, None

    def _screen_anthropic(self, response: dict) -> "tuple[dict, list[dict]]":
        events: list[dict] = []
        content, changed = [], False
        for b in response.get("content", []):
            if isinstance(b, dict) and b.get("type") == "tool_use":
                name, tool_input = b.get("name", ""), b.get("input", {})
                allowed, event = self._judge(name, tool_input)
                if event:
                    events.append(event)
                if not allowed:
                    reason = _reason(event)
                    content.append({"type": "text", "text": _blocked_text(name, tool_input, reason)})
                    changed = True
                    continue
            content.append(b)
        if not changed:
            return response, events
        out = {**response, "content": content}
        if not any(isinstance(b, dict) and b.get("type") == "tool_use" for b in content):
            out["stop_reason"] = "end_turn"
        return out, events

    def _screen_openai(self, response: dict) -> "tuple[dict, list[dict]]":
        events: list[dict] = []
        choices = response.get("choices") or []
        if not choices:
            return response, events
        message = choices[0].get("message", {}) or {}
        kept, notices, changed = [], [], False
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function", {}) or {}
            name = fn.get("name", "")
            try:
                tool_input = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                tool_input = {"_raw": fn.get("arguments", "")}
            allowed, event = self._judge(name, tool_input)
            if event:
                events.append(event)
            if allowed:
                kept.append(tc)
            else:
                notices.append(_blocked_text(name, tool_input, _reason(event)))
                changed = True
        if not changed:
            return response, events
        new_message = {**message, "content": "\n".join(filter(None, [message.get("content") or ""] + notices))}
        if kept:
            new_message["tool_calls"] = kept
        else:
            new_message.pop("tool_calls", None)
        new_choice = {**choices[0], "message": new_message}
        if not kept:
            new_choice["finish_reason"] = "stop"
        return {**response, "choices": [new_choice] + choices[1:]}, events


def _reason(event: "dict | None") -> str:
    via = (event or {}).get("via", "rule")
    if via == "rule":
        return f"matched deny rule '{(event or {}).get('rule', '')}'."
    if via == "timeout":
        return "approval timed out."
    return "denied by the operator."
