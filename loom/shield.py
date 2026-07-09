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

**Sequence rules** constrain runs, not just single calls -- real incidents are
sequences ("it read the .env, THEN posted somewhere"):

    loom record --rule 'after Read(*.env*): deny WebFetch*, deny Bash(*curl*)' -- ...
    loom record --rule 'taint sk-ant-*: confirm *' -- ...

``after <call-pattern>:`` arms when a matching tool call is allowed through;
``taint <text-pattern>:`` arms when any tool RESULT matches (the proxy sees
results in the next request). Once armed, the consequences (``deny <pattern>``
/ ``confirm <pattern>``, comma-separated) apply to every later call in the
session -- and a sequence deny beats a static allow, because "this run touched
secrets" outranks "reads are generally fine". Sequence confirms never
auto-approve via the trust ratchet: a tripped tripwire always gets a human.

Stdlib only, like the rest of the kernel.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from fnmatch import fnmatchcase as fnmatch

ALLOW = "allow"
CONFIRM = "confirm"
DENY = "deny"


def _signature(name: str, tool_input) -> str:
    try:
        args = json.dumps(tool_input, sort_keys=True, default=str)
    except (TypeError, ValueError):
        args = str(tool_input)
    return f"{name}({args})"


def _normalize(sig: str) -> str:
    """Collapse whitespace runs so ``rm   -rf`` still matches ``*rm -rf*``."""
    return re.sub(r"\s+", " ", sig)


def _blocked_text(name: str, tool_input, reason: str) -> str:
    return (
        f"[loom shield] Blocked tool call {_signature(name, tool_input)} -- {reason} "
        "The call was not executed. Do not retry it; continue without it or explain "
        "to the user that the action was blocked by policy."
    )


@dataclass
class SequenceRule:
    """A temporal tripwire: once ``trigger`` fires, ``consequences`` apply.

    ``after`` triggers on an allowed tool CALL matching the pattern; ``taint``
    triggers on a tool RESULT whose text matches. Consequences are
    ``(action, call-pattern)`` pairs applied to every subsequent call.
    """

    trigger_type: str  # "after" | "taint"
    trigger: str
    consequences: "list[tuple[str, str]]"
    raw: str
    triggered: bool = False
    evidence: str = ""


def parse_sequence_rule(raw: str) -> SequenceRule:
    """Parse ``'after Read(*.env*): deny WebFetch*, confirm Bash*'``."""
    head, sep, tail = raw.partition(":")
    head = head.strip()
    for trigger_type in ("after", "taint"):
        if head.startswith(trigger_type + " "):
            trigger = head[len(trigger_type) + 1 :].strip()
            break
    else:
        raise ValueError(
            f"sequence rule must start with 'after <pattern>:' or 'taint <pattern>:', got {raw!r}"
        )
    if not sep or not trigger:
        raise ValueError(f"sequence rule needs '<trigger>: <consequences>', got {raw!r}")
    consequences = []
    for part in tail.split(","):
        action, _, pattern = part.strip().partition(" ")
        if action not in (DENY, CONFIRM) or not pattern.strip():
            raise ValueError(
                f"consequence must be 'deny <pattern>' or 'confirm <pattern>', got {part.strip()!r}"
            )
        consequences.append((action, pattern.strip()))
    return SequenceRule(trigger_type, trigger, consequences, raw)


def _call_matches(pattern: str, name: str, tool_input) -> bool:
    if pattern.startswith("cap:"):  # capability patterns work in sequence rules too
        from .capabilities import matches_cap

        return matches_cap(pattern, name, tool_input)
    sig = _signature(name, tool_input)
    return fnmatch(name, pattern) or fnmatch(sig, pattern) or fnmatch(_normalize(sig), pattern)


def _text_matches(text: str, pattern: str) -> bool:
    """Taint patterns are SEARCHED in the text: 'sk-ant-*' hits anywhere."""
    return fnmatch(_normalize(text), f"*{pattern}*")


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
    decided_by: str = ""  # who decided (operator identity), for the audit trail

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
        default: str = ALLOW,
        timeout: float = 300.0,
        webhook: str = "",
        notify=None,
        judge=None,
        judge_threshold: float = 0.7,
        trust=None,
        trust_after: int = 0,
        sequence: "list[str] | tuple" = (),
        sign_key: "bytes | None" = None,
        approvers: "dict[str, list[str]] | None" = None,
    ):
        if default not in (ALLOW, CONFIRM, DENY):
            raise ValueError(f"default must be allow/confirm/deny, not {default!r}")
        self.deny = list(deny)
        self.confirm = list(confirm)
        self.allow = list(allow)
        self.default = default
        self.timeout = timeout
        self.webhook = webhook
        self.notify = notify  # callable(PendingApproval) -> None, e.g. a console printer
        # LLM-judge: a cheap model risk-scores calls no explicit rule matched;
        # a score >= judge_threshold escalates to the confirm flow. The verdict
        # is recorded in shield_events either way -- auditable intelligence.
        self.judge = judge
        self.judge_threshold = judge_threshold
        # Trust ratchet: after `trust_after` consecutive operator approvals a
        # tool's confirms auto-approve (via="ratchet"); any deny demotes it.
        self.trust = trust  # a TrustLedger
        self.trust_after = trust_after
        # Temporal tripwires; state is per-Shield, i.e. per proxy session.
        self.sequence = [
            r if isinstance(r, SequenceRule) else parse_sequence_rule(r) for r in sequence
        ]
        # Signed decisions: an HMAC over each operator decision makes the audit
        # trail tamper-PROOF (verify with the same key). Approver policy: a
        # capability pattern -> the identities allowed to approve it, so a
        # money-movement confirm can't be self-approved by just anyone.
        self.sign_key = sign_key
        self.approvers = dict(approvers or {})
        self.pending: dict[str, PendingApproval] = {}
        self.lock = threading.Lock()

    # -- rules ------------------------------------------------------------

    def classify(self, name: str, tool_input) -> "tuple[str, str]":
        """First matching action wins, checked deny > allow > confirm.

        A ``cap:<capability>`` pattern matches by inferred tool capability
        (read/write/exec/network/secret/destructive) rather than by name, so
        ``deny cap:exec`` blocks every shell-shaped tool whatever it's called.
        """
        sig = _signature(name, tool_input)
        norm = _normalize(sig)
        for patterns, action in ((self.deny, DENY), (self.allow, ALLOW), (self.confirm, CONFIRM)):
            for p in patterns:
                if p.startswith("cap:"):
                    from .capabilities import matches_cap

                    if matches_cap(p, name, tool_input):
                        return action, p
                elif fnmatch(name, p) or fnmatch(sig, p) or fnmatch(norm, p):
                    return action, p
        return self.default, ""

    # -- approval inbox ----------------------------------------------------

    def pending_list(self) -> "list[dict]":
        with self.lock:
            return [p.to_dict() for p in self.pending.values() if not p.event.is_set()]

    def approver_allowed(self, name: str, tool_input, who: str) -> bool:
        """Is ``who`` permitted to APPROVE this call under the approver policy?

        A denial never needs a permitted identity -- anyone may stop a call.
        An approval of a capability with a required-approver list must come
        from a listed identity."""
        if not self.approvers:
            return True
        from .capabilities import matches_cap

        for pattern, allowed in self.approvers.items():
            matches = (matches_cap(pattern, name, tool_input) if pattern.startswith("cap:")
                       else fnmatch(name, pattern) or fnmatch(_signature(name, tool_input), pattern))
            if matches and who not in allowed:
                return False
        return True

    def decide_pending(self, pending_id: str, approve: bool, who: str = "") -> bool:
        with self.lock:
            p = self.pending.get(pending_id)
            if p is None or p.event.is_set():
                return False
            # Approver policy: an unauthorized approval is refused outright (the
            # call stays pending until an authorized approver or the timeout).
            if approve and not self.approver_allowed(p.tool, p.input, who):
                return False
            p.decision = "approve" if approve else "deny"
            p.decided_by = who
            p.event.set()
            return True

    def _sign(self, event: dict) -> dict:
        """Attach an HMAC over the decision fields, if a signing key is set."""
        if not self.sign_key:
            return event
        import hmac

        payload = json.dumps(
            {k: event.get(k) for k in ("id", "tool", "action", "via", "by", "ts")},
            sort_keys=True, default=str).encode()
        event["signature"] = "hmac-sha256:" + hmac.new(
            self.sign_key, payload, __import__("hashlib").sha256).hexdigest()
        return event

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

    def _await_approval(self, name: str, tool_input, rule: str) -> "tuple[bool, str, str, str]":
        """File a pending approval and block until decided or timed out.

        Returns (approved, via, id, who) -- ``who`` is the operator identity
        that decided, for the audit trail (empty on timeout)."""
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
            return False, "timeout", p.id, ""
        return p.decision == "approve", "operator", p.id, p.decided_by

    # -- screening ---------------------------------------------------------

    def screen(self, response: dict) -> "tuple[dict, list[dict]]":
        """Return (response the client may see, shield events). May block on confirm."""
        if "choices" in response:
            return self._screen_openai(response)
        return self._screen_anthropic(response)

    def _judge(self, name: str, tool_input) -> "tuple[bool, dict | None]":
        """Decide one tool call. Returns (allowed, event-or-None)."""
        allowed, event = self._decide(name, tool_input)
        if allowed:
            self._arm_after_rules(name, tool_input)
        return allowed, event

    def _decide(self, name: str, tool_input) -> "tuple[bool, dict | None]":
        # Tripped sequence rules come first: "this run touched secrets" beats
        # any static allow. Denies before confirms across all tripped rules.
        gate = self._sequence_gate(name, tool_input)
        if gate is not None:
            return gate
        action, rule = self.classify(name, tool_input)
        base = {"ts": time.time(), "tool": name, "input": tool_input, "rule": rule}
        if action == DENY:
            return False, {**base, "action": "deny", "via": "rule" if rule else "default"}
        if action == CONFIRM:
            return self._confirm_flow(name, tool_input, rule or "(shield default: confirm)", base)
        if not rule and self.judge is not None:
            # No explicit rule matched: let the judge model risk-score it.
            # Explicit allow rules bypass the judge on purpose.
            risk, reason, ok = self._assess_risk(name, tool_input)
            base = {**base, "judge_risk": risk, "judge_reason": reason}
            if not ok:
                return True, {**base, "action": "allow", "via": "judge-error"}  # fail-open
            if risk >= self.judge_threshold:
                return self._confirm_flow(
                    name, tool_input, f"judge({risk:.2f}): {reason}", base
                )
            return True, {**base, "action": "allow", "via": "judge"}
        return True, None

    # -- sequence rules ------------------------------------------------------

    def _sequence_gate(self, name: str, tool_input) -> "tuple[bool, dict] | None":
        """Consequences of tripped sequence rules, or None if none apply."""
        with self.lock:
            tripped = [r for r in self.sequence if r.triggered]
        confirm_rule = None
        for r in tripped:
            for action, pattern in r.consequences:
                if not _call_matches(pattern, name, tool_input):
                    continue
                label = f"{r.raw} (tripped by {r.evidence})"
                if action == DENY:
                    return False, {
                        "ts": time.time(), "tool": name, "input": tool_input,
                        "rule": label, "action": "deny", "via": "sequence",
                    }
                confirm_rule = confirm_rule or label
        if confirm_rule is None:
            return None
        # A tripped tripwire always gets a human: no trust-ratchet shortcut.
        base = {"ts": time.time(), "tool": name, "input": tool_input, "rule": confirm_rule}
        approved, via, pid, who = self._await_approval(name, tool_input, confirm_rule)
        return approved, self._sign({
            **base, "id": pid, "action": "approve" if approved else "deny",
            "via": via if via != "operator" else "sequence-operator",
            **({"by": who} if who else {}),
        })

    def _arm_after_rules(self, name: str, tool_input) -> None:
        """An allowed call arms every 'after' rule it matches."""
        with self.lock:
            for r in self.sequence:
                if r.trigger_type == "after" and not r.triggered and _call_matches(
                    r.trigger, name, tool_input
                ):
                    r.triggered = True
                    r.evidence = _signature(name, tool_input)[:200]

    def observe_request(self, request: dict) -> "list[dict]":
        """Scan a request's tool RESULTS for taint triggers (both dialects).

        The proxy calls this before forwarding: results of the last round's
        tool calls ride in the next request, which is the earliest the wire
        can know what a tool returned. Arming is recorded once per rule.
        """
        with self.lock:
            watching = [
                r for r in self.sequence if r.trigger_type == "taint" and not r.triggered
            ]
        if not watching:
            return []
        events = []
        texts = _tool_result_texts(request)
        with self.lock:
            for r in watching:
                for text in texts:
                    if _text_matches(text, r.trigger):
                        r.triggered = True
                        r.evidence = f"a tool result matching {r.trigger!r}"
                        events.append(
                            {"ts": time.time(), "action": "tainted", "via": "sequence",
                             "rule": r.raw}
                        )
                        break
        return events

    def _confirm_flow(self, name: str, tool_input, rule: str, base: dict) -> "tuple[bool, dict]":
        """The confirm path: ratchet short-circuit, else wait for the operator."""
        if self.trust is not None and 0 < self.trust_after <= self.trust.streak(name):
            return True, {
                **base, "action": "approve", "via": "ratchet", "streak": self.trust.streak(name)
            }
        approved, via, pid, who = self._await_approval(name, tool_input, rule)
        if self.trust is not None and via == "operator":
            # Only explicit human decisions move the ratchet; timeouts don't.
            self.trust.record(name, approved, {"id": pid, "ts": base["ts"], "rule": rule,
                                               "by": who})
        return approved, self._sign({
            **base, "id": pid, "action": "approve" if approved else "deny", "via": via,
            **({"by": who} if who else {}),
        })

    def _assess_risk(self, name: str, tool_input) -> "tuple[float, str, bool]":
        """Ask the judge model for a 0..1 risk score. Fail-open on any trouble."""
        judge = self.judge
        if isinstance(judge, str):
            from .providers import AnthropicProvider

            judge = self.judge = AnthropicProvider(judge)
        system = (
            "You are a security reviewer for AI-agent tool calls. You will see ONE "
            "proposed call as name({json arguments}). Reply with ONLY a JSON object "
            '{"risk": <number 0..1>, "reason": "<one short sentence>"}. '
            "High risk (>=0.7): destructive or irreversible actions (deleting or "
            "overwriting data, force-pushes), reading or exfiltrating credentials "
            "and secrets, piping downloads into a shell, sending data to unknown "
            "hosts, privilege changes. Low risk (<0.3): read-only inspection of "
            "ordinary project files, listing, searching, building, running tests."
        )
        try:
            resp = judge.complete(
                system, [{"role": "user", "content": _signature(name, tool_input)}], []
            )
            match = re.search(r"\{.*\}", getattr(resp, "text", "") or "", re.S)
            data = json.loads(match.group(0))
            risk = min(1.0, max(0.0, float(data["risk"])))
            return risk, str(data.get("reason", ""))[:200], True
        except Exception:
            return 0.0, "judge unavailable", False

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
        # Screen EVERY choice, not just choices[0]: with n>1 a blocked tool call
        # in a later choice must not slip through the firewall unscreened.
        new_choices, any_changed = [], False
        for choice in choices:
            new_choice, changed, choice_events = self._screen_openai_choice(choice)
            events.extend(choice_events)
            new_choices.append(new_choice)
            any_changed = any_changed or changed
        if not any_changed:
            return response, events
        return {**response, "choices": new_choices}, events

    def _screen_openai_choice(self, choice: dict) -> "tuple[dict, bool, list[dict]]":
        events: list[dict] = []
        message = choice.get("message", {}) or {}
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
            return choice, False, events
        new_message = {**message, "content": "\n".join(filter(None, [message.get("content") or ""] + notices))}
        if kept:
            new_message["tool_calls"] = kept
        else:
            new_message.pop("tool_calls", None)
        new_choice = {**choice, "message": new_message}
        if not kept:
            new_choice["finish_reason"] = "stop"
        return new_choice, True, events


def _tool_result_texts(request: dict) -> "list[str]":
    """Every tool-result string in a request body, Anthropic or OpenAI shaped."""
    texts: list[str] = []
    for message in request.get("messages") or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if message.get("role") == "tool" and isinstance(content, str):
            texts.append(content)  # OpenAI dialect
        if not isinstance(content, list):
            continue
        for block in content:  # Anthropic dialect
            if isinstance(block, dict) and block.get("type") == "tool_result":
                inner = block.get("content")
                if isinstance(inner, str):
                    texts.append(inner)
                elif isinstance(inner, list):
                    texts.extend(
                        b.get("text", "") for b in inner
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
    return texts


def _reason(event: "dict | None") -> str:
    via = (event or {}).get("via", "rule")
    if via == "rule":
        return f"matched deny rule '{(event or {}).get('rule', '')}'."
    if via == "sequence":
        return f"blocked by sequence rule '{(event or {}).get('rule', '')}'."
    if via == "default":
        return "no rule matched and this shield denies by default."
    if via == "timeout":
        return "approval timed out."
    return "denied by the operator."


def verify_approvals(data: dict, key: bytes) -> "tuple[list[dict], list[dict]]":
    """Verify the HMAC on every signed shield decision in a trace.

    Returns (valid, invalid) -- events whose signature recomputes correctly
    vs. those that were tampered with or signed with a different key. Events
    without a signature are ignored (they were recorded unsigned)."""
    import hashlib
    import hmac

    valid, invalid = [], []
    for ev in data.get("shield_events") or []:
        sig = ev.get("signature")
        if not sig:
            continue
        payload = json.dumps(
            {k: ev.get(k) for k in ("id", "tool", "action", "via", "by", "ts")},
            sort_keys=True, default=str).encode()
        expected = "hmac-sha256:" + hmac.new(key, payload, hashlib.sha256).hexdigest()
        (valid if hmac.compare_digest(sig, expected) else invalid).append(ev)
    return valid, invalid


class TrustLedger:
    """Consecutive-approval streaks per tool -- the evidence for the ratchet.

    Every entry links to the pending-approval ids that earned the trust, so a
    promotion is always auditable ("Bash was auto-allowed because you approved
    it 5 times: a3f2c1, 9b1e77, ..."). One explicit operator DENY demotes the
    tool and clears the streak; ``loom trust --demote <tool>`` does it by hand.
    """

    def __init__(self, path: str):
        self.path = path
        try:
            with open(path) as f:
                self.data: dict = json.load(f)
        except (OSError, json.JSONDecodeError):
            self.data = {}

    def streak(self, tool: str) -> int:
        return self.data.get(tool, {}).get("streak", 0)

    def record(self, tool: str, approved: bool, evidence: dict) -> None:
        entry = self.data.setdefault(tool, {"streak": 0, "evidence": []})
        if approved:
            entry["streak"] += 1
            entry["evidence"].append(evidence)
        else:
            entry["streak"] = 0
            entry["evidence"] = []
            entry["demoted_at"] = time.time()
        self._save()

    def demote(self, tool: str) -> bool:
        if tool not in self.data:
            return False
        self.data[tool] = {"streak": 0, "evidence": [], "demoted_at": time.time()}
        self._save()
        return True

    def _save(self) -> None:
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)
