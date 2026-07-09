"""The Support/Ops Pack -- debug agents that touch customers.

Opt-in (``import loom.packs.support`` registers it). A support agent's side
effects land on *people*: refunds move money, replies reach a real inbox,
CRM writes change the record other teammates act on. What it teaches Loom:

  owns        refund / message / ticket / CRM / customer-record tools
  capabilities  the core taxonomy already flags money_movement, pii_access
              and user_communication; this pack adds ticket/CRM writes
  state_diff  "refunded $50 (order #123)", "emailed customer", "updated
              ticket #45: status -> closed" -- and when the tool input carries
              the prior value (``old``/``previous``), a real field diff
  undo        a refund gets a compensating-action plan (reverse or re-charge,
              policy permitting); an email cannot be unsent -- the plan is a
              follow-up correction; a CRM field write is revertible only when
              the prior value was recorded

The audit story: every one of these actions carries the firewall's
PolicyDecision (who approved, by which rule) on the Action itself.
"""

from __future__ import annotations

from fnmatch import fnmatchcase as fnmatch

from ..action import Action, StateDiff
from . import Pack, UndoPlan, register

_MONEY = ["refund*", "*_refund", "*payout*", "charge*", "*payment*", "credit_account*"]
_COMM = ["send_email*", "*send_message*", "send_sms*", "reply*", "notify*",
         "email_customer*", "post_message*"]
_TICKET = ["*ticket*", "close_case*", "escalate*"]
_CRM = ["update_customer*", "update_record*", "update_field*", "set_field*",
        "*crm*", "update_account*", "merge_contacts*"]
_RECORD = ["get_customer*", "*customer_record*", "lookup_user*", "get_account*"]
_ALL = _MONEY + _COMM + _TICKET + _CRM + _RECORD


def _matches(name: str, globs: "list[str]") -> bool:
    return any(fnmatch(name.lower(), g) for g in globs)


def _get(action: Action, *keys: str) -> str:
    if isinstance(action.input, dict):
        for k in keys:
            v = action.input.get(k)
            if v not in (None, ""):
                return str(v)
    return ""


class SupportPack(Pack):
    name = "support"

    def owns(self, action: Action) -> bool:
        return action.type == "call" and _matches(action.tool, _ALL)

    def capabilities(self, name: str, tool_input) -> "set[str]":
        caps: set[str] = set()
        if _matches(name, _TICKET) or _matches(name, _CRM):
            caps |= {"write", "external_side_effect"}
        if _matches(name, _RECORD):
            caps |= {"pii_access", "read"}
        return caps

    def state_diff(self, action: Action, trace: dict) -> "StateDiff | None":
        if _matches(action.tool, _MONEY):
            amount = _get(action, "amount", "value")
            ref = _get(action, "order_id", "charge_id", "invoice", "customer_id")
            return StateDiff("record",
                             f"moved money: {amount or '?'}" + (f" ({ref})" if ref else ""),
                             detail={"amount": amount, "ref": ref})
        if _matches(action.tool, _COMM):
            to = _get(action, "to", "recipient", "customer_id", "channel")
            return StateDiff("record", f"messaged {to or 'a user'}",
                             detail={"to": to})
        if _matches(action.tool, _TICKET) or _matches(action.tool, _CRM):
            target = _get(action, "ticket_id", "record_id", "customer_id", "account_id")
            field = _get(action, "field", "status")
            new = _get(action, "value", "new", "status")
            old = _get(action, "old", "previous", "prior")
            arrow = f": {field or 'field'} -> {new}" if new else ""
            if old:
                arrow = f": {field or 'field'} {old} -> {new}"
            return StateDiff("field", f"updated {target or 'a record'}{arrow}",
                             detail={"target": target, "field": field,
                                     "old": old, "new": new})
        return None

    def undo(self, action: Action, trace: dict) -> "UndoPlan | None":
        if _matches(action.tool, _MONEY):
            return UndoPlan(
                "compensate", "reverse the transaction (a compensating charge/refund, "
                              "policy permitting)",
                ["issue the opposite transaction for the same amount and reference"],
                reversible=False)  # money moved; only a new movement offsets it
        if _matches(action.tool, _COMM):
            return UndoPlan(
                "noop", "a sent message cannot be unsent -- send a follow-up correction",
                reversible=False)
        if _matches(action.tool, _TICKET) or _matches(action.tool, _CRM):
            old = _get(action, "old", "previous", "prior")
            if old:
                return UndoPlan("revert", f"restore the previous value ({old})",
                                [f"set the field back to {old!r}"])
            return UndoPlan(
                "noop", "prior value not recorded -- fetch it from the CRM's own "
                        "audit history to revert", reversible=False)
        return None

    def safe_runtime(self) -> str:
        return ("point the tools at a SANDBOX tenant with fake customers; enable "
                "idempotency keys on refund/payment calls so a replay or retry can't "
                "double-charge; route email to a catch-all mailbox")

    def replay_hint(self, action: Action) -> str:
        return ("point the tools at a sandbox tenant (or record with --shield confirm "
                "rules) before replaying actions that touch real customers")


register(SupportPack())
