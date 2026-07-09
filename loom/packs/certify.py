"""Pack certification: is this domain pack safe to trust in production?

A wrong pack is worse than no pack -- it will mislabel a destructive SQL as
reversible, or claim a sent email can be undone. Two tools keep packs honest:

  lint_pack(pack)         static checks against a corpus of tool shapes:
                          over-broad owns(), fake reversibility on external
                          side effects, and undo plans that lie.
  test_pack(pack, cases)  golden cases: an action -> expected capabilities,
                          state-diff kind, and undo kind/reversibility.

    loom packs lint --pack mypkg:MyPack
    loom packs test cases.yml

Both are offline and deterministic.
"""

from __future__ import annotations

from typing import Any

from ..action import Action, actions
from . import Pack

# Benign, unmistakably-not-my-domain tools. A well-scoped pack owns NONE of
# these; a pack that claims them will fire on unrelated agents.
_BENIGN = [
    ("get_weather", {"city": "Paris"}),
    ("add", {"a": 1, "b": 2}),
    ("list_items", {}),
    ("search", {"q": "docs"}),
    ("summarize", {"text": "..."}),
    ("ping", {}),
    ("translate", {"text": "hi"}),
]

# Shapes that, if a pack owns them, must NOT be reported as truly reversible.
_EXTERNAL_SHAPES = [
    ("issue_refund", {"amount": 50}),
    ("send_email", {"to": "x@y.com"}),
    ("charge_card", {"amount": 9}),
    ("run_sql", {"query": "DROP TABLE users"}),
    ("run_sql", {"query": "DELETE FROM orders"}),
    ("click", {"selector": "#buy"}),
]

_EXTERNAL_CAPS = {"money_movement", "user_communication", "browser_submit"}


def _action(pack: Pack, name: str, tool_input: dict, result: str = "ok") -> "Action | None":
    """Build the single call Action this pack sees for one tool call."""
    trace = {"log": [
        {"seq": 0, "kind": "model", "key": "k",
         "result": {"text": "", "tool_calls": [{"id": "t", "name": name, "input": tool_input}],
                    "stop_reason": "tool_use", "usage": {}}},
        {"seq": 1, "kind": f"tool:{name}", "key": "k2", "result": result},
    ]}
    # Isolate: swap the global registry for JUST this pack for the duration,
    # then restore it -- so certifying a built-in doesn't unregister it.
    import loom.packs as _pkgs

    saved = list(_pkgs._REGISTRY)
    _pkgs._REGISTRY = [pack]
    try:
        for a in actions(trace):
            if a.type == "call":
                return a
    finally:
        _pkgs._REGISTRY = saved
    return None


def lint_pack(pack: Pack) -> "list[str]":
    """Static correctness warnings for a pack (empty list == clean)."""
    problems: list[str] = []

    # 1. over-broad owns(): a pack must not claim benign, unrelated tools.
    owned_benign = [n for n, i in _BENIGN if pack.owns(_stub_action(n, i))]
    if owned_benign:
        problems.append(
            f"owns() is over-broad: it claims unrelated tools {owned_benign} -- "
            "it will fire on agents outside this domain. Tighten owns().")

    # 2. fake reversibility: an external side effect can't be truly undone.
    for name, tinput in _EXTERNAL_SHAPES:
        a = _action(pack, name, tinput)
        if a is None or not pack.owns(a):
            continue
        plan = pack.undo(a, {})
        caps = set(a.capabilities)
        external = bool(caps & _EXTERNAL_CAPS) or "destructive" in caps
        if plan is not None and plan.reversible and external:
            problems.append(
                f"undo({name}) claims reversible=True for an external/destructive "
                f"action ({sorted(caps & (_EXTERNAL_CAPS | {'destructive'}))}) -- "
                "those can only be COMPENSATED, not reverted; set reversible=False.")

    # 3. silent ownership: a pack that owns an action but adds no capability
    #    AND no state_diff is just noise on the timeline.
    for name, tinput in _EXTERNAL_SHAPES:
        a = _action(pack, name, tinput)
        if a is None or not pack.owns(a):
            continue
        if not pack.capabilities(name, tinput) and pack.state_diff(a, {}) is None:
            problems.append(
                f"owns({name}) but contributes neither a capability nor a state_diff "
                "-- either enrich it or narrow owns().")
            break
    return problems


def _stub_action(name: str, tool_input: dict) -> Action:
    """A bare Action for owns() checks (no registry, no enrichment)."""
    from ..capabilities import capabilities as _caps
    from ..risk import classify

    return Action(step=0, depth=0, type="call", tool=name, input=tool_input,
                  capabilities=sorted(_caps(name, tool_input)), risk=classify(name, tool_input))


def test_pack(pack: Pack, cases: "list[dict]") -> "list[dict]":
    """Run golden cases against a pack. Returns per-case results.

    A case is ``{action: {tool, input, result?}, expect: {...}}`` where expect
    may set: ``owns`` (bool), ``capabilities_include`` (list), ``state_diff_kind``
    (str), ``undo_kind`` (revert/compensate/noop), ``reversible`` (bool).
    """
    results = []
    for case in cases:
        act = case.get("action", {})
        exp = case.get("expect", {})
        a = _action(pack, act.get("tool", ""), act.get("input", {}) or {},
                    act.get("result", "ok"))
        fails = []
        if a is None:
            fails.append("pack produced no action")
        else:
            if "owns" in exp and pack.owns(a) != exp["owns"]:
                fails.append(f"owns: expected {exp['owns']}, got {pack.owns(a)}")
            for cap in exp.get("capabilities_include", []):
                if cap not in a.capabilities:
                    fails.append(f"missing capability {cap!r} (got {a.capabilities})")
            if "state_diff_kind" in exp:
                got = a.state_diff.kind if a.state_diff else None
                if got != exp["state_diff_kind"]:
                    fails.append(f"state_diff kind: expected {exp['state_diff_kind']!r}, got {got!r}")
            if "undo_kind" in exp or "reversible" in exp:
                plan = pack.undo(a, {})
                if plan is None:
                    fails.append("expected an undo plan, got none")
                else:
                    if "undo_kind" in exp and plan.kind != exp["undo_kind"]:
                        fails.append(f"undo kind: expected {exp['undo_kind']!r}, got {plan.kind!r}")
                    if "reversible" in exp and plan.reversible != exp["reversible"]:
                        fails.append(f"reversible: expected {exp['reversible']}, got {plan.reversible}")
        results.append({"tool": act.get("tool", ""), "ok": not fails, "failures": fails})
    return results


def load_pack(spec: str) -> Pack:
    """Resolve ``module:attr`` to a Pack instance (attr may be a class or instance)."""
    import importlib

    module_name, _, attr = spec.partition(":")
    if not attr:
        raise ValueError("--pack must look like module:attr")
    obj = getattr(importlib.import_module(module_name), attr)
    return obj() if isinstance(obj, type) else obj
