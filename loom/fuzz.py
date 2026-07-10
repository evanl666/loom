"""``loom fuzz``: a hostile-trace corpus that proves the debugger is crash-proof.

A trace is an untrusted artifact -- it can be hand-edited, shared by someone
else, produced by a third-party recorder, or truncated by a crash. Every Loom
analyzer runs on user files, so a malformed trace must never turn into a Python
traceback (a clear ``ValueError`` is fine -- that is the tool refusing bad
input, not falling over).

``fuzz`` takes a real trace and derives a battery of *malformed* mutations --
non-dict effects, missing keys, huge blobs, hostile artifact pointers, broken
model results, tampered approvals -- then runs every analyzer against each one
and reports which (if any) crashed:

    loom fuzz session.loom.json --check     # exit 1 if any analyzer crashed

Wire it into CI to keep the "tolerate untrusted traces" guarantee from
regressing. A crash here is a bug; a ``ValueError`` naming the problem is a pass.
"""

from __future__ import annotations

import copy
from typing import Any, Callable

# Exceptions that mean "this input was refused cleanly" -- expected, a PASS.
_CLEAN = (ValueError,)
# Exceptions that mean "the analyzer fell over" -- a real robustness bug.
_CRASH = (AttributeError, TypeError, KeyError, IndexError, RecursionError)


def _seed() -> dict:
    """A minimal well-formed trace to mutate when none is provided."""
    return {
        "version": 2,
        "prompt": "do the thing",
        "output": "done",
        "episodes": ["do the thing"],
        "log": [
            {"seq": 0, "kind": "model", "key": "k0", "result": {
                "text": "calling", "tool_calls": [
                    {"id": "t1", "name": "read_file", "input": {"path": ".env"}}],
                "stop_reason": "tool_use", "usage": {"input_tokens": 10, "output_tokens": 4}}},
            {"seq": 1, "kind": "tool:read_file", "key": "k1", "result": "API_KEY=sk-test-123"},
            {"seq": 2, "kind": "model", "key": "k2", "result": {
                "text": "done", "tool_calls": [], "stop_reason": "end_turn",
                "usage": {"input_tokens": 20, "output_tokens": 6}}},
        ],
        "shield_events": [],
    }


def mutations(trace: "dict | None" = None) -> "list[tuple[str, Any]]":
    """(name, hostile-trace) pairs derived from ``trace`` (or a built-in seed)."""
    base = copy.deepcopy(trace) if trace else _seed()
    out: list[tuple[str, Any]] = []

    def add(name: str, fn: "Callable[[dict], Any]") -> None:
        t = copy.deepcopy(base)
        out.append((name, fn(t)))

    # -- shape attacks: the whole document is the wrong type ----------------
    out.append(("doc-is-a-list", [1, 2, 3]))
    out.append(("doc-is-null", None))
    out.append(("doc-is-scalar", 42))
    add("log-not-a-list", lambda t: {**t, "log": "nope"})
    add("log-null", lambda t: {**t, "log": None})
    add("missing-prompt-output", lambda t: {"log": t["log"]})
    add("empty", lambda t: {})

    # -- effect-entry attacks ----------------------------------------------
    add("effects-non-dict", lambda t: {**t, "log": [None, 42, "x", *t["log"]]})
    add("effect-missing-keys", lambda t: {**t, "log": [{}, {"seq": 1}, *t["log"]]})
    add("effect-huge-seq", lambda t: {**t, "log": [{"seq": 10**18, "kind": "model",
                                                    "result": {}}, *t["log"]]})
    add("effect-negative-seq", lambda t: {**t, "log": [{"seq": -5, "kind": "model",
                                                        "result": {}}, *t["log"]]})

    # -- model-result attacks ----------------------------------------------
    def _first_model(t: dict) -> dict:
        for e in t["log"]:
            if isinstance(e, dict) and e.get("kind") == "model":
                return e
        return t["log"][0]

    def m_result_not_dict(t: dict) -> dict:
        _first_model(t)["result"] = "a string result"
        return t
    add("model-result-not-dict", m_result_not_dict)

    def m_null_toolcalls(t: dict) -> dict:
        _first_model(t)["result"]["tool_calls"] = None
        return t
    add("model-tool_calls-null", m_null_toolcalls)

    def m_junk_toolcalls(t: dict) -> dict:
        _first_model(t)["result"]["tool_calls"] = [None, 7, {"name": None}, {}]
        return t
    add("model-tool_calls-junk", m_junk_toolcalls)

    def m_null_usage(t: dict) -> dict:
        _first_model(t)["result"]["usage"] = None
        return t
    add("model-usage-null", m_null_usage)

    # -- huge payloads ------------------------------------------------------
    def m_huge_result(t: dict) -> dict:
        t["log"].append({"seq": 99, "kind": "tool:dump", "result": "A" * 2_000_000})
        return t
    add("huge-tool-result", m_huge_result)

    def m_deep_nesting(t: dict) -> dict:
        d: Any = "x"
        for _ in range(200):
            d = {"n": d}
        _first_model(t)["result"]["tool_calls"] = [{"id": "d", "name": "f", "input": d}]
        return t
    add("deeply-nested-input", m_deep_nesting)

    # -- hostile pointers / traversal --------------------------------------
    def m_bad_artifact(t: dict) -> dict:
        t["log"].append({"seq": 98, "kind": "tool:x",
                         "result": {"_loom_artifact": {"sha": "../../etc/passwd", "bytes": 9}}})
        return t
    add("artifact-pointer-traversal", m_bad_artifact)

    # -- shield / approval attacks -----------------------------------------
    add("shield_events-null", lambda t: {**t, "shield_events": None})
    add("shield_events-junk", lambda t: {**t, "shield_events": [None, 7, {"action": None}]})

    # -- control chars / weird unicode -------------------------------------
    def m_weird_text(t: dict) -> dict:
        _first_model(t)["result"]["text"] = "\x00\x07\x1b[31m\ud800 weird"
        return t
    add("control-chars-in-text", m_weird_text)

    return out


# The analyzers a malformed trace must survive. Each takes the trace dict.
def _battery() -> "list[tuple[str, Callable[[dict], Any]]]":
    from .action import actions, effect_dicts
    from .autopsy import autopsy_html
    from .cost import analyze_cost
    from .diagnose import diagnose
    from .diff import score_breakdown
    from .export import trace_to_html
    from .incident import build_report
    from .inject import find_injections
    from .taint import dlp_report, taint_paths

    return [
        ("actions", lambda d: actions(d)),
        ("effect_dicts", lambda d: list(effect_dicts(d))),
        ("taint_paths", lambda d: taint_paths(d)),
        ("dlp_report", lambda d: dlp_report(d)),
        ("analyze_cost", lambda d: analyze_cost(d)),
        ("score_breakdown", lambda d: score_breakdown(d)),
        ("find_injections", lambda d: find_injections(d)),
        ("incident", lambda d: build_report(d, "fuzz.loom.json")),
        ("diagnose", lambda d: diagnose(d)),
        ("trace_to_html", lambda d: trace_to_html(d)),
        ("autopsy_html", lambda d: autopsy_html(d)),
    ]


def fuzz_check(trace: "dict | None" = None) -> dict:
    """Run every analyzer against every mutation; report crashes (not ValueErrors).

    Returns {"total", "crashes": [{mutation, analyzer, error}], "clean", "ok"}.
    ``ok`` is True when nothing crashed (clean ValueErrors are allowed).
    """
    battery = _battery()
    crashes: list[dict] = []
    clean = 0
    muts = mutations(trace)
    for mname, mtrace in muts:
        for aname, fn in battery:
            try:
                fn(mtrace)
            except _CRASH as e:
                crashes.append({"mutation": mname, "analyzer": aname,
                                "error": f"{type(e).__name__}: {e}"[:160]})
            except _CLEAN:
                clean += 1
            except Exception as e:  # noqa: BLE001 -- unexpected type: treat as a crash
                crashes.append({"mutation": mname, "analyzer": aname,
                                "error": f"{type(e).__name__}: {e}"[:160]})
    return {
        "total": len(muts) * len(battery),
        "mutations": len(muts),
        "analyzers": len(battery),
        "crashes": crashes,
        "clean_refusals": clean,
        "ok": not crashes,
    }


def describe_fuzz(report: dict) -> str:
    lines = [
        f"fuzzed {report['mutations']} hostile mutation(s) x {report['analyzers']} "
        f"analyzer(s) = {report['total']} runs",
    ]
    if report["ok"]:
        lines.append(f"  ✓ no crashes ({report['clean_refusals']} clean refusals) -- "
                     "every analyzer tolerates untrusted traces")
    else:
        lines.append(f"  ✗ {len(report['crashes'])} crash(es):")
        for c in report["crashes"][:40]:
            lines.append(f"      {c['mutation']:28} {c['analyzer']:16} {c['error']}")
    return "\n".join(lines)
