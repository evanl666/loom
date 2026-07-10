"""``loom shadow``: shadow-deploy a policy change against real history.

Before you enforce a new firewall policy, shadow it: replay a corpus of recorded
runs through it and see, offline, exactly what it would have changed -- which
runs get denied or gated, how many that previously SUCCEEDED would break, the
capability risk it removes, and the exfiltration paths it would have caught.

    loom shadow runs/ --policy new.yml
    loom shadow runs/ --policy new.yml --baseline current.yml   # diff two policies

It's a canary without touching production. Pairs with `policy rollout` (which
gates promotion on the same breakage signal).
"""

from __future__ import annotations

from typing import Any


def shadow_eval(paths: "list[str]", policy_path: str,
                baseline_path: str = "") -> dict:
    """What a policy would do to a corpus, offline. Optionally diff a baseline."""
    import os
    from glob import glob

    from .policy_file import load_document, simulate, simulate_diff, to_shield_kwargs
    from .shield import Shield
    from .taint import taint_paths

    expanded: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            expanded += sorted(glob(os.path.join(p, "**", "*.loom.json"), recursive=True))
        else:
            expanded.append(p)
    paths = expanded

    new = Shield(**to_shield_kwargs(load_document(policy_path)))
    sim = simulate(new, paths)

    # capability risk the policy removes (dangerous caps it would deny/confirm)
    gated_caps = {}
    for c in sim["capabilities"]:
        if c["deny"] or c["confirm"]:
            gated_caps[c["capability"]] = c["deny"] + c["confirm"]

    # exfiltration paths across the corpus (what a DLP-shaped policy protects)
    import json as _json
    exfil = 0
    for p in paths:
        try:
            with open(p) as f:
                exfil += len(taint_paths(_json.load(f)))
        except (OSError, ValueError):
            pass

    out = {
        "runs": sim["runs"], "calls": sim["calls"],
        "would_deny_runs": len(sim["denied"]),
        "would_confirm_runs": len(sim["confirm_only"]),
        "untouched": sim["untouched"],
        "breakages": [d["name"] for d in sim["false_positives"]],
        "gated_capabilities": gated_caps,
        "exfil_paths": exfil,
        "top_rules": sim["rule_hits"][:6],
    }
    if baseline_path:
        old = Shield(**to_shield_kwargs(load_document(baseline_path)))
        d = simulate_diff(old, new, paths)
        out["diff"] = {"newly_denied": d["newly_denied"][:20],
                       "newly_confirmed": d["newly_confirmed"][:20],
                       "released": d["released"][:20]}
    safe = not out["breakages"]
    out["verdict"] = ("safe to enforce -- no successful run breaks"
                      if safe else
                      f"HOLD -- {len(out['breakages'])} successful run(s) would break; "
                      "review or canary first")
    out["safe"] = safe
    return out


def describe_shadow(r: dict) -> str:
    lines = [f"shadow deployment — {r['runs']} run(s), {r['calls']} tool call(s)",
             f"  would DENY {r['would_deny_runs']} run(s), CONFIRM {r['would_confirm_runs']}, "
             f"leave {r['untouched']} untouched",
             f"  exfiltration paths in corpus: {r['exfil_paths']}"]
    if r["gated_capabilities"]:
        lines.append(f"  capability risk removed: {', '.join(sorted(r['gated_capabilities']))}")
    if r["breakages"]:
        lines.append(f"  ⚠ {len(r['breakages'])} successful run(s) would BREAK: "
                     f"{', '.join(r['breakages'][:6])}")
    if "diff" in r:
        d = r["diff"]
        lines.append(f"  vs baseline: +{len(d['newly_denied'])} denied, "
                     f"+{len(d['newly_confirmed'])} confirmed, {len(d['released'])} released")
    lines.append(f"  {'✓' if r['safe'] else '⛔'} {r['verdict']}")
    return "\n".join(lines)
