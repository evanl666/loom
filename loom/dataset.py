"""``loom dataset``: turn a corpus of runs into training / eval data.

Every recorded run is a labeled example -- the prompt, what the agent did, the
answer, and (via the trace) whether it succeeded. ``dataset`` compiles a corpus
into the common formats:

    loom dataset from runs/ --format sft   -o sft.jsonl     # prompt -> completion
    loom dataset from runs/ --format trajectory -o traj.jsonl  # + tool steps
    loom dataset from runs/ --format eval  -o eval.jsonl    # prompt + expected
    loom dataset from runs/ --format dpo   -o dpo.jsonl     # chosen vs rejected

Paused / truncated / invalid-output runs are excluded from SFT and are the
*rejected* side of DPO pairs (grouped by prompt). Secrets are scrubbed from
every field. Traces are debugging assets; this makes them training assets too.
"""

from __future__ import annotations

import json
import os
from glob import glob
from typing import Any


def _succeeded(data: dict) -> bool:
    return (not data.get("paused") and not data.get("truncated")
            and data.get("stop_reason", "") in ("", "end_turn")
            and bool(str(data.get("output", "")).strip()))


def _prompt(data: dict) -> str:
    eps = data.get("episodes") or [data.get("prompt", "")]
    return str(eps[0]) if eps else ""


def _tool_steps(data: dict) -> list[dict]:
    from .action import actions

    return [{"tool": a.tool, "input": a.input,
             "result": (a.observation.text[:2000] if a.observation else "")}
            for a in actions(data) if a.type == "call" and a.tool]


def _scrub(obj: Any) -> Any:
    from .scrub import scrub_text

    if isinstance(obj, str):
        return scrub_text(obj)[0]
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub(v) for v in obj]
    return obj


def compile_dataset(source: Any, fmt: str = "sft") -> "list[dict]":
    """Compile a corpus (dir path or list of trace dicts) into ``fmt`` records."""
    datas: list[dict] = []
    if isinstance(source, str):
        for p in sorted(glob(os.path.join(source, "**", "*.loom.json"), recursive=True)):
            try:
                with open(p) as f:
                    d = json.load(f)
                if isinstance(d, dict):
                    datas.append(d)
            except (OSError, json.JSONDecodeError):
                continue
    else:
        datas = [d for d in source if isinstance(d, dict)]

    if fmt == "dpo":
        return _dpo(datas)

    records: list[dict] = []
    for d in datas:
        prompt, output = _prompt(d), str(d.get("output", ""))
        if fmt in ("sft", "trajectory") and not _succeeded(d):
            continue  # SFT learns from good behavior only
        if fmt == "sft":
            records.append(_scrub({
                "messages": [{"role": "user", "content": prompt},
                             {"role": "assistant", "content": output}]}))
        elif fmt == "trajectory":
            records.append(_scrub({
                "prompt": prompt, "steps": _tool_steps(d), "output": output}))
        elif fmt == "eval":
            records.append(_scrub({
                "prompt": prompt, "expected_output": output,
                "expected_tools": sorted({s["tool"] for s in _tool_steps(d)}),
                "succeeded": _succeeded(d)}))
        else:
            raise ValueError(f"unknown format {fmt!r} (sft|trajectory|eval|dpo)")
    return records


def _dpo(datas: "list[dict]") -> "list[dict]":
    """Preference pairs: for a prompt with both a good and a bad outcome, the
    successful output is `chosen` and a failing one is `rejected`."""
    by_prompt: dict[str, dict[str, list[str]]] = {}
    for d in datas:
        p = _prompt(d)
        out = str(d.get("output", ""))
        bucket = by_prompt.setdefault(p, {"good": [], "bad": []})
        (bucket["good"] if _succeeded(d) else bucket["bad"]).append(out)
    pairs: list[dict] = []
    for p, b in by_prompt.items():
        for chosen in b["good"]:
            for rejected in b["bad"]:
                if chosen.strip() and rejected.strip() and chosen != rejected:
                    pairs.append(_scrub({"prompt": p, "chosen": chosen, "rejected": rejected}))
    return pairs


def write_jsonl(records: "list[dict]", path: str) -> int:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return len(records)
