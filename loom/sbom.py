"""``loom sbom``: a Software Bill of Materials for an AI agent.

Before an enterprise ships an agent, security review asks: what model, what
tools, what can they reach, what's gating them? ``sbom`` reads a recorded run
(or a corpus) and emits a CycloneDX-flavored bill of materials -- the model,
every tool and its capabilities, whether a firewall gated it, the exfiltration
surface, and a risk summary -- as audit-ready JSON:

    loom sbom session.loom.json -o agent.sbom.json

The procurement/security artifact that gets an agent through review.
"""

from __future__ import annotations

import json
import os
from glob import glob
from typing import Any


def build_sbom(source: Any) -> dict:
    """A CycloneDX-like SBOM for the agent(s) in a trace or corpus."""
    datas: list[dict] = []
    if isinstance(source, str) and os.path.isdir(source):
        for p in sorted(glob(os.path.join(source, "**", "*.loom.json"), recursive=True)):
            try:
                with open(p) as f:
                    datas.append(json.load(f))
            except (OSError, json.JSONDecodeError):
                continue
    elif isinstance(source, str):
        with open(source) as f:
            datas.append(json.load(f))
    else:
        datas.append(source)

    from .scan import scan

    models = set()
    tool_caps: dict[str, set] = {}
    gated: set[str] = set()
    for d in datas:
        if not isinstance(d, dict):
            continue
        if d.get("model"):
            models.add(d["model"])
    rep = scan(source)  # reuses the supply-chain surface + findings + grade
    for t in rep["tools"]:
        tool_caps.setdefault(t["name"], set()).update(t["capabilities"])
        if t["gated"]:
            gated.add(t["name"])

    components = [{
        "type": "machine-learning-model", "name": m, "group": "provider",
    } for m in sorted(models)]
    for name, caps in sorted(tool_caps.items()):
        components.append({
            "type": "tool", "name": name,
            "properties": [{"name": "capability", "value": c} for c in sorted(caps)],
            "properties_gated": name in gated,
        })

    return {
        "bomFormat": "LoomSBOM",
        "specVersion": "1.0",
        "metadata": {"tool": "loom sbom", "runs": len(datas),
                     "component": {"type": "application", "name": "ai-agent"}},
        "components": components,
        "summary": {
            "models": sorted(models),
            "tools": len(tool_caps),
            "ungated_dangerous": [f["tool"] for f in rep["findings"]
                                  if f["severity"] == "high" and f.get("tool")],
            "risk_grade": rep["grade"],
            "exfiltration_paths": sum(1 for f in rep["findings"]
                                      if "exfiltration" in f["issue"]),
        },
        "findings": rep["findings"],
    }


def describe_sbom(sbom: dict) -> str:
    s = sbom["summary"]
    lines = [
        f"AI-agent SBOM — risk grade {s['risk_grade']} ({sbom['metadata']['runs']} run(s))",
        f"  models: {', '.join(s['models']) or '(none recorded)'}",
        f"  tools: {s['tools']}   ungated dangerous: {len(s['ungated_dangerous'])}   "
        f"exfil paths: {s['exfiltration_paths']}",
    ]
    if s["ungated_dangerous"]:
        lines.append(f"  ⚠ ungated dangerous tools: {', '.join(s['ungated_dangerous'])}")
    return "\n".join(lines)
