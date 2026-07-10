"""Recover the agent hierarchy from a trace -- for ANY framework.

A recording proxy sees the wire (HTTP calls to the model API); the parent/child
structure of a multi-agent system (a supervisor delegating to workers, a
DeepAgents / CrewAI / LangGraph sub-agent) lives in the *application* and never
reaches the wire. But each sub-agent almost always has a **distinct system
prompt and tool set**, so we can reconstruct "which agent made this call" from
the per-call fingerprint the proxy now records -- without the framework
cooperating.

Two sources of truth, in order of confidence:

  1. ``meta.agent``   -- the agent name, present when Loom's own harness recorded
                         the run (it knows the hierarchy natively, incl. depth).
  2. ``meta.sys_hash``/``sys_head``/``tools`` -- a wire fingerprint, present for
                         proxied third-party agents; calls are clustered by it.

``infer_agents`` returns a per-step agent label the debugger uses to lane and
color the timeline, plus an agents overview and the hand-off edges between them.
"""

from __future__ import annotations

import re

# Role phrases we can lift from a system prompt to name an anonymous wire agent.
_ROLE_PATTERNS = [
    re.compile(r"you are (?:the |a |an )?([\w][\w -]{1,28}?)(?:[.,:;]| that | who | which | agent| assistant| whose)", re.I),
    re.compile(r"\b([\w-]+)[ -](?:agent|worker|subagent|specialist)\b", re.I),
]
_STOP = {"you", "a", "an", "the", "helpful", "an", "ai", "assistant"}


def _label_from_system(head: str) -> "str | None":
    head = (head or "").strip()
    if not head:
        return None
    for pat in _ROLE_PATTERNS:
        m = pat.search(head)
        if m:
            phrase = re.sub(r"\s+", " ", m.group(1)).strip(" -.,")
            words = [w for w in phrase.split(" ") if w.lower() not in _STOP]
            if words:
                return " ".join(words[:3])[:24]
    return None


def _entries(data: dict) -> list:
    log = data.get("log")
    return [e for e in (log if isinstance(log, list) else []) if isinstance(e, dict)]


def infer_agents(data: dict) -> dict:
    """Attribute every step to an agent. Returns::

        {
          "multi": bool,                     # more than one agent seen
          "agents": [ {id,label,model,tools,calls,is_root,color} ],
          "step_agent": { seq(str): agent_id },   # for the debugger
          "edges": [ {"from":id,"to":id,"seq":int} ],  # hand-offs / delegations
          "source": "native" | "wire" | "flat",
        }
    """
    entries = _entries(data)
    agents: dict[tuple, dict] = {}   # identity key -> record
    order: list[tuple] = []
    step_agent: dict[str, str] = {}
    active_by_depth: dict[int, tuple] = {}
    edges: list[dict] = []
    source = "flat"

    def _ident(meta: dict, depth: int) -> tuple:
        nonlocal source
        if meta.get("agent"):
            source = "native"
            return ("name", meta["agent"])
        if meta.get("sys_hash"):
            if source != "native":
                source = "wire"
            return ("fp", meta["sys_hash"], tuple(meta.get("tools", [])))
        return ("depth", depth)

    for e in entries:
        seq = e.get("seq")
        kind = e.get("kind", "")
        depth = e.get("depth", 0) or 0
        meta = e.get("meta") or {}
        if kind == "model":
            key = _ident(meta, depth)
            rec = agents.get(key)
            if rec is None:
                rec = {"key": key, "label": None, "sys_head": meta.get("sys_head", ""),
                       "model": meta.get("model", "") or data.get("model", ""),
                       "tools": list(meta.get("tools", [])), "calls": 0, "depth": depth}
                agents[key] = rec
                order.append(key)
            elif meta.get("tools") and not rec["tools"]:
                rec["tools"] = list(meta.get("tools", []))
            rec["calls"] += 1
            prev = active_by_depth.get(depth)
            active_by_depth[depth] = key
            # a hand-off: the active agent at this depth changed, or we descended
            parent = active_by_depth.get(depth - 1) if depth else None
            if parent and parent != key and not any(
                ed for ed in edges if ed["_pk"] == (parent, key)):
                edges.append({"_pk": (parent, key), "from": parent, "to": key, "seq": seq})
            elif prev and prev != key and depth == 0 and not any(
                ed for ed in edges if ed["_pk"] == (prev, key)):
                edges.append({"_pk": (prev, key), "from": prev, "to": key, "seq": seq})
            if seq is not None:
                step_agent[str(seq)] = key
        else:
            # a tool / other effect belongs to the agent active at its depth
            key = active_by_depth.get(depth) or active_by_depth.get(0)
            if key is not None and seq is not None:
                step_agent[str(seq)] = key

    # assign readable labels + ids, stable in first-appearance order
    id_of: dict[tuple, str] = {}
    used: set[str] = set()
    out_agents: list[dict] = []
    for i, key in enumerate(order):
        rec = agents[key]
        if key[0] == "name":
            label = key[1]
        else:
            label = _label_from_system(rec["sys_head"]) or f"agent {i + 1}"
        base, n = label, 2
        while label in used:  # disambiguate collisions
            label = f"{base} ({n})"; n += 1
        used.add(label)
        aid = f"a{i + 1}"
        id_of[key] = aid
        out_agents.append({
            "id": aid, "label": label, "model": rec["model"],
            "tools": rec["tools"], "calls": rec["calls"],
            "is_root": i == 0, "color": i % 8,
        })

    return {
        "multi": len(out_agents) > 1,
        "agents": out_agents,
        "step_agent": {seq: id_of[key] for seq, key in step_agent.items()},
        "edges": [{"from": id_of[e["from"]], "to": id_of[e["to"]], "seq": e["seq"]}
                  for e in edges if e["from"] in id_of and e["to"] in id_of],
        "source": source,
    }
