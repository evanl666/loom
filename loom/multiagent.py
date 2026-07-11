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
    re.compile(r"you are (?:the |a |an )?([\w][\w -]{1,34}?)(?:[.,:;\n]| that | who | which | whose | designed| responsible)", re.I),
    re.compile(r"\b([\w][\w -]{1,28}?)[ -](?:agent|worker|subagent|specialist|assistant|analyst|reviewer|planner|orchestrator|supervisor)\b", re.I),
]
_STOP = {"you", "a", "an", "the", "helpful", "ai", "are", "is", "was", "be",
         "your", "this", "that", "our", "my", "an"}
# Generic identities that don't distinguish a sub-agent (a framework preamble
# like Claude Code's "You are Claude Code..."). Skip these and keep scanning
# for the specific role a sub-agent was given.
_GENERIC = {
    "", "claude", "claude code", "assistant", "an assistant", "ai assistant",
    "helpful assistant", "chatbot", "agent", "an agent", "language model",
}
# Role keywords that mark a *specific* role -- preferred over a bare noun.
_SPECIFIC = re.compile(
    r"specialist|researcher|writer|worker|analyst|reviewer|planner|orchestrator|"
    r"supervisor|coder|engineer|manager|expert|assistant\b", re.I)


# Tokens too generic to link a delegation call to its sub-agent.
_STEM_STOP = {"the", "and", "ask", "for", "delegate", "task", "work", "agent",
              "sub", "call", "lead", "team", "coworker", "assistant", "specialist",
              "return", "result", "using", "then", "with", "this"}
# Arg keys on a delegation tool-call that name/describe the target sub-agent.
_DELEG_ARG_KEYS = ("subagent_type", "type", "name", "agent", "task",
                   "description", "query", "prompt", "instruction", "goal", "input")


def stem_tokens(*parts) -> set:
    """5-char stems of the meaningful words in ``parts``.

    Stemming to 5 chars lets a delegation tool ('ask_research', 'calculator')
    match its sub-agent's role/tools ('Research Specialist', 'calculate')
    across the naming gap. Used both to name agents and, in ``infer_agents``,
    to attach each sub-agent to the parent whose tool call actually spawned it
    (correct even when a parent fans out several siblings in ONE turn)."""
    out: set = set()
    for p in parts:
        for w in re.findall(r"[a-z]{4,}", str(p).lower()):
            if w not in _STEM_STOP:
                out.add(w[:5])
    return out


def _clean_role(phrase: str) -> str:
    phrase = re.sub(r"\s+", " ", phrase or "").strip(" -.,'\"")
    words = [w for w in phrase.split(" ") if w.lower() not in _STOP]
    return " ".join(words[:3])[:26]


def best_role(system: str) -> "str | None":
    """The most specific role phrase in a (possibly long) system prompt.

    A sub-agent framework often prepends a generic identity ("You are Claude
    Code...") and only later states the sub-agent's real role ("You are a
    Landmark Researcher"). We scan the WHOLE prompt, drop generic identities,
    and prefer a phrase carrying a role keyword."""
    if not system:
        return None
    # The agent's OWN stated identity -- the first non-generic "You are (a) X" --
    # is the most reliable ("You are a Coordinator. Delegate to the researcher..."
    # must read Coordinator, not the delegation target). Only fall back to the
    # "X specialist/agent" phrasing when there's no such self-identification.
    for m in _ROLE_PATTERNS[0].finditer(system):
        role = _clean_role(m.group(1))
        if role and role.lower() not in _GENERIC:
            return role
    cands: list[str] = []
    for m in _ROLE_PATTERNS[1].finditer(system):
        role = _clean_role(m.group(1))
        if role and role.lower() not in _GENERIC:
            cands.append(role)
    if not cands:
        return None
    specific = [c for c in cands if _SPECIFIC.search(c)]
    return (specific or cands)[0]


def _label_from_system(head: str) -> "str | None":
    return best_role(head)


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
    edges: list[dict] = []
    depth_of: dict[tuple, int] = {}   # each agent's call-tree depth
    source = "flat"

    # Reconstruct the call tree from the transcript: an agent that emits a final
    # answer (no tool calls) has RETURNED; a new agent that appears is a child of
    # whichever agent is still awaiting a delegation (the top of the stack). This
    # works even for a flat wire trace where every call is depth 0 -- and handles
    # a parent that fires several sub-agents from ONE turn (they're siblings, not
    # a chain), which trace-order hand-offs got wrong.
    stack: list[tuple] = []   # agents awaiting a delegation result, root-first
    last_agent: "tuple | None" = None
    pending_tools: list = []  # (tool_name, agent_key) FIFO -> attribute a result
                              # to the agent that actually requested it (correct even
                              # when a peer's turn is interleaved -- e.g. AutoGen group chat)
    open_calls: dict = {}     # agent_key -> [ {name, stems} ] tool calls awaiting a
                              # result; used to attach a fresh sub-agent to the parent
                              # whose call actually spawned it (siblings, not a chain)
    peers: set = set()        # agents classified as group-chat peers (never nested)
    named_by: dict = {}       # agent_key -> explicit name its hand-off gave it
                              # (e.g. Task tool subagent_type='researcher'), used to
                              # label a sub-agent whose own system prompt is generic

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

    def _add_edge(frm: tuple, to: tuple, seq) -> None:
        if not any(ed["_pk"] == (frm, to) for ed in edges):
            edges.append({"_pk": (frm, to), "from": frm, "to": to, "seq": seq})

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
                       "sys_role": meta.get("sys_role", ""), "sys_hash": meta.get("sys_hash", ""),
                       "model": meta.get("model", "") or data.get("model", ""),
                       "tools": list(meta.get("tools", [])), "calls": 0, "depth": depth}
                agents[key] = rec
                order.append(key)
            elif meta.get("tools") and not rec["tools"]:
                rec["tools"] = list(meta.get("tools", []))
            rec["calls"] += 1
            res = e.get("result") if isinstance(e.get("result"), dict) else {}
            tcs = res.get("tool_calls") or []
            has_tc = bool(tcs)
            for tc in tcs:                   # remember who requested each tool
                nm = tc.get("name", "")
                pending_tools.append((nm, key))
                args = tc.get("input") or tc.get("args") or {}
                arg_vals = [args.get(k) for k in _DELEG_ARG_KEYS] if isinstance(args, dict) else []
                # an explicit target name the hand-off carries (Task subagent_type,
                # a handoff's agent/name) -- authoritative when the child's own
                # system prompt is too generic to name it.
                target = next((str(args[k]) for k in ("subagent_type", "name", "agent", "type")
                               if isinstance(args, dict) and args.get(k)), "")
                open_calls.setdefault(key, []).append(
                    {"name": nm, "target": target,
                     "stems": stem_tokens(nm, *[v for v in arg_vals if v])})
            msgs = meta.get("msgs")
            is_peer = key in peers           # peer-ness is sticky across turns
            if key in stack:                 # this agent is resuming: pop those it called
                while stack and stack[-1] != key:
                    stack.pop()
            elif key in depth_of:            # reappearing (was popped) -- keep its depth
                pass
            else:                            # genuine first appearance -- attach to a parent
                # This sub-agent's own naming signal (its role + tool set).
                child_stems = stem_tokens(
                    rec.get("sys_role") or _label_from_system(rec.get("sys_head", "")) or "",
                    *rec.get("tools", []))
                # Prefer the ancestor whose OPEN tool call actually spawned this
                # agent -- matched by name/args, so a parent that fans out several
                # siblings in ONE turn attaches them all to itself (not a chain),
                # and a genuine grandchild attaches to the deeper parent.
                m_parent, m_call, m_score = None, None, 0
                for a in reversed(stack):    # deeper agents win ties
                    for call in open_calls.get(a, []):
                        sc = len(child_stems & call["stems"])
                        if sc > m_score:
                            m_parent, m_call, m_score = a, call, sc
                if m_parent is not None:     # spawned by a matched delegation call
                    depth_of[key] = depth_of.get(m_parent, 0) + 1
                    _add_edge(m_parent, key, seq)
                    if m_call.get("target"):
                        named_by[key] = m_call["target"]
                    open_calls[m_parent].remove(m_call)
                else:
                    parent = stack[-1] if stack else None
                    # No name match. A FRESH context (msgs<=1) under an awaiting
                    # parent is still a sequential delegation child; a call that
                    # already carries the shared conversation (msgs>=2) is a PEER
                    # turn -- a group chat (e.g. AutoGen), not a child.
                    is_peer = parent is not None and isinstance(msgs, int) and msgs >= 2
                    if is_peer:
                        peers.add(key)
                        depth_of[key] = depth_of.get(parent, 0)
                    else:
                        depth_of[key] = (depth_of[parent] + 1) if parent is not None else 0
                        if parent is not None:
                            _add_edge(parent, key, seq)
                            # the child couldn't be name-matched (generic prompt, no
                            # tools). Consume the parent's oldest un-consumed hand-off
                            # that named a target, in order, so it can still be labeled.
                            nc = next((c for c in open_calls.get(parent, []) if c.get("target")), None)
                            if nc is not None:
                                named_by[key] = nc["target"]
                                open_calls[parent].remove(nc)
            # a peer does not create delegation nesting, so it never goes on the stack
            if has_tc and not is_peer:
                if not stack or stack[-1] != key:
                    stack.append(key)
            elif not has_tc:                 # final answer: this agent returns
                if stack and stack[-1] == key:
                    stack.pop()
            last_agent = key
            if seq is not None:
                step_agent[str(seq)] = key
        else:
            # a tool / other effect belongs to the agent that REQUESTED it. Match
            # the tool result to the pending tool call by NAME (precise, correct
            # even when a peer's turn interleaves); fall back to native meta.agent,
            # the awaiting stack top, then the last agent to act.
            m = e.get("meta") or {}
            key = None
            if kind.startswith("tool:"):
                tname = kind[5:]
                idx = next((i for i, (n, _a) in enumerate(pending_tools) if n == tname), None)
                if idx is not None:
                    key = pending_tools.pop(idx)[1]
                    # NOTE: do NOT drop this call from open_calls here. A hand-off
                    # tool (transfer_to_X) emits its result BEFORE the target agent
                    # takes over; the call must stay available so that agent can
                    # match it. A leaf tool simply never matches a child anyway.
            if key is None:
                if m.get("agent") and ("name", m["agent"]) in agents:
                    key = ("name", m["agent"])
                else:
                    key = (stack[-1] if stack else None) or last_agent
            if key is not None and seq is not None:
                step_agent[str(seq)] = key

    # assign readable labels + ids, stable in first-appearance order
    systems = data.get("systems") if isinstance(data.get("systems"), dict) else {}
    id_of: dict[tuple, str] = {}
    used: set[str] = set()
    out_agents: list[dict] = []
    for i, key in enumerate(order):
        rec = agents[key]
        if key[0] == "name":
            label = key[1]
        else:
            label = rec.get("sys_role") or _label_from_system(rec["sys_head"]) or f"agent {i + 1}"
            # if the agent's OWN prompt was too generic to name it, fall back to the
            # explicit name its hand-off gave it (Task subagent_type='researcher').
            base_l = re.sub(r"\s*\(\d+\)$", "", str(label)).strip().lower()
            if key in named_by and (base_l in _GENERIC or base_l.startswith("agent ")
                                    or base_l in ("claude agent", "claude code")):
                label = named_by[key].replace("_", " ").replace("-", " ").strip().title()
        base, n = label, 2
        while label in used:  # disambiguate collisions
            label = f"{base} ({n})"; n += 1
        used.add(label)
        aid = f"a{i + 1}"
        id_of[key] = aid
        # The agent's full system prompt (stored once per sys_hash by the proxy);
        # the root of a native run falls back to the trace-level system, and any
        # agent from an older trace without the map falls back to its 160-char head.
        system = systems.get(rec.get("sys_hash", ""))
        if not system and i == 0:
            system = data.get("system", "")
        if not system:
            system = rec.get("sys_head", "")
        system = system or ""
        out_agents.append({
            "id": aid, "label": label, "model": rec["model"],
            "tools": rec["tools"], "calls": rec["calls"],
            "is_root": i == 0, "color": i % 8,
            "system": system, "sys_head": rec.get("sys_head", ""),
            "sys_hash": rec.get("sys_hash", ""),
        })

    # Delegation-tree depth per agent comes straight from the call-stack
    # reconstruction above (root = 0, a delegatee = parent + 1).
    for a, key in zip(out_agents, order):
        a["level"] = depth_of.get(key, 0)

    return {
        "multi": len(out_agents) > 1,
        "agents": out_agents,
        "step_agent": {seq: id_of[key] for seq, key in step_agent.items()},
        "agent_level": {a["id"]: a["level"] for a in out_agents},
        "edges": [{"from": id_of[e["from"]], "to": id_of[e["to"]], "seq": e["seq"]}
                  for e in edges if e["from"] in id_of and e["to"] in id_of],
        "source": source,
    }
