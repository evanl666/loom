"""``loom debug``: a step-debugger UI for an agent run.

Static ``loom studio`` shows a finished trace. ``loom debug`` makes it
*interactive*, like a source debugger:

    loom debug session.loom.json --agent app:agent

opens a page where you step through the run one action at a time (◀ ▶ / arrow
keys), inspect each step's model reasoning, tool call + arguments, and result,
then -- the debugger part -- **edit a turn and re-run it live**: inject a note
into the model's context, or switch the model, at any forkable turn, hit *Fork
& Run*, and the new branch is executed against the real model and shown beside
the original with the first divergence highlighted.

The edit is exactly ``Run.fork(at, edit=...)`` wired to buttons: turns 0..at-1
replay from the log for free; only the divergent tail costs a live call. The
``--agent module:attr`` supplies the agent (and its tools) used for the live
tail; without it the page is read-only (step + inspect, no re-run).
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def steps_for(data: dict) -> list[dict]:
    """The run as an ordered list of inspectable Action dicts.

    Builtin packs are installed first so each step carries its domain StateDiff
    -- the file diff for a coding agent, the row diff for SQL, the DOM diff for a
    browser agent -- i.e. the actual *code/world change* behind the step.
    """
    from .action import actions
    from .multiagent import infer_agents
    from .packs import install_builtin

    install_builtin()
    # Attribute every step to its agent (works for our harness AND proxied
    # third-party multi-agent systems), so the UI can lane/color by agent.
    ia = infer_agents(data)
    label = {a["id"]: a["label"] for a in ia["agents"]}
    color = {a["id"]: a["color"] for a in ia["agents"]}
    step_agent = ia["step_agent"]
    agent_level = ia["agent_level"]
    # Show each user turn as a node (the conversation's roots), interleaved at
    # episode boundaries, so the step list / tree reads as a real dialogue --
    # not just the agent's moves.
    episodes = [e for e in (data.get("episodes") or [data.get("prompt", "")]) if e]
    # A proxy trace's "episodes" also holds internal user-role turns (tool
    # results, framework reminders), so only the first is a real user prompt --
    # interleave follow-ups only for our own (native / live) multi-episode runs.
    if data.get("recorded_via") == "proxy":
        episodes = episodes[:1]
    acts = list(actions(data))
    ends_after = [i for i, a in enumerate(acts) if a.type == "answer" and a.depth == 0]

    def _user_node(text: str) -> dict:
        return {"step": -1, "depth": 0, "type": "user", "intent": str(text),
                "observation": {"text": str(text)}, "nest": 0,
                **({"agent": "you", "agent_id": "user", "agent_color": 3} if ia["multi"] else {})}

    out = []
    ep_i = 0
    if episodes:
        out.append(_user_node(episodes[0])); ep_i = 1
    for idx, a in enumerate(acts):
        d = a.to_dict()
        aid = step_agent.get(str(a.step))
        if aid and ia["multi"]:
            d["agent"] = label.get(aid, aid)
            d["agent_id"] = aid
            d["agent_color"] = color.get(aid, 0)
            # nesting for the tree view: the agent's delegation-tree depth, but a
            # native trace's own recorded depth wins when it goes deeper.
            d["nest"] = max(agent_level.get(aid, 0), a.depth)
        out.append(d)
        # a top-level final answer ends an episode; if another episode follows,
        # its user message opens the next turn.
        if idx in ends_after and ep_i < len(episodes) and idx != ends_after[-1]:
            out.append(_user_node(episodes[ep_i])); ep_i += 1
    # Flag DELEGATION calls (labelled "subagent"). Three signals, most-reliable
    # first: (1) the drain marker actions() leaves on an unpaired hand-off
    # (catches parallel delegations); (2) a delegate tool that DID return a
    # result, detected because a DEEPER agent ran between the call's request and
    # its result (requested_at..step); (3) the immediate next step goes deeper.
    for i, d in enumerate(out):
        if d.get("type") != "call":
            continue
        nest = d.get("nest", 0)
        obs = (d.get("observation") or {}).get("text") or ""
        if obs.startswith("(delegated to sub-agent"):
            d["is_delegation"] = True
            continue
        req = d.get("requested_at", -1)
        if req >= 0 and any(x.get("type") in ("reason", "answer", "call")
                            and req <= x.get("step", -1) < d.get("step", -1)
                            and x.get("nest", 0) > nest for x in out):
            d["is_delegation"] = True
            continue
        for nxt in out[i + 1:]:
            if nxt.get("type") == "user" or (nxt.get("type") == "call"
                    and nxt.get("nest", 0) == nest):
                continue  # skip a user node or a sibling delegation at this level
            if nxt.get("nest", 0) > nest:
                d["is_delegation"] = True
            break

    # Re-anchor a delegation call whose result came back LATE (a paired delegate
    # tool: the sub-agent ran, THEN the result arrived) to right BEFORE its
    # sub-agent's first step, so the tree reads request -> sub-agent -> return
    # instead of showing the hand-off after the work it triggered.
    def _child_start(deleg):
        req, res, n = deleg.get("requested_at", -1), deleg.get("step", -1), deleg.get("nest", 0)
        for x in out:
            if x is deleg:
                continue
            st = x.get("step", -1)
            if req <= st < res and x.get("nest", 0) > n:
                return x  # the sub-agent's first step (out is in result-seq order)
        return None

    before = {}   # id(child_first_step) -> the delegation to insert before it
    move_ids = set()
    for d in out:
        if d.get("is_delegation") and d.get("requested_at", -1) >= 0:
            c = _child_start(d)
            if c is not None and id(c) not in before:
                before[id(c)] = d
                move_ids.add(id(d))
                # record which sub-agent this hand-off spawned, so the context
                # panel can show that agent's delegated task (reliable for a
                # late-returning delegation; parallel drain hand-offs stay
                # unmapped -- their child is structurally ambiguous on the wire).
                if c.get("agent_id"):
                    d["delegates_to"] = c["agent_id"]
    if move_ids:
        reordered = []
        for d in out:
            if id(d) in move_ids:
                continue  # skip its original (late) position
            if id(d) in before:
                reordered.append(before[id(d)])  # the hand-off, right before the child
            reordered.append(d)
        out = reordered

    # Best-effort SEMANTIC mapping for parallel drain delegations, which are
    # structurally ambiguous on the wire (a coordinator firing ask_research +
    # ask_support at once -- nothing says which spawned which child). Match each
    # unmapped hand-off to the child the parent spawned whose name/role best
    # overlaps the delegate tool name + task text ("ask_research" -> "Research
    # Lead"). Greedy 1:1; only assigns on a real token overlap, never a guess.
    import re as _re

    def _toks(s):
        return set(_re.findall(r"[a-z]{3,}", str(s).lower())) - {
            "the", "and", "ask", "for", "delegate", "task", "work", "agent",
            "sub", "call", "lead", "team", "coworker", "assistant"}

    mapped = {d.get("delegates_to") for d in out if d.get("delegates_to")}
    label_of = {a["id"]: a["label"] for a in ia["agents"]}
    by_parent: dict = {}
    for d in out:
        if (d.get("type") == "call" and d.get("is_delegation")
                and not d.get("delegates_to") and d.get("agent_id")):
            by_parent.setdefault(d["agent_id"], []).append(d)
    for parent, delegs in by_parent.items():
        children = [e["to"] for e in ia["edges"]
                    if e["from"] == parent and e["to"] not in mapped]
        children = list(dict.fromkeys(children))
        for d in delegs:
            inp = d.get("input") if isinstance(d.get("input"), dict) else {}
            dt = _toks(d.get("tool", "")) | _toks(inp.get("task", ""))
            best, best_score = None, 0
            for cid in children:
                sc = len(dt & _toks(label_of.get(cid, "")))
                if sc > best_score:
                    best, best_score = cid, sc
            if best is not None:
                d["delegates_to"] = best
                children.remove(best)
                mapped.add(best)
    return out


def context_at(data: dict, step: int, acts=None) -> list[dict]:
    """The conversation the model had seen up to (and including) ``step`` --
    the debugger's "current frame": the prompt, prior reasoning, tool calls, and
    tool results that were in context when this step ran. ``acts`` may be a
    precomputed ``actions(data)`` to avoid re-parsing (see static_data)."""
    from .action import actions

    prompt = (data.get("episodes") or [data.get("prompt", "")])[0]
    frame: list[dict] = [{"role": "user", "content": str(prompt), "step": -1}]
    for a in (actions(data) if acts is None else acts):
        if a.step > step:
            break
        if a.type in ("reason", "answer") and a.intent:
            frame.append({"role": "assistant", "content": a.intent, "step": a.step})
        elif a.type == "call":
            import json as _json
            frame.append({"role": "assistant",
                          "content": f"→ call {a.tool}({_json.dumps(a.input, default=str)})",
                          "step": a.step})
            if a.observation is not None and a.observation.text:
                frame.append({"role": "tool", "content": a.observation.text[:2000],
                              "step": a.step, "tool": a.tool})
        elif a.type == "ask-human":
            frame.append({"role": "human", "content": (a.observation.text if a.observation else ""),
                          "step": a.step})
    return frame


def explain_step(data: dict, step: int, model) -> dict:
    """Explain ONE step to a developer -- tightly. The old version reused the
    whole-run copilot, so it narrated the run (and often the wrong step); this
    gives the model only THIS step + the context the agent had seen up to it, and
    forbids describing later steps. Returns {reply} or {error}."""
    import json as _json

    from .action import actions
    from .judge import _resolve

    acts = actions(data)
    a = next((x for x in acts if x.step == step), None)
    if a is None:
        return {"reply": "(no such step)"}
    frame = context_at(data, step - 1, acts=acts)  # what it had seen BEFORE this step
    convo = "\n".join(f"{m['role']}: {str(m.get('content', ''))[:300]}" for m in frame[-10:])
    if a.type == "call":
        obs = (a.observation.text or "")[:400] if a.observation else ""
        this = (f"The agent called tool `{a.tool}` with input "
                f"{_json.dumps(a.input, default=str)[:400]} and got back: {obs}")
    elif a.type in ("reason", "answer"):
        this = f"The agent (the model) produced this text: {(a.intent or '')[:600]}"
    else:
        this = f"A {a.type} step."
    system = ("You explain ONE step of an AI agent's run to a developer debugging it. "
              "Explain what the agent did at THIS step and WHY, based ONLY on the "
              "context it had seen so far. 2-3 sentences, concrete. Do NOT narrate "
              "later steps or the whole run -- just this step. If it looks wrong or "
              "risky, say so.")
    user = f"CONTEXT the agent had seen:\n{convo}\n\nTHIS STEP (step {step}):\n{this}"
    try:
        provider = _resolve(model)
        resp = provider.complete(system, [{"role": "user", "content": user}], [])
        return {"reply": (resp.text or "").strip()}
    except Exception as e:  # noqa: BLE001
        return {"error": f"explain error: {type(e).__name__}: {e}"}


def context_delta(data: dict, step: int) -> dict:
    """What CHANGED in the model's context at this step vs the previous model call.

    Flags what agent bugs usually come from: newly-added messages, a tool result
    that dominates the token budget, untrusted text entering context, and
    repeated content (context rot)."""
    from .action import actions
    from .inject import _INJECTION, _is_untrusted

    acts = actions(data)
    prev_texts: set[str] = set()
    added: list[dict] = []
    seen_before = True
    for a in acts:
        if a.type == "call" and a.observation is not None:
            txt = a.observation.text or ""
            tokens = max(1, len(txt) // 4)
            item = {"step": a.step, "tool": a.tool, "tokens": tokens,
                    "untrusted": _is_untrusted(a),
                    "poisoned": bool(_INJECTION.search(txt)),
                    "repeated": txt in prev_texts and len(txt) > 40}
            if a.step <= step:
                if a.step == step or not seen_before:
                    pass
            added.append(item)
            prev_texts.add(txt)
    # items up to this step; the "new" ones are those at this step's turn
    upto = [x for x in added if x["step"] <= step]
    total = sum(x["tokens"] for x in upto) or 1
    for x in upto:
        x["share"] = round(100 * x["tokens"] / total)
    biggest = max(upto, key=lambda x: x["tokens"], default=None)
    return {
        "step": step,
        "items": upto[-8:],  # the recent context
        "dominant": biggest,
        "untrusted_in_context": [x for x in upto if x["untrusted"]],
        "poisoned_in_context": [x for x in upto if x["poisoned"]],
        "repeated": [x for x in upto if x["repeated"]],
        "total_tokens": total,
    }


def copilot_report(data: dict) -> dict:
    """The Debug Copilot: point at the suspicious steps, suggest fork edits and a
    policy patch, and summarize the run -- so you don't have to read the trace."""
    from .action import actions
    from .diagnose import diagnose
    from .scan import scan

    diag = diagnose(data)
    rep = scan(data)
    acts = actions(data)

    suspicious = []
    for a in acts:
        if a.type != "call":
            continue
        reasons = []
        if a.risky:
            reasons.append(f"risky ({a.risk})")
        if a.policy is not None and a.policy.blocked:
            reasons.append("firewall-blocked")
        if set(a.capabilities) & {"money_movement", "destructive", "database_write"}:
            reasons.append("high-impact")
        if reasons:
            suspicious.append({"step": a.step, "turn": (a.replay.turn if a.replay else 0),
                               "tool": a.tool, "why": ", ".join(reasons)})

    fork_edits = [{"turn": s["turn"],
                   "suggestion": f"Try forking at turn {s['turn']} with: "
                                 f"“Do NOT call {s['tool']}; only do what the user asked.”"}
                  for s in suspicious[:3]]
    policy = sorted({f"{s['tool']}*" for s in suspicious})
    summary = (f"This run is graded {rep['grade']}. "
               + (diag.get("suggestion", "") or "No obvious failure category. ")
               + (f" {len(suspicious)} step(s) worth a look." if suspicious
                  else " Nothing suspicious stood out."))
    return {"summary": summary, "grade": rep["grade"],
            "category": diag.get("category", "unknown"),
            "suspicious": suspicious, "fork_edits": fork_edits,
            "policy_suggestion": policy}


def memory_blame(data: dict, step: int) -> dict:
    """Which recalled memories could have influenced the action at ``step``?

    Lists the memory recalls that were in context before the action, flags any
    that carry injected instructions, and points at counterfactual verification
    (loom why --causal) to prove which one actually drove it.
    """
    from .action import actions, effect_dicts
    from .inject import _INJECTION

    acts = actions(data)
    target = next((a for a in acts if a.step == step and a.type == "call"), None)
    influences = []
    for e in effect_dicts(data):
        if e.get("kind") == "memory" and (e.get("seq") or 0) < step:
            text = e["result"] if isinstance(e.get("result"), str) else ""
            influences.append({"step": e.get("seq"),
                               "poisoned": bool(_INJECTION.search(text)),
                               "preview": (text[:220] + "…") if len(text) > 220 else text})
    return {
        "step": step, "tool": target.tool if target else "",
        "turn": (target.replay.turn if target and target.replay else 0),
        "influences": influences,
        "verify": (f"loom why {'<trace>'} --step {step} --causal --agent m:a"
                   if influences else ""),
        "note": ("this action ran after a POISONED memory recall — verify causation"
                 if any(i["poisoned"] for i in influences)
                 else "no memory recalls preceded this action" if not influences
                 else "memory recalls preceded this action; fork to test their effect"),
    }


def _run_summary(data: dict) -> str:
    """A compact, model-readable view of the run for the chat copilot."""
    from .action import actions

    lines = []
    for a in actions(data):
        if a.type == "call":
            import json as _json
            obs = (a.observation.text[:120] if a.observation else "")
            lines.append(f"[{a.step}] turn {a.replay.turn if a.replay else '?'} CALL {a.tool}"
                         f"({_json.dumps(a.input, default=str)[:120]}) "
                         f"caps={','.join(a.capabilities)}{' RISKY' if a.risky else ''} -> {obs}")
        elif a.type in ("reason", "answer"):
            lines.append(f"[{a.step}] {a.type.upper()}: {a.intent[:140]}")
    return "\n".join(lines)


def copilot_chat(data: dict, messages: "list[dict]", model: Any,
                 step: "int | None" = None) -> dict:
    """A conversational debug copilot backed by a real model.

    ``messages`` is the chat history [{role, content}]. The model is given the
    run's steps, diagnosis, and the currently-selected step, and asked to help
    debug. When it proposes an experiment it emits a fenced ```fork block with
    {"turn", "edit"} JSON (and ```policy for a rule) which the UI turns into a
    one-click *Adopt* button. Returns {reply, suggestions}.
    """
    import json as _json
    import re as _re

    from .diagnose import describe_diagnosis, diagnose

    if isinstance(model, str):
        from .agent import _resolve_provider
        model = _resolve_provider(model, None)

    diag = describe_diagnosis(diagnose(data))
    cur = ""
    if step is not None:
        from .action import actions
        a = next((x for x in actions(data) if x.step == step), None)
        if a:
            cur = f"\nCurrently selected: step {a.step} ({a.type} {a.tool}), turn " \
                  f"{a.replay.turn if a.replay else '?'}."
    system = (
        "You are a debugging copilot embedded in an interactive agent-run debugger, "
        "like a senior engineer pairing with the user. Be concise and concrete.\n\n"
        f"USER REQUEST: {str((data.get('episodes') or [data.get('prompt','')])[0])[:300]}\n"
        f"FINAL OUTPUT: {str(data.get('output',''))[:300]}\n\n"
        f"THE RUN, action by action:\n{_run_summary(data)}\n\n"
        f"DIAGNOSIS:\n{diag}{cur}\n\n"
        "When you propose an experiment to TEST a hypothesis, emit a fenced block:\n"
        "```fork\n{\"turn\": <int>, \"edit\": \"<instruction to inject into the model's "
        "context at that turn>\"}\n```\n"
        "The user can adopt it with one click to re-run that turn live (this is how "
        "they change behavior or code -- e.g. \"do NOT issue the refund\" or \"write the "
        "function using binary search instead\"). For a firewall fix, emit:\n"
        "```policy\n{\"deny\": [\"tool*\"], \"confirm\": [\"tool*\"]}\n```")

    resp = model.complete(system, messages, [])
    reply = getattr(resp, "text", "") or ""
    suggestions = []
    for m in _re.finditer(r"```fork\s*(\{.*?\})\s*```", reply, _re.S):
        try:
            d = _json.loads(m.group(1))
            suggestions.append({"kind": "fork", "turn": int(d.get("turn", 0)),
                                "edit": str(d.get("edit", ""))})
        except (ValueError, TypeError):
            pass
    for m in _re.finditer(r"```policy\s*(\{.*?\})\s*```", reply, _re.S):
        try:
            suggestions.append({"kind": "policy", **_json.loads(m.group(1))})
        except (ValueError, TypeError):
            pass
    # strip the raw fenced blocks from the human-facing reply
    clean = _re.sub(r"```(?:fork|policy)\s*\{.*?\}\s*```", "", reply, flags=_re.S).strip()
    return {"reply": clean, "suggestions": suggestions}


def _branch_payload(base_data: dict, branch_data: dict, at: int) -> dict:
    """Original vs branch steps + the first step that differs."""
    a = steps_for(base_data)
    b = steps_for(branch_data)
    diverge = None
    for i in range(min(len(a), len(b))):
        ka = (a[i].get("type"), a[i].get("tool"), json.dumps(a[i].get("input"), sort_keys=True))
        kb = (b[i].get("type"), b[i].get("tool"), json.dumps(b[i].get("input"), sort_keys=True))
        if ka != kb:
            diverge = i
            break
    if diverge is None and len(a) != len(b):
        diverge = min(len(a), len(b))
    return {"branch_steps": b, "branch_output": branch_data.get("output", ""),
            "diverge": diverge, "forked_at": at}


class DebugSession:
    """Holds the trace + optional agent, and executes forks on demand."""

    def __init__(self, trace_path: "str | None" = None, agent: Any = None,
                 copilot_model: str = "", live: Any = None):
        self.trace_path = trace_path
        self.agent = agent
        self.live = live  # a LiveSession -> self.data streams from a running agent
        # model the chat copilot talks to; falls back to the fork agent's model
        self.copilot_model = copilot_model or (getattr(agent, "model", "") if agent else "")
        self.branches: list[dict] = []  # a tree of experiments (Git-branch-style)
        self.comments: list[dict] = []  # step annotations / root-cause labels
        self.macro: list[dict] = []     # recorded human debug actions -> a recipe
        self._static_data: dict = {}
        if live is not None:
            return  # data is served live from the session (see the `data` property)
        with open(trace_path) as f:
            loaded = json.load(f)
        if isinstance(loaded, dict) and loaded.get("loomdebug"):
            # a .loomdebug session artifact: restore base + branches + comments
            self._static_data = loaded.get("base", {})
            self.branches = loaded.get("branches", [])
            self.comments = loaded.get("comments", [])
            self.macro = loaded.get("macro", [])
        else:
            self._static_data = loaded

    @property
    def data(self) -> dict:
        # In live mode the trace is whatever the running agent has produced so
        # far, so every analyzer (steps, agents, taint, copilot) sees it grow.
        return self.live.trace() if self.live is not None else self._static_data

    @data.setter
    def data(self, value: dict) -> None:
        self._static_data = value

    def _agent_for(self, model: str, system: "str | None" = None,
                   tools: "list[str] | None" = None):
        if self.agent is None:
            raise RuntimeError("no --agent given; the run cannot be re-forked live")
        change_model = bool(model) and model != "keep"
        sys_p = self.agent.system if system is None else system
        keep_tools = tools is None
        # unchanged config -> reuse the exact agent (so the replayed prefix's
        # keys still match under strict replay)
        if not change_model and sys_p == self.agent.system and keep_tools:
            return self.agent
        from .agent import Agent
        picked = (list(self.agent.tools.values()) if keep_tools
                  else [t for n, t in self.agent.tools.items() if n in tools])
        # new model -> a model-id string; otherwise reuse the existing provider
        # object (so a scripted/local provider stays itself, not a live lookup)
        model_arg = model if change_model else self.agent.provider
        return Agent(model=model_arg, tools=picked, system=sys_p, max_steps=self.agent.max_steps)

    def fork(self, at: int, append: str = "", model: str = "keep",
             system: "str | None" = None, tools: "list[str] | None" = None,
             set_results: "dict | None" = None) -> dict:
        """Replay 0..at-1 from the log, apply the edits (context / system / tools /
        injected tool results), run the tail live.

        ``set_results`` = {step: fake_result} FAULT-INJECTS a tool result: the
        replayed prefix serves the fake value, so the live tail reacts to "what
        if this tool had returned X?" -- an error, empty, or hostile output --
        without touching the agent's code.
        """
        from .trace import Run

        agent = self._agent_for(model, system, tools)
        base = Run.load(self.trace_path, agent=agent)
        if set_results:
            want = {int(k): v for k, v in set_results.items()}
            for e in base.log:
                if e.seq in want:
                    e.result = want[e.seq]
        edit = None
        if append.strip():
            text = append  # captured for the callback
            edit = lambda ctx: ctx.add_user(text, source="debugger")  # noqa: E731
        branch = base.fork(at=at, edit=edit)
        bdata = branch.to_dict()
        payload = _branch_payload(self.data, bdata, at)
        # record the branch in the tree (Git-branch-style experiment history)
        from .cost import analyze_cost
        from .diff import score_breakdown
        label = (append[:40] or (f"system: {system[:24]}" if system else "")
                 or (f"tools: {','.join(tools)}" if tools else "")
                 or (f"fault@{','.join(map(str, sorted(set_results)))}" if set_results else "")
                 or (f"model: {model}" if model != "keep" else "re-run"))
        node = {"id": len(self.branches) + 1, "at": at, "label": label,
                "model": model, "output": bdata.get("output", "")[:120],
                "score": score_breakdown(bdata)["overall"],
                "tokens": analyze_cost(bdata)["total_tokens"],
                "diverge": payload["diverge"], "_data": bdata}
        self.branches.append(node)
        payload["branch_id"] = node["id"]
        return payload

    def _branch_data(self, bid: int) -> "dict | None":
        """The trace for a branch id (0 = the original run)."""
        if bid == 0:
            return self.data
        for n in self.branches:
            if n["id"] == bid:
                return n.get("_data")
        return None

    def compare(self, a_id: int, b_id: int) -> dict:
        """Side-by-side diff of two branches (0 = the original run) -- the
        experimentation payoff of the branch tree: fork three ways, then SEE
        exactly where they diverge, and which won on score/tokens."""
        from .cost import analyze_cost
        from .diff import score_breakdown

        da, db = self._branch_data(a_id), self._branch_data(b_id)
        if da is None or db is None:
            raise ValueError("unknown branch id")
        sa, sb = steps_for(da), steps_for(db)
        diverge = None
        rows: list[dict] = []
        for i in range(max(len(sa), len(sb))):
            x, y = (sa[i] if i < len(sa) else None), (sb[i] if i < len(sb) else None)
            kx = (x.get("type"), x.get("tool"), json.dumps(x.get("input"), sort_keys=True)) if x else None
            ky = (y.get("type"), y.get("tool"), json.dumps(y.get("input"), sort_keys=True)) if y else None
            same = kx == ky
            if not same and diverge is None:
                diverge = i
            rows.append({
                "i": i, "same": same,
                "a": {"type": x.get("type"), "tool": x.get("tool"), "risk": x.get("risk")} if x else None,
                "b": {"type": y.get("type"), "tool": y.get("tool"), "risk": y.get("risk")} if y else None,
            })

        def _side(bid: int, d: dict) -> dict:
            lbl = "original run" if bid == 0 else next(
                (n["label"] for n in self.branches if n["id"] == bid), f"branch {bid}")
            return {"id": bid, "label": lbl, "output": (d.get("output") or "")[:600],
                    "score": score_breakdown(d)["overall"],
                    "tokens": analyze_cost(d)["total_tokens"]}

        A, B = _side(a_id, da), _side(b_id, db)
        winner = A["id"] if (A["score"], -A["tokens"]) >= (B["score"], -B["tokens"]) else B["id"]
        return {"a": A, "b": B, "rows": rows, "diverge": diverge, "winner": winner}

    _FIXES = [
        ("add a stop instruction", {"append": "You have enough information. Give the final "
                                              "answer now; do not call any more tools."}),
        ("forbid the risky tool", {"tools_drop_risky": True}),
        ("switch to a stronger model", {"model": "claude-sonnet-5"}),
    ]

    def _proposed_fixes(self) -> "list[tuple[str, dict]]":
        """Ask the copilot model for fixes TAILORED to this trace (vs the canned
        list): each is a context edit to inject at the fork point. Best-effort --
        no copilot model, a bad reply, or an error just means no extra fixes."""
        if not self.copilot_model:
            return []
        import re as _re

        from .judge import _resolve, run_summary

        try:
            provider = _resolve(self.copilot_model)
            system = ("You are a debugging copilot. Given an agent run's transcript, "
                      "propose up to 2 SPECIFIC one-line instructions that, injected "
                      "into the model's context at the failure point, would fix the "
                      "run. Reply with ONLY a JSON array: "
                      '[{"label": "<3-6 words>", "edit": "<the instruction>"}]')
            resp = provider.complete(
                system, [{"role": "user", "content": run_summary(self.data)}], [])
            m = _re.search(r"\[.*\]", resp.text or "", _re.S)
            fixes = json.loads(m.group(0)) if m else []
            return [(f"copilot: {f['label'][:40]}", {"append": str(f["edit"])[:500]})
                    for f in fixes[:2]
                    if isinstance(f, dict) and f.get("label") and f.get("edit")]
        except Exception:  # noqa: BLE001 -- smart fixes are a bonus, never a blocker
            return []

    def auto_fix(self, at: int) -> "list[dict]":
        """Try several fixes at ``at`` and compare them: the canned playbook plus,
        when a copilot model is available, fixes the model tailored to THIS trace."""
        from .action import actions

        risky = sorted({a.tool for a in actions(self.data)
                        if a.type == "call" and (a.risky or set(a.capabilities)
                        & {"money_movement", "destructive", "database_write"})})
        keep = sorted(set(self.agent.tools) - set(risky)) if self.agent else []
        out = []
        for name, spec in list(self._FIXES) + self._proposed_fixes():
            kw: dict = {}
            if "append" in spec:
                kw["append"] = spec["append"]
            if spec.get("tools_drop_risky") and keep:
                kw["tools"] = keep
            if "model" in spec:
                kw["model"] = spec["model"]
            try:
                r = self.fork(at=at, **kw)
                out.append({"fix": name, "output": r["branch_output"][:120],
                            "branch_id": r.get("branch_id"),
                            "score": self.branches[-1]["score"],
                            "tokens": self.branches[-1]["tokens"]})
            except Exception as e:  # noqa: BLE001
                out.append({"fix": name, "error": f"{type(e).__name__}: {e}"[:80]})
        out.sort(key=lambda x: (x.get("score", -1), -x.get("tokens", 1e9)), reverse=True)
        return out

    def dry_run(self, step: int, args: "dict | None" = None) -> dict:
        """Run ONLY the tool at ``step`` with (possibly edited) args -- no model
        calls, no full re-run. See the result + world diff before you commit to a
        fork. Runs the real tool, so it has the tool's side effects."""
        import time

        from .action import actions

        if self.agent is None:
            raise RuntimeError("dry-run needs --agent (the tool functions)")
        a = next((x for x in actions(self.data) if x.step == step and x.type == "call"), None)
        if a is None:
            raise ValueError(f"step {step} is not a tool call")
        tool = self.agent.tools.get(a.tool)
        if tool is None:
            raise ValueError(f"no tool named {a.tool!r} on this agent")
        use = args if args is not None else (a.input or {})
        t0 = time.time()
        try:
            result = tool.fn(**use)
            err = ""
        except Exception as e:  # noqa: BLE001 -- surface the tool's own failure
            result, err = "", f"{type(e).__name__}: {e}"
        return {"tool": a.tool, "args": use, "result": str(result)[:3000], "error": err,
                "ms": round((time.time() - t0) * 1000),
                "orig_result": (a.observation.text[:3000] if a.observation else "")}

    def policy_preview(self, deny: "list[str] | None" = None,
                       confirm: "list[str] | None" = None, corpus: str = "") -> dict:
        """What a candidate policy would do to THIS run (and optionally a corpus)."""
        from .action import actions
        from .shield import Shield

        sh = Shield(deny=deny or [], confirm=confirm or [])
        this_run = []
        for a in actions(self.data):
            if a.type != "call":
                continue
            act, rule = sh.classify(a.tool, a.input or {})
            if act in ("deny", "confirm"):
                this_run.append({"step": a.step, "tool": a.tool, "action": act, "rule": rule})
        out = {"this_run": this_run}
        if corpus:
            import os
            from glob import glob

            from .policy_file import simulate
            paths = sorted(glob(os.path.join(corpus, "**", "*.loom.json"), recursive=True))
            sim = simulate(sh, paths)
            out["corpus"] = {"runs": sim["runs"], "would_deny": len(sim["denied"]),
                             "breakages": [d["name"] for d in sim["false_positives"]]}
        return out

    def panels_for(self, step: int) -> "list[dict]":
        """Domain panels contributed by packs for the action at ``step``."""
        from .action import actions
        from .packs import install_builtin, packs

        install_builtin()
        a = next((x for x in actions(self.data) if x.step == step), None)
        if a is None:
            return []
        out = []
        for pack in packs():
            hook = getattr(pack, "debugger_panels", None)
            if hook is None:
                continue
            try:
                for p in (hook(a, self.data) or []):
                    out.append(p)
            except Exception:  # noqa: BLE001 -- a bad pack panel never breaks the debugger
                continue
        return out

    def export_session(self) -> dict:
        """A .loomdebug artifact: base trace + branches + comments + macro."""
        return {"loomdebug": 1, "base": self.data, "branches": self.branches,
                "comments": self.comments, "macro": self.macro,
                "model": getattr(self.agent, "model", "") if self.agent else ""}

    def next_model_turn_after(self, step: int) -> "int | None":
        """The turn index of the first top-level model call after ``step`` --
        the fork point where a tool-result override at ``step`` takes effect."""
        from .action import actions

        for a in actions(self.data):
            if a.step > step and a.type in ("reason", "answer") and a.replay \
                    and a.replay.forkable:
                return a.replay.turn
        return None


# -- HTTP layer -------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence per-request stderr noise
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict) -> None:
        self._send(code, json.dumps(payload).encode(), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        sess: DebugSession = self.server.session  # type: ignore[attr-defined]
        if self.path.split("?", 1)[0] == "/":
            self._send(200, _PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/api/run":
            self._json(200, {
                "prompt": sess.data.get("prompt", ""),
                "output": sess.data.get("output", ""),
                "model": sess.data.get("model", ""),
                "steps": steps_for(sess.data),
                "can_fork": sess.agent is not None,
                "can_chat": bool(sess.copilot_model),
                "live": sess.live is not None,
                "running": bool(sess.live and sess.live.running),
                "system": (getattr(sess.agent, "system", "") if sess.agent
                           else sess.data.get("system", "")),
                "all_tools": (sorted(sess.agent.tools) if sess.agent else []),
            })
        elif self.path == "/api/live":
            if sess.live is None:
                self._json(400, {"error": "not a live session"})
            else:
                self._json(200, sess.live.snapshot())
        elif self.path.startswith("/api/breaks"):
            from urllib.parse import parse_qs, urlparse
            from .breakpoint import find_all_breaks
            cond = parse_qs(urlparse(self.path).query).get("cond", [""])[0]
            try:
                hits = find_all_breaks(sess.data, cond) if cond else []
            except Exception:  # noqa: BLE001
                hits = []
            self._json(200, {"steps": hits})
        elif self.path == "/api/copilot":
            self._json(200, copilot_report(sess.data))
        elif self.path == "/api/branches":
            lean = [{k: v for k, v in n.items() if not k.startswith("_")}
                    for n in sess.branches]
            self._json(200, {"branches": lean})
        elif self.path == "/api/agents":
            from .multiagent import infer_agents
            self._json(200, infer_agents(sess.data))
        elif self.path.startswith("/api/branch?"):
            from urllib.parse import parse_qs, urlparse
            try:
                bid = int(parse_qs(urlparse(self.path).query).get("id", ["0"])[0])
            except (TypeError, ValueError):
                bid = 0
            bd = sess._branch_data(bid)
            if bd is None:
                self._json(404, {"error": "unknown branch id"})
            else:
                lbl = "original run" if bid == 0 else next(
                    (n["label"] for n in sess.branches if n["id"] == bid), f"branch {bid}")
                self._json(200, {"id": bid, "label": lbl,
                                 "steps": steps_for(bd), "output": bd.get("output", "")})
        elif self.path.startswith("/api/compare"):
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            try:
                a_id = int(q.get("a", ["0"])[0]); b_id = int(q.get("b", ["0"])[0])
                self._json(200, sess.compare(a_id, b_id))
            except (TypeError, ValueError) as e:
                self._json(400, {"error": str(e)})
        elif self.path == "/api/export":
            self._send(200, json.dumps(sess.export_session(), indent=2).encode(),
                       "application/json")
        elif self.path.startswith("/api/panels"):
            from urllib.parse import parse_qs, urlparse
            try:
                step = int(parse_qs(urlparse(self.path).query).get("step", ["0"])[0])
            except (TypeError, ValueError):
                step = 0
            self._json(200, {"panels": sess.panels_for(step)})
        elif self.path == "/api/rootcause":
            from .rootcause import first_bad_step
            self._json(200, first_bad_step(sess.data))
        elif self.path.startswith("/api/delta"):
            from urllib.parse import parse_qs, urlparse
            try:
                step = int(parse_qs(urlparse(self.path).query).get("step", ["0"])[0])
            except (TypeError, ValueError):
                step = 0
            self._json(200, context_delta(sess.data, step))
        elif self.path.startswith("/api/autofix"):
            from urllib.parse import parse_qs, urlparse
            if sess.agent is None:
                self._json(400, {"error": "auto-fix needs --agent"})
                return
            try:
                at = int(parse_qs(urlparse(self.path).query).get("at", ["0"])[0])
                self._json(200, {"fixes": sess.auto_fix(at)})
            except Exception as e:  # noqa: BLE001
                self._json(502, {"error": f"auto-fix failed: {e}"})
        elif self.path.startswith("/api/blame"):
            from urllib.parse import parse_qs, urlparse
            try:
                step = int(parse_qs(urlparse(self.path).query).get("step", ["0"])[0])
            except (TypeError, ValueError):
                step = 0
            self._json(200, memory_blame(sess.data, step))
        elif self.path.startswith("/api/context"):
            from urllib.parse import parse_qs, urlparse
            try:
                step = int(parse_qs(urlparse(self.path).query).get("step", ["0"])[0])
            except (TypeError, ValueError):
                step = 0
            self._json(200, {"frame": context_at(sess.data, step)})
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        sess: DebugSession = self.server.session  # type: ignore[attr-defined]
        if self.path in ("/api/dryrun", "/api/policypreview", "/api/comment", "/api/macro"):
            try:
                length = int(self.headers.get("content-length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                assert isinstance(body, dict)
            except (ValueError, AssertionError):
                self._json(400, {"error": "bad json body"})
                return
            if self.path == "/api/dryrun":
                try:
                    self._json(200, sess.dry_run(int(body.get("step", 0)), body.get("args")))
                except (ValueError, RuntimeError) as e:
                    self._json(400, {"error": str(e)})
                except Exception as e:  # noqa: BLE001
                    self._json(502, {"error": f"dry-run failed: {e}"})
            elif self.path == "/api/policypreview":
                self._json(200, sess.policy_preview(body.get("deny"), body.get("confirm"),
                                                    str(body.get("corpus", ""))))
            elif self.path == "/api/comment":
                sess.comments.append({"step": body.get("step"), "text": str(body.get("text", ""))[:2000],
                                      "label": str(body.get("label", ""))[:40]})
                self._json(200, {"ok": True, "comments": len(sess.comments)})
            else:  # /api/macro
                sess.macro.append({"action": str(body.get("action", ""))[:40],
                                   "detail": str(body.get("detail", ""))[:200]})
                self._json(200, {"ok": True})
            return
        if self.path == "/api/chat":
            try:
                length = int(self.headers.get("content-length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                assert isinstance(body, dict)
            except (ValueError, AssertionError):
                self._json(400, {"error": "bad json body"})
                return
            if not sess.copilot_model:
                self._json(400, {"error": "no copilot model -- start with --copilot-model MODEL "
                                          "or --agent (uses its model)"})
                return
            try:
                out = copilot_chat(sess.data, body.get("messages") or [],
                                   sess.copilot_model, step=body.get("step"))
                self._json(200, out)
            except Exception as e:  # noqa: BLE001
                self._json(502, {"error": f"copilot error: {type(e).__name__}: {e}"})
            return
        if self.path == "/api/ask":
            if sess.live is None:
                self._json(400, {"error": "not a live session"})
                return
            try:
                length = int(self.headers.get("content-length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                prompt = str(body.get("prompt", "")).strip()
            except (ValueError, TypeError):
                self._json(400, {"error": "bad json body"})
                return
            if not prompt:
                self._json(400, {"error": "empty prompt"})
                return
            started = sess.live.ask(prompt)
            self._json(200, {"started": started, "running": sess.live.running})
            return
        if self.path == "/api/assert":
            from .assertions import check_assertions
            try:
                length = int(self.headers.get("content-length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                # the copilot model doubles as the judge for `judge:` lines
                self._json(200, check_assertions(sess.data, str(body.get("q", "")),
                                                 judge=sess.copilot_model or None))
            except (ValueError, TypeError) as e:
                self._json(400, {"error": str(e)})
            return
        if self.path == "/api/explain":
            try:
                length = int(self.headers.get("content-length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                step = int(body.get("step", 0))
            except (ValueError, TypeError):
                self._json(400, {"error": "bad json body"})
                return
            if not sess.copilot_model:
                self._json(400, {"error": "no copilot model -- start with --copilot-model MODEL or --agent"})
                return
            self._json(200, explain_step(sess.data, step, sess.copilot_model))
            return
        if self.path != "/api/fork":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("content-length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            assert isinstance(body, dict)
        except (ValueError, AssertionError):
            self._json(400, {"error": "bad json body"})
            return
        try:
            at = int(body.get("at", 0))
        except (TypeError, ValueError):
            self._json(400, {"error": "'at' must be an integer turn"})
            return
        system = body.get("system")
        tools = body.get("tools")
        sr = body.get("set_results")
        try:
            result = sess.fork(at, str(body.get("append", "")), str(body.get("model", "keep")),
                               system=system if isinstance(system, str) else None,
                               tools=tools if isinstance(tools, list) else None,
                               set_results=sr if isinstance(sr, dict) else None)
            self._json(200, result)
        except (IndexError, RuntimeError, ValueError) as e:
            self._json(400, {"error": str(e)})
        except Exception as e:  # noqa: BLE001 -- a live-call failure shouldn't kill the server
            self._json(502, {"error": f"fork failed: {type(e).__name__}: {e}"})


class DebugServer:
    """Serves the step-debugger for one trace. Bind port 0 to pick a free port."""

    def __init__(self, trace_path: "str | None" = None, agent: Any = None,
                 port: int = 8790, host: str = "127.0.0.1", copilot_model: str = "",
                 live: Any = None):
        self.session = DebugSession(trace_path, agent, copilot_model=copilot_model, live=live)
        self.httpd = ThreadingHTTPServer((host, port), _Handler)
        self.httpd.session = self.session  # type: ignore[attr-defined]
        self.httpd.daemon_threads = True

    @property
    def port(self) -> int:
        return self.httpd.server_address[1]

    def serve_forever(self) -> None:
        self.httpd.serve_forever()

    def shutdown(self) -> None:
        self.httpd.shutdown()


_PAGE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Loom debugger</title><style>
*{box-sizing:border-box}
body{margin:0;font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;color:#e6e6e6;background:#0d0d0f}
header{padding:10px 16px;border-bottom:1px solid #26262b;background:#161619;display:flex;gap:14px;align-items:baseline;flex-wrap:wrap}
header b{font-size:14px}.muted{color:#8a8a92}
#wrap{display:flex;height:calc(100vh - 46px)}
#steps{width:300px;border-right:1px solid #26262b;overflow:auto;flex:none}
.step{padding:7px 12px;border-bottom:1px solid #1c1c20;cursor:pointer;display:flex;gap:8px;align-items:center}
.step:hover{background:#161619}.step.cur{background:#1e2a3a;border-left:3px solid #4a9eff;padding-left:9px}
.step.brk{box-shadow:inset 3px 0 0 #e5484d}.step.brk::after{content:"⏹";margin-left:auto;color:#e5484d;font-size:11px}
.badge{font-size:10px;padding:1px 6px;border-radius:4px;background:#2a2a30;color:#bdbdc4;text-transform:uppercase;letter-spacing:.3px}
.b-call{background:#2d2438;color:#c79bff}.b-answer{background:#1f3a2a;color:#7ee0a0}.b-reason{background:#25333f;color:#7fc3ff}
.b-blocked,.b-ask-human{background:#3a2323;color:#ff9b9b}
.risky{color:#ff6b6b}.depth{color:#6a6a72}
#detail{flex:1;overflow:auto;padding:16px 20px}
#tlbar{border-top:1px solid #26262b;background:#131316;padding:6px 10px;display:flex;gap:8px;align-items:flex-end;position:sticky;bottom:0}
#tlbar #play{padding:2px 9px;font-size:13px}#tlbar #play.on{background:#3a2c1a;border-color:#7a5a2a}
#tlinfo{font-size:10px;color:#8a8a92;white-space:nowrap;padding-bottom:2px}
#timeline{flex:1;display:flex;gap:2px;align-items:flex-end;height:36px;overflow-x:auto}
#timeline span{width:8px;flex:none;border-radius:2px 2px 0 0;cursor:pointer;opacity:.75;transition:opacity .1s}
#timeline span:hover{opacity:1}#timeline span.cur{opacity:1;outline:2px solid #e6e6e6;outline-offset:-1px}
#timeline .reason{background:#7fc3ff}#timeline .call{background:#c79bff}#timeline .answer{background:#7ee0a0}#timeline .risky{background:#ff6b6b}
#toolbar{padding:8px 12px;border-bottom:1px solid #26262b;display:flex;gap:6px;align-items:center;background:#131316;position:sticky;top:0}
button{font:inherit;background:#22222a;color:#e6e6e6;border:1px solid #33333c;border-radius:6px;padding:4px 10px;cursor:pointer}
button:hover{background:#2c2c36}button:disabled{opacity:.4;cursor:default}
.k{color:#8a8a92;margin:14px 0 4px;font-size:11px;text-transform:uppercase;letter-spacing:.4px}
pre{margin:0;background:#161619;border:1px solid #24242a;border-radius:8px;padding:10px 12px;white-space:pre-wrap;word-break:break-word;overflow:auto;max-height:340px}
.chip{display:inline-block;font-size:11px;padding:1px 7px;border-radius:10px;background:#22222a;border:1px solid #33333c;margin:0 4px 4px 0}
.chip.risk{background:#3a2323;color:#ff9b9b;border-color:#5a3030}
textarea,select{font:inherit;width:100%;background:#161619;color:#e6e6e6;border:1px solid #33333c;border-radius:6px;padding:8px}
#fork{margin-top:22px;border-top:1px dashed #33333c;padding-top:14px}
#fork.hidden{display:none}
.cols{display:flex;gap:14px}.cols>div{flex:1;min-width:0}
.branchstep{padding:5px 9px;border-bottom:1px solid #1c1c20;font-size:12px}
.branchstep.new{background:#12261a}.branchstep.div{background:#2a2233;border-left:3px solid #c79bff;padding-left:6px}
kbd{background:#22222a;border:1px solid #33333c;border-radius:4px;padding:0 5px;font-size:11px}
.spin{display:inline-block;animation:s 1s linear infinite}@keyframes s{to{transform:rotate(360deg)}}
.sub2{color:#b7b7bf;margin-bottom:6px}
button.mini{padding:1px 8px;font-size:11px;margin-left:6px}
pre.diff .add{color:#7ee0a0;display:block}pre.diff .del{color:#ff9b9b;display:block}pre.diff .hunk{color:#7fc3ff;display:block}
pre.code{background:#0c0c0f;border-color:#2c2c34;color:#d7d7de;max-height:420px;font-size:12.5px}
.frame{border:1px solid #24242a;border-radius:8px;margin:6px 0;overflow:hidden}
.frame .rl{display:block;font-size:11px;color:#8a8a92;padding:4px 10px;background:#161619;border-bottom:1px solid #22222a}
.frame.curframe{border-color:#4a9eff}.frame.curframe .rl{color:#4a9eff;background:#12202f}
.frame pre{border:0;border-radius:0;max-height:180px;background:#0f0f12}
/* floating right drawer (copilot / branches / assert) */
#drawer{position:fixed;top:0;right:0;bottom:0;width:430px;max-width:92vw;z-index:40;
  background:#141518;border-left:1px solid #2a2c33;box-shadow:-18px 0 50px rgba(0,0,0,.5);
  display:flex;flex-direction:column;transform:translateX(0);transition:transform .2s cubic-bezier(.4,0,.2,1)}
#drawer.hidden{transform:translateX(103%);box-shadow:none}
#drawerhead{display:flex;align-items:center;padding:13px 16px;border-bottom:1px solid #24262c;
  font-size:13px;font-weight:600;letter-spacing:.2px;background:#17191d}
#drawerx{margin-left:auto;background:transparent;border:0;color:#8b8d94;font-size:15px;padding:2px 6px;border-radius:6px}
#drawerx:hover{background:#262931;color:#e7e8ea}
#copilotpanel{flex:1;overflow:auto;padding:14px 16px}
.cop-sum{margin-bottom:8px;color:#cfe3ff}
.chip.jump{cursor:pointer}.chip.jump:hover{border-color:#4a9eff}
.cop-edit{font-size:12px;color:#c79bff;margin:4px 0}
.frame.poison{border-color:#e5484d}.frame.poison .rl{color:#ff9b9b;background:#2a1518}
#chatlog{max-height:280px;overflow:auto;margin:6px 0}
.msg{padding:7px 11px;border-radius:10px;margin:5px 0;max-width:90%;white-space:pre-wrap}
.msg.u{background:#1e2a3a;margin-left:auto;color:#dbeaff}
.msg.a{background:#18181b;border:1px solid #26262b}
.chatbar{display:flex;gap:6px}.chatbar input{flex:1}
button.adopt{background:#2d2438;border-color:#4a3a5c;color:#e0c9ff;margin:6px 6px 2px 0;font-size:12px}
button.adopt:hover{background:#3a2f4a}
.msg.a code{background:#26262b;padding:1px 5px;border-radius:4px}
pre.mdcode{background:#0c0c0f;border:1px solid #26262b;border-radius:8px;padding:9px 11px;margin:6px 0;white-space:pre-wrap;font-size:12px;overflow:auto}
.msg.a ul{margin:4px 0 4px 2px;padding-left:16px}.msg.a li{margin:2px 0}
/* fork panel */
.forkhead{font-size:13px;font-weight:600;margin-bottom:10px}
.fl{display:block;font-size:11px;color:#8a8a92;text-transform:uppercase;letter-spacing:.4px;margin:10px 0 4px}
.forkrow{display:flex;gap:10px;align-items:flex-start;flex-wrap:wrap;margin-top:6px}
.forkrow>label.fl{margin:8px 0 0}
.fdet{flex:1;min-width:220px}.fdet summary{cursor:pointer;color:#8a8a92;font-size:12px;padding:6px 0;user-select:none}
.fdet summary:hover{color:#e6e6e6}
#toolbox{display:flex;flex-wrap:wrap;gap:4px 12px}
label.tk{font-size:12px;color:#cfcfd6;display:flex;align-items:center;gap:5px;cursor:pointer}
#run{margin-top:12px;background:#1d3a5c;border-color:#2c5a8c;color:#dbeaff;font-weight:600;padding:6px 14px}
#run:hover{background:#244a72}
.branchhead{color:#7ee0a0;font-weight:600;margin:14px 0 4px}
#fork textarea,#fork select{margin-top:2px}
button.fault{background:#3a2f1a;border-color:#5c4a2c;color:#ffe0a0;margin-top:8px}
button.fault:hover{background:#4a3c22}
#faultval{background:#161311;border-color:#3a2f1a}
.bnode{border:1px solid #24242a;border-radius:8px;padding:8px 11px;margin:6px 0;font-size:12.5px}
.bmeta{color:#7fc3ff;font-size:11px;margin-left:6px}
#cmprow{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
#cmprow select{background:#161619;color:#e6e6e6;border:1px solid #33333c;border-radius:6px;padding:3px 6px;font:inherit;max-width:190px}
.cmphead{display:flex;gap:8px}.cmphead .cmpside{flex:1;min-width:0}
.cmpside{border:1px solid #24242a;border-radius:8px;padding:7px 10px;font-size:12px;overflow:auto}
.cmpside.win{border-color:#2f7a45;background:#152a1c}
.cmptbl{width:100%;border-collapse:collapse;font-size:12px;margin:6px 0}
.cmptbl th{text-align:left;color:#8a8a92;font-weight:400;padding:2px 6px}
.cmptbl td{padding:2px 6px;border-top:1px solid #1c1c20}
.cmptbl tr.diff{background:#241b1b}.cmptbl tr.dv td{border-top:2px solid #ff6b6b}
#viewbanner{background:#2a2418;border-bottom:1px solid #7a5a2a;color:#e8c98a;padding:5px 12px;font-size:12px;display:flex;gap:10px;align-items:center}
#viewbanner.hidden{display:none}#viewbanner #backorig{padding:2px 8px;font-size:11px}
.bnode[data-view]{cursor:pointer}.bnode[data-view]:hover{border-color:#7fc3ff;background:#14202b}
button.on{background:#1d3a5c;color:#dbeaff}
.lane{font-size:9px;padding:0 5px;border-radius:8px;margin-right:5px;text-transform:uppercase;letter-spacing:.3px}
.lane.l0{background:#25333f;color:#7fc3ff}.lane.l1{background:#2d2438;color:#c79bff}.lane.l2{background:#1f3a2a;color:#7ee0a0}.lane.l3{background:#3a2323;color:#ff9b9b}
.step.swim.d1{margin-left:16px}.step.swim.d2{margin-left:32px}.step.swim.d3{margin-left:48px}
#palette{position:fixed;inset:0;background:rgba(0,0,0,.5);display:flex;justify-content:center;align-items:flex-start;z-index:50}
#palette.hidden{display:none}
#palbox{margin-top:11vh;width:min(560px,90vw);background:#17171b;border:1px solid #3a3a44;border-radius:12px;box-shadow:0 18px 60px rgba(0,0,0,.6);overflow:hidden}
#palin{width:100%;box-sizing:border-box;background:#17171b;border:0;border-bottom:1px solid #2a2a30;color:#e6e6e6;font:inherit;font-size:15px;padding:13px 16px;outline:none}
#pallist{max-height:52vh;overflow:auto}
.palitem{padding:8px 16px;cursor:pointer;display:flex;gap:10px;align-items:center;font-size:13px;border-bottom:1px solid #1c1c20}
.palitem .pk{font-size:10px;color:#8a8a92;text-transform:uppercase;letter-spacing:.3px;min-width:52px}
.palitem.sel{background:#1d3a5c}.palitem:hover{background:#22222a}
.palitem .pm{color:#8a8a92;font-size:11px;margin-left:auto}
#assertwrap textarea{width:100%;box-sizing:border-box;background:#161619;color:#e6e6e6;border:1px solid #33333c;border-radius:8px;padding:9px 11px;font:12px ui-monospace,Menlo,monospace;min-height:92px}
.asr{display:flex;gap:8px;align-items:baseline;padding:4px 8px;border-radius:6px;font-size:12.5px;margin:3px 0}
.asr.ok{background:#152a1c}.asr.fail{background:#2c1a1a}.asr.err{background:#2a2418}
.asr .ai{font-weight:600;min-width:18px}.asr .ad{color:#8a8a92;font-size:11px;margin-left:auto}
#explainbtn{background:#25203a;border-color:#443a66;color:#c9b8ff;margin-top:8px}
#explainbtn:hover{background:#2e2748}
/* ---- visual refresh -------------------------------------------------- */
body{background:#0b0c0e;color:#e7e8ea}
#main{flex:1;display:flex;flex-direction:column;min-width:0}
header{background:#111214;padding:11px 18px;border-bottom:1px solid #202228}
header b{font-size:13.5px;letter-spacing:.2px}
.muted{color:#8b8d94}
#steps{width:288px;border-right:1px solid #202228;background:#0e0f11}
.step{padding:8px 12px;border-bottom:1px solid #17181b;transition:background .08s}
.step:hover{background:#151619}
.step.cur{background:#152232;border-left:3px solid #4a9eff;padding-left:9px}
#detail{padding:18px 22px}
/* toolbar: grouped, roomy, wraps cleanly */
#toolbar{padding:8px 14px;gap:8px;background:#111214;border-bottom:1px solid #202228;flex-wrap:wrap}
#toolbar .tgroup{display:flex;gap:4px;align-items:center;background:#17181b;border:1px solid #24262c;border-radius:9px;padding:3px}
#toolbar .tspring{flex:1}
#toolbar #pos{font-size:11px;white-space:nowrap}
#toolbar #brk{width:250px;background:#0e0f11;border:1px solid #24262c;border-radius:8px;padding:6px 10px;font-size:12px}
#toolbar #brk:focus{outline:none;border-color:#3a6ea5}
button{background:#1c1d21;border:1px solid #2a2c33;border-radius:7px;padding:5px 11px;transition:background .1s,border-color .1s}
button:hover{background:#26282e;border-color:#34363e}
#toolbar .tgroup button{border:0;background:transparent;padding:5px 9px;border-radius:6px}
#toolbar .tgroup button:hover{background:#26282e}
button.ico{min-width:32px;text-align:center}
button.accent{background:#173a5e;border:1px solid #235a8c;color:#dbeaff;font-weight:600}
#toolbar .tgroup button.accent{background:#173a5e}
button.accent:hover{background:#1d4a76;border-color:#2c6ba5}
button.on{background:#173a5e !important;color:#dbeaff}
.k{color:#7d7f87;margin:16px 0 6px;font-size:10.5px;font-weight:600}
.badge{border-radius:5px}
.chip{border-radius:7px;background:#1a1b1f;border-color:#2a2c33}
pre{background:#0e0f11;border-color:#212228;border-radius:9px}
.frame{border-color:#212228;border-radius:9px}
.frame .rl{background:#141518}
.msg{border-radius:12px}.msg.a{background:#151619;border-color:#212228}
.chatbar input{background:#0e0f11;border:1px solid #24262c;border-radius:8px;padding:7px 10px}
.chatbar input:focus{outline:none;border-color:#3a6ea5}
#tlbar{background:#101113;border-top:1px solid #202228}
/* multi-agent attribution: per-agent color chips, lanes, overview */
.atag{font-size:9.5px;padding:1px 6px;border-radius:6px;margin:0 5px 0 1px;letter-spacing:.2px;white-space:nowrap;font-weight:600}
.lane{font-weight:600}
.anode{border:1px solid #24262c;border-left-width:3px;border-radius:9px;padding:9px 12px;margin:7px 0}
.edge{font-size:12px;padding:3px 0;color:#c9cbd2}
.c0{background:#16283d;color:#8cc2ff;border-left-color:#3f7fd0}
.c1{background:#2a1f38;color:#c9a4ff;border-left-color:#8a5fd0}
.c2{background:#123528;color:#7fe0ad;border-left-color:#2f9d6b}
.c3{background:#3a2a16;color:#ffcf8f;border-left-color:#c78a3a}
.c4{background:#3a1f27;color:#ff9db0;border-left-color:#c74f6b}
.c5{background:#123a3a;color:#7fe0e0;border-left-color:#2f9d9d}
.c6{background:#2e2e14;color:#dbe07f;border-left-color:#9d9d2f}
.c7{background:#2a2333;color:#b8a4d0;border-left-color:#6f5a8c}
.anode .chip{background:#0e0f11}
/* tree view */
.treehead{display:flex;gap:6px;align-items:center;cursor:pointer;padding:6px 8px;position:relative;
  border-bottom:1px solid #17181b;background:#0e0f11;user-select:none;font-size:12px}
.treehead:hover{background:#141518}
.treehead .tw{color:#8b8d94;font-size:10px;width:10px}
.tstep{position:relative}
.tguide{position:absolute;top:0;bottom:0;width:1px;background:#2a2c33}
.step.tstep{border-bottom:1px solid #141518}
.b-user{background:#3a2a16;color:#ffcf8f}
.step .b-user+*{color:#e8c98a}
/* live session bar */
#livebar{display:flex;gap:8px;align-items:center;padding:8px 14px;border-bottom:1px solid #202228;background:#101418}
#livebar.hidden{display:none}
#livebar #askin{flex:1;background:#0e0f11;border:1px solid #24262c;border-radius:8px;padding:8px 11px;color:#e7e8ea;font:inherit}
#livebar #askin:focus{outline:none;border-color:#3a6ea5}
#livebar #askin:disabled{opacity:.5}
.livedot{width:9px;height:9px;border-radius:50%;background:#2f9d6b;flex:none}
.livedot.busy{background:#e5a54a;animation:pulse 1s ease-in-out infinite}
@keyframes pulse{50%{opacity:.35}}
/* ============ CLEAN LIGHT THEME (overrides the dark rules above) ============ */
body{background:#f6f7f9;color:#1f2328}
.muted{color:#8a9099}
header{background:#fff;border-bottom:1px solid #e6e8eb}
header b{color:#1f2328}
#steps{background:#fbfbfc;border-right:1px solid #e6e8eb}
.step{border-bottom:1px solid #f0f1f3}
.step:hover{background:#f0f2f5}
.step.cur{background:#e8f0fe;border-left:3px solid #2563eb}
#toolbar{background:#fff;border-bottom:1px solid #e6e8eb}
#toolbar .tgroup{background:#f3f4f6;border:1px solid #e6e8eb}
#toolbar .tgroup button{background:transparent;color:#3a4048}
#toolbar .tgroup button:hover{background:#e6e8eb}
#toolbar #brk{background:#fff;border:1px solid #d6d9de;color:#1f2328}
button{background:#fff;border:1px solid #d6d9de;color:#2a2f36}
button:hover{background:#f0f2f5;border-color:#c4c8ce}
button.accent,#toolbar .tgroup button.accent{background:#2563eb;border-color:#2563eb;color:#fff}
button.accent:hover{background:#1d4fd7}
button.on,#toolbar .tgroup button.on{background:#e8f0fe !important;color:#2563eb;border-color:#bcd2fb}
#detail{color:#1f2328}
.k{color:#8a9099;font-weight:600}
pre{background:#f6f7f9;border:1px solid #e6e8eb;color:#1f2328}
pre.code{background:#f8fafc;border-color:#e2e8f0;color:#0f172a}
.chip{background:#f0f2f5;border:1px solid #e0e3e8;color:#3a4048}
.chip.risk{background:#fdecec;border-color:#f5c6c6;color:#c0392b}
.frame{border:1px solid #e6e8eb;background:#fff}.frame .rl{background:#f6f7f9;color:#6b7280;border-bottom:1px solid #eef0f2}
.frame.curframe{border-color:#2563eb}.frame.curframe .rl{background:#e8f0fe;color:#2563eb}
.frame pre{background:#fbfbfc}
.risky{color:#e5484d}.depth{color:#b6bcc4}
/* badges: model / tool / subagent / you -- soft pastel + darker ink */
.badge{border:1px solid transparent}
.b-answer{background:#e8f0fe;color:#1d4fd7}     /* model */
.b-call{background:#e4f6f1;color:#0d7a5f}       /* tool */
.b-sub{background:#efe7fb;color:#6b3fc0}        /* subagent */
.b-reason{background:#eef1f4;color:#5a636e}     /* meta */
.b-user{background:#fdf0da;color:#b26a00}       /* you */
.b-blocked,.b-ask-human{background:#fdecec;color:#c0392b}
.callrow{display:flex;gap:8px;align-items:center;padding:5px 9px;border:1px solid #e6e8eb;border-radius:8px;margin:4px 0;cursor:pointer;background:#fff}
.callrow:hover{background:#f0f2f5;border-color:#cdd2d8}
/* the floating drawer + panels */
#drawer{background:#fff;border-left:1px solid #e0e3e8;box-shadow:-12px 0 40px rgba(20,25,35,.12)}
#drawerhead{background:#fff;border-bottom:1px solid #eef0f2;color:#1f2328}
#drawerx{color:#8a9099}#drawerx:hover{background:#f0f2f5;color:#1f2328}
.cop-sum{color:#1f2937}
.msg.a{background:#f6f7f9;border:1px solid #e6e8eb;color:#1f2328}
.msg.u{background:#e8f0fe;color:#1d4fd7}
.msg.a code{background:#eceff2}
pre.mdcode{background:#f8fafc;border-color:#e2e8f0}
.chatbar input,#drawer input,#drawer textarea,#drawer select{background:#fff;border:1px solid #d6d9de;color:#1f2328}
.bnode{border:1px solid #e6e8eb;background:#fff}.bmeta{color:#2563eb}
.anode{border-color:#e6e8eb}
/* tree */
.treehead{background:#fbfbfc;border-bottom:1px solid #f0f1f3;color:#1f2328}
.treehead:hover{background:#f0f2f5}.treehead .tw{color:#8a9099}
.tguide{background:#e0e3e8}.step.tstep{border-bottom:1px solid #f4f5f7}
/* timeline + live bar */
#tlbar{background:#fff;border-top:1px solid #e6e8eb}
#tlinfo,#livestat{color:#8a9099}
#livebar{background:#f0f7ff;border-bottom:1px solid #d8e6fb}
#livebar #askin{background:#fff;border:1px solid #cdd8e8;color:#1f2328}
/* command palette */
#palbox{background:#fff;border:1px solid #e0e3e8;box-shadow:0 20px 60px rgba(20,25,35,.22)}
#palin{background:#fff;color:#1f2328;border-bottom:1px solid #eef0f2}
.palitem{border-bottom:1px solid #f0f1f3}.palitem.sel{background:#e8f0fe}.palitem:hover{background:#f0f2f5}
.palitem .pk,.palitem .pm{color:#8a9099}
kbd{background:#f0f2f5;border:1px solid #d6d9de;color:#3a4048}
#explainbtn{background:#efe7fb;border:1px solid #ddccf5;color:#6b3fc0}#explainbtn:hover{background:#e6d9fa}
button.fault{background:#fff6e6;border:1px solid #f3dca6;color:#b26a00}button.fault:hover{background:#fdeecd}
.asr.ok{background:#e9f7ef}.asr.fail{background:#fdecec}.asr.err{background:#fef5e6}
.cmpside.win{border-color:#0d7a5f;background:#e9f7ef}
.cmptbl tr.diff{background:#fdf0f0}
#viewbanner{background:#fff6e6;border-bottom:1px solid #f3dca6;color:#8a5a00}
.atag{filter:none}
</style></head><body>
<header><b>🔬 Loom debugger</b><span class="muted" id="prompt">loading…</span>
<span class="muted" style="margin-left:auto" id="model"></span></header>
<div id="wrap">
  <div id="steps"></div>
  <div id="main">
    <div id="toolbar">
      <div class="tgroup">
        <button id="first" class="ico" title="first (Home)">⏮</button>
        <button id="prev" class="ico" title="prev (←)">◀</button>
        <button id="next" class="ico" title="next (→)">▶</button>
        <button id="last" class="ico" title="last (End)">⏭</button>
      </div>
      <span class="muted" id="pos"></span>
      <input id="brk" placeholder="⏹ breakpoint — tool:send_email · cap:network" title="conditional breakpoint">
      <span class="tspring"></span>
      <div class="tgroup">
        <button id="rootcause" title="jump to the first bad step">🎯 root cause</button>
        <button id="palettebtn" class="ico" title="command palette (⌘K)">⌘K</button>
        <button id="swim" class="ico" title="tree view — nest steps by agent/sub-agent">🌲</button>
        <button id="export" class="ico" title="download a shareable .loomdebug session">💾</button>
      </div>
      <div class="tgroup">
        <button id="agentsbtn" title="agents in this run (multi-agent map)">🕸 agents</button>
        <button id="branches" title="branch tree · compare">🌳 branches</button>
        <button id="assertbtn" title="check behavioural assertions">✔ assert</button>
        <button id="copilot" class="accent" title="AI copilot">🤖 Copilot</button>
      </div>
    </div>
    <div id="livebar" class="hidden">
      <span class="livedot"></span>
      <input id="askin" placeholder="ask the agent — it runs live, steps stream in below…">
      <button id="asksend" class="accent">▶ run</button>
      <span class="muted" id="livestat"></span>
    </div>
    <div id="detail"></div>
    <div id="tlbar"><button id="play" title="watch the run animate">▶</button>
      <span class="muted" id="tlinfo"></span><div id="timeline"></div></div>
  </div>
</div>
<aside id="drawer" class="hidden">
  <div id="drawerhead"><span id="drawertitle">🤖 Copilot</span>
    <button id="drawerx" title="close (Esc)">✕</button></div>
  <div id="copilotpanel"></div>
</aside>
<div id="palette" class="hidden"><div id="palbox">
  <input id="palin" placeholder="jump to a step or run a command — type to filter, ↑↓ Enter, Esc">
  <div id="pallist"></div></div></div>
<script>
const E=s=>String(s==null?"":s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
// Static Studio export: the SAME page, but data is inlined and server-only
// features (fork / live / copilot / assert) are off. See debugger.static_page().
const STATIC=!!window.LOOM_STATIC, SD=window.LOOM_STATIC||{};
function md(t){ // minimal, XSS-safe markdown: escape first, protect code blocks
  const blocks=[];
  t=String(t||"").replace(/```(\w*)\n?([\s\S]*?)```/g,(m,l,c)=>{blocks.push(c.replace(/\n$/,"")); return "~CB~"+(blocks.length-1)+"~CB~";});
  t=E(t);
  t=t.replace(/`([^`]+)`/g,"<code>$1</code>");
  t=t.replace(/^\s*#{1,4}\s+(.*)$/gm,"<b>$1</b>");
  t=t.replace(/\*\*([^*]+)\*\*/g,"<b>$1</b>").replace(/(^|[^*])\*([^*\n]+)\*/g,"$1<i>$2</i>");
  t=t.replace(/^\s*[-*]\s+(.*)$/gm,"<li>$1</li>").replace(/^\s*\d+\.\s+(.*)$/gm,"<li>$1</li>");
  t=t.replace(/(?:<li>[\s\S]*?<\/li>\s*)+/g,m=>"<ul>"+m+"</ul>");
  t=t.replace(/\n/g,"<br>");
  return t.replace(/~CB~(\d+)~CB~/g,(m,i)=>'<pre class="mdcode">'+E(blocks[i])+"</pre>");
}
const J=x=>{try{return JSON.stringify(x,null,2)}catch(e){return String(x)}};
function codeFields(inp){
  // pull the file path + code out of a write/edit tool input so it shows as
  // real source, not buried JSON. Handles content / new_str / code / old_str→new_str.
  if(!inp||typeof inp!=="object") return null;
  const path=inp.path||inp.file_path||inp.filename||inp.file||inp.notebook_path||"";
  let code=null, verb="";
  if(inp.content!=null){code=inp.content; verb="write";}
  else if(inp.new_str!=null){code=(inp.old_str!=null?"- "+inp.old_str+"\n+ ":"")+inp.new_str; verb="edit";}
  else if(inp.code!=null){code=inp.code;}
  if(!path||code==null) return null;
  const rest={}; for(const k in inp) if(!["path","file_path","filename","file","notebook_path","content","new_str","old_str","code"].includes(k)) rest[k]=inp[k];
  return {path, code:String(code), verb, rest:Object.keys(rest).length?J(rest):""};
}
let RUN=null, steps=[], cur=0, canFork=false, CAN_CHAT=false;

async function load(){
  RUN = STATIC ? SD.run : await (await fetch("/api/run")).json();
  steps=RUN.steps; canFork=RUN.can_fork; CAN_CHAT=RUN.can_chat;
  if(STATIC){ // hide server-only controls; this is a frozen, shareable snapshot
    ["copilot","assertbtn","export","brk"].forEach(id=>{const e=document.getElementById(id); if(e)e.style.display="none";});
    const b=document.querySelector("header b"); if(b)b.textContent="🔬 Loom Studio";
  }
  document.getElementById("prompt").textContent=RUN.prompt.slice(0,120);
  document.getElementById("model").textContent=RUN.model||"";
  autoTree();
  renderSteps(); renderTimeline(); select(0);
  const pb=document.getElementById("play"); if(pb) pb.onclick=togglePlay;
  LIVE=RUN.live;
  if(LIVE){
    document.getElementById("livebar").classList.remove("hidden");
    const s=document.getElementById("asksend"), inp=document.getElementById("askin");
    s.onclick=askAgent; inp.onkeydown=e=>{if(e.key==="Enter")askAgent();};
    setLiveStat(steps.length?(steps.length+" steps"):"ready — ask the agent something");
    if(RUN.running) startPolling();
  }
}
function autoTree(){  // turn on the tree when a real hierarchy appears
  const ags=new Set(steps.filter(s=>s.agent_id&&s.agent_id!=="user").map(s=>s.agent_id));
  if(ags.size>1&&!TREE){TREE=true; document.getElementById("swim").classList.add("on");}
}
// ---- live session: ask + stream steps as the agent runs ----
let LIVE=false, POLL=null;
async function askAgent(){
  const inp=document.getElementById("askin"), q=inp.value.trim(); if(!q)return;
  inp.value="";
  const r=await (await fetch("/api/ask",{method:"POST",headers:{"content-type":"application/json"},
    body:JSON.stringify({prompt:q})})).json();
  if(r.error){setLiveStat("⚠ "+r.error);return;}
  startPolling();
}
function startPolling(){setBusy(true); if(POLL)clearInterval(POLL); POLL=setInterval(pollLive,700); pollLive();}
async function pollLive(){
  let snap; try{snap=await (await fetch("/api/live")).json();}catch(e){return;}
  if(snap.error&&!snap.steps){setLiveStat("⚠ "+snap.error);return;}
  const atEnd=cur>=steps.length-1, grew=snap.steps.length!==steps.length;
  steps=snap.steps; autoTree(); renderSteps(); renderTimeline();
  if(grew&&atEnd&&steps.length) select(steps.length-1);
  else document.querySelectorAll(".step").forEach(d=>d.classList.toggle("cur",+d.dataset.i===cur));
  setLiveStat(snap.running?("running · "+steps.length+" steps"):
    (snap.error?("⚠ "+snap.error):("done · "+steps.length+" steps · "+(snap.turns||0)+" turn"+(snap.turns===1?"":"s"))));
  if(!snap.running){setBusy(false); if(POLL){clearInterval(POLL);POLL=null;}}
}
function setBusy(b){
  const d=document.querySelector(".livedot"); if(d)d.classList.toggle("busy",b);
  const i=document.getElementById("askin"),s=document.getElementById("asksend");
  if(i)i.disabled=b; if(s)s.disabled=b;
}
function setLiveStat(t){const e=document.getElementById("livestat"); if(e)e.textContent=t;}
function typeClass(t){return "b-"+t}
// Clean, human labels: model / tool / subagent / you (instead of reason/call/answer).
function stepKind(s){
  if(s.type==="user") return {label:"you", cls:"b-user"};
  if(s.type==="call") return s.is_delegation ? {label:"subagent", cls:"b-sub"} : {label:"tool", cls:"b-call"};
  if(s.type==="ask-human") return {label:"human", cls:"b-user"};
  if(s.type==="meta") return {label:"meta", cls:"b-reason"};
  return {label:"model", cls:"b-answer"};  // reason + answer are the model speaking
}
const LANES=["main agent","subagent","sub-subagent","depth 3"];
// the tool calls a model turn (step i) decided to make: the calls at the same
// nesting level that follow it (deeper sub-agent steps are skipped over).
function callsOfModel(i){
  const s=steps[i], n=s.nest||0, out=[];
  for(let j=i+1;j<steps.length;j++){
    const x=steps[j], xn=x.nest||0;
    if(x.type==="call"&&xn===n) out.push(j);
    else if(xn>n) continue;           // a sub-agent's own steps -- skip
    else break;                        // back to this level, non-call -> done
  }
  return out;
}
function agentTag(s){return s.agent?`<span class="atag c${s.agent_color||0}" title="agent: ${E(s.agent)}">${E(s.agent)}</span>`:"";}
let COLLAPSED={};  // tree: collapsed agent segments, by segment index
function renderSteps(){
  const el=document.getElementById("steps");
  if(TREE){renderTree(el);return;}
  el.innerHTML=steps.map((s,i)=>{
    const risky=s.risk?` <span class="risky" title="${E(s.risk)}">⚠</span>`:"";
    const k=stepKind(s), lbl=s.tool?E(s.tool):"";
    const ind=s.depth?`<span class="depth">${"› ".repeat(s.depth)}</span>`:"";
    return `<div class="step${i===cur?' cur':''}" data-i="${i}"><span class="muted">${s.step}</span>`+
      `<span class="badge ${k.cls}">${k.label}</span>${agentTag(s)}${ind}<span>${lbl}</span>${risky}</div>`;
  }).join("");
  el.querySelectorAll(".step").forEach(d=>d.onclick=()=>select(+d.dataset.i));
}
function renderTree(el){
  // group consecutive same-agent steps into segments; indent by the agent's
  // delegation-tree depth (s.nest) so parent -> sub-agent reads as a tree.
  const segs=[];
  steps.forEach((s,idx)=>{
    const aid=s.agent_id||"main";
    const last=segs[segs.length-1];
    if(last&&last.aid===aid) last.items.push(idx);
    else segs.push({aid,agent:s.agent||"main agent",color:s.agent_color||0,nest:s.nest||0,items:[idx]});
  });
  el.innerHTML=segs.map((seg,si)=>{
    const pad=6+seg.nest*15, col=COLLAPSED[si];
    const guide=seg.nest?`<span class="tguide" style="left:${pad-8}px"></span>`:"";
    const head=`<div class="treehead" style="padding-left:${pad}px" data-seg="${si}">${guide}`+
      `<span class="tw">${col?"▸":"▾"}</span><span class="atag c${seg.color}">${E(seg.agent)}</span>`+
      `<span class="muted" style="font-size:10px">${seg.items.length}</span></div>`;
    const kids=col?"":seg.items.map(idx=>{
      const s=steps[idx], risky=s.risk?` <span class="risky">⚠</span>`:"", k=stepKind(s), lbl=s.tool?E(s.tool):"";
      return `<div class="step tstep${idx===cur?' cur':''}" data-i="${idx}" style="padding-left:${pad+18}px">`+
        `<span class="muted">${s.step}</span><span class="badge ${k.cls}">${k.label}</span> <span>${lbl}</span>${risky}</div>`;
    }).join("");
    return head+kids;
  }).join("");
  el.querySelectorAll(".treehead").forEach(d=>d.onclick=()=>{const i=+d.dataset.seg;COLLAPSED[i]=!COLLAPSED[i];renderSteps();});
  el.querySelectorAll(".tstep").forEach(d=>d.onclick=()=>select(+d.dataset.i));
}
function stepTokens(s){
  const o=s.observation||{};
  if(o.tokens&&(o.tokens.input_tokens||o.tokens.output_tokens)) return (o.tokens.input_tokens||0)+(o.tokens.output_tokens||0);
  return o.text?Math.max(1,Math.round(o.text.length/4)):0;
}
function renderTimeline(){
  const tl=document.getElementById("timeline"); if(!tl)return;
  const toks=steps.map(stepTokens), mx=Math.max(1,...toks);
  tl.innerHTML=steps.map((s,i)=>{
    const h=6+Math.round(28*toks[i]/mx);
    const cls=s.risk?"tk risky":(s.type==="call"?"tk call":s.type==="answer"?"tk answer":"tk reason");
    return `<span class="${cls}${i===cur?" cur":""}" data-i="${i}" style="height:${h}px" title="step ${s.step} ${E(s.tool||s.type)} · ${toks[i]} tok${s.risk?" ⚠"+E(s.risk):""}"></span>`;
  }).join("");
  tl.querySelectorAll("span").forEach(d=>d.onclick=()=>select(+d.dataset.i));
  const total=toks.reduce((a,b)=>a+b,0);
  document.getElementById("tlinfo").textContent=`${steps.length} steps · ${total.toLocaleString()} tok`;
}
let PLAY=null;
function togglePlay(){
  const b=document.getElementById("play");
  if(PLAY){clearInterval(PLAY);PLAY=null;b.textContent="▶";b.classList.remove("on");return;}
  b.textContent="⏸";b.classList.add("on");
  PLAY=setInterval(()=>{ if(cur>=steps.length-1){togglePlay();return;} select(cur+1); },900);
}
function select(i){
  cur=Math.max(0,Math.min(steps.length-1,i));
  document.querySelectorAll(".step").forEach(d=>d.classList.toggle("cur",+d.dataset.i===cur));
  document.querySelectorAll("#timeline span").forEach((d,j)=>d.classList.toggle("cur",j===cur));
  const c=document.querySelector(".step.cur"); if(c)c.scrollIntoView({block:"nearest"});
  document.getElementById("pos").textContent=`step ${cur+1} / ${steps.length}`;
  renderDetail();
  document.getElementById("prev").disabled=cur===0;
  document.getElementById("first").disabled=cur===0;
  document.getElementById("next").disabled=cur===steps.length-1;
  document.getElementById("last").disabled=cur===steps.length-1;
}
function renderDetail(){
  const s=steps[cur], o=s.observation||{}, d=document.getElementById("detail");
  const k=stepKind(s);
  let h=`<span class="badge ${k.cls}">${k.label}</span> `+
        (s.tool?`<b>${E(s.tool)}</b>`:"")+
        ` <span class="muted">step ${s.step}${s.agent?" · "+E(s.agent):""}</span>`;
  if(s.type==="call"){
    // a tool / sub-agent call: INPUT and OUTPUT only (no model reasoning here)
    const cf=codeFields(s.input);
    if(cf){
      h+=`<div class="k">📄 ${E(cf.path)}${cf.verb?` <span class="muted">(${cf.verb})</span>`:""}</div><pre class="code">${E(cf.code)}</pre>`;
      if(cf.rest) h+=`<div class="k">args</div><pre>${E(cf.rest)}</pre>`;
    } else if(s.input!=null){
      h+=`<div class="k">input</div><pre>${E(J(s.input))}</pre>`;
    }
    if(o.text) h+=`<div class="k">output</div><pre>${E(o.text.length>4000?o.text.slice(0,4000)+"\n… (truncated)":o.text)}</pre>`;
  } else if(s.type==="user"){
    h+=`<div class="k">user request</div><pre>${E(s.intent||o.text||"")}</pre>`;
  } else {
    // a model turn: its text, then the tool calls it decided to make
    if(s.intent) h+=`<div class="k">${s.type==="answer"?"final answer":"reasoning"}</div><pre>${E(s.intent)}</pre>`;
    const cj=callsOfModel(cur);
    if(cj.length) h+=`<div class="k">→ decided to call</div>`+cj.map(j=>{
      const x=steps[j], xk=stepKind(x);
      return `<div class="callrow" data-j="${j}"><span class="badge ${xk.cls}">${xk.label}</span> <b>${E(x.tool)}</b> <span class="muted">${E(J(x.input||{}).slice(0,90))}</span></div>`;
    }).join("");
    if(o.text&&!s.intent) h+=`<div class="k">output</div><pre>${E(o.text.slice(0,4000))}</pre>`;
  }
  if(s.type==="call"&&o.text!=null&&canFork&&steps.some(x=>x.step>s.step&&(x.replay||{}).forkable)){
    h+=`<div class="k">🧪 fault injection <span class="muted">— what if this tool had returned something else?</span></div>
      <textarea id="faultval" rows="2" placeholder="a different tool result to test error handling / edge cases / hostile output">${E((o.text||"").slice(0,2000))}</textarea>
      <button id="faultrun" class="fault">🧪 Inject &amp; re-run from here</button><div id="faultbranch"></div>`;
  }
  if(s.state_diff&&s.state_diff.kind&&s.state_diff.kind!=="none"){
    h+=`<div class="k">🌍 world change · ${E(s.state_diff.kind)}</div>`;
    if(s.state_diff.summary) h+=`<div class="sub2">${E(s.state_diff.summary)}</div>`;
    if(s.state_diff.detail) h+=`<pre class="diff">${diffHtml(typeof s.state_diff.detail==="string"?s.state_diff.detail:J(s.state_diff.detail))}</pre>`;
  }
  h+=`<div class="k">🧠 context the model saw here <button id="ctxbtn" class="mini">show</button></div><div id="ctx"></div>`;
  if(CAN_CHAT) h+=`<button id="explainbtn" title="ask the copilot to explain this step">🔎 Explain this step</button><div id="explainout"></div>`;
  if(s.type==="call"&&!STATIC) h+=`<div class="k">🩸 memory blame <button id="blamebtn" class="mini">what influenced this?</button></div><div id="blame"></div>`;
  h+=`<div id="dynpanels"></div>`;  // pack-contributed panels (SQL plan, file, screenshot…)
  if(s.capabilities&&s.capabilities.length) h+=`<div class="k">capabilities</div>`+s.capabilities.map(c=>`<span class="chip">${E(c)}</span>`).join("");
  if(s.risk) h+=`<div class="k">risk</div><span class="chip risk">⚠ ${E(s.risk)}</span>`;
  if(s.policy) h+=`<div class="k">firewall</div><span class="chip">${E(s.policy.action)} ${E(s.policy.rule||"")}</span>`;
  if(o.tokens&&(o.tokens.input_tokens||o.tokens.output_tokens)) h+=`<div class="k">tokens</div><span class="chip">in ${o.tokens.input_tokens||0}</span><span class="chip">out ${o.tokens.output_tokens||0}</span>`;
  const forkable=(s.replay||{}).forkable;
  const mc=(s.replay||{}).turn;
  h+=`<div id="fork" class="${forkable&&canFork?"":"hidden"}">
    <div class="forkhead">🍴 Fork from model call #${mc} <span class="muted">— replay up to here, change something, run the rest live</span></div>
    <label class="fl">inject into context</label>
    <textarea id="append" rows="2" placeholder="a message for the model at this point — e.g. “handle n &lt; 2 correctly” or “do NOT issue the refund”"></textarea>
    <div class="forkrow">
      <label class="fl">model</label>
      <select id="model"><option value="keep">keep (${E(RUN.model||"recorded")})</option>
        <option value="claude-haiku-4-5-20251001">claude-haiku-4-5</option>
        <option value="claude-sonnet-5">claude-sonnet-5</option>
        <option value="claude-opus-4-8">claude-opus-4-8</option></select>
      <details class="fdet"><summary>⚙ system + tools</summary>
        <label class="fl">system prompt</label>
        <textarea id="sysp" rows="3">${E(RUN.system||"")}</textarea>
        <label class="fl">tools available to the fork</label>
        <div id="toolbox">${(RUN.all_tools||[]).map(t=>`<label class="tk"><input type="checkbox" class="tchk" value="${E(t)}" checked> ${E(t)}</label>`).join("")}</div>
      </details>
    </div>
    <button id="run">▶ Fork &amp; Run live</button>
    <button id="autofix" class="fault" title="try several canned fixes and compare">🔧 Auto-fix (try several)</button>
    <div id="branch"></div>
  </div>`;
  if(!canFork&&forkable) h+=`<div class="k muted">re-run disabled — start with <kbd>--agent module:attr</kbd> to fork live</div>`;
  d.innerHTML=h;
  d.querySelectorAll(".callrow").forEach(r=>r.onclick=()=>select(+r.dataset.j));
  const rb=document.getElementById("run"); if(rb) rb.onclick=doFork;
  const af=document.getElementById("autofix"); if(af) af.onclick=doAutoFix;
  const cb=document.getElementById("ctxbtn"); if(cb) cb.onclick=loadContext;
  const bb=document.getElementById("blamebtn"); if(bb) bb.onclick=loadBlame;
  const fr=document.getElementById("faultrun"); if(fr) fr.onclick=faultInject;
  const eb=document.getElementById("explainbtn"); if(eb) eb.onclick=()=>explainStep(s.step);
  loadPanels(s.step);
  if(s.type==="call"&&canFork){const dp=document.getElementById("dryrunbtn"); if(dp)dp.onclick=doDryRun;}
}
async function loadPanels(step){
  const box=document.getElementById("dynpanels"); if(!box)return;
  const r = STATIC ? {panels: SD.panels[step]||[]} : await (await fetch("/api/panels?step="+step)).json();
  let h=r.panels.map(p=>`<div class="k">🧩 ${E(p.title)}</div>`+(p.code!=null?`<pre class="code">${E(p.code)}</pre>`:`<div>${E(p.text||"")}</div>`)).join("");
  const s=steps[cur];
  if(s.type==="call"&&canFork) h+=`<div class="k">▷ dry-run <span class="muted">— run just this tool with edited args (no model call)</span></div>
    <textarea id="dryargs" rows="2">${E(J(s.input||{}))}</textarea><button id="dryrunbtn" class="mini">▷ run tool</button><div id="dryout"></div>`;
  box.innerHTML=h;
  const dp=document.getElementById("dryrunbtn"); if(dp)dp.onclick=doDryRun;
}
async function doDryRun(){
  const s=steps[cur], out=document.getElementById("dryout");
  let args; try{args=JSON.parse(document.getElementById("dryargs").value);}catch(e){out.innerHTML='<pre class="risky">args must be JSON</pre>';return;}
  out.innerHTML='<span class="muted">running…</span>';
  const r=await fetch("/api/dryrun",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({step:s.step,args})});
  const res=await r.json();
  if(!r.ok){out.innerHTML=`<pre class="risky">${E(res.error||"failed")}</pre>`;}
  else out.innerHTML=`<div class="fl">result (${res.ms}ms)${res.error?" · ERROR":""}</div><pre class="code">${E(res.error||res.result)}</pre>`;
}
async function loadBlame(){
  const s=steps[cur], box=document.getElementById("blame");
  document.getElementById("blamebtn").remove();
  box.innerHTML='<span class="muted">tracing…</span>';
  const r=await (await fetch("/api/blame?step="+s.step)).json();
  let h=`<div class="sub2">${E(r.note)}</div>`;
  if(r.influences.length){
    h+=r.influences.map(m=>`<div class="frame${m.poisoned?' poison':''}"><span class="rl">🧠 memory recall @${m.step}${m.poisoned?' ⚠ POISONED':''}</span><pre>${E(m.preview)}</pre></div>`).join("");
    if(r.verify) h+=`<div class="cop-edit">verify causation: <code>${E(r.verify)}</code></div>`;
  }
  box.innerHTML=h;
}
function diffHtml(t){
  return E(t).split("\n").map(l=>{
    if(l.startsWith("+")&&!l.startsWith("+++")) return `<span class="add">${l}</span>`;
    if(l.startsWith("-")&&!l.startsWith("---")) return `<span class="del">${l}</span>`;
    if(l.startsWith("@@")) return `<span class="hunk">${l}</span>`;
    return l;
  }).join("\n");
}
async function loadContext(){
  const s=steps[cur], box=document.getElementById("ctx");
  document.getElementById("ctxbtn").remove();
  box.innerHTML='<span class="muted">loading…</span>';
  let frame;
  if(s.agent_id){
    // MULTI-AGENT: show only THIS agent's own sub-conversation (it never saw the
    // other agents' work), strictly BEFORE this step (its own output isn't input).
    frame=agentFrame(s);
  } else {
    // single agent: the whole prior conversation, strictly before this step.
    frame = STATIC ? (SD.context_all||[]).filter(m=>m.step<s.step)
                   : (await (await fetch("/api/context?step="+(s.step-1))).json()).frame;
  }
  if(!frame.length){box.innerHTML='<span class="muted">(only the user request — this is the first model call)</span>';return;}
  box.innerHTML=frame.map(m=>{
    const role={user:"👤 user",assistant:"🤖 model",tool:"🔧 "+(m.tool||"tool"),human:"🧑 human",task:"📩 delegated task"}[m.role]||m.role;
    return `<div class="frame"><span class="rl">${role}</span><pre>${E((m.content||"").slice(0,1500))}</pre></div>`;
  }).join("");
}
function agentFrame(s){
  const aid=s.agent_id;
  const rootId=(steps.find(x=>x.agent_id&&x.type!=="user")||{}).agent_id;
  const frame=[];
  if(aid===rootId){
    frame.push({role:"user",content:RUN.prompt||"",step:-1});
  } else {
    const deleg=steps.find(x=>x.type==="call"&&x.delegates_to===aid);
    const task=deleg&&deleg.input?(deleg.input.task||J(deleg.input)):"";
    frame.push({role:"task",content:task||"(this sub-agent's task was delegated by its parent)",step:-1});
  }
  for(let j=0;j<cur;j++){            // this agent's OWN prior steps, in order
    const x=steps[j];
    if(x.agent_id!==aid) continue;
    if((x.type==="reason"||x.type==="answer")&&x.intent) frame.push({role:"assistant",content:x.intent,step:x.step});
    else if(x.type==="call"){
      frame.push({role:"assistant",content:"→ call "+x.tool+"("+J(x.input||{})+")",step:x.step});
      const o=x.observation||{};
      if(o.text&&!o.text.startsWith("(delegated")) frame.push({role:"tool",tool:x.tool,content:o.text,step:x.step});
    }
  }
  return frame;
}
async function faultInject(){
  const s=steps[cur];
  const after=steps.find(x=>x.step>s.step&&(x.replay||{}).forkable);
  if(!after){alert("no model call after this step to re-run");return;}
  const at=(after.replay||{}).turn;
  const btn=document.getElementById("faultrun"), bx=document.getElementById("faultbranch");
  btn.disabled=true; btn.innerHTML='<span class="spin">⟳</span> re-running…'; bx.innerHTML="";
  try{
    const val=document.getElementById("faultval").value, sr={}; sr[s.step]=val;
    const r=await fetch("/api/fork",{method:"POST",headers:{"content-type":"application/json"},
      body:JSON.stringify({at,set_results:sr})});
    const res=await r.json();
    if(!r.ok){bx.innerHTML=`<pre class="risky">${E(res.error||"failed")}</pre>`;}
    else{
      const div=res.diverge;
      const bs=res.branch_steps.map((b,i)=>`<div class="branchstep${div!=null&&i>=div?' div':''}"><span class="muted">${b.step}</span> <b>${E(b.tool||b.type)}</b> ${E((b.intent||"").slice(0,60))}</div>`).join("");
      bx.innerHTML=`<div class="branchhead">✅ how the agent reacts to the injected result</div><div class="fl">output</div><pre>${E(res.branch_output)}</pre>${bs}`;
    }
  }catch(e){bx.innerHTML=`<pre class="risky">${E(e)}</pre>`;}
  finally{btn.disabled=false; btn.innerHTML="🧪 Inject &amp; re-run from here";}
}
async function doAutoFix(){
  const s=steps[cur], at=(s.replay||{}).turn;
  const btn=document.getElementById("autofix"), bx=document.getElementById("branch");
  btn.disabled=true; btn.innerHTML='<span class="spin">⟳</span> trying fixes…'; bx.innerHTML="";
  try{
    const r=await fetch("/api/autofix?at="+at); const res=await r.json();
    if(!r.ok){bx.innerHTML=`<pre class="risky">${E(res.error||"failed")}</pre>`;}
    else{
      bx.innerHTML=`<div class="branchhead">🔧 auto-fix results (best first)</div>`+res.fixes.map((f,i)=>
        `<div class="bnode">${i===0?"🏆 ":""}<b>${E(f.fix)}</b> ${f.error?`<span class="risky">${E(f.error)}</span>`:`<span class="bmeta">score ${f.score} · ${(f.tokens||0).toLocaleString()} tok</span><div class="muted">${E(f.output||"")}</div>`}</div>`).join("");
    }
  }catch(e){bx.innerHTML=`<pre class="risky">${E(e)}</pre>`;}
  finally{btn.disabled=false; btn.innerHTML="🔧 Auto-fix (try several)";}
}
async function doFork(){
  const s=steps[cur], at=(s.replay||{}).turn;
  const btn=document.getElementById("run"), bx=document.getElementById("branch");
  btn.disabled=true; btn.innerHTML='<span class="spin">⟳</span> running…';
  bx.innerHTML="";
  try{
    const sysEl=document.getElementById("sysp");
    const payload={at,append:document.getElementById("append").value,model:document.getElementById("model").value};
    if(sysEl && sysEl.value!==(RUN.system||"")) payload.system=sysEl.value;
    const chks=[...document.querySelectorAll(".tchk")];
    const on=chks.filter(c=>c.checked).map(c=>c.value);
    if(chks.length && on.length!==chks.length) payload.tools=on;   // only send if a subset
    const r=await fetch("/api/fork",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify(payload)});
    const res=await r.json();
    if(!r.ok){bx.innerHTML=`<pre class="risky">${E(res.error||"fork failed")}</pre>`;return;}
    const div=res.diverge;
    const bs=res.branch_steps.map((b,i)=>{
      const cls=(div!=null&&i>=div)?"branchstep div":(i>=(div??1e9)?"branchstep new":"branchstep");
      return `<div class="${cls}"><span class="muted">${b.step}</span> <b>${E(b.tool||b.type)}</b> ${E((b.intent||"").slice(0,60))}</div>`;
    }).join("");
    bx.innerHTML=`<div class="branchhead">✅ new branch${div!=null?` · diverges at step ${div}`:""}</div>`+
                 `<div class="fl">output</div><pre>${E(res.branch_output)}</pre>`+
                 `<div class="fl">branch steps</div>${bs}`;
  }catch(e){bx.innerHTML=`<pre class="risky">${E(e)}</pre>`;}
  finally{btn.disabled=false; btn.innerHTML="▶ Fork &amp; Run live";}
}
document.addEventListener("keydown",e=>{
  if((e.metaKey||e.ctrlKey)&&e.key.toLowerCase()==="k"){e.preventDefault();openPalette();return;}
  if(!document.getElementById("palette").classList.contains("hidden")){paletteKey(e);return;}
  if(e.key==="Escape"&&!document.getElementById("drawer").classList.contains("hidden")){closeDrawer();return;}
  if(e.target.tagName==="TEXTAREA"||e.target.tagName==="INPUT")return;
  if(e.key==="ArrowLeft")select(cur-1); else if(e.key==="ArrowRight")select(cur+1);
  else if(e.key==="Home")select(0); else if(e.key==="End")select(steps.length-1);
});
// ---- floating right drawer (copilot / branches / assert) ----
function drawerOpen(view){const d=document.getElementById("drawer");return !d.classList.contains("hidden")&&d.dataset.view===view;}
function openDrawer(view,title){const d=document.getElementById("drawer");d.classList.remove("hidden");d.dataset.view=view;
  document.getElementById("drawertitle").innerHTML=title;return document.getElementById("copilotpanel");}
function closeDrawer(){document.getElementById("drawer").classList.add("hidden");}
// ---- explain this step (copilot) ----
async function explainStep(step){
  const out=document.getElementById("explainout"); if(!out)return;
  out.innerHTML='<div class="muted">asking the copilot…</div>';
  try{
    const r=await (await fetch("/api/explain",{method:"POST",headers:{"content-type":"application/json"},
      body:JSON.stringify({step})})).json();
    out.innerHTML=r.error?`<div class="muted">${E(r.error)}</div>`:`<div class="cop-sum">${md(r.reply||"")}</div>`;
  }catch(_){out.innerHTML='<div class="muted">explain failed</div>';}
}
// ---- agents overview (multi-agent map) ----
async function showAgents(){
  if(drawerOpen("agents")){closeDrawer();return;}
  const p=openDrawer("agents","🕸 Agents");
  const r = STATIC ? SD.agents : await (await fetch("/api/agents")).json();
  if(!r.multi){p.innerHTML='<div class="muted">single-agent run — no sub-agents detected.</div>';return;}
  const src={native:"recorded by Loom's harness (agent names known)",wire:"recovered from the wire (system-prompt fingerprint)",flat:"depth only"}[r.source]||r.source;
  let h=`<div class="muted" style="font-size:11px;margin-bottom:10px">${r.agents.length} agents · ${E(src)}</div>`;
  h+=r.agents.map(a=>`<div class="anode c${a.color}">
      <div><span class="atag c${a.color}">${E(a.label)}</span>${a.is_root?' <span class="muted">root</span>':''}
        <span class="bmeta">${a.calls} call${a.calls===1?'':'s'}</span></div>
      ${a.model?`<div class="muted" style="font-size:11px">model: ${E(a.model)}</div>`:''}
      ${a.tools&&a.tools.length?`<div style="font-size:11px;margin-top:3px">${a.tools.map(t=>`<span class="chip">${E(t)}</span>`).join("")}</div>`:''}
    </div>`).join("");
  if(r.edges.length){
    const lab={}; r.agents.forEach(a=>lab[a.id]=a.label);
    h+=`<div class="k">hand-offs</div>`+r.edges.map(e=>
      `<div class="edge"><span class="chip jump" data-step="${e.seq}">→</span> <b>${E(lab[e.from]||e.from)}</b> delegates to <b>${E(lab[e.to]||e.to)}</b> <span class="muted">@${e.seq}</span></div>`).join("");
  }
  p.innerHTML=h;
  p.querySelectorAll(".jump").forEach(c=>c.onclick=()=>{const i=steps.findIndex(s=>s.step===+c.dataset.step);if(i>=0){closeDrawer();select(i);}});
}
// ---- assertion bar ----
async function showAssert(){
  if(drawerOpen("assert")){closeDrawer();return;}
  const p=openDrawer("assert","✔ Assertions");
  p.innerHTML=`<div class="muted" style="font-size:11px;margin-bottom:8px">one per line — turns your expectations into a repeatable check</div>
    <div id="assertwrap"><textarea id="asrin" placeholder="judge: the agent verified the order before refunding
output contains order 42
never issue_refund*
no blocked
calls get_*
steps < 20
tokens < 50000"></textarea>
    <button id="asrgo" style="margin-top:6px">▶ check</button></div><div id="asrout"></div>`;
  document.getElementById("asrgo").onclick=runAssert;
}
async function runAssert(){
  const q=document.getElementById("asrin").value;
  const out=document.getElementById("asrout"); out.innerHTML='<div class="muted">checking…</div>';
  const r=await (await fetch("/api/assert",{method:"POST",headers:{"content-type":"application/json"},
    body:JSON.stringify({q})})).json();
  if(r.error){out.innerHTML=`<div class="muted">${E(r.error)}</div>`;return;}
  const rows=r.results.map(x=>{
    const cls=x.error?"err":(x.ok?"ok":"fail"), icon=x.error?"⚠":(x.ok?"✓":"✗");
    return `<div class="asr ${cls}"><span class="ai">${icon}</span><span>${E(x.expr)}</span><span class="ad">${E(x.detail||x.error||"")}</span></div>`;
  }).join("");
  const banner=r.total?`<div class="k">${r.passed}/${r.total} passed ${r.all_pass?"🎉":""}</div>`:"";
  out.innerHTML=banner+rows;
}
// ---- command palette (⌘K) ----
let PAL=[], PALSEL=0;
function paletteItems(){
  const items=steps.map((s,i)=>({kind:s.type,label:(s.type==="call"?"🔧 "+s.tool:s.type==="answer"?"✅ final answer":"💭 reasoning"),
    meta:"call "+((s.replay||{}).turn??s.step)+(s.risk?" ⚠"+s.risk:""),run:()=>select(i)}));
  const cmds=[
    {kind:"cmd",label:"🎯 jump to root cause",meta:"first bad step",run:gotoRootCause},
    {kind:"cmd",label:"🕸 agents",meta:"multi-agent map",run:showAgents},
    {kind:"cmd",label:"🌳 branch tree",meta:"compare / walk branches",run:showBranches},
    {kind:"cmd",label:"✔ assertions",meta:"check expectations",run:showAssert},
    {kind:"cmd",label:"🤖 Copilot",meta:"chat",run:loadCopilot},
    {kind:"cmd",label:"▶ play the run",meta:"animate",run:()=>{togglePlay();}},
    {kind:"cmd",label:"🌲 tree view",meta:"nest by agent",run:toggleTree},
    {kind:"cmd",label:"💾 export session",meta:".loomdebug",run:exportSession},
    {kind:"cmd",label:"⏮ first step",meta:"",run:()=>select(0)},
    {kind:"cmd",label:"⏭ last step",meta:"",run:()=>select(steps.length-1)},
  ];
  const use = STATIC ? cmds.filter(c=>!/assert|copilot|export session/i.test(c.label)) : cmds;
  return items.concat(use);
}
function openPalette(){
  const p=document.getElementById("palette"); p.classList.remove("hidden");
  const inp=document.getElementById("palin"); inp.value=""; PALSEL=0;
  renderPalette(""); inp.focus();
  inp.oninput=()=>{PALSEL=0;renderPalette(inp.value);};
}
function closePalette(){document.getElementById("palette").classList.add("hidden");}
function renderPalette(q){
  const all=paletteItems(); q=q.toLowerCase().trim();
  PAL=q?all.filter(x=>(x.label+" "+x.meta).toLowerCase().includes(q)):all;
  const list=document.getElementById("pallist");
  list.innerHTML=PAL.slice(0,60).map((x,i)=>
    `<div class="palitem ${i===PALSEL?"sel":""}" data-i="${i}"><span class="pk">${E(x.kind)}</span><span>${E(x.label)}</span><span class="pm">${E(x.meta)}</span></div>`).join("")
    ||'<div class="palitem"><span class="muted">no matches</span></div>';
  list.querySelectorAll(".palitem[data-i]").forEach(d=>d.onclick=()=>runPalette(+d.dataset.i));
}
function paletteKey(e){
  if(e.key==="Escape"){e.preventDefault();closePalette();}
  else if(e.key==="ArrowDown"){e.preventDefault();PALSEL=Math.min(PAL.length-1,PALSEL+1);renderPalette(document.getElementById("palin").value);}
  else if(e.key==="ArrowUp"){e.preventDefault();PALSEL=Math.max(0,PALSEL-1);renderPalette(document.getElementById("palin").value);}
  else if(e.key==="Enter"){e.preventDefault();runPalette(PALSEL);}
}
function runPalette(i){const x=PAL[i]; if(!x)return; closePalette(); x.run();}
let CHAT=[];  // conversation history
async function loadCopilot(){
  if(drawerOpen("chat")){closeDrawer();return;}
  const p=openDrawer("chat","🤖 Copilot");
  const r=await (await fetch("/api/copilot")).json();
  let h=`<div class="cop-sum">🤖 ${E(r.summary)}</div>`;
  if(r.suspicious.length){
    h+=`<div class="k">suspicious steps</div>`+r.suspicious.map(s=>
      `<span class="chip jump" data-step="${s.step}">step ${s.step} · ${E(s.tool)} <span class="muted">(${E(s.why)})</span></span>`).join("");
  }
  h+=`<div class="k">💬 chat with the copilot ${CAN_CHAT?"":"<span class=muted>(start with --copilot-model or --agent)</span>"}</div>
    <div id="chatlog"></div>
    <div class="chatbar"><input id="chatin" placeholder="ask: why did it issue the refund? / suggest a fix…" ${CAN_CHAT?"":"disabled"}>
      <button id="chatsend" ${CAN_CHAT?"":"disabled"}>send</button></div>
    <div class="muted" style="font-size:11px">try: “explain the risky steps”, “suggest a fork that prevents the refund”, “write a policy to fix this”</div>`;
  p.innerHTML=h;
  p.querySelectorAll(".jump").forEach(c=>c.onclick=()=>{
    const i=steps.findIndex(s=>s.step===+c.dataset.step); if(i>=0)select(i);});
  renderChat();
  const send=document.getElementById("chatsend"), inp=document.getElementById("chatin");
  if(send){send.onclick=sendChat; inp.onkeydown=e=>{if(e.key==="Enter")sendChat();};}
}
function renderChat(){
  const log=document.getElementById("chatlog"); if(!log)return;
  log.innerHTML=CHAT.map(m=>{
    if(m.role==="user")return `<div class="msg u">${E(m.content)}</div>`;
    let h=`<div class="msg a">${m.content==="…thinking"?'<span class="spin">⟳</span> thinking…':md(m.content)}</div>`;
    (m.suggestions||[]).forEach((s,i)=>{
      if(s.kind==="fork") h+=`<button class="adopt" data-turn="${s.turn}" data-edit="${E(s.edit)}">🍴 Adopt · fork model call #${s.turn}</button>`;
      else if(s.kind==="policy") h+=`<div class="cop-edit">🛡 policy: deny ${E((s.deny||[]).join(", "))||"—"} · confirm ${E((s.confirm||[]).join(", "))||"—"}</div>`;
    });
    return h;
  }).join("");
  log.querySelectorAll(".adopt").forEach(b=>b.onclick=()=>adoptFork(+b.dataset.turn,b.dataset.edit));
  log.scrollTop=log.scrollHeight;
}
async function sendChat(){
  const inp=document.getElementById("chatin"), q=inp.value.trim(); if(!q)return;
  inp.value=""; CHAT.push({role:"user",content:q}); renderChat();
  CHAT.push({role:"assistant",content:"…thinking",suggestions:[]}); renderChat();
  try{
    const r=await fetch("/api/chat",{method:"POST",headers:{"content-type":"application/json"},
      body:JSON.stringify({messages:CHAT.filter(m=>m.content!=="…thinking"),step:steps[cur].step})});
    const res=await r.json(); CHAT.pop();
    if(!r.ok){CHAT.push({role:"assistant",content:"⚠ "+(res.error||"error"),suggestions:[]});}
    else CHAT.push({role:"assistant",content:res.reply||"(no reply)",suggestions:res.suggestions||[]});
  }catch(e){CHAT.pop();CHAT.push({role:"assistant",content:"⚠ "+e,suggestions:[]});}
  renderChat();
}
function adoptFork(turn,edit){
  // find the forkable step for this turn; if none, snap to the nearest earlier one
  const forkable=steps.map((s,i)=>[i,(s.replay||{})]).filter(([i,r])=>r.forkable);
  let hit=forkable.find(([i,r])=>r.turn===turn);
  if(!hit) hit=forkable.filter(([i,r])=>r.turn<=turn).pop()||forkable[0];
  if(!hit){alert("no forkable turn available");return;}
  select(hit[0]);
  setTimeout(()=>{const t=document.getElementById("append"); if(t){t.value=edit; document.getElementById("run").scrollIntoView({block:"center"});}},50);
}
let BREAKS=[];
async function setBreak(){
  const cond=document.getElementById("brk").value.trim();
  document.querySelectorAll(".step").forEach(d=>d.classList.remove("brk"));
  if(!cond){BREAKS=[];return;}
  const r=await (await fetch("/api/breaks?cond="+encodeURIComponent(cond))).json();
  BREAKS=r.steps||[];
  document.querySelectorAll(".step").forEach((d,i)=>d.classList.toggle("brk",BREAKS.includes(steps[i].step)));
  // jump to the next breakpoint at/after the current step
  const nxt=BREAKS.find(st=>{const i=steps.findIndex(s=>s.step===st);return i>=cur;});
  const tgt=nxt!=null?nxt:BREAKS[0];
  if(tgt!=null){const i=steps.findIndex(s=>s.step===tgt); if(i>=0)select(i);}
}
document.getElementById("brk").onkeydown=e=>{if(e.key==="Enter")setBreak();};
async function gotoRootCause(){
  const r = STATIC ? SD.rootcause : await (await fetch("/api/rootcause")).json();
  if(!r.found){alert("no root-cause signal — the run looks clean");return;}
  const i=steps.findIndex(s=>s.step===r.step);
  if(i>=0)select(i);
  const p=document.getElementById("copilotpanel"); p.classList.remove("hidden");
  p.innerHTML=`<div class="cop-sum">🎯 first bad step: <b>${r.step} ${E(r.tool)}</b></div>`+
    `<div class="k">why</div><div>${r.signals.map(E).join("<br>")}</div>`+
    `<div class="k">cascade</div>${r.cascade.map(c=>`<span class="chip jump" data-step="${c.step}">[${c.step}] ${E(c.tool)}</span>`).join("")}`;
  p.querySelectorAll(".jump").forEach(c=>c.onclick=()=>{const i=steps.findIndex(s=>s.step===+c.dataset.step);if(i>=0)select(i);});
}
async function showBranches(){
  if(drawerOpen("branches")){closeDrawer();return;}
  const p=openDrawer("branches","🌳 Branches");
  const r = STATIC ? {branches: SD.branches||[]} : await (await fetch("/api/branches")).json();
  if(!r.branches.length){p.innerHTML='<div class="muted">no branches yet — fork a step to start the tree</div>';return;}
  const opts=[`<option value="0">#0 original run</option>`].concat(
    r.branches.map(b=>`<option value="${b.id}">#${b.id} ${E(b.label).slice(0,30)}</option>`)).join("");
  p.innerHTML=`<div class="muted" style="font-size:11px;margin-bottom:8px">${r.branches.length} branch(es) — click a node to walk it</div>`+r.branches.map(b=>
    `<div class="bnode" data-view="${b.id}"><b>#${b.id}</b> <span class="muted">@call ${b.at}</span> — ${E(b.label)}
      <span class="bmeta">score ${b.score} · ${b.tokens.toLocaleString()} tok</span>
      <div class="muted">${E(b.output)}</div></div>`).join("")+
    `<div class="k" style="margin-top:12px">⇄ compare branches</div>
     <div id="cmprow"><select id="cmpa">${opts}</select><span class="muted">vs</span>
       <select id="cmpb">${opts.replace('value="0"','value="0"')}</select>
       <button id="cmpgo">compare</button></div><div id="cmpout"></div>`;
  const sb=document.getElementById("cmpb"); if(sb&&r.branches.length)sb.value=String(r.branches[r.branches.length-1].id);
  document.getElementById("cmpgo").onclick=runCompare;
  p.querySelectorAll(".bnode[data-view]").forEach(d=>d.onclick=e=>{if(e.target.tagName!=="SELECT")viewBranch(+d.dataset.view);});
}
let VIEWING=0;  // 0 = original run
async function viewBranch(id){
  const r=await (await fetch("/api/branch?id="+id)).json();
  if(r.error)return;
  VIEWING=id; steps=r.steps;
  renderSteps(); renderTimeline(); select(0);
  let bn=document.getElementById("viewbanner");
  if(!bn){bn=document.createElement("div");bn.id="viewbanner";document.getElementById("toolbar").after(bn);}
  bn.innerHTML=`👁 viewing branch <b>#${id}</b> — ${E(r.label)} <button id="backorig">← back to original run</button>`;
  bn.classList.toggle("hidden",id===0);
  const bo=document.getElementById("backorig"); if(bo)bo.onclick=()=>viewBranch(0);
}
async function runCompare(){
  const a=document.getElementById("cmpa").value, b=document.getElementById("cmpb").value;
  const out=document.getElementById("cmpout"); out.innerHTML='<div class="muted">comparing…</div>';
  const c=await (await fetch(`/api/compare?a=${a}&b=${b}`)).json();
  if(c.error){out.innerHTML=`<div class="muted">${E(c.error)}</div>`;return;}
  const head=s=>`<div class="cmpside${c.winner===s.id?' win':''}"><b>#${s.id}</b> ${E(s.label)}
    <div class="bmeta">score ${s.score} · ${s.tokens.toLocaleString()} tok ${c.winner===s.id?'🏆':''}</div></div>`;
  const cell=x=>x?`${x.type==="call"?"🔧 "+E(x.tool):x.type==="answer"?"✅ answer":"💭 reason"}${x.risk?" ⚠":""}`:'<span class="muted">—</span>';
  const rows=c.rows.map(rw=>`<tr class="${rw.same?'':'diff'}${rw.i===c.diverge?' dv':''}">
    <td>${rw.i}</td><td>${cell(rw.a)}</td><td>${cell(rw.b)}</td></tr>`).join("");
  out.innerHTML=`<div class="cmphead">${head(c.a)}${head(c.b)}</div>
    ${c.diverge==null?'<div class="muted" style="margin:6px 0">identical trajectories</div>':`<div class="muted" style="margin:6px 0">first divergence at call ${c.diverge}</div>`}
    <table class="cmptbl"><tr><th>#</th><th>A</th><th>B</th></tr>${rows}</table>
    <div class="k">outputs</div>
    <div class="cmphead"><div class="cmpside">${md(c.a.output)}</div><div class="cmpside">${md(c.b.output)}</div></div>`;
}
let TREE=false;
function toggleTree(){TREE=!TREE; document.getElementById("swim").classList.toggle("on",TREE); renderSteps(); select(cur);}
async function exportSession(){
  const r=await fetch("/api/export"); const blob=await r.blob();
  const u=URL.createObjectURL(blob), a=document.createElement("a");
  a.href=u; a.download="session.loomdebug"; a.click(); URL.revokeObjectURL(u);
}
document.getElementById("rootcause").onclick=gotoRootCause;
document.getElementById("branches").onclick=showBranches;
document.getElementById("agentsbtn").onclick=showAgents;
document.getElementById("assertbtn").onclick=showAssert;
document.getElementById("palettebtn").onclick=openPalette;
document.getElementById("palette").onclick=e=>{if(e.target.id==="palette")closePalette();};
document.getElementById("swim").onclick=toggleTree;
document.getElementById("export").onclick=exportSession;
document.getElementById("copilot").onclick=loadCopilot;
document.getElementById("drawerx").onclick=closeDrawer;
document.getElementById("prev").onclick=()=>select(cur-1);
document.getElementById("next").onclick=()=>select(cur+1);
document.getElementById("first").onclick=()=>select(0);
document.getElementById("last").onclick=()=>select(steps.length-1);
load();
</script></body></html>"""


def _panels_for(data: dict, step: int, acts=None) -> "list[dict]":
    """Domain panels contributed by packs for the action at ``step`` (module-level
    twin of DebugSession.panels_for, for the static export). ``acts`` may be a
    precomputed ``actions(data)`` to avoid re-parsing."""
    from .action import actions
    from .packs import install_builtin, packs

    install_builtin()
    a = next((x for x in (actions(data) if acts is None else acts) if x.step == step), None)
    if a is None:
        return []
    out: list[dict] = []
    for pack in packs():
        hook = getattr(pack, "debugger_panels", None)
        if hook is None:
            continue
        try:
            out.extend(hook(a, data) or [])
        except Exception:  # noqa: BLE001
            continue
    return out


def static_data(data: dict) -> dict:
    """Everything the debugger page fetches, precomputed -- so the SAME UI can be
    frozen into a self-contained file (no server, no agent, no model).

    Defensive throughout: ``trace_to_html`` delegates here and is an analyzer
    that must never raise on a hand-edited / hostile / malformed trace."""
    from .action import require_trace
    from .multiagent import infer_agents
    from .rootcause import first_bad_step

    data = require_trace(data)

    def _safe(fn, default):
        try:
            return fn()
        except Exception:  # noqa: BLE001
            return default

    steps = _safe(lambda: steps_for(data), [])
    # Parse the log ONCE and reuse it for every step's context + panels, instead
    # of re-parsing per step (which made a 300-step trace O(n^2) / seconds).
    from .action import actions
    acts = _safe(lambda: actions(data), [])
    d = data if isinstance(data, dict) else {}
    run = {
        "prompt": d.get("prompt", ""), "output": d.get("output", ""),
        "model": d.get("model", ""), "steps": steps,
        "can_fork": False, "can_chat": False, "live": False, "running": False,
        "system": d.get("system", ""), "all_tools": [],
    }
    # The conversation frame is CUMULATIVE (frame N contains all messages 0..N),
    # so inlining one per step is O(n^2) data -- a 300-step run ballooned the
    # HTML to ~23 MB. Inline the full message list ONCE; the page reconstructs
    # "the frame at step N" by filtering messages with step <= N client-side.
    mx = max((s.get("step", -1) for s in steps if isinstance(s, dict)), default=-1)
    context_all = _safe(lambda: context_at(data, mx, acts=acts), [])
    panels: dict = {}
    for s in steps:
        st = s.get("step") if isinstance(s, dict) else None
        if not isinstance(st, int) or st < 0 or str(st) in panels:
            continue
        p = _safe(lambda st=st: _panels_for(data, st, acts=acts), [])
        if p:
            panels[str(st)] = p
    return {"run": run, "agents": _safe(lambda: infer_agents(data), {"multi": False, "agents": []}),
            "context_all": context_all, "panels": panels,
            "rootcause": _safe(lambda: first_bad_step(data), {"found": False}),
            "branches": []}


def _scrub_banner(data: dict) -> str:
    """Green if `loom share` scrubbed it, amber otherwise -- the safety signal
    that must survive on a shareable report."""
    if data.get("scrubbed"):
        return ('<div style="padding:7px 16px;font-size:12px;background:#152a1c;'
                'color:#7ee0a0;border-bottom:1px solid #2f7a45">🛡️ Scrubbed &mdash; '
                'secrets redacted, safe to share.</div>')
    return ('<div style="padding:7px 16px;font-size:12px;background:#2a2418;'
            'color:#e8c98a;border-bottom:1px solid #7a5a2a">⚠️ Not scrubbed &mdash; '
            'this trace may contain secrets the agent saw. Run <code>loom share</code> '
            'before sharing it.</div>')


def static_page(data: dict) -> str:
    """The interactive debugger UI as a self-contained static file (Loom Studio):
    the same page, with its data inlined and server-only features (fork / live /
    copilot / assert) switched off. Shareable -- the viewer needs neither Loom
    nor the agent."""
    try:
        payload = json.dumps(static_data(data), default=str).replace("</", "<\\/")
    except Exception:  # noqa: BLE001 -- never let a bad trace break the viewer
        payload = json.dumps({"run": {"steps": []}, "agents": {"multi": False, "agents": []},
                              "context": {}, "panels": {}, "rootcause": {"found": False},
                              "branches": []})
    inject = "<script>window.LOOM_STATIC=" + payload + ";</script>\n"
    page = _PAGE.replace("<script>", inject + "<script>", 1)
    banner = _scrub_banner(data if isinstance(data, dict) else {})
    return page.replace("</header>", "</header>\n" + banner, 1)
