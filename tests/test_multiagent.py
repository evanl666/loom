"""Multi-agent attribution: native names + wire-fingerprint recovery."""

import hashlib

from loom import Agent, tool
from loom.action import actions
from loom.multiagent import best_role, infer_agents
from loom.providers import ModelResponse, ScriptedProvider, ToolCall


@tool
def search(q: str) -> str:
    "Search."
    return f"result for {q}"


def _lead():
    researcher = Agent(model=ScriptedProvider([
        ModelResponse(tool_calls=[ToolCall("s1", "search", {"q": "loom"})], stop_reason="tool_use"),
        ModelResponse(text="Loom is an agent harness.", stop_reason="end_turn")]),
        tools=[search], name="researcher")
    return Agent(model=ScriptedProvider([
        ModelResponse(tool_calls=[ToolCall("d1", "researcher", {"task": "x"})], stop_reason="tool_use"),
        ModelResponse(text="Per research: harness.", stop_reason="end_turn")]),
        tools=[researcher.as_tool()], name="lead")


def test_native_names_and_delegation_step():
    data = _lead().run("Explain Loom.").to_dict()
    # the delegation call is no longer dropped
    tools = [a.tool for a in actions(data) if a.type == "call"]
    assert "researcher" in tools and "search" in tools
    ia = infer_agents(data)
    assert ia["multi"] and ia["source"] == "native"
    labels = {a["label"] for a in ia["agents"]}
    assert labels == {"lead", "researcher"}
    # the parent delegates to the child
    assert any(e for e in ia["edges"])
    # each step maps to an agent
    assert ia["step_agent"]


def _fp(system, tools):
    return {"sys_hash": hashlib.sha1(system.encode()).hexdigest()[:12],
            "sys_head": system, "tools": tools, "model": "claude"}


def test_wire_fingerprint_recovers_two_agents():
    SUP = "You are the supervisor. Delegate to workers."
    COD = "You are the coder worker. Write code."
    data = {"recorded_via": "proxy", "model": "claude", "output": "done", "tools": {}, "log": [
        {"seq": 0, "kind": "model", "depth": 0, "meta": _fp(SUP, ["delegate"]),
         "result": {"tool_calls": [{"id": "1", "name": "delegate", "input": {}}], "stop_reason": "tool_use"}},
        {"seq": 1, "kind": "tool:delegate", "depth": 0, "result": "ok"},
        {"seq": 2, "kind": "model", "depth": 0, "meta": _fp(COD, ["write_file"]),
         "result": {"tool_calls": [{"id": "2", "name": "write_file", "input": {}}], "stop_reason": "tool_use"}},
        {"seq": 3, "kind": "tool:write_file", "depth": 0, "result": "ok"},
        {"seq": 4, "kind": "model", "depth": 0, "meta": _fp(COD, ["write_file"]),
         "result": {"text": "done", "stop_reason": "end_turn"}},
    ]}
    ia = infer_agents(data)
    assert ia["multi"] and ia["source"] == "wire"
    labels = [a["label"] for a in ia["agents"]]
    assert "supervisor" in labels and "coder worker" in labels
    # the coder's tool call is attributed to the coder, not the supervisor
    assert ia["step_agent"]["3"] == ia["step_agent"]["2"]
    assert ia["step_agent"]["0"] != ia["step_agent"]["2"]


def test_titlegen_side_call_is_a_utility_not_the_root():
    """The Claude CLI fires a one-shot, tool-less title-generation call BEFORE the
    main loop. It must be flagged a UTILITY side-call -- not made the run's root,
    which would demote the real interactive agent to a bogus 'SUBAGENT'."""
    TITLE = ("You are a Claude agent, built on Anthropic's Claude Agent SDK. "
             "Generate a concise, sentence-case title for this session.")
    MAIN = "You are a Claude agent. You are an interactive CLI tool that helps with coding."
    data = {"recorded_via": "proxy", "model": "claude", "output": "done", "tools": {}, "log": [
        # title-gen: no tools, one call, returns text -- never delegates / delegated-to
        {"seq": 0, "kind": "model", "depth": 0, "meta": _fp(TITLE, []),
         "result": {"text": '{"title": "Review files"}', "stop_reason": "end_turn"}},
        # the real agent: has tools, does the work
        {"seq": 1, "kind": "model", "depth": 0, "meta": _fp(MAIN, ["Read", "Glob"]),
         "result": {"tool_calls": [{"id": "1", "name": "Read", "input": {}}], "stop_reason": "tool_use"}},
        {"seq": 2, "kind": "tool:Read", "depth": 0, "result": "ok"},
        {"seq": 3, "kind": "model", "depth": 0, "meta": _fp(MAIN, ["Read", "Glob"]),
         "result": {"text": "done", "stop_reason": "end_turn"}},
    ]}
    ia = infer_agents(data)
    by_hash = {a["sys_hash"]: a for a in ia["agents"]}
    title = by_hash[_fp(TITLE, [])["sys_hash"]]
    main = by_hash[_fp(MAIN, ["Read", "Glob"])["sys_hash"]]
    assert title["utility"] is True and title["is_root"] is False
    assert title["label"] == "title"          # generic "agent N" relabeled
    assert main["utility"] is False and main["is_root"] is True   # the REAL root
    assert ia["edges"] == []                   # no false delegation between them


def test_toolless_agent_in_a_real_delegation_is_not_marked_utility():
    """A 0-tool agent that IS part of a delegation (a pure-reasoning planner a
    supervisor hands off to) must NOT be mistaken for a throwaway utility call."""
    SUP = "You are the supervisor. Delegate to the planner."
    PLAN = "You are the planner. Think, then reply."      # no tools, but delegated-to
    data = {"recorded_via": "proxy", "model": "claude", "output": "ok", "tools": {}, "log": [
        {"seq": 0, "kind": "model", "depth": 0, "meta": _fp(SUP, ["delegate_planner"]),
         "result": {"tool_calls": [{"id": "1", "name": "delegate_planner", "input": {}}],
                    "stop_reason": "tool_use"}},
        {"seq": 1, "kind": "tool:delegate_planner", "depth": 0, "result": "ok"},
        {"seq": 2, "kind": "model", "depth": 0, "meta": _fp(PLAN, []),
         "result": {"text": "the plan", "stop_reason": "end_turn"}},
    ]}
    ia = infer_agents(data)
    planner = next(a for a in ia["agents"] if a["sys_hash"] == _fp(PLAN, [])["sys_hash"])
    assert planner["utility"] is False        # linked by a delegation edge -> real
    assert any(e["to"] for e in ia["edges"])  # the delegation is preserved


def test_best_role_finds_specific_role_past_generic_preamble():
    # a framework preamble ("You are Claude Code...") must not mask the sub-agent's
    # real role stated later in the same (possibly long) system prompt.
    assert best_role("You are Claude Code. [800 chars] You are a Landmark Researcher. "
                     "Be terse.") == "Landmark Researcher"
    assert best_role("You are the Supervisor. Delegate to workers.") == "Supervisor"
    assert best_role("You are a Math Specialist. Use calculate.") == "Math Specialist"
    # generic identities give no label (caller falls back to "agent N")
    assert best_role("You are a helpful assistant.") is None
    assert best_role("You are Claude, made by Anthropic.") is None


def test_single_agent_is_not_multi():
    @tool
    def get() -> str:
        "get"
        return "x"
    a = Agent(model=ScriptedProvider([
        ModelResponse(tool_calls=[ToolCall("1", "get", {})], stop_reason="tool_use"),
        ModelResponse(text="ok", stop_reason="end_turn")]), tools=[get], name="solo")
    ia = infer_agents(a.run("go").to_dict())
    assert ia["multi"] is False


def test_wire_delayed_delegation_attributed_to_the_requester():
    """A delegate tool whose result comes back AFTER the sub-agent ran (a paired
    delegation) must be attributed to the REQUESTER (not whoever acted last),
    flagged as a sub-agent hand-off, and re-anchored before the sub-agent's work.
    Mirrors LangGraph's Research Lead -> ask_data_analyst -> Data Analyst."""
    import hashlib
    from loom.debugger import steps_for

    def fp(system, tools):
        return {"sys_hash": hashlib.sha1(system.encode()).hexdigest()[:12],
                "sys_head": system, "sys_role": system.split(".")[0].replace("You are the ", ""),
                "tools": tools, "model": "m"}
    LEAD, WORKER = "You are the Research Lead.", "You are the Data Analyst."
    data = {"recorded_via": "proxy", "model": "m", "output": "990", "tools": {}, "log": [
        # Research Lead asks the analyst; the tool result (seq 4) comes back LATE
        {"seq": 0, "kind": "model", "depth": 0, "meta": fp(LEAD, ["ask_analyst"]),
         "result": {"tool_calls": [{"id": "1", "name": "ask_analyst", "input": {"t": "3*3"}}], "stop_reason": "tool_use"}},
        {"seq": 1, "kind": "model", "depth": 0, "meta": fp(WORKER, ["calc"]),
         "result": {"tool_calls": [{"id": "2", "name": "calc", "input": {"e": "3*3"}}], "stop_reason": "tool_use"}},
        {"seq": 2, "kind": "tool:calc", "depth": 0, "result": "9"},
        {"seq": 3, "kind": "model", "depth": 0, "meta": fp(WORKER, ["calc"]),
         "result": {"text": "9", "stop_reason": "end_turn"}},
        {"seq": 4, "kind": "tool:ask_analyst", "depth": 0, "result": "9"},  # late result
        {"seq": 5, "kind": "model", "depth": 0, "meta": fp(LEAD, ["ask_analyst"]),
         "result": {"text": "990", "stop_reason": "end_turn"}},
    ]}
    steps = steps_for(data)
    by_tool = {s["tool"]: s for s in steps if s.get("type") == "call"}
    assert by_tool["ask_analyst"]["agent"] == "Research Lead"   # requester, not the analyst
    assert by_tool["ask_analyst"]["is_delegation"] is True      # a sub-agent hand-off
    assert by_tool["calc"]["agent"] == "Data Analyst"
    # re-anchored: the hand-off is shown BEFORE the analyst's calc
    order = [s.get("tool") for s in steps if s.get("type") == "call"]
    assert order.index("ask_analyst") < order.index("calc")


def test_tool_only_model_call_still_shows_a_model_step():
    """A model call that makes a tool call but produces no text must still be a
    visible 'model' step -- so a tool call is never orphaned (no model deciding
    it). Regression for the debugger reading 'tool call' with no model before it."""
    data = {"log": [
        {"seq": 0, "kind": "model", "result": {"tool_calls": [
            {"id": "1", "name": "search", "input": {}}], "stop_reason": "tool_use"}},  # no text
        {"seq": 1, "kind": "tool:search", "result": "r"},
        {"seq": 2, "kind": "model", "result": {"text": "done", "stop_reason": "end_turn"}}],
        "prompt": "x", "output": "done", "tools": {}}
    types = [a.type for a in actions(data)]
    assert types[0] == "reason" and types[1] == "call"   # model decision, then the call


def test_parallel_delegations_mapped_to_children_semantically():
    """A coordinator that fires several hand-offs from ONE turn is structurally
    ambiguous on the wire; match each to the child whose role overlaps the
    delegate tool name/task (ask_research -> Research Lead)."""
    import hashlib
    from loom.debugger import steps_for

    def fp(system, tools):
        role = system.replace("You are the ", "").rstrip(".")
        return {"sys_hash": hashlib.sha1(system.encode()).hexdigest()[:12],
                "sys_head": system, "sys_role": role, "tools": tools, "model": "m"}
    data = {"recorded_via": "proxy", "model": "m", "output": "done", "tools": {}, "log": [
        {"seq": 0, "kind": "model", "depth": 0, "meta": fp("You are the Coordinator.", ["ask_research", "ask_support"]),
         "result": {"tool_calls": [
             {"id": "1", "name": "ask_research", "input": {"task": "find X"}},
             {"id": "2", "name": "ask_support", "input": {"task": "help Y"}}], "stop_reason": "tool_use"}},
        # support runs FIRST (execution order != tool_call order), then research
        {"seq": 1, "kind": "model", "depth": 0, "meta": fp("You are the Support Lead.", ["email"]),
         "result": {"text": "helped", "stop_reason": "end_turn"}},
        {"seq": 2, "kind": "model", "depth": 0, "meta": fp("You are the Research Lead.", ["search"]),
         "result": {"text": "found", "stop_reason": "end_turn"}},
        {"seq": 3, "kind": "model", "depth": 0, "meta": fp("You are the Coordinator.", ["ask_research", "ask_support"]),
         "result": {"text": "done", "stop_reason": "end_turn"}},
    ]}
    steps = steps_for(data)
    lab = {s["agent_id"]: s.get("agent") for s in steps if s.get("agent_id")}
    m = {s["tool"]: lab.get(s.get("delegates_to")) for s in steps if s.get("is_delegation")}
    # matched by role token overlap, NOT by positional order (support ran first)
    assert m.get("ask_research") == "Research Lead"
    assert m.get("ask_support") == "Support Lead"


def test_leaf_tool_not_flagged_delegation_across_interleaved_branches():
    """A leaf tool whose result arrives AFTER a deeper agent from another parallel
    branch ran must NOT be mislabeled a delegation. A tool call is a delegation
    only if the agent that made it actually spawned a matching child agent.
    Mirrors LangGraph fan-out: Coordinator -> {Research Lead -> Data Analyst,
    Support Lead(draft_email leaf)}, with Data Analyst running between the Support
    Lead's draft_email call and its result."""
    from loom.debugger import steps_for

    def fp(system, tools):
        return {"sys_hash": hashlib.sha1(system.encode()).hexdigest()[:12],
                "sys_head": system[:160], "sys_role": system.split(".")[0].replace("You are the ", "").replace("You are a ", ""),
                "tools": tools, "model": "m"}

    def model(system, tools, tcs=None, text=None, seq=0):
        res = {"tool_calls": tcs, "stop_reason": "tool_use"} if tcs else {"text": text or "", "stop_reason": "end_turn"}
        return {"seq": seq, "kind": "model", "meta": fp(system, tools), "result": res}

    def tc(name):
        return {"id": name, "name": name, "input": {}}

    COORD, RL, SL, DA = ("You are the Coordinator.", "You are the Research Lead.",
                         "You are the Support Lead.", "You are the Data Analyst.")
    data = {"recorded_via": "proxy", "model": "m", "output": "done", "tools": {}, "log": [
        model(COORD, ["ask_research", "ask_support"], [tc("ask_research"), tc("ask_support")], seq=0),
        model(RL, ["ask_data_analyst"], [tc("ask_data_analyst")], seq=1),
        model(SL, ["draft_email"], [tc("draft_email")], seq=2),
        model(DA, ["calculate"], [tc("calculate")], seq=3),
        {"seq": 4, "kind": "tool:calculate", "result": "990"},
        model(DA, ["calculate"], text="990", seq=5),
        {"seq": 6, "kind": "tool:draft_email", "result": "DRAFT to jane@x"},   # SL leaf, AFTER DA ran
        model(SL, ["draft_email"], text="sent", seq=7),
        {"seq": 8, "kind": "tool:ask_data_analyst", "result": "990"},
        model(RL, ["ask_data_analyst"], text="330x3=990", seq=9),
        {"seq": 10, "kind": "tool:ask_research", "result": "990"},
        {"seq": 11, "kind": "tool:ask_support", "result": "sent"},
        model(COORD, ["ask_research", "ask_support"], text="all done", seq=12),
    ]}
    steps = steps_for(data)
    byname = {s["tool"]: s for s in steps if s.get("type") == "call"}
    # the leaf tool is NOT a delegation, despite Data Analyst running before its result
    assert not byname["draft_email"].get("is_delegation")
    # the real delegations are flagged and mapped to the right child
    lab = {s.get("agent_id"): s.get("agent") for s in steps if s.get("agent_id")}
    assert byname["ask_research"].get("is_delegation") and lab[byname["ask_research"]["delegates_to"]] == "Research Lead"
    assert byname["ask_support"].get("is_delegation") and lab[byname["ask_support"]["delegates_to"]] == "Support Lead"
    assert byname["ask_data_analyst"].get("is_delegation") and lab[byname["ask_data_analyst"]["delegates_to"]] == "Data Analyst"


def test_infer_agents_surfaces_the_full_system_prompt_per_agent():
    """Each agent carries its FULL system prompt (resolved from the trace's
    per-sys_hash `systems` map the proxy records), so the debugger can show and
    edit it; a sys_head fallback covers older traces without the map."""
    LONG = "You are the Coordinator. " + "Delegate to the research specialist. " * 12
    RES = "You are a Research Specialist. Use web_search. Be terse."
    hc = hashlib.sha1(LONG.encode()).hexdigest()[:12]
    hr = hashlib.sha1(RES.encode()).hexdigest()[:12]

    def fp(system, tools):
        return {"sys_hash": hashlib.sha1(system.encode()).hexdigest()[:12],
                "sys_head": system[:160], "sys_role": system.split(".")[0].replace("You are the ", "").replace("You are a ", ""),
                "tools": tools, "model": "m"}
    data = {"recorded_via": "proxy", "model": "m", "system": LONG,
            "systems": {hc: LONG, hr: RES}, "output": "x", "tools": {}, "log": [
        {"seq": 0, "kind": "model", "meta": fp(LONG, ["ask_research"]),
         "result": {"tool_calls": [{"id": "1", "name": "ask_research", "input": {"task": "t"}}], "stop_reason": "tool_use"}},
        {"seq": 1, "kind": "model", "meta": fp(RES, ["web_search"]),
         "result": {"text": "done", "stop_reason": "end_turn"}}]}
    ags = {a["label"]: a for a in infer_agents(data)["agents"]}
    assert ags["Coordinator"]["system"] == LONG      # full, not the 160-char head
    assert len(ags["Coordinator"]["system"]) > 160
    assert ags["Research Specialist"]["system"] == RES

    # older trace without a `systems` map: fall back to the recorded head
    data.pop("systems")
    ags2 = {a["label"]: a for a in infer_agents(data)["agents"]}
    assert ags2["Research Specialist"]["system"] == RES[:160]


def _wire(system, tools, tcs=None, text=None, seq=0, msgs=None):
    role = system.replace("You are the ", "").replace("You are a ", "").rstrip(".")
    meta = {"sys_hash": hashlib.sha1(system.encode()).hexdigest()[:12],
            "sys_head": system, "sys_role": role, "tools": tools, "model": "m"}
    if msgs is not None:
        meta["msgs"] = msgs
    res = {"stop_reason": "tool_use", "tool_calls": tcs} if tcs else {"text": text or "", "stop_reason": "end_turn"}
    return {"seq": seq, "kind": "model", "depth": 0, "meta": meta, "result": res}


def test_parallel_siblings_attach_to_parent_not_a_chain():
    """A coordinator that fans out TWO sub-agents from one turn, whose calls then
    interleave on the wire, must produce two SIBLINGS under the coordinator -- not
    a chain (research -> math). General across frameworks (mirrors PydanticAI)."""
    data = {"recorded_via": "proxy", "model": "m", "output": "990", "tools": {}, "log": [
        _wire("You are the Coordinator.", ["ask_research", "ask_calculator"], seq=0, msgs=1,
              tcs=[{"id": "1", "name": "ask_research", "input": {"task": "Eiffel height"}},
                   {"id": "2", "name": "ask_calculator", "input": {"task": "multiply 330"}}]),
        _wire("You are a Research Specialist.", ["web_search"], seq=1, msgs=1,
              tcs=[{"id": "3", "name": "web_search", "input": {"q": "eiffel"}}]),
        _wire("You are a Math Specialist.", ["calculate"], seq=2, msgs=1,
              tcs=[{"id": "4", "name": "calculate", "input": {"e": "330*3"}}]),
        _wire("You are a Research Specialist.", ["web_search"], seq=4, msgs=3, text="330m"),
        _wire("You are a Math Specialist.", ["calculate"], seq=5, msgs=3, text="990"),
        _wire("You are the Coordinator.", ["ask_research", "ask_calculator"], seq=6, msgs=3, text="done"),
    ]}
    ia = infer_agents(data)
    lvl = {a["label"]: a["level"] for a in ia["agents"]}
    assert lvl["Coordinator"] == 0
    assert lvl["Research Specialist"] == 1 and lvl["Math Specialist"] == 1  # siblings
    edges = {(e["from"], e["to"]) for e in ia["edges"]}
    ids = {a["label"]: a["id"] for a in ia["agents"]}
    assert (ids["Coordinator"], ids["Research Specialist"]) in edges
    assert (ids["Coordinator"], ids["Math Specialist"]) in edges
    # NOT a chain: math is not a child of research
    assert (ids["Research Specialist"], ids["Math Specialist"]) not in edges


def test_grandchild_nesting_attaches_to_deeper_parent():
    """A three-level delegation (coordinator -> research lead -> data analyst) must
    nest the analyst under the LEAD, not the coordinator -- matched by name/args."""
    data = {"recorded_via": "proxy", "model": "m", "output": "9", "tools": {}, "log": [
        _wire("You are the Coordinator.", ["ask_research"], seq=0, msgs=1,
              tcs=[{"id": "1", "name": "ask_research", "input": {"task": "analyze data"}}]),
        _wire("You are the Research Lead.", ["ask_data_analyst"], seq=1, msgs=1,
              tcs=[{"id": "2", "name": "ask_data_analyst", "input": {"task": "compute"}}]),
        _wire("You are the Data Analyst.", ["calculate"], seq=2, msgs=1,
              tcs=[{"id": "3", "name": "calculate", "input": {"e": "3*3"}}]),
        _wire("You are the Data Analyst.", ["calculate"], seq=4, msgs=3, text="9"),
        _wire("You are the Research Lead.", ["ask_data_analyst"], seq=5, msgs=3, text="9"),
        _wire("You are the Coordinator.", ["ask_research"], seq=6, msgs=3, text="9"),
    ]}
    ia = infer_agents(data)
    lvl = {a["label"]: a["level"] for a in ia["agents"]}
    assert lvl["Coordinator"] == 0 and lvl["Research Lead"] == 1 and lvl["Data Analyst"] == 2
    ids = {a["label"]: a["id"] for a in ia["agents"]}
    edges = {(e["from"], e["to"]) for e in ia["edges"]}
    assert (ids["Research Lead"], ids["Data Analyst"]) in edges  # analyst under the lead


def test_generic_subagent_labeled_from_handoff_subagent_type():
    """When a sub-agent's OWN system prompt is too generic to name it (the Claude
    Agent SDK gives every call a 'You are Claude Code' preamble), label it from
    the explicit target its hand-off carried (Task tool subagent_type)."""
    GEN = "You are Claude Code, an AI assistant."   # generic -> best_role gives no name

    def gen(who, tools, seq, tcs=None, text=None):
        # a generic agent: sys_role EMPTY (best_role found nothing), like the CLI
        meta = {"sys_hash": hashlib.sha1(who.encode()).hexdigest()[:12],
                "sys_head": GEN, "sys_role": "", "tools": tools, "model": "m", "msgs": 1}
        res = {"stop_reason": "tool_use", "tool_calls": tcs} if tcs else {"text": text or "", "stop_reason": "end_turn"}
        return {"seq": seq, "kind": "model", "depth": 0, "meta": meta, "result": res}

    data = {"recorded_via": "proxy", "model": "m", "output": "x", "tools": {}, "log": [
        gen("orch", ["Task"], 0, tcs=[{"id": "1", "name": "Task", "input": {"subagent_type": "researcher", "prompt": "find X"}}]),
        {"seq": 1, "kind": "tool:Task", "result": "found"},
        gen("sub", [], 2, text="330m"),   # the sub-agent: generic prompt, no tools of its own
    ]}
    labels = [a["label"] for a in infer_agents(data)["agents"]]
    assert "Researcher" in labels   # named from subagent_type, not "agent 2"


def test_handoff_with_early_result_still_creates_an_edge():
    """A CONTROL-TRANSFER handoff (OpenAI Agents SDK / Semantic Kernel) emits its
    transfer_to_X result BEFORE the target agent takes over, and the target
    carries the whole conversation (msgs>=2). It must still be recovered as a
    directed edge (Triage -> Refund), not mistaken for a flat peer turn."""
    data = {"recorded_via": "proxy", "model": "m", "output": "done", "tools": {}, "log": [
        _wire("You are a Triage Specialist.", ["transfer_to_refund_agent"], seq=0, msgs=1,
              tcs=[{"id": "1", "name": "transfer_to_refund_agent", "input": {}}]),
        {"seq": 1, "kind": "tool:transfer_to_refund_agent", "result": "handed off"},  # result BEFORE target
        _wire("You are a Refund Specialist.", ["issue_refund"], seq=2, msgs=3,     # carries the convo
              tcs=[{"id": "2", "name": "issue_refund", "input": {}}]),
        {"seq": 3, "kind": "tool:issue_refund", "result": "refunded"},
        _wire("You are a Refund Specialist.", ["issue_refund"], seq=4, msgs=5, text="done"),
    ]}
    ia = infer_agents(data)
    lvl = {a["label"]: a["level"] for a in ia["agents"]}
    assert lvl["Triage Specialist"] == 0 and lvl["Refund Specialist"] == 1   # nested, not flat peers
    ids = {a["label"]: a["id"] for a in ia["agents"]}
    assert (ids["Triage Specialist"], ids["Refund Specialist"]) in {(e["from"], e["to"]) for e in ia["edges"]}


def test_peer_group_chat_is_flat_not_delegation():
    """A round-robin group chat (AutoGen): peers share ONE growing conversation and
    take turns. The msgs signal (context already large on a peer's first turn)
    keeps them flat -- same level, no delegation edges, no false sub-agent flags."""
    from loom.debugger import steps_for
    data = {"recorded_via": "proxy", "model": "m", "output": "990", "tools": {}, "log": [
        _wire("You are a Research Specialist.", ["web_search"], seq=0, msgs=1,
              tcs=[{"id": "1", "name": "web_search", "input": {"q": "eiffel"}}]),
        _wire("You are a Math Specialist.", ["calculate"], seq=1, msgs=2,
              tcs=[{"id": "2", "name": "calculate", "input": {"e": "330*3"}}]),
        _wire("You are a Research Specialist.", ["web_search"], seq=3, msgs=4, text="330m"),
        _wire("You are a Math Specialist.", ["calculate"], seq=4, msgs=5, text="990 DONE"),
    ]}
    ia = infer_agents(data)
    lvl = {a["label"]: a["level"] for a in ia["agents"]}
    assert lvl["Research Specialist"] == 0 and lvl["Math Specialist"] == 0  # flat peers
    assert ia["edges"] == []                                                # no delegation
    assert not any(s.get("is_delegation") for s in steps_for(data))         # no false flags


def test_claude_cli_markers_orchestrator_subagents_no_title_gen():
    """The Claude Agent SDK (via the claude CLI) is the hard case: every call gets
    a generic 'You are a Claude agent' preamble, sub-agents are flagged with an
    explicit cc_is_subagent marker, and the CLI slips in an internal title-
    generation call. Use the marker as authoritative: sub-agents are children of
    the one main session, the title-gen is NOT a separate agent, and roles come
    from the real 'You are a X' past the preamble."""
    def sysf(role, sub=False):
        h = "x-anthropic-billing-header: cc_version=1.2.3; cc_entrypoint=sdk-py; "
        if sub:
            h += "cc_is_subagent=true; "
        return h + "You are a Claude agent, built on Anthropic's Claude Agent SDK. " + role

    def h(s):
        return hashlib.sha1(s.encode()).hexdigest()[:12]

    ORCH = sysf("You are an Orchestrator. You MUST delegate.")
    TITLE = sysf("Generate a concise, sentence-case title (3-7 words).")
    RES = sysf("You are a Landmark Researcher. Reply with one numeric fact.", sub=True)
    WRI = sysf("You are a Summary Writer. Write one sentence.", sub=True)
    systems = {h(ORCH): ORCH, h(TITLE): TITLE, h(RES): RES, h(WRI): WRI}

    def model(system, seq, tcs=None, text=""):
        res = {"tool_calls": tcs, "stop_reason": "tool_use"} if tcs else {"text": text, "stop_reason": "end_turn"}
        return {"seq": seq, "kind": "model", "result": res,
                "meta": {"sys_hash": h(system), "sys_head": system[:160], "model": "m"}}

    def task(sub):
        return [{"id": sub, "name": "Agent", "input": {"subagent_type": sub}}]

    data = {"recorded_via": "proxy", "model": "m", "output": "done", "tools": {},
            "systems": systems, "log": [
        model(ORCH, 0, tcs=task("researcher"), text="delegating"),
        model(TITLE, 1, text='{"title": "Eiffel height"}'),        # CLI title-gen -- not an agent
        model(RES, 2, text="330 meters"),                          # the researcher sub-agent
        {"seq": 3, "kind": "tool:Agent", "result": "330 meters"},
        model(ORCH, 4, tcs=task("writer"), text="now writing"),
        model(WRI, 5, text="The Eiffel Tower is 330 meters tall."),  # the writer sub-agent
        {"seq": 6, "kind": "tool:Agent", "result": "..."},
        model(ORCH, 7, text="DONE")]}
    ia = infer_agents(data)
    labels = [a["label"] for a in ia["agents"]]
    assert labels == ["Orchestrator", "Landmark Researcher", "Summary Writer"]  # 3, no title-gen
    lvl = {a["label"]: a["level"] for a in ia["agents"]}
    assert lvl["Landmark Researcher"] == 1 and lvl["Summary Writer"] == 1  # children of the main session
    ids = {a["label"]: a["id"] for a in ia["agents"]}
    edges = {(e["from"], e["to"]) for e in ia["edges"]}
    assert (ids["Orchestrator"], ids["Landmark Researcher"]) in edges
    assert (ids["Orchestrator"], ids["Summary Writer"]) in edges


def test_late_returning_grandchild_attaches_to_its_real_parent():
    """A grandchild whose first call appears AFTER its parent already returned
    (popped off the stack) -- an interleaving a fork's re-run can produce -- must
    still nest under its REAL parent, not fall back to the root. Regression for
    Data Analyst showing at the Coordinator's level instead of under Research
    Lead in a branch view."""
    def fp(system, tools):
        return {"sys_hash": hashlib.sha1(system.encode()).hexdigest()[:12], "sys_head": system,
                "sys_role": system.split(".")[0].replace("You are the ", ""), "tools": tools, "model": "m"}

    def model(system, tools, seq, tcs=None, text=None):
        res = {"tool_calls": tcs, "stop_reason": "tool_use"} if tcs else {"text": text or "", "stop_reason": "end_turn"}
        return {"seq": seq, "kind": "model", "meta": fp(system, tools), "result": res}

    def tc(n):
        return {"id": n, "name": n, "input": {}}
    C, R, D = "You are the Coordinator.", "You are the Research Lead.", "You are the Data Analyst."
    data = {"recorded_via": "proxy", "model": "m", "output": "x", "tools": {}, "log": [
        model(C, ["ask_research"], 0, tcs=[tc("ask_research")]),
        model(R, ["ask_data_analyst"], 1, tcs=[tc("ask_data_analyst")]),
        model(R, ["ask_data_analyst"], 2, text="done"),          # RL returns before Data Analyst runs
        model(D, ["calculate"], 3, tcs=[tc("calculate")]),       # Data Analyst appears LATE
        {"seq": 4, "kind": "tool:calculate", "result": "9"},
        model(D, ["calculate"], 5, text="9"),
        model(C, ["ask_research"], 6, text="all done")]}
    ia = infer_agents(data)
    lvl = {a["label"]: a["level"] for a in ia["agents"]}
    assert lvl["Data Analyst"] == 2   # under Research Lead, not the Coordinator
    ids = {a["label"]: a["id"] for a in ia["agents"]}
    assert (ids["Research Lead"], ids["Data Analyst"]) in {(e["from"], e["to"]) for e in ia["edges"]}
