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
    from .packs import install_builtin

    install_builtin()
    return [a.to_dict() for a in actions(data)]


def context_at(data: dict, step: int) -> list[dict]:
    """The conversation the model had seen up to (and including) ``step`` --
    the debugger's "current frame": the prompt, prior reasoning, tool calls, and
    tool results that were in context when this step ran."""
    from .action import actions

    prompt = (data.get("episodes") or [data.get("prompt", "")])[0]
    frame: list[dict] = [{"role": "user", "content": str(prompt), "step": -1}]
    for a in actions(data):
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

    def __init__(self, trace_path: str, agent: Any = None, copilot_model: str = ""):
        self.trace_path = trace_path
        self.agent = agent
        # model the chat copilot talks to; falls back to the fork agent's model
        self.copilot_model = copilot_model or (getattr(agent, "model", "") if agent else "")
        with open(trace_path) as f:
            self.data = json.load(f)

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
        return _branch_payload(self.data, branch.to_dict(), at)

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
                "system": (getattr(sess.agent, "system", "") if sess.agent
                           else sess.data.get("system", "")),
                "all_tools": (sorted(sess.agent.tools) if sess.agent else []),
            })
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

    def __init__(self, trace_path: str, agent: Any = None,
                 port: int = 8790, host: str = "127.0.0.1", copilot_model: str = ""):
        self.session = DebugSession(trace_path, agent, copilot_model=copilot_model)
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
#copilotpanel{padding:12px 16px;border-bottom:1px solid #26262b;background:#12151a}
#copilotpanel.hidden{display:none}.cop-sum{margin-bottom:8px;color:#cfe3ff}
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
</style></head><body>
<header><b>🔬 Loom debugger</b><span class="muted" id="prompt">loading…</span>
<span class="muted" style="margin-left:auto" id="model"></span></header>
<div id="wrap">
  <div id="steps"></div>
  <div style="flex:1;display:flex;flex-direction:column;min-width:0">
    <div id="toolbar">
      <button id="first" title="first (Home)">⏮</button>
      <button id="prev" title="prev (←)">◀ step</button>
      <button id="next" title="next (→)">step ▶</button>
      <button id="last" title="last (End)">⏭</button>
      <span class="muted" id="pos" style="margin-left:8px"></span>
      <input id="brk" placeholder="⏹ break: tool:send_email · cap:network" title="conditional breakpoint" style="margin-left:12px;width:230px">
      <button id="copilot" style="margin-left:auto">🤖 Copilot</button>
    </div>
    <div id="copilotpanel" class="hidden"></div>
    <div id="detail"></div>
  </div>
</div>
<script>
const E=s=>String(s==null?"":s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
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
  RUN=await (await fetch("/api/run")).json();
  steps=RUN.steps; canFork=RUN.can_fork; CAN_CHAT=RUN.can_chat;
  document.getElementById("prompt").textContent=RUN.prompt.slice(0,120);
  document.getElementById("model").textContent=RUN.model||"";
  renderSteps(); select(0);
}
function typeClass(t){return "b-"+t}
function renderSteps(){
  const el=document.getElementById("steps");
  el.innerHTML=steps.map((s,i)=>{
    const risky=s.risk?` <span class="risky" title="${E(s.risk)}">⚠</span>`:"";
    const lbl=s.tool?E(s.tool):E(s.type);
    const ind=s.depth?`<span class="depth">${"› ".repeat(s.depth)}</span>`:"";
    return `<div class="step" data-i="${i}"><span class="muted">${s.step}</span>`+
      `<span class="badge ${typeClass(s.type)}">${E(s.type)}</span>${ind}<span>${lbl}</span>${risky}</div>`;
  }).join("");
  el.querySelectorAll(".step").forEach(d=>d.onclick=()=>select(+d.dataset.i));
}
function select(i){
  cur=Math.max(0,Math.min(steps.length-1,i));
  document.querySelectorAll(".step").forEach((d,j)=>d.classList.toggle("cur",j===cur));
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
  let h=`<span class="badge ${typeClass(s.type)}">${E(s.type)}</span> `+
        (s.tool?`<b>${E(s.tool)}</b>`:"")+` <span class="muted">step ${s.step} · turn ${(s.replay||{}).turn??"?"}${s.depth?" · depth "+s.depth:""}</span>`;
  if(s.intent) h+=`<div class="k">model reasoning</div><pre>${E(s.intent)}</pre>`;
  const cf=codeFields(s.input);
  if(cf){
    h+=`<div class="k">📄 ${E(cf.path)}${cf.verb?` <span class="muted">(${cf.verb})</span>`:""}</div><pre class="code">${E(cf.code)}</pre>`;
    if(cf.rest) h+=`<div class="k">args</div><pre>${E(cf.rest)}</pre>`;
  } else if(s.input!=null){
    h+=`<div class="k">${s.tool?"tool input":"input"}</div><pre>${E(J(s.input))}</pre>`;
  }
  if(o.text) h+=`<div class="k">result</div><pre>${E(o.text.length>4000?o.text.slice(0,4000)+"\n… (truncated)":o.text)}</pre>`;
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
  if(s.type==="call") h+=`<div class="k">🩸 memory blame <button id="blamebtn" class="mini">what influenced this?</button></div><div id="blame"></div>`;
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
    <div id="branch"></div>
  </div>`;
  if(!canFork&&forkable) h+=`<div class="k muted">re-run disabled — start with <kbd>--agent module:attr</kbd> to fork live</div>`;
  d.innerHTML=h;
  const rb=document.getElementById("run"); if(rb) rb.onclick=doFork;
  const cb=document.getElementById("ctxbtn"); if(cb) cb.onclick=loadContext;
  const bb=document.getElementById("blamebtn"); if(bb) bb.onclick=loadBlame;
  const fr=document.getElementById("faultrun"); if(fr) fr.onclick=faultInject;
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
  const r=await (await fetch("/api/context?step="+s.step)).json();
  box.innerHTML=r.frame.map(m=>{
    const role={user:"👤 user",assistant:"🤖 model",tool:"🔧 "+(m.tool||"tool"),human:"🧑 human"}[m.role]||m.role;
    const cur=m.step===s.step?" curframe":"";
    return `<div class="frame${cur}"><span class="rl">${role}</span><pre>${E((m.content||"").slice(0,1500))}</pre></div>`;
  }).join("");
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
  if(e.target.tagName==="TEXTAREA")return;
  if(e.key==="ArrowLeft")select(cur-1); else if(e.key==="ArrowRight")select(cur+1);
  else if(e.key==="Home")select(0); else if(e.key==="End")select(steps.length-1);
});
let CHAT=[];  // conversation history
async function loadCopilot(){
  const p=document.getElementById("copilotpanel");
  if(!p.classList.contains("hidden")){p.classList.add("hidden");return;}
  p.classList.remove("hidden");
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
document.getElementById("copilot").onclick=loadCopilot;
document.getElementById("prev").onclick=()=>select(cur-1);
document.getElementById("next").onclick=()=>select(cur+1);
document.getElementById("first").onclick=()=>select(0);
document.getElementById("last").onclick=()=>select(steps.length-1);
load();
</script></body></html>"""
