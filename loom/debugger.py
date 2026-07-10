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

    def __init__(self, trace_path: str, agent: Any = None):
        self.trace_path = trace_path
        self.agent = agent
        with open(trace_path) as f:
            self.data = json.load(f)

    def _agent_for(self, model: str):
        if self.agent is None:
            raise RuntimeError("no --agent given; the run cannot be re-forked live")
        if not model or model == "keep":
            return self.agent
        from .agent import Agent
        return Agent(model=model, tools=list(self.agent.tools.values()),
                     system=self.agent.system, max_steps=self.agent.max_steps)

    def fork(self, at: int, append: str = "", model: str = "keep") -> dict:
        """Replay 0..at-1 from the log, apply the edit, run the tail live."""
        from .trace import Run

        agent = self._agent_for(model)
        base = Run.load(self.trace_path, agent=agent)
        edit = None
        if append.strip():
            text = append  # captured for the callback
            edit = lambda ctx: ctx.add_user(text, source="debugger")  # noqa: E731
        branch = base.fork(at=at, edit=edit)
        return _branch_payload(self.data, branch.to_dict(), at)


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
            })
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
        try:
            result = sess.fork(at, str(body.get("append", "")), str(body.get("model", "keep")))
            self._json(200, result)
        except (IndexError, RuntimeError, ValueError) as e:
            self._json(400, {"error": str(e)})
        except Exception as e:  # noqa: BLE001 -- a live-call failure shouldn't kill the server
            self._json(502, {"error": f"fork failed: {type(e).__name__}: {e}"})


class DebugServer:
    """Serves the step-debugger for one trace. Bind port 0 to pick a free port."""

    def __init__(self, trace_path: str, agent: Any = None,
                 port: int = 8790, host: str = "127.0.0.1"):
        self.session = DebugSession(trace_path, agent)
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
.frame{border:1px solid #24242a;border-radius:8px;margin:6px 0;overflow:hidden}
.frame .rl{display:block;font-size:11px;color:#8a8a92;padding:4px 10px;background:#161619;border-bottom:1px solid #22222a}
.frame.curframe{border-color:#4a9eff}.frame.curframe .rl{color:#4a9eff;background:#12202f}
.frame pre{border:0;border-radius:0;max-height:180px;background:#0f0f12}
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
      <span class="muted" style="margin-left:auto">← → step · click a step to jump</span>
    </div>
    <div id="detail"></div>
  </div>
</div>
<script>
const E=s=>String(s==null?"":s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const J=x=>{try{return JSON.stringify(x,null,2)}catch(e){return String(x)}};
let RUN=null, steps=[], cur=0, canFork=false;

async function load(){
  RUN=await (await fetch("/api/run")).json();
  steps=RUN.steps; canFork=RUN.can_fork;
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
  if(s.input!=null) h+=`<div class="k">${s.tool?"tool input / code":"input"}</div><pre>${E(J(s.input))}</pre>`;
  if(o.text) h+=`<div class="k">result</div><pre>${E(o.text.length>4000?o.text.slice(0,4000)+"\n… (truncated)":o.text)}</pre>`;
  if(s.state_diff&&s.state_diff.kind&&s.state_diff.kind!=="none"){
    h+=`<div class="k">🌍 world change · ${E(s.state_diff.kind)}</div>`;
    if(s.state_diff.summary) h+=`<div class="sub2">${E(s.state_diff.summary)}</div>`;
    if(s.state_diff.detail) h+=`<pre class="diff">${diffHtml(typeof s.state_diff.detail==="string"?s.state_diff.detail:J(s.state_diff.detail))}</pre>`;
  }
  h+=`<div class="k">🧠 context the model saw here <button id="ctxbtn" class="mini">show</button></div><div id="ctx"></div>`;
  if(s.capabilities&&s.capabilities.length) h+=`<div class="k">capabilities</div>`+s.capabilities.map(c=>`<span class="chip">${E(c)}</span>`).join("");
  if(s.risk) h+=`<div class="k">risk</div><span class="chip risk">⚠ ${E(s.risk)}</span>`;
  if(s.policy) h+=`<div class="k">firewall</div><span class="chip">${E(s.policy.action)} ${E(s.policy.rule||"")}</span>`;
  if(o.tokens&&(o.tokens.input_tokens||o.tokens.output_tokens)) h+=`<div class="k">tokens</div><span class="chip">in ${o.tokens.input_tokens||0}</span><span class="chip">out ${o.tokens.output_tokens||0}</span>`;
  const forkable=(s.replay||{}).forkable;
  h+=`<div id="fork" class="${forkable&&canFork?"":"hidden"}">
    <div class="k">🍴 fork from turn ${(s.replay||{}).turn} — edit context / model, run the tail live</div>
    <textarea id="append" rows="3" placeholder="Inject a message into the model's context at this turn (optional) — e.g. 'Actually, do NOT issue the refund.'"></textarea>
    <div style="display:flex;gap:8px;margin-top:8px;align-items:center">
      <select id="model"><option value="keep">model: keep (${E(RUN.model||"recorded")})</option>
        <option value="claude-haiku-4-5-20251001">claude-haiku-4-5</option>
        <option value="claude-sonnet-5">claude-sonnet-5</option>
        <option value="claude-opus-4-8">claude-opus-4-8</option></select>
      <button id="run">▶ Fork &amp; Run live</button>
    </div>
    <div id="branch"></div>
  </div>`;
  if(!canFork&&forkable) h+=`<div class="k muted">re-run disabled — start with <kbd>--agent module:attr</kbd> to fork live</div>`;
  d.innerHTML=h;
  const rb=document.getElementById("run"); if(rb) rb.onclick=doFork;
  const cb=document.getElementById("ctxbtn"); if(cb) cb.onclick=loadContext;
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
async function doFork(){
  const s=steps[cur], at=(s.replay||{}).turn;
  const btn=document.getElementById("run"), bx=document.getElementById("branch");
  btn.disabled=true; btn.innerHTML='<span class="spin">⟳</span> running…';
  bx.innerHTML="";
  try{
    const r=await fetch("/api/fork",{method:"POST",headers:{"content-type":"application/json"},
      body:JSON.stringify({at,append:document.getElementById("append").value,model:document.getElementById("model").value})});
    const res=await r.json();
    if(!r.ok){bx.innerHTML=`<pre class="risky">${E(res.error||"fork failed")}</pre>`;return;}
    const div=res.diverge;
    const bs=res.branch_steps.map((b,i)=>{
      const cls=(div!=null&&i>=div)?"branchstep div":(i>=(div??1e9)?"branchstep new":"branchstep");
      return `<div class="${cls}"><span class="muted">${b.step}</span> <b>${E(b.tool||b.type)}</b> ${E((b.intent||"").slice(0,60))}</div>`;
    }).join("");
    bx.innerHTML=`<div class="k">new branch output</div><pre>${E(res.branch_output)}</pre>`+
                 `<div class="k">branch steps ${div!=null?"(diverges at "+div+")":""}</div>${bs}`;
  }catch(e){bx.innerHTML=`<pre class="risky">${E(e)}</pre>`;}
  finally{btn.disabled=false; btn.innerHTML="▶ Fork &amp; Run live";}
}
document.addEventListener("keydown",e=>{
  if(e.target.tagName==="TEXTAREA")return;
  if(e.key==="ArrowLeft")select(cur-1); else if(e.key==="ArrowRight")select(cur+1);
  else if(e.key==="Home")select(0); else if(e.key==="End")select(steps.length-1);
});
document.getElementById("prev").onclick=()=>select(cur-1);
document.getElementById("next").onclick=()=>select(cur+1);
document.getElementById("first").onclick=()=>select(0);
document.getElementById("last").onclick=()=>select(steps.length-1);
load();
</script></body></html>"""
