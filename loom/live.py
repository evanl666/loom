"""Live Studio: watch an agent run in real time, and approve/deny as it goes.

``loom proxy --live`` (or ``loom record --live``) serves a page that polls the
growing trace and the shield's pending approvals, so you can *watch Claude
Code work* -- every model call, tool call, token, and firewall decision as it
happens -- and click Approve / Deny on anything the firewall is holding.

It reuses the recorder (which builds the trace incrementally) and the shield
control plane (``/loom/shield/pending`` + ``/decide``); the only new surface
is ``/loom/live`` (this page) and ``/loom/live/state`` (its JSON feed).
"""

from __future__ import annotations

import html
import json


def live_state(server) -> dict:
    """A snapshot for the live page: effects so far, cost, pending approvals."""
    from .providers.base import ModelResponse

    rec = server.recorder
    effects = []
    inp = out = 0
    with server.lock:  # persist() mutates log and shield_events under this lock
        log = list(rec.log)
        denied = sum(1 for ev in rec.shield_events if ev.get("action") == "deny")
    for e in log:
        row = {"seq": e.seq, "kind": e.kind, "depth": getattr(e, "depth", 0)}
        if e.kind == "model" and isinstance(e.result, dict):
            resp = ModelResponse.from_dict(e.result)
            usage = e.result.get("usage") or {}
            inp += usage.get("input_tokens", 0) or 0
            out += usage.get("output_tokens", 0) or 0
            if resp.tool_calls:
                row["detail"] = "calls " + ", ".join(
                    f"{tc.name}({json.dumps(tc.input, default=str)})" for tc in resp.tool_calls)
            else:
                row["detail"] = resp.text[:200]
        else:
            row["detail"] = (e.result if isinstance(e.result, str) else json.dumps(e.result))[:200]
        effects.append(row)

    pending = server.shield.pending_list() if server.shield is not None else []
    return {
        "effects": effects,
        "input_tokens": inp, "output_tokens": out,
        "model": rec.model, "episodes": rec.episodes,
        "pending": pending, "shield_denied": denied,
        "done": server._finalized,
    }


def live_html(port: int, token: str) -> str:
    """The live viewer page. Polls /loom/live/state; approves via the control plane."""
    return _PAGE.replace("__PORT__", str(port)).replace("__TOKEN__", html.escape(token or ""))


_PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Loom Live</title>
<style>
:root { --bg:#0d0d0d; --card:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --muted:#898781;
  --model:#3987e5; --tool:#199e70; --human:#d95926; --warn:#fab219; --deny:#ff6b6b;
  --ring:rgba(255,255,255,.1); }
@media (prefers-color-scheme: light) { :root { --bg:#f9f9f7; --card:#fff; --ink:#0b0b0b;
  --ink2:#52514e; --muted:#898781; --ring:rgba(0,0,0,.1); } }
* { box-sizing:border-box; margin:0; }
body { background:var(--bg); color:var(--ink); font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; padding:24px; }
h1 { font-size:18px; } h1 .live { color:var(--deny); font-size:12px; vertical-align:middle; }
.sub { color:var(--muted); margin-bottom:16px; font-size:13px; }
.tiles { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:16px; }
.tile { background:var(--card); border:1px solid var(--ring); border-radius:8px; padding:10px 14px; min-width:110px; }
.tile .v { font-size:22px; font-weight:650; } .tile .k { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.05em; }
.pending { background:var(--card); border:1px solid var(--warn); border-radius:10px; padding:12px 14px; margin-bottom:10px; }
.pending .q { color:var(--warn); font-weight:600; margin-bottom:8px; }
.pending code { background:rgba(127,127,127,.15); padding:1px 5px; border-radius:4px; }
.btn { font:inherit; cursor:pointer; border-radius:99px; padding:5px 16px; margin-right:8px; border:1px solid; }
.approve { color:var(--tool); border-color:var(--tool); background:transparent; }
.deny { color:var(--deny); border-color:var(--deny); background:transparent; }
.feed { display:flex; flex-direction:column; gap:4px; }
.row { background:var(--card); border:1px solid var(--ring); border-radius:8px; padding:7px 12px; display:flex; gap:10px; align-items:baseline; }
.row.new { animation:rise .4s ease; }
@keyframes rise { from { opacity:0; transform:translateY(6px);} to { opacity:1; } }
.seq { color:var(--muted); font-variant-numeric:tabular-nums; min-width:28px; }
.pill { font-size:11px; font-weight:600; padding:1px 7px; border-radius:99px; }
.k-model{color:var(--model);border:1px solid var(--model);} .k-tool{color:var(--tool);border:1px solid var(--tool);}
.k-human{color:var(--human);border:1px solid var(--human);} .k-meta{color:var(--muted);border:1px solid var(--muted);}
.detail { color:var(--ink2); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.done { color:var(--tool); }
</style></head><body>
<h1>Loom Live <span class="live" id="livedot">● recording</span></h1>
<p class="sub" id="prompt">connecting…</p>
<div class="tiles">
  <div class="tile"><div class="v" id="steps">0</div><div class="k">steps</div></div>
  <div class="tile"><div class="v" id="tokens">0</div><div class="k">tokens</div></div>
  <div class="tile"><div class="v" id="blocked">0</div><div class="k">blocked</div></div>
</div>
<div id="pendingwrap"></div>
<div class="feed" id="feed"></div>
<script>
const PORT = __PORT__, TOKEN = "__TOKEN__";
const H = TOKEN ? {"x-loom-token": TOKEN} : {};
const kclass = k => k==="model"?"k-model":k.startsWith("tool:")?"k-tool":k==="human"?"k-human":"k-meta";
let shown = 0;
async function decide(id, ok) {
  await fetch(`http://127.0.0.1:${PORT}/loom/shield/decide`,
    {method:"POST", headers:{"content-type":"application/json", ...H},
     body:JSON.stringify({id, decision: ok?"approve":"deny"})});
  tick();
}
async function tick() {
  let s; try { s = await (await fetch(`http://127.0.0.1:${PORT}/loom/live/state`, {headers:H})).json(); }
  catch(e){ document.getElementById("livedot").textContent="● proxy closed"; return; }
  document.getElementById("prompt").textContent = (s.episodes&&s.episodes[0]) || "(waiting for the agent)";
  document.getElementById("steps").textContent = s.effects.length;
  document.getElementById("tokens").textContent = (s.input_tokens+s.output_tokens).toLocaleString();
  document.getElementById("blocked").textContent = s.shield_denied;
  if (s.done) { const d=document.getElementById("livedot"); d.textContent="● done"; d.className="live done"; }
  const pw = document.getElementById("pendingwrap");
  pw.innerHTML = (s.pending||[]).map(p =>
    `<div class="pending"><div class="q">⏸ approve this tool call?</div>
      <code>${escapeHtml(p.tool)}(${escapeHtml(JSON.stringify(p.input||{}))})</code>
      <div style="margin-top:8px">
        <button class="btn approve" onclick="decide('${p.id}',true)">Approve</button>
        <button class="btn deny" onclick="decide('${p.id}',false)">Deny</button>
      </div></div>`).join("");
  const feed = document.getElementById("feed");
  for (let i=shown; i<s.effects.length; i++) {
    const e = s.effects[i];
    const div = document.createElement("div");
    div.className = "row new";
    div.innerHTML = `<span class="seq">${e.seq}</span>
      <span class="pill ${kclass(e.kind)}">${escapeHtml(e.kind)}</span>
      <span class="detail">${escapeHtml(e.detail||"")}</span>`;
    feed.appendChild(div);
  }
  shown = s.effects.length;
}
function escapeHtml(s){ return String(s).replace(/[&<>"]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
tick(); setInterval(tick, 700);
</script></body></html>"""
