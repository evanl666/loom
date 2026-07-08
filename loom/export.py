"""Export a saved trace to Loom Studio: a self-contained HTML time-travel viewer.

No external assets, no JavaScript dependencies -- the file can be attached to a
bug report, emailed, or committed next to the trace itself. The timeline reads
without JavaScript; with it you get the scrubber (drag or ←/→), the per-step
cost strip, the reconstructed conversation at any point in time, and a
copy-paste fork snippet for the selected turn.

    loom export run.loom.json            # writes run.loom.html
    loom studio run.loom.json            # same, then opens it in the browser
"""

from __future__ import annotations

import html
import json

from .providers.base import ModelResponse

# Palette: the dataviz reference instance (validated categorical slots +
# sequential blue ramp). Categorical hues are assigned by entity, fixed order:
# model=blue, tool=aqua, human=orange, harness meta-effects=violet.
_CSS = """
:root {
  --surface: #fcfcfb; --page: #f9f9f7; --ink: #0b0b0b; --ink-2: #52514e;
  --muted: #898781; --grid: #e1e0d9; --ring: rgba(11,11,11,0.10);
  --model: #2a78d6; --tool: #1baf7a; --human: #eb6834; --meta: #4a3aa7;
  --warn: #fab219;
  --c1:#cde2fb; --c2:#9ec5f4; --c3:#6da7ec; --c4:#3987e5; --c5:#256abf;
  --c6:#184f95; --c7:#0d366b;
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface: #1a1a19; --page: #0d0d0d; --ink: #ffffff; --ink-2: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --ring: rgba(255,255,255,0.10);
    --model: #3987e5; --tool: #199e70; --human: #d95926; --meta: #9085e9;
    --c1:#0d366b; --c2:#104281; --c3:#184f95; --c4:#1c5cab; --c5:#256abf;
    --c6:#3987e5; --c7:#6da7ec;
  }
}
* { box-sizing: border-box; margin: 0; }
body { background: var(--page); color: var(--ink);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
  padding: 1.25rem; max-width: 1200px; margin: 0 auto; }
h1 { font-size: 1.1rem; }
h1 .brand { color: var(--muted); font-weight: 500; }
.sub { color: var(--ink-2); font-size: .9rem; margin: .15rem 0 1rem;
  overflow-wrap: anywhere; }
.tiles { display: flex; flex-wrap: wrap; gap: .6rem; margin-bottom: 1rem; }
.tile { background: var(--surface); border: 1px solid var(--ring);
  border-radius: 10px; padding: .55rem .9rem; min-width: 6.5rem; }
.tile .k { color: var(--muted); font-size: .72rem; text-transform: uppercase;
  letter-spacing: .04em; }
.tile .v { font-size: 1.15rem; font-weight: 650; }
.scrub { display: flex; align-items: center; gap: .75rem; margin: .25rem 0 .4rem; }
.scrub input[type=range] { flex: 1; accent-color: var(--model); }
.scrub .pos { font-variant-numeric: tabular-nums; color: var(--ink-2);
  font-size: .85rem; min-width: 8.5rem; text-align: right; }
.strip { display: flex; gap: 2px; height: 26px; margin-bottom: 1rem; }
.strip .cell { flex: 1; border-radius: 4px 4px 0 0; align-self: end;
  background: var(--grid); min-width: 3px; cursor: pointer; }
.strip .cell.future { opacity: .3; }
.striplabel { color: var(--muted); font-size: .74rem; margin-bottom: .25rem; }
.cols { display: grid; grid-template-columns: minmax(320px, 1.2fr) minmax(280px, 1fr);
  gap: 1rem; align-items: start; }
@media (max-width: 820px) { .cols { grid-template-columns: 1fr; } }
.panel { background: var(--surface); border: 1px solid var(--ring);
  border-radius: 10px; overflow: hidden; }
.panel h2 { font-size: .8rem; color: var(--muted); text-transform: uppercase;
  letter-spacing: .04em; padding: .6rem .8rem .2rem; }
.timeline { max-height: 72vh; overflow-y: auto; }
.step { display: flex; gap: .6rem; padding: .45rem .8rem; cursor: pointer;
  border-top: 1px solid var(--grid); align-items: baseline; }
.step:hover { background: color-mix(in srgb, var(--model) 7%, transparent); }
.step.sel { background: color-mix(in srgb, var(--model) 14%, transparent); }
.step.future { opacity: .35; }
.seq { color: var(--muted); font-size: .78rem; min-width: 2rem;
  font-variant-numeric: tabular-nums; }
.dot { width: 9px; height: 9px; border-radius: 50%; flex: none;
  align-self: center; }
.k-model .dot { background: var(--model); } .k-tool .dot { background: var(--tool); }
.k-human .dot { background: var(--human); } .k-meta .dot { background: var(--meta); }
.kind { font-size: .78rem; font-weight: 600; white-space: nowrap; }
.k-model .kind { color: var(--model); } .k-tool .kind { color: var(--tool); }
.k-human .kind { color: var(--human); } .k-meta .kind { color: var(--meta); }
.snippet { color: var(--ink-2); overflow-wrap: anywhere; white-space: pre-wrap;
  font-size: .85rem; }
.snippet .calls { color: var(--muted); }
.depth1 { padding-left: 2.2rem; } .depth2 { padding-left: 3.6rem; }
.depth3 { padding-left: 5rem; }
.side { display: flex; flex-direction: column; gap: 1rem; }
.convo { max-height: 34vh; overflow-y: auto; padding: .4rem .8rem .8rem; }
.msg { margin: .45rem 0; padding: .5rem .7rem; border-radius: 10px;
  background: color-mix(in srgb, var(--ink) 5%, transparent);
  white-space: pre-wrap; overflow-wrap: anywhere; font-size: .88rem; }
.msg .who { color: var(--muted); font-size: .72rem; text-transform: uppercase;
  letter-spacing: .04em; display: block; margin-bottom: .15rem; }
.msg.user { border-left: 3px solid var(--ink-2); }
.msg.assistant { border-left: 3px solid var(--model); }
.msg.tool { border-left: 3px solid var(--tool); }
.msg.human { border-left: 3px solid var(--human); }
.detail { padding: .4rem .8rem .8rem; font-size: .85rem; }
.detail dl { display: grid; grid-template-columns: auto 1fr; gap: .15rem .8rem; }
.detail dt { color: var(--muted); } .detail dd { font-variant-numeric: tabular-nums; }
.detail pre { background: color-mix(in srgb, var(--ink) 5%, transparent);
  border-radius: 8px; padding: .6rem; overflow-x: auto; margin-top: .5rem;
  font-size: .8rem; max-height: 26vh; }
.forkbox { margin-top: .6rem; }
.forkbox pre { margin-top: .25rem; }
.forkbox button { margin-top: .35rem; font: inherit; font-size: .8rem;
  padding: .3rem .7rem; border-radius: 8px; border: 1px solid var(--ring);
  background: var(--surface); color: var(--ink); cursor: pointer; }
.forkbox button:hover { border-color: var(--model); color: var(--model); }
.paused { color: var(--warn); font-weight: 600; margin: .75rem 0;
  padding: .5rem .8rem; border: 1px solid var(--warn); border-radius: 10px; }
.paused::before { content: "⏸ "; }
footer { color: var(--muted); font-size: .78rem; margin-top: 1.25rem; }
.hint { color: var(--muted); font-size: .78rem; }
"""

_JS = """
const N = TRACE.log.length;
const slider = document.getElementById('scrub');
const pos = document.getElementById('pos');
const rows = Array.from(document.querySelectorAll('.step'));
const cells = Array.from(document.querySelectorAll('.strip .cell'));
let selected = N - 1;

function turnOf(seq) {
  let t = 0;
  for (const e of TRACE.log) {
    if (e.seq >= seq) break;
    if (e.kind === 'model' && (e.depth || 0) === 0) t++;
  }
  return t;
}

function renderConvo(k) {
  const box = document.getElementById('convo');
  box.textContent = '';
  const eps = TRACE.episodes || [];
  let ep = 0;
  const add = (who, cls, text) => {
    const d = document.createElement('div');
    d.className = 'msg ' + cls;
    const w = document.createElement('span');
    w.className = 'who'; w.textContent = who;
    d.appendChild(w); d.appendChild(document.createTextNode(text));
    box.appendChild(d);
  };
  if (eps.length) { add('user', 'user', eps[0]); ep = 1; }
  for (const e of TRACE.log) {
    if (e.seq > k) break;
    if (e.kind === 'model') {
      const r = e.result || {};
      let text = r.text || '';
      if (r.tool_calls && r.tool_calls.length)
        text += (text ? '\\n' : '') + r.tool_calls.map(
          tc => '→ ' + tc.name + '(' + JSON.stringify(tc.input) + ')').join('\\n');
      add('assistant', 'assistant', text || '(empty)');
      if (r.stop_reason === 'end_turn' && ep < eps.length) { add('user', 'user', eps[ep]); ep++; }
    } else if (e.kind.startsWith('tool:')) {
      add(e.kind, 'tool', typeof e.result === 'string' ? e.result : JSON.stringify(e.result));
    } else if (e.kind === 'human') {
      add('human', 'human', String(e.result));
    }
  }
  box.scrollTop = box.scrollHeight;
}

function renderDetail(seq) {
  const e = TRACE.log[seq];
  if (!e) return;
  const set = (id, v) => { document.getElementById(id).textContent = v; };
  set('d-kind', e.kind); set('d-seq', e.seq); set('d-depth', e.depth || 0);
  set('d-key', e.key || '');
  const u = (e.result && e.result.usage) || {};
  set('d-tokens', e.kind === 'model'
    ? (u.input_tokens || 0) + ' in / ' + (u.output_tokens || 0) + ' out' : '—');
  document.getElementById('d-json').textContent = JSON.stringify(e.result, null, 2);
  document.getElementById('d-fork').textContent =
    'run = Run.load("trace.loom.json", agent=agent)\\n' +
    'branch = run.fork(at=' + turnOf(e.seq) + ', edit=lambda ctx: ...)';
}

function apply(k, sel) {
  selected = sel === undefined ? k : sel;
  slider.value = k;
  pos.textContent = 'after step ' + k + ' of ' + (N - 1);
  rows.forEach((r, i) => {
    r.classList.toggle('future', i > k);
    r.classList.toggle('sel', i === selected);
  });
  cells.forEach((c, i) => c.classList.toggle('future', i > k));
  renderConvo(k);
  renderDetail(selected);
}

slider.addEventListener('input', () => apply(+slider.value));
rows.forEach((r, i) => r.addEventListener('click', () => apply(i, i)));
cells.forEach((c, i) => c.addEventListener('click', () => apply(i, i)));
document.addEventListener('keydown', (ev) => {
  if (ev.key === 'ArrowLeft' && selected > 0) apply(selected - 1, selected - 1);
  if (ev.key === 'ArrowRight' && selected < N - 1) apply(selected + 1, selected + 1);
});
document.getElementById('copyfork').addEventListener('click', () => {
  navigator.clipboard.writeText(document.getElementById('d-fork').textContent);
});
apply(N - 1, N - 1);
"""


def _kind_class(kind: str) -> str:
    if kind == "model":
        return "k-model"
    if kind == "human":
        return "k-human"
    if kind.startswith("tool:"):
        return "k-tool"
    return "k-meta"  # edit / compact / memory / critic / sample / choose / time...


def _step_detail(kind: str, result) -> str:
    """Escaped HTML for one step's content."""
    if kind == "model":
        resp = ModelResponse.from_dict(result)
        parts = []
        if resp.text:
            parts.append(html.escape(resp.text))
        if resp.tool_calls:
            calls = ", ".join(
                f"{tc.name}({json.dumps(tc.input)})" for tc in resp.tool_calls
            )
            parts.append(f'<span class="calls">→ calls {html.escape(calls)}</span>')
        return "<br>".join(parts) or "<em>(empty)</em>"
    text = result if isinstance(result, str) else json.dumps(result)
    return html.escape(text)


def _tokens_of(e: dict) -> int:
    if e.get("kind") != "model":
        return 0
    u = e.get("result", {}).get("usage", {}) or {}
    return u.get("input_tokens", 0) + u.get("output_tokens", 0)


def trace_to_html(data: dict) -> str:
    """Render a saved trace dict (``Run.to_dict`` / a ``.loom.json`` file) to HTML."""
    log = data.get("log", [])
    episodes = data.get("episodes") or [data.get("prompt", "")]

    inp = out = turns = 0
    for e in log:
        if e.get("kind") == "model":
            u = e.get("result", {}).get("usage", {}) or {}
            inp += u.get("input_tokens", 0)
            out += u.get("output_tokens", 0)
            if e.get("depth", 0) == 0:
                turns += 1

    rows: list[str] = []
    for e in log:
        kind = e.get("kind", "?")
        depth = min(e.get("depth", 0), 3)
        rows.append(
            f'<div class="step {_kind_class(kind)} depth{depth}" data-seq="{e.get("seq", 0)}">'
            f'<span class="seq">{e.get("seq", "")}</span>'
            f'<span class="dot"></span>'
            f'<span class="kind">{html.escape(kind)}</span>'
            f'<span class="snippet">{_step_detail(kind, e.get("result"))}</span>'
            f"</div>"
        )

    # Cost strip: sequential ramp, bucketed by each step's share of the max.
    max_tokens = max((_tokens_of(e) for e in log), default=0) or 1
    cells: list[str] = []
    for e in log:
        t = _tokens_of(e)
        if t == 0:
            style = "height:22%"
        else:
            bucket = min(7, 1 + int(6 * t / max_tokens))
            share = 30 + int(70 * t / max_tokens)
            style = f"height:{share}%;background:var(--c{bucket})"
        title = f"step {e.get('seq')}: {e.get('kind')} — {t or 'no'} tokens"
        cells.append(f'<div class="cell" style="{style}" title="{html.escape(title)}"></div>')

    paused_html = (
        f'<p class="paused">paused — waiting for a human answer to: '
        f"{html.escape(str(data.get('pending')))}</p>"
        if data.get("paused")
        else ""
    )
    title = html.escape(episodes[0][:80]) or "Loom trace"
    # <, >, & escaped inside the JSON payload so markup can never leak out of it.
    trace_json = (
        json.dumps({"log": log, "episodes": episodes})
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Loom Studio — {title}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Loom Studio <span class="brand">— read, replay, rewind</span></h1>
<p class="sub">{title}</p>
<div class="tiles">
  <div class="tile"><div class="k">model</div><div class="v">{html.escape(str(data.get("model", "?")))}</div></div>
  <div class="tile"><div class="k">steps</div><div class="v">{len(log)}</div></div>
  <div class="tile"><div class="k">turns</div><div class="v">{turns}</div></div>
  <div class="tile"><div class="k">input tokens</div><div class="v">{inp}</div></div>
  <div class="tile"><div class="k">output tokens</div><div class="v">{out}</div></div>
</div>
{paused_html}
<div class="scrub">
  <input id="scrub" type="range" min="0" max="{max(len(log) - 1, 0)}" value="{max(len(log) - 1, 0)}">
  <span class="pos" id="pos"></span>
</div>
<div class="striplabel">tokens per step (click a bar to jump) · ← / → to scrub</div>
<div class="strip">{"".join(cells)}</div>
<div class="cols">
  <div class="panel"><h2>Timeline</h2><div class="timeline">{"".join(rows)}</div></div>
  <div class="side">
    <div class="panel"><h2>Conversation at this point</h2><div class="convo" id="convo"></div></div>
    <div class="panel"><h2>Selected effect</h2><div class="detail">
      <dl>
        <dt>kind</dt><dd id="d-kind"></dd>
        <dt>seq</dt><dd id="d-seq"></dd>
        <dt>depth</dt><dd id="d-depth"></dd>
        <dt>input key</dt><dd id="d-key"></dd>
        <dt>tokens</dt><dd id="d-tokens"></dd>
      </dl>
      <pre id="d-json"></pre>
      <div class="forkbox">
        <span class="hint">rewind here in Python:</span>
        <pre id="d-fork"></pre>
        <button id="copyfork">copy fork snippet</button>
      </div>
    </div></div>
  </div>
</div>
<footer>output: {html.escape(str(data.get("output", "")))}<br>
generated by loom — the agent harness you can read, replay, and rewind</footer>
<script>
const TRACE = {trace_json};
{_JS}
</script>
</body>
</html>
"""
