"""Export a saved trace to Loom Studio: a self-contained HTML time-travel viewer.

No external assets, no JavaScript dependencies -- the file can be attached to a
bug report, emailed, or committed next to the trace itself. The timeline reads
without JavaScript; with it you get the scrubber (drag, ←/→, or press play and
watch the run happen), the per-step cost strip, the reconstructed conversation
at any point in time, and a copy-paste fork snippet for the selected turn.

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
  --shadow: 0 1px 2px rgba(11,11,11,.05), 0 10px 30px -12px rgba(11,11,11,.12);
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface: #1a1a19; --page: #0d0d0d; --ink: #ffffff; --ink-2: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --ring: rgba(255,255,255,0.10);
    --model: #3987e5; --tool: #199e70; --human: #d95926; --meta: #9085e9;
    --c1:#0d366b; --c2:#104281; --c3:#184f95; --c4:#1c5cab; --c5:#256abf;
    --c6:#3987e5; --c7:#6da7ec;
    --shadow: 0 1px 2px rgba(0,0,0,.4), 0 10px 30px -12px rgba(0,0,0,.6);
  }
}
* { box-sizing: border-box; margin: 0; }
html { scroll-behavior: smooth; }
body {
  background:
    radial-gradient(1100px 420px at 15% -8%, color-mix(in srgb, var(--model) 7%, transparent), transparent 60%),
    radial-gradient(900px 380px at 95% -12%, color-mix(in srgb, var(--meta) 6%, transparent), transparent 55%),
    var(--page);
  color: var(--ink);
  font: 14.5px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
  padding: 1.5rem 1.25rem 3rem; max-width: 1220px; margin: 0 auto;
}
::selection { background: color-mix(in srgb, var(--model) 25%, transparent); }
.topbar { display: flex; align-items: center; gap: .65rem; }
.logo { width: 20px; height: 20px; border-radius: 6px; flex: none;
  background:
    linear-gradient(90deg, transparent 42%, var(--page) 42% 58%, transparent 58%),
    linear-gradient(0deg,  transparent 42%, var(--page) 42% 58%, transparent 58%),
    linear-gradient(135deg, var(--model), var(--meta));
  box-shadow: var(--shadow); }
h1 { font-size: 1.15rem; letter-spacing: -.01em; }
h1 .brand { color: var(--muted); font-weight: 450; font-size: .95rem; }
.accentline { height: 2px; border: 0; border-radius: 2px; margin: .8rem 0 1.1rem;
  background: linear-gradient(90deg, var(--model), var(--tool), var(--human), var(--meta));
  opacity: .75; }
.sub { color: var(--ink-2); font-size: .95rem; margin: .35rem 0 0;
  overflow-wrap: anywhere; }
.banner { margin: .8rem 0 0; padding: .5rem .8rem; border-radius: 8px;
  font-size: .9rem; border: 1px solid; }
.banner code { background: rgba(127,127,127,.15); padding: 0 .3em; border-radius: 4px; }
.banner.safe { color: var(--tool); border-color: var(--tool);
  background: color-mix(in srgb, var(--tool) 10%, transparent); }
.banner.unsafe { color: var(--warn); border-color: var(--warn);
  background: color-mix(in srgb, var(--warn) 12%, transparent); }
.tiles { display: flex; flex-wrap: wrap; gap: .7rem; margin: 1.1rem 0 1.2rem; }
.tile { background: var(--surface); border: 1px solid var(--ring);
  border-radius: 14px; padding: .65rem 1.05rem; min-width: 7rem;
  box-shadow: var(--shadow); transition: transform .15s ease; }
.tile:hover { transform: translateY(-1px); }
.tile .k { color: var(--muted); font-size: .7rem; text-transform: uppercase;
  letter-spacing: .07em; }
.tile .v { font-size: 1.3rem; font-weight: 680; letter-spacing: -.01em;
  font-variant-numeric: tabular-nums; }
.tile.accent .v { color: var(--model); }
.deck { background: var(--surface); border: 1px solid var(--ring);
  border-radius: 14px; padding: .9rem 1rem .8rem; box-shadow: var(--shadow);
  margin-bottom: 1.1rem; }
.scrub { display: flex; align-items: center; gap: .8rem; }
.play { width: 34px; height: 34px; border-radius: 50%; border: 1px solid var(--ring);
  background: color-mix(in srgb, var(--model) 12%, var(--surface));
  color: var(--model); font-size: .8rem; cursor: pointer; flex: none;
  display: grid; place-items: center; transition: transform .12s ease, background .15s; }
.play:hover { transform: scale(1.08);
  background: color-mix(in srgb, var(--model) 22%, var(--surface)); }
.play:active { transform: scale(.94); }
input[type=range] { -webkit-appearance: none; appearance: none; flex: 1;
  height: 6px; border-radius: 99px; outline-offset: 4px; --p: 100%;
  background: linear-gradient(90deg, var(--model) var(--p), var(--grid) var(--p)); }
input[type=range]::-webkit-slider-thumb { -webkit-appearance: none;
  width: 18px; height: 18px; border-radius: 50%; background: var(--model);
  border: 3px solid var(--surface); box-shadow: 0 0 0 1px var(--ring), var(--shadow);
  cursor: grab; transition: transform .12s ease; }
input[type=range]::-webkit-slider-thumb:hover { transform: scale(1.15); }
input[type=range]::-moz-range-thumb { width: 15px; height: 15px; border-radius: 50%;
  background: var(--model); border: 3px solid var(--surface);
  box-shadow: 0 0 0 1px var(--ring); cursor: grab; }
.pos { font-variant-numeric: tabular-nums; color: var(--ink-2);
  font-size: .85rem; min-width: 9rem; text-align: right; }
.striplabel { color: var(--muted); font-size: .74rem; margin: .7rem 0 .35rem;
  letter-spacing: .02em; }
.strip { display: flex; gap: 3px; height: 34px; align-items: end; }
.strip .cell { flex: 1; border-radius: 4px 4px 2px 2px; background: var(--grid);
  min-width: 4px; cursor: pointer;
  transition: transform .12s ease, opacity .25s ease, filter .15s; }
.strip .cell:hover { transform: translateY(-3px) scaleY(1.04); filter: brightness(1.12); }
.strip .cell.future { opacity: .25; }
.cols { display: grid; grid-template-columns: minmax(330px, 1.25fr) minmax(290px, 1fr);
  gap: 1.1rem; align-items: start; }
@media (max-width: 840px) { .cols { grid-template-columns: 1fr; } }
.panel { background: var(--surface); border: 1px solid var(--ring);
  border-radius: 14px; overflow: hidden; box-shadow: var(--shadow); }
.panel h2 { font-size: .74rem; color: var(--muted); text-transform: uppercase;
  letter-spacing: .08em; padding: .75rem .95rem .3rem; }
.timeline { max-height: 70vh; overflow-y: auto; }
.step { display: flex; gap: .65rem; padding: .55rem .95rem; cursor: pointer;
  border-top: 1px solid var(--grid); border-left: 3px solid transparent;
  align-items: baseline;
  transition: background .15s, opacity .3s, border-color .15s; }
.step:hover { background: color-mix(in srgb, var(--model) 6%, transparent); }
.step.sel { background: color-mix(in srgb, var(--model) 12%, transparent);
  border-left-color: var(--model); }
.step.future { opacity: .3; }
.seq { color: var(--muted); font-size: .76rem; min-width: 2rem;
  font-variant-numeric: tabular-nums; }
.pill { display: inline-flex; align-items: center; gap: .4rem; flex: none;
  font-size: .74rem; font-weight: 620; padding: .12rem .6rem;
  border-radius: 99px; white-space: nowrap; align-self: center; }
.pill .dot { width: 7px; height: 7px; border-radius: 50%; }
.k-model .pill { color: var(--model);
  background: color-mix(in srgb, var(--model) 11%, transparent); }
.k-model .dot { background: var(--model); }
.k-tool .pill { color: var(--tool);
  background: color-mix(in srgb, var(--tool) 11%, transparent); }
.k-tool .dot { background: var(--tool); }
.k-human .pill { color: var(--human);
  background: color-mix(in srgb, var(--human) 12%, transparent); }
.k-human .dot { background: var(--human); }
.k-meta .pill { color: var(--meta);
  background: color-mix(in srgb, var(--meta) 11%, transparent); }
.k-meta .dot { background: var(--meta); }
.snippet { color: var(--ink-2); overflow-wrap: anywhere; white-space: pre-wrap;
  font-size: .86rem; }
.snippet .calls { color: var(--muted); font-style: italic; }
.depth1 { padding-left: 2.4rem; } .depth2 { padding-left: 3.9rem; }
.depth3 { padding-left: 5.4rem; }
.side { display: flex; flex-direction: column; gap: 1.1rem; }
.convo { max-height: 33vh; overflow-y: auto; padding: .4rem .95rem .9rem; }
.msg { margin: .5rem 0; padding: .55rem .8rem; border-radius: 12px;
  white-space: pre-wrap; overflow-wrap: anywhere; font-size: .88rem;
  animation: rise .22s ease both; }
@keyframes rise { from { opacity: 0; transform: translateY(4px); }
  to { opacity: 1; transform: none; } }
.msg .who { font-size: .68rem; text-transform: uppercase; letter-spacing: .07em;
  display: block; margin-bottom: .2rem; color: var(--muted); }
.msg.user { background: color-mix(in srgb, var(--ink) 6%, transparent);
  border-top-left-radius: 4px; margin-right: 12%; }
.msg.assistant { background: color-mix(in srgb, var(--model) 10%, transparent);
  border-top-right-radius: 4px; margin-left: 12%; }
.msg.assistant .who { color: var(--model); }
.msg.tool { background: color-mix(in srgb, var(--tool) 9%, transparent);
  margin-right: 12%; border-top-left-radius: 4px; }
.msg.tool .who { color: var(--tool); }
.msg.human { background: color-mix(in srgb, var(--human) 10%, transparent);
  margin-right: 12%; border-top-left-radius: 4px; }
.msg.human .who { color: var(--human); }
.detail { padding: .45rem .95rem .95rem; font-size: .86rem; }
.detail dl { display: grid; grid-template-columns: auto 1fr; gap: .2rem .9rem; }
.detail dt { color: var(--muted); }
.detail dd { font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
.detail pre { background: color-mix(in srgb, var(--ink) 5%, transparent);
  border: 1px solid var(--grid); border-radius: 10px; padding: .65rem .75rem;
  overflow-x: auto; margin-top: .55rem; font-size: .79rem; max-height: 25vh;
  line-height: 1.5; }
.forkbox { margin-top: .7rem; }
.forkbox .hint { color: var(--muted); font-size: .76rem; letter-spacing: .02em; }
button.copy { margin-top: .45rem; font: inherit; font-size: .8rem;
  padding: .35rem .85rem; border-radius: 99px; border: 1px solid var(--ring);
  background: var(--surface); color: var(--ink); cursor: pointer;
  transition: all .15s ease; }
button.copy:hover { border-color: var(--model); color: var(--model);
  background: color-mix(in srgb, var(--model) 8%, var(--surface)); }
button.copy:active { transform: scale(.96); }
.paused { color: var(--warn); font-weight: 600; margin: .75rem 0;
  padding: .55rem .9rem; border: 1px solid var(--warn); border-radius: 12px;
  background: color-mix(in srgb, var(--warn) 8%, transparent); }
.paused::before { content: "⏸ "; }
#tip { position: fixed; pointer-events: none; background: var(--ink);
  color: var(--page); padding: .3rem .6rem; border-radius: 8px;
  font-size: .76rem; opacity: 0; transition: opacity .12s;
  transform: translate(-50%, -130%); white-space: nowrap; z-index: 10;
  font-variant-numeric: tabular-nums; }
footer { color: var(--muted); font-size: .78rem; margin-top: 1.4rem;
  overflow-wrap: anywhere; }
.timeline::-webkit-scrollbar, .convo::-webkit-scrollbar { width: 8px; }
.timeline::-webkit-scrollbar-thumb, .convo::-webkit-scrollbar-thumb {
  background: var(--grid); border-radius: 99px; }
"""

_JS = """
const N = TRACE.log.length;
const slider = document.getElementById('scrub');
const pos = document.getElementById('pos');
const playBtn = document.getElementById('play');
const tip = document.getElementById('tip');
const rows = Array.from(document.querySelectorAll('.step'));
const cells = Array.from(document.querySelectorAll('.strip .cell'));
let selected = N - 1;
let timer = null;

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
  slider.style.setProperty('--p', (N > 1 ? (k / (N - 1)) * 100 : 100) + '%');
  pos.textContent = 'after step ' + k + ' of ' + (N - 1);
  rows.forEach((r, i) => {
    r.classList.toggle('future', i > k);
    r.classList.toggle('sel', i === selected);
  });
  cells.forEach((c, i) => c.classList.toggle('future', i > k));
  renderConvo(k);
  renderDetail(selected);
}

function stop() { clearInterval(timer); timer = null; playBtn.textContent = '▶'; }
playBtn.addEventListener('click', () => {
  if (timer) { stop(); return; }
  playBtn.textContent = '⏸';
  if (+slider.value >= N - 1) apply(0, 0);
  timer = setInterval(() => {
    const k = +slider.value;
    if (k >= N - 1) { stop(); return; }
    apply(k + 1, k + 1);
  }, 650);
});

slider.addEventListener('input', () => { stop(); apply(+slider.value); });
rows.forEach((r, i) => r.addEventListener('click', () => { stop(); apply(i, i); }));
cells.forEach((c, i) => {
  c.addEventListener('click', () => { stop(); apply(i, i); });
  c.addEventListener('mousemove', (ev) => {
    tip.textContent = c.dataset.tip;
    tip.style.left = ev.clientX + 'px';
    tip.style.top = ev.clientY + 'px';
    tip.style.opacity = 1;
  });
  c.addEventListener('mouseleave', () => { tip.style.opacity = 0; });
});
document.addEventListener('keydown', (ev) => {
  if (ev.key === 'ArrowLeft' && selected > 0) { stop(); apply(selected - 1, selected - 1); }
  if (ev.key === 'ArrowRight' && selected < N - 1) { stop(); apply(selected + 1, selected + 1); }
  if (ev.key === ' ' && ev.target === document.body) { ev.preventDefault(); playBtn.click(); }
});
const copyBtn = document.getElementById('copyfork');
copyBtn.addEventListener('click', () => {
  navigator.clipboard.writeText(document.getElementById('d-fork').textContent);
  copyBtn.textContent = 'copied ✓';
  setTimeout(() => { copyBtn.textContent = 'copy fork snippet'; }, 1200);
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


def _scrub_banner(data: dict) -> str:
    """A safe/unsafe banner: green if `loom share` scrubbed it, amber otherwise."""
    if data.get("scrubbed"):
        return ('<div class="banner safe">🛡️ Scrubbed &mdash; secrets redacted, '
                'safe to share.</div>')
    return ('<div class="banner unsafe">⚠️ Not scrubbed &mdash; this trace may contain '
            'secrets the agent saw. Run <code>loom share</code> before sharing it.</div>')


def _workspace_tile(ws: "dict | None") -> str:
    if not ws:
        return ""
    g = ws.get("git") or {}
    label = g.get("commit", "")[:10] or ws.get("os", "?")
    if g.get("dirty"):
        label += " ·dirty"
    return (
        '<div class="tile"><div class="k">workspace</div>'
        f'<div class="v" title="{html.escape(ws.get("cwd", ""))}">{html.escape(label)}</div></div>'
    )


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
            f'<span class="pill"><span class="dot"></span>{html.escape(kind)}</span>'
            f'<span class="snippet">{_step_detail(kind, e.get("result"))}</span>'
            f"</div>"
        )

    # Cost strip: sequential ramp, bucketed by each step's share of the max.
    max_tokens = max((_tokens_of(e) for e in log), default=0) or 1
    cells: list[str] = []
    for e in log:
        t = _tokens_of(e)
        if t == 0:
            style = "height:20%"
        else:
            bucket = min(7, 1 + int(6 * t / max_tokens))
            share = 32 + int(68 * t / max_tokens)
            style = f"height:{share}%;background:var(--c{bucket})"
        tipline = f"step {e.get('seq')} · {e.get('kind')} · {t or 'no'} tokens"
        cells.append(
            f'<div class="cell" style="{style}" data-tip="{html.escape(tipline)}"></div>'
        )

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
<div class="topbar">
  <div class="logo"></div>
  <h1>Loom Studio <span class="brand">— read · replay · rewind</span></h1>
</div>
<p class="sub">{title}</p>
{_scrub_banner(data)}
<hr class="accentline">
<div class="tiles">
  <div class="tile accent"><div class="k">model</div><div class="v">{html.escape(str(data.get("model", "?")))}</div></div>
  <div class="tile"><div class="k">steps</div><div class="v">{len(log)}</div></div>
  <div class="tile"><div class="k">turns</div><div class="v">{turns}</div></div>
  <div class="tile"><div class="k">input tokens</div><div class="v">{inp}</div></div>
  <div class="tile"><div class="k">output tokens</div><div class="v">{out}</div></div>
  {_workspace_tile(data.get("workspace"))}
</div>
{paused_html}
<div class="deck">
  <div class="scrub">
    <button id="play" class="play" title="replay the run (space)">▶</button>
    <input id="scrub" type="range" min="0" max="{max(len(log) - 1, 0)}" value="{max(len(log) - 1, 0)}">
    <span class="pos" id="pos"></span>
  </div>
  <div class="striplabel">tokens per step · click a bar to jump · ← / → to scrub · space to replay</div>
  <div class="strip">{"".join(cells)}</div>
</div>
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
        <button id="copyfork" class="copy">copy fork snippet</button>
      </div>
    </div></div>
  </div>
</div>
<div id="tip"></div>
<footer>output: {html.escape(str(data.get("output", "")))}<br>
generated by loom — the agent harness you can read, replay, and rewind</footer>
<script>
const TRACE = {trace_json};
{_JS}
</script>
</body>
</html>
"""
