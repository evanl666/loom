"""``loom movie``: the 30-second incident animation.

A trace is for engineers; a *movie* is for everyone else. This auto-cuts a run
into scenes -- what the agent did, which sensitive value it read, where it
tried to send it, what the firewall decided, and the final behavior score --
and renders them as one self-playing HTML file you can drop in a tweet, a
README, or a postmortem review:

    loom movie session.loom.json            # -> session.movie.html

Auto-advances scene by scene (space = pause, arrows = step). Self-contained:
no network, no build, safe to share after `loom share`.
"""

from __future__ import annotations

import html as _html
import json


def _scenes(data: dict) -> "list[dict]":
    """Cut the run into the scenes worth watching."""
    from .action import actions as _actions
    from .diff import score_breakdown
    from .packs import install_builtin
    from .taint import taint_paths

    install_builtin()
    acts = _actions(data)
    calls = [a for a in acts if a.type == "call"]
    prompt = (data.get("episodes") or [data.get("prompt", "")])[0]

    scenes: list[dict] = [{
        "kind": "title", "icon": "🎬",
        "title": "An agent went to work",
        "body": f"“{prompt[:120]}”",
        "sub": f"{data.get('model', 'agent')} · {len(calls)} action(s) recorded by Loom",
    }]

    # The interesting actions: risky, world-changing, or blocked.
    for a in calls:
        blocked = bool(a.policy and a.policy.blocked)
        if not (a.risk or a.state_diff is not None or blocked):
            continue
        if blocked:
            icon, title = "🛡️", f"Loom blocked {a.tool}"
            sub = f"rule: {a.policy.rule}" if a.policy.rule else "firewall deny"
        elif a.risk:
            icon, title = "⚠️", f"{a.tool} — {a.risk}"
            sub = a.state_diff.summary if a.state_diff else ", ".join(
                c for c in a.capabilities if c not in ("read", "idempotent"))
        else:
            icon, title = "🔧", a.tool
            sub = a.state_diff.summary if a.state_diff else ""
        scenes.append({
            "kind": "blocked" if blocked else ("risky" if a.risk else "action"),
            "icon": icon, "title": title,
            "body": f'“{a.intent[:110]}”' if a.intent else "",
            "sub": sub, "step": a.step if a.step >= 0 else None,
        })

    # The leak chain, if the values actually flowed.
    for p in taint_paths(data)[:2]:
        scenes.append({
            "kind": "taint", "icon": "⛓️",
            "title": f"{p['kind']} left the box",
            "body": (f"read at step {p['source']['step']} ({p['source']['tool']}) → "
                     f"carried into step {p['sink']['step']} ({p['sink']['tool']})"),
            "sub": f"value {p['value_preview']} via {', '.join(p['sink']['via'])}",
        })

    # The verdict.
    bd = score_breakdown(data)
    worst = min(bd["dimensions"].items(), key=lambda kv: kv[1]["score"])
    denies = sum(1 for ev in (data.get("shield_events") or [])
                 if ev.get("action") == "deny")
    scenes.append({
        "kind": "score", "icon": "📊",
        "title": f"Behavior score: {bd['overall']}/100",
        "body": f"weakest: {worst[0]} ({worst[1]['score']}) — {worst[1]['why']}",
        "sub": (f"{denies} call(s) blocked by the firewall" if denies
                else "no firewall interventions"),
        "score": bd["overall"],
    })
    scenes.append({
        "kind": "end", "icon": "🧵",
        "title": "Recorded. Firewalled. Explained.",
        "body": "Every step above is replayable offline from the trace.",
        "sub": "loom-harness — the black-box debugger for tool-using AI agents",
    })
    return scenes


_CSS = """
* { box-sizing: border-box; margin: 0; }
body { background: #0d0d0d; color: #fff; overflow: hidden; height: 100vh;
  font: 18px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.scene { position: absolute; inset: 0; display: flex; flex-direction: column;
  align-items: center; justify-content: center; text-align: center;
  padding: 8vh 10vw; opacity: 0; transform: translateY(14px) scale(.985);
  transition: opacity .55s ease, transform .55s ease; pointer-events: none; }
.scene.on { opacity: 1; transform: none; }
.icon { font-size: 88px; margin-bottom: 26px;
  animation: pop .6s cubic-bezier(.2,.9,.3,1.4) both; }
.scene.on .icon { animation-play-state: running; }
@keyframes pop { from { transform: scale(.4); opacity: 0; } to { transform: none; opacity: 1; } }
h1 { font-size: clamp(26px, 4.4vw, 44px); letter-spacing: -.02em; margin-bottom: 16px; }
.body { font-size: clamp(16px, 2.2vw, 22px); color: #c3c2b7; max-width: 760px; }
.sub { margin-top: 14px; color: #898781; font-size: 15px; }
.k-blocked h1 { color: #ff8f92; } .k-blocked .icon { filter: drop-shadow(0 0 34px rgba(229,72,77,.55)); }
.k-risky h1 { color: #fab219; }
.k-taint h1 { color: #ff8f92; }
.k-score h1 { color: #6da7ec; }
.stepchip { position: absolute; top: 26px; right: 30px; color: #898781;
  font-size: 13px; border: 1px solid #2c2c2a; border-radius: 99px; padding: 3px 12px; }
.bar { position: fixed; left: 0; top: 0; height: 3px; background: #3987e5;
  width: 0%; transition: width .2s linear; z-index: 5; }
.hint { position: fixed; bottom: 18px; width: 100%; text-align: center;
  color: #52514e; font-size: 12.5px; }
.dots { position: fixed; bottom: 44px; width: 100%; text-align: center; }
.dots span { display: inline-block; width: 7px; height: 7px; border-radius: 50%;
  background: #2c2c2a; margin: 0 4px; transition: background .3s; }
.dots span.on { background: #3987e5; }
"""

_JS = """
const scenes = Array.from(document.querySelectorAll('.scene'));
const dots = Array.from(document.querySelectorAll('.dots span'));
const bar = document.querySelector('.bar');
const HOLD = 2800;
let i = 0, timer = null, t0 = null, paused = false;
function show(n) {
  i = Math.max(0, Math.min(scenes.length - 1, n));
  scenes.forEach((s, k) => s.classList.toggle('on', k === i));
  dots.forEach((d, k) => d.classList.toggle('on', k <= i));
}
function tick(ts) {
  if (paused) { t0 = ts - (t0 ? 0 : 0); requestAnimationFrame(tick); return; }
  if (t0 === null) t0 = ts;
  const p = Math.min(1, (ts - t0) / HOLD);
  bar.style.width = (100 * (i + p) / scenes.length) + '%';
  if (p >= 1) { if (i < scenes.length - 1) { show(i + 1); t0 = ts; } }
  requestAnimationFrame(tick);
}
document.addEventListener('keydown', e => {
  if (e.key === ' ') { paused = !paused; e.preventDefault(); }
  if (e.key === 'ArrowRight') { show(i + 1); t0 = null; }
  if (e.key === 'ArrowLeft') { show(i - 1); t0 = null; }
});
document.addEventListener('click', () => { show(i + 1); t0 = null; });
show(0); requestAnimationFrame(tick);
"""


def movie_html(data: dict) -> str:
    """Render the incident movie for a trace dict."""
    scenes = _scenes(data)
    parts = []
    for s in scenes:
        chip = (f'<span class="stepchip">step {s["step"]}</span>'
                if s.get("step") is not None else "")
        parts.append(
            f'<section class="scene k-{s["kind"]}">{chip}'
            f'<div class="icon">{s["icon"]}</div>'
            f'<h1>{_html.escape(s["title"])}</h1>'
            + (f'<p class="body">{_html.escape(s["body"])}</p>' if s["body"] else "")
            + (f'<p class="sub">{_html.escape(s["sub"])}</p>' if s["sub"] else "")
            + "</section>")
    dots = "".join("<span></span>" for _ in scenes)
    title = (data.get("episodes") or [data.get("prompt", "loom run")])[0][:60]
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Loom movie — {_html.escape(title)}</title>
<style>{_CSS}</style></head><body>
<div class="bar"></div>
{"".join(parts)}
<div class="dots">{dots}</div>
<div class="hint">space pause · ←/→ step · click next</div>
<script>{_JS}</script>
</body></html>"""
