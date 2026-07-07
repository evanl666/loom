"""Export a saved trace to a single self-contained HTML page.

No external assets, no JavaScript dependencies -- the file can be attached to a
bug report, emailed, or committed next to the trace itself.

    loom export run.loom.json            # writes run.loom.html
    loom export run.loom.json -o out.html
"""

from __future__ import annotations

import html
import json

from .providers.base import ModelResponse

_CSS = """
:root {
  --bg: #ffffff; --fg: #1c1c1e; --muted: #6e6e73; --line: #e5e5ea;
  --card: #f6f6f7; --model: #5856d6; --tool: #0a7ea4; --human: #b25000;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #131316; --fg: #ececf1; --muted: #98989f; --line: #2c2c31;
    --card: #1d1d21; --model: #9d9bff; --tool: #57c4e5; --human: #ffb266;
  }
}
* { box-sizing: border-box; margin: 0; }
body {
  background: var(--bg); color: var(--fg);
  font: 15px/1.55 -apple-system, "Segoe UI", Roboto, sans-serif;
  max-width: 860px; margin: 0 auto; padding: 2.5rem 1.25rem 4rem;
}
h1 { font-size: 1.35rem; margin-bottom: .25rem; }
.sub { color: var(--muted); font-size: .9rem; margin-bottom: 1.5rem; }
.meta { display: flex; flex-wrap: wrap; gap: 1.5rem; padding: 1rem 1.25rem;
  background: var(--card); border-radius: 10px; margin-bottom: 1.75rem; }
.meta div { min-width: 7rem; }
.meta .k { color: var(--muted); font-size: .78rem; text-transform: uppercase;
  letter-spacing: .04em; }
.meta .v { font-variant-numeric: tabular-nums; font-weight: 600; }
.episode { color: var(--muted); font-size: .9rem; margin: 1.25rem 0 .5rem; }
.step { display: flex; gap: .75rem; padding: .6rem .75rem; border-top: 1px solid var(--line);
  align-items: baseline; }
.step:first-of-type { border-top: none; }
.seq { color: var(--muted); font-size: .8rem; min-width: 2.2rem;
  font-variant-numeric: tabular-nums; }
.badge { font-size: .74rem; font-weight: 600; padding: .1rem .5rem;
  border-radius: 99px; white-space: nowrap; }
.badge.model { color: var(--model); border: 1px solid var(--model); }
.badge.tool  { color: var(--tool);  border: 1px solid var(--tool); }
.badge.human { color: var(--human); border: 1px solid var(--human); }
.detail { overflow-wrap: anywhere; white-space: pre-wrap; }
.detail .calls { color: var(--muted); }
.depth1 { margin-left: 2rem; } .depth2 { margin-left: 4rem; }
.depth3 { margin-left: 6rem; }
.paused { color: var(--human); font-weight: 600; margin: 1rem 0; }
footer { color: var(--muted); font-size: .8rem; margin-top: 2.5rem; }
"""


def _badge_class(kind: str) -> str:
    if kind == "model":
        return "model"
    if kind == "human":
        return "human"
    return "tool"


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


def trace_to_html(data: dict) -> str:
    """Render a saved trace dict (``Run.to_dict`` / a ``.loom.json`` file) to HTML."""
    log = data.get("log", [])
    episodes = data.get("episodes") or [data.get("prompt", "")]

    inp = out = 0
    for e in log:
        if e.get("kind") == "model":
            u = e.get("result", {}).get("usage", {})
            inp += u.get("input_tokens", 0)
            out += u.get("output_tokens", 0)

    rows: list[str] = []
    for e in log:
        kind = e.get("kind", "?")
        depth = min(e.get("depth", 0), 3)
        badge = _badge_class(kind)
        rows.append(
            f'<div class="step depth{depth}">'
            f'<span class="seq">{e.get("seq", "")}</span>'
            f'<span class="badge {badge}">{html.escape(kind)}</span>'
            f'<span class="detail">{_step_detail(kind, e.get("result"))}</span>'
            f"</div>"
        )

    episode_html = "".join(
        f'<p class="episode">user #{i + 1}: {html.escape(ep)}</p>' for i, ep in enumerate(episodes)
    )
    paused_html = (
        f'<p class="paused">⏸ paused — waiting for a human answer to: '
        f"{html.escape(str(data.get('pending')))}</p>"
        if data.get("paused")
        else ""
    )
    title = html.escape(episodes[0][:80]) or "Loom trace"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Loom trace — {title}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Loom trace</h1>
<p class="sub">{title}</p>
<div class="meta">
  <div><div class="k">model</div><div class="v">{html.escape(str(data.get("model", "?")))}</div></div>
  <div><div class="k">steps</div><div class="v">{len(log)}</div></div>
  <div><div class="k">input tokens</div><div class="v">{inp}</div></div>
  <div><div class="k">output tokens</div><div class="v">{out}</div></div>
</div>
{episode_html}
{paused_html}
<section>
{"".join(rows)}
</section>
<footer>output: {html.escape(str(data.get("output", "")))}<br>
generated by loom — the agent harness you can read, replay, and rewind</footer>
</body>
</html>
"""
