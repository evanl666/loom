"""Analyzer visual panels reused by the incident / autopsy HTML bundles.

These SVG fragments were once part of the old Studio renderer; that full-page
renderer is retired (``loom.export.trace_to_html`` delegates to the debugger UI
now), but the panels themselves -- an Impact Map (actions -> world) and a Data
Flow map (sensitive values source -> sink) -- are still useful, so they live
here as standalone builders. They emit HTML fragments that use CSS variables
(``var(--warn)`` etc.) supplied by the embedding page.
"""

from __future__ import annotations

import html

_WORLD = {  # state-diff kind (or "network") -> (emoji, label)
    "file": ("📄", "files"), "database": ("🗄", "database"),
    "dom": ("🌐", "browser"), "record": ("👤", "records"),
    "field": ("👤", "records"), "network": ("↗", "network"),
}


def _impact_map(data: dict) -> str:
    """A bipartite Impact Map: agent actions (left) → the parts of the outside
    world they touched (right), edges tinted by risk and dashed when blocked.
    Self-contained inline SVG, no JS."""
    from .action import actions as _actions

    edges = []  # (action_label, world_key, risk, blocked)
    for a in _actions(data):
        if a.type != "call" or a.step < 0:
            continue
        caps = set(a.capabilities)
        world = None
        if a.state_diff is not None:
            world = a.state_diff.kind if a.state_diff.kind in _WORLD else "file"
        elif "network" in caps:
            world = "network"
        if world is None:
            continue
        blocked = bool(a.policy and a.policy.blocked)
        edges.append((f"{a.step} {a.tool}", world, a.risk, blocked))
    if not edges:
        return ""

    left = list(dict.fromkeys(e[0] for e in edges))[:14]
    worlds = list(dict.fromkeys(e[1] for e in edges))
    row_h, pad_top = 34, 24
    height = max(len(left), len(worlds)) * row_h + pad_top + 20
    ly = {name: pad_top + i * row_h for i, name in enumerate(left)}
    wy = {name: pad_top + i * row_h for i, name in enumerate(worlds)}
    lx, wx, w = 20, 470, 700

    parts = [f'<svg viewBox="0 0 {w} {height}" width="100%" '
             f'style="max-width:{w}px" role="img" aria-label="side-effect map">']
    # edges first (under the nodes)
    for label, world, risk, blocked in edges:
        if label not in ly or world not in wy:
            continue
        y1, y2 = ly[label] + 11, wy[world] + 11
        cx = (lx + 190 + wx) / 2
        stroke = "var(--warn)" if risk else "var(--tool)"
        if blocked:
            stroke = "#e5484d"
        dash = ' stroke-dasharray="4 3"' if blocked else ""
        parts.append(
            f'<path d="M{lx + 190},{y1} C{cx},{y1} {cx},{y2} {wx},{y2}" '
            f'fill="none" stroke="{stroke}" stroke-width="1.6" opacity="0.7"{dash}/>')
        if risk:
            parts.append(
                f'<text x="{cx}" y="{(y1 + y2) / 2 - 3}" text-anchor="middle" '
                f'font-size="9.5" fill="var(--muted)">{html.escape(risk)}'
                + ("  🛡" if blocked else "") + "</text>")
    # left nodes: actions
    for label, y in ly.items():
        parts.append(
            f'<rect x="{lx}" y="{y}" width="190" height="22" rx="7" '
            f'fill="var(--surface)" stroke="var(--ring)"/>'
            f'<text x="{lx + 10}" y="{y + 15}" font-size="12" fill="var(--ink)">'
            f'{html.escape(label[:26])}</text>')
    # right nodes: world
    for world, y in wy.items():
        emoji, wlabel = _WORLD[world]
        parts.append(
            f'<rect x="{wx}" y="{y}" width="210" height="22" rx="7" '
            f'fill="color-mix(in srgb, var(--meta) 9%, var(--surface))" stroke="var(--ring)"/>'
            f'<text x="{wx + 10}" y="{y + 15}" font-size="12" fill="var(--ink)">'
            f'{emoji} {html.escape(wlabel)}</text>')
    parts.append("</svg>")
    return ('<div class="panel"><h2>Impact map — what it touched in the world</h2>'
            '<div style="padding:.6rem .95rem;overflow-x:auto">'
            + "".join(parts) + "</div></div>")


def _data_flow(data: dict) -> str:
    """The Data Flow panel: where sensitive VALUES came from and where they
    went. Sources (the reads that produced them) on the left, sinks (the
    egress actions that carried them) on the right, red edges labeled with a
    non-leaking value preview. Rendered only when taint paths exist."""
    from .taint import taint_paths

    paths = taint_paths(data)
    if not paths:
        return ""
    sources = list(dict.fromkeys(
        f"[{p['source']['step']}] {p['source']['tool']}" for p in paths))[:8]
    sinks = list(dict.fromkeys(
        f"[{p['sink']['step']}] {p['sink']['tool']} ({', '.join(p['sink']['via'])})"
        for p in paths))[:8]
    row_h, pad_top = 40, 26
    height = max(len(sources), len(sinks)) * row_h + pad_top + 22
    sy = {n: pad_top + i * row_h for i, n in enumerate(sources)}
    ky = {n: pad_top + i * row_h for i, n in enumerate(sinks)}
    lx, wx, w = 20, 452, 700

    parts = [f'<svg viewBox="0 0 {w} {height}" width="100%" '
             f'style="max-width:{w}px" role="img" aria-label="sensitive data flow">']
    for p in paths:
        src = f"[{p['source']['step']}] {p['source']['tool']}"
        snk = f"[{p['sink']['step']}] {p['sink']['tool']} ({', '.join(p['sink']['via'])})"
        if src not in sy or snk not in ky:
            continue
        y1, y2 = sy[src] + 12, ky[snk] + 12
        cx = (lx + 190 + wx) / 2
        parts.append(
            f'<path d="M{lx + 190},{y1} C{cx},{y1} {cx},{y2} {wx},{y2}" fill="none" '
            f'stroke="#e5484d" stroke-width="2" opacity="0.85"/>')
        sev = f" · {p['severity']}" if p.get("severity") else ""
        cls = p.get("sensitivity", p["kind"])
        parts.append(
            f'<text x="{cx}" y="{(y1 + y2) / 2 - 4}" text-anchor="middle" font-size="10" '
            f'fill="#e5484d">⛓ {html.escape(cls)}{html.escape(sev)} '
            f'{html.escape(p["value_preview"])}</text>')
    for label, y in sy.items():
        parts.append(
            f'<rect x="{lx}" y="{y}" width="190" height="24" rx="7" '
            f'fill="var(--surface)" stroke="var(--ring)"/>'
            f'<text x="{lx + 10}" y="{y + 16}" font-size="12" fill="var(--ink)">'
            f'🔑 {html.escape(label[:24])}</text>')
    for label, y in ky.items():
        parts.append(
            f'<rect x="{wx}" y="{y}" width="228" height="24" rx="7" '
            f'fill="color-mix(in srgb, #e5484d 10%, var(--surface))" stroke="var(--ring)"/>'
            f'<text x="{wx + 10}" y="{y + 16}" font-size="12" fill="var(--ink)">'
            f'↗ {html.escape(label[:30])}</text>')
    parts.append("</svg>")
    return ('<div class="panel"><h2>Data flow — sensitive values that left the box</h2>'
            '<div style="padding:.6rem .95rem;overflow-x:auto">' + "".join(parts)
            + '</div><p style="padding:0 .95rem .8rem;color:var(--muted);font-size:.78rem">'
              'verbatim value lineage (previews never show the value) — a paraphrased '
              'leak would not appear here</p></div>')
