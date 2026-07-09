"""``loom serve``: the team trace server, one process, zero dependencies.

Point it at the directory where your runs land and every teammate gets a
browser view of the corpus -- searchable run list, per-run Studio (the Action
Debugger), and the incident report -- without anyone installing anything:

    loom serve runs/                 # http://127.0.0.1:8790
    loom serve runs/ --host 0.0.0.0  # share on the LAN (no auth -- trusted nets only)

Built on the same pieces as the CLI: the Lake index (incremental, mtime-keyed)
for list/search, ``trace_to_html`` for the per-run debugger, and
``build_report`` for incidents. Single-tenant by design: RBAC/SSO/retention
belong to a real deployment, not a library. The index refreshes on every list
request, so new runs appear as they land.
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import __version__
from .lake import Lake, _esc

_CSS = """
:root { --surface:#fcfcfb; --page:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e;
  --muted:#898781; --grid:#e1e0d9; --model:#2a78d6; --warn:#b3261e;
  --shadow: 0 1px 2px rgba(11,11,11,.05), 0 10px 30px -12px rgba(11,11,11,.12); }
@media (prefers-color-scheme: dark) {
  :root { --surface:#1a1a19; --page:#0d0d0d; --ink:#fff; --ink2:#c3c2b7;
    --muted:#898781; --grid:#2c2c2a; --model:#3987e5; --warn:#ff8a80;
    --shadow: 0 1px 2px rgba(0,0,0,.4), 0 10px 30px -12px rgba(0,0,0,.6); } }
* { box-sizing:border-box; margin:0; }
body { background:var(--page); color:var(--ink); padding:32px; max-width:1080px;
  margin:0 auto; font:14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
h1 { font-size:20px; } .sub { color:var(--muted); margin-bottom:20px; }
form { margin: 14px 0 18px; }
input[type=search] { font:inherit; padding:8px 14px; border-radius:99px;
  border:1px solid var(--grid); background:var(--surface); color:var(--ink);
  width: min(480px, 100%); }
.tiles { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:20px; }
.tile { background:var(--surface); border:1px solid var(--grid); border-radius:10px;
  padding:12px 16px; min-width:130px; box-shadow:var(--shadow); }
.tile .v { font-size:26px; font-weight:650; } .tile .v.warn { color:var(--warn); }
.tile .k { color:var(--muted); font-size:11px; text-transform:uppercase;
  letter-spacing:.06em; }
table { width:100%; border-collapse:collapse; background:var(--surface);
  border:1px solid var(--grid); border-radius:10px; overflow:hidden;
  box-shadow:var(--shadow); }
th { text-align:left; font-size:11px; color:var(--muted); text-transform:uppercase;
  letter-spacing:.06em; padding:10px 12px; border-bottom:1px solid var(--grid); }
td { padding:8px 12px; border-bottom:1px solid var(--grid); vertical-align:top; }
tr:last-child td { border-bottom:none; }
td.num { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
a { color:var(--model); text-decoration:none; } a:hover { text-decoration:underline; }
.prompt { color:var(--ink2); overflow:hidden; text-overflow:ellipsis;
  display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; }
.chip { display:inline-block; font-size:11px; border:1px solid currentColor;
  border-radius:99px; padding:0 7px; margin-right:4px; white-space:nowrap; }
.chip.deny, .chip.risk { color:var(--warn); } .chip.ok { color:var(--muted); }
footer { color:var(--muted); font-size:12px; margin-top:22px; }
.empty { color:var(--muted); padding:18px; }
"""


def _row_html(r: dict, root: str) -> str:
    rel = os.path.relpath(r["path"], root)
    tokens = (r["input_tokens"] or 0) + (r["output_tokens"] or 0)
    chips = []
    room = _read_json(r["path"] + ".room.json", {})
    if room.get("resolved"):
        chips.append('<span class="chip" style="color:var(--model)">✓ resolved</span>')
    if room.get("owner"):
        chips.append(f'<span class="chip ok">👤 {_esc(room["owner"])}</span>')
    if r["shield_denies"]:
        chips.append(f'<span class="chip deny">🛡 {r["shield_denies"]} denied</span>')
    if r["risk"]:
        chips.append(f'<span class="chip risk">⚠ {_esc(r["risk"])}</span>')
    if r["stop_reason"] not in ("end_turn", ""):
        chips.append(f'<span class="chip ok">{_esc(r["stop_reason"])}</span>')
    prompt = (r["episodes"] or "").split(" | ")[0]
    return (
        f"<tr><td><a href=\"/run?p={_esc(rel)}\">{_esc(os.path.basename(r['path']))}</a>"
        f"<br><small>{_esc(r['model'])}</small></td>"
        f'<td><div class="prompt">{_esc(prompt[:160])}</div>{"".join(chips)}</td>'
        f'<td class="num">{r["num_effects"]}</td>'
        f'<td class="num">{tokens:,}</td>'
        f'<td class="num"><a href="/run?p={_esc(rel)}&view=incident">incident</a></td></tr>'
    )


def index_html(lake: Lake, root: str, query: str = "") -> str:
    stats = lake.stats()
    rows = lake.search(query) if query else lake.db.execute(
        "SELECT * FROM runs ORDER BY mtime DESC LIMIT 200").fetchall()
    body = "".join(_row_html(dict(r), root) for r in rows) or \
        '<tr><td colspan="5" class="empty">no runs match</td></tr>'
    warn = ' class="v warn"' if stats["denies"] else ' class="v"'
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Loom — {_esc(os.path.basename(os.path.abspath(root)))}</title>
<style>{_CSS}</style></head><body>
<h1>Loom trace server</h1>
<p class="sub">{_esc(os.path.abspath(root))} — {stats["runs"]} run(s) indexed</p>
<div class="tiles">
  <div class="tile"><div class="v">{stats["runs"]}</div><div class="k">runs</div></div>
  <div class="tile"><div class="v">{stats["input_tokens"] + stats["output_tokens"]:,}</div><div class="k">tokens</div></div>
  <div class="tile"><div{warn}>{stats["denies"]}</div><div class="k">firewall denies</div></div>
  <div class="tile"><div class="v">{stats["failed"]}</div><div class="k">failed runs</div></div>
  <div class="tile"><div class="v">{stats["risky"]}</div><div class="k">risky runs</div></div>
</div>
<form method="get" action="/">
  <input type="search" name="q" value="{_esc(query)}"
   placeholder="search: risk:secret-read shield:deny cost&gt;50000 tool:Bash failed …">
</form>
<table><tr><th>run</th><th>prompt</th><th>steps</th><th>tokens</th><th></th></tr>
{body}</table>
<footer>loom {__version__} — search syntax: free text, tool:NAME, model:GLOB,
risk:CATEGORY, risky, failed, shield:deny, healed, cost&gt;N</footer>
</body></html>"""


def _read_json(path: str, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: str, data) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


_ROOM_CSS = """
body { padding: 0; max-width: none; }
.roombar { display:flex; gap:14px; align-items:center; flex-wrap:wrap;
  padding:14px 22px; border-bottom:1px solid var(--grid); background:var(--surface); }
.roombar h1 { font-size:16px; margin:0; }
.roombar .chip { font-size:11px; }
.roombar .chip.resolved { color:var(--model); }
.roombar input[type=text] { font:inherit; font-size:13px; padding:4px 10px;
  border-radius:8px; border:1px solid var(--grid); background:var(--page);
  color:var(--ink); width:150px; }
.roombar label { font-size:12px; color:var(--muted); display:flex; gap:6px;
  align-items:center; }
.roombar button, .roombar a.btn { font:inherit; font-size:12px; padding:5px 12px;
  border-radius:99px; border:1px solid var(--grid); background:var(--page);
  color:var(--ink); cursor:pointer; text-decoration:none; }
.roombar button:hover, .roombar a.btn:hover { border-color:var(--model);
  color:var(--model); }
.roomcols { display:grid; grid-template-columns: minmax(0,1fr) 340px; height: calc(100vh - 61px); }
@media (max-width: 900px) { .roomcols { grid-template-columns:1fr; height:auto; } }
.roomcols iframe { width:100%; height:100%; border:none; min-height:70vh; }
.comments { border-left:1px solid var(--grid); background:var(--surface);
  overflow-y:auto; padding:16px; }
.comments h2 { font-size:12px; color:var(--muted); text-transform:uppercase;
  letter-spacing:.06em; margin-bottom:10px; }
.note { border:1px solid var(--grid); border-radius:10px; padding:8px 12px;
  margin-bottom:8px; font-size:13px; }
.note .who { color:var(--muted); font-size:11px; margin-top:4px; }
.note .step { color:var(--model); font-variant-numeric:tabular-nums;
  font-size:11px; }
.comments form { margin-top:12px; display:flex; flex-direction:column; gap:6px; }
.comments input, .comments textarea { font:inherit; font-size:13px; padding:6px 10px;
  border-radius:8px; border:1px solid var(--grid); background:var(--page);
  color:var(--ink); }
.comments textarea { min-height:60px; resize:vertical; }
"""


def room_html(rel: str, full: str, data: dict, base_url: str) -> str:
    """The replay room: triage state + step comments around the embedded Studio."""
    room = _read_json(full + ".room.json", {})
    notes = _read_json(full + ".notes.json", [])
    name = os.path.basename(full)
    permalink = f"{base_url}/run?p={rel}"
    prompt = (data.get("episodes") or [data.get("prompt", "")])[0]

    note_html = "".join(
        f'<div class="note"><span class="step">[{_esc(n.get("step", "?"))}]</span> '
        f'{_esc(n.get("text", ""))}'
        f'<div class="who">{_esc(n.get("by") or "anonymous")} · {_esc(n.get("ts", ""))}</div></div>'
        for n in sorted(notes, key=lambda x: (x.get("step") or 0))
    ) or '<div class="empty">no comments yet — be the first to mark a step</div>'

    resolved = bool(room.get("resolved"))
    status_chip = ('<span class="chip resolved">✓ resolved</span>' if resolved
                   else '<span class="chip risk">● open</span>')
    issue_body = (f"Loom replay room: {permalink}%0A%0ARun: {name}%0APrompt: "
                  f"{_esc(prompt[:120])}")
    issue_url = f"https://github.com/issues/new?title=Agent%20run%20incident:%20{name}&body={issue_body}"

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Room — {_esc(name)}</title>
<style>{_CSS}{_ROOM_CSS}</style></head><body>
<div class="roombar">
  <a href="/" class="btn">← runs</a>
  <h1>{_esc(name)}</h1>
  {status_chip}
  <label>owner <input type="text" id="owner" value="{_esc(room.get("owner", ""))}"
    placeholder="who's on it?"></label>
  <label>root cause <input type="text" id="root_cause"
    value="{_esc(room.get("root_cause", ""))}" placeholder="label it"></label>
  <label><input type="checkbox" id="resolved" {"checked" if resolved else ""}> resolved</label>
  <button onclick="saveRoom()">save</button>
  <button onclick="navigator.clipboard.writeText('{_esc(permalink)}');this.textContent='copied!'">
    copy permalink</button>
  <a class="btn" href="{issue_url}" target="_blank">open GitHub issue</a>
  <a class="btn" href="/run?p={_esc(rel)}&view=incident" target="_blank">incident report</a>
</div>
<div class="roomcols">
  <iframe src="/run/studio?p={_esc(rel)}" title="Action Debugger"></iframe>
  <div class="comments">
    <h2>Step comments ({len(notes)})</h2>
    {note_html}
    <form onsubmit="addNote(event)">
      <input type="number" id="n-step" placeholder="step #" min="0">
      <textarea id="n-text" placeholder="what happened at this step?" required></textarea>
      <input type="text" id="n-by" placeholder="your name">
      <button type="submit">comment</button>
    </form>
  </div>
</div>
<script>
const P = {json.dumps(rel)};
async function post(path, body) {{
  const r = await fetch(path + '?p=' + encodeURIComponent(P), {{
    method: 'POST', headers: {{'content-type': 'application/json'}},
    body: JSON.stringify(body)}});
  if (r.ok) location.reload(); else alert('failed: ' + (await r.text()));
}}
function saveRoom() {{
  post('/api/room', {{
    owner: document.getElementById('owner').value,
    root_cause: document.getElementById('root_cause').value,
    resolved: document.getElementById('resolved').checked }});
}}
function addNote(ev) {{
  ev.preventDefault();
  const step = document.getElementById('n-step').value;
  post('/api/notes', {{
    step: step === '' ? null : +step,
    text: document.getElementById('n-text').value,
    by: document.getElementById('n-by').value }});
}}
</script>
</body></html>"""


class _Handler(BaseHTTPRequestHandler):
    server_version = f"loom-serve/{__version__}"

    def log_message(self, fmt, *args):  # quiet by default
        pass

    def _send(self, code: int, body: str, ctype: str = "text/html; charset=utf-8"):
        raw = body.encode()
        self.send_response(code)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _trace_path(self, rel: str) -> "str | None":
        """Resolve a ?p= parameter safely inside the served directory."""
        root = os.path.realpath(self.server.root)  # type: ignore[attr-defined]
        full = os.path.realpath(os.path.join(root, rel))
        if full != root and not full.startswith(root + os.sep):
            return None  # traversal attempt
        return full if os.path.isfile(full) else None

    def do_GET(self):  # noqa: N802 (stdlib naming)
        url = urlparse(self.path)
        qs = parse_qs(url.query)
        root: str = self.server.root  # type: ignore[attr-defined]

        if url.path == "/":
            # SQLite connections are thread-bound and every request runs on
            # its own thread: open a per-request Lake (cheap -- the index is
            # a file) rather than sharing one across threads.
            lake = Lake(root)
            try:
                lake.index()  # refresh: new runs appear as they land
                self._send(200, index_html(lake, root, query=(qs.get("q") or [""])[0]))
            finally:
                lake.close()
            return
        if url.path == "/api/runs":
            lake = Lake(root)
            try:
                lake.index()
                q = (qs.get("q") or [""])[0]
                rows = lake.search(q) if q else lake.db.execute(
                    "SELECT * FROM runs ORDER BY mtime DESC LIMIT 200").fetchall()
                self._send(200, json.dumps([dict(r) for r in rows], indent=2),
                           "application/json")
            finally:
                lake.close()
            return
        if url.path in ("/run", "/run/studio"):
            rel = (qs.get("p") or [""])[0]
            full = self._trace_path(rel)
            if full is None:
                self._send(404, "<h1>404</h1><p>no such trace</p>")
                return
            try:
                with open(full) as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                self._send(500, "<h1>unreadable trace</h1>")
                return
            if (qs.get("view") or [""])[0] == "incident":
                from .incident import build_report

                report = build_report(data, os.path.basename(full))
                self._send(200, report, "text/plain; charset=utf-8")
                return
            from .export import trace_to_html

            if url.path == "/run/studio":  # the raw debugger, embedded by the room
                self._send(200, trace_to_html(data, path=os.path.basename(full)))
                return
            self._send(200, room_html(rel, full, data, self._server_url()))
            return
        self._send(404, "<h1>404</h1>")

    def _server_url(self) -> str:
        host, port = self.server.server_address[0], self.server.server_address[1]  # type: ignore[attr-defined]
        return f"http://{host}:{port}"

    def do_POST(self):  # noqa: N802 (stdlib naming)
        url = urlparse(self.path)
        qs = parse_qs(url.query)
        full = self._trace_path((qs.get("p") or [""])[0])
        if full is None or url.path not in ("/api/notes", "/api/room"):
            self._send(404, json.dumps({"error": "not found"}), "application/json")
            return
        try:
            length = int(self.headers.get("content-length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            assert isinstance(body, dict)
        except (ValueError, AssertionError):
            self._send(400, json.dumps({"error": "bad json body"}), "application/json")
            return

        if url.path == "/api/notes":
            # Same sidecar format as `loom note`, so CLI and room interoperate.
            text = str(body.get("text") or "").strip()
            if not text:
                self._send(400, json.dumps({"error": "text required"}), "application/json")
                return
            import time

            note = {"step": body.get("step"), "by": str(body.get("by") or ""),
                    "text": text[:2000], "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
            notes = _read_json(full + ".notes.json", [])
            notes.append(note)
            _write_json(full + ".notes.json", notes)
            self._send(200, json.dumps({"ok": True, "notes": len(notes)}), "application/json")
            return

        # /api/room: root-cause label, owner, resolved -- the triage state.
        room = _read_json(full + ".room.json", {})
        for key in ("root_cause", "owner"):
            if key in body:
                room[key] = str(body[key])[:200]
        if "resolved" in body:
            room["resolved"] = bool(body["resolved"])
        _write_json(full + ".room.json", room)
        self._send(200, json.dumps({"ok": True, **room}), "application/json")


class TraceServer:
    """The single-tenant team trace server. ``serve_forever`` or use as a handle."""

    def __init__(self, directory: str, host: str = "127.0.0.1", port: int = 8790):
        self.directory = directory
        # Build the index once up front (fast first paint); handlers open
        # their own per-request Lake because SQLite is thread-bound.
        boot = Lake(directory)
        boot.index()
        boot.close()
        self.httpd = ThreadingHTTPServer((host, port), _Handler)
        self.httpd.root = directory  # type: ignore[attr-defined]
        self.host, self.port = self.httpd.server_address[0], self.httpd.server_address[1]

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def serve_forever(self) -> None:
        self.httpd.serve_forever()

    def shutdown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
