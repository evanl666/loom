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
        if url.path == "/run":
            full = self._trace_path((qs.get("p") or [""])[0])
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

            self._send(200, trace_to_html(data, path=os.path.basename(full)))
            return
        self._send(404, "<h1>404</h1>")


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
