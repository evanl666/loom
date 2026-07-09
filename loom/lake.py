"""A local trace lake: search every run you've ever recorded.

Point it at a directory (traces are just files -- commit them, rsync them,
drop them in a shared folder) and ask questions across runs:

    loom search runs/ 'cost>50000'                 # expensive runs
    loom search runs/ 'tool:Bash failed'           # failed runs that shelled out
    loom search runs/ 'shield:deny'                # runs where the firewall fired
    loom search runs/ 'database migration'         # free text over prompts/outputs
    loom lake runs/ -o dashboard.html              # cost dashboard for the corpus

The index is a SQLite file (``.loom-lake.db``) inside the directory, rebuilt
incrementally by mtime -- indexing is invisible, there is no service to run.
Stdlib only, like the rest of the kernel.
"""

from __future__ import annotations

import json
import os
import sqlite3
from glob import glob

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    path TEXT PRIMARY KEY,
    mtime REAL,
    model TEXT,
    recorded_via TEXT,
    stop_reason TEXT,
    episodes TEXT,
    output TEXT,
    num_effects INTEGER,
    input_tokens INTEGER,
    output_tokens INTEGER,
    tools TEXT,
    shield_denies INTEGER,
    healed_by TEXT
);
"""


def _summarize(path: str, data: dict) -> tuple:
    log = data.get("log") or []
    input_tokens = output_tokens = 0
    tools: list[str] = []
    for e in log:
        kind = e.get("kind", "")
        if kind == "model" and isinstance(e.get("result"), dict):
            usage = e["result"].get("usage") or {}
            input_tokens += usage.get("input_tokens", 0) or 0
            output_tokens += usage.get("output_tokens", 0) or 0
            for tc in e["result"].get("tool_calls") or []:
                name = tc.get("name")
                if name and name not in tools:
                    tools.append(name)
        elif kind.startswith("tool:"):
            name = kind[5:]
            if name not in tools:
                tools.append(name)
    denies = sum(
        1 for ev in (data.get("shield_events") or []) if ev.get("action") == "deny"
    )
    episodes = data.get("episodes") or [data.get("prompt", "")]
    return (
        path,
        os.path.getmtime(path),
        data.get("model", ""),
        data.get("recorded_via", "harness"),
        data.get("stop_reason", ""),
        " | ".join(str(e) for e in episodes),
        str(data.get("output", "")),
        len(log),
        input_tokens,
        output_tokens,
        " ".join(tools),
        denies,
        data.get("healed_by") or "",
    )


class Lake:
    """An incrementally indexed directory of traces."""

    def __init__(self, directory: str):
        self.directory = directory
        self.db_path = os.path.join(directory, ".loom-lake.db")
        self.db = sqlite3.connect(self.db_path)
        self.db.row_factory = sqlite3.Row
        self.db.execute(_SCHEMA)

    def index(self) -> "tuple[int, int]":
        """Bring the index up to date. Returns (indexed now, total runs)."""
        on_disk = {
            p: os.path.getmtime(p)
            for p in glob(os.path.join(self.directory, "**", "*.loom.json"), recursive=True)
        }
        known = dict(self.db.execute("SELECT path, mtime FROM runs"))
        for stale in known.keys() - on_disk.keys():
            self.db.execute("DELETE FROM runs WHERE path = ?", (stale,))
        fresh = [
            p for p, m in on_disk.items() if known.get(p) is None or m > known[p]
        ]
        for path in fresh:
            try:
                with open(path) as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            self.db.execute(
                "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                _summarize(path, data),
            )
        self.db.commit()
        total = self.db.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        return len(fresh), total

    def search(self, query: str) -> "list[sqlite3.Row]":
        """A tiny query language, safe to feed from a CLI.

        Space-separated terms, all must hold:
          ``cost>N`` / ``cost<N``   total tokens above/below N
          ``tool:NAME``             the run called this tool
          ``model:PATTERN``         model name GLOB
          ``failed``                stop_reason is not a clean end_turn
          ``shield:deny``           the firewall blocked something
          ``healed``                the run was repaired by heal()
          anything else             substring over prompts + output
        """
        where, args = [], []
        for term in query.split():
            if term.startswith("cost>") or term.startswith("cost<"):
                op = term[4]
                try:
                    n = int(term[5:])
                except ValueError:
                    raise ValueError(f"cost filter needs a number: {term!r}")
                where.append(f"(input_tokens + output_tokens) {op} ?")
                args.append(n)
            elif term.startswith("tool:"):
                where.append("(' ' || tools || ' ') LIKE ?")
                args.append(f"% {term[5:]} %")
            elif term.startswith("model:"):
                where.append("model GLOB ?")
                args.append(term[6:])
            elif term == "failed":
                where.append("stop_reason NOT IN ('end_turn', '')")
            elif term == "shield:deny":
                where.append("shield_denies > 0")
            elif term == "healed":
                where.append("healed_by != ''")
            else:
                where.append("(episodes LIKE ? OR output LIKE ?)")
                args.extend([f"%{term}%", f"%{term}%"])
        sql = "SELECT * FROM runs"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY (input_tokens + output_tokens) DESC"
        return self.db.execute(sql, args).fetchall()

    def stats(self) -> dict:
        """Corpus-level numbers for the dashboard."""
        row = self.db.execute(
            "SELECT COUNT(*) AS runs, COALESCE(SUM(input_tokens),0) AS input_tokens, "
            "COALESCE(SUM(output_tokens),0) AS output_tokens, "
            "COALESCE(SUM(shield_denies),0) AS denies, "
            "SUM(CASE WHEN stop_reason NOT IN ('end_turn','') THEN 1 ELSE 0 END) AS failed "
            "FROM runs"
        ).fetchone()
        tools: dict[str, int] = {}
        for (joined,) in self.db.execute("SELECT tools FROM runs"):
            for name in (joined or "").split():
                tools[name] = tools.get(name, 0) + 1
        by_cost = self.db.execute(
            "SELECT path, model, episodes, stop_reason, "
            "(input_tokens + output_tokens) AS tokens, shield_denies "
            "FROM runs ORDER BY tokens DESC"
        ).fetchall()
        return {
            "runs": row["runs"],
            "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"],
            "denies": row["denies"],
            "failed": row["failed"] or 0,
            "top_tools": sorted(tools.items(), key=lambda kv: -kv[1])[:12],
            "by_cost": [dict(r) for r in by_cost],
        }

    def close(self) -> None:
        self.db.close()


# ------------------------------------------------------------- the dashboard

_DASH_CSS = """
:root {
  --surface: #fcfcfb; --page: #f9f9f7; --ink: #0b0b0b; --ink-2: #52514e;
  --muted: #898781; --grid: #e1e0d9;
  --tool: #1baf7a; --warn: #b3261e;
  --c2:#9ec5f4; --c3:#6da7ec; --c4:#3987e5; --c5:#256abf; --c6:#184f95; --c7:#0d366b;
  --shadow: 0 1px 2px rgba(11,11,11,.05), 0 10px 30px -12px rgba(11,11,11,.12);
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface: #1a1a19; --page: #0d0d0d; --ink: #ffffff; --ink-2: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a;
    --tool: #199e70; --warn: #ff8a80;
    --c2:#104281; --c3:#184f95; --c4:#1c5cab; --c5:#256abf; --c6:#3987e5; --c7:#6da7ec;
    --shadow: 0 1px 2px rgba(0,0,0,.4), 0 10px 30px -12px rgba(0,0,0,.6);
  }
}
* { box-sizing: border-box; margin: 0; }
body { background: var(--page); color: var(--ink);
  font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; padding: 32px; }
h1 { font-size: 20px; margin-bottom: 4px; }
.sub { color: var(--muted); margin-bottom: 24px; }
.tiles { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 28px; }
.tile { background: var(--surface); border: 1px solid var(--grid); border-radius: 10px;
  padding: 14px 18px; min-width: 150px; box-shadow: var(--shadow); }
.tile .v { font-size: 30px; font-weight: 650; letter-spacing: -0.02em; }
.tile .k { color: var(--muted); font-size: 12px; text-transform: uppercase;
  letter-spacing: 0.06em; }
.tile .v.warn { color: var(--warn); }
h2 { font-size: 14px; color: var(--ink-2); margin: 26px 0 10px; }
.rows { background: var(--surface); border: 1px solid var(--grid); border-radius: 10px;
  padding: 10px 14px; box-shadow: var(--shadow); }
.row { display: grid; grid-template-columns: minmax(140px, 34%) 1fr 90px;
  gap: 12px; align-items: center; padding: 5px 4px; border-radius: 6px; }
.row:hover { background: var(--page); }
.row .name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  color: var(--ink-2); }
.row .name small { color: var(--muted); }
.bar { height: 10px; border-radius: 0 4px 4px 0; min-width: 2px; }
.row .val { text-align: right; font-variant-numeric: tabular-nums; color: var(--ink); }
.badge { display: inline-block; font-size: 11px; color: var(--warn);
  border: 1px solid currentColor; border-radius: 99px; padding: 0 6px; margin-left: 6px; }
.empty { color: var(--muted); padding: 8px 4px; }
footer { color: var(--muted); font-size: 12px; margin-top: 28px; }
"""

_RAMP = ("--c2", "--c3", "--c4", "--c5", "--c6", "--c7")


def _esc(s: str) -> str:
    return (
        str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _bar_rows(items: "list[tuple[str, str, int, str]]", color=None) -> str:
    """Rows of (label, sublabel, value, extra-html) as labeled horizontal bars."""
    if not items:
        return '<div class="empty">nothing here yet</div>'
    top = max(v for _, _, v, _ in items) or 1
    rows = []
    for label, sub, value, extra in items:
        share = value / top
        # Sequential ramp: longer bar, darker step -- same encoding twice, on purpose.
        fill = color or f"var({_RAMP[min(int(share * len(_RAMP)), len(_RAMP) - 1)]})"
        sub_html = f" <small>{_esc(sub)}</small>" if sub else ""
        rows.append(
            f'<div class="row" title="{_esc(label)}: {value:,}">'
            f'<span class="name">{_esc(label)}{sub_html}</span>'
            f'<span class="bar" style="width:{max(share * 100, 0.5):.1f}%;'
            f'background:{fill}"></span>'
            f'<span class="val">{value:,}{extra}</span></div>'
        )
    return "\n".join(rows)


def dashboard_html(stats: dict, directory: str) -> str:
    """A self-contained cost dashboard for an indexed trace corpus."""
    runs = stats["by_cost"]
    run_items = []
    for r in runs[:50]:
        label = os.path.basename(r["path"])
        prompt = (r["episodes"] or "").split(" | ")[0][:60]
        badge = '<span class="badge">shield</span>' if r["shield_denies"] else ""
        run_items.append((label, prompt, r["tokens"], badge))
    tool_items = [(name, "", count, "") for name, count in stats["top_tools"]]
    failed = stats["failed"]
    tiles = f"""
<div class="tiles">
  <div class="tile"><div class="v">{stats["runs"]:,}</div><div class="k">runs</div></div>
  <div class="tile"><div class="v">{stats["input_tokens"] + stats["output_tokens"]:,}</div>
    <div class="k">total tokens</div></div>
  <div class="tile"><div class="v{' warn' if failed else ''}">{failed:,}</div>
    <div class="k">failed runs</div></div>
  <div class="tile"><div class="v{' warn' if stats["denies"] else ''}">{stats["denies"]:,}</div>
    <div class="k">shield blocks</div></div>
</div>"""
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>loom lake -- {_esc(directory)}</title>
<style>{_DASH_CSS}</style></head><body>
<h1>Trace lake</h1>
<div class="sub">{_esc(directory)} &middot; generated by <code>loom lake</code></div>
{tiles}
<h2>Runs by token cost{" (top 50)" if len(runs) > 50 else ""}</h2>
<div class="rows">{_bar_rows(run_items)}</div>
<h2>Tools by runs that used them</h2>
<div class="rows">{_bar_rows(tool_items, color="var(--tool)")}</div>
<footer>search this corpus: <code>loom search {_esc(directory)} 'cost&gt;50000'</code>
&middot; <code>'tool:Bash failed'</code> &middot; <code>'shield:deny'</code></footer>
</body></html>"""
