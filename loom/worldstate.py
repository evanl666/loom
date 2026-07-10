"""``loom snapshot``: capture and restore an agent's WORLD, not just its trace.

Replay rewinds what the agent *said*; a real debugger also rewinds what the
agent *did to the world*. A WorldSnapshot captures one or more external states
-- a workspace directory, a SQLite database -- into a store, and restores them
atomically so a fork can re-run turn N against the real world it saw then:

    loom snapshot capture snap/ --dir ./workspace --sqlite ./app.db
    #  ... let the agent run / experiment ...
    loom snapshot restore snap/           # world is back to the captured state

The store is a directory: a ``manifest.json`` plus the captured tar / db copies,
so capture and restore work across separate processes (and CI). Backends
(FileTreeSnapshot, SqliteSnapshot) implement a tiny capture/restore protocol;
new worlds (a browser profile, a CRM tenant) plug in the same way.
"""

from __future__ import annotations

import json
import os


class WorldSnapshot:
    """A persisted, multi-backend snapshot of an agent's external world."""

    def __init__(self, store_dir: str):
        self.store_dir = os.path.abspath(store_dir)
        os.makedirs(self.store_dir, exist_ok=True)
        self.entries: list[dict] = []   # {kind, target, store}
        self._backends: list = []

    # -- capture ------------------------------------------------------------
    def add_dir(self, directory: str) -> "WorldSnapshot":
        from .packs.snapshots import FileTreeSnapshot

        store = os.path.join(self.store_dir, f"dir_{len(self.entries)}.tar")
        FileTreeSnapshot(directory, tar_path=store)
        self.entries.append({"kind": "dir", "target": os.path.abspath(directory), "store": store})
        return self

    def add_sqlite(self, db_path: str) -> "WorldSnapshot":
        from .packs.snapshots import SqliteSnapshot

        store = os.path.join(self.store_dir, f"db_{len(self.entries)}.sqlite")
        SqliteSnapshot(db_path, store_path=store)
        self.entries.append({"kind": "sqlite", "target": os.path.abspath(db_path), "store": store})
        return self

    def save(self) -> str:
        path = os.path.join(self.store_dir, "manifest.json")
        with open(path, "w") as f:
            json.dump({"entries": self.entries}, f, indent=2)
        return path

    # -- restore ------------------------------------------------------------
    @classmethod
    def load(cls, store_dir: str) -> "WorldSnapshot":
        w = cls(store_dir)
        with open(os.path.join(w.store_dir, "manifest.json")) as f:
            w.entries = (json.load(f) or {}).get("entries", [])
        return w

    def restore_all(self) -> "list[dict]":
        """Restore every captured world. Returns per-entry {kind, target, ok}."""
        from .packs.snapshots import FileTreeSnapshot, SqliteSnapshot

        results = []
        for e in self.entries:
            ok = False
            try:
                if e["kind"] == "dir" and os.path.isdir(e["target"]):
                    b = FileTreeSnapshot.__new__(FileTreeSnapshot)
                    b.directory, b.tar_path = e["target"], e["store"]
                    ok = b.restore()
                elif e["kind"] == "sqlite":
                    import sqlite3
                    b = SqliteSnapshot.__new__(SqliteSnapshot)
                    b.db_path, b.store_path, b._sqlite3 = e["target"], e["store"], sqlite3
                    ok = b.restore()
            except Exception:  # noqa: BLE001 -- one bad entry shouldn't abort the rest
                ok = False
            results.append({"kind": e["kind"], "target": e["target"], "ok": ok})
        return results


class WorldRepo:
    """Git-branch-style management of an agent's world snapshots.

    ``branch`` captures the current world under a name, ``checkout`` restores a
    named branch, ``diff`` compares two branches. The repo is a ``.loom-world/``
    directory holding one snapshot store per branch -- so you can run an agent,
    branch its world, try a different path, and jump back, like source control
    for state instead of code.
    """

    def __init__(self, repo_dir: str = ".loom-world"):
        self.repo_dir = os.path.abspath(repo_dir)
        os.makedirs(self.repo_dir, exist_ok=True)

    def _branch_dir(self, name: str) -> str:
        safe = "".join(c for c in name if c.isalnum() or c in "-_.")
        if not safe:
            raise ValueError(f"invalid branch name {name!r}")
        return os.path.join(self.repo_dir, safe)

    def branch(self, name: str, dirs: "list[str]" = (), sqlite: "list[str]" = ()) -> str:
        if not dirs and not sqlite:
            raise ValueError("a branch needs at least one --dir or --sqlite to capture")
        w = WorldSnapshot(self._branch_dir(name))
        for d in dirs:
            w.add_dir(d)
        for db in sqlite:
            w.add_sqlite(db)
        w.save()
        return self._branch_dir(name)

    def list(self) -> "list[dict]":
        out = []
        for name in sorted(os.listdir(self.repo_dir)):
            m = os.path.join(self.repo_dir, name, "manifest.json")
            if os.path.isfile(m):
                try:
                    with open(m) as f:
                        entries = (json.load(f) or {}).get("entries", [])
                    out.append({"branch": name, "worlds": len(entries),
                                "targets": [e["target"] for e in entries]})
                except (OSError, json.JSONDecodeError):
                    continue
        return out

    def checkout(self, name: str) -> "list[dict]":
        d = self._branch_dir(name)
        if not os.path.isfile(os.path.join(d, "manifest.json")):
            raise ValueError(f"no branch named {name!r}")
        return WorldSnapshot.load(d).restore_all()

    def diff(self, a: str, b: str) -> dict:
        """Compare two branches: per world, what differs (files / table rows)."""
        ea = self._entries(a)
        eb = self._entries(b)
        worlds = []
        by_target_b = {e["target"]: e for e in eb}
        for e in ea:
            other = by_target_b.get(e["target"])
            if not other:
                continue
            if e["kind"] == "dir":
                worlds.append({"target": e["target"], "kind": "dir",
                               **_dir_diff(e["store"], other["store"])})
            elif e["kind"] == "sqlite":
                worlds.append({"target": e["target"], "kind": "sqlite",
                               **_sqlite_diff(e["store"], other["store"])})
        return {"a": a, "b": b, "worlds": worlds}

    def _entries(self, name: str) -> "list[dict]":
        with open(os.path.join(self._branch_dir(name), "manifest.json")) as f:
            return (json.load(f) or {}).get("entries", [])


def _dir_diff(tar_a: str, tar_b: str) -> dict:
    import tarfile
    def names(p):
        with tarfile.open(p) as t:
            return {m.name for m in t.getmembers() if m.isfile()}
    na, nb = names(tar_a), names(tar_b)
    return {"added": sorted(nb - na), "removed": sorted(na - nb)}


def _sqlite_diff(db_a: str, db_b: str) -> dict:
    import sqlite3
    def counts(p):
        c = sqlite3.connect(p)
        try:
            tables = [r[0] for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")]
            return {t: c.execute(f"SELECT count(*) FROM '{t}'").fetchone()[0] for t in tables}
        finally:
            c.close()
    ca, cb = counts(db_a), counts(db_b)
    changed = {t: [ca.get(t, 0), cb.get(t, 0)] for t in set(ca) | set(cb)
               if ca.get(t) != cb.get(t)}
    return {"tables_changed": changed}


def describe_world_diff(d: dict) -> str:
    lines = [f"world diff: {d['a']} → {d['b']}"]
    if not d["worlds"]:
        return lines[0] + "  (no shared worlds)"
    for w in d["worlds"]:
        if w["kind"] == "dir":
            lines.append(f"  📁 {w['target']}: +{len(w['added'])} / -{len(w['removed'])} file(s)")
            for f in w["added"][:5]:
                lines.append(f"      + {f}")
            for f in w["removed"][:5]:
                lines.append(f"      - {f}")
        else:
            ch = w["tables_changed"]
            lines.append(f"  🗄  {w['target']}: {len(ch)} table(s) changed")
            for t, (x, y) in list(ch.items())[:6]:
                lines.append(f"      {t}: {x} → {y} rows")
    return "\n".join(lines)


def describe_restore(results: "list[dict]") -> str:
    if not results:
        return "nothing to restore (empty snapshot)"
    lines = [f"restored {sum(1 for r in results if r['ok'])}/{len(results)} world(s):"]
    for r in results:
        lines.append(f"  {'✓' if r['ok'] else '✗'} {r['kind']:<8} {r['target']}")
    return "\n".join(lines)
