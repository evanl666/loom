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


def describe_restore(results: "list[dict]") -> str:
    if not results:
        return "nothing to restore (empty snapshot)"
    lines = [f"restored {sum(1 for r in results if r['ok'])}/{len(results)} world(s):"]
    for r in results:
        lines.append(f"  {'✓' if r['ok'] else '✗'} {r['kind']:<8} {r['target']}")
    return "\n".join(lines)
