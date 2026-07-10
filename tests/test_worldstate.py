"""World-state time travel: capture + restore a workspace dir and a SQLite DB."""

import sqlite3

from loom.worldstate import WorldSnapshot


def test_snapshot_restores_dir_and_sqlite(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "data.txt").write_text("original")
    db = str(tmp_path / "app.db")
    c = sqlite3.connect(db)
    c.execute("CREATE TABLE users(id int, name text)")
    c.executemany("INSERT INTO users VALUES (?, ?)", [(1, "alice"), (2, "bob")])
    c.commit(); c.close()

    # capture
    store = str(tmp_path / "store")
    snap = WorldSnapshot(store).add_dir(str(ws)).add_sqlite(db)
    snap.save()

    # the agent destroys the world
    (ws / "data.txt").unlink()
    (ws / "evil.txt").write_text("junk")
    c = sqlite3.connect(db); c.execute("DELETE FROM users"); c.commit(); c.close()

    # restore in a FRESH WorldSnapshot loaded from the store (cross-process shape)
    results = WorldSnapshot.load(store).restore_all()
    assert all(r["ok"] for r in results) and len(results) == 2

    assert (ws / "data.txt").read_text() == "original"
    assert not (ws / "evil.txt").exists()  # the agent's junk is gone
    rows = sqlite3.connect(db).execute("SELECT name FROM users ORDER BY id").fetchall()
    assert [r[0] for r in rows] == ["alice", "bob"]
