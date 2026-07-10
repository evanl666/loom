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


def test_world_repo_branch_diff_checkout(tmp_path):
    import sqlite3

    from loom.worldstate import WorldRepo

    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "f.txt").write_text("v1")
    db = str(tmp_path / "db.sqlite")
    c = sqlite3.connect(db); c.execute("CREATE TABLE t(x int)"); c.execute("INSERT INTO t VALUES (1)")
    c.commit(); c.close()

    repo = WorldRepo(str(tmp_path / ".repo"))
    repo.branch("main", dirs=[str(ws)], sqlite=[db])

    # mutate the world, branch again
    (ws / "f.txt").write_text("v2"); (ws / "g.txt").write_text("new")
    c = sqlite3.connect(db); c.execute("INSERT INTO t VALUES (2), (3)"); c.commit(); c.close()
    repo.branch("experiment", dirs=[str(ws)], sqlite=[db])

    assert {b["branch"] for b in repo.list()} == {"main", "experiment"}
    d = repo.diff("main", "experiment")
    dir_w = next(w for w in d["worlds"] if w["kind"] == "dir")
    assert "./g.txt" in dir_w["added"]
    db_w = next(w for w in d["worlds"] if w["kind"] == "sqlite")
    assert db_w["tables_changed"]["t"] == [1, 3]

    # checkout main -> the whole world jumps back
    repo.checkout("main")
    assert (ws / "f.txt").read_text() == "v1" and not (ws / "g.txt").exists()
    assert sqlite3.connect(db).execute("SELECT count(*) FROM t").fetchone()[0] == 1
