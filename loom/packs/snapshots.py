"""Executable world-state snapshot backends for packs.

``Pack.restore`` returns a *plan*; a backend actually captures and restores the
world so a fork can continue from step N against real state, not just replayed
text. The generic backend here snapshots a directory (a tar), which any pack
whose world is a filesystem can reuse; domains with their own state (a Postgres
test DB, a browser session, a CRM tenant) plug in their own backend by
implementing the same tiny protocol.

    snap = FileTreeSnapshot("./sandbox")     # capture
    snap.restore()                           # put it back before re-running
"""

from __future__ import annotations

import os
import tarfile
import tempfile


class SnapshotBackend:
    """The protocol: capture() before, restore() to rewind, drop() to clean up."""

    def restore(self) -> bool:
        raise NotImplementedError

    def drop(self) -> None:
        pass


class FileTreeSnapshot(SnapshotBackend):
    """A tar snapshot of a directory -- restore wipes it back to the captured tree.

    Executable and offline: usable by any pack whose external world is a
    filesystem (a scratch workspace, a fixture dir). Captures on construction.
    """

    def __init__(self, directory: str, tar_path: "str | None" = None):
        self.directory = os.path.abspath(directory)
        if not os.path.isdir(self.directory):
            raise ValueError(f"{directory} is not a directory")
        fd, self.tar_path = (None, tar_path) if tar_path else tempfile.mkstemp(suffix=".tar")
        if fd is not None:
            os.close(fd)
        with tarfile.open(self.tar_path, "w") as tar:
            tar.add(self.directory, arcname=".")

    def restore(self) -> bool:
        """Replace the directory's contents with the captured snapshot."""
        if not os.path.isfile(self.tar_path):
            return False
        # Remove current contents (not the dir itself), then extract.
        for entry in os.listdir(self.directory):
            p = os.path.join(self.directory, entry)
            if os.path.isdir(p) and not os.path.islink(p):
                import shutil

                shutil.rmtree(p, ignore_errors=True)
            else:
                try:
                    os.remove(p)
                except OSError:
                    pass
        with tarfile.open(self.tar_path, "r") as tar:
            _safe_extractall(tar, self.directory)
        return True

    def drop(self) -> None:
        try:
            os.remove(self.tar_path)
        except OSError:
            pass


class SqliteSnapshot(SnapshotBackend):
    """A snapshot of a SQLite database -- the world of a SQL agent.

    Uses the sqlite backup API (consistent even with an open/WAL database), so
    a fork can rewind a test DB to its state at turn N and re-run against real
    rows, not replayed text. Captures on construction.
    """

    def __init__(self, db_path: str, store_path: "str | None" = None):
        import sqlite3

        self.db_path = os.path.abspath(db_path)
        if not os.path.isfile(self.db_path):
            raise ValueError(f"{db_path} is not a SQLite file")
        fd, self.store_path = (None, store_path) if store_path else tempfile.mkstemp(suffix=".sqlite")
        if fd is not None:
            os.close(fd)
        self._sqlite3 = sqlite3
        self._copy(self.db_path, self.store_path)  # capture

    def _copy(self, src: str, dst: str) -> None:
        s = self._sqlite3.connect(src)
        d = self._sqlite3.connect(dst)
        try:
            with d:
                s.backup(d)
        finally:
            s.close()
            d.close()

    def restore(self) -> bool:
        """Overwrite the live DB with the captured snapshot."""
        if not os.path.isfile(self.store_path):
            return False
        self._copy(self.store_path, self.db_path)
        return True

    def drop(self) -> None:
        try:
            os.remove(self.store_path)
        except OSError:
            pass


def _safe_extractall(tar: "tarfile.TarFile", dest: str) -> None:
    """Extract, refusing any member that would escape ``dest`` (tar traversal)."""
    dest = os.path.realpath(dest)
    for member in tar.getmembers():
        target = os.path.realpath(os.path.join(dest, member.name))
        if target != dest and not target.startswith(dest + os.sep):
            raise ValueError(f"unsafe path in archive: {member.name}")
    # 'data' filter (py3.12+) also strips unsafe members; fall back on older.
    try:
        tar.extractall(dest, filter="data")
    except TypeError:  # pragma: no cover  (Python < 3.12)
        tar.extractall(dest)  # noqa: S202  (paths validated above)
