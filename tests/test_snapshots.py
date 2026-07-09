"""Executable world-state snapshot backends."""

import os

import pytest

from loom.packs.snapshots import FileTreeSnapshot


def test_filetree_snapshot_restores_the_directory(tmp_path):
    d = tmp_path / "world"
    d.mkdir()
    (d / "a.txt").write_text("original")
    (d / "keep").mkdir()
    (d / "keep" / "n.txt").write_text("nested")

    snap = FileTreeSnapshot(str(d))

    # mutate the world: change a file, add one, delete a dir
    (d / "a.txt").write_text("CHANGED")
    (d / "new.txt").write_text("added by the agent")
    import shutil
    shutil.rmtree(d / "keep")

    assert snap.restore() is True
    assert (d / "a.txt").read_text() == "original"      # change reverted
    assert not (d / "new.txt").exists()                 # addition removed
    assert (d / "keep" / "n.txt").read_text() == "nested"  # deletion restored
    snap.drop()
    assert not os.path.isfile(snap.tar_path)


def test_snapshot_refuses_a_non_directory(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("x")
    with pytest.raises(ValueError):
        FileTreeSnapshot(str(f))


def test_extract_rejects_path_traversal(tmp_path):
    import io
    import tarfile

    from loom.packs.snapshots import _safe_extractall

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo(name="../escape.txt")
        data = b"evil"
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    buf.seek(0)
    with tarfile.open(fileobj=buf) as tar:
        with pytest.raises(ValueError, match="unsafe path"):
            _safe_extractall(tar, str(tmp_path))
