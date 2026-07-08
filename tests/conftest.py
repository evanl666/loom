import pytest


@pytest.fixture(autouse=True)
def _isolated_loom_runtime(tmp_path, monkeypatch):
    """Keep proxy control tokens out of the real ~/.loom during tests."""
    monkeypatch.setenv("LOOM_RUNTIME_DIR", str(tmp_path / "loom-runtime"))
