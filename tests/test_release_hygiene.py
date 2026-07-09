"""Release hygiene: version consistency across the repo (also gated in CI)."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _pyproject_version() -> str:
    return re.search(r'^version = "([^"]+)"',
                     (ROOT / "pyproject.toml").read_text(), re.M).group(1)


def test_version_is_consistent():
    import loom

    assert loom.__version__ == _pyproject_version()


def test_readme_has_no_stale_version_pin():
    readme = (ROOT / "README.md").read_text()
    pins = re.findall(r"loom-harness==(\d+\.\d+\.\d+)", readme)
    current = _pyproject_version()
    stale = [p for p in pins if p != current]
    assert not stale, f"README pins stale version(s) {stale} (current {current})"
