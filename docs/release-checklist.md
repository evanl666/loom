# Release checklist

Every release follows the same sequence — the version-consistency CI job
enforces the invariant (`loom.__version__` == `pyproject.toml` == the tag),
this document records the rest.

1. **CI green on `main`** for the commit being released (all jobs: the
   3.10–3.13 test matrix, the flight-recorder demo, the docker jail, the
   containerized recording, version-consistency).
2. **Bump the version** in *both* places — `pyproject.toml` and
   `loom/__init__.py` — in one commit.
3. **Build fresh**: `rm -rf dist && python -m build`.
4. **Full test run locally** (`pytest -q`), *check the exit code*, then commit
   and push the bump.
5. **Upload**: `twine upload --non-interactive dist/*`.
6. **Tag and release**: `git tag vX.Y.Z && git push origin vX.Y.Z`, then a
   GitHub Release with the dist artifacts attached and notes that lead with
   the headline features.
7. **Verify from a clean venv in a neutral directory**, pinned to the exact
   version (`pip install "loom-harness==X.Y.Z"` — the index can lag a minute
   or two): import it, exercise the release's headline features, check the
   new CLI surface exists.
8. If anything fails after upload: PyPI uploads are immutable — fix forward
   with a patch release, never delete.

Known pitfalls (learned the hard way):

- `raw.githubusercontent.com` caches for ~5 minutes; "main still shows the
  old version" right after a push is usually the CDN, not the repo. Check
  `git show origin/main:pyproject.toml` instead.
- Don't pipe `pytest` through `tail` and trust the output — zsh needs
  `pipestatus[1]`; write to a file and `echo $?`.
- Verify the released package from a **neutral cwd**: running `python -c
  "import loom"` from the repo root resolves the local source, not
  site-packages.
