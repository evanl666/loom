"""Pack certification: lint_pack + test_pack keep domain packs honest."""

from loom.packs import Pack, UndoPlan, packs
from loom.packs.browser import BrowserPack
from loom.packs.certify import lint_pack, load_pack
from loom.packs.certify import test_pack as run_pack_cases
from loom.packs.coding import CodingPack
from loom.packs.sql import SqlPack
from loom.packs.support import SupportPack


def test_builtin_packs_lint_clean():
    for pack in (CodingPack(), SqlPack(), BrowserPack(), SupportPack()):
        assert lint_pack(pack) == [], f"{pack.name} should certify clean"


def test_certify_does_not_disturb_the_global_registry():
    before = [p.name for p in packs()]
    lint_pack(SqlPack())
    run_pack_cases(SqlPack(), [{"action": {"tool": "run_query",
                                      "input": {"query": "INSERT INTO t VALUES (1)"}},
                           "expect": {"owns": True}}])
    assert [p.name for p in packs()] == before  # restored


class _OverBroadPack(Pack):
    name = "over-broad"

    def owns(self, action):
        return True  # claims everything


def test_lint_flags_over_broad_owns():
    problems = lint_pack(_OverBroadPack())
    assert any("over-broad" in p for p in problems)


class _LyingUndoPack(Pack):
    name = "liar"

    def owns(self, action):
        return action.tool == "issue_refund"

    def undo(self, action, trace):
        return UndoPlan("revert", "just re-charge", reversible=True)


def test_lint_flags_fake_reversibility_on_money():
    problems = lint_pack(_LyingUndoPack())
    assert any("reversible=True" in p and "issue_refund" in p for p in problems)


def test_test_pack_golden_cases_pass_and_fail():
    cases = [
        {"action": {"tool": "run_query", "input": {"query": "INSERT INTO orders VALUES (1)"},
                    "result": "1 row inserted"},
         "expect": {"owns": True, "capabilities_include": ["database_write"],
                    "state_diff_kind": "database", "undo_kind": "compensate",
                    "reversible": False}},
        {"action": {"tool": "run_query", "input": {"query": "DROP TABLE users"}},
         "expect": {"undo_kind": "revert"}},  # wrong on purpose: DROP is a noop plan
    ]
    results = run_pack_cases(SqlPack(), cases)
    assert results[0]["ok"] is True
    assert results[1]["ok"] is False
    assert any("undo kind" in f for f in results[1]["failures"])


def test_insert_is_not_labeled_a_clean_revert():
    # Regression: certification caught an INSERT compensation marked reversible
    # even though the trace lacks the inserted rows' keys.
    from loom.packs.certify import _action

    a = _action(SqlPack(), "run_query", {"query": "INSERT INTO t VALUES (1)"})
    plan = SqlPack().undo(a, {})
    assert plan.kind == "compensate" and plan.reversible is False


def test_load_pack_resolves_class_and_instance():
    assert load_pack("loom.packs.sql:SqlPack").name == "sql"


def test_scaffolded_pack_certifies_clean(tmp_path, monkeypatch):
    import subprocess
    import sys

    from loom.cli import main

    outdir = str(tmp_path / "pk")
    assert main(["packs", "new", "acme", "-o", outdir]) == 0
    import os
    assert os.path.isfile(os.path.join(outdir, "acme_pack.py"))
    assert os.path.isfile(os.path.join(outdir, "cases.yml"))

    # the scaffold lints clean and its golden case passes, out of the box
    env = {**os.environ, "PYTHONPATH": outdir}
    lint = subprocess.run([sys.executable, "-m", "loom", "packs", "lint",
                           "--pack", "acme_pack:AcmePack"], env=env,
                          capture_output=True, text=True)
    assert lint.returncode == 0, lint.stdout + lint.stderr
    test = subprocess.run([sys.executable, "-m", "loom", "packs", "test",
                           os.path.join(outdir, "cases.yml")], env=env,
                          capture_output=True, text=True)
    assert test.returncode == 0 and "1/1 case(s) passed" in test.stdout
