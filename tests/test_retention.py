"""loom retention: age-based scrub/delete lifecycle for a corpus."""

import json
import os
import time

from loom import Agent, tool
from loom.cli import main
from loom.providers import ModelResponse, ScriptedProvider, ToolCall
from loom.retention import apply_retention, plan_retention

SECRET = "sk-ant-api03-" + "a1B2" * 8


@tool
def Read(file_path: str) -> str:
    "read"
    return f"ANTHROPIC_API_KEY={SECRET} contact jane@example.com"


def _corpus(tmp_path):
    now = time.time()
    for name, age_days in [("fresh", 1), ("old", 45), ("ancient", 120)]:
        run = Agent(model=ScriptedProvider([
            ModelResponse(tool_calls=[ToolCall("t", "Read", {"file_path": "x"})],
                          stop_reason="tool_use"),
            ModelResponse(text="done"),
        ]), tools=[Read]).run("go")
        p = str(tmp_path / f"{name}.loom.json")
        run.save(p)
        old = now - age_days * 86400
        os.utime(p, (old, old))
    return str(tmp_path)


def test_plan_by_age(tmp_path):
    d = _corpus(tmp_path)
    plan = {os.path.basename(i["path"]): i["action"]
            for i in plan_retention(d, {"scrub_after": "30d", "delete_after": "90d"})}
    assert plan["fresh.loom.json"] == "keep"
    assert plan["old.loom.json"] == "scrub"
    assert plan["ancient.loom.json"] == "delete"


def test_dry_run_changes_nothing(tmp_path):
    d = _corpus(tmp_path)
    before = set(os.listdir(d))
    apply_retention(d, {"scrub_after": "30d", "delete_after": "90d"}, dry_run=True)
    assert set(os.listdir(d)) == before  # nothing removed
    assert SECRET in (tmp_path / "old.loom.json").read_text()  # not scrubbed


def test_apply_scrubs_and_deletes_with_pii(tmp_path):
    d = _corpus(tmp_path)
    audit = apply_retention(d, {"scrub_after": "30d", "delete_after": "90d",
                                "redact_pii": True}, dry_run=False)
    assert not os.path.exists(tmp_path / "ancient.loom.json")   # deleted
    old = json.loads((tmp_path / "old.loom.json").read_text())
    assert old["scrubbed"] is True
    blob = json.dumps(old)
    assert SECRET not in blob and "jane@example.com" not in blob   # secret + PII gone
    assert (tmp_path / "fresh.loom.json").exists()                 # kept
    assert any(a["action"] == "delete" and a["applied"] for a in audit)


def test_cli_retention_dry_run_then_apply(tmp_path, capsys):
    d = _corpus(tmp_path)
    assert main(["retention", d, "--scrub-after", "30d", "--delete-after", "90d"]) == 0
    out = capsys.readouterr().out
    assert "dry run" in out and "would" in out
    assert os.path.exists(tmp_path / "ancient.loom.json")  # dry run kept it

    audit_path = str(tmp_path / "audit.json")
    assert main(["retention", d, "--scrub-after", "30d", "--delete-after", "90d",
                 "--apply", "--audit", audit_path]) == 0
    assert not os.path.exists(tmp_path / "ancient.loom.json")
    assert os.path.exists(audit_path)


def test_legal_hold_exempts_from_scrub_and_delete(tmp_path):
    d = _corpus(tmp_path)
    audit = apply_retention(d, {"scrub_after": "30d", "delete_after": "90d",
                                "legal_hold": ["ancient.loom.json"]}, dry_run=False)
    actions = {os.path.basename(a["path"]): a["action"] for a in audit}
    assert actions["ancient.loom.json"] == "hold"          # NOT deleted
    assert os.path.exists(tmp_path / "ancient.loom.json")
    assert actions["old.loom.json"] == "scrub"             # hold is per-pattern


def test_dsar_plan_scrub_delete(tmp_path, capsys):
    from loom.cli import main
    from loom.retention import dsar

    d = _corpus(tmp_path)  # every trace result contains jane@example.com
    plan = dsar(d, "jane@example.com", mode="plan")
    assert len(plan) == 3 and all(p["occurrences"] >= 1 for p in plan)

    # scrub redacts in place and re-checksums
    dsar(d, "jane@example.com", mode="scrub")
    blob = (tmp_path / "fresh.loom.json").read_text()
    assert "jane@example.com" not in blob and "[scrubbed:dsar]" in blob
    assert dsar(d, "jane@example.com", mode="plan") == []  # gone everywhere

    # CLI: an identifier nobody has
    assert main(["dsar", d, "--value", "no-such-person@x.com"]) == 0
    assert "no trace contains" in capsys.readouterr().out


def test_dsar_rejects_short_identifiers(tmp_path):
    import pytest as _pytest
    from loom.retention import dsar

    with _pytest.raises(ValueError):
        dsar(str(tmp_path), "ab", mode="plan")
