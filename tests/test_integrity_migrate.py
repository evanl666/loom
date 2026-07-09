"""Checksums make edits visible; loom migrate brings old traces forward."""

import json
import warnings

import pytest

from loom import Agent, Run, tool
from loom.cli import main
from loom.effect import ReplayMismatch
from loom.migrate import migrate
from loom.providers import ModelResponse, ScriptedProvider, ToolCall
from loom.testing import verify_trace
from loom.trace import TRACE_VERSION, trace_checksum


@tool
def lookup(city: str) -> str:
    "Look up a city."
    return f"data for {city}"


def _agent():
    return Agent(
        model=ScriptedProvider([
            ModelResponse(tool_calls=[ToolCall("t1", "lookup", {"city": "Berlin"})],
                          stop_reason="tool_use"),
            ModelResponse(text="answered"),
        ]),
        tools=[lookup],
        system="geo bot",
    )


def _record(tmp_path):
    path = str(tmp_path / "run.loom.json")
    _agent().run("Berlin?").save(path)
    return path


# ------------------------------------------------------------------ checksum


def test_saved_traces_are_stamped_and_clean_loads_are_silent(tmp_path):
    path = _record(tmp_path)
    data = json.load(open(path))
    assert data["checksum"].startswith("sha256:")
    assert data["checksum"] == trace_checksum(data)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        Run.load(path)
    assert verify_trace(path) == []


def test_hand_edits_are_visible(tmp_path):
    path = _record(tmp_path)
    data = json.load(open(path))
    data["output"] = "doctored answer"
    json.dump(data, open(path, "w"))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Run.load(path)
    assert any("modified after it was written" in str(w.message) for w in caught)
    assert any("checksum mismatch" in p for p in verify_trace(path))

    # migrate blesses a deliberate edit: content untouched, stamp renewed
    migrate(path)
    stamped = json.load(open(path))
    assert stamped["output"] == "doctored answer"
    assert stamped["checksum"] == trace_checksum(stamped)


def test_scrub_restamps_its_edit(tmp_path):
    path = str(tmp_path / "leaky.loom.json")
    run_path = _record(tmp_path)
    data = json.load(open(run_path))
    data["output"] = "key: sk-ant-api03-" + "a1B2" * 8
    data["checksum"] = trace_checksum(data)
    json.dump(data, open(path, "w"))

    assert main(["scrub", path, "--in-place"]) == 0
    clean = json.load(open(path))
    assert "sk-ant" not in clean["output"]
    assert clean["checksum"] == trace_checksum(clean)  # stamp matches the new content


# ------------------------------------------------------------------- migrate


def _fake_v1(tmp_path):
    """A trace whose model keys predate v2 (no tool schemas in the hash)."""
    path = _record(tmp_path)
    data = json.load(open(path))
    for e in data["log"]:
        if e["kind"] == "model":
            e["key"] = "0" * 40  # stale v1-era hash
    data["version"] = 1
    data.pop("checksum", None)
    old = str(tmp_path / "v1.loom.json")
    json.dump(data, open(old, "w"))
    return old


def test_migrate_rekeys_a_v1_harness_trace(tmp_path):
    old = _fake_v1(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ReplayMismatch):
            Run.load(old, agent=_agent()).replay()  # v1 keys fail strict replay

    rekeyed, out = migrate(old, agent=_agent())
    assert rekeyed == 2  # both model effects
    data = json.load(open(out))
    assert data["version"] == TRACE_VERSION

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # migrated trace loads clean...
        run = Run.load(out, agent=_agent())
    assert run.replay().output == "answered"  # ...and passes strict replay


def test_migrate_harness_trace_without_agent_is_a_clear_error(tmp_path):
    old = _fake_v1(tmp_path)
    with pytest.raises(ValueError, match="--agent module:attr"):
        migrate(old)


def test_cli_migrate(tmp_path, capsys, monkeypatch):
    old = _fake_v1(tmp_path)
    (tmp_path / "agentmod.py").write_text(
        "from tests.test_integrity_migrate import _agent\nagent = _agent()\n"
    )
    monkeypatch.chdir(tmp_path)
    assert main(["migrate", old, "--agent", "agentmod:agent"]) == 0
    assert "2 effect key(s) recomputed" in capsys.readouterr().out
