"""CLI: loom record / heal / skills -- the black-box workflow end to end."""

import json
import sys
import threading

from loom.cli import main
from tests.test_proxy import FINAL_ANSWER, _FakeUpstream

CHILD = """
import json, os, urllib.request
url = os.environ["ANTHROPIC_BASE_URL"] + "/v1/messages"
req = urllib.request.Request(
    url,
    data=json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}]}).encode(),
    headers={"content-type": "application/json"},
    method="POST",
)
urllib.request.urlopen(req, timeout=10).read()
raise SystemExit({exit_code})
"""


def _run_record(tmp_path, exit_code=0):
    upstream = _FakeUpstream([FINAL_ANSWER])
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    child = tmp_path / "child.py"
    child.write_text(CHILD.replace("{exit_code}", str(exit_code)))
    save = str(tmp_path / "session.loom.json")
    code = main(
        [
            "record",
            "--save", save,
            "--target", f"http://127.0.0.1:{upstream.server_address[1]}",
            "--",
            sys.executable, str(child),
        ]
    )
    upstream.shutdown()
    return code, save


def test_record_wraps_a_command_and_saves_the_trace(tmp_path):
    code, save = _run_record(tmp_path)
    assert code == 0
    with open(save) as f:
        data = json.load(f)
    assert data["recorded_via"] == "proxy"
    assert data["output"] == "It is raining in Berlin."


def test_record_passes_the_childs_exit_code_through(tmp_path):
    code, _ = _run_record(tmp_path, exit_code=3)
    assert code == 3


def test_record_without_a_command_is_an_error(tmp_path, capsys):
    assert main(["record", "--save", str(tmp_path / "x.json")]) == 2
    assert "needs a command" in capsys.readouterr().err


def test_heal_cli_fixes_and_saves_regression(tmp_path, monkeypatch, capsys):
    from tests.test_health import build_poisoned_agent

    run = build_poisoned_agent().run("What is the answer?")
    trace = str(tmp_path / "failed.loom.json")
    run.save(trace)
    (tmp_path / "healmod.py").write_text(
        "from tests.test_health import build_poisoned_agent\nagent = build_poisoned_agent()\n"
    )
    monkeypatch.chdir(tmp_path)

    code = main(
        ["heal", trace, "--agent", "healmod:agent", "--forbid", "ERROR",
         "--save-regression", str(tmp_path / "regressions")]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "healed by: redact-" in out
    assert "saved regression:" in out
    assert list((tmp_path / "regressions").glob("healed-*.loom.json"))


def test_heal_cli_requires_a_check(tmp_path, capsys):
    assert main(["heal", "x.json", "--agent", "m:a"]) == 2
    assert "--forbid and/or --require" in capsys.readouterr().err


def test_skills_cli_mines_and_saves(tmp_path, capsys):
    from tests.test_skills import record_run

    for i, city in enumerate(["Berlin", "Lisbon"]):
        record_run(city).save(str(tmp_path / f"r{i}.loom.json"))
    lib = str(tmp_path / "skills.json")
    code = main(["skills", str(tmp_path), "--save", lib])
    out = capsys.readouterr().out
    assert code == 0
    assert "skill: skill_geocode_then_forecast" in out
    assert "support: 2 runs" in out
    assert "parameters: " in out
    with open(lib) as f:
        assert json.load(f)[0]["name"] == "skill_geocode_then_forecast"


def test_skills_cli_approve_and_filters(tmp_path, capsys):
    from tests.test_skills import record_run

    for i, city in enumerate(["Berlin", "Lisbon"]):
        record_run(city).save(str(tmp_path / f"r{i}.loom.json"))
    lib = str(tmp_path / "skills.json")
    main(["skills", str(tmp_path), "--save", lib])
    assert "unapproved" in capsys.readouterr().out
    with open(lib) as f:
        assert json.load(f)[0]["approved"] is False

    assert main(["skills", lib, "--approve", "skill_geocode_then_forecast"]) == 0
    with open(lib) as f:
        assert json.load(f)[0]["approved"] is True
    assert main(["skills", lib, "--approve", "nope"]) == 1

    # a success filter that no run passes -> nothing mined
    assert main(["skills", str(tmp_path), "--require", "IMPOSSIBLE"]) == 1


def test_skills_cli_reports_nothing_found(tmp_path, capsys):
    from tests.test_skills import record_run

    record_run("Berlin").save(str(tmp_path / "only.loom.json"))
    assert main(["skills", str(tmp_path)]) == 1  # one run is an anecdote