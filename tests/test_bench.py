"""loom bench: same task, several agents, one comparison table."""

import json
import sys
import threading

from loom.bench import _oracle, load_task, report, score
from loom.cli import main
from tests.test_proxy import FINAL_ANSWER, WEATHER_TOOL_USE, _FakeUpstream

# Two "agents": one makes a single clean call, one makes a blocked tool call.
GOOD_AGENT = """
import json, os, urllib.request
url = os.environ["ANTHROPIC_BASE_URL"] + "/v1/messages"
req = urllib.request.Request(url,
    data=json.dumps({"model": "m", "messages": [{"role": "user", "content": os.environ.get("PROMPT","")}]}).encode(),
    headers={"content-type": "application/json"}, method="POST")
urllib.request.urlopen(req, timeout=10).read()
"""


def test_oracle_variants(tmp_path):
    assert _oracle({"success": {"contains": "OK"}}, "all OK here", ".")[0] is True
    assert _oracle({"success": {"contains": "OK"}}, "nope", ".")[0] is False
    assert _oracle({"success": {"absent": "ERROR"}}, "clean run", ".")[0] is True
    assert _oracle({"success": {"absent": "ERROR"}}, "ERROR!", ".")[0] is False
    ok, how = _oracle({"success": {"command": "true"}}, "", ".")
    assert ok and "exit 0" in how
    assert _oracle({"success": {"command": "false"}}, "", ".")[0] is False


def test_score_reads_the_trace():
    trace = {
        "log": [
            {"kind": "model", "result": {"tool_calls": [{"name": "Bash"}],
                                          "usage": {"input_tokens": 10, "output_tokens": 4}}},
            {"kind": "model", "result": {"tool_calls": [], "usage": {"input_tokens": 20, "output_tokens": 6}}},
        ],
        "shield_events": [{"action": "deny"}, {"action": "approve"}],
    }
    s = score(trace, passed=True)
    assert s == {"passed": True, "tokens": 40, "steps": 2, "tools": 1, "blocked": 1}


def test_load_task_requires_a_prompt(tmp_path):
    import pytest

    p = tmp_path / "t.yaml"
    p.write_text("success:\n  contains: OK\n")
    with pytest.raises(ValueError, match="needs a 'prompt'"):
        load_task(str(p))


def test_report_ranks_cheapest_passing():
    results = [
        {"name": "big", "passed": True, "tokens": 900, "steps": 5, "tools": 2, "blocked": 0},
        {"name": "small", "passed": True, "tokens": 200, "steps": 3, "tools": 1, "blocked": 0},
        {"name": "broken", "passed": False, "tokens": 100, "steps": 1, "tools": 0, "blocked": 1},
    ]
    text = report("task.yaml", results)
    assert "cheapest passing: small" in text
    assert "✅" in text and "❌" in text


def test_bench_cli_runs_two_agents(tmp_path, capsys):
    upstream = _FakeUpstream([FINAL_ANSWER, FINAL_ANSWER, FINAL_ANSWER, FINAL_ANSWER])
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    target = f"http://127.0.0.1:{upstream.server_address[1]}"

    agent = tmp_path / "agent.py"
    agent.write_text(GOOD_AGENT)
    task = tmp_path / "task.yaml"
    task.write_text('prompt: "say something"\nsuccess:\n  contains: "raining"\n')

    code = main([
        "bench", str(task),
        "--agent", f"a:{sys.executable} {agent}",
        "--agent", f"b:{sys.executable} {agent}",
        "--target", target,
        "--outdir", str(tmp_path / "traces"),
    ])
    upstream.shutdown()
    assert code == 0  # FINAL_ANSWER contains "raining" -> both pass
    out = capsys.readouterr().out
    assert "Task:" in out
    assert out.count("✅") == 2  # both agents passed
    assert (tmp_path / "traces" / "a.loom.json").exists()
    assert (tmp_path / "traces" / "b.loom.json").exists()


def test_bench_cli_rejects_bad_agent_spec(tmp_path, capsys):
    task = tmp_path / "task.yaml"
    task.write_text('prompt: "x"\n')
    assert main(["bench", str(task), "--agent", "noseparator"]) == 2
    assert "name:command" in capsys.readouterr().err
