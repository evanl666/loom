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


def test_target_inference():
    from loom.bench import target_for

    # dialect is inferred only when the run-wide target is still a default API
    assert target_for("codex exec {prompt}", "https://api.anthropic.com") == "https://api.openai.com"
    assert target_for("claude -p {prompt}", "https://api.openai.com") == "https://api.anthropic.com"
    # an explicit custom target (a local mock, a vLLM endpoint) ALWAYS wins --
    # 'claude' in the command must not reroute traffic to the real API
    assert target_for("claude -p {prompt}", "http://127.0.0.1:9999") == "http://127.0.0.1:9999"
    assert target_for("my-agent run", "https://default") == "https://default"


def test_reset_workspace_git(tmp_path):
    import subprocess

    from loom.bench import reset_workspace

    subprocess.run(["git", "init", "-q"], cwd=tmp_path)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path)
    (tmp_path / "f.py").write_text("v1\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path,
                          capture_output=True, text=True).stdout.strip()

    # simulate an agent's mess: edit a file, add an untracked one
    (tmp_path / "f.py").write_text("agent broke this\n")
    (tmp_path / "junk.py").write_text("agent litter\n")

    assert reset_workspace("git", str(tmp_path), head) == ""
    assert (tmp_path / "f.py").read_text() == "v1\n"       # reverted
    assert not (tmp_path / "junk.py").exists()             # cleaned
    assert reset_workspace("bogus", str(tmp_path), head)   # unknown mode -> error string


def test_bench_reset_refuses_dirty_tree(tmp_path, capsys, monkeypatch):
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path)
    (tmp_path / "f.py").write_text("v1\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path)
    (tmp_path / "f.py").write_text("uncommitted work I care about\n")  # dirty

    task = tmp_path / "task.yaml"
    task.write_text('prompt: "x"\n')
    monkeypatch.chdir(tmp_path)
    code = main(["bench", str(task), "--agent", "a:true", "--reset", "git"])
    assert code == 2
    assert "dirty" in capsys.readouterr().err
    assert (tmp_path / "f.py").read_text() == "uncommitted work I care about\n"  # untouched


def test_bench_reset_copy_isolates_workspaces(tmp_path, capsys, monkeypatch):
    # Two agents each write the SAME file; with copy isolation, neither sees
    # the other's edit (each runs in its own copy of the repo).
    import sys as _sys
    import threading as _th

    from tests.test_proxy import FINAL_ANSWER, _FakeUpstream

    (tmp_path / "seed.txt").write_text("original\n")
    agent = tmp_path / "agent.py"
    agent.write_text(
        "import json, os, urllib.request\n"
        "assert open('seed.txt').read() == 'original\\n', 'workspace was polluted!'\n"
        "open('seed.txt','w').write('touched by agent\\n')\n"
        "url = os.environ['ANTHROPIC_BASE_URL'] + '/v1/messages'\n"
        "req = urllib.request.Request(url,"
        " data=json.dumps({'model':'m','messages':[{'role':'user','content':'hi'}]}).encode(),"
        " headers={'content-type':'application/json'}, method='POST')\n"
        "urllib.request.urlopen(req, timeout=10).read()\n"
    )
    task = tmp_path / "task.yaml"
    task.write_text('prompt: "x"\nsuccess:\n  contains: "raining"\n')

    up = _FakeUpstream([FINAL_ANSWER, FINAL_ANSWER])
    _th.Thread(target=up.serve_forever, daemon=True).start()
    monkeypatch.chdir(tmp_path)
    code = main([
        "bench", str(task),
        "--agent", f"a:{_sys.executable} {agent}",
        "--agent", f"b:{_sys.executable} {agent}",
        "--target", f"http://127.0.0.1:{up.server_address[1]}",
        "--outdir", str(tmp_path / "bt"),
        "--reset", "copy",
    ])
    up.shutdown()
    # both agents' asserts passed (neither saw the other's write) -> exit 0
    assert code == 0
    # the real repo's file is untouched by the copies
    assert (tmp_path / "seed.txt").read_text() == "original\n"
