"""Tool capability contracts: policy on what a tool DOES, not its name."""

import json

from loom import Agent, tool
from loom.capabilities import capabilities, manifest, matches_cap
from loom.cli import main
from loom.providers import ScriptedProvider
from loom.shield import ALLOW, CONFIRM, DENY, Shield


def test_inference_from_name_and_input():
    assert capabilities("Read", {"file_path": "a.py"}) == {"read", "idempotent"}
    assert "secret" in capabilities("Read", {"file_path": "/app/.env"})
    assert capabilities("WebFetch", {"url": "x"}) >= {"network"}
    assert capabilities("Bash", {"command": "rm -rf /"}) >= {"exec", "destructive"}
    assert capabilities("get_weather", {}) == {"read", "idempotent"}


def test_shell_synonyms_all_read_as_exec():
    for name in ["sh", "bash", "run_command", "execute_code", "python_repl"]:
        assert "exec" in capabilities(name, {}), name


def test_business_capabilities_generalize_past_coding():
    # The pivot: capabilities describe business risk, not just shell/fs.
    assert "money_movement" in capabilities("issue_refund", {"amount": 50})
    assert "pii_access" in capabilities("get_customer", {"id": 7})
    assert "user_communication" in capabilities("send_email", {"to": "x"})
    assert "database_write" in capabilities("run_sql", {"query": "INSERT INTO t VALUES (1)"})
    assert "browser_submit" in capabilities("click_button", {})
    # money/user-comm/browser/db writes are all observable external side effects
    for name in ["issue_refund", "send_email", "click_button"]:
        assert "external_side_effect" in capabilities(name, {}), name


def test_business_capability_rules_gate_by_capability():
    shield = Shield(confirm=["cap:money_movement"], allow=["cap:read"])
    assert shield.classify("issue_refund", {"amount": 9})[0] == CONFIRM
    assert shield.classify("process_payment", {})[0] == CONFIRM
    assert shield.classify("get_status", {})[0] == ALLOW  # a read, not money


def test_business_capabilities_do_not_false_positive():
    # anchored globs: 'recharge_cache' isn't money; 'update_status' display isn't db
    assert "money_movement" not in capabilities("recharge_cache", {})
    assert "pii_access" not in capabilities("get_config", {})


def test_name_hints_do_not_false_positive_on_substrings():
    # Token-anchored globs must not flag innocent tools: 'prune' isn't exec,
    # 'capital' isn't network, 'truncate' isn't exec.
    for name in ["prune_logs", "truncate_display", "capital_gains", "rapid_sort",
                 "evaluate_answer", "brunt_force"]:
        caps = capabilities(name, {})
        assert "exec" not in caps and "network" not in caps, (name, caps)


def test_unclassifiable_tool_is_not_assumed_idempotent():
    # A tool we can't classify at all is *unknown*, not safe-to-rerun.
    assert capabilities("prune_logs", {}) == set()


def test_declared_capabilities_win_over_inference():
    # a tool named innocently but declared as network
    assert capabilities("summarize", {}, declared={"network"}) == {"network"}
    assert matches_cap("cap:network", "summarize", {}, declared={"network"})


def test_shield_cap_rules_match_by_capability():
    shield = Shield(deny=["cap:exec"], allow=["cap:read"])
    assert shield.classify("run_command", {"command": "ls"})[0] == DENY   # exec, any name
    assert shield.classify("sh", {})[0] == DENY
    assert shield.classify("Glob", {"pattern": "*.py"})[0] == ALLOW       # read


def test_tool_declares_capabilities():
    @tool(capabilities={"network", "write"})
    def deploy(target: str) -> str:
        "Deploy."
        return "ok"

    assert deploy.capabilities == {"network", "write"}


def test_manifest_lists_tools():
    @tool
    def Read(path: str) -> str:
        "read"
        return "x"

    @tool(capabilities={"exec", "destructive"})
    def danger() -> str:
        "danger"
        return "x"

    rows = {r["tool"]: r for r in manifest([Read, danger])}
    assert rows["danger"]["declared"] is True
    assert set(rows["danger"]["capabilities"]) == {"exec", "destructive"}
    assert rows["Read"]["declared"] is False and "read" in rows["Read"]["capabilities"]


def test_cli_tools_manifest(tmp_path, capsys, monkeypatch):
    (tmp_path / "agentmod.py").write_text(
        "from loom import Agent, tool\n"
        "from loom.providers import ScriptedProvider\n"
        "@tool(capabilities={'exec'})\n"
        "def run(cmd: str) -> str:\n"
        "    'run'\n"
        "    return 'ok'\n"
        "agent = Agent(model=ScriptedProvider([]), tools=[run])\n"
    )
    monkeypatch.chdir(tmp_path)
    assert main(["tools", "--agent", "agentmod:agent"]) == 0
    out = capsys.readouterr().out
    assert "run" in out and "exec" in out and "declared" in out
    assert main(["tools", "--agent", "agentmod:agent", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["tool"] == "run"
