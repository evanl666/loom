"""loom redteam: canonical attacks screened through a policy."""

from loom.cli import main
from loom.policy_file import PROFILES, resolve, to_shield_kwargs
from loom.redteam import run_all, run_scenario, scenarios
from loom.shield import Shield


def _shield(profile="", policy=""):
    return Shield(**to_shield_kwargs(resolve(profile=profile, policy_path=policy)))


def test_claude_code_safe_stops_every_scenario():
    results = run_all(_shield(profile="claude-code-safe"))
    assert len(results) == len(scenarios())
    assert all(r["stopped"] for r in results)


def test_permissive_policy_lets_attacks_through():
    r = run_scenario("secret_exfil", Shield(default="allow"))
    assert r["stopped"] is False and r["firewall"] == "allow"


def test_prod_db_safe_denies_the_drop():
    r = run_scenario("sql_destroy", _shield(profile="prod-db-safe"))
    assert r["stopped"] and r["firewall"] == "deny"


def test_cli_redteam_gates(tmp_path, capsys):
    assert main(["redteam", "run", "--profile", "claude-code-safe"]) == 0
    assert "5/5 attack(s) stopped" in capsys.readouterr().out

    weak = tmp_path / "weak.yml"
    weak.write_text("default: allow\n")
    assert main(["redteam", "run", "--policy", str(weak), "--scenario", "secret_exfil"]) == 1
    out = capsys.readouterr().out
    assert "GOT THROUGH" in out and "secret_exfil" in out
