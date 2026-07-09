"""loom harden: deployment recommendation from the threat model."""

from loom.cli import main
from loom.harden import describe, harden, policy_yaml, scenarios


def test_every_scenario_recommends_a_real_profile():
    from loom.policy_file import PROFILES

    for s in scenarios():
        rec = harden(s)
        assert rec["profile"] in PROFILES
        assert rec["why"]


def test_support_policy_encodes_the_approval_chain():
    yaml = policy_yaml("support")
    assert "profile: customer-data-safe" in yaml
    assert "cap:money_movement" in yaml and "min: 2" in yaml

    # and it loads into a Shield with the chain intact
    import os

    from loom.policy_file import resolve, to_shield_kwargs
    from loom.shield import Shield

    p = os.path.join(os.environ.get("PYTEST_CURRENT_TEST", "").split("::")[0].rsplit("/", 1)[0]
                     if False else "/tmp", "h.yml")
    with open(p, "w") as f:
        f.write(yaml)
    shield = Shield(**to_shield_kwargs(resolve(policy_path=p)))
    assert shield.approvers["cap:money_movement"] == {"names": ["manager", "finance"], "min": 2}


def test_cli_harden_prints_and_writes(tmp_path, capsys):
    assert main(["harden", "--scenario", "coding"]) == 0
    out = capsys.readouterr().out
    assert "claude-code-safe" in out and "--sandbox" in out

    outfile = tmp_path / "policy.yml"
    assert main(["harden", "--scenario", "ci", "-o", str(outfile)]) == 0
    assert "profile: github-actions-safe" in outfile.read_text()


def test_cli_harden_unknown_scenario_is_a_clean_error(capsys):
    assert main(["harden", "--scenario", "nope"]) == 2
    assert "unknown scenario" in capsys.readouterr().err
