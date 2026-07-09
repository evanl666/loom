"""Signed shield decisions + the approver policy."""

import threading
import time

from loom.shield import Shield, verify_approvals


def _approve_on(shield, who, approve=True, delay=0.05):
    time.sleep(delay)
    for _ in range(60):
        pend = shield.pending_list()
        if pend:
            return shield.decide_pending(pend[0]["id"], approve, who=who)
        time.sleep(0.05)
    return None


def _confirm(shield, name="issue_refund", tool_input=None):
    """Fire a confirm-flow decision with a background approver already set up."""
    return shield._judge(name, tool_input or {"amount": 500})


def test_only_a_listed_identity_may_approve():
    shield = Shield(confirm=["cap:money_movement"],
                    approvers={"cap:money_movement": ["alice"]}, timeout=1.5)
    result = {}
    threading.Thread(target=lambda: result.update(bob=_approve_on(shield, "bob"))).start()
    _, event = _confirm(shield)
    assert result["bob"] is False              # bob's approval was refused
    assert event["action"] == "deny"           # so the call timed out -> denied


def test_a_listed_identity_approves_and_it_is_signed():
    shield = Shield(confirm=["cap:money_movement"],
                    approvers={"cap:money_movement": ["alice", "bob"]},
                    sign_key=b"secret", timeout=3)
    result = {}
    threading.Thread(target=lambda: result.update(a=_approve_on(shield, "alice"))).start()
    allowed, event = _confirm(shield)
    assert result["a"] is True and allowed is True
    assert event["by"] == "alice"
    assert event["signature"].startswith("hmac-sha256:")


def test_anyone_may_deny_regardless_of_approver_policy():
    shield = Shield(confirm=["cap:money_movement"],
                    approvers={"cap:money_movement": ["alice"]}, timeout=3)
    result = {}
    threading.Thread(
        target=lambda: result.update(bob=_approve_on(shield, "bob", approve=False))).start()
    allowed, event = _confirm(shield)
    assert result["bob"] is True               # a DENY from bob is accepted
    assert allowed is False and event["action"] == "deny"


def test_verify_approvals_detects_tampering():
    event = {"id": "abc", "tool": "issue_refund", "action": "approve",
             "via": "operator", "by": "alice", "ts": 123}
    signed = Shield(sign_key=b"secret")._sign(dict(event))
    ok, bad = verify_approvals({"shield_events": [signed]}, b"secret")
    assert len(ok) == 1 and not bad

    tampered = dict(signed, by="mallory")       # forge the approver
    ok2, bad2 = verify_approvals({"shield_events": [tampered]}, b"secret")
    assert not ok2 and len(bad2) == 1

    _, bad3 = verify_approvals({"shield_events": [signed]}, b"wrong-key")
    assert len(bad3) == 1


def test_unsigned_events_are_ignored_by_the_verifier():
    ok, bad = verify_approvals(
        {"shield_events": [{"action": "deny", "tool": "x"}]}, b"secret")
    assert not ok and not bad


def test_cli_verify_approvals(tmp_path, capsys, monkeypatch):
    import json

    from loom.cli import main

    event = Shield(sign_key=b"topsecret")._sign(
        {"id": "a1", "tool": "issue_refund", "action": "approve",
         "via": "operator", "by": "alice", "ts": 1})
    path = tmp_path / "r.loom.json"
    path.write_text(json.dumps({"log": [], "shield_events": [event]}))

    monkeypatch.setenv("LOOM_APPR_KEY", "topsecret")
    assert main(["trace", "verify-approvals", str(path), "--key-env", "LOOM_APPR_KEY"]) == 0
    assert "1 valid, 0 invalid" in capsys.readouterr().out

    # tamper, then it must fail
    data = json.loads(path.read_text())
    data["shield_events"][0]["by"] = "mallory"
    path.write_text(json.dumps(data))
    assert main(["trace", "verify-approvals", str(path), "--key-env", "LOOM_APPR_KEY"]) == 1
    assert "SIGNATURE INVALID" in capsys.readouterr().out


def test_build_shield_parses_require_approver_flag():
    import argparse

    from loom.cli import _build_shield

    args = argparse.Namespace(
        profile="", policy="", deny=[], confirm=["cap:money_movement"], allow=[], rule=[],
        shield_default="allow", confirm_timeout=300.0, webhook="", judge="",
        judge_threshold=0.7, trust_after=0, trust_ledger="",
        sign_approvals_key_env="", require_approver=["cap:money_movement=alice,bob"],
        break_glass=[])
    shield = _build_shield(args)
    assert shield.approvers == {"cap:money_movement": {"names": ["alice", "bob"], "min": 1}}


def test_approval_chain_requires_two_distinct_identities():
    shield = Shield(confirm=["cap:money_movement"],
                    approvers={"cap:money_movement": {"names": ["alice", "bob", "carol"],
                                                      "min": 2}},
                    sign_key=b"k", timeout=4)

    def chain():
        for _ in range(80):
            pend = shield.pending_list()
            if pend:
                pid = pend[0]["id"]
                assert pend[0]["required"] == 2
                assert shield.decide_pending(pid, True, who="alice") is True
                # one approval is NOT enough; still pending with alice recorded
                still = shield.pending_list()
                assert still and still[0]["approvals"] == ["alice"]
                # alice approving twice doesn't satisfy the chain
                shield.decide_pending(pid, True, who="alice")
                assert shield.pending_list()
                # bob completes the chain
                assert shield.decide_pending(pid, True, who="bob") is True
                return
            time.sleep(0.05)

    t = threading.Thread(target=chain)
    t.start()
    allowed, event = shield._judge("issue_refund", {"amount": 900})
    t.join()
    assert allowed is True
    assert event["by"] == "alice+bob"
    assert event["signature"].startswith("hmac-sha256:")


def test_break_glass_approves_single_handedly_and_is_flagged():
    shield = Shield(confirm=["cap:money_movement"],
                    approvers={"cap:money_movement": {"names": ["alice", "bob"], "min": 2}},
                    break_glass=["oncall"], timeout=4)

    def glass():
        for _ in range(80):
            pend = shield.pending_list()
            if pend:
                assert shield.decide_pending(pend[0]["id"], True, who="oncall") is True
                return
            time.sleep(0.05)

    t = threading.Thread(target=glass)
    t.start()
    allowed, event = shield._judge("issue_refund", {"amount": 900})
    t.join()
    assert allowed is True
    assert event["via"] == "break-glass"          # loudly flagged
    assert event["by"] == "oncall"


def test_deny_short_circuits_a_chain():
    shield = Shield(confirm=["cap:money_movement"],
                    approvers={"cap:money_movement": {"names": ["alice", "bob"], "min": 2}},
                    timeout=4)

    def deny():
        for _ in range(80):
            pend = shield.pending_list()
            if pend:
                pid = pend[0]["id"]
                shield.decide_pending(pid, True, who="alice")   # first approval
                shield.decide_pending(pid, False, who="mallory")  # anyone may deny
                return
            time.sleep(0.05)

    t = threading.Thread(target=deny)
    t.start()
    allowed, event = shield._judge("issue_refund", {"amount": 900})
    t.join()
    assert allowed is False and event["action"] == "deny"


def test_build_shield_parses_chain_syntax_and_break_glass():
    import argparse

    from loom.cli import _build_shield

    args = argparse.Namespace(
        profile="", policy="", deny=[], confirm=["cap:money_movement"], allow=[], rule=[],
        shield_default="allow", confirm_timeout=300.0, webhook="", judge="",
        judge_threshold=0.7, trust_after=0, trust_ledger="",
        sign_approvals_key_env="",
        require_approver=["cap:money_movement=2:alice,bob"],
        break_glass=["oncall"])
    shield = _build_shield(args)
    assert shield.approvers["cap:money_movement"] == {"names": ["alice", "bob"], "min": 2}
    assert shield.break_glass == ["oncall"]
    assert shield.required_approvals("issue_refund", {}) == 2
