"""Trace diff distinguishes control-flow, input, and result divergences."""

from loom import Agent, diff_logs
from loom.diff import INPUTS_DIFFER, KINDS_DIFFER, ONLY_B, RESULTS_DIFFER
from loom.effect import EffectEntry
from loom.providers import ModelResponse, ScriptedProvider


def e(seq, kind, key, result):
    return EffectEntry(seq=seq, kind=kind, key=key, result=result)


def test_identical_logs():
    a = [e(0, "model", "k0", {"text": "hi"})]
    d = diff_logs(a, list(a))
    assert d.identical
    assert d.first_divergence is None
    assert "identical" in d.summary()


def test_inputs_differ_detected_via_key():
    a = [e(0, "model", "aaa", {"text": "hi"})]
    b = [e(0, "model", "bbb", {"text": "hi"})]
    d = diff_logs(a, b)
    assert d.steps[0].status == INPUTS_DIFFER
    assert d.first_divergence == 0


def test_results_differ_same_inputs():
    a = [e(0, "model", "same", {"text": "hi"})]
    b = [e(0, "model", "same", {"text": "hello"})]
    d = diff_logs(a, b)
    assert d.steps[0].status == RESULTS_DIFFER


def test_kinds_differ_is_control_flow_divergence():
    a = [e(0, "model", "k", {}), e(1, "tool:add", "k", 5)]
    b = [e(0, "model", "k", {}), e(1, "tool:multiply", "k", 6)]
    d = diff_logs(a, b)
    assert d.steps[0].status == "identical"
    assert d.steps[1].status == KINDS_DIFFER
    assert d.first_divergence == 1


def test_length_mismatch_marks_extra_steps():
    a = [e(0, "model", "k", {})]
    b = [e(0, "model", "k", {}), e(1, "tool:add", "k2", 5)]
    d = diff_logs(a, b)
    assert d.steps[1].status == ONLY_B
    assert d.counts() == {"identical": 1, "only-b": 1}


def test_run_diff_integration():
    def agent(text):
        return Agent(model=ScriptedProvider([ModelResponse(text=text, stop_reason="end_turn")]))

    r1 = agent("answer A").run("same question")
    r2 = agent("answer B").run("same question")  # same inputs, different reply
    d = r1.diff(r2)
    assert d.steps[0].status == RESULTS_DIFFER

    r3 = agent("answer A").run("different question")  # different inputs
    d2 = r1.diff(r3)
    assert d2.steps[0].status == INPUTS_DIFFER


# -- action-level behavior diff ------------------------------------------------


def _saved(tmp_path, name, calls):
    from loom import tool
    from loom.providers import ToolCall

    made = []
    for tool_name in {c[0] for c in calls}:
        @tool
        def t(**kwargs) -> str:
            "a tool"
            return "x"
        t.name = tool_name
        made.append(t)
    responses = [
        ModelResponse(tool_calls=[ToolCall(f"t{i}", n, inp)], stop_reason="tool_use")
        for i, (n, inp) in enumerate(calls)
    ] + [ModelResponse(text="done")]
    run = Agent(model=ScriptedProvider(responses), tools=made).run("go")
    path = str(tmp_path / name)
    run.save(path)
    return path


def test_diff_actions_reports_added_risk(tmp_path):
    import json

    from loom.diff import describe_action_diff, diff_actions

    a = _saved(tmp_path, "a.loom.json", [("Read", {"file_path": "a.py"})])
    b = _saved(tmp_path, "b.loom.json", [("Read", {"file_path": "a.py"}),
                                          ("send_email", {"to": "jane@x.com"})])
    d = diff_actions(json.load(open(a)), json.load(open(b)))
    assert d["added"] == [{"tool": "send_email", "risk": "user-comm", "count": 1}]
    assert d["removed"] == []
    assert d["risk_gained"] == ["user-comm"]
    assert d["score"]["b"] < d["score"]["a"]                 # overall behavior score dropped
    assert any(m["dimension"] == "security" for m in d["score_moved"])
    text = describe_action_diff(d)
    assert "behavior score:" in text and "⬇" in text
    assert "security:" in text                                # a moved dimension is explained
    assert "+ send_email x1  ⚠ user-comm" in text


def test_diff_actions_identical_runs_score_unchanged(tmp_path):
    import json

    from loom.diff import diff_actions

    a = _saved(tmp_path, "a.loom.json", [("Read", {"file_path": "a.py"})])
    b = _saved(tmp_path, "b.loom.json", [("Read", {"file_path": "a.py"})])
    d = diff_actions(json.load(open(a)), json.load(open(b)))
    assert d["added"] == [] and d["removed"] == []
    assert d["score"]["a"] == d["score"]["b"] == 100


def test_cli_diff_actions_gates_on_change(tmp_path, capsys):
    from loom.cli import main

    a = _saved(tmp_path, "a.loom.json", [("Read", {"file_path": "a.py"})])
    b = _saved(tmp_path, "b.loom.json", [("Read", {"file_path": "a.py"}),
                                          ("run_sql", {"query": "INSERT INTO t VALUES (1)"})])
    assert main(["diff", a, b, "--actions"]) == 1  # behavior changed
    out = capsys.readouterr().out
    assert "behavior score" in out and "run_sql" in out
    assert main(["diff", a, a, "--actions"]) == 0  # same behavior passes


def test_score_breakdown_is_explainable(tmp_path):
    import json

    from loom.diff import describe_score, score_breakdown

    # a run that moves money, emails a user, and gates nothing
    b = _saved(tmp_path, "b.loom.json", [
        ("issue_refund", {"amount": 500, "order_id": "A-17"}),
        ("send_email", {"to": "jane@x.com", "body": "done"})])
    bd = score_breakdown(json.load(open(b)))
    dims = bd["dimensions"]
    assert 0 <= bd["overall"] <= 100
    assert dims["security"]["score"] < 100                   # risk exercised
    assert dims["reversibility"]["score"] < 100              # sent email can't be undone
    assert dims["policy_coverage"]["score"] < 100            # nothing gated
    assert "money-movement" in dims["security"]["why"]
    text = describe_score(bd)
    assert "behavior score:" in text and "reversibility" in text


def test_score_breakdown_clean_run_is_high(tmp_path):
    import json

    from loom.diff import score_breakdown

    a = _saved(tmp_path, "a.loom.json", [("Read", {"file_path": "a.py"})])
    bd = score_breakdown(json.load(open(a)))
    assert bd["overall"] >= 90  # a plain read scores near-perfect
