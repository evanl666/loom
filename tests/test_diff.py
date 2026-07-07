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
