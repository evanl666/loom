"""The shared LLM-judge engine: semantic assertions, experiment ranking,
copilot-proposed auto-fixes."""

import json

from loom import Agent, tool
from loom.assertions import check_assertions
from loom.experiment import run_experiment
from loom.judge import llm_judge, run_summary
from loom.providers import ModelResponse, ScriptedProvider, ToolCall


class TranscriptJudge:
    """Passes iff `needle` appears in the TRANSCRIPT section (not the expectation)."""

    model = "fake-judge"

    def __init__(self, needle: str):
        self.needle = needle

    def complete(self, system, messages, tools):
        transcript = messages[0]["content"].split("TRANSCRIPT:", 1)[1]
        ok = self.needle in transcript
        return ModelResponse(text=json.dumps(
            {"pass": ok, "reason": "shown" if ok else "not shown"}), stop_reason="end_turn")


def _trace(output="Refunded order 42"):
    return {"log": [
        {"seq": 0, "kind": "model", "result": {"tool_calls": [
            {"id": "1", "name": "get_order", "input": {}}], "stop_reason": "tool_use"}},
        {"seq": 1, "kind": "tool:get_order", "result": "order 42 verified"},
        {"seq": 2, "kind": "model", "result": {"text": output, "stop_reason": "end_turn"}}],
        "prompt": "refund order 42", "output": output, "tools": {}}


def test_llm_judge_verdicts_and_summary():
    s = run_summary(_trace())
    assert "get_order" in s and "FINAL OUTPUT" in s and "refund order 42" in s
    assert llm_judge(TranscriptJudge("order 42 verified"), "x", _trace())["ok"] is True
    assert llm_judge(TranscriptJudge("never-there"), "x", _trace())["ok"] is False


def test_llm_judge_never_raises_on_bad_replies():
    class Bad:
        model = "bad"
        def complete(self, s, m, t):
            return ModelResponse(text="sure!", stop_reason="end_turn")

    class Boom:
        model = "boom"
        def complete(self, s, m, t):
            raise RuntimeError("api down")

    assert "error" in llm_judge(Bad(), "x", _trace())
    assert "error" in llm_judge(Boom(), "x", _trace())


def test_semantic_assertion_line():
    r = check_assertions(_trace(), [
        "judge: the order was verified",
        "output contains order 42",
    ], judge=TranscriptJudge("verified"))
    assert r["all_pass"] and r["results"][0]["detail"] == "shown"
    # failing judge verdict fails the assertion (with the reason)
    r2 = check_assertions(_trace(), ["judge: x"], judge=TranscriptJudge("nope"))
    assert r2["results"][0]["ok"] is False


def test_judge_line_without_a_model_is_an_error_not_a_pass():
    r = check_assertions(_trace(), ["judge: anything"])
    x = r["results"][0]
    assert "error" in x and not x.get("ok")


def test_experiment_ranks_by_semantic_judge():
    agent = Agent(model=ScriptedProvider([
        ModelResponse(text="Refunded after verifying the order", stop_reason="end_turn"),
        ModelResponse(text="Refunded blindly", stop_reason="end_turn")]), tools=[])
    res = run_experiment(agent, "refund", systems=["verify first", "just do it"],
                         judge=TranscriptJudge("verifying"),
                         criteria="the agent verified before refunding")
    assert res[0]["success"] is True and res[1]["success"] is False
    assert res[0]["system"].startswith("verify")  # the semantic winner ranks first
    assert "judge_reason" in res[0]


def test_auto_fix_includes_copilot_proposed_fixes():
    from loom.debugger import DebugSession

    @tool
    def get() -> str:
        "get"
        return "data"

    class Echo:
        model = "echo"
        def complete(self, system, messages, tools):
            n = sum(1 for m in messages if m.get("role") == "assistant")
            if n == 0:
                return ModelResponse(tool_calls=[ToolCall("1", "get", {})],
                                     stop_reason="tool_use")
            return ModelResponse(text="answer", stop_reason="end_turn")

    class ProposerCopilot:
        model = "proposer"
        def complete(self, system, messages, tools):
            return ModelResponse(text=json.dumps(
                [{"label": "be specific", "edit": "Answer with the exact number."}]),
                stop_reason="end_turn")

    import tempfile, os
    agent = Agent(model=Echo(), tools=[get])
    p = os.path.join(tempfile.mkdtemp(), "t.loom.json")
    agent.run("go").save(p)
    sess = DebugSession(p, agent=agent, copilot_model=ProposerCopilot())
    fixes = sess._proposed_fixes()
    assert fixes == [("copilot: be specific", {"append": "Answer with the exact number."})]
    # and a broken copilot yields no extra fixes, never an exception
    class Broken:
        model = "broken"
        def complete(self, s, m, t):
            raise RuntimeError("down")
    sess2 = DebugSession(p, agent=agent, copilot_model=Broken())
    assert sess2._proposed_fixes() == []


def test_run_summary_always_keeps_request_and_output_on_long_runs():
    # regression: the FINAL OUTPUT used to be appended then truncated away on a
    # long run, so a `judge:` assertion about the output silently saw nothing.
    log = []
    for i in range(300):
        log.append({"seq": len(log), "kind": "model", "result": {"tool_calls": [
            {"id": str(i), "name": "lookup", "input": {"q": "x" * 60}}], "stop_reason": "tool_use"}})
        log.append({"seq": len(log), "kind": "tool:lookup", "result": "r" * 300})
    log.append({"seq": len(log), "kind": "model",
                "result": {"text": "OUTPUT_SENTINEL", "stop_reason": "end_turn"}})
    data = {"log": log, "prompt": "REQUEST_SENTINEL", "output": "OUTPUT_SENTINEL", "tools": {}}
    s = run_summary(data)
    assert len(s) <= 4000
    assert "REQUEST_SENTINEL" in s and "OUTPUT_SENTINEL" in s   # both survive
    assert "elided" in s                                        # the middle is elided
