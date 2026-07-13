"""Dialect-blind reconstruction: the SAME multi-agent scenario, recorded off the
wire in Anthropic vs OpenAI vs Gemini format, must normalize to the same agents,
steps, and taint findings -- the analyzers never see the dialect."""
import pytest

from loom.debugger import steps_for
from loom.multiagent import infer_agents
from loom.proxy import WireRecorder
from loom.taint import taint_paths

SUP = "You are the Supervisor. Delegate to the coder, then read the customer record."
COD = "You are the Coder. Write code and email the result out."
SSN = "123-45-6789"


def _anthropic():
    r = WireRecorder()
    r.record({"model": "claude", "system": SUP, "messages": [{"role": "user", "content": "build it"}]},
             {"content": [{"type": "tool_use", "id": "d1", "name": "ask_coder", "input": {}}],
              "stop_reason": "tool_use", "usage": {}})
    r.record({"model": "claude", "system": COD,
              "messages": [{"role": "user", "content": "write code and read the customer"}]},
             {"content": [{"type": "tool_use", "id": "c1", "name": "read_customer", "input": {}}],
              "stop_reason": "tool_use", "usage": {}})
    r.record({"model": "claude", "system": COD, "messages": [
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": f"SSN={SSN}"}]}]},
        {"content": [{"type": "tool_use", "id": "e1", "name": "send_email", "input": {"body": f"ssn {SSN}"}}],
         "stop_reason": "tool_use", "usage": {}})
    return r.to_dict()


def _openai():
    r = WireRecorder()
    r.record({"model": "gpt", "messages": [{"role": "system", "content": SUP}, {"role": "user", "content": "build it"}]},
             {"choices": [{"message": {"content": None, "tool_calls": [
                 {"id": "d1", "type": "function", "function": {"name": "ask_coder", "arguments": "{}"}}]}}], "usage": {}})
    r.record({"model": "gpt", "messages": [{"role": "system", "content": COD}, {"role": "user", "content": "write code and read the customer"}]},
             {"choices": [{"message": {"content": None, "tool_calls": [
                 {"id": "c1", "type": "function", "function": {"name": "read_customer", "arguments": "{}"}}]}}], "usage": {}})
    r.record({"model": "gpt", "messages": [{"role": "system", "content": COD},
                                           {"role": "tool", "tool_call_id": "c1", "content": f"SSN={SSN}"}]},
             {"choices": [{"message": {"content": None, "tool_calls": [
                 {"id": "e1", "type": "function", "function": {"name": "send_email", "arguments": '{"body": "ssn ' + SSN + '"}'}}]}}], "usage": {}})
    return r.to_dict()


def _gemini():
    r = WireRecorder()
    r.record({"systemInstruction": {"parts": [{"text": SUP}]}, "contents": [{"role": "user", "parts": [{"text": "build it"}]}]},
             {"candidates": [{"content": {"parts": [{"functionCall": {"name": "ask_coder", "args": {}}}]}}], "usageMetadata": {}})
    r.record({"systemInstruction": {"parts": [{"text": COD}]}, "contents": [{"role": "user", "parts": [{"text": "write code and read the customer"}]}]},
             {"candidates": [{"content": {"parts": [{"functionCall": {"name": "read_customer", "args": {}}}]}}], "usageMetadata": {}})
    r.record({"systemInstruction": {"parts": [{"text": COD}]}, "contents": [
        {"role": "user", "parts": [{"functionResponse": {"name": "read_customer", "response": {"result": f"SSN={SSN}"}}}]}]},
        {"candidates": [{"content": {"parts": [{"functionCall": {"name": "send_email", "args": {"body": f"ssn {SSN}"}}}]}}], "usageMetadata": {}})
    return r.to_dict()


DIALECTS = {"anthropic": _anthropic, "openai": _openai, "gemini": _gemini}


@pytest.mark.parametrize("name", list(DIALECTS))
def test_each_dialect_recovers_two_agents(name):
    data = DIALECTS[name]()
    ia = infer_agents(data)
    assert ia["source"] == "wire"
    labels = sorted(a["label"] for a in ia["agents"])
    assert labels == ["Coder", "Supervisor"], f"{name} recovered {labels}"
    assert steps_for(data), f"{name} produced no steps"


def test_reconstruction_is_identical_across_dialects():
    """The agent ROSTER a run reconstructs to must not depend on the wire dialect."""
    rosters = {name: sorted(a["label"] for a in infer_agents(fn())["agents"])
               for name, fn in DIALECTS.items()}
    assert rosters["anthropic"] == rosters["openai"] == rosters["gemini"] == ["Coder", "Supervisor"]


@pytest.mark.parametrize("name", list(DIALECTS))
def test_taint_catches_the_exfil_in_every_dialect(name):
    """The SSN read then emailed out must be caught regardless of dialect."""
    paths = taint_paths(DIALECTS[name]())
    assert any(p["sink"]["tool"] == "send_email" and p["kind"] == "ssn" for p in paths), \
        f"{name} missed the exfil"
