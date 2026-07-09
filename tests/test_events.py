"""loom export --jsonl/--otel: a trace as observability events."""

import io
import json

from loom import Agent, tool
from loom.cli import main
from loom.events import events_for, export_events, to_otel
from loom.providers import ModelResponse, ScriptedProvider, ToolCall


@tool
def Bash(command: str) -> str:
    "shell"
    return "ok"


def _trace(tmp_path):
    provider = ScriptedProvider([
        ModelResponse(tool_calls=[ToolCall("t1", "Bash", {"command": "ls"})],
                      stop_reason="tool_use", usage={"input_tokens": 10, "output_tokens": 4}),
        ModelResponse(text="done", usage={"input_tokens": 20, "output_tokens": 6}),
    ])
    run = Agent(model=provider, tools=[Bash]).run("go")
    path = str(tmp_path / "r.loom.json")
    run.save(path)
    data = json.load(open(path))
    data["shield_events"] = [{"action": "deny", "tool": "Bash",
                              "input": {"command": "curl x"}, "rule": "cap:exec", "via": "rule"}]
    json.dump(data, open(path, "w"))
    return path


def test_events_flatten_a_trace(tmp_path):
    path = _trace(tmp_path)
    evs = events_for(path, json.load(open(path)))
    kinds = [e["kind"] for e in evs]
    assert "model" in kinds and "tool:Bash" in kinds and "shield" in kinds

    tool_ev = next(e for e in evs if e["kind"] == "tool:Bash")
    assert "exec" in tool_ev["capabilities"]        # capability tagged
    model_ev = next(e for e in evs if e["kind"] == "model")
    assert model_ev["input_tokens"] == 10 and model_ev["tool_calls"] == ["Bash"]
    shield_ev = next(e for e in evs if e["kind"] == "shield")
    assert shield_ev["action"] == "deny" and "network-egress" in shield_ev["risk"]

    # all events share one run id
    assert len({e["run"] for e in evs}) == 1


def test_otel_wrapping():
    otel = to_otel({"run": "abc", "kind": "model", "seq": 0, "input_tokens": 5,
                    "output_tokens": 2})
    assert otel["resource"]["loom.run"] == "abc"
    assert otel["name"] == "loom.model"
    attrs = otel["attributes"]
    assert attrs["loom.seq"] == 0
    assert attrs["loom.tokens.input"] == 5 and attrs["loom.tokens.output"] == 2
    assert "run" not in attrs and "input_tokens" not in attrs  # namespaced only


def test_export_events_to_buffer(tmp_path):
    path = _trace(tmp_path)
    buf = io.StringIO()
    n = export_events([path], buf)
    lines = [json.loads(x) for x in buf.getvalue().splitlines()]
    assert n == len(lines) and n >= 4
    assert all("run" in ln for ln in lines)


def test_cli_export_jsonl(tmp_path, capsys):
    path = _trace(tmp_path)
    out = str(tmp_path / "events.jsonl")
    assert main(["export", path, "--jsonl", out]) == 0
    lines = open(out).read().splitlines()
    assert len(lines) >= 4 and all(json.loads(x)["run"] for x in lines)

    # a directory corpus streams too
    assert main(["export", str(tmp_path), "--jsonl", "-"]) == 0
    assert '"kind": "shield"' in capsys.readouterr().out


def test_action_events_carry_the_stable_semantic_fields(tmp_path):
    path = _trace(tmp_path)
    evs = events_for(path, json.load(open(path)))
    action_evs = [e for e in evs if e["kind"] == "action"]
    assert action_evs, "action-level events missing"

    call = next(e for e in action_evs if e.get("tool") == "Bash" and e.get("action_type") == "call")
    assert "exec" in call["capabilities"]
    # the denied call surfaces with its policy decision
    denied = next(e for e in action_evs if e.get("policy_action") == "deny")
    assert denied["policy_rule"] == "cap:exec"

    # OTel mapping uses the STABLE names from docs/action-schema.md
    attrs = to_otel(call)["attributes"]
    assert attrs["loom.action.type"] == "call"
    assert "exec" in attrs["loom.capability"]
    otel_denied = to_otel(denied)["attributes"]
    assert otel_denied["loom.policy.action"] == "deny"
    assert otel_denied["loom.policy.rule"] == "cap:exec"


def test_action_events_include_state_diff_kind(tmp_path):
    @tool
    def Edit(file_path: str, new: str) -> str:
        "edit"
        return "edited"

    provider = ScriptedProvider([
        ModelResponse(tool_calls=[ToolCall("t1", "Edit", {"file_path": "a.py", "new": "x"})],
                      stop_reason="tool_use"),
        ModelResponse(text="done"),
    ])
    path = str(tmp_path / "e.loom.json")
    Agent(model=provider, tools=[Edit]).run("go").save(path)
    evs = events_for(path, json.load(open(path)))
    call = next(e for e in evs if e["kind"] == "action" and e.get("tool") == "Edit")
    assert call["state_diff_kind"] == "file"
    assert to_otel(call)["attributes"]["loom.state_diff.kind"] == "file"
