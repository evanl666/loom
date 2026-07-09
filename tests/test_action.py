"""The generic Action schema -- the Action Debugger's vocabulary."""

from loom import Agent, actions, tool
from loom.action import Action, PolicyDecision
from loom.providers import ModelResponse, ScriptedProvider, ToolCall


@tool
def get_customer(id: int) -> str:
    "Look up a customer record."
    return f"customer {id}: Jane Doe"


@tool
def send_email(to: str, body: str) -> str:
    "Email a customer."
    return "sent"


def _run():
    prov = ScriptedProvider([
        ModelResponse(text="Looking up the customer to answer their question.",
                      tool_calls=[ToolCall("t1", "get_customer", {"id": 7})],
                      stop_reason="tool_use"),
        ModelResponse(text="All set."),
    ])
    return Agent(model=prov, tools=[get_customer, send_email]).run("who is customer 7?")


def test_actions_pair_intent_call_and_observation():
    acts = _run().actions()
    types = [a.type for a in acts]
    assert types == ["reason", "call", "answer"]

    call = acts[1]
    assert call.tool == "get_customer"
    assert call.input == {"id": 7}
    assert call.intent.startswith("Looking up the customer")  # WHY, from the model
    assert call.observation.text == "customer 7: Jane Doe"
    assert "pii_access" in call.capabilities
    assert call.risk == "pii-access"


def test_replay_points_mark_top_level_turns():
    acts = _run().actions()
    forkable = [a.step for a in acts if a.replay.forkable]
    # both model calls are top-level turn boundaries; the tool call is not
    assert acts[1].replay.forkable is False
    assert len(forkable) == 2
    assert acts[0].replay.turn == 0


def test_policy_decisions_attach_from_shield_events():
    data = {
        "log": [
            {"seq": 0, "kind": "model", "key": "k",
             "result": ModelResponse(tool_calls=[ToolCall("t1", "Bash", {"command": "ls"})],
                                     stop_reason="tool_use").to_dict()},
            {"seq": 1, "kind": "tool:Bash", "key": "k2", "result": "file1\nfile2"},
        ],
        "shield_events": [
            {"tool": "Bash", "input": {"command": "ls"}, "action": "allow",
             "rule": "cap:exec", "via": "rule"},
        ],
    }
    acts = actions(data)
    call = [a for a in acts if a.type == "call"][0]
    assert isinstance(call.policy, PolicyDecision)
    assert call.policy.action == "allow" and call.policy.rule == "cap:exec"
    assert call.policy.blocked is False


def test_blocked_calls_become_their_own_actions():
    data = {
        "log": [
            {"seq": 0, "kind": "model", "key": "k",
             "result": ModelResponse(text="I'll read the env.").to_dict()},
        ],
        "shield_events": [
            {"tool": "Read", "input": {"file_path": "/app/.env"}, "action": "deny",
             "rule": "Read(*.env*)", "via": "rule"},
        ],
    }
    acts = actions(data)
    blocked = [a for a in acts if a.policy and a.policy.blocked]
    assert len(blocked) == 1
    assert blocked[0].tool == "Read"
    assert blocked[0].observation.error is True
    assert "secret" in blocked[0].capabilities


def test_action_to_dict_is_json_shaped():
    call = _run().actions()[1]
    d = call.to_dict()
    assert d["type"] == "call" and d["tool"] == "get_customer"
    assert d["risk"] == "pii-access"
    assert d["observation"]["text"] == "customer 7: Jane Doe"
    assert d["replay"]["forkable"] is False


def test_actions_accepts_a_plain_trace_dict(tmp_path):
    run = _run()
    path = str(tmp_path / "r.loom.json")
    run.save(path)
    import json

    data = json.load(open(path))
    acts = actions(data)
    assert [a.type for a in acts] == ["reason", "call", "answer"]
