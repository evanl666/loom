"""The OpenAI-compatible provider translates messages both ways (no network)."""

from types import SimpleNamespace

from loom.providers.base import ToolCall

# The translation helpers are pure static methods -- importable without the SDK.
from loom.providers.openai import OpenAIProvider


def test_to_openai_messages_shapes():
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "let me check",
            "tool_calls": [ToolCall("call_1", "add", {"a": 1, "b": 2})],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "add", "content": "3"},
    ]
    out = OpenAIProvider._to_openai_messages("be helpful", messages)

    assert out[0] == {"role": "system", "content": "be helpful"}
    assert out[1] == {"role": "user", "content": "hi"}
    assistant = out[2]
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"][0]["id"] == "call_1"
    assert assistant["tool_calls"][0]["function"]["name"] == "add"
    # arguments are serialized to a JSON string
    assert assistant["tool_calls"][0]["function"]["arguments"] == '{"a": 1, "b": 2}'
    assert out[3] == {"role": "tool", "tool_call_id": "call_1", "content": "3"}


def test_from_openai_extracts_tool_call():
    fake = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call_9",
                            function=SimpleNamespace(name="add", arguments='{"a": 5, "b": 6}'),
                        )
                    ],
                ),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=8),
    )
    resp = OpenAIProvider._from_openai(fake)
    assert resp.stop_reason == "tool_use"
    assert resp.tool_calls[0].name == "add"
    assert resp.tool_calls[0].input == {"a": 5, "b": 6}
    assert resp.usage == {"input_tokens": 12, "output_tokens": 8}


def test_from_openai_plain_text():
    fake = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="hello there", tool_calls=None),
            )
        ],
        usage=None,
    )
    resp = OpenAIProvider._from_openai(fake)
    assert resp.text == "hello there"
    assert resp.stop_reason == "end_turn"
    assert resp.tool_calls == []


def test_bad_tool_arguments_degrade_gracefully():
    fake = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="c", function=SimpleNamespace(name="f", arguments="{not json")
                        )
                    ],
                ),
            )
        ],
        usage=None,
    )
    resp = OpenAIProvider._from_openai(fake)
    assert resp.tool_calls[0].input == {}  # malformed args -> empty dict, no crash
