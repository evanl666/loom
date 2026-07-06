"""The Anthropic provider translates messages both ways correctly (no network).

These tests exercise the pure translation helpers so the live path is trustworthy
without an API key or the SDK's network layer.
"""

from types import SimpleNamespace

import pytest

anthropic_mod = pytest.importorskip("anthropic")  # skip if SDK not installed

from loom.providers.anthropic import AnthropicProvider
from loom.providers.base import ToolCall


def test_to_anthropic_messages_merges_tool_results():
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "let me check",
            "tool_calls": [ToolCall("a", "add", {"x": 1}), ToolCall("b", "add", {"x": 2})],
        },
        {"role": "tool", "tool_call_id": "a", "name": "add", "content": "1"},
        {"role": "tool", "tool_call_id": "b", "name": "add", "content": "2"},
    ]
    out = AnthropicProvider._to_anthropic_messages(messages)

    assert out[0] == {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    # assistant carries a text block + two tool_use blocks
    assistant = out[1]
    assert assistant["role"] == "assistant"
    kinds = [b["type"] for b in assistant["content"]]
    assert kinds == ["text", "tool_use", "tool_use"]
    # the two tool results are merged into ONE following user message
    assert out[2]["role"] == "user"
    assert [b["type"] for b in out[2]["content"]] == ["tool_result", "tool_result"]
    assert len(out) == 3


def test_from_anthropic_extracts_text_and_tool_calls():
    fake = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="the answer is"),
            SimpleNamespace(type="tool_use", id="t1", name="add", input={"a": 1, "b": 2}),
        ],
        usage=SimpleNamespace(input_tokens=11, output_tokens=7),
    )
    resp = AnthropicProvider._from_anthropic(fake)
    assert resp.text == "the answer is"
    assert resp.stop_reason == "tool_use"
    assert resp.tool_calls[0].name == "add"
    assert resp.tool_calls[0].input == {"a": 1, "b": 2}
    assert resp.usage == {"input_tokens": 11, "output_tokens": 7}


def test_from_anthropic_plain_text_ends_turn():
    fake = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="done")],
        usage=SimpleNamespace(input_tokens=3, output_tokens=1),
    )
    resp = AnthropicProvider._from_anthropic(fake)
    assert resp.stop_reason == "end_turn"
    assert resp.tool_calls == []
