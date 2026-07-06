"""Model providers. The kernel only needs the ``ModelProvider`` protocol.

Offline (no deps): ``ScriptedProvider``, ``RuleProvider``.
Live (optional):   ``AnthropicProvider`` -- import triggers the anthropic SDK.
"""

from .base import ModelProvider, ModelResponse, ToolCall
from .scripted import RuleProvider, ScriptedProvider


def __getattr__(name: str):  # lazy so `import loom.providers` never needs an SDK
    if name == "AnthropicProvider":
        from .anthropic import AnthropicProvider

        return AnthropicProvider
    if name == "OpenAIProvider":
        from .openai import OpenAIProvider

        return OpenAIProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ModelProvider",
    "ModelResponse",
    "ToolCall",
    "ScriptedProvider",
    "RuleProvider",
    "AnthropicProvider",
    "OpenAIProvider",
]
