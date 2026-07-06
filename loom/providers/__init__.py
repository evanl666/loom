"""Model providers. The kernel only needs the ``ModelProvider`` protocol.

Offline (no deps): ``ScriptedProvider``, ``RuleProvider``.
Live (optional):   ``AnthropicProvider`` -- import triggers the anthropic SDK.
"""

from .base import ModelProvider, ModelResponse, ToolCall
from .scripted import RuleProvider, ScriptedProvider


def __getattr__(name: str):  # lazy so `import loom.providers` never needs the SDK
    if name == "AnthropicProvider":
        from .anthropic import AnthropicProvider

        return AnthropicProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ModelProvider",
    "ModelResponse",
    "ToolCall",
    "ScriptedProvider",
    "RuleProvider",
    "AnthropicProvider",
]
