"""Offline, deterministic providers -- no SDK, no API key, no network.

These make the harness genuinely runnable and testable out of the box, and they
produce reproducible traces that double as replay fixtures.

  * ``ScriptedProvider`` -- returns a fixed sequence of responses. Perfect for
    tests and for demonstrating the agent loop deterministically.
  * ``RuleProvider``     -- picks a response based on the current context via
    user-supplied rules. Because its output depends on context, editing the
    context at a fork point changes the downstream branch -- ideal for showing
    off ``run.fork(...)``.
"""

from __future__ import annotations

from typing import Callable

from .base import ModelResponse


class ScriptedProvider:
    """Replays a fixed list of ``ModelResponse`` objects, one per model call."""

    def __init__(self, responses: list[ModelResponse], model: str = "scripted"):
        self.responses = list(responses)
        self.model = model
        self.name = "scripted"
        self._i = 0

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> ModelResponse:
        if self._i >= len(self.responses):
            # The script ran out; end the turn gracefully instead of crashing.
            return ModelResponse(text="", stop_reason="end_turn")
        resp = self.responses[self._i]
        self._i += 1
        return resp


Rule = Callable[[list[dict]], "ModelResponse | None"]


class RuleProvider:
    """Chooses a response by testing rules against the live message list.

    Each rule is ``fn(messages) -> ModelResponse | None``; the first rule to
    return non-None wins. ``default`` is used when no rule matches.
    """

    def __init__(
        self,
        rules: list[Rule],
        default: "ModelResponse | None" = None,
        model: str = "rule",
    ):
        self.rules = rules
        self.default = default or ModelResponse(text="(no rule matched)")
        self.model = model
        self.name = "rule"

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> ModelResponse:
        for rule in self.rules:
            resp = rule(messages)
            if resp is not None:
                return resp
        return self.default
