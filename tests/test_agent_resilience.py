"""Production essentials in the loop: model retry/backoff, tool timeout."""

import time

import pytest

from loom import Agent, tool
from loom.agent import _transient
from loom.providers import ModelResponse, ScriptedProvider


class FlakyProvider:
    """Fails N times with a given exception, then delegates to the script."""

    def __init__(self, inner, failures, exc):
        self.inner = inner
        self.model = inner.model
        self.name = "flaky"
        self.failures = failures
        self.exc = exc
        self.calls = 0

    def complete(self, system, messages, tools):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc
        return self.inner.complete(system, messages, tools)


class RateLimitError(Exception):
    status_code = 429


class AuthError(Exception):
    status_code = 401


def test_transient_classifier():
    assert _transient(ConnectionError())
    assert _transient(TimeoutError())
    assert _transient(RateLimitError())
    assert not _transient(AuthError())
    assert not _transient(ValueError("bad request"))

    class APIConnectionError(Exception):  # name-hint fallback, no status
        pass

    assert _transient(APIConnectionError())


def test_model_retries_weather_rate_limits_invisibly():
    flaky = FlakyProvider(
        ScriptedProvider([ModelResponse(text="made it")]), failures=2, exc=RateLimitError()
    )
    agent = Agent(model=flaky, model_retries=2, retry_backoff=0)
    run = agent.run("hello")
    assert run.output == "made it"
    assert flaky.calls == 3
    assert len([e for e in run.log if e.kind == "model"]) == 1  # retries never recorded
    assert run.replay().output == "made it"  # strict replay unaffected by the weather


def test_retries_exhausted_reraises():
    flaky = FlakyProvider(
        ScriptedProvider([ModelResponse(text="never")]), failures=5, exc=RateLimitError()
    )
    with pytest.raises(RateLimitError):
        Agent(model=flaky, model_retries=2, retry_backoff=0).run("hello")
    assert flaky.calls == 3  # 1 try + 2 retries


def test_non_transient_errors_do_not_retry():
    flaky = FlakyProvider(
        ScriptedProvider([ModelResponse(text="never")]), failures=1, exc=AuthError()
    )
    with pytest.raises(AuthError):
        Agent(model=flaky, model_retries=3, retry_backoff=0).run("hello")
    assert flaky.calls == 1


@tool
def slow() -> str:
    "Sleeps far longer than the cap."
    time.sleep(5)
    return "too late"


@tool
def quick() -> str:
    "Returns fast."
    return "fast"


def _tool_use(name):
    from loom.providers import ToolCall

    return ScriptedProvider(
        [
            ModelResponse(tool_calls=[ToolCall("t1", name, {})], stop_reason="tool_use"),
            ModelResponse(text="done"),
        ]
    )


def test_tool_timeout_records_an_error_result():
    agent = Agent(model=_tool_use("slow"), tools=[slow, quick], tool_timeout=0.2)
    start = time.time()
    run = agent.run("go")
    assert time.time() - start < 3  # did not wait out the sleep
    tool_effects = [e for e in run.log if e.kind == "tool:slow"]
    assert "ERROR: TimeoutError" in tool_effects[0].result
    assert "exceeded 0.2s" in tool_effects[0].result
    assert run.output == "done"  # the run continues; the model saw the error


def test_fast_tools_are_unaffected_by_the_cap():
    run = Agent(model=_tool_use("quick"), tools=[slow, quick], tool_timeout=5).run("go")
    assert [e.result for e in run.log if e.kind == "tool:quick"] == ["fast"]
