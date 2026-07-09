"""The flight-recorder demo's patient: a deploy bot misled by a stale config.

The bug is genuine context rot. ``read_deploy_config()`` returns a huge,
14-months-stale config naming a database host that no longer exists; the bot
trusts it, health-checks the dead host, and declares the deploy failed. The
real host is one ``discover_services()`` call away -- the bot never makes it,
because the stale config already "answered" the question.

Offline by construction: the "model" is a RuleProvider that reacts to what's
in its context, exactly the kind of behavior heal() flips by redacting the
poisoned item and taking the fork.
"""

import sys

from loom import Agent, tool
from loom.providers import ModelResponse, RuleProvider, ToolCall

STALE_HOST = "db-legacy.internal"
LIVE_HOST = "db-prod-3.internal"

STALE_CONFIG = (
    "# deploy.toml -- last regenerated 14 months ago\n"
    f'[database]\nhost = "{STALE_HOST}"\nport = 5432\n\n# '
    + "compatibility shims, unused feature flags, dead service entries, "
    "commented-out rollback plans, historical notes nobody reads... " * 30
)


@tool
def read_deploy_config() -> str:
    "Read deploy.toml from the repo."
    return STALE_CONFIG


@tool
def discover_services() -> str:
    "Ask service discovery for the active database primary."
    return f"active primary: {LIVE_HOST} (healthy, 12 replicas)"


@tool
def check_database(host: str) -> str:
    "Health-check a database host."
    if host == LIVE_HOST:
        return f"{host}: ok (34ms, 12/12 replicas)"
    return f"{host}: UNREACHABLE (no such host)"


def _text(messages) -> str:
    return " ".join(str(m.get("content", "")) for m in messages)


def build_agent() -> Agent:
    def start(messages):
        if not any(m["role"] == "tool" for m in messages):
            return ModelResponse(
                text="Reading the deploy config first.",
                tool_calls=[ToolCall("t1", "read_deploy_config", {})],
                stop_reason="tool_use",
            )
        return None

    # The bot only believes deploy.toml while its text is actually in context;
    # heal's redaction removes it, and with it the bad conclusion.
    config_marker = f'host = "{STALE_HOST}"'

    def trust_stale_config(messages):
        t = _text(messages)
        if config_marker in t and "UNREACHABLE" not in t:
            return ModelResponse(
                text=f"Config says the database is {STALE_HOST}. Checking it.",
                tool_calls=[ToolCall("t2", "check_database", {"host": STALE_HOST})],
                stop_reason="tool_use",
            )
        return None

    def declare_failure(messages):
        t = _text(messages)
        if config_marker in t and "UNREACHABLE" in t:
            return ModelResponse(
                text=f"DEPLOY FAILED: database {STALE_HOST} is unreachable. Aborting."
            )
        return None

    def rediscover(messages):
        t = _text(messages)
        if "active primary" not in t:
            return ModelResponse(
                text="No trustworthy config in context -- asking service discovery.",
                tool_calls=[ToolCall("t3", "discover_services", {})],
                stop_reason="tool_use",
            )
        if "ok (" not in t:
            return ModelResponse(
                text=f"Discovery says {LIVE_HOST}. Verifying.",
                tool_calls=[ToolCall("t4", "check_database", {"host": LIVE_HOST})],
                stop_reason="tool_use",
            )
        return ModelResponse(text=f"Deploy is GREEN: {LIVE_HOST} healthy, 12/12 replicas.")

    return Agent(
        model=RuleProvider(rules=[start, trust_stale_config, declare_failure, rediscover]),
        tools=[read_deploy_config, discover_services, check_database],
        system="You are the release bot for checkout-service.",
    )


# `loom heal/impact --agent agent:build_agent` resolves this factory.
agent = build_agent


if __name__ == "__main__":
    save_to = sys.argv[1] if len(sys.argv) > 1 else "flight.loom.json"
    run = build_agent().run("Ship checkout-service to production.")
    run.save(save_to)
    print(f"bot: {run.output}")
    print(f"\nflight recording -> {save_to} "
          f"({run.num_turns} turns, {len(run.log)} effects)")
