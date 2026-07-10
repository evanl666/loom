"""Record a call to a REMOTE agent -- one you reach over HTTP/gRPC and can't
instrument from inside -- at its boundary.

Loom normally records at the model-API boundary (the proxy), which captures an
agent's every step *when that agent's LLM traffic routes through Loom*. A remote
agent on someone else's server is a black box: its internal model/tool calls
never reach you. What you CAN see is the boundary -- the request you send and the
response you get. ``RemoteAgent`` turns that boundary into a first-class Loom
Action, so even a black-box remote is:

  * **replayable**   -- the recorded response serves the call, no network;
  * **firewallable** -- a Shield/Policy can gate ``remote_<name>`` (or its
                        ``remote_agent`` / ``network`` capabilities) before it runs;
  * **taintable**    -- its result enters the trace, so value-lineage tracks a
                        secret flowing INTO or OUT OF the remote across a run.

Two ways to use it:

    ra = RemoteAgent("planner", call=grpc_or_http_client)  # call: (prompt)->str

    # (a) as a tool inside a normal loom.Agent -- full taint/firewall/replay:
    agent = Agent(model=..., tools=[ra.as_tool(), ...])

    # (b) record a single black-box call as a one-Action trace:
    trace = ra.record("plan the launch")            # dict, replayable/inspectable
"""

from __future__ import annotations

from typing import Any, Callable

# Declared capabilities every remote-agent call carries: it crosses the network
# to code you don't control, so it reads as a network + remote_agent action.
REMOTE_CAPS = ["network", "remote_agent"]


class RemoteAgent:
    def __init__(self, name: str, call: "Callable[[str], str]",
                 transport: str = "http", endpoint: str = ""):
        self.name = name
        self._call = call
        self.transport = transport
        self.endpoint = endpoint

    @property
    def tool_name(self) -> str:
        return f"remote_{self.name}"

    def as_tool(self):
        """A Loom tool that invokes the remote agent -- drop it into an Agent's
        ``tools=[...]`` and the remote call is recorded like any other tool."""
        from .tools import tool

        endpoint, transport, call, name = self.endpoint, self.transport, self._call, self.name

        @tool(name=self.tool_name, capabilities=set(REMOTE_CAPS))
        def _remote(prompt: str) -> str:
            return call(prompt)

        _remote.description = (f"Delegate to the remote '{name}' agent over "
                               f"{transport}{(' at ' + endpoint) if endpoint else ''}.")
        return _remote

    def call(self, prompt: str) -> str:
        """Invoke the remote agent directly (no recording)."""
        return self._call(prompt)

    def record(self, prompt: str, save: "str | None" = None) -> dict:
        """Invoke the remote agent and record the call as a ONE-Action Loom trace
        (a synthetic trigger + the remote tool call), so a black-box remote is
        still replayable / firewallable / taintable at the boundary.

        Returns the trace dict; writes it to ``save`` if given."""
        import json

        from .effect import Recorder

        rec = Recorder.record()
        result = rec.run(
            f"tool:{self.tool_name}", {"prompt": prompt}, lambda: self._call(prompt),
        )
        # A synthetic model "trigger" declaring the remote call, so actions()
        # pairs it into one call Action carrying the prompt as input.
        log = [
            {"seq": 0, "kind": "model", "depth": 0,
             "result": {"tool_calls": [{"id": "r0", "name": self.tool_name,
                                        "input": {"prompt": prompt}}],
                        "stop_reason": "tool_use"}},
            {"seq": 1, "kind": f"tool:{self.tool_name}", "depth": 0,
             "result": rec.log[0].result if rec.log else result},
            {"seq": 2, "kind": "model", "depth": 0,
             "result": {"text": str(result), "stop_reason": "end_turn"}},
        ]
        data = {
            "recorded_via": "remote",
            "transport": self.transport,
            "endpoint": self.endpoint,
            "prompt": prompt,
            "episodes": [prompt],
            "output": str(result),
            "model": f"remote:{self.name}",
            "tools": {self.tool_name: list(REMOTE_CAPS)},
            "log": log,
        }
        if save:
            with open(save, "w") as f:
                json.dump(data, f, indent=2)
        return data
