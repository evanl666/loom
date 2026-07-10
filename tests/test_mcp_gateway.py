"""MCP Gateway: firewall + record in front of an upstream MCP server.

Uses a fake upstream (no live MCP server needed) to test the screen/forward/
record core; the stdio re-serve is exercised separately with a real server.
"""
from loom import tool
from loom.mcp import MCPGateway
from loom.scan import scan
from loom.shield import Shield


@tool
def read_file(path: str) -> str:
    "Read a file."
    return f"contents of {path}"


@tool(capabilities={"write"})
def write_file(path: str, content: str) -> str:
    "Write a file."
    _RAN.append(path)
    return "written"


_RAN = []


class _FakeUpstream:
    def tools(self):
        return [read_file, write_file]

    def call(self, name, **kwargs):
        return {"read_file": read_file, "write_file": write_file}[name].fn(**kwargs)


def test_gateway_blocks_denied_call_and_records():
    _RAN.clear()
    gw = MCPGateway(_FakeUpstream(), shield=Shield(deny=["write_file*"]))
    assert "contents of x" in gw.call("read_file", path="x")  # allowed -> forwarded
    blocked = gw.call("write_file", path="y", content="z")
    assert blocked.startswith("BLOCKED")  # denied -> not forwarded
    assert _RAN == []  # the real write never ran
    decisions = [(c["tool"], c["decision"]) for c in gw.calls]
    assert decisions == [("read_file", "allow"), ("write_file", "deny")]


def test_gateway_traffic_is_a_scannable_trace():
    gw = MCPGateway(_FakeUpstream(), shield=Shield(deny=["write_file*"]))
    gw.call("read_file", path="x")
    gw.call("write_file", path="y", content="z")
    trace = gw.to_trace()
    assert trace["log"] and trace["tools"]
    rep = scan(trace)  # the recorded traffic analyzes like any loom trace
    assert {"read_file", "write_file"} <= {t["name"] for t in rep["tools"]}


def test_gateway_trust_score():
    gw = MCPGateway(_FakeUpstream())
    assert 0 <= gw.trust()["score"] <= 100
