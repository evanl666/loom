"""MCP gateway: capability manifest + shield-guarded tools (no live server)."""

from loom import tool
from loom.mcp import guarded_tools, mcp_manifest
from loom.shield import Shield


@tool(capabilities={"network"})
def send_webhook(url: str) -> str:
    "post to a url"
    return "sent"


@tool
def read_file(path: str) -> str:
    "read a file"
    return "contents"


@tool
def run_shell(cmd: str) -> str:
    "run a shell command"
    return "ok"


def test_manifest_classifies_each_tool():
    m = {r["tool"]: r for r in mcp_manifest([send_webhook, read_file, run_shell])}
    assert m["send_webhook"]["declared"] is True
    assert "network" in m["send_webhook"]["capabilities"]
    assert "read" in m["read_file"]["capabilities"]
    assert "exec" in m["run_shell"]["capabilities"]  # inferred from the name shape
    assert all("schema" in r for r in m.values())


def test_guarded_tools_block_denied_calls():
    shield = Shield(deny=["cap:exec"])
    guarded = {t.name: t for t in guarded_tools([read_file, run_shell], shield)}
    # a denied exec tool returns a BLOCKED result instead of running
    assert guarded["run_shell"].fn(cmd="rm -rf /").startswith("BLOCKED")
    # an allowed tool runs normally
    assert guarded["read_file"].fn(path="a.py") == "contents"


def test_guarded_confirm_falls_back_to_block():
    shield = Shield(confirm=["cap:network"])
    g = {t.name: t for t in guarded_tools([send_webhook], shield)}
    assert g["send_webhook"].fn(url="http://x").startswith("BLOCKED")
