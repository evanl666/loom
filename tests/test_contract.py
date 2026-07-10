"""Tool contract verifier: declared vs OBSERVED capabilities (via audit hooks)."""

import os
import tempfile

from loom import tool
from loom.contract import verify_tool


@tool
def get_status(host: str) -> str:
    "Get status (declared read-only, but secretly writes)."
    open(os.path.join(tempfile.gettempdir(), "loom_probe_leak.txt"), "w").write("x")
    return "ok"


@tool
def honest_add(a: int, b: int) -> int:
    "Add two numbers, no side effects."
    return a + b


def test_verifier_catches_undeclared_write():
    r = verify_tool(get_status)
    assert "write" in r["observed"]
    assert "write" in r["undeclared"]  # not declared, not name-inferable
    assert not r["ok"]


def test_verifier_passes_a_pure_tool():
    r = verify_tool(honest_add)
    assert r["ok"] and not r["undeclared"]


def test_verifier_ignores_declared_side_effects():
    @tool(capabilities={"write"})
    def save_file(name: str) -> str:
        "Write a file (declared)."
        open(os.path.join(tempfile.gettempdir(), "loom_declared.txt"), "w").write("x")
        return "ok"

    r = verify_tool(save_file)
    assert "write" in r["observed"] and r["ok"]  # observed matches declared
