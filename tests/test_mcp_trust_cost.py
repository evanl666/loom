"""MCP trust score + cost surgeon patches."""

from loom.cost import cost_patches
from loom.mcp import mcp_trust


def test_mcp_trust_scores_risky_below_safe():
    risky = mcp_trust([
        {"tool": "run_sql", "capabilities": ["database_write", "destructive"], "declared": False},
        {"tool": "exec_shell", "capabilities": ["exec"], "declared": False},
    ])
    safe = mcp_trust([
        {"tool": "read_file", "capabilities": ["read", "idempotent"], "declared": True},
    ])
    assert risky["score"] < safe["score"]
    assert safe["grade"] == "A" and risky["grade"] in ("C", "D", "F")
    assert "run_sql" in risky["risky"] and "exec_shell" in risky["risky"]


def test_cost_patches_computes_compaction_for_bloat():
    log = []
    for i, inp in enumerate([1000, 3000, 6000, 9000]):
        log.append({"seq": i * 2, "kind": "model",
                    "result": {"usage": {"input_tokens": inp, "output_tokens": 50}, "tool_calls": []}})
    data = {"log": log, "prompt": "p", "output": "done"}
    patches = cost_patches(data)
    assert any("compact_after" in p["patch"] for p in patches)
