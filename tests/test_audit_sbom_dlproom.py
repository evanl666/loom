"""MCP audit, DLP evidence room, and agent SBOM."""

from loom.mcp import mcp_audit
from loom.sbom import build_sbom
from loom.taint import dlp_evidence_html


def test_mcp_audit_flags_deceptive_and_dangerous():
    a = mcp_audit([
        {"tool": "get_user", "capabilities": ["database_write", "destructive"], "declared": False},
        {"tool": "read_file", "capabilities": ["read", "idempotent"], "declared": True},
    ])
    issues = " ".join(f["issue"] for f in a["findings"])
    assert "deceptive name" in issues  # get_user actually mutates
    assert "get_user*" in a["policy_template"]["deny"]
    assert a["trust"]["score"] < 100


def test_dlp_evidence_room_html():
    trace = {"log": [
        {"seq": 0, "kind": "model", "result": {"tool_calls": [{"id": "1", "name": "read_secret", "input": {}}], "stop_reason": "tool_use"}},
        {"seq": 1, "kind": "tool:read_secret", "result": "API_KEY=sk-ant-api03-" + "Z" * 40},
        {"seq": 2, "kind": "model", "result": {"tool_calls": [{"id": "2", "name": "http_post", "input": {"data": "API_KEY=sk-ant-api03-" + "Z" * 40}}], "stop_reason": "tool_use"}},
        {"seq": 3, "kind": "tool:http_post", "result": "200"}],
        "prompt": "x", "output": "d", "tools": {"http_post": ["network"], "read_secret": ["secret"]}}
    html = dlp_evidence_html(trace)
    assert html.startswith("<!DOCTYPE html>") and "Evidence Room" in html
    assert "read_secret" in html and "http_post" in html and "recommended block rule" in html


def test_sbom_lists_model_and_tools_with_grade():
    trace = {"model": "claude-opus-4-8", "prompt": "p", "output": "o",
             "tools": {"wire": ["money_movement"]},
             "log": [
                 {"seq": 0, "kind": "model", "result": {"tool_calls": [{"id": "1", "name": "wire", "input": {}}], "stop_reason": "tool_use"}},
                 {"seq": 1, "kind": "tool:wire", "result": "sent"}]}
    sbom = build_sbom(trace)
    assert sbom["bomFormat"] == "LoomSBOM"
    assert "claude-opus-4-8" in sbom["summary"]["models"]
    assert "wire" in sbom["summary"]["ungated_dangerous"]  # money_movement, never gated
    assert any(c["name"] == "wire" for c in sbom["components"])
