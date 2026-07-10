"""MCP: plug Model Context Protocol servers into the harness.

    from loom.mcp import MCPServer

    with MCPServer("npx", ["-y", "@modelcontextprotocol/server-filesystem", "."]) as fs:
        agent = Agent(model="claude-opus-4-8", tools=fs.tools())
        run = agent.run("What files are here?")

Every MCP tool is wrapped as an ordinary loom ``Tool``, so its calls flow
through the Effect boundary like any other tool: recorded on the way out,
served from the log on replay. A trace recorded against a live MCP server
replays byte-identically with the server gone -- ``verify_replay`` in CI
needs no MCP processes at all.

Requires the optional extra:  pip install "loom-harness[mcp]"
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

from .tools import Tool


def mcp_manifest(tools: "list[Tool]") -> "list[dict]":
    """A capability + risk manifest for a set of (MCP) tools.

    The MCP gateway's core value: before you let an agent reach a server's
    tools, see what each one CAN DO -- its capabilities (read/write/exec/
    network/secret/... plus business classes), its risk category, whether it
    declared its own capabilities, and its input schema. This is what a policy
    is written against.
    """
    from .capabilities import capabilities
    from .risk import classify

    out = []
    for t in tools:
        declared = getattr(t, "capabilities", None)
        caps = sorted(capabilities(t.name, {}, declared=declared))
        out.append({
            "tool": t.name,
            "capabilities": caps,
            "risk": classify(t.name, {}),
            "declared": bool(declared),
            "schema": getattr(t, "input_schema", {}),
        })
    return out


def mcp_trust(manifest: "list[dict]") -> dict:
    """A 0-100 trust score for an MCP server's tool surface, from its manifest.

    "Should I install this server?" -- before you let an agent reach it. Lower =
    riskier: dangerous capabilities (money/destructive/db-write/browser-submit),
    broad reach (exec/network), secret/PII access, and tools that didn't declare
    their own capabilities (opaque). Returns {score, grade, factors, risky}.
    """
    weights = {
        "money_movement": 16, "destructive": 16, "database_write": 12,
        "browser_submit": 10, "exec": 12, "secret": 9, "pii_access": 8,
        "network": 5, "user_communication": 6, "write": 3,
    }
    factors: dict[str, int] = {}
    undeclared = 0
    for m in manifest:
        caps = set(m.get("capabilities") or [])
        for cap, w in weights.items():
            if cap in caps:
                factors[cap] = factors.get(cap, 0) + w
        # a tool with real reach that didn't declare its contract is opaque
        if not m.get("declared") and (caps - {"idempotent"}):
            undeclared += 1
    if undeclared:
        factors["undeclared-tools"] = min(20, undeclared * 4)
    score = max(0, 100 - sum(factors.values()))
    grade = ("A" if score >= 90 else "B" if score >= 75 else "C" if score >= 55
             else "D" if score >= 35 else "F")
    risky = sorted(m["tool"] for m in manifest
                   if set(m.get("capabilities") or []) &
                   {"money_movement", "destructive", "database_write", "browser_submit", "exec"})
    return {"score": score, "grade": grade, "factors": factors,
            "tools": len(manifest), "risky": risky}


def mcp_audit(manifest: "list[dict]") -> dict:
    """`npm audit` for an MCP server: trust score + the specific reasons.

    Flags dangerous tools (money/destructive/db-write/exec), DECEPTIVELY NAMED
    tools (a read-sounding name whose capabilities say it writes/deletes -- the
    classic 'get_user that deletes data'), and undeclared-capability tools, then
    emits a ready-to-use deny/confirm policy template. The 'should I install
    this?' verdict.
    """
    trust = mcp_trust(manifest)
    dangerous = {"money_movement", "destructive", "database_write", "browser_submit", "exec"}
    read_words = ("get", "read", "list", "fetch", "search", "find", "view", "show", "lookup")
    findings: list[dict] = []
    deny: list[str] = []
    confirm: list[str] = []
    for m in manifest:
        name = m["tool"]
        caps = set(m.get("capabilities") or [])
        hits = caps & dangerous
        # deceptive naming: sounds read-only, actually mutates
        looks_read = any(name.lower().startswith(w) or f"_{w}" in name.lower() for w in read_words)
        mutates = caps & {"write", "destructive", "database_write", "money_movement"}
        if looks_read and mutates:
            findings.append({"severity": "high", "tool": name,
                             "issue": f"deceptive name: reads-like '{name}' but has "
                                      f"{', '.join(sorted(mutates))}"})
            deny.append(f"{name}*")
        elif hits:
            findings.append({"severity": "high" if hits & {"money_movement", "destructive"} else "medium",
                             "tool": name, "issue": f"dangerous capability: {', '.join(sorted(hits))}"})
            (deny if hits & {"destructive", "money_movement"} else confirm).append(f"{name}*")
        elif not m.get("declared") and (caps - {"idempotent", "read"}):
            findings.append({"severity": "low", "tool": name,
                             "issue": "undeclared capabilities (opaque -- trust unverified)"})
    verdict = ("avoid -- high-risk tools, write a policy first" if trust["score"] < 45
               else "review -- gate the flagged tools before use" if findings
               else "ok -- no dangerous tools found")
    return {"trust": trust, "findings": findings, "verdict": verdict,
            "policy_template": {"default": "allow", "deny": deny, "confirm": confirm}}


def describe_mcp_audit(audit: dict) -> str:
    t = audit["trust"]
    lines = [f"MCP audit — trust {t['score']}/100 ({t['grade']}) — {audit['verdict']}", ""]
    icon = {"high": "🔴", "medium": "🟠", "low": "🟡"}
    for f in audit["findings"]:
        lines.append(f"  {icon.get(f['severity'], '·')} {f['tool']}: {f['issue']}")
    if not audit["findings"]:
        lines.append("  ✓ no findings")
    pt = audit["policy_template"]
    if pt["deny"] or pt["confirm"]:
        lines += ["", "  suggested policy:"]
        if pt["deny"]:
            lines.append(f"    deny:    {', '.join(pt['deny'])}")
        if pt["confirm"]:
            lines.append(f"    confirm: {', '.join(pt['confirm'])}")
        lines.append(f"    → loom mcp gateway {' '.join('--deny ' + g for g in pt['deny'])} -- <server>")
    return "\n".join(lines)


def describe_mcp_trust(trust: dict) -> str:
    top = sorted(trust["factors"].items(), key=lambda kv: -kv[1])[:5]
    reasons = ", ".join(f"{k} -{v}" for k, v in top) or "clean surface"
    line = (f"trust score: {trust['score']}/100 (grade {trust['grade']}) "
            f"over {trust['tools']} tool(s)\n  deductions: {reasons}")
    if trust["risky"]:
        line += f"\n  ⚠ high-reach tools: {', '.join(trust['risky'])}"
    return line


def guarded_tools(tools: "list[Tool]", shield) -> "list[Tool]":
    """Wrap tools so every call is screened by ``shield`` before it runs.

    A denied call never reaches the server -- the tool returns a BLOCKED
    result the model sees, exactly like Shield at the proxy, so MCP tools used
    directly (not just via the recording proxy) are still firewalled. Confirm
    rules here fall back to deny (there is no human in a direct call path).
    """
    from dataclasses import replace

    wrapped = []
    for t in tools:
        inner = t.fn

        def screened(_inner=inner, _name=t.name, **kwargs):
            action, rule = shield.classify(_name, kwargs)
            if action in ("deny", "confirm"):
                return (f"BLOCKED: {_name} was not run -- firewall {action}"
                        + (f" (rule: {rule})" if rule else ""))
            return _inner(**kwargs)

        wrapped.append(replace(t, fn=screened))
    return wrapped


class MCPGateway:
    """A firewall + black-box recorder in front of an upstream MCP server.

    The MCP equivalent of ``loom proxy``: every ``tools/call`` is screened by a
    Shield (deny/confirm block it before it runs), forwarded to the real server,
    and recorded -- so the traffic replays, taints, and scans like any other
    Loom trace. Point a *loom agent* at ``gateway.guarded_tools()``, or re-serve
    it to Claude Desktop / Cursor over stdio with ``serve_stdio()`` (a drop-in
    firewalled MCP endpoint).
    """

    def __init__(self, upstream: "MCPServer", shield: Any = None, save_path: str = ""):
        self.upstream = upstream
        self.shield = shield
        self.save_path = save_path
        self.calls: list[dict] = []  # {tool, input, decision, rule, result}

    def manifest(self) -> "list[dict]":
        return mcp_manifest(self.upstream.tools())

    def trust(self) -> dict:
        return mcp_trust(self.manifest())

    def list_tools(self) -> "list[Tool]":
        return self.upstream.tools()

    def call(self, name: str, **kwargs: Any) -> str:
        """Screen, forward, record. A denied/confirmed call never reaches upstream."""
        decision, rule = "allow", ""
        if self.shield is not None:
            decision, rule = self.shield.classify(name, kwargs)
        if decision in ("deny", "confirm"):
            result = (f"BLOCKED: {name} not run -- MCP gateway firewall {decision}"
                      + (f" (rule: {rule})" if rule else ""))
            self._record(name, kwargs, decision, rule, result)
            return result
        result = self.upstream.call(name, **kwargs)
        self._record(name, kwargs, "allow", rule, result)
        return result

    def _record(self, name: str, args: dict, decision: str, rule: str, result: str) -> None:
        self.calls.append({"tool": name, "input": args, "decision": decision,
                           "rule": rule, "result": result})
        if self.save_path:
            self.save(self.save_path)

    def to_trace(self) -> dict:
        """The recorded traffic as a loom trace (so taint / scan / replay work)."""
        log: list[dict] = []
        seq = 0
        for c in self.calls:
            log.append({"seq": seq, "kind": "model", "key": f"g{seq}", "result": {
                "text": "", "tool_calls": [{"id": f"c{seq}", "name": c["tool"], "input": c["input"]}],
                "stop_reason": "tool_use", "usage": {}}})
            seq += 1
            log.append({"seq": seq, "kind": f"tool:{c['tool']}", "key": f"g{seq}",
                        "result": c["result"]})
            seq += 1
        caps = {m["tool"]: m["capabilities"] for m in self.manifest()}
        shield_events = [{"tool": c["tool"], "input": c["input"], "action": "deny",
                          "rule": c["rule"], "via": "gateway"}
                         for c in self.calls if c["decision"] in ("deny", "confirm")]
        return {"version": 2, "prompt": "MCP gateway session", "output": "",
                "episodes": ["MCP gateway session"], "log": log, "tools": caps,
                "stop_reason": "end_turn", "shield_events": shield_events}

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_trace(), f, indent=2)

    def guarded_tools(self) -> "list[Tool]":
        """Upstream tools, each routed through this gateway (screen + record)."""
        from dataclasses import replace

        out = []
        for t in self.upstream.tools():
            out.append(replace(t, fn=lambda _n=t.name, **kw: self.call(_n, **kw)))
        return out

    def serve_stdio(self, name: str = "loom-mcp-gateway") -> None:
        """Re-expose the upstream (firewalled + recorded) as an MCP server on
        stdio, so any MCP client can use the guarded endpoint. Blocks."""
        import anyio
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import TextContent, Tool as MCPTool

        server: Server = Server(name)
        tools = self.upstream.tools()

        @server.list_tools()
        async def _list():  # noqa: ANN202
            return [MCPTool(name=t.name, description=t.description,
                            inputSchema=getattr(t, "input_schema", None)
                            or {"type": "object", "properties": {}}) for t in tools]

        @server.call_tool()
        async def _call(tool_name: str, arguments: dict):  # noqa: ANN202
            result = await anyio.to_thread.run_sync(lambda: self.call(tool_name, **(arguments or {})))
            return [TextContent(type="text", text=str(result))]

        async def _run():
            async with stdio_server() as (r, w):
                await server.run(r, w, server.create_initialization_options())

        anyio.run(_run)


def _require_mcp():
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError:  # pragma: no cover
        raise ImportError(
            "MCP support needs the 'mcp' package. "
            'Install it with: pip install "loom-harness[mcp]"'
        ) from None
    return ClientSession, StdioServerParameters, stdio_client


class MCPServer:
    """A connection to one MCP server (stdio transport), usable as a context manager.

    The MCP SDK is async; loom's loop is sync. The connection lives on a
    dedicated background event loop, and tool calls block until the server
    answers (``timeout`` seconds, default 60).
    """

    def __init__(
        self,
        command: str,
        args: "list[str] | None" = None,
        env: "dict[str, str] | None" = None,
        timeout: float = 60.0,
    ):
        ClientSession, StdioServerParameters, stdio_client = _require_mcp()
        self.timeout = timeout
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._session: Any = None

        async def connect():
            params = StdioServerParameters(command=command, args=args or [], env=env)
            self._stdio_cm = stdio_client(params)
            read, write = await self._stdio_cm.__aenter__()
            self._session_cm = ClientSession(read, write)
            self._session = await self._session_cm.__aenter__()
            await self._session.initialize()

        self._await(connect())

    def _await(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(self.timeout)

    # -- the public surface -------------------------------------------------

    def tools(self) -> list[Tool]:
        """The server's tools, wrapped as ordinary loom Tools."""
        listed = self._await(self._session.list_tools())
        return [
            Tool(
                name=t.name,
                description=t.description or "",
                fn=self._make_fn(t.name),
                input_schema=t.inputSchema or {"type": "object", "properties": {}},
            )
            for t in listed.tools
        ]

    def manifest(self) -> "list[dict]":
        """The capability + risk manifest for this server's tools."""
        return mcp_manifest(self.tools())

    def guarded_tools(self, shield) -> "list[Tool]":
        """This server's tools, each screened by ``shield`` before it runs."""
        return guarded_tools(self.tools(), shield)

    def call(self, name: str, **kwargs: Any) -> str:
        """Call one tool synchronously, flattening the reply to text."""
        result = self._await(self._session.call_tool(name, kwargs))
        parts = []
        for block in result.content:
            text = getattr(block, "text", None)
            parts.append(text if text is not None else json.dumps(block.__dict__, default=str))
        text = "\n".join(parts)
        if getattr(result, "isError", False):
            return f"MCP ERROR: {text}"
        return text

    def close(self) -> None:
        """Shut down the server connection and the background loop."""
        if self._loop.is_closed():
            return

        async def teardown():
            if self._session is not None:
                await self._session_cm.__aexit__(None, None, None)
            await self._stdio_cm.__aexit__(None, None, None)

        try:
            self._await(teardown())
        except Exception:
            pass  # the subprocess may already be gone; closing must not raise
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        self._loop.close()

    def _make_fn(self, name: str):
        def fn(**kwargs: Any) -> str:
            return self.call(name, **kwargs)

        return fn

    def __enter__(self) -> "MCPServer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
