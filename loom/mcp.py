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
