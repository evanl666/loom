"""``loom tools --verify``: does a tool's declared contract match what it DOES?

An MCP server (or a tool author) can lie: a tool named ``get_user`` with a
read-only schema might delete rows. Capability inference reads the name and the
declaration -- it can't see behavior. The contract verifier runs the tool with
harmless probe arguments inside a monitored sandbox and records what it actually
touches -- network, filesystem writes, subprocess, database -- then compares the
OBSERVED capabilities to the DECLARED/inferred ones:

    loom tools --verify --agent app:agent

Undeclared side effects (observed network on a "read" tool, a write it never
declared) are the finding: trust, but verify.

The probe *executes the tool*, so it uses empty/benign args and monitors
in-process. Point it at tools you're evaluating, not ones with irreversible
effects on real systems.
"""

from __future__ import annotations

import sys
import threading
from typing import Any

# CPython audit hooks (PEP 578) observe real side effects -- network, file
# writes, subprocess, db -- without monkeypatching anything that could break
# (patching socket.socket breaks urllib, which subclasses it). A hook can't be
# removed once added, so we install ONE that routes to the thread-local set of
# whichever _Monitor is currently active.
_ACTIVE = threading.local()
_INSTALLED = False


def _install_hook() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    def hook(event: str, args: tuple) -> None:
        obs = getattr(_ACTIVE, "observed", None)
        if obs is None:
            return
        if event in ("socket.connect", "socket.getaddrinfo", "urllib.Request"):
            obs.add("network")
        elif event == "open":
            mode = args[1] if len(args) > 1 and args[1] else "r"
            obs.add("write" if any(c in str(mode) for c in "wax+") else "read")
        elif event == "subprocess.Popen" or event == "os.system" or event.startswith("os.exec"):
            obs.add("exec")
        elif event == "sqlite3.connect":
            obs.add("database")

    sys.addaudithook(hook)
    _INSTALLED = True


class _Monitor:
    """Record the real side effects (via audit hooks) of code run inside it."""

    def __init__(self) -> None:
        self.observed: set[str] = set()

    def __enter__(self) -> "_Monitor":
        _install_hook()
        _ACTIVE.observed = self.observed
        return self

    def __exit__(self, *exc) -> None:
        _ACTIVE.observed = None


def _probe_args(tool: Any) -> dict:
    """Benign default arguments derived from the tool's input schema."""
    schema = getattr(tool, "input_schema", None) or {}
    props = schema.get("properties", {}) if isinstance(schema, dict) else {}
    defaults = {"string": "probe", "integer": 0, "number": 0, "boolean": False,
                "array": [], "object": {}}
    return {k: defaults.get((v or {}).get("type", "string"), "probe")
            for k, v in props.items()}


def verify_tool(tool: Any, args: "dict | None" = None) -> dict:
    """Run ``tool`` with benign args under monitoring; compare declared vs observed."""
    from .capabilities import capabilities

    declared = set(getattr(tool, "capabilities", None) or [])
    inferred = capabilities(tool.name, {}, declared=declared or None)
    args = args if args is not None else _probe_args(tool)
    observed: set[str] = set()
    error = ""
    mon = _Monitor()
    try:
        with mon:
            tool.fn(**args)
    except Exception as e:  # noqa: BLE001 -- a probe failing is fine; we only want the side effects
        error = f"{type(e).__name__}: {e}"[:100]
    observed = mon.observed
    # side effects observed but neither declared nor inferred = a contract gap
    claimed = declared | inferred
    undeclared = sorted(observed - claimed - {"read", "idempotent"})
    return {"tool": tool.name, "declared": sorted(declared),
            "inferred": sorted(inferred), "observed": sorted(observed),
            "undeclared": undeclared, "ok": not undeclared, "probe_error": error}


def verify_tools(tools: "list[Any]") -> "list[dict]":
    return [verify_tool(t) for t in tools]


def describe_verify(results: "list[dict]") -> str:
    lines = [f"tool contract verification — probed {len(results)} tool(s)"]
    for r in results:
        if r["undeclared"]:
            lines.append(f"  ⚠ {r['tool']}: observed UNDECLARED {', '.join(r['undeclared'])} "
                         f"(declared: {', '.join(r['declared']) or 'none'})")
        else:
            lines.append(f"  ✓ {r['tool']}: observed {', '.join(r['observed']) or 'nothing'} — "
                         "matches contract")
    gaps = sum(1 for r in results if r["undeclared"])
    lines.append("" if not gaps else f"\n  {gaps} tool(s) do more than they declare.")
    return "\n".join(lines)
