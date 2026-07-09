"""Tool capability contracts: judge tools by what they DO, not what they're named.

A firewall rule like ``deny 'Bash*'`` is brittle -- the next agent calls its
shell tool ``run_command`` or ``sh`` and sails through. What you actually mean
is "deny anything that can execute code". Capabilities make that expressible:

    read          reads files / inspects state
    write         creates or edits files
    exec          runs arbitrary code (shell)
    network       reaches off the machine
    secret        reads credentials / keys
    destructive   deletes or overwrites irreversibly
    idempotent    safe to run more than once (no external side effect)

A tool can declare its own capabilities (``@tool(capabilities={"network"})``
or ``Tool(..., capabilities=...)``); otherwise they're inferred from the call
the same way ``risk`` classifies it, plus name heuristics. Shield patterns
that start with ``cap:`` match on capability, so ``--deny 'cap:exec'`` blocks
every shell-shaped tool regardless of its name.
"""

from __future__ import annotations

from fnmatch import fnmatchcase as fnmatch

# Risk categories map onto capability names.
_FROM_RISK = {
    "secret-read": {"read", "secret"},
    "network-egress": {"network"},
    "code-exec": {"exec"},
    "fs-destructive": {"write", "destructive"},
    "fs-write": {"write"},
}

# Name hints beyond what risk's (display-tuned) globs cover.
# Name hints are token-anchored (start*, *_token, *_end) rather than bare
# *substring* to avoid false positives -- '*run*' would flag 'prune'/'truncate',
# '*api*' would flag 'capital'. Declared capabilities are the reliable path;
# this is a conservative fallback.
_READ_NAMES = ["read*", "get_*", "*_get", "glob*", "grep*", "ls", "list_*", "*_list",
               "cat", "search*", "find_*", "view*", "*_fetch", "fetch_*", "*status*",
               "head", "tail", "stat"]
_WRITE_NAMES = ["write*", "edit*", "create_*", "*_create", "update_*", "*_update",
                "*patch*", "append_*", "*_append", "put_*"]
_EXEC_NAMES = ["sh", "bash*", "zsh", "shell*", "*shell*", "run", "run_*", "*_run",
               "exec*", "*_exec*", "*command*", "terminal*", "*execute*", "python*", "node*"]
_NETWORK_NAMES = ["*http*", "*_url", "url_*", "*download*", "curl*", "wget*", "*request*",
                  "api_*", "*_api", "fetch*", "*webhook*", "*email*", "*slack*"]


def capabilities(name: str, tool_input=None, declared: "set[str] | None" = None) -> "set[str]":
    """The capability set of a tool call. Declared capabilities win; otherwise
    inferred from risk classification + name heuristics."""
    if declared:
        return set(declared)
    from .risk import classify_all

    caps: set[str] = set()
    for cat in classify_all(name, tool_input or {}):
        caps |= _FROM_RISK.get(cat, set())
    lname = name.lower()
    if any(fnmatch(lname, p) for p in _READ_NAMES):
        caps.add("read")
    if any(fnmatch(lname, p) for p in _WRITE_NAMES):
        caps.add("write")
    if any(fnmatch(lname, p) for p in _EXEC_NAMES):
        caps.add("exec")
    if any(fnmatch(lname, p) for p in _NETWORK_NAMES):
        caps.add("network")
    # A CONFIRMED read-only tool is idempotent. A tool we couldn't classify at
    # all (empty caps) is *unknown*, not safe -- don't claim it's idempotent
    # (e.g. 'prune_logs' matches nothing but is destructive).
    if caps == {"read"}:
        caps.add("idempotent")
    return caps


def matches_cap(pattern: str, name: str, tool_input=None,
                declared: "set[str] | None" = None) -> bool:
    """Does ``cap:<capability>`` match this call? (``pattern`` includes 'cap:')."""
    if not pattern.startswith("cap:"):
        return False
    wanted = pattern[4:].strip()
    return wanted in capabilities(name, tool_input, declared=declared)


def manifest(tools) -> "list[dict]":
    """A capability manifest for a set of tools (name -> declared/inferred caps)."""
    out = []
    items = tools.values() if hasattr(tools, "values") else tools
    for t in items:
        declared = getattr(t, "capabilities", None)
        caps = sorted(capabilities(t.name, {}, declared=declared))
        out.append({"tool": t.name, "capabilities": caps,
                    "declared": bool(declared)})
    return out
