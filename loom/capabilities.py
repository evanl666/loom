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
_READ_NAMES = ["read*", "*get*", "glob*", "grep*", "ls*", "list*", "cat*", "search*",
               "find*", "view*", "*fetch*", "*status*", "head*", "tail*", "stat*"]
_WRITE_NAMES = ["write*", "edit*", "*create*", "*update*", "*patch*", "*append*", "put*"]
_EXEC_NAMES = ["sh", "bash*", "zsh", "shell*", "*run*", "exec*", "*command*", "terminal*",
               "*execute*", "eval*", "python*", "node*"]
_NETWORK_NAMES = ["*http*", "*url*", "*download*", "curl*", "wget*", "*request*", "*api*",
                  "*webhook*", "*email*", "*slack*"]


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
    # A read that touches nothing external is idempotent; exec/write/network
    # are assumed to have side effects unless the tool declares otherwise.
    if caps <= {"read"}:
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
