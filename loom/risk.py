"""A small, offline risk taxonomy for tool calls.

One shared classifier so the incident report, the impact Action, and the
trace lake all speak the same language about *what kind* of dangerous a call
is -- not just "a tool ran" but "a tool read credentials" or "a tool could
send data off the box". Pattern-based, no model, deliberately conservative.

Categories, most-severe first:

  secret-read      reading credentials/keys (.env, ~/.ssh, *secret*)
  network-egress   sending anywhere off the machine (WebFetch, curl, wget)
  code-exec        running arbitrary code (Bash/Shell/Exec, download|sh)
  fs-destructive   deleting or overwriting (rm, mkfs, truncating redirects)
  fs-write         creating/editing files (Write, Edit)

``classify`` takes a call (it can see the arguments, so it's precise);
``categories_for_names`` is the coarser name-only version for when only tool
names are known (the impact tool-inventory diff).
"""

from __future__ import annotations

from fnmatch import fnmatchcase as fnmatch

# (category, [name-globs], [signature-globs]). Order = severity, high first.
_RULES: "list[tuple[str, list[str], list[str]]]" = [
    ("secret-read",
     ["*secret*", "*credential*"],
     ["*(*.env*)", "*(*/.ssh/*)", "*(*/.aws/*)", "*(*secret*)", "*(*credential*)",
      "*(*/.netrc*)", "*(*id_rsa*)", "*(*.pem*)"]),
    ("network-egress",
     ["WebFetch*", "*fetch*", "*http*request*", "*send_email*", "*upload*"],
     ["*(*curl *)", "*(*wget *)", "*(*http://*)", "*(*https://*)", "*(*nc -*)",
      "*(*scp *)", "*(*rsync *)"]),
    ("code-exec",
     ["Bash*", "Shell*", "Exec*", "*run_shell*", "*run_command*", "*execute*"],
     ["*(*| sh*)", "*(*| bash*)", "*(*eval *)", "*(*python -c*)"]),
    ("fs-destructive",
     ["Delete*", "*remove*", "*rmtree*"],
     ["*(*rm -rf*)", "*(*rm -f*)", "*(* rm *)", "*(*mkfs*)", "*(*> /*)", "*(*truncate*)",
      "*(*git push --force*)", "*(*git reset --hard*)", "*(*DROP *)", "*(*DELETE FROM*)"]),
    ("fs-write",
     ["Write*", "Edit*", "*write_file*", "*create_file*", "*patch*"],
     ["*(*>> *)", "*(*tee *)"]),
]

# Capabilities worth flagging when a change GRANTS one (impact tool-diff):
# gaining shell or file-write is a real capability increase.
DANGEROUS = {"secret-read", "network-egress", "code-exec", "fs-destructive"}

# The subset that is alarming when merely EXERCISED in a run (incident
# severity): running pytest via a shell is code-exec but not an incident;
# reading a secret or reaching the network is.
ALARMING = {"secret-read", "network-egress", "fs-destructive"}


def _sig(name: str, tool_input) -> str:
    import json

    try:
        return f"{name}({json.dumps(tool_input, sort_keys=True, default=str)})"
    except (TypeError, ValueError):
        return f"{name}({tool_input})"


def classify_all(name: str, tool_input=None) -> "list[str]":
    """Every risk category a call matches, most-severe first. A curl that
    reads a .env is both secret-read AND network-egress -- callers judging
    exfiltration need to see both."""
    sig = _sig(name, tool_input or {})
    hits = []
    for category, name_globs, sig_globs in _RULES:
        if any(fnmatch(name, g) for g in name_globs) or any(fnmatch(sig, g) for g in sig_globs):
            hits.append(category)
    return hits


def classify(name: str, tool_input=None) -> str:
    """The single most-severe risk category a call matches, or '' for none."""
    hits = classify_all(name, tool_input)
    return hits[0] if hits else ""


def categories_for_names(names: "list[str]") -> "set[str]":
    """Coarse name-only classification (no arguments available)."""
    out = set()
    for n in names:
        cat = classify(n, {})
        if cat:
            out.add(cat)
    return out


def recommended_rule(category: str) -> str:
    """A Shield rule that would have gated a call of this category."""
    return {
        "secret-read": "deny 'Read(*.env*)'  (and add: --rule 'taint sk-*: confirm *')",
        "network-egress": "confirm 'WebFetch*' --confirm 'Bash(*curl*)'",
        "code-exec": "confirm 'Bash(*)'  (or --profile claude-code-safe)",
        "fs-destructive": "deny 'Bash(*rm -rf*)'",
        "fs-write": "confirm 'Write(*)'",
    }.get(category, "")
