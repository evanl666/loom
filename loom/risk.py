"""A small, offline risk taxonomy for tool actions -- of any agent, not just coding.

One shared classifier so the incident report, the impact Action, and the
trace lake all speak the same language about *what kind* of dangerous an
action is -- not just "a tool ran" but "it read credentials", "it could send
data off the box", "it moved money", or "it wrote to the database". Pattern-
based, no model, deliberately conservative.

Two tiers of category:

  Infrastructure (coding / shell / fs), most-severe first:
    secret-read      reading credentials/keys (.env, ~/.ssh, *secret*)
    network-egress   sending anywhere off the machine (WebFetch, curl, wget)
    code-exec        running arbitrary code (Bash/Shell/Exec, download|sh)
    fs-destructive   deleting or overwriting (rm, mkfs, truncating redirects)
    fs-write         creating/editing files (Write, Edit)

  Business (data / browser / support / commerce agents):
    money-movement   refunds, payments, transfers, charges
    pii-access       reading personal data (customer/patient records, SSNs)
    user-comm        messaging real users (email, SMS, ticket replies)
    db-write         writing/updating rows (SQL INSERT/UPDATE/UPSERT)
    browser-submit   submitting forms / clicking through a browser agent

These map onto generic capabilities in ``loom.capabilities`` (money_movement,
pii_access, database_write, ...), so a firewall can gate business risk the
same way it gates a shell -- ``--confirm cap:money_movement``.

``classify`` takes an action (it can see the arguments, so it's precise);
``categories_for_names`` is the coarser name-only version for when only tool
names are known (the impact tool-inventory diff).
"""

from __future__ import annotations

from fnmatch import fnmatchcase as fnmatch

# (category, [name-globs], [signature-globs]). Order = severity, high first.
# Globs are token-anchored to avoid false positives (see loom.capabilities).
_RULES: "list[tuple[str, list[str], list[str]]]" = [
    ("money-movement",
     ["refund*", "*_refund", "issue_refund*", "*payout*", "create_charge*",
      "charge_card*", "capture_payment*", "*create_payment*", "transfer_funds*",
      "*wire_transfer*", "process_payment*"],
     ["*(*create_refund*)", "*(*capture_payment*)"]),
    ("secret-read",
     ["*secret*", "*credential*"],
     ["*(*.env*)", "*(*/.ssh/*)", "*(*/.aws/*)", "*(*secret*)", "*(*credential*)",
      "*(*/.netrc*)", "*(*id_rsa*)", "*(*.pem*)"]),
    ("pii-access",
     ["get_customer*", "*customer_record*", "*patient*", "get_profile*",
      "*personal_data*", "*get_pii*", "lookup_user*", "*ssn*", "*passport*"],
     ["*(*social_security*)", "*(*date_of_birth*)", "*(*ssn*)"]),
    ("network-egress",
     ["WebFetch*", "*fetch*", "*http*request*", "*upload*"],
     ["*(*curl *)", "*(*wget *)", "*(*http://*)", "*(*https://*)", "*(*nc -*)",
      "*(*scp *)", "*(*rsync *)"]),
    ("user-comm",
     ["send_email*", "*send_message*", "send_sms*", "reply_to*", "notify_customer*",
      "notify_user*", "post_message*", "*send_notification*", "email_customer*"],
     []),
    ("code-exec",
     ["Bash*", "Shell*", "Exec*", "*run_shell*", "*run_command*", "*execute*"],
     ["*(*| sh*)", "*(*| bash*)", "*(*eval *)", "*(*python -c*)"]),
    ("db-write",
     ["sql_insert*", "sql_update*", "db_insert*", "db_update*", "insert_row*",
      "update_record*", "upsert*"],
     ["*(*INSERT INTO*)", "*(*insert into*)", "*(*UPDATE *SET*)", "*(*update *set*)",
      "*(*UPSERT*)"]),
    ("browser-submit",
     ["click*", "*_click", "submit*", "fill_form*", "type_text*", "press_button*",
      "*form_submit*"],
     []),
    ("fs-destructive",
     ["Delete*", "*remove*", "*rmtree*"],
     ["*(*rm -rf*)", "*(*rm -f*)", "*(* rm *)", "*(*mkfs*)", "*(*> /*)", "*(*truncate*)",
      "*(*git push --force*)", "*(*git reset --hard*)", "*(*DROP *)", "*(*DELETE FROM*)"]),
    ("fs-write",
     ["Write*", "Edit*", "*write_file*", "*create_file*", "*patch*"],
     ["*(*>> *)", "*(*tee *)"]),
]

# Capabilities worth flagging when a change GRANTS one (impact tool-diff):
# gaining shell, file-write, money movement, or a DB write is a real increase.
DANGEROUS = {"secret-read", "network-egress", "code-exec", "fs-destructive",
             "money-movement", "pii-access", "user-comm", "db-write", "browser-submit"}

# The subset that is alarming when merely EXERCISED in a run (incident
# severity): running pytest via a shell is code-exec but not an incident;
# reading a secret, reaching the network, moving money, touching PII, writing
# the DB, or messaging a real user is.
ALARMING = {"secret-read", "network-egress", "fs-destructive",
            "money-movement", "pii-access", "user-comm", "db-write"}


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
        "money-movement": "confirm 'cap:money_movement'  (never auto-run refunds/payments)",
        "pii-access": "confirm 'cap:pii_access'  (gate reads of customer/patient data)",
        "user-comm": "confirm 'cap:user_communication'  (approve outbound messages to users)",
        "db-write": "confirm 'cap:database_write'  (dry-run writes first)",
        "browser-submit": "confirm 'cap:browser_submit'  (approve form submissions)",
    }.get(category, "")
