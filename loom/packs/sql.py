"""The Data/SQL Pack -- debug agents that talk to databases.

Opt-in (``import loom.packs.sql`` registers it). What it teaches Loom:

  owns        tools shaped like a database client, or any call whose input
              carries a SQL statement
  capabilities  parsed from the statement itself: SELECT reads, INSERT/UPDATE
              write, DELETE/DROP/TRUNCATE are destructive -- and a query that
              touches PII-shaped columns (ssn, email, date_of_birth...) is
              flagged ``pii_access`` even when it's "just a SELECT"
  state_diff  "UPDATE on orders" / "+N rows" when the result reports a count
  undo        a compensating statement where one exists (INSERT -> DELETE);
              an honest "restore from backup" plan for DELETE/DROP, because
              pretending a dropped table is revertible helps nobody

The pack never connects to a database: it reads the recorded trace only.
"""

from __future__ import annotations

import re
from fnmatch import fnmatchcase as fnmatch

from ..action import Action, StateDiff
from . import Pack, UndoPlan, register

_TOOL_NAMES = ["*sql*", "*query*", "db_*", "*_db", "*database*", "execute_query*"]

# Input keys likely to carry a SQL statement.
_SQL_KEYS = ("query", "sql", "statement", "command")

_PII_COLUMNS = re.compile(
    r"\b(ssn|social_security|date_of_birth|dob|passport|email|phone|address|"
    r"credit_card|card_number|salary|diagnosis)\b", re.I)

_ROWCOUNT = re.compile(r"\b(\d+)\s+rows?\b", re.I)


_SQL_SHAPE = re.compile(
    r"^\s*(select|insert|update|delete|drop|truncate|alter|create|upsert)\b", re.I)


def _statement(action_input) -> str:
    """The SQL statement in a tool input -- only if it actually LOOKS like SQL.

    'command' is a key shells use too; a `rm -rf` must not read as a query."""
    if isinstance(action_input, dict):
        for k in _SQL_KEYS:
            v = action_input.get(k)
            if isinstance(v, str) and _SQL_SHAPE.match(v):
                return v.strip()
    return ""


def _op_and_table(stmt: str) -> "tuple[str, str]":
    """('INSERT', 'orders') from a statement, best-effort."""
    m = re.match(
        r"\s*(select|insert|update|delete|drop|truncate|alter|create|upsert)\b", stmt, re.I)
    op = m.group(1).upper() if m else ""
    t = re.search(
        r"\b(?:from|into|update|table|join)\s+[\"'`]?([A-Za-z_][\w.]*)", stmt, re.I)
    return op, (t.group(1) if t else "")


class SqlPack(Pack):
    name = "sql"

    def owns(self, action: Action) -> bool:
        if action.type != "call":
            return False
        return bool(_statement(action.input)) or any(
            fnmatch(action.tool.lower(), g) for g in _TOOL_NAMES)

    def debugger_panels(self, action: Action, trace: dict) -> "list[dict]":
        stmt = _statement(action.input)
        if not stmt:
            return []
        op, table = _op_and_table(stmt)
        return [{"title": f"🗄 SQL · {op} on {table or '?'}", "code": stmt[:4000]}]

    def capabilities(self, name: str, tool_input) -> "set[str]":
        stmt = _statement(tool_input)
        if not stmt:
            return set()
        op, _ = _op_and_table(stmt)
        caps: set[str] = set()
        if op == "SELECT":
            caps.add("read")
        elif op in ("INSERT", "UPDATE", "UPSERT"):
            caps |= {"database_write", "write"}
        elif op in ("DELETE", "DROP", "TRUNCATE"):
            caps |= {"database_write", "write", "destructive"}
        elif op in ("ALTER", "CREATE"):
            caps |= {"database_write", "write"}
        if _PII_COLUMNS.search(stmt):
            caps.add("pii_access")
        return caps

    def state_diff(self, action: Action, trace: dict) -> "StateDiff | None":
        stmt = _statement(action.input)
        if not stmt:
            return None
        op, table = _op_and_table(stmt)
        if op in ("", "SELECT"):
            return None  # reads don't change the world
        target = f" on {table}" if table else ""
        rows = ""
        if action.observation is not None:
            m = _ROWCOUNT.search(action.observation.text or "")
            if m:
                rows = f" ({m.group(1)} rows)"
        return StateDiff("database", f"{op}{target}{rows}",
                         detail={"op": op, "table": table, "statement": stmt[:500]})

    def undo(self, action: Action, trace: dict) -> "UndoPlan | None":
        stmt = _statement(action.input)
        if not stmt:
            return None
        op, table = _op_and_table(stmt)
        t = table or "<table>"
        if op == "INSERT":
            # Compensable but not a clean revert: the trace doesn't hold the
            # keys of the inserted rows, so this needs a human to fill them in.
            return UndoPlan("compensate", f"DELETE the rows inserted into {t}",
                            [f"DELETE FROM {t} WHERE <keys of the inserted rows>"],
                            reversible=False)
        if op in ("UPDATE", "UPSERT"):
            return UndoPlan(
                "compensate", f"restore the previous values in {t}",
                [f"UPDATE {t} SET <columns to prior values> WHERE <same predicate>"],
                reversible=False)  # prior values aren't in the trace
        if op in ("DELETE", "DROP", "TRUNCATE"):
            return UndoPlan(
                "noop", f"{op} on {t} is not reversible from the trace -- "
                        "restore from a backup or point-in-time recovery",
                reversible=False)
        return None

    def safe_runtime(self) -> str:
        return ("point the SQL tool at a read-replica or a disposable test database, "
                "wrap writes in a transaction you roll back, or run it EXPLAIN-only "
                "(dry-run) while debugging -- never at production data")

    def replay_hint(self, action: Action) -> str:
        return ("restore the database to its state before this step "
                "(snapshot / point-in-time recovery), then replay")


register(SqlPack())
