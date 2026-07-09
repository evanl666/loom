"""A data agent, debugged: capabilities parsed from the SQL itself.

The SELECT touching PII columns is flagged pii_access even though it "only
reads"; the INSERT gets a database StateDiff and a compensating undo.
"""

from _shared import show

from loom import Agent, tool
from loom.providers import ModelResponse, ScriptedProvider, ToolCall


@tool
def run_sql(query: str) -> str:
    "Run a SQL statement."
    return "1 row inserted" if "INSERT" in query else "12 rows"


model = ScriptedProvider([
    ModelResponse(text="Checking which customers churned last month.",
                  tool_calls=[ToolCall("t1", "run_sql", {
                      "query": "SELECT email, date_of_birth FROM customers WHERE churned=1"})],
                  stop_reason="tool_use"),
    ModelResponse(text="Logging the cohort into the retention table.",
                  tool_calls=[ToolCall("t2", "run_sql", {
                      "query": "INSERT INTO retention_cohort SELECT id FROM customers WHERE churned=1"})],
                  stop_reason="tool_use"),
    ModelResponse(text="Done: cohort of 12 recorded."),
])

run = Agent(model=model, tools=[run_sql], name="analyst").run(
    "build last month's churn cohort")
show(run, "sql_agent.loom.json")

print("""
try the firewall on this agent:
  loom policy simulate sql_agent.loom.json --profile prod-db-safe
  (or gate it live: loom proxy --confirm 'cap:database_write')""")
