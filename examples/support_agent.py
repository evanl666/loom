"""A support agent, debugged: business risk and the incident report.

A refund is money movement, an email is unsendable-back, and the incident
report reads in business terms -- affected customers and money, not files.
"""

import json

from _shared import show

from loom import Agent, tool
from loom.incident import build_report
from loom.providers import ModelResponse, ScriptedProvider, ToolCall


@tool
def get_customer(id: int) -> str:
    "Look up a customer record."
    return "Jane Doe <jane@example.com>, order A-17, $500"


@tool
def issue_refund(amount: int, order_id: str) -> str:
    "Refund an order."
    return "refunded"


@tool
def send_email(to: str, body: str) -> str:
    "Email a customer."
    return "sent"


model = ScriptedProvider([
    ModelResponse(text="Looking up the order first.",
                  tool_calls=[ToolCall("t1", "get_customer", {"id": 7})],
                  stop_reason="tool_use"),
    ModelResponse(text="Order qualifies -- issuing the refund.",
                  tool_calls=[ToolCall("t2", "issue_refund",
                                       {"amount": 500, "order_id": "A-17"})],
                  stop_reason="tool_use"),
    ModelResponse(text="Confirming with the customer.",
                  tool_calls=[ToolCall("t3", "send_email",
                                       {"to": "jane@example.com",
                                        "body": "Your $500 refund for A-17 is on its way."})],
                  stop_reason="tool_use"),
    ModelResponse(text="Refund issued and customer notified."),
])

run = Agent(model=model, tools=[get_customer, issue_refund, send_email],
            name="support").run("customer 7 wants a refund for order A-17")
show(run, "support_agent.loom.json")

print("\n== Incident report (excerpt) " + "=" * 31)
report = build_report(json.load(open("support_agent.loom.json")), "support_agent.loom.json")
for line in report.splitlines():
    if line.startswith(("**Severity", "- customers/records", "- risky",
                        "- add firewall rule")):
        print(" " + line)
