"""A browser agent, debugged: DOM diffs and the unsubmittable-form truth.

Navigation is trivially reversible (go back); a submitted form is not -- the
undo plan says so instead of pretending.
"""

from _shared import show

from loom import Agent, tool
from loom.providers import ModelResponse, ScriptedProvider, ToolCall


@tool
def navigate(url: str) -> str:
    "Open a page."
    return "loaded"


@tool
def fill_form(selector: str, text: str) -> str:
    "Type into a field."
    return "filled"


@tool
def click(selector: str) -> dict:
    "Click an element."
    return {"dom_before": "<button>Submit</button>",
            "dom_after": "<p>Order #1027 confirmed</p>"}


model = ScriptedProvider([
    ModelResponse(text="Opening the checkout page.",
                  tool_calls=[ToolCall("t1", "navigate",
                                       {"url": "https://shop.example/checkout"})],
                  stop_reason="tool_use"),
    ModelResponse(text="Entering the quantity the user asked for.",
                  tool_calls=[ToolCall("t2", "fill_form",
                                       {"selector": "#qty", "text": "2"})],
                  stop_reason="tool_use"),
    ModelResponse(text="Placing the order.",
                  tool_calls=[ToolCall("t3", "click", {"selector": "#place-order"})],
                  stop_reason="tool_use"),
    ModelResponse(text="Order #1027 placed."),
])

run = Agent(model=model, tools=[navigate, fill_form, click], name="shopper").run(
    "order two of item X")
show(run, "browser_agent.loom.json")

print("""
gate the risky step next time:
  loom proxy --confirm 'cap:browser_submit'   # a human approves every submit""")
