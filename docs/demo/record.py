"""RECORD: run the agent once; every model + tool call is recorded."""

from demo_agent import make_agent

agent = make_agent()
question = "Should I bike to work in Berlin today?"

print(f"Q: {question}")
run = agent.run(question)
print(f"A: {run.output}")
print()
run.save("trace.json")
print(f"saved -> trace.json  ({len(run.log)} recorded effects)")
