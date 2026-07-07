"""REWIND: load the trace, rewind to turn 0, change the city, branch off."""

from demo_agent import make_agent

from loom import Run

run = Run.load("trace.json", agent=make_agent())


def edit(ctx):
    ctx.items[0].content = "Should I bike to work in Lisbon today?"


branch = run.fork(at=0, edit=edit)

print(f"original: {run.output}")
print(f"rewound : {branch.output}")
