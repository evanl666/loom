# Loom examples — one per agent domain

Every example runs **offline** (a `ScriptedProvider` plays the model), so you
can run them straight from a clone with no API key:

```
python examples/coding_agent.py
python examples/sql_agent.py
python examples/browser_agent.py
python examples/support_agent.py
```

| Example | Domain pack | What it shows |
|---|---|---|
| `coding_agent.py` | coding (built-in) | file-edit state diffs, git undo plan, the Action timeline |
| `sql_agent.py` | sql | capabilities parsed from SQL, PII detection, compensating undo |
| `browser_agent.py` | browser | DOM before/after diffs, unsubmittable form honesty |
| `support_agent.py` | support | refund/email business risk, the agent-type-aware incident report |

Each script records a run, prints its **Action timeline** (what it did, why,
risk, what changed), the **undo/compensation plans**, and writes the trace +
Studio HTML next to itself — open the HTML to see the Action Debugger.

To wire in a *real* agent instead of the scripted one, see "Record any
agent" in the top-level README: `loom record claude "..." --safe` needs no
code at all.
