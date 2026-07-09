# Loom incident gallery

Canonical agent incidents, each recorded offline into a replayable trace with a
Studio view and a 30-second movie. Regenerate: `python generate.py`.

| Incident | What happens | See |
|---|---|---|
| **secret-leak** | agent reads `.env`, then curls the key out | `loom taint`, Data Flow panel, movie |
| **sql-delete** | a `DELETE FROM orders` with no useful WHERE | Impact Map (database), undo = restore-from-backup |
| **browser-submit** | an irreversible form submit | undo says "cannot be unsubmitted" |
| **refund-mistake** | a refund issued for 10× the amount | money-movement risk, score drop |

Try on any of them:
```
loom movie secret-leak.loom.json --open      # the shareable animation
loom taint secret-leak.loom.json             # the leak chain
loom diagnose sql-delete.loom.json --plan     # root cause + fix
loom score refund-mistake.loom.json           # the behavior scorecard
loom fix from secret-leak.loom.json           # a fix PR
```
