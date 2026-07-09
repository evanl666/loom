"""Shared rendering for the example gallery: print a run the debugger way."""

from loom.packs import install_builtin, undo_plan


def show(run, save_as: str) -> None:
    """Print the Action timeline + undo plans; save the trace and Studio HTML."""
    install_builtin()
    print("\n== Action timeline " + "=" * 41)
    for a in run.actions():
        if a.type == "reason":
            print(f'  [{a.step}] 💭 "{a.intent[:70]}"')
        elif a.type == "call":
            risk = f"  ⚠ {a.risk}" if a.risk else ""
            print(f"  [{a.step}] 🔧 {a.tool}{risk}")
            if a.state_diff is not None:
                print(f"        Δ {a.state_diff.summary}")
        elif a.type == "answer":
            print(f'  [{a.step}] ✅ "{a.intent[:70]}"')

    plans = run.undo_plans()
    if plans:
        print("\n== Undo / compensation (newest first) " + "=" * 22)
        for action, plan in plans:
            mark = {"revert": "↩", "compensate": "⇄", "noop": "✋"}[plan.kind]
            rev = "" if plan.reversible else "  [irreversible]"
            print(f"  {mark} {action.tool}: {plan.summary}{rev}")

    run.save(save_as)
    html_path = save_as.replace(".loom.json", ".html")
    from loom import trace_to_html

    with open(html_path, "w") as f:
        f.write(trace_to_html(run.to_dict(), path=save_as))
    print(f"\ntrace: {save_as}\nstudio: {html_path}  (open in a browser)")
