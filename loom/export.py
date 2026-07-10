"""Export a saved trace to Loom Studio: a self-contained HTML viewer.

Studio is the interactive debugger UI (``loom debug``) *frozen into a file* --
the same page, with its data inlined and server-only features (fork / live /
copilot / assert) switched off. No external assets, no server, no agent needed,
so the file can be attached to a bug report, emailed, or committed next to the
trace. You get the agent tree, timeline scrubber, per-step inspector, and the
reconstructed conversation at any point.

    loom export run.loom.json            # writes run.loom.html
    loom studio run.loom.json            # same, then opens it in the browser

There is a single viewer to maintain: this module delegates to
``loom.debugger.static_page``. (The old bespoke Studio renderer was retired; its
reusable analyzer panels moved to ``loom.report_panels``.)
"""

from __future__ import annotations


def trace_to_html(data: dict, path: str = "session.loom.json") -> str:
    """Render a trace as the self-contained Studio page (the frozen debugger UI).

    ``path`` is accepted for backwards compatibility but no longer affects the
    output -- the page carries the run inline."""
    from .debugger import static_page

    return static_page(data)
