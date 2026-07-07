"""The ``loom`` command-line interface.

    loom run "What is 2 + 3?" --model claude-opus-4-8
    loom timeline trace.loom.json
    loom replay   trace.loom.json

Uses only the standard library so the CLI ships with the zero-dependency core.
``loom run`` needs a live provider (e.g. Anthropic) and its API key.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .effect import EffectEntry
from .providers.base import ModelResponse


def _cmd_run(args: argparse.Namespace) -> int:
    from .agent import Agent

    try:
        agent = Agent(model=args.model, system=args.system or "")
        run = agent.run(args.prompt)
    except Exception as e:  # surface provider/auth errors cleanly
        print(f"loom: run failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(run.output)
    if args.timeline:
        print("\n--- timeline ---", file=sys.stderr)
        run.print_timeline()
    if args.save:
        run.save(args.save)
        print(f"\nsaved trace -> {args.save}", file=sys.stderr)
    return 0


def _load_log(path: str) -> "tuple[list[EffectEntry], dict]":
    with open(path) as f:
        data = json.load(f)
    return [EffectEntry.from_dict(e) for e in data["log"]], data


def _cmd_timeline(args: argparse.Namespace) -> int:
    log, data = _load_log(args.path)
    print(f"model:  {data.get('model')}")
    print(f"prompt: {data.get('prompt')}")
    print(f"turns:  {sum(1 for e in log if e.kind == 'model')}")
    print("--- timeline ---")
    turn = 0
    for e in log:
        if e.kind == "model":
            resp = ModelResponse.from_dict(e.result)
            detail = (
                "calls " + ", ".join(f"{tc.name}({json.dumps(tc.input)})" for tc in resp.tool_calls)
                if resp.tool_calls
                else resp.text
            )
            print(f"  [{e.seq:>2}] turn {turn:>2} model          {detail[:100]}")
            turn += 1
        else:
            result = e.result if isinstance(e.result, str) else json.dumps(e.result)
            print(f"  [{e.seq:>2}] turn {turn - 1:>2} {e.kind:<14} {result[:100]}")
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    """Reconstruct the final output from the recorded trace -- deterministic, offline."""
    log, data = _load_log(args.path)
    model_entries = [e for e in log if e.kind == "model"]
    output = ModelResponse.from_dict(model_entries[-1].result).text if model_entries else ""
    print(output)
    if output != data.get("output", output):
        print("loom: warning: replayed output differs from stored output", file=sys.stderr)
    return 0


def _cmd_test(args: argparse.Namespace) -> int:
    """Verify a suite of saved traces; exit 1 if any fail."""
    from .testing import verify_trace

    paths = _expand_trace_paths(args.paths)
    if not paths:
        print("no traces found", file=sys.stderr)
        return 1

    failed = 0
    for p in paths:
        problems = verify_trace(p)
        if problems:
            failed += 1
            print(f"FAIL {p}")
            for problem in problems:
                print(f"     - {problem}")
        else:
            print(f"PASS {p}")
    print(f"\n{len(paths) - failed}/{len(paths)} traces passed")
    return 1 if failed else 0


def _cmd_watch(args: argparse.Namespace) -> int:
    """Follow a run's journal live (like tail -f for an agent)."""
    import time

    from .journal import Journal

    seen = 0
    shown_header = False
    while True:
        try:
            header, entries = Journal.read(args.path)
        except FileNotFoundError:
            if args.once:
                print(f"no journal at {args.path}", file=sys.stderr)
                return 1
            time.sleep(args.interval)
            continue
        if header and not shown_header:
            episodes = header.get("episodes") or []
            print(f"model: {header.get('model')}  |  prompt: {episodes[0] if episodes else '?'}")
            shown_header = True
        for e in entries[seen:]:
            result = e.result if isinstance(e.result, str) else json.dumps(e.result)
            detail = (result[:100] + "...") if len(result) > 100 else result
            indent = "  " * e.depth
            print(f"  [{e.seq:>3}] {indent}{e.kind:<14} {detail}")
        seen = len(entries)
        if args.once:
            return 0
        time.sleep(args.interval)


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Check a saved trace for context rot; exit 1 if findings exist."""
    from .health import analyze

    log, data = _load_log(args.path)
    episodes = data.get("episodes") or [data.get("prompt", "")]
    report = analyze(episodes, log)
    print(report.summary())
    return 0 if report.ok else 1


def _cmd_export(args: argparse.Namespace) -> int:
    """Render a saved trace to a self-contained HTML page."""
    from .export import trace_to_html

    with open(args.path) as f:
        data = json.load(f)
    out = args.output or (args.path.rsplit(".json", 1)[0] + ".html")
    with open(out, "w") as f:
        f.write(trace_to_html(data))
    print(f"wrote {out}")
    return 0


def _cmd_diff(args: argparse.Namespace) -> int:
    """Compare two saved traces; exit 0 if identical, 1 if they diverge."""
    from .diff import diff_logs

    log_a, _ = _load_log(args.a)
    log_b, _ = _load_log(args.b)
    d = diff_logs(log_a, log_b)
    print(d.summary())
    return 0 if d.identical else 1


def _expand_trace_paths(targets: list[str]) -> list[str]:
    import glob
    import os

    paths: list[str] = []
    for target in targets:
        if os.path.isdir(target):
            paths.extend(sorted(glob.glob(os.path.join(target, "*.loom.json"))))
        else:
            paths.append(target)
    return paths


def _cmd_impact(args: argparse.Namespace) -> int:
    """Replay a trace corpus against a changed agent config; exit 1 if any run is affected."""
    import importlib
    import os

    from .impact import assess, report

    sys.path.insert(0, os.getcwd())
    module_name, _, attr = args.agent.partition(":")
    if not attr:
        print("loom: --agent must look like module:attr", file=sys.stderr)
        return 2
    try:
        obj = getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError) as e:
        print(f"loom: could not load agent {args.agent!r}: {e}", file=sys.stderr)
        return 2
    agent = obj() if callable(obj) and not hasattr(obj, "run") else obj

    paths = _expand_trace_paths(args.paths)
    if not paths:
        print("no traces found", file=sys.stderr)
        return 2
    impacts = assess(paths, agent, live=args.live)
    print(report(impacts))
    return 1 if any(i.changed for i in impacts) else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="loom", description="Read, replay, and rewind agent runs.")
    p.add_argument("--version", action="version", version=f"loom {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run an agent with a live model")
    run.add_argument("prompt")
    run.add_argument("--model", default="claude-opus-4-8")
    run.add_argument("--system", default="")
    run.add_argument("--save", default="", help="write the trace to this path")
    run.add_argument("--timeline", action="store_true", help="print the step-by-step timeline")
    run.set_defaults(func=_cmd_run)

    tl = sub.add_parser("timeline", help="print the timeline of a saved trace")
    tl.add_argument("path")
    tl.set_defaults(func=_cmd_timeline)

    rp = sub.add_parser("replay", help="replay a saved trace offline")
    rp.add_argument("path")
    rp.set_defaults(func=_cmd_replay)

    df = sub.add_parser("diff", help="compare two saved traces at the effect level")
    df.add_argument("a")
    df.add_argument("b")
    df.set_defaults(func=_cmd_diff)

    ex = sub.add_parser("export", help="render a saved trace to self-contained HTML")
    ex.add_argument("path")
    ex.add_argument("-o", "--output", default="", help="output path (default: <trace>.html)")
    ex.set_defaults(func=_cmd_export)

    dr = sub.add_parser("doctor", help="check a saved trace for context rot")
    dr.add_argument("path")
    dr.set_defaults(func=_cmd_doctor)

    ts = sub.add_parser("test", help="verify a suite of saved traces")
    ts.add_argument("paths", nargs="+", help="trace files or directories of *.loom.json")
    ts.set_defaults(func=_cmd_test)

    im = sub.add_parser("impact", help="which recorded runs does a config change affect?")
    im.add_argument("paths", nargs="+", help="trace files or directories of *.loom.json")
    im.add_argument(
        "--agent",
        required=True,
        help="where to find the (re)configured agent: module:attr, an Agent or a zero-arg factory",
    )
    im.add_argument(
        "--live",
        action="store_true",
        help="re-run affected conversations to show HOW outputs change (costs API calls)",
    )
    im.set_defaults(func=_cmd_impact)

    wa = sub.add_parser("watch", help="follow a run's journal live")
    wa.add_argument("path")
    wa.add_argument("--interval", type=float, default=0.5)
    wa.add_argument("--once", action="store_true", help="print current state and exit")
    wa.set_defaults(func=_cmd_watch)
    return p


def main(argv: "list[str] | None" = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
