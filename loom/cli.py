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


def _build_shield(args: argparse.Namespace):
    """Turn --deny/--confirm/--allow flags into a Shield (or None)."""
    if not (args.deny or args.confirm or args.allow):
        return None
    from .shield import Shield

    return Shield(
        deny=args.deny or [],
        confirm=args.confirm or [],
        allow=args.allow or [],
        timeout=args.confirm_timeout,
        webhook=args.webhook,
    )


def _shield_notifier(port: int):
    """Console printer for pending approvals, with the exact command to run."""

    def notify(p) -> None:
        print(
            f"\nloom shield: CONFIRM [{p.id}] {p.tool}({json.dumps(p.input, sort_keys=True, default=str)})\n"
            f"  approve:  loom approve {p.id} --port {port}\n"
            f"  deny:     loom approve {p.id} --deny --port {port}",
            file=sys.stderr,
        )

    return notify


def _print_shield_rules(shield) -> None:
    for action, patterns in (("deny", shield.deny), ("confirm", shield.confirm), ("allow", shield.allow)):
        for pat in patterns:
            print(f"  shield {action:7s} {pat}", file=sys.stderr)


def _cmd_record(args: argparse.Namespace) -> int:
    """Black-box a real agent session: proxy up, env var set, command run."""
    import os
    import subprocess
    import threading

    from .proxy import ProxyServer

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("loom: record needs a command, e.g. loom record -- claude -p 'hi'", file=sys.stderr)
        return 2

    shield = _build_shield(args)
    server = ProxyServer(port=args.port, target=args.target, save_path=args.save, shield=shield)
    if shield is not None:
        shield.notify = _shield_notifier(server.port)
        _print_shield_rules(shield)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    env = dict(os.environ)
    base = f"http://127.0.0.1:{server.port}"
    if "openai" in args.target:
        env["OPENAI_BASE_URL"] = base + "/v1"
    else:
        env["ANTHROPIC_BASE_URL"] = base
    print(f"loom record: proxying {args.target} on {base}", file=sys.stderr)

    code = subprocess.call(command, env=env)
    server.shutdown()

    try:
        with open(args.save) as f:
            data = json.load(f)
        log = data.get("log", [])
        inp = sum(
            e["result"].get("usage", {}).get("input_tokens", 0)
            for e in log
            if e.get("kind") == "model"
        )
        out = sum(
            e["result"].get("usage", {}).get("output_tokens", 0)
            for e in log
            if e.get("kind") == "model"
        )
        print(
            f"\nrecorded {len(log)} step(s), {inp + out} tokens -> {args.save}",
            file=sys.stderr,
        )
        blocked = [e for e in data.get("shield_events", []) if e.get("action") == "deny"]
        if blocked:
            print(f"  shield blocked {len(blocked)} tool call(s):", file=sys.stderr)
            for e in blocked:
                print(f"    {e.get('tool')}({json.dumps(e.get('input', {}), sort_keys=True, default=str)})", file=sys.stderr)
        print(f"  replay it:  loom replay {args.save}", file=sys.stderr)
        print(f"  inspect it: loom studio {args.save}", file=sys.stderr)
    except (OSError, json.JSONDecodeError):
        print("\nno traffic recorded (did the agent talk to the API?)", file=sys.stderr)
    return code


def _load_agent(spec: str) -> "tuple[Any, str] | tuple[None, str]":
    """Resolve module:attr to an Agent (or factory). Returns (agent, error)."""
    import importlib
    import os

    sys.path.insert(0, os.getcwd())
    module_name, _, attr = spec.partition(":")
    if not attr:
        return None, "--agent must look like module:attr"
    try:
        obj = getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError) as e:
        return None, f"could not load agent {spec!r}: {e}"
    return (obj() if callable(obj) and not hasattr(obj, "run") else obj), ""


def _cmd_heal(args: argparse.Namespace) -> int:
    """Diagnose a failed run, try repairs, optionally save the fix as a test."""
    from .trace import Run

    if not args.forbid and not args.require:
        print("loom: heal needs --forbid and/or --require to know what 'fixed' means", file=sys.stderr)
        return 2
    agent, err = _load_agent(args.agent)
    if agent is None:
        print(f"loom: {err}", file=sys.stderr)
        return 2

    run = Run.load(args.path, agent=agent)
    report = run.checkup()
    print(report.summary())

    def check(text: str) -> bool:
        ok = True
        if args.forbid:
            ok = ok and args.forbid not in text
        if args.require:
            ok = ok and args.require in text
        return ok

    if check(run.output):
        print("\nrun already passes the check; nothing to heal")
        return 0
    healed = run.heal(check, regression_dir=args.save_regression or None)
    if healed is None:
        print("\nno repair fixed the run")
        return 1
    print(f"\n✅ healed by: {healed.healed_by}")
    print(f"   output now: {healed.output[:100]}")
    if healed.regression_path:
        print(f"   saved regression: {healed.regression_path}")
    return 0


def _cmd_skills(args: argparse.Namespace) -> int:
    """Mine proven tool sequences from a trace corpus into a skill library."""
    from .skills import mine, save
    from .trace import Run

    paths = _expand_trace_paths(args.paths)
    if not paths:
        print("no traces found", file=sys.stderr)
        return 2
    runs = [Run.load(p) for p in paths]
    skills = mine(runs, min_support=args.min_support)
    if not skills:
        print(f"no sequences seen in >= {args.min_support} successful runs")
        return 1
    for s in skills:
        print(f"skill: {s.name}   (support: {s.support} runs)")
        for i, step in enumerate(s.steps, 1):
            print(f"  {i}. {step['tool']}({json.dumps(step['args'])})")
        if s.params:
            print(f"  parameters: {', '.join(s.params)}")
    if args.save:
        save(skills, args.save)
        print(f"\nsaved {len(skills)} skill(s) -> {args.save}")
    return 0


def _cmd_studio(args: argparse.Namespace) -> int:
    """Export a trace to the Studio viewer and open it in the default browser."""
    import webbrowser

    code = _cmd_export(args)
    if code == 0:
        out = args.output or (args.path.rsplit(".json", 1)[0] + ".html")
        webbrowser.open(f"file://{__import__('os').path.abspath(out)}")
    return code


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
    from .impact import assess, report, to_json

    agent, err = _load_agent(args.agent)
    if agent is None:
        print(f"loom: {err}", file=sys.stderr)
        return 2

    paths = _expand_trace_paths(args.paths)
    if not paths:
        print("no traces found", file=sys.stderr)
        return 2
    try:
        impacts = assess(paths, agent, live=args.live)
    except (OSError, json.JSONDecodeError) as e:
        print(f"loom: could not read traces: {e}", file=sys.stderr)
        return 2
    if args.json:
        with open(args.json, "w") as f:
            json.dump(to_json(impacts), f, indent=2)
    print(report(impacts))
    return 1 if any(i.changed for i in impacts) else 0


def _cmd_proxy(args: argparse.Namespace) -> int:
    from .proxy import ProxyServer

    shield = _build_shield(args)
    server = ProxyServer(
        port=args.port,
        target=args.target,
        save_path=args.save if not args.replay else None,
        replay_path=args.replay or None,
        shield=shield if not args.replay else None,
    )
    mode = f"replaying {args.replay}" if args.replay else f"recording -> {args.save}"
    print(f"loom proxy on http://127.0.0.1:{server.port} ({mode})")
    print(f"  export ANTHROPIC_BASE_URL=http://127.0.0.1:{server.port}")
    if shield is not None and not args.replay:
        shield.notify = _shield_notifier(server.port)
        _print_shield_rules(shield)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def _cmd_approvals(args: argparse.Namespace) -> int:
    """List a running proxy's pending shield approvals."""
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{args.port}/loom/shield/pending", timeout=10
        ) as r:
            pending = json.load(r).get("pending", [])
    except (urllib.error.URLError, OSError) as e:
        print(f"loom: no shielded proxy on port {args.port} ({e})", file=sys.stderr)
        return 2
    if not pending:
        print("no pending approvals")
        return 0
    for p in pending:
        print(f"[{p['id']}] {p['tool']}({json.dumps(p.get('input', {}), sort_keys=True, default=str)})"
              f"  rule={p.get('rule', '')!r}  waiting {p.get('age_s', 0)}s")
    return 0


def _cmd_approve(args: argparse.Namespace) -> int:
    """Decide a pending shield approval on a running proxy."""
    import urllib.error
    import urllib.request

    decision = "deny" if args.deny else "approve"
    req = urllib.request.Request(
        f"http://127.0.0.1:{args.port}/loom/shield/decide",
        data=json.dumps({"id": args.id, "decision": decision}).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=10).close()
    except urllib.error.HTTPError as e:
        print(f"loom: {json.load(e).get('error', e.reason)}", file=sys.stderr)
        return 1
    except (urllib.error.URLError, OSError) as e:
        print(f"loom: no shielded proxy on port {args.port} ({e})", file=sys.stderr)
        return 2
    print(f"{decision}d [{args.id}]")
    return 0


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
        "--json",
        default="",
        metavar="FILE",
        help="also write a machine-readable report (used for cost comparison across branches)",
    )
    im.add_argument(
        "--live",
        action="store_true",
        help="re-run affected conversations to show HOW outputs change (costs API calls)",
    )
    im.set_defaults(func=_cmd_impact)

    def shield_flags(sp) -> None:
        sp.add_argument("--deny", action="append", default=[], metavar="PATTERN",
                        help="block tool calls matching this glob, e.g. 'Read(*.env*)' (repeatable)")
        sp.add_argument("--confirm", action="append", default=[], metavar="PATTERN",
                        help="hold matching tool calls for approval (loom approve <id>)")
        sp.add_argument("--allow", action="append", default=[], metavar="PATTERN",
                        help="bypass confirm for matching tool calls")
        sp.add_argument("--confirm-timeout", type=float, default=300.0,
                        help="seconds to wait for approval before denying (default 300)")
        sp.add_argument("--webhook", default="",
                        help="POST pending approvals to this URL (approval inbox)")

    rc = sub.add_parser("record", help="black-box a real agent session: loom record -- <command>")
    rc.add_argument("command", nargs=argparse.REMAINDER, help="the agent command to run")
    rc.add_argument("--save", default="session.loom.json")
    rc.add_argument("--target", default="https://api.anthropic.com",
                    help="upstream API (use https://api.openai.com for OpenAI agents)")
    rc.add_argument("--port", type=int, default=0, help="proxy port (default: pick a free one)")
    shield_flags(rc)
    rc.set_defaults(func=_cmd_record)

    he = sub.add_parser("heal", help="diagnose a failed run, try repairs, save the fix as a test")
    he.add_argument("path")
    he.add_argument("--agent", required=True, help="module:attr (Agent or zero-arg factory)")
    he.add_argument("--forbid", default="", help="healed when the output no longer contains this")
    he.add_argument("--require", default="", help="healed when the output contains this")
    he.add_argument("--save-regression", default="", dest="save_regression",
                    help="directory to save the healed run as a golden trace")
    he.set_defaults(func=_cmd_heal)

    sk = sub.add_parser("skills", help="mine proven tool sequences from traces into skills")
    sk.add_argument("paths", nargs="+", help="trace files or directories of *.loom.json")
    sk.add_argument("--min-support", type=int, default=2, dest="min_support")
    sk.add_argument("--save", default="", help="write the skill library to this JSON file")
    sk.set_defaults(func=_cmd_skills)

    st = sub.add_parser("studio", help="export a trace to the Studio viewer and open it")
    st.add_argument("path")
    st.add_argument("-o", "--output", default="", help="output path (default: <trace>.html)")
    st.set_defaults(func=_cmd_studio)

    px = sub.add_parser("proxy", help="record any Anthropic-API agent through a local proxy")
    px.add_argument("--port", type=int, default=8788)
    px.add_argument("--target", default="https://api.anthropic.com")
    px.add_argument("--save", default="session.loom.json", help="trace written after every exchange")
    px.add_argument("--replay", default="", help="serve recorded responses from this trace instead")
    shield_flags(px)
    px.set_defaults(func=_cmd_proxy)

    av = sub.add_parser("approvals", help="list pending shield approvals on a running proxy")
    av.add_argument("--port", type=int, default=8788)
    av.set_defaults(func=_cmd_approvals)

    ap = sub.add_parser("approve", help="approve (or --deny) a pending shield tool call")
    ap.add_argument("id")
    ap.add_argument("--deny", action="store_true", help="deny instead of approving")
    ap.add_argument("--port", type=int, default=8788)
    ap.set_defaults(func=_cmd_approve)

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
