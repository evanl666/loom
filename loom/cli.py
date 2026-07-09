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


class CLIError(Exception):
    """A user-facing problem: printed as ``loom: <message>``, exit 2.

    Raise it with a message that says BOTH what went wrong and what to do
    next -- a traceback is a bug report, this is an answer.
    """


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
    data = _load_trace_json(path)
    if not isinstance(data.get("log"), list):
        raise CLIError(
            f"{path} is JSON but has no 'log' -- not a loom trace. Traces are "
            f"written by run.save(), `loom record`, or the proxy."
        )
    return [EffectEntry.from_dict(e) for e in data["log"]], data


def _load_trace_json(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        raise CLIError(f"no such file: {path}")
    except IsADirectoryError:
        raise CLIError(
            f"{path} is a directory -- pass a single trace file "
            f"(or use `loom test {path}` / `loom search {path}` for a corpus)"
        )
    except json.JSONDecodeError as e:
        raise CLIError(f"{path} is not valid JSON (line {e.lineno}): expected a loom trace")


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

    data = _load_trace_json(args.path)
    out = args.output or (args.path.rsplit(".json", 1)[0] + ".html")
    with open(out, "w") as f:
        f.write(trace_to_html(data))
    print(f"wrote {out}")
    return 0


def _build_shield(args: argparse.Namespace):
    """Turn --deny/--confirm/--allow/--judge/--rule flags into a Shield (or None)."""
    if not (args.deny or args.confirm or args.allow or args.judge or args.rule
            or args.shield_default != "allow"):
        return None
    from .shield import Shield, TrustLedger

    trust = None
    if args.trust_after > 0:
        import os

        ledger_path = args.trust_ledger or os.path.join(
            os.path.expanduser("~"), ".loom", "trust.json"
        )
        trust = TrustLedger(ledger_path)
    return Shield(
        deny=args.deny or [],
        confirm=args.confirm or [],
        allow=args.allow or [],
        default=args.shield_default,
        timeout=args.confirm_timeout,
        webhook=args.webhook,
        judge=args.judge or None,
        judge_threshold=args.judge_threshold,
        trust=trust,
        trust_after=args.trust_after,
        sequence=args.rule or [],
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
    for rule in shield.sequence:
        print(f"  shield rule    {rule.raw}", file=sys.stderr)


def _recover_wirelog(save: str) -> None:
    """A leftover .wirelog means the last session crashed mid-run: salvage it."""
    import os

    wirelog = save + ".wirelog"
    if not os.path.exists(wirelog):
        return
    from .proxy import compact_wirelog

    if save.endswith(".loom.json"):
        recovered = save[: -len(".loom.json")] + ".recovered.loom.json"
    else:
        recovered = save + ".recovered"
    compact_wirelog(wirelog, recovered)
    os.remove(wirelog)
    print(f"loom: recovered a crashed session's wirelog -> {recovered}", file=sys.stderr)


def _cmd_record(args: argparse.Namespace) -> int:
    """Black-box a real agent session: proxy up, env var set, command run."""
    import os
    import subprocess
    import threading

    from .proxy import ProxyServer

    _recover_wirelog(args.save)

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("loom: record needs a command, e.g. loom record -- claude -p 'hi'", file=sys.stderr)
        return 2

    shield = _build_shield(args)
    server = ProxyServer(port=args.port, target=args.target, save_path=args.save,
                         shield=shield, scrub=args.scrub,
                         max_body=args.max_body_mb * 1024 * 1024,
                         upstream_timeout=args.upstream_timeout, auth=args.auth)
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

    profile_path = ""
    if args.sandbox:
        from .sandbox import wrap_sandboxed

        try:
            command, profile_path = wrap_sandboxed(
                command, ports=[server.port], allow=args.sandbox_allow
            )
        except RuntimeError as e:
            print(f"loom: {e}", file=sys.stderr)
            server.shutdown()
            server.finalize()
            return 2
        print("loom record: sandboxed -- the proxy is the only network door",
              file=sys.stderr)

    try:
        code = subprocess.call(command, env=env)
    finally:
        if profile_path:
            os.unlink(profile_path)
    server.shutdown()
    server.finalize()

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
        if "openai" not in args.target:
            print("  talking to OpenAI instead? add: --target https://api.openai.com",
                  file=sys.stderr)
        print(f"  expected the agent to honor {'OPENAI_BASE_URL' if 'openai' in args.target else 'ANTHROPIC_BASE_URL'}",
              file=sys.stderr)
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
    from .skills import approve, mine, save
    from .trace import Run

    if args.approve:
        library = args.paths[0]
        if approve(library, args.approve):
            print(f"approved {args.approve} in {library}")
            return 0
        print(f"loom: no skill named {args.approve!r} in {library}", file=sys.stderr)
        return 1

    paths = _expand_trace_paths(args.paths)
    if not paths:
        print("no traces found", file=sys.stderr)
        return 2
    runs = [Run.load(p) for p in paths]

    check = None
    if args.forbid or args.require:
        def check(run) -> bool:
            ok = True
            if args.forbid:
                ok = ok and args.forbid not in run.output
            if args.require:
                ok = ok and args.require in run.output
            return ok

    skills = mine(runs, min_support=args.min_support, check=check)
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
        print(f"\nsaved {len(skills)} skill(s) -> {args.save} (unapproved)")
        print(f"review the steps, then arm one: loom skills {args.save} --approve <name>")
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

    if not args.replay:
        _recover_wirelog(args.save)
    shield = _build_shield(args)
    try:
        server = ProxyServer(
        port=args.port,
        target=args.target,
        save_path=args.save if not args.replay else None,
        replay_path=args.replay or None,
        shield=shield if not args.replay else None,
        scrub=args.scrub,
            max_body=args.max_body_mb * 1024 * 1024,
            upstream_timeout=args.upstream_timeout,
            auth=args.auth,
        )
    except ValueError:
        raise CLIError(
            f"{args.replay} has no recorded wire traffic -- it's a harness trace, "
            f"not a proxy recording. `loom replay {args.replay}` replays those; "
            f"`loom proxy --replay` needs a trace made by `loom record`/`loom proxy`."
        )
    except FileNotFoundError:
        raise CLIError(f"no such file: {args.replay}")
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
    finally:
        server.finalize()
    return 0


def _control_headers(port: int) -> dict:
    """Auth header for a shielded proxy's control endpoints (loom approve...)."""
    from .proxy import control_token_for

    token = control_token_for(port)
    return {"x-loom-token": token} if token else {}


def _cmd_approvals(args: argparse.Namespace) -> int:
    """List a running proxy's pending shield approvals."""
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        f"http://127.0.0.1:{args.port}/loom/shield/pending",
        headers=_control_headers(args.port),
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            pending = json.load(r).get("pending", [])
    except urllib.error.HTTPError as e:
        print(f"loom: {json.load(e).get('error', e.reason)}", file=sys.stderr)
        return 1
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
        headers={"content-type": "application/json", **_control_headers(args.port)},
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


def _cmd_scrub(args: argparse.Namespace) -> int:
    """Redact secrets from a saved trace (or just report them with --check)."""
    from .scrub import scrub_trace

    data = _load_trace_json(args.path)
    clean, found = scrub_trace(data, aggressive=args.aggressive)
    total = sum(found.values())
    for kind in sorted(found):
        print(f"  {found[kind]:>3}x {kind}")

    if args.check:
        if total:
            print(f"loom: {total} secret(s) in {args.path}", file=sys.stderr)
            return 1
        print(f"clean: {args.path}")
        return 0

    if args.in_place:
        out = args.path
    elif args.path.endswith(".loom.json"):
        out = args.path[: -len(".loom.json")] + ".scrubbed.loom.json"
    else:
        out = args.path + ".scrubbed"
    if "checksum" in clean:  # scrubbing is a deliberate edit: re-stamp it
        from .trace import trace_checksum

        clean["checksum"] = trace_checksum(clean)
    with open(out, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"scrubbed {total} secret(s) -> {out}")
    return 0


def _cmd_why(args: argparse.Namespace) -> int:
    """Ask a debugger agent a question about a saved trace."""
    from .why import why

    try:
        run = why(args.path, args.question, model=args.model)
    except Exception as e:  # surface provider/auth errors cleanly
        print(f"loom: why failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(run.output)
    if args.save:
        run.save(args.save)
        print(f"\nsaved diagnosis trace -> {args.save}", file=sys.stderr)
    return 0


def _cmd_trust(args: argparse.Namespace) -> int:
    """Show (or demote entries in) the shield's trust ledger."""
    import os

    from .shield import TrustLedger

    path = args.ledger or os.path.join(os.path.expanduser("~"), ".loom", "trust.json")
    ledger = TrustLedger(path)
    if args.demote:
        if ledger.demote(args.demote):
            print(f"demoted {args.demote}: streak reset to 0")
            return 0
        print(f"loom: no trust recorded for {args.demote!r}", file=sys.stderr)
        return 1
    if not ledger.data:
        print(f"no trust recorded yet ({path})")
        return 0
    for tool, entry in sorted(ledger.data.items()):
        ids = ", ".join(e.get("id", "?") for e in entry.get("evidence", [])[-5:])
        line = f"{tool}: streak {entry.get('streak', 0)}"
        if ids:
            line += f"  (recent approvals: {ids})"
        print(line)
    return 0


def _cmd_migrate(args: argparse.Namespace) -> int:
    """Bring traces to the current format version (and re-stamp checksums)."""
    from .migrate import migrate

    agent = None
    if args.agent:
        agent, err = _load_agent(args.agent)
        if agent is None:
            print(f"loom: {err}", file=sys.stderr)
            return 2

    paths = _expand_trace_paths(args.paths)
    if not paths:
        print("no traces found", file=sys.stderr)
        return 2
    for path in paths:
        _load_trace_json(path)  # friendly errors for non-traces
        try:
            rekeyed, out = migrate(path, agent=agent, out=args.output or None)
        except ValueError as e:
            raise CLIError(str(e))
        detail = f"{rekeyed} effect key(s) recomputed" if rekeyed else "re-stamped"
        print(f"migrated {path} -> {out}  ({detail})")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    """Query an auto-indexed directory of traces."""
    from .lake import Lake

    lake = Lake(args.directory)
    lake.index()
    try:
        rows = lake.search(args.query)
    except ValueError as e:
        print(f"loom: {e}", file=sys.stderr)
        return 2
    finally:
        lake.close()
    if not rows:
        print("no matching runs")
        return 1
    for r in rows:
        tokens = r["input_tokens"] + r["output_tokens"]
        flags = []
        if r["stop_reason"] not in ("end_turn", ""):
            flags.append(r["stop_reason"])
        if r["shield_denies"]:
            flags.append(f"shield:{r['shield_denies']}")
        prompt = (r["episodes"] or "").split(" | ")[0][:60]
        print(f"{tokens:>9,} tok  {r['path']}  {prompt!r}"
              + (f"  [{', '.join(flags)}]" if flags else ""))
    print(f"\n{len(rows)} run(s)")
    return 0


def _cmd_lake(args: argparse.Namespace) -> int:
    """Index a trace corpus and render its cost dashboard."""
    import os

    from .lake import Lake, dashboard_html

    lake = Lake(args.directory)
    fresh, total = lake.index()
    stats = lake.stats()
    lake.close()
    print(f"indexed {total} run(s) ({fresh} new/changed) in {args.directory}")
    print(f"  tokens: {stats['input_tokens'] + stats['output_tokens']:,}"
          f"  failed: {stats['failed']}  shield blocks: {stats['denies']}")
    out = args.output or os.path.join(args.directory, "lake.html")
    with open(out, "w") as f:
        f.write(dashboard_html(stats, args.directory))
    print(f"dashboard -> {out}")
    if args.open:
        import webbrowser

        webbrowser.open(f"file://{os.path.abspath(out)}")
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
        sp.add_argument("--rule", action="append", default=[], metavar="RULE",
                        help="sequence rule, e.g. 'after Read(*.env*): deny WebFetch*, "
                             "deny Bash(*curl*)' or 'taint sk-ant-*: confirm *' (repeatable)")
        sp.add_argument("--confirm-timeout", type=float, default=300.0,
                        help="seconds to wait for approval before denying (default 300)")
        sp.add_argument("--webhook", default="",
                        help="POST pending approvals to this URL (approval inbox)")
        sp.add_argument("--shield-default", dest="shield_default", default="allow",
                        choices=["allow", "confirm", "deny"],
                        help="action when no rule matches (default: allow)")
        sp.add_argument("--judge", default="", metavar="MODEL",
                        help="risk-score unmatched tool calls with this model; "
                             "risky ones are held for approval")
        sp.add_argument("--judge-threshold", dest="judge_threshold", type=float, default=0.7,
                        help="risk score at which the judge escalates to confirm (default 0.7)")
        sp.add_argument("--trust-after", dest="trust_after", type=int, default=0, metavar="N",
                        help="auto-approve a tool's confirms after N consecutive "
                             "operator approvals (see: loom trust)")
        sp.add_argument("--trust-ledger", dest="trust_ledger", default="", metavar="FILE",
                        help="where approval streaks live (default ~/.loom/trust.json)")

    def scrub_flag(sp) -> None:
        sp.add_argument("--scrub", action="store_true",
                        help="redact secrets (API keys, tokens...) before the trace is written")
        sp.add_argument("--max-body-mb", dest="max_body_mb", type=int, default=64,
                        help="reject request bodies larger than this (default 64; 0 = no cap)")
        sp.add_argument("--upstream-timeout", dest="upstream_timeout", type=float, default=600.0,
                        help="seconds to wait on the upstream API (default 600)")
        sp.add_argument("--auth", default="",
                        help="require this token in x-loom-auth on data-plane requests "
                             "(guards replay serving; needs an agent that can add a header)")

    rc = sub.add_parser("record", help="black-box a real agent session: loom record -- <command>")
    rc.add_argument("command", nargs=argparse.REMAINDER, help="the agent command to run")
    rc.add_argument("--save", default="session.loom.json")
    rc.add_argument("--target", default="https://api.anthropic.com",
                    help="upstream API (use https://api.openai.com for OpenAI agents)")
    rc.add_argument("--port", type=int, default=0, help="proxy port (default: pick a free one)")
    rc.add_argument("--sandbox", action="store_true",
                    help="deny the agent ALL network except the proxy (macOS sandbox-exec); "
                         "shield rules become impossible to bypass")
    rc.add_argument("--sandbox-allow", action="append", default=[], metavar="HOST:PORT",
                    help="extra host:port the sandboxed agent may reach (repeatable)")
    shield_flags(rc)
    scrub_flag(rc)
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
    sk.add_argument("paths", nargs="+",
                    help="trace files or directories of *.loom.json "
                         "(with --approve: the skill library JSON)")
    sk.add_argument("--min-support", type=int, default=2, dest="min_support")
    sk.add_argument("--require", default="",
                    help="only mine runs whose output contains this (success filter)")
    sk.add_argument("--forbid", default="",
                    help="only mine runs whose output does NOT contain this")
    sk.add_argument("--save", default="", help="write the skill library to this JSON file")
    sk.add_argument("--approve", default="", metavar="NAME",
                    help="mark a skill in the given library as human-approved")
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
    scrub_flag(px)
    px.set_defaults(func=_cmd_proxy)

    av = sub.add_parser("approvals", help="list pending shield approvals on a running proxy")
    av.add_argument("--port", type=int, default=8788)
    av.set_defaults(func=_cmd_approvals)

    ap = sub.add_parser("approve", help="approve (or --deny) a pending shield tool call")
    ap.add_argument("id")
    ap.add_argument("--deny", action="store_true", help="deny instead of approving")
    ap.add_argument("--port", type=int, default=8788)
    ap.set_defaults(func=_cmd_approve)

    mg = sub.add_parser("migrate", help="bring traces to the current format version")
    mg.add_argument("paths", nargs="+", help="trace files or directories of *.loom.json")
    mg.add_argument("--agent", default="",
                    help="module:attr of the recording agent (harness traces only)")
    mg.add_argument("-o", "--output", default="",
                    help="write here instead of in place (single trace only)")
    mg.set_defaults(func=_cmd_migrate)

    se = sub.add_parser("search", help="query a directory of traces (auto-indexed)")
    se.add_argument("directory")
    se.add_argument("query", nargs="?", default="",
                    help="terms: cost>N cost<N tool:NAME model:GLOB failed "
                         "shield:deny healed, plus free text (all must hold)")
    se.set_defaults(func=_cmd_search)

    lk = sub.add_parser("lake", help="index a trace corpus + cost dashboard HTML")
    lk.add_argument("directory")
    lk.add_argument("-o", "--output", default="", help="dashboard path (default <dir>/lake.html)")
    lk.add_argument("--open", action="store_true", help="open the dashboard in a browser")
    lk.set_defaults(func=_cmd_lake)

    sc = sub.add_parser("scrub", help="redact secrets from a trace before sharing it")
    sc.add_argument("path")
    sc.add_argument("--in-place", action="store_true", dest="in_place",
                    help="overwrite the trace instead of writing *.scrubbed.loom.json")
    sc.add_argument("--check", action="store_true",
                    help="report only; exit 1 if secrets are found (CI gate)")
    sc.add_argument("--aggressive", action="store_true",
                    help="also redact long high-entropy tokens (may false-positive)")
    sc.set_defaults(func=_cmd_scrub)

    wy = sub.add_parser("why", help="ask a debugger agent about a saved trace")
    wy.add_argument("path")
    wy.add_argument("question")
    wy.add_argument("--model", default="claude-opus-4-8")
    wy.add_argument("--save", default="", help="record the diagnosis run to this path")
    wy.set_defaults(func=_cmd_why)

    tr = sub.add_parser("trust", help="show the shield's trust ledger (approval streaks)")
    tr.add_argument("--ledger", default="", help="ledger file (default ~/.loom/trust.json)")
    tr.add_argument("--demote", default="", metavar="TOOL",
                    help="reset a tool's streak so its confirms need approval again")
    tr.set_defaults(func=_cmd_trust)

    wa = sub.add_parser("watch", help="follow a run's journal live")
    wa.add_argument("path")
    wa.add_argument("--interval", type=float, default=0.5)
    wa.add_argument("--once", action="store_true", help="print current state and exit")
    wa.set_defaults(func=_cmd_watch)
    return p


def main(argv: "list[str] | None" = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except CLIError as e:
        print(f"loom: {e}", file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f"loom: no such file: {getattr(e, 'filename', e)}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as e:
        print(f"loom: invalid JSON (line {e.lineno}) -- expected a loom trace", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
