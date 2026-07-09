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
    """No path: check the environment. With a path: check a trace for context rot."""
    if args.path:
        from .health import analyze

        log, data = _load_log(args.path)
        episodes = data.get("episodes") or [data.get("prompt", "")]
        report = analyze(episodes, log)
        print(report.summary())
        return 0 if report.ok else 1
    return _doctor_environment()


def _doctor_environment() -> int:
    """Will a recording actually work here? Check versions, agents, extras."""
    import importlib.util
    import os
    import platform
    import shutil

    from . import __version__

    ok = "✅"
    warn = "⚠️ "
    print(f"loom {__version__}   python {platform.python_version()}   {platform.system()}")
    print()

    problems = 0
    print("agents (loom record <name> \"...\"):")
    any_agent = False
    for name, (_, binary, is_openai) in sorted(AGENTS.items()):
        found = shutil.which(binary)
        env = "OPENAI_BASE_URL" if is_openai else "ANTHROPIC_BASE_URL"
        if found:
            any_agent = True
            print(f"  {ok} {name:8} {found}  (honors {env})")
        else:
            print(f"  {warn}{name:8} not on PATH")
    if not any_agent:
        print("     (install Claude Code or Codex, or use `loom record -- <command>`)")

    print("\noptional extras:")
    for extra, module, what in [
        ("anthropic", "anthropic", "live Claude runs / the LLM judge"),
        ("openai", "openai", "OpenAI agents / embeddings"),
        ("yaml", "yaml", "full YAML for policy files (a bounded parser ships built in)"),
        ("mcp", "mcp", "MCP tool servers"),
    ]:
        present = importlib.util.find_spec(module) is not None
        mark = ok if present else warn
        note = "" if present else f"  (pip install \"loom-harness[{extra}]\")"
        print(f"  {mark}{module:10} {what}{note}")

    print("\nfirewall profiles (--profile):")
    from .policy_file import profile_names

    print("  " + ", ".join(profile_names()))

    runtime = os.path.join(os.path.expanduser("~"), ".loom")
    try:
        os.makedirs(runtime, exist_ok=True)
        testfile = os.path.join(runtime, ".doctor-write-test")
        with open(testfile, "w") as f:
            f.write("ok")
        os.remove(testfile)
        print(f"\n{ok} runtime dir writable: {runtime}")
    except OSError as e:
        problems += 1
        print(f"\n{warn}runtime dir not writable ({runtime}): {e}")

    print()
    if problems:
        print(f"{problems} problem(s) found.")
        return 1
    print("ready to record.")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    """Render a saved trace to HTML, or flatten it to observability events."""
    if args.jsonl or args.otel:
        from .events import export_events

        paths = _expand_trace_paths([args.path])
        dest = args.jsonl or "-"
        if dest == "-":
            n = export_events(paths, sys.stdout, otel=args.otel)
        else:
            with open(dest, "w") as f:
                n = export_events(paths, f, otel=args.otel)
            print(f"wrote {n} event(s) -> {dest}", file=sys.stderr)
        return 0

    from .export import trace_to_html

    data = _load_trace_json(args.path)
    out = args.output or (args.path.rsplit(".json", 1)[0] + ".html")
    with open(out, "w") as f:
        f.write(trace_to_html(data, path=args.path))
    print(f"wrote {out}")
    return 0


def _build_shield(args: argparse.Namespace):
    """Turn --profile/--policy plus --deny/--confirm/--allow/--judge/--rule into a Shield."""
    profile = getattr(args, "profile", "")
    policy_path = getattr(args, "policy", "")
    has_flags = (args.deny or args.confirm or args.allow or args.judge or args.rule
                 or args.shield_default != "allow")
    if not (has_flags or profile or policy_path):
        return None
    from .shield import Shield, TrustLedger

    # Start from the resolved policy (profile and/or file), then let explicit
    # command-line flags extend it -- the flags always win by being additive.
    policy: dict = {}
    if profile or policy_path:
        from .policy_file import resolve, to_shield_kwargs

        policy = to_shield_kwargs(resolve(profile=profile, policy_path=policy_path))
    default = policy.get("default", "allow")
    if args.shield_default != "allow":  # an explicit flag overrides the policy
        default = args.shield_default

    trust = None
    if args.trust_after > 0:
        import os

        ledger_path = args.trust_ledger or os.path.join(
            os.path.expanduser("~"), ".loom", "trust.json"
        )
        trust = TrustLedger(ledger_path)
    import os

    sign_key = None
    if getattr(args, "sign_approvals_key_env", ""):
        val = os.environ.get(args.sign_approvals_key_env)
        if not val:
            raise CLIError(f"env var {args.sign_approvals_key_env} is not set")
        sign_key = val.encode()
    # Approver policy from --require-approver 'pattern=a,b' plus any in the file.
    approvers = dict(policy.get("approvers", {}) or {})
    for spec in getattr(args, "require_approver", []) or []:
        pattern, _, names = spec.partition("=")
        if not names.strip():
            raise CLIError(f"--require-approver needs 'PATTERN=NAMES', got {spec!r}")
        approvers[pattern.strip()] = [n.strip() for n in names.split(",") if n.strip()]

    return Shield(
        deny=list(policy.get("deny", [])) + (args.deny or []),
        confirm=list(policy.get("confirm", [])) + (args.confirm or []),
        allow=list(policy.get("allow", [])) + (args.allow or []),
        default=default,
        timeout=args.confirm_timeout,
        webhook=args.webhook,
        judge=args.judge or None,
        judge_threshold=args.judge_threshold,
        trust=trust,
        trust_after=args.trust_after,
        sequence=list(policy.get("sequence", [])) + (args.rule or []),
        sign_key=sign_key,
        approvers=approvers,
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


# Known coding agents: name -> (build argv from a prompt, the binary to check,
# is it an OpenAI-dialect agent?). The shortcut form `loom record claude "..."`
# expands through this; `loom record -- <anything>` bypasses it.
AGENTS: dict = {
    "claude": (lambda p: ["claude", "-p", p], "claude", False),
    "codex": (lambda p: ["codex", "exec", p], "codex", True),
}


def _expand_agent_shortcut(command: "list[str]", args) -> "tuple[list[str], str]":
    """Turn `[agent, prompt]` into the real argv. Returns (command, error)."""
    if len(command) >= 1 and command[0] in AGENTS and not (
        len(command) > 1 and command[1].startswith("-")
    ):
        name = command[0]
        build, binary, is_openai = AGENTS[name]
        if len(command) < 2:
            return command, (f"`loom record {name}` needs a prompt, "
                             f'e.g. loom record {name} "fix the failing test"')
        prompt = " ".join(command[1:])
        import shutil

        if shutil.which(binary) is None:
            return command, (f"{binary!r} is not on your PATH -- install the agent first, "
                             f"or use the passthrough form: loom record -- <command>")
        if is_openai and args.target == "https://api.anthropic.com":
            # Route codex to the right dialect -- but only when the target is
            # still the default; an explicit --target (a local mock, a vLLM
            # endpoint) must never be silently replaced with a real API.
            args.target = "https://api.openai.com"
        return build(prompt), ""
    return command, ""


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
        print("loom: record needs a command: `loom record claude \"fix the test\"` "
              "or `loom record -- <command>`", file=sys.stderr)
        return 2

    command, err = _expand_agent_shortcut(command, args)
    if err:
        print(f"loom: {err}", file=sys.stderr)
        return 2

    if args.safe:  # one flag = the sane coding-agent defaults
        args.profile = args.profile or "claude-code-safe"
        args.scrub = True
        args.report = True
        print("safe mode:", file=sys.stderr)
        print(f"  ✓ policy: {args.profile}", file=sys.stderr)
        print("  ✓ scrub on   ✓ report on", file=sys.stderr)
        if args.sandbox:
            print("  ✓ sandbox on", file=sys.stderr)
        elif sys.platform == "darwin":
            print("  ! sandbox off — add --sandbox for full network isolation", file=sys.stderr)
        else:
            print("  ! sandbox not built in on this OS — see examples/docker-sandbox "
                  "for full isolation", file=sys.stderr)

    if args.container and args.sandbox:
        print("loom: use --container OR --sandbox, not both (container includes network "
              "isolation)", file=sys.stderr)
        return 2

    shield = _build_shield(args)
    # A container reaches the host proxy via host.docker.internal, so the proxy
    # must bind beyond loopback.
    proxy_host = "0.0.0.0" if args.container else "127.0.0.1"
    server = ProxyServer(port=args.port, target=args.target, save_path=args.save,
                         shield=shield, scrub=args.scrub,
                         max_body=args.max_body_mb * 1024 * 1024,
                         upstream_timeout=args.upstream_timeout, auth=args.auth,
                         host=proxy_host)
    before_snap = None
    if not args.no_workspace:
        from .workspace import collect, diff_snapshot

        ws = collect(command=command, target=args.target)
        server.recorder.workspace = ws
        if ws.get("git"):  # snapshot the working tree so we can diff the delta
            before_snap = diff_snapshot(os.getcwd())
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

    if args.container:
        from .sandbox import wrap_container

        try:
            command = wrap_container(command, port=server.port, image=args.container_image,
                                     workdir=os.getcwd(), target=args.target,
                                     read_only=args.container_readonly,
                                     memory=args.container_memory, cpus=args.container_cpus)
        except RuntimeError as e:
            print(f"loom: {e}", file=sys.stderr)
            server.shutdown()
            server.finalize()
            return 2
        print(f"loom record: containerized in {args.container_image} -- repo mounted, "
              "API routed through the proxy", file=sys.stderr)

    try:
        code = subprocess.call(command, env=env)
    finally:
        if profile_path:
            os.unlink(profile_path)
    # What did the agent do to the workspace? Diff the tree against the
    # pre-run snapshot before the trace is finalized.
    if before_snap is not None and server.recorder.workspace is not None:
        from .workspace import changes_since, diff_snapshot

        after_snap = diff_snapshot(os.getcwd())
        server.recorder.workspace["changes"] = changes_since(
            before_snap, after_snap, agent_exit_code=code, capture_diff=args.capture_diff,
            cwd=os.getcwd(),
        )
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
            calls = ", ".join(
                f"{e.get('tool')}({json.dumps(e.get('input', {}), sort_keys=True, default=str)})"
                for e in blocked[:3])
            print(f"🛡️ shield blocked {len(blocked)} risky call(s): {calls}", file=sys.stderr)
        if args.report:
            base = args.save[: -len(".loom.json")] if args.save.endswith(".loom.json") else args.save
            from .export import trace_to_html
            from .incident import build_report

            html_path = base + ".html"
            with open(html_path, "w") as f:
                f.write(trace_to_html(data, path=args.save))
            md_path = base + ".incident.md"
            with open(md_path, "w") as f:
                f.write(build_report(data, args.save) + "\n")
            print(f"report:    {html_path} + {md_path}", file=sys.stderr)
            # A safe-to-share copy: scrub the trace, stamp it, so it can go
            # straight into an issue alongside the incident report.
            from .scrub import scrub_trace
            from .trace import trace_checksum

            shared, found = scrub_trace(data)
            if "checksum" in shared:
                shared["checksum"] = trace_checksum(shared)
            shared["scrubbed"] = True
            shared_path = base + ".shared.loom.json"
            with open(shared_path, "w") as f:
                json.dump(shared, f, indent=2)
            print(f"shareable: {shared_path}"
                  + (f"   ({sum(found.values())} secret(s) scrubbed)" if found else "   (secrets scrubbed)"),
                  file=sys.stderr)
        print(f"replay:    loom replay {args.save}", file=sys.stderr)
        print(f"inspect:   loom studio {args.save}", file=sys.stderr)
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
    print("diagnosis:")
    print("  " + report.summary().replace("\n", "\n  "))

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
        print("\n❌ no single context repair fixed the run (tried redacting each "
              "suspect item). The failure may not be context rot.")
        return 1

    what = _explain_repair(healed.healed_by)
    print(f"\n✅ fixed by {what}")
    print(f"   before: {run.output[:100]!r}")
    print(f"   after:  {healed.output[:100]!r}")
    if healed.regression_path:
        print(f"\n   saved as a regression test -> {healed.regression_path}")
        print(f"   it runs in CI with:  loom test {args.save_regression}")
    else:
        print("\n   keep it as a regression test:  "
              "add --save-regression tests/traces")
    return 0


def _explain_repair(healed_by: str) -> str:
    """Turn a healed_by tag like 'redact-oversized-0' into a sentence."""
    parts = (healed_by or "").split("-")
    if len(parts) >= 2 and parts[0] == "redact":
        kind = parts[1]
        reason = {
            "oversized": "redacting the oversized context item that was crowding out the answer",
            "unused": "redacting a context item nothing later referenced",
            "duplicate": "redacting a duplicated context item",
        }.get(kind, f"redacting the {kind} context item")
        return f"{reason} (`{healed_by}`)"
    return f"`{healed_by}`"


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

    if getattr(args, "actions", False):
        # Behavior diff: what does run B DO differently -- new/removed actions
        # and the exercised-risk movement. The PR-review view.
        from .diff import describe_action_diff, diff_actions

        data_a = _load_trace_json(args.a)
        data_b = _load_trace_json(args.b)
        d = diff_actions(data_a, data_b)
        if getattr(args, "json", False):
            print(json.dumps(d, indent=2))
        else:
            print(describe_action_diff(d))
        changed = d["added"] or d["removed"] or d["score"]["a"] != d["score"]["b"]
        return 1 if changed else 0

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
            json.dump(to_json(impacts, agent=agent), f, indent=2)
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
            host=args.host,
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
    live_url = f"http://127.0.0.1:{server.port}/loom/live"
    if server.control_token:  # the page is token-gated when a shield is active
        live_url += f"?token={server.control_token}"
    print(f"  live studio: {live_url}")
    if args.live:
        import webbrowser

        # Open once the server is accepting connections (a beat after start).
        threading.Timer(0.6, lambda: webbrowser.open(live_url)).start()
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
    import getpass
    import os as _os

    who = args.by or _os.environ.get("LOOM_APPROVER") or getpass.getuser()
    req = urllib.request.Request(
        f"http://127.0.0.1:{args.port}/loom/shield/decide",
        data=json.dumps({"id": args.id, "decision": decision, "by": who}).encode(),
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
    from .scrub import load_scrub_config, scrub_trace

    config = None
    if args.config:
        config = load_scrub_config(args.config)

    data = _load_trace_json(args.path)

    if args.audit:
        from .scrub import audit_report

        report = audit_report(data, aggressive=args.aggressive, config=config)
        out = args.audit if args.audit != "-" else None
        text = json.dumps(report, indent=2)
        if out:
            with open(out, "w") as f:
                f.write(text + "\n")
            print(f"audit -> {out}  ({report['total']} redaction(s))")
        else:
            print(text)
        return 0

    clean, found = scrub_trace(data, aggressive=args.aggressive, config=config)
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


def _import_builtin_packs() -> None:
    """Register every built-in domain pack (for pack-aware CLI views)."""
    from .packs import install_builtin

    install_builtin()


def _cmd_undo(args: argparse.Namespace) -> int:
    """Revert the file changes an agent made, from the trace's workspace record."""
    import os

    from .undo import undo

    data = _load_trace_json(args.path)

    if args.plan:
        # The generic view: per-action undo/compensation plans from whichever
        # domain pack owns each action (files, SQL, browser, CRM) -- newest
        # first, because undo runs backwards.
        from .action import actions as _actions
        from .packs import undo_plan as _undo_plan

        _import_builtin_packs()
        shown = 0
        for a in reversed([x for x in _actions(data) if x.type == "call"]):
            plan = _undo_plan(a, data)
            if plan is None:
                continue
            shown += 1
            mark = {"revert": "↩", "compensate": "⇄", "noop": "✋"}.get(plan.kind, "·")
            rev = "" if plan.reversible else "  [irreversible -- compensation only]"
            step = f"[{a.step:>3}]" if a.step >= 0 else "[ - ]"
            print(f"  {mark} {step} {a.tool}: {plan.summary}{rev}")
            for c in plan.commands:
                print(f"          $ {c}")
        if not shown:
            print("nothing to undo (no reversible actions recorded)")
        else:
            print(f"\n{shown} plan(s). File reverts can run via `loom undo {args.path}`;"
                  " other domains list the compensating steps above.")
        return 0

    ok, log = undo(data, os.getcwd(), only=args.only, dry_run=args.dry_run, force=args.force)
    for line in log:
        print(line)
    if not ok:
        return 1
    return 0


def _cmd_fork(args: argparse.Namespace) -> int:
    """Fork a recorded run at a step/turn: replay the prefix free, continue live."""
    from .trace import Run

    data = _load_trace_json(args.path)
    log = data.get("log", [])
    turn_seqs = [e["seq"] for e in log
                 if e.get("kind") == "model" and not e.get("depth", 0)]
    if not turn_seqs:
        raise CLIError("this trace has no model turns to fork from")

    if args.turn is not None:
        turn = args.turn
    else:
        # --from-step S rewinds to the turn containing step S.
        matching = [i for i, s in enumerate(turn_seqs) if s <= args.from_step]
        turn = matching[-1] if matching else 0
    if turn < 0 or turn >= len(turn_seqs):
        raise CLIError(f"turn {turn} out of range (run has {len(turn_seqs)} turns)")
    fork_seq = turn_seqs[turn]

    # Ask the owning packs how to restore external state before re-running:
    # replaying is free, but the WORLD (db, browser, customers) isn't rewound.
    from .action import actions as _actions
    from .packs import restore_plans

    _import_builtin_packs()
    # Only the domains touched from the fork point onward need restoring.
    touched = [a for a in _actions(data) if a.type == "call" and a.step >= fork_seq]
    plans = restore_plans(touched, data)

    print(f"fork at turn {turn} (step {fork_seq}): "
          f"steps 0..{max(fork_seq - 1, 0)} replay free, the rest runs live")
    if plans:
        print("restore external state before continuing live:")
        for name, plan in plans:
            mark = "▶ runnable" if plan.executable else "manual"
            print(f"  [{name}] ({mark}) {plan.summary}")
            for c in plan.commands:
                print(f"      $ {c}")

    if not args.agent:
        print("\nno --agent given (module:attr), so nothing was run. To continue live:")
        print(f"  loom fork {args.path} --turn {turn} --agent yourmodule:agent")
        print("or in Python:")
        print(f'  run = Run.load("{args.path}", agent=agent)')
        print(f"  branch = run.fork(at={turn}, edit=lambda ctx: ...)")
        return 0

    agent, err = _load_agent(args.agent)
    if agent is None:
        raise CLIError(err)
    run = Run.load(args.path, agent=agent)
    edit = None
    if args.inject:
        note = args.inject

        def edit(ctx, _note=note):  # a recorded edit: replays deterministically
            ctx.add_user(_note, source="fork-edit")

    branch = run.fork(at=turn, edit=edit)
    base = args.path[: -len(".loom.json")] if args.path.endswith(".loom.json") else args.path
    out = args.output or f"{base}.fork{turn}.loom.json"
    branch.save(out)
    spent = branch.cost(since=fork_seq)["total_tokens"]
    print(f"\nbranch: {out}")
    print(f"  output: {branch.output[:200]}")
    print(f"  live tokens spent after the fork: {spent}")
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    """The behavior scorecard: security / side-effect / reversibility / ..."""
    from .diff import describe_score, score_breakdown

    _import_builtin_packs()
    b = score_breakdown(_load_trace_json(args.path))
    if args.json:
        print(json.dumps(b, indent=2))
    else:
        print(describe_score(b))
    return 0


def _cmd_taint(args: argparse.Namespace) -> int:
    """Trace sensitive VALUES from where they were read to where they left."""
    from .taint import describe_taint, taint_paths

    _import_builtin_packs()
    paths = taint_paths(_load_trace_json(args.path))
    print(describe_taint(paths))
    return 1 if (paths and args.fail_on_leak) else 0


def _cmd_map(args: argparse.Namespace) -> int:
    """The side-effect map: everything the run changed or reached, one view."""
    from .insight import describe_map, side_effect_map

    _import_builtin_packs()
    print(describe_map(side_effect_map(_load_trace_json(args.path))))
    return 0


def _cmd_graph(args: argparse.Namespace) -> int:
    """The delegation/causality tree across subagent depths."""
    from .insight import causality_tree

    _import_builtin_packs()
    out = causality_tree(_load_trace_json(args.path))
    print(out or "no actions recorded")
    return 0


def _cmd_provenance(args: argparse.Namespace) -> int:
    """Link each claim in the final answer to the tool results behind it."""
    from .insight import provenance

    _import_builtin_packs()
    rows = provenance(_load_trace_json(args.path))
    if not rows:
        print("no final answer to attribute")
        return 1
    unsupported = 0
    for row in rows:
        print(f"• {row['claim']}")
        if row["evidence"]:
            for e in row["evidence"]:
                print(f"    ⤷ [{e['step']}] {e['tool']}: {e['snippet']}")
        else:
            unsupported += 1
            print("    ⤷ (no supporting tool result found)")
    if unsupported:
        print(f"\n⚠️  {unsupported} claim(s) without supporting evidence")
    return 0


def _cmd_flake(args: argparse.Namespace) -> int:
    """Divergence heatmap across repeated recordings of the same task."""
    from .insight import describe_flakiness, flakiness

    paths = _expand_trace_paths(args.paths)
    traces = []
    for p in paths:
        try:
            with open(p) as f:
                traces.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            continue
    if len(traces) < 2:
        raise CLIError("flake needs at least two traces of the same task")
    print(describe_flakiness(flakiness(traces)))
    return 0


def _cmd_note(args: argparse.Namespace) -> int:
    """Annotate a trace step (sidecar file; the seed of the replay room)."""
    import os
    import time

    _load_trace_json(args.path)  # friendly error for non-traces
    notes_path = args.path + ".notes.json"
    try:
        with open(notes_path) as f:
            notes = json.load(f)
    except (OSError, json.JSONDecodeError):
        notes = []

    if args.message:
        who = args.by or os.environ.get("USER", "")
        notes.append({"step": args.step, "by": who, "text": args.message,
                      "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
        with open(notes_path, "w") as f:
            json.dump(notes, f, indent=2)
        print(f"noted step {args.step} -> {notes_path}")
        return 0

    if not notes:
        print("no notes yet (add one: loom note <trace> --step N -m 'text')")
        return 0
    for n in sorted(notes, key=lambda x: (x.get("step") or 0)):
        who = f" — {n['by']}" if n.get("by") else ""
        print(f"  [{n.get('step', '?'):>3}] {n['text']}{who}  ({n.get('ts', '')})")
    return 0


def _load_yaml_or_json(path: str) -> dict:
    """Load a .yml/.json doc via the bounded policy parser (pyyaml if present)."""
    from .policy_file import _parse

    try:
        with open(path) as f:
            return _parse(f.read(), path) or {}
    except OSError as e:
        raise CLIError(f"cannot read {path}: {e}")


def _cmd_packs(args: argparse.Namespace) -> int:
    """List, lint, or test domain packs."""
    if getattr(args, "packs_cmd", None) == "lint":
        from .packs.certify import lint_pack, load_pack

        try:
            pack = load_pack(args.pack)
        except (ImportError, AttributeError, ValueError) as e:
            raise CLIError(f"could not load pack {args.pack!r}: {e}")
        problems = lint_pack(pack)
        for p in problems:
            print(f"  ⚠️  {p}")
        if problems:
            print(f"\n{len(problems)} issue(s) — this pack may mislabel actions", file=sys.stderr)
            return 1
        print(f"pack {pack.name!r} looks good ✓")
        return 0

    if getattr(args, "packs_cmd", None) == "test":
        from .packs.certify import load_pack, test_pack

        cases_doc = _load_yaml_or_json(args.cases)
        spec = args.pack or cases_doc.get("pack")
        if not spec:
            raise CLIError("provide --pack module:attr, or a 'pack:' key in the cases file")
        try:
            pack = load_pack(spec)
        except (ImportError, AttributeError, ValueError) as e:
            raise CLIError(f"could not load pack {spec!r}: {e}")
        results = test_pack(pack, cases_doc.get("cases", []))
        failed = 0
        for r in results:
            if r["ok"]:
                print(f"  ok   {r['tool']}")
            else:
                failed += 1
                print(f"  FAIL {r['tool']}")
                for f in r["failures"]:
                    print(f"         - {f}")
        print(f"\n{len(results) - failed}/{len(results)} case(s) passed")
        return 1 if failed else 0

    from importlib import metadata

    from .packs import install_builtin, packs, register

    install_builtin()
    # Third-party packs: any installed package exposing a "loom.packs" entry
    # point is discovered here -- the marketplace is pip itself.
    plugin_names = set()
    for ep in metadata.entry_points(group="loom.packs"):
        try:
            obj = ep.load()
            pack = obj() if isinstance(obj, type) else obj
            register(pack)
            plugin_names.add(pack.name)
        except Exception as e:  # a broken plugin shouldn't kill the listing
            print(f"  ⚠️  plugin {ep.name!r} failed to load: {e}", file=sys.stderr)

    import importlib

    for p in packs():
        doc = p.__class__.__doc__ or ""
        if not doc:  # the built-ins document at module level
            mod = importlib.import_module(p.__class__.__module__)
            doc = mod.__doc__ or ""
        first = doc.strip().splitlines()[0] if doc.strip() else ""
        origin = "plugin" if p.name in plugin_names else "built-in"
        print(f"  {p.name:<10} [{origin}]  {first}")
    print("\ninstall more: pip install <package> — any package with a "
          "'loom.packs' entry point is discovered automatically")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    """Serve a trace directory to the team: list, search, Studio, incidents."""
    import os

    from .serve import TraceServer

    if not os.path.isdir(args.directory):
        raise CLIError(f"{args.directory} is not a directory")
    server = TraceServer(args.directory, host=args.host, port=args.port)
    print(f"loom trace server: {server.url}  (dir: {os.path.abspath(args.directory)})")
    if args.host not in ("127.0.0.1", "localhost"):
        print("  ⚠️  serving beyond localhost -- there is no auth; trusted networks only")
    print("  Ctrl-C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.shutdown()
    return 0


def _cmd_tools(args: argparse.Namespace) -> int:
    """Show the capability contract of an agent's tools."""
    from .capabilities import manifest

    agent, err = _load_agent(args.agent)
    if agent is None:
        raise CLIError(err)
    rows = manifest(agent.tools)
    if not rows:
        print("this agent has no tools")
        return 0
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    for r in rows:
        mark = "declared" if r["declared"] else "inferred"
        caps = ", ".join(r["capabilities"]) or "(none)"
        print(f"  {r['tool']:<24} {caps:<45} [{mark}]")
    return 0


def _cmd_pack(args: argparse.Namespace) -> int:
    """Bundle a trace + incident + studio + patch into a shareable .loompack."""
    from .pack import build_pack

    _load_trace_json(args.path)  # friendly error for non-traces
    out, redacted = build_pack(args.path, out=args.output or None)
    note = f" ({redacted} secret(s) scrubbed)" if redacted else " (secrets scrubbed)"
    print(f"packed -> {out}{note}")
    print("  a self-contained incident bundle: replay, studio, incident, patch, manifest")
    return 0


def _cmd_share(args: argparse.Namespace) -> int:
    """Produce a shareable copy: scrub secrets, then REFUSE to emit if any remain."""
    from .scrub import load_scrub_config, scrub_text, scrub_trace

    config = load_scrub_config(args.config) if args.config else None
    data = _load_trace_json(args.path)
    clean, found = scrub_trace(data, aggressive=args.aggressive, config=config)
    total = sum(found.values())
    for kind in sorted(found):
        print(f"  redacted {found[kind]:>3}x {kind}")

    # Belt and suspenders: scan the SCRUBBED output too. If anything a human
    # would call a secret survives, don't hand out a file that looks safe.
    residual = 0
    _, still = scrub_text(json.dumps(clean), aggressive=True)
    residual = sum(still.values())
    if residual:
        print(f"loom: {residual} possible secret(s) survived scrubbing -- not sharing. "
              f"Try --aggressive, or inspect the trace.", file=sys.stderr)
        return 1

    clean["scrubbed"] = True  # Studio reads this to show a "safe to share" banner
    if "checksum" in clean:
        from .trace import trace_checksum

        clean["checksum"] = trace_checksum(clean)
    if args.path.endswith(".loom.json"):
        out = args.output or args.path[: -len(".loom.json")] + ".shared.loom.json"
    else:
        out = args.output or args.path + ".shared"
    with open(out, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"redacted {total} secret(s); safe to share -> {out}")
    return 0


def _cmd_why(args: argparse.Namespace) -> int:
    """Ask a debugger agent a question about a saved trace."""
    if args.step is not None:
        # Offline "why did it do THAT?": stated intent + the observations the
        # action most plausibly drew on. Instant, deterministic, no API calls.
        from .insight import describe_why, why_action

        _import_builtin_packs()
        try:
            print(describe_why(why_action(_load_trace_json(args.path), args.step)))
        except ValueError as e:
            raise CLIError(str(e))
        return 0
    if not args.question:
        raise CLIError("why needs a question, or --step N for the offline explanation")

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


def _cmd_bench(args: argparse.Namespace) -> int:
    """Run one task through several agents and compare their traces."""
    import os

    from .bench import load_task, report, reset_workspace, run_agent, target_for

    try:
        task = load_task(args.task)
    except (OSError, ValueError) as e:
        raise CLIError(str(e))

    agents = []
    for spec in args.agent:
        name, sep, command = spec.partition(":")
        if not sep or not command.strip():
            raise CLIError(f"--agent must look like name:command, got {spec!r}")
        agents.append((name.strip(), command.strip()))
    if not agents:
        raise CLIError("bench needs at least one --agent name:command")

    # Workspace reset needs a clean baseline: refuse to nuke uncommitted work.
    baseline = ""
    if args.reset == "git":
        from .workspace import _git

        cwd = os.getcwd()
        baseline = _git(["rev-parse", "HEAD"], cwd)
        if not baseline:
            raise CLIError("--reset git needs a git repo (none found here)")
        if _git(["status", "--porcelain"], cwd, strip=False).strip() and not args.force:
            raise CLIError("the workspace is dirty and --reset git would hard-reset it; "
                           "commit or stash first, or pass --force")

    os.makedirs(args.outdir, exist_ok=True)
    shield = _build_shield(args)
    results = []
    tmpdirs = []
    for i, (name, command) in enumerate(agents):
        workdir = None
        if args.reset == "git" and i > 0:  # clean slate before each agent but the first
            err = reset_workspace(args.reset, os.getcwd(), baseline)
            if err:
                raise CLIError(err)
            print(f"reset workspace to {baseline[:10]}", file=sys.stderr)
        elif args.reset == "copy":
            import shutil
            import tempfile

            workdir = tempfile.mkdtemp(prefix=f"loom-bench-{name}-")
            tmpdirs.append(workdir)
            shutil.copytree(os.getcwd(), workdir, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns(args.outdir, "bench-traces"))
            print(f"{name}: isolated workspace {workdir}", file=sys.stderr)
        agent_target = target_for(command, args.target)
        print(f"running {name} ({'openai' if 'openai' in agent_target else 'anthropic'})...",
              file=sys.stderr)
        results.append(run_agent(name, command, task, agent_target,
                                  shield=shield, outdir=args.outdir, studio=args.studio,
                                  workdir=workdir))
    if args.reset == "git" and len(agents) > 1:
        reset_workspace(args.reset, os.getcwd(), baseline)  # leave a clean tree
    for d in tmpdirs:
        import shutil

        shutil.rmtree(d, ignore_errors=True)
    text = report(args.task, results)
    print(text)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text + "\n")
    return 0 if any(r.get("passed") for r in results) else 1


def _cmd_policy(args: argparse.Namespace) -> int:
    """Scaffold, test, or explain a firewall policy."""
    from .policy_file import PROFILES, profile_names, resolve, to_shield_kwargs
    from .shield import Shield

    if args.policy_cmd == "init":
        if args.name not in PROFILES:
            raise CLIError(f"unknown profile {args.name!r}; built-in: {', '.join(profile_names())}")
        out = args.output or "loom-policy.yml"
        prof = PROFILES[args.name]
        lines = [f"# loom policy -- generated from the {args.name!r} profile", "",
                 f"# {prof.get('description', '')}", f"default: {prof.get('default', 'allow')}"]
        for section in ("allow", "confirm", "deny", "sequence"):
            if prof.get(section):
                lines.append(f"{section}:")
                # Quote items with a colon so YAML reads them as strings, not maps
                # (sequence rules like 'after X: deny Y' would otherwise parse wrong).
                lines += [f'  - "{p}"' if ": " in p else f"  - {p}" for p in prof[section]]
        with open(out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"wrote {out} (from profile {args.name!r}) -- edit, then: loom record ... --policy {out}")
        return 0

    if args.policy_cmd == "lint":
        from .policy_file import lint, load_document

        doc = load_document(args.policy) if args.policy else PROFILES.get(args.profile, {})
        if not doc:
            raise CLIError("lint needs --policy FILE or --profile NAME")
        problems = lint(doc)
        for w in problems:
            print(f"  ⚠️  {w}")
        if problems:
            print(f"\n{len(problems)} issue(s) -- these rules may not do what they look like",
                  file=sys.stderr)
            return 1
        print("policy looks good")
        return 0

    if args.policy_cmd == "simulate":
        # The rollout-impact report: what would this policy have done to the
        # runs you already recorded? Production teams don't hard-cut a deny
        # rule -- they look at the blast radius first.
        from .policy_file import (simulate, simulate_html, simulate_markdown,
                                  simulate_text)

        shield = Shield(**to_shield_kwargs(resolve(profile=args.profile, policy_path=args.policy)))
        paths = _expand_trace_paths(args.paths)
        if not paths:
            raise CLIError("no traces found (pass files or a directory of *.loom.json)")
        result = simulate(shield, paths)
        if not result["runs"]:
            raise CLIError("no readable traces found")

        if args.html:
            with open(args.html, "w") as f:
                f.write(simulate_html(result))
            print(f"policy simulation dashboard -> {args.html}")
        if args.md:
            out = simulate_markdown(result)
            if args.md == "-":
                print(out)
            else:
                with open(args.md, "w") as f:
                    f.write(out + "\n")
                print(f"policy simulation (markdown) -> {args.md}")
        if not args.html and not args.md:
            print(simulate_text(result))
        return 1 if args.fail_on_deny and result["denied"] else 0

    if args.policy_cmd == "explain":
        # Explain how the policy would classify each tool call in one trace.
        shield = Shield(**to_shield_kwargs(resolve(profile=args.profile, policy_path=args.policy)))
        data = _load_trace_json(args.path)
        seen: dict = {}
        for e in data.get("log", []):
            if e.get("kind") == "model" and isinstance(e.get("result"), dict):
                for tc in e["result"].get("tool_calls") or []:
                    action, rule = shield.classify(tc.get("name", ""), tc.get("input", {}))
                    sig = f"{tc.get('name')}({json.dumps(tc.get('input', {}), sort_keys=True, default=str)})"
                    seen[sig[:100]] = (action, rule)
        if not seen:
            print("no tool calls in this trace")
            return 0
        for sig, (action, rule) in seen.items():
            mark = {"deny": "🚫", "confirm": "⏸️ ", "allow": "✅"}.get(action, "  ")
            print(f"{mark} {action:8} {sig}" + (f"   (rule: {rule})" if rule else "   (default)"))
        return 0

    # test: run the policy against a JSON list of {name, input} calls
    shield = Shield(**to_shield_kwargs(resolve(profile=args.profile, policy_path=args.policy)))
    with open(args.calls) as f:
        cases = json.load(f)
    failures = 0
    for case in cases:
        action, rule = shield.classify(case["name"], case.get("input", {}))
        expected = case.get("expect")
        ok = expected is None or expected == action
        if not ok:
            failures += 1
        status = "ok  " if ok else "FAIL"
        line = f"{status} {action:8} {case['name']}({json.dumps(case.get('input', {}), default=str)})"
        if expected and expected != action:
            line += f"   expected {expected}"
        if case.get("why"):
            line += f"   — {case['why']}"
        print(line)
    if failures:
        print(f"\n{failures} case(s) did not match expectations", file=sys.stderr)
        return 1
    print(f"\nall {len(cases)} case(s) as expected")
    return 0


def _signing_key(args) -> "bytes | None":
    """Resolve --key-env / --key-file into signing-key bytes, or None."""
    import os

    if getattr(args, "key_env", ""):
        val = os.environ.get(args.key_env)
        if not val:
            raise CLIError(f"env var {args.key_env} is not set")
        return val.encode()
    if getattr(args, "key_file", ""):
        try:
            with open(args.key_file, "rb") as f:
                return f.read().strip()
        except OSError as e:
            raise CLIError(f"could not read key file: {e}")
    return None


def _cmd_trace(args: argparse.Namespace) -> int:
    """Validate / verify / sign / explain-version a trace's format contract."""
    from .trace import TRACE_VERSION, trace_checksum, trace_signature
    from .testing import verify_trace

    data = _load_trace_json(args.path)
    version = data.get("version", 1)

    if args.trace_cmd == "sign":
        key = _signing_key(args)
        if key is None:
            raise CLIError("sign needs a key: --key-env VAR or --key-file PATH")
        data["signature"] = trace_signature(data, key)
        if "checksum" in data:
            data["checksum"] = trace_checksum(data)
        with open(args.path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"{args.path}: signed ({data['signature'][:24]}…)")
        return 0

    if args.trace_cmd == "verify-approvals":
        key = _signing_key(args)
        if key is None:
            raise CLIError("verify-approvals needs the signing key: --key-env VAR or --key-file PATH")
        from .shield import verify_approvals

        valid, invalid = verify_approvals(data, key)
        if not valid and not invalid:
            print(f"{args.path}: no signed approvals to verify")
            return 0
        for ev in valid:
            by = ev.get("by", "?")
            print(f"  ✓ {ev.get('action'):8} {ev.get('tool')}  by {by}  (id {ev.get('id', '?')})")
        for ev in invalid:
            print(f"  ✗ {ev.get('action'):8} {ev.get('tool')}  by {ev.get('by', '?')}  "
                  "-- SIGNATURE INVALID (tampered or wrong key)")
        print(f"\n{len(valid)} valid, {len(invalid)} invalid signed decision(s)")
        return 1 if invalid else 0

    if args.trace_cmd == "explain-version":
        print(f"{args.path}: trace format version {version} (this loom writes v{TRACE_VERSION})")
        if version < TRACE_VERSION:
            print("  older format: effect keys were computed differently, so strict replay "
                  "and `loom impact` may report inputs-differ. Bring it forward: loom migrate")
        elif version > TRACE_VERSION:
            print("  newer format: written by a newer loom; upgrade loom-harness if anything "
                  "looks off.")
        else:
            print("  current: strict replay and impact are apples-to-apples.")
        return 0

    if args.trace_cmd == "verify":
        key = _signing_key(args)
        if key is not None:  # cryptographic verification against a shared secret
            sig = data.get("signature")
            if not sig:
                print(f"loom: {args.path} is not signed", file=sys.stderr)
                return 1
            import hmac

            if hmac.compare_digest(sig, trace_signature(data, key)):
                print(f"{args.path}: signature valid — authentic and unmodified")
                return 0
            print(f"loom: {args.path} signature INVALID (wrong key or tampered)",
                  file=sys.stderr)
            return 1
        stored = data.get("checksum")
        if not stored:
            print(f"{args.path}: no checksum (written by an older loom or hand-made)")
            return 0
        if stored == trace_checksum(data):
            print(f"{args.path}: checksum OK — unmodified since it was written")
            return 0
        print(f"loom: {args.path} was MODIFIED after it was written (checksum mismatch)",
              file=sys.stderr)
        return 1

    # validate: structure + checksum together, the CI gate
    problems = verify_trace(args.path)
    if version > TRACE_VERSION:
        problems.append(f"trace version {version} is newer than this loom's v{TRACE_VERSION}")
    if problems:
        for p in problems:
            print(f"  ✗ {p}")
        print(f"\n{len(problems)} problem(s)", file=sys.stderr)
        return 1
    print(f"{args.path}: valid (v{version}, structure + checksum OK)")
    return 0


def _cmd_incident(args: argparse.Namespace) -> int:
    """Write an agent postmortem from a saved trace."""
    from .incident import build_report

    data = _load_trace_json(args.path)
    why_output = ""
    if args.why:
        from .why import why

        try:
            run = why(args.path, args.question, model=args.model)
            why_output = run.output
        except Exception as e:  # the offline report still stands
            why_output = f"_(why agent unavailable: {type(e).__name__}: {e})_"
    report = build_report(data, args.path, why_output=why_output)
    if args.output:
        with open(args.output, "w") as f:
            f.write(report + "\n")
        print(f"wrote {args.output}")
    else:
        print(report)
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

    # `path:A->B` terms are temporal (ordered within a run) -- SQL can't see
    # order, so they post-filter the SQL candidates by walking each trace's
    # Action timeline. Terms match a capability, risk category, or tool name:
    #   loom search runs/ "path:pii_access->user_communication"
    terms = args.query.split()
    paths_terms = [t[5:] for t in terms if t.startswith("path:") and "->" in t]
    sql_query = " ".join(t for t in terms if not t.startswith("path:"))

    lake = Lake(args.directory)
    lake.index()
    try:
        rows = lake.search(sql_query)
    except ValueError as e:
        print(f"loom: {e}", file=sys.stderr)
        return 2
    finally:
        lake.close()

    evidence: dict[str, list[str]] = {}
    if paths_terms:
        from .action import sequence_hits
        from .packs import install_builtin

        install_builtin()
        kept = []
        for r in rows:
            try:
                with open(r["path"]) as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            ok = True
            for term in paths_terms:
                first, _, then = term.partition("->")
                hits = sequence_hits(data, first.strip(), then.strip())
                if not hits:
                    ok = False
                    break
                a, b = hits[0]
                evidence[r["path"]] = [f"[{a.step}] {a.tool} → [{b.step}] {b.tool}"]
            if ok:
                kept.append(r)
        rows = kept

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
        if r["risk"]:
            flags.append(f"⚠️ {r['risk'].replace(' ', ',')}")
        prompt = (r["episodes"] or "").split(" | ")[0][:60]
        print(f"{tokens:>9,} tok  {r['path']}  {prompt!r}"
              + (f"  [{', '.join(flags)}]" if flags else ""))
        for line in evidence.get(r["path"], []):
            print(f"           path: {line}")
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
    df.add_argument("--actions", action="store_true",
                    help="behavior diff at the Action level: added/removed actions "
                         "and the exercised-risk movement (exit 1 on any change)")
    df.add_argument("--json", action="store_true", help="machine-readable (--actions only)")
    df.set_defaults(func=_cmd_diff)

    ex = sub.add_parser("export", help="render a trace to HTML, or --jsonl/--otel events")
    ex.add_argument("path", help="a trace file or a directory of *.loom.json")
    ex.add_argument("-o", "--output", default="", help="HTML output path (default: <trace>.html)")
    ex.add_argument("--jsonl", default="", metavar="FILE",
                    help="flatten to one JSON event per effect (FILE, or '-' for stdout) "
                         "for Datadog/Splunk/Grafana ingestion")
    ex.add_argument("--otel", action="store_true",
                    help="with --jsonl (or alone -> stdout): OpenTelemetry-style log records")
    ex.set_defaults(func=_cmd_export)

    dr = sub.add_parser("doctor",
                        help="check your environment (no args) or a trace for context rot")
    dr.add_argument("path", nargs="?", default="",
                    help="a trace to check for context rot; omit to check the environment")
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
        sp.add_argument("--profile", default="", metavar="NAME",
                        help="apply a built-in safety profile (claude-code-safe, ci-safe, "
                             "prod-data-safe); flags below extend it")
        sp.add_argument("--policy", default="", metavar="FILE",
                        help="apply a policy file (loom-policy.yml/.json)")
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
        sp.add_argument("--sign-approvals-key-env", dest="sign_approvals_key_env", default="",
                        metavar="VAR",
                        help="HMAC-sign every operator decision with the key in this env var "
                             "(verify later with: loom trace verify-approvals --key-env VAR)")
        sp.add_argument("--require-approver", dest="require_approver", action="append",
                        default=[], metavar="PATTERN=NAMES",
                        help="only these identities may APPROVE a capability, e.g. "
                             "'cap:money_movement=alice,bob' (repeatable; anyone may still deny)")

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

    rc = sub.add_parser(
        "record",
        help="black-box a real agent session",
        description="Record an agent through a proxy. Two forms:\n"
                    "  loom record [--profile P] [--report] claude \"fix the test\"\n"
                    "  loom record [--profile P] -- <any command>\n"
                    "(put loom's flags BEFORE the agent). Known shortcuts: "
                    + ", ".join(sorted(AGENTS)),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    rc.add_argument("command", nargs=argparse.REMAINDER,
                    help="`claude \"prompt\"` (a known agent) or `-- <command>`")
    rc.add_argument("--save", default="session.loom.json")
    rc.add_argument("--target", default="https://api.anthropic.com",
                    help="upstream API (use https://api.openai.com for OpenAI agents)")
    rc.add_argument("--port", type=int, default=0, help="proxy port (default: pick a free one)")
    rc.add_argument("--safe", action="store_true",
                    help="shorthand for --profile claude-code-safe --scrub --report "
                         "(the sane defaults for a coding agent)")
    rc.add_argument("--report", action="store_true",
                    help="after recording, also write <save>.html (Studio) and "
                         "<save>.incident.md (postmortem)")
    rc.add_argument("--no-workspace", action="store_true",
                    help="don't record cwd/git/argv/os metadata or the file-change delta")
    rc.add_argument("--capture-diff", action="store_true",
                    help="embed the full git patch of the agent's changes in the trace "
                         "(may be large / contain secrets -- scrub before sharing)")
    rc.add_argument("--sandbox", action="store_true",
                    help="deny the agent ALL network except the proxy (macOS sandbox-exec); "
                         "shield rules become impossible to bypass")
    rc.add_argument("--container", action="store_true",
                    help="run the agent in Docker: filesystem AND network isolation, "
                         "API routed through the proxy (repo mounted at /workspace)")
    rc.add_argument("--container-image", dest="container_image", default="python:3.12-slim",
                    help="the Docker image to run the agent in (default python:3.12-slim)")
    rc.add_argument("--container-readonly", dest="container_readonly", action="store_true",
                    help="mount the repo read-only inside the container")
    rc.add_argument("--container-memory", dest="container_memory", default="",
                    help="memory limit for the container (e.g. 2g)")
    rc.add_argument("--container-cpus", dest="container_cpus", default="",
                    help="CPU limit for the container (e.g. 2)")
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
    px.add_argument("--host", default="127.0.0.1",
                    help="bind address (default loopback; 0.0.0.0 for the docker-sandbox "
                         "topology -- pair a wide bind with --auth on open networks)")
    px.add_argument("--target", default="https://api.anthropic.com")
    px.add_argument("--save", default="session.loom.json", help="trace written after every exchange")
    px.add_argument("--replay", default="", help="serve recorded responses from this trace instead")
    px.add_argument("--live", action="store_true",
                    help="open Live Studio in a browser: watch the run in real time, "
                         "approve/deny held tool calls")
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
    ap.add_argument("--as", dest="by", default="",
                    help="record who decided (default: $LOOM_APPROVER or the OS user)")
    ap.set_defaults(func=_cmd_approve)

    bn = sub.add_parser("bench", help="run one task through several agents and compare")
    bn.add_argument("task", help="task file (yaml/json): prompt + success check")
    bn.add_argument("--agent", action="append", default=[], metavar="NAME:COMMAND",
                    help="an agent to benchmark, e.g. 'claude:claude -p {prompt}' (repeatable)")
    bn.add_argument("--target", default="https://api.anthropic.com",
                    help="upstream API (https://api.openai.com for OpenAI agents)")
    bn.add_argument("--outdir", default="bench-traces", help="where per-agent traces go")
    bn.add_argument("-o", "--output", default="", help="also write the report here")
    bn.add_argument("--reset", default="none", choices=["none", "git", "copy"],
                    help="isolate agents so one's edits don't pollute the next: "
                         "git = hard-reset to HEAD + clean between agents; "
                         "copy = each agent runs in its own copy of the repo")
    bn.add_argument("--force", action="store_true",
                    help="allow --reset git on a dirty tree (destroys uncommitted work)")
    bn.add_argument("--studio", action="store_true",
                    help="export each agent's trace to Studio HTML (a clickable trace per cell)")
    shield_flags(bn)
    bn.set_defaults(func=_cmd_bench)

    po = sub.add_parser("policy", help="scaffold, test, or explain a firewall policy")
    posub = po.add_subparsers(dest="policy_cmd", required=True)
    po_init = posub.add_parser("init", help="write a policy file from a built-in profile")
    po_init.add_argument("name", help="profile: claude-code-safe, ci-safe, prod-data-safe")
    po_init.add_argument("-o", "--output", default="", help="output path (default loom-policy.yml)")
    po_init.set_defaults(func=_cmd_policy)
    po_test = posub.add_parser("test", help="classify a JSON list of tool calls against a policy")
    po_test.add_argument("calls", help="JSON file: [{\"name\":..., \"input\":..., \"expect\":\"deny\"}]")
    po_test.add_argument("--profile", default="")
    po_test.add_argument("--policy", default="")
    po_test.set_defaults(func=_cmd_policy)
    po_exp = posub.add_parser("explain", help="show how a policy classifies a trace's tool calls")
    po_exp.add_argument("path", help="a saved trace")
    po_exp.add_argument("--profile", default="")
    po_exp.add_argument("--policy", default="")
    po_exp.set_defaults(func=_cmd_policy)
    po_lint = posub.add_parser("lint", help="catch rules that don't do what they look like")
    po_lint.add_argument("--policy", default="", help="policy file to lint")
    po_lint.add_argument("--profile", default="", help="or a built-in profile to lint")
    po_lint.set_defaults(func=_cmd_policy)
    po_sim = posub.add_parser("simulate",
                              help="rollout impact: what would this policy have done to "
                                   "your recorded runs?")
    po_sim.add_argument("paths", nargs="+", help="trace files and/or directories of *.loom.json")
    po_sim.add_argument("--profile", default="")
    po_sim.add_argument("--policy", default="")
    po_sim.add_argument("--fail-on-deny", action="store_true", dest="fail_on_deny",
                        help="exit 1 if any run would be denied (CI gate)")
    po_sim.add_argument("--html", default="", metavar="FILE",
                        help="write a self-contained dashboard for security review")
    po_sim.add_argument("--md", default="", metavar="FILE",
                        help="write a Markdown summary for a PR comment ('-' = stdout)")
    po_sim.set_defaults(func=_cmd_policy)

    tr_p = sub.add_parser("trace", help="validate / verify / sign / explain a trace's format")
    trsub = tr_p.add_subparsers(dest="trace_cmd", required=True)
    for cmd, helptext in [
        ("validate", "structure + checksum check (CI gate; exit 1 on problems)"),
        ("verify", "tamper check: checksum, or HMAC signature with --key-* (exit 1 on fail)"),
        ("sign", "add an HMAC signature with --key-env/--key-file (tamper-proof)"),
        ("verify-approvals", "verify the HMAC on each signed shield decision (exit 1 on any invalid)"),
        ("explain-version", "report the trace format version and what to expect"),
    ]:
        sp = trsub.add_parser(cmd, help=helptext)
        sp.add_argument("path")
        if cmd in ("sign", "verify", "verify-approvals"):
            sp.add_argument("--key-env", dest="key_env", default="",
                            help="read the signing key from this environment variable")
            sp.add_argument("--key-file", dest="key_file", default="",
                            help="read the signing key from this file")
        sp.set_defaults(func=_cmd_trace)

    ic = sub.add_parser("incident", help="write an agent postmortem from a saved trace")
    ic.add_argument("path")
    ic.add_argument("-o", "--output", default="", help="write markdown here instead of stdout")
    ic.add_argument("--why", action="store_true",
                    help="add an AI root-cause narrative (runs the loom why debugger; costs API calls)")
    ic.add_argument("--question", default="What was the root cause of the failure? Cite seqs.",
                    help="the question the why agent investigates (with --why)")
    ic.add_argument("--model", default="claude-opus-4-8", help="model for --why")
    ic.set_defaults(func=_cmd_incident)

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

    ud = sub.add_parser("undo", help="revert the file changes an agent made (from its trace)")
    ud.add_argument("path")
    ud.add_argument("--only", default="", help="only revert paths under this prefix")
    ud.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="show what would be reverted, change nothing")
    ud.add_argument("--force", action="store_true",
                    help="undo even if the tree changed since the recording")
    ud.add_argument("--plan", action="store_true",
                    help="show per-action undo/compensation plans for every domain "
                         "(files, SQL, browser, CRM), newest first; changes nothing")
    ud.set_defaults(func=_cmd_undo)

    fk = sub.add_parser("fork", help="fork a recorded run at a step: replay the prefix, continue live")
    fk.add_argument("path")
    group = fk.add_mutually_exclusive_group(required=True)
    group.add_argument("--turn", type=int, default=None, help="fork at this top-level turn")
    group.add_argument("--from-step", type=int, dest="from_step", default=None,
                       help="fork at the turn containing this step")
    fk.add_argument("--agent", default="", help="module:attr to continue the run live")
    fk.add_argument("--inject", default="",
                    help="append this user note to the context at the fork point")
    fk.add_argument("-o", "--output", default="", help="where to save the branch trace")
    fk.set_defaults(func=_cmd_fork)

    pks = sub.add_parser("packs", help="list / lint / test domain packs")
    pks.set_defaults(func=_cmd_packs, packs_cmd=None)
    pkssub = pks.add_subparsers(dest="packs_cmd")
    pks_lint = pkssub.add_parser("lint", help="static correctness checks for a pack")
    pks_lint.add_argument("--pack", required=True, help="module:attr (a Pack class or instance)")
    pks_lint.set_defaults(func=_cmd_packs)
    pks_test = pkssub.add_parser("test", help="run golden cases against a pack")
    pks_test.add_argument("cases", help="a .yml/.json file of {pack, cases:[{action, expect}]}")
    pks_test.add_argument("--pack", default="", help="module:attr (overrides the file's pack:)")
    pks_test.set_defaults(func=_cmd_packs)

    sv = sub.add_parser("serve", help="serve a trace directory to the team (list, search, "
                                      "Studio, incidents)")
    sv.add_argument("directory", nargs="?", default=".", help="directory of *.loom.json")
    sv.add_argument("--port", type=int, default=8790)
    sv.add_argument("--host", default="127.0.0.1",
                    help="bind address (0.0.0.0 shares on the LAN; no auth)")
    sv.set_defaults(func=_cmd_serve)

    tls = sub.add_parser("tools", help="show an agent's tools and their capability contract")
    tls.add_argument("--agent", required=True, help="module:attr (Agent or zero-arg factory)")
    tls.add_argument("--json", action="store_true", help="machine-readable manifest")
    tls.set_defaults(func=_cmd_tools)

    pk = sub.add_parser("pack", help="bundle trace + incident + studio + patch into a .loompack")
    pk.add_argument("path")
    pk.add_argument("-o", "--output", default="", help="output path (default *.loompack)")
    pk.set_defaults(func=_cmd_pack)

    sh = sub.add_parser("share", help="make a shareable copy: scrub, then refuse if secrets remain")
    sh.add_argument("path")
    sh.add_argument("-o", "--output", default="", help="output path (default *.shared.loom.json)")
    sh.add_argument("--aggressive", action="store_true", help="also redact high-entropy tokens")
    sh.add_argument("--config", default="", metavar="FILE",
                    help="loom-scrub.yml: custom detectors + an allowlist")
    sh.set_defaults(func=_cmd_share)

    sc = sub.add_parser("scrub", help="redact secrets from a trace before sharing it")
    sc.add_argument("path")
    sc.add_argument("--in-place", action="store_true", dest="in_place",
                    help="overwrite the trace instead of writing *.scrubbed.loom.json")
    sc.add_argument("--check", action="store_true",
                    help="report only; exit 1 if secrets are found (CI gate)")
    sc.add_argument("--aggressive", action="store_true",
                    help="also redact long high-entropy tokens (may false-positive)")
    sc.add_argument("--config", default="", metavar="FILE",
                    help="loom-scrub.yml: custom detectors + an allowlist")
    sc.add_argument("--audit", default="", metavar="FILE",
                    help="write a redaction audit report (what/where, no values; '-' for stdout)")
    sc.set_defaults(func=_cmd_scrub)

    wy = sub.add_parser("why", help="ask a debugger agent about a saved trace "
                                    "(or --step N for the offline action explanation)")
    wy.add_argument("path")
    wy.add_argument("question", nargs="?", default="")
    wy.add_argument("--step", type=int, default=None,
                    help="explain the action at this step offline: intent, risk, "
                         "policy, and the observations it drew on (no API calls)")
    wy.add_argument("--model", default="claude-opus-4-8")
    wy.add_argument("--save", default="", help="record the diagnosis run to this path")
    wy.set_defaults(func=_cmd_why)

    sc = sub.add_parser("score", help="behavior scorecard: security/side-effect/reversibility/...")
    sc.add_argument("path")
    sc.add_argument("--json", action="store_true", help="machine-readable breakdown")
    sc.set_defaults(func=_cmd_score)

    tt = sub.add_parser("taint", help="value-lineage exfiltration paths (secret/PII → egress)")
    tt.add_argument("path")
    tt.add_argument("--fail-on-leak", action="store_true", dest="fail_on_leak",
                    help="exit 1 if any exfiltration path is found (CI gate)")
    tt.set_defaults(func=_cmd_taint)

    mp = sub.add_parser("map", help="side-effect map: everything the run changed or reached")
    mp.add_argument("path")
    mp.set_defaults(func=_cmd_map)

    gr = sub.add_parser("graph", help="delegation/causality tree across subagent depths")
    gr.add_argument("path")
    gr.set_defaults(func=_cmd_graph)

    pv = sub.add_parser("provenance",
                        help="link each claim in the final answer to its tool-result evidence")
    pv.add_argument("path")
    pv.set_defaults(func=_cmd_provenance)

    fl = sub.add_parser("flake", help="divergence heatmap across repeated runs of one task")
    fl.add_argument("paths", nargs="+", help="trace files and/or directories (first = baseline)")
    fl.set_defaults(func=_cmd_flake)

    nt = sub.add_parser("note", help="annotate a trace step (shared sidecar notes)")
    nt.add_argument("path")
    nt.add_argument("--step", type=int, default=None, help="the step the note is about")
    nt.add_argument("-m", "--message", default="", help="the note text (omit to list notes)")
    nt.add_argument("--by", default="", help="author (default: $USER)")
    nt.set_defaults(func=_cmd_note)

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
    import sys as _sys

    argv = list(argv) if argv is not None else _sys.argv[1:]
    # `loom claude "fix the tests"` is the tightest entrypoint: sugar for
    # `loom record --safe claude "..."` (firewall + scrub + report).
    if argv and argv[0] in AGENTS:
        argv = ["record", "--safe", argv[0]] + argv[1:]

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
