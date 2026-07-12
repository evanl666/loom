"""Live agent sessions: run a complex agent for the FIRST time with the debugger
open, watch each step stream in, and keep asking follow-up questions -- results
arriving step-by-step -- instead of recording a trace first and replaying it.

One primitive makes both cases work: a **live-growing trace**. A background
thread runs the agent and appends effects to a log; the debugger polls a
snapshot and renders new steps as they land.

Two agent sources, same streaming abstraction:

  * a native ``loom.Agent`` -- run with an injected Recorder whose ``log`` grows
    turn by turn; a follow-up replays the prior turns for free and runs only the
    new one live.
  * any external agent, given as a ``callable(prompt) -> str`` (a LangGraph /
    Claude-SDK / CrewAI adapter) -- an in-process recording proxy captures every
    model call it makes (it just has to honor ``ANTHROPIC_BASE_URL`` /
    ``OPENAI_BASE_URL``), and that proxy's log is what grows.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Callable


def start_proxy(target: str = "https://api.anthropic.com"):
    """Start an in-process recording proxy and point the model-client env vars
    at it. Call this BEFORE importing a framework whose clients read
    ANTHROPIC_BASE_URL / OPENAI_BASE_URL at construction time."""
    from .proxy import ProxyServer

    proxy = ProxyServer(port=0, target=target, save_path=None)
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{proxy.server_address[1]}"
    os.environ["ANTHROPIC_BASE_URL"] = base
    os.environ["ANTHROPIC_API_BASE"] = base      # litellm / CrewAI dialect
    os.environ["OPENAI_BASE_URL"] = base + "/v1"
    return proxy


class LiveSession:
    def __init__(self, agent: Any = None, func: "Callable[[str], Any] | None" = None,
                 target: str = "https://api.anthropic.com", proxy: Any = None,
                 spec: str = ""):
        if (agent is None) == (func is None):
            raise ValueError("LiveSession needs exactly one of agent= or func=")
        self.agent = agent
        self.func = func
        self.spec = spec          # "module:attr" -- lets an external agent be re-run to fork
        self.target = target
        self.running = False
        self.error = ""
        self.output = ""
        self.turns = 0
        self._episodes: list[str] = []
        # (wire_index_at_ask_start, prompt) per ask() on the proxy path, so a
        # multi-turn live session shows each user message as its own dialogue
        # turn (a proxy trace's raw "episodes" also holds internal tool-result
        # turns, so it can't tell the real asks apart -- this can).
        self._user_turns: list[list] = []
        self._log: list = []            # accumulated EffectEntry (native path)
        self._live_rec = None           # the Recorder/WireRecorder growing right now
        self._proxy = proxy
        self._lock = threading.Lock()
        if func is not None:
            # A caller that must set ANTHROPIC_BASE_URL before importing the
            # framework (clients bake base_url at construction) starts the proxy
            # first and passes it in; otherwise start one now.
            if self._proxy is None:
                self._proxy = start_proxy(target)
            self._live_rec = self._proxy.recorder

    # -- introspection ------------------------------------------------------
    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._proxy.server_address[1]}" if self._proxy else ""

    @property
    def can_ask(self) -> bool:
        return not self.running

    @property
    def can_proxy_fork(self) -> bool:
        """An external adapter with a re-runnable spec can be proxy-forked:
        replay its recorded prefix, then re-run the tail live with edits."""
        return (self.func is not None and bool(self.spec)
                and self._proxy is not None and bool(self._proxy.recorder.wire))

    def fork_external(self, at: int, edits: dict) -> dict:
        """Fork an EXTERNAL agent at model-turn ``at``: a fresh proxy replays the
        recorded prefix (the exchanges before ``at``) BY CONTENT -- so a re-run
        that reorders its calls still matches -- then rewrites the request (edited
        system / injected message / model) and forwards LIVE for the fork-point
        call and everything downstream. The adapter is re-run in a SUBPROCESS so
        its model client picks up the fork proxy's base_url (frameworks bake it at
        import). Returns the branch trace."""
        import subprocess
        import sys

        if not self.can_proxy_fork:
            raise RuntimeError(
                "proxy-fork needs an external adapter started with "
                "`loom live --agent module:attr` and at least one recorded turn")
        from .proxy import ProxyServer

        rec = self._proxy.recorder
        keys, wire = list(rec.request_keys), [dict(r) for r in rec.wire]
        at = max(0, min(int(at), len(keys)))
        prefix: dict = {}                      # content key -> [recorded responses]
        for i in range(at):                    # everything BEFORE the fork point replays
            prefix.setdefault(keys[i], []).append(wire[i])
        prompt = self._episodes[0] if self._episodes else ""
        fproxy = ProxyServer(port=0, target=self.target, save_path=None)
        fproxy.fork = {
            "prefix": prefix,
            "inject_key": keys[at] if (edits.get("append") and at < len(keys)) else None,
            "edit_sys_hash": edits.get("edit_sys_hash"),
            "new_system": edits.get("new_system"),
            "append": edits.get("append"),
            "model": edits.get("model", "keep"),
            "result_overrides": edits.get("result_overrides") or {},
            "_injected": False,
        }
        threading.Thread(target=fproxy.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{fproxy.server_address[1]}"
        module_name, _, attr = self.spec.partition(":")
        env = {**os.environ, "ANTHROPIC_BASE_URL": base, "ANTHROPIC_API_BASE": base,
               "OPENAI_BASE_URL": base + "/v1", "LOOM_FORK_PROMPT": prompt,
               "PYTHONPATH": os.getcwd() + os.pathsep + os.environ.get("PYTHONPATH", "")}
        runner = ("import importlib,os,sys;sys.path.insert(0, os.getcwd());"
                  f"getattr(importlib.import_module({module_name!r}), {attr!r})"
                  "(os.environ['LOOM_FORK_PROMPT'])")
        try:
            proc = subprocess.run([sys.executable, "-c", runner], env=env,
                                  capture_output=True, text=True, timeout=600)
            if not fproxy.recorder.wire and proc.returncode != 0:
                raise RuntimeError(f"fork re-run failed: {proc.stderr[-500:] or proc.stdout[-500:]}")
        finally:
            fproxy.shutdown()
        return fproxy.recorder.to_dict()

    # -- driving the agent --------------------------------------------------
    def ask(self, prompt: str) -> bool:
        """Start a turn (non-blocking). Returns False if one is already running."""
        with self._lock:
            if self.running:
                return False
            self.running = True
            self.error = ""
            # mark where this ask's model calls begin (wire index for the proxy
            # path, cumulative model-call count in the log for the native path) so
            # every follow-up turn shows in the step list AND the context frame.
            if self._proxy is not None:
                idx = len(self._proxy.recorder.wire)
            else:
                # a trace's "turn" counts only DEPTH-0 model calls (sub-agent calls
                # keep the parent's turn), so the boundary must match -- counting
                # every model entry over-shoots on a multi-agent run and never lands.
                idx = sum(1 for e in self._log
                          if getattr(e, "kind", "") == "model" and getattr(e, "depth", 0) == 0)
            self._user_turns.append([idx, str(prompt)])
        threading.Thread(target=self._run, args=(str(prompt),), daemon=True).start()
        return True

    def _run(self, prompt: str) -> None:
        try:
            if self.agent is not None:
                self._run_agent(prompt)
            else:
                self._run_func(prompt)
            self.turns += 1
        except Exception as e:  # noqa: BLE001 -- surface, don't kill the server
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self.running = False

    def _run_agent(self, prompt: str) -> None:
        from .effect import Recorder

        self._episodes.append(prompt)
        # Replay prior turns from the log for free; run only the new turn live.
        rec = Recorder(log=list(self._log), replay_until=len(self._log), allow_live=True)
        self._live_rec = rec
        run = self.agent.run(self._episodes, recorder=rec)
        self._log = list(rec.log)
        self.output = run.output

    def _run_func(self, prompt: str) -> None:
        # Point the external agent's model client at our in-process proxy, then
        # let it run -- every model call is captured and streamed.
        self._episodes.append(prompt)   # remember the prompt so a fork can re-run it
        os.environ["ANTHROPIC_BASE_URL"] = self.base_url
        os.environ["OPENAI_BASE_URL"] = self.base_url + "/v1"
        out = self.func(prompt)
        self.output = (str(out) if out else "") or (self._proxy.recorder.output or "")

    # -- the live trace + snapshot -----------------------------------------
    def trace(self) -> dict:
        if self._proxy is not None:
            d = self._proxy.recorder.to_dict()
            if len(self._user_turns) > 1:   # multi-turn: let steps_for split by ask
                d["user_turns"] = [list(t) for t in self._user_turns]
            return d
        log = self._live_rec.log if self._live_rec is not None else self._log
        tools = {}
        for name, t in getattr(self.agent, "tools", {}).items():
            caps = sorted(getattr(t, "capabilities", []) or [])
            if caps:
                tools[name] = caps
        d = {
            "log": [e.to_dict() for e in log],
            "prompt": self._episodes[0] if self._episodes else "",
            "episodes": list(self._episodes) or [""],
            "output": self.output,
            "model": getattr(self.agent, "model", "") if isinstance(
                getattr(self.agent, "model", ""), str) else "",
            "system": getattr(self.agent, "system", ""),
            "tools": tools,
        }
        if len(self._user_turns) > 1:   # multi-turn native run: split by ask
            d["user_turns"] = [list(t) for t in self._user_turns]
        return d

    def snapshot(self) -> dict:
        from .debugger import steps_for

        return {
            "steps": steps_for(self.trace()),
            "running": self.running,
            "output": self.output,
            "error": self.error,
            "turns": self.turns,
        }
