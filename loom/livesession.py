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


class LiveSession:
    def __init__(self, agent: Any = None, func: "Callable[[str], Any] | None" = None,
                 target: str = "https://api.anthropic.com"):
        if (agent is None) == (func is None):
            raise ValueError("LiveSession needs exactly one of agent= or func=")
        self.agent = agent
        self.func = func
        self.running = False
        self.error = ""
        self.output = ""
        self.turns = 0
        self._episodes: list[str] = []
        self._log: list = []            # accumulated EffectEntry (native path)
        self._live_rec = None           # the Recorder/WireRecorder growing right now
        self._proxy = None
        self._lock = threading.Lock()
        if func is not None:
            from .proxy import ProxyServer

            self._proxy = ProxyServer(port=0, target=target, save_path=None)
            threading.Thread(target=self._proxy.serve_forever, daemon=True).start()
            self._live_rec = self._proxy.recorder

    # -- introspection ------------------------------------------------------
    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._proxy.server_address[1]}" if self._proxy else ""

    @property
    def can_ask(self) -> bool:
        return not self.running

    # -- driving the agent --------------------------------------------------
    def ask(self, prompt: str) -> bool:
        """Start a turn (non-blocking). Returns False if one is already running."""
        with self._lock:
            if self.running:
                return False
            self.running = True
            self.error = ""
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
        os.environ["ANTHROPIC_BASE_URL"] = self.base_url
        os.environ["OPENAI_BASE_URL"] = self.base_url + "/v1"
        out = self.func(prompt)
        self.output = (str(out) if out else "") or (self._proxy.recorder.output or "")

    # -- the live trace + snapshot -----------------------------------------
    def trace(self) -> dict:
        if self._proxy is not None:
            return self._proxy.recorder.to_dict()
        log = self._live_rec.log if self._live_rec is not None else self._log
        tools = {}
        for name, t in getattr(self.agent, "tools", {}).items():
            caps = sorted(getattr(t, "capabilities", []) or [])
            if caps:
                tools[name] = caps
        return {
            "log": [e.to_dict() for e in log],
            "prompt": self._episodes[0] if self._episodes else "",
            "episodes": list(self._episodes) or [""],
            "output": self.output,
            "model": getattr(self.agent, "model", "") if isinstance(
                getattr(self.agent, "model", ""), str) else "",
            "system": getattr(self.agent, "system", ""),
            "tools": tools,
        }

    def snapshot(self) -> dict:
        from .debugger import steps_for

        return {
            "steps": steps_for(self.trace()),
            "running": self.running,
            "output": self.output,
            "error": self.error,
            "turns": self.turns,
        }
