"""Effect-level caching: identical inputs reuse recorded results across runs.

Every effect already carries a stable hash of its inputs (the trace ``key``),
so a cache falls out of the boundary for free:

    cache = EffectCache("dev-cache.jsonl")     # persistent; or EffectCache() in-memory
    agent = Agent(model=..., cache=cache)

    agent.run("same prompt")   # pays for the model call
    agent.run("same prompt")   # served from cache -- zero API calls

By default only ``model`` effects are cached: tool calls have side effects and
should re-execute unless you opt them in with ``kinds=("model", "tool:*")``.
Cached results land in the trace like any recorded effect, so replay, fork,
and diff behave identically.
"""

from __future__ import annotations

import json
import os
from fnmatch import fnmatch
from typing import Any


class EffectCache:
    """A key -> encoded-result store, optionally persisted as JSONL."""

    def __init__(self, path: "str | None" = None, kinds: tuple = ("model",)):
        self.path = path
        self.kinds = tuple(kinds)
        self.hits = 0
        self.misses = 0
        self._store: dict[str, Any] = {}
        if path and os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # tolerate a torn tail
                    self._store[d["key"]] = d["result"]

    def wants(self, kind: str) -> bool:
        return any(fnmatch(kind, pattern) for pattern in self.kinds)

    def get(self, key: str) -> Any:
        result = self._store.get(key)
        if result is not None:
            self.hits += 1
        return result

    def put(self, key: str, result: Any) -> None:
        self.misses += 1
        self._store[key] = result
        if self.path:
            with open(self.path, "a") as f:
                f.write(json.dumps({"key": key, "result": result}) + "\n")
