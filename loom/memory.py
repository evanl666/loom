"""Trace memory: agents that learn from their own history.

Every Loom run leaves a complete trace. ``TraceMemory`` turns a directory of
them into recallable experience: before a run starts, the most similar past
runs are summarized into a context item, so the agent walks in knowing what
worked (and what failed) last time.

    memory = TraceMemory("runs/", auto_store=True)
    agent = Agent(model=..., tools=[...], memory=memory)
    agent.run("Migrate the staging database.")   # recalls similar past runs
    # auto_store=True saves each completed live run back into runs/

Recall is nondeterminism (the store changes over time), so it is recorded as a
``"memory"`` effect -- replays reproduce exactly what was recalled at the time.

Recall quality: the default is word-overlap (jaccard) -- zero dependencies,
fine for small stores. Pass an ``embedder`` (any callable mapping a list of
strings to a list of vectors) for semantic recall; ``OpenAIEmbedder`` wraps
the ``[openai]`` extra, and vectors are cached next to the traces so each one
is embedded once:

    memory = TraceMemory("runs/", embedder=OpenAIEmbedder())
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from glob import glob

_WORD = re.compile(r"[A-Za-z0-9]{4,}")


def _words(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text)}


def _cosine(a: "list[float]", b: "list[float]") -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class OpenAIEmbedder:
    """Embeddings via the ``[openai]`` extra (works with any base_url clone)."""

    def __init__(self, model: str = "text-embedding-3-small", **client_kwargs):
        from openai import OpenAI  # pip install "loom-harness[openai]"

        self.model = model
        self._client = OpenAI(**client_kwargs)

    def __call__(self, texts: "list[str]") -> "list[list[float]]":
        response = self._client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]


class TraceMemory:
    """Similarity recall over a directory of saved traces."""

    def __init__(self, directory: str, k: int = 3, auto_store: bool = False, embedder=None):
        self.directory = directory
        self.k = k
        self.auto_store = auto_store
        self.embedder = embedder  # callable: list[str] -> list[list[float]]
        os.makedirs(directory, exist_ok=True)
        self._index: list[dict] = []
        self._vectors: dict = {}  # text-hash -> vector, persisted per store
        self.refresh()

    def refresh(self) -> None:
        """Re-scan the directory. Called automatically after add()."""
        self._index = []
        for path in sorted(glob(os.path.join(self.directory, "*.loom.json"))):
            try:
                with open(path) as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            episodes = data.get("episodes") or [data.get("prompt", "")]
            output = data.get("output", "")
            self._index.append(
                {
                    "path": path,
                    "episodes": episodes,
                    "output": output,
                    "words": _words(" ".join(episodes) + " " + output),
                }
            )

    def add(self, run) -> str:
        """Store a finished run as experience. Returns the saved path."""
        blob = json.dumps(run.to_dict(), sort_keys=True)
        digest = hashlib.sha256(blob.encode()).hexdigest()[:12]
        path = os.path.join(self.directory, f"{digest}.loom.json")
        run.save(path)
        self.refresh()
        return path

    # -- similarity backends ------------------------------------------------

    def _vector_cache_path(self) -> str:
        return os.path.join(self.directory, ".loom-vectors.json")

    def _embed(self, texts: "list[str]") -> "list[list[float]]":
        """Embed with a per-store cache: each distinct text is embedded once."""
        if not self._vectors:
            try:
                with open(self._vector_cache_path()) as f:
                    self._vectors = json.load(f)
            except (OSError, json.JSONDecodeError):
                self._vectors = {}
        keys = [hashlib.sha256(t.encode()).hexdigest()[:16] for t in texts]
        missing = [(k, t) for k, t in zip(keys, texts) if k not in self._vectors]
        if missing:
            for (k, _), vec in zip(missing, self.embedder([t for _, t in missing])):
                self._vectors[k] = vec
            with open(self._vector_cache_path(), "w") as f:
                json.dump(self._vectors, f)
        return [self._vectors[k] for k in keys]

    def recall(self, query: str) -> list[dict]:
        """Top-k most similar past runs (cosine over embeddings when an
        embedder is configured, word-overlap jaccard otherwise)."""
        if not self._index:
            return []
        if self.embedder is not None:
            texts = [" ".join(e["episodes"]) + " " + e["output"] for e in self._index]
            vectors = self._embed(texts + [query])
            query_vec, entry_vecs = vectors[-1], vectors[:-1]
            scored = [
                (_cosine(vec, query_vec), entry)
                for vec, entry in zip(entry_vecs, self._index)
            ]
            scored.sort(key=lambda pair: -pair[0])
            return [e for score, e in scored[: self.k] if score > 0]
        qw = _words(query)
        if not qw:
            return []
        scored = []
        for entry in self._index:
            overlap = len(qw & entry["words"])
            if overlap:
                scored.append((overlap / len(qw | entry["words"]), entry))
        scored.sort(key=lambda pair: -pair[0])
        return [e for _, e in scored[: self.k]]

    def recall_text(self, query: str) -> str:
        """A context-ready summary of similar past runs ("" when none)."""
        hits = self.recall(query)
        if not hits:
            return ""
        lines = ["Relevant experience from similar past runs:"]
        for i, h in enumerate(hits, 1):
            question = h["episodes"][0][:120]
            outcome = h["output"][:160]
            lines.append(f"{i}. Task: {question}")
            lines.append(f"   Outcome: {outcome}")
        return "\n".join(lines)
