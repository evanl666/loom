"""TraceMemory with an embedder: semantic recall, vectors cached per store."""

import json
import os

from loom.memory import TraceMemory, _cosine


class _KeywordEmbedder:
    """Deterministic 'semantic' embeddings: axis 0 = weather-ish, axis 1 = db-ish."""

    def __init__(self):
        self.calls = 0

    def __call__(self, texts):
        self.calls += 1
        return [
            [
                float(any(w in t.lower() for w in ("rain", "weather", "forecast"))),
                float(any(w in t.lower() for w in ("sql", "database", "query"))),
                0.1,  # keep vectors non-zero
            ]
            for t in texts
        ]


def _store(tmp_path):
    for name, episodes, output in [
        ("a", ["will it rain tomorrow?"], "yes, heavy rain"),
        ("b", ["optimize this sql join"], "use an index on user_id"),
    ]:
        with open(tmp_path / f"{name}.loom.json", "w") as f:
            json.dump({"episodes": episodes, "output": output, "log": []}, f)


def test_embedder_recall_matches_meaning_not_words(tmp_path):
    _store(tmp_path)
    memory = TraceMemory(str(tmp_path), k=1, embedder=_KeywordEmbedder())
    # "forecast" shares no words with "will it rain tomorrow?" -- jaccard would
    # miss it, the embedding space puts them on the same axis.
    hits = memory.recall("weather forecast for berlin")
    assert len(hits) == 1 and hits[0]["output"] == "yes, heavy rain"
    hits = memory.recall("slow database")
    assert hits[0]["output"] == "use an index on user_id"


def test_vectors_are_cached_per_store(tmp_path):
    _store(tmp_path)
    embedder = _KeywordEmbedder()
    memory = TraceMemory(str(tmp_path), embedder=embedder)
    memory.recall("weather forecast")
    assert embedder.calls == 1  # one batched call for entries + query
    memory.recall("weather forecast")
    assert embedder.calls == 1  # everything already cached

    assert os.path.exists(os.path.join(str(tmp_path), ".loom-vectors.json"))
    # a fresh process reuses the on-disk cache: the embedder is never called
    cold = TraceMemory(str(tmp_path), embedder=_KeywordEmbedder())
    cold.recall("weather forecast")
    assert cold.embedder.calls == 0


def test_no_embedder_still_uses_word_overlap(tmp_path):
    _store(tmp_path)
    hits = TraceMemory(str(tmp_path), k=1).recall("rain tomorrow")
    assert hits and hits[0]["output"] == "yes, heavy rain"


def test_cosine_basics():
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert _cosine([0.0, 0.0], [0.0, 0.0]) == 0.0  # zero vectors don't crash
