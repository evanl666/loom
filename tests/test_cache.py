"""EffectCache load tolerates malformed persisted lines."""

from loom.cache import EffectCache


def test_cache_load_skips_malformed_lines_and_keeps_valid_ones(tmp_path):
    """A valid-JSON line missing key/result (or a bare scalar) must be skipped,
    not crash the load and lose every entry after it."""
    p = tmp_path / "c.jsonl"
    p.write_text(
        '{"key": "a", "result": 1}\n'
        '{"garbage": true}\n'   # valid JSON, missing key/result
        "42\n"                  # valid JSON, not an object
        "{torn\n"               # torn tail
        '{"key": "b", "result": 2}\n'
    )
    cache = EffectCache(str(p))
    assert cache.get("a") == 1
    assert cache.get("b") == 2
