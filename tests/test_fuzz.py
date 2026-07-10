"""The hostile-trace fuzzer must find zero crashes -- every analyzer tolerates
malformed / third-party / corrupted traces (a clear error is fine, a traceback
is not). This is the CI guard for the untrusted-artifact robustness guarantee."""

from loom.fuzz import fuzz_check, mutations


def test_fuzzer_finds_no_analyzer_crashes():
    report = fuzz_check()
    assert report["ok"], "analyzers crashed on hostile traces:\n" + "\n".join(
        f"  {c['mutation']} / {c['analyzer']}: {c['error']}" for c in report["crashes"]
    )
    assert report["mutations"] >= 15 and report["analyzers"] >= 8


def test_fuzzer_mutates_a_real_seed():
    seed = {"prompt": "p", "output": "o", "log": [
        {"seq": 0, "kind": "model", "result": {"text": "hi", "tool_calls": []}}]}
    muts = mutations(seed)
    assert len(muts) >= 15
    names = {m[0] for m in muts}
    assert "log-not-a-list" in names and "artifact-pointer-traversal" in names
