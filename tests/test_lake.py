"""loom lake / loom search: a local, service-free trace lake."""

import json
import os

from loom.cli import main
from loom.lake import Lake, dashboard_html


def _write_trace(path, prompt, output, tokens=(10, 5), tools=(), stop="end_turn",
                 denies=0):
    log = [
        {
            "seq": 0, "kind": "model", "key": "k", "depth": 0,
            "result": {
                "text": output, "stop_reason": stop,
                "tool_calls": [{"id": f"t{i}", "name": t, "input": {}} for i, t in enumerate(tools)],
                "usage": {"input_tokens": tokens[0], "output_tokens": tokens[1]},
            },
        }
    ] + [
        {"seq": i + 1, "kind": f"tool:{t}", "key": "k", "depth": 0, "result": "ok"}
        for i, t in enumerate(tools)
    ]
    data = {
        "model": "scripted", "episodes": [prompt], "output": output,
        "stop_reason": stop, "log": log,
        "shield_events": [{"action": "deny", "tool": "Read"}] * denies,
    }
    with open(path, "w") as f:
        json.dump(data, f)


def _corpus(tmp_path):
    _write_trace(tmp_path / "cheap.loom.json", "say hi", "hi there", tokens=(10, 5))
    _write_trace(tmp_path / "pricey.loom.json", "migrate the database",
                 "migration done", tokens=(90000, 400), tools=("Bash",))
    _write_trace(tmp_path / "blocked.loom.json", "read config", "was blocked",
                 tokens=(50, 20), stop="budget", denies=2)
    return str(tmp_path)


def test_index_is_incremental_and_forgets_deleted(tmp_path):
    d = _corpus(tmp_path)
    lake = Lake(d)
    assert lake.index() == (3, 3)
    assert lake.index() == (0, 3)  # nothing changed, nothing re-read

    os.remove(tmp_path / "cheap.loom.json")
    assert lake.index() == (0, 2)  # deletions fall out of the index
    lake.close()


def test_search_filters(tmp_path):
    lake = Lake(_corpus(tmp_path))
    lake.index()

    assert [r["path"] for r in lake.search("cost>50000")] == [
        str(tmp_path / "pricey.loom.json")
    ]
    assert len(lake.search("cost<1000")) == 2
    assert [os.path.basename(r["path"]) for r in lake.search("tool:Bash")] == [
        "pricey.loom.json"
    ]
    assert [os.path.basename(r["path"]) for r in lake.search("failed")] == [
        "blocked.loom.json"
    ]
    assert [os.path.basename(r["path"]) for r in lake.search("shield:deny")] == [
        "blocked.loom.json"
    ]
    assert [os.path.basename(r["path"]) for r in lake.search("migration")] == [
        "pricey.loom.json"
    ]
    # terms AND together; results ordered by cost desc
    assert lake.search("tool:Bash failed") == []
    assert [os.path.basename(r["path"]) for r in lake.search("")][0] == "pricey.loom.json"
    lake.close()


def test_search_rejects_bad_cost(tmp_path):
    lake = Lake(_corpus(tmp_path))
    lake.index()
    import pytest

    with pytest.raises(ValueError):
        lake.search("cost>lots")
    lake.close()


def test_stats_and_dashboard(tmp_path):
    d = _corpus(tmp_path)
    lake = Lake(d)
    lake.index()
    stats = lake.stats()
    lake.close()
    assert stats["runs"] == 3
    assert stats["input_tokens"] == 90060 and stats["denies"] == 2
    assert stats["failed"] == 1
    assert ("Bash", 1) in stats["top_tools"]

    html = dashboard_html(stats, d)
    assert "pricey.loom.json" in html and "90,400" in html
    assert "shield" in html  # the blocked run wears its badge
    assert "prefers-color-scheme: dark" in html


def test_cli_search_and_lake(tmp_path, capsys):
    d = _corpus(tmp_path)
    assert main(["search", d, "cost>50000"]) == 0
    out = capsys.readouterr().out
    assert "pricey.loom.json" in out and "1 run(s)" in out

    assert main(["search", d, "no-such-text-anywhere"]) == 1
    capsys.readouterr()

    assert main(["lake", d]) == 0
    out = capsys.readouterr().out
    assert "indexed 3 run(s)" in out
    assert os.path.exists(os.path.join(d, "lake.html"))


def test_lake_indexes_and_searches_risk(tmp_path):
    _write_trace(tmp_path / "safe.loom.json", "list files", "done", tools=("Glob",))
    # a run that shells out and fetches the network
    import json as _json
    log = [{
        "seq": 0, "kind": "model", "key": "k", "depth": 0,
        "result": {"text": "", "stop_reason": "tool_use",
                   "tool_calls": [{"id": "t1", "name": "Bash", "input": {"command": "curl http://x"}}],
                   "usage": {"input_tokens": 5, "output_tokens": 2}},
    }]
    (tmp_path / "risky.loom.json").write_text(_json.dumps(
        {"model": "m", "episodes": ["fetch"], "output": "ok", "stop_reason": "end_turn", "log": log}
    ))
    lake = Lake(str(tmp_path))
    lake.index()
    hits = [os.path.basename(r["path"]) for r in lake.search("risky")]
    assert hits == ["risky.loom.json"]
    assert [os.path.basename(r["path"]) for r in lake.search("risk:network-egress")] == ["risky.loom.json"]
    assert lake.stats()["risky"] == 1
    lake.close()
