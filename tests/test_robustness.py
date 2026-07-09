"""Analyzers tolerate hand-edited / third-party / corrupted traces."""

import pytest

from loom.action import actions, effect_dicts
from loom.autopsy import autopsy_html
from loom.cost import analyze_cost
from loom.diagnose import diagnose
from loom.diff import score_breakdown
from loom.export import trace_to_html
from loom.incident import build_report
from loom.insight import provenance, side_effect_map
from loom.movie import movie_html
from loom.packs import install_builtin
from loom.providers.base import ModelResponse
from loom.taint import dlp_report, taint_paths

install_builtin()

MALFORMED = [
    {"log": "not a list"},
    {"log": ["a string entry", 42, None]},
    {"log": [{"seq": 0, "kind": "model", "key": "k", "result": "not a dict"}]},
    {"log": [{"seq": 0, "kind": "model", "key": "k",
              "result": {"text": "", "tool_calls": None, "usage": None}}]},
    {"log": [], "shield_events": None},
    {"log": [{"seq": 0, "kind": "tool:x", "key": "k", "result": None}]},
    {},  # no keys at all
]

ANALYZERS = [actions, analyze_cost, score_breakdown, movie_html, taint_paths,
             dlp_report, diagnose, side_effect_map, provenance, trace_to_html,
             autopsy_html]


@pytest.mark.parametrize("data", MALFORMED)
def test_analyzers_never_crash_on_malformed_traces(data):
    for fn in ANALYZERS:
        fn(data)                      # must not raise
    build_report(data, "x.loom.json")  # nor the incident report


def test_model_response_from_dict_tolerates_null_and_non_dict():
    assert ModelResponse.from_dict({"tool_calls": None}).tool_calls == []
    assert ModelResponse.from_dict({"usage": None}).usage == {}
    assert ModelResponse.from_dict("not a dict").text == ""
    # a non-dict tool_call entry is skipped, not crashed on
    assert ModelResponse.from_dict({"tool_calls": ["junk", {"id": "t", "name": "X"}]}).tool_calls[0].name == "X"


def test_effect_dicts_normalizes_the_log():
    assert effect_dicts({"log": "oops"}) == []
    assert effect_dicts({"log": ["x", {"seq": 0, "kind": "model"}, 5]}) == [{"seq": 0, "kind": "model"}]
    assert effect_dicts({}) == []
