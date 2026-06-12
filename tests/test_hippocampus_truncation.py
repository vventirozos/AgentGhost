"""Hippocampus consolidation truncation.

Production failure: the smart-memory consolidation call capped output at
max_tokens=1024, the model's JSON died mid-object (right after
`"profile_update":`), and `extract_json_from_text` dropped the whole
extraction. Two-sided fix: a larger cap + json_object response_format on
the call, and a truncated-JSON repair pass in the extractor so a cap hit
salvages every complete key/value pair instead of losing the memory.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import inspect
import re

from ghost_agent.core.agent import (
    GhostAgent,
    _repair_truncated_json,
    extract_json_from_text,
)


# ------------------------------------------------------------ repair helper

def test_repair_production_truncation_shape():
    """The exact shape observed in the production log."""
    text = ('{ "score": 0.8,   "fact": "User has a PostgreSQL table named '
            'web_order_line_options with columns id and header_id",   '
            '"profile_update":')
    res = extract_json_from_text(text, repair_truncated=True)
    assert res.get("score") == 0.8
    assert "web_order_line_options" in res.get("fact", "")
    # the dangling key is completed with null, not dropped silently
    assert "profile_update" in res and res["profile_update"] is None


def test_repair_mid_string_truncation():
    res = _repair_truncated_json('{"score": 0.8, "fact": "User has a Post')
    assert res == {"score": 0.8, "fact": "User has a Post"}


def test_repair_nested_open_structures():
    assert _repair_truncated_json('{"a": {"b": [1, 2') == {"a": {"b": [1, 2]}}


def test_repair_trailing_comma():
    assert _repair_truncated_json('{"a": 1,') == {"a": 1}


def test_repair_partial_key_dropped():
    res = _repair_truncated_json('{"a": 1, "partial_ke')
    assert res == {"a": 1}


def test_repair_garbage_returns_empty():
    assert _repair_truncated_json("hello") == {}
    assert _repair_truncated_json("{not json") == {}
    assert _repair_truncated_json("") == {}


def test_extractor_well_formed_unchanged():
    assert extract_json_from_text('{"a": 1, "b": [2]}') == {"a": 1, "b": [2]}


def test_extractor_no_json_still_empty():
    assert extract_json_from_text("no json here at all") == {}


def test_extractor_escaped_quote_in_truncated_string():
    res = _repair_truncated_json('{"fact": "user said \\"hi')
    assert res == {"fact": 'user said "hi'}


# ----------------------------------------------------- consolidation payload

def test_smart_memory_payload_has_room_and_json_mode():
    """Pin the call-site fix: a 1024 cap reliably truncated real
    consolidations (score + fact + profile_update + graph_triplets)."""
    src = inspect.getsource(GhostAgent.run_smart_memory_task)
    m = re.search(r'"max_tokens":\s*(\d+)', src)
    assert m, "smart memory payload must set an explicit max_tokens"
    assert int(m.group(1)) >= 2048
    assert "json_object" in src
    # and the extraction opts in to truncation salvage
    assert "repair_truncated=True" in src
