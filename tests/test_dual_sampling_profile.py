"""Tests for the dual-temperature policy keyed on tool-use vs conversation.

Model-card policy (Qwen 3.5):
* General conversation: temperature=1.0, top_p=0.95, top_k=20,
  min_p=0, presence_penalty=1.5.
* Coding / precise tasks: temperature=0.6, top_p=0.95, top_k=20,
  min_p=0, presence_penalty=0.

The earlier wiring only dropped to 0.6 when the sub-classifier decided
the turn was "coding". Any other tool-using turn (profile updates,
web searches, manage_tasks, recall, etc.) ran at 1.0 — which produced
a reproducible over-eagerness bug: a successful `update_profile` was
re-issued on the following turn, caught by the idempotency guard, and
the loop burned an extra turn before the model conceded.

These tests pin the new policy:
* is_tool_turn=False → general profile (temp 1.0, pp 1.5).
* is_tool_turn=True + is_coding=False → base precise profile
  (temp 0.6, pp 0).
* is_tool_turn=True + is_coding=True → sub-classified (creative /
  precise / balanced).
"""
from __future__ import annotations

import pytest

from ghost_agent.core.agent import (
    CODING_SAMPLING_PARAMS,
    GENERAL_SAMPLING_PARAMS,
    get_sampling_params,
)


# ---------------------------------------------------------------------------
# Model-card alignment — the two fixed profiles must match the card.
# ---------------------------------------------------------------------------

def test_general_profile_matches_model_card():
    assert GENERAL_SAMPLING_PARAMS["temperature"] == 1.0
    assert GENERAL_SAMPLING_PARAMS["top_p"] == 0.95
    assert GENERAL_SAMPLING_PARAMS["top_k"] == 20
    assert GENERAL_SAMPLING_PARAMS["min_p"] == 0
    assert GENERAL_SAMPLING_PARAMS["presence_penalty"] == 1.5


def test_coding_profile_matches_model_card():
    assert CODING_SAMPLING_PARAMS["temperature"] == 0.6
    assert CODING_SAMPLING_PARAMS["top_p"] == 0.95
    assert CODING_SAMPLING_PARAMS["top_k"] == 20
    assert CODING_SAMPLING_PARAMS["min_p"] == 0
    assert CODING_SAMPLING_PARAMS["presence_penalty"] == 0


# ---------------------------------------------------------------------------
# Routing rules.
# ---------------------------------------------------------------------------

def test_conversation_turn_uses_general_profile():
    p = get_sampling_params(is_tool_turn=False)
    assert p == GENERAL_SAMPLING_PARAMS
    assert p["temperature"] == 1.0
    assert p["presence_penalty"] == 1.5


def test_non_coding_tool_turn_uses_base_precise_profile():
    """The regression this whole change fixes — a non-coding tool turn
    (update_profile, web_search, manage_tasks, etc.) must route through
    the coding base profile, not the warm conversational one."""
    p = get_sampling_params(is_tool_turn=True, is_coding=False)
    assert p == CODING_SAMPLING_PARAMS
    assert p["temperature"] == 0.6
    assert p["presence_penalty"] == 0


def test_coding_tool_turn_sub_classifies_creative():
    # "design" and "brainstorm" both hit _CREATIVE_KEYWORDS; no precise
    # keywords in the query, so the classifier picks "creative".
    p = get_sampling_params(
        is_tool_turn=True,
        query="design and brainstorm alternative approaches",
        is_coding=True,
    )
    assert p["temperature"] == 0.8  # creative profile


def test_coding_tool_turn_sub_classifies_precise_for_sql():
    p = get_sampling_params(
        is_tool_turn=True,
        query="write the exact SQL migration for the users table",
        is_coding=True,
    )
    assert p["temperature"] == 0.3  # precise profile


def test_coding_tool_turn_sub_classifies_balanced_for_plain_code():
    p = get_sampling_params(
        is_tool_turn=True,
        query="implement a helper function",
        is_coding=True,
    )
    assert p["temperature"] == 0.6  # balanced (same temp as base precise)


# ---------------------------------------------------------------------------
# Source-level wiring — the call site must pass the tool-turn flag, not
# only the coding flag. A future change that reverts to the old
# "is_coding only" shape would re-introduce the bug; this guard catches
# it loudly.
# ---------------------------------------------------------------------------

def test_agent_call_site_passes_is_tool_turn():
    from pathlib import Path
    src = (
        Path(__file__).resolve().parents[1]
        / "src/ghost_agent/core/agent.py"
    ).read_text()
    assert "is_tool_turn = not turn_is_conversational" in src, (
        "agent.py must derive is_tool_turn from turn_is_conversational"
    )
    assert "get_sampling_params(\n                        is_tool_turn" in src or (
        "is_tool_turn," in src and "is_coding=has_coding_intent" in src
    ), "call site must forward is_tool_turn + is_coding to get_sampling_params"
