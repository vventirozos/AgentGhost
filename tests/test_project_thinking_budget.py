"""Tests for the in_active_project signal on classify_thinking_budget.

Regression target (2026-04-19 trace): during long-running project
work the model spent 15-25s of 'thinking' per turn reasoning about
which task to mark done next. Those turns should be tight.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.core.agent import classify_thinking_budget


# ---- baseline: pre-existing logic unchanged when not in project -------------

def test_extended_still_fires_for_complex_query_outside_project():
    # Two strong keywords should still trigger extended without a project
    q = "debug this race condition in the concurrent job queue"
    assert classify_thinking_budget(q) == "extended"


def test_tight_for_trivial_query_outside_project():
    assert classify_thinking_budget("hi") == "tight"


def test_meta_task_always_tight():
    assert classify_thinking_budget("debug this race condition",
                                    is_meta_task=True) == "tight"


# ---- in-project behavior ----------------------------------------------------

def test_tight_when_inside_project_with_two_keywords():
    """Two extended keywords normally trigger extended. Inside a
    project we raise the bar — most turns are bookkeeping, not
    new reasoning."""
    q = "debug this race condition"
    assert classify_thinking_budget(q, in_active_project=True) == "tight"


def test_extended_when_inside_project_with_three_keywords():
    """Genuinely complex new ask still gets extended room."""
    q = "debug the concurrent race in the optimization algorithm"
    # Keywords: debug, race, concurrent, optim, algorithm
    assert classify_thinking_budget(q, in_active_project=True) == "extended"


def test_tight_for_empty_query_in_project():
    assert classify_thinking_budget("", in_active_project=True) == "tight"


def test_tight_for_task_update_continuation_in_project():
    """The exact shape of turn content we saw in the trace: just a
    short follow-up describing what to do next."""
    q = "mark the parser task as done and move to testing"
    assert classify_thinking_budget(q, in_active_project=True) == "tight"


def test_in_project_ignores_coding_intent_single_keyword():
    """Outside a project, coding_intent + one keyword → extended.
    Inside a project, same input → tight. This is the key difference."""
    q = "implement the refactor"
    assert classify_thinking_budget(q, has_coding_intent=True) == "extended"
    assert classify_thinking_budget(q, has_coding_intent=True,
                                    in_active_project=True) == "tight"
