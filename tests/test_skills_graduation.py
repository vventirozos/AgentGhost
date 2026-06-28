"""Tests for the skills-layer graduation/utility fixes (review follow-up).

Covers three behaviours that the review flagged as broken:

1. ``_normalize_trigger`` collapses semantically-equivalent / paraphrased
   triggers onto the SAME key (so the frequency counter can climb).
2. ``learn_lesson`` frequency actually accumulates across paraphrased
   triggers (the exact-string match never collided before).
3. ``credit_recent_retrievals`` is DISCRIMINATIVE — it credits lessons that
   are relevant to the succeeding query and skips irrelevant ones, while
   still honouring the legacy "credit everything" path when no query is
   supplied.

Pure-python; no docker / live model.
"""

import pytest

from ghost_agent.memory.skills import (
    SkillMemory,
    _normalize_trigger,
    _trigger_token_set,
)


# --------------------------------------------------------------------------
# 1. _normalize_trigger
# --------------------------------------------------------------------------
def test_normalize_trigger_collapses_equivalent_triggers():
    # Punctuation, casing, stopwords and word-order should all wash out.
    assert _normalize_trigger("How do I parse JSON?") == _normalize_trigger("parse JSON")
    assert _normalize_trigger("Parse JSON!!!") == _normalize_trigger("parse json")
    assert _normalize_trigger("Please parse JSON!") == _normalize_trigger("How do I parse JSON?")
    # Word order does not matter (significant tokens are sorted).
    assert _normalize_trigger("read file safely") == _normalize_trigger("safely read file")


def test_normalize_trigger_distinguishes_unrelated_triggers():
    assert _normalize_trigger("parse JSON") != _normalize_trigger("deploy kubernetes cluster")


def test_normalize_trigger_handles_empty_and_all_stopwords():
    assert _normalize_trigger("") == ""
    # All-stopword input falls back to raw tokens rather than collapsing to "".
    assert _normalize_trigger("how do I") != ""


def test_trigger_token_set_strips_stopwords():
    assert _trigger_token_set("How do I parse JSON?") == {"parse", "json"}


# --------------------------------------------------------------------------
# 2. frequency accumulates across paraphrased triggers
# --------------------------------------------------------------------------
def test_frequency_accumulates_across_paraphrased_triggers(tmp_path):
    sm = SkillMemory(tmp_path)

    sm.learn_lesson("How do I parse JSON?", "used eval()", "use json.loads")
    sm.learn_lesson("parse JSON", "used eval()", "use json.loads")
    sm.learn_lesson("Please parse JSON!", "used eval()", "use json.loads")

    playbook = sm._load_playbook()
    # All three paraphrases collapse onto a single lesson...
    assert len(playbook) == 1
    # ...whose frequency climbed to 3 (1 create + 2 merges).
    assert int(playbook[0].get("frequency") or 0) == 3


def test_distinct_triggers_do_not_merge(tmp_path):
    sm = SkillMemory(tmp_path)
    sm.learn_lesson("parse JSON", "m", "s")
    sm.learn_lesson("deploy kubernetes cluster", "m2", "s2")
    playbook = sm._load_playbook()
    assert len(playbook) == 2


# --------------------------------------------------------------------------
# 3. credit_recent_retrievals is discriminative
# --------------------------------------------------------------------------
def _seed_two_retrieved_lessons(tmp_path):
    sm = SkillMemory(tmp_path)
    sm.learn_lesson("parse JSON file", "used eval()", "use json.loads")
    sm.learn_lesson("deploy kubernetes cluster", "kubectl typo", "use kubectl apply -f")
    # Mark both as just-retrieved so they fall inside the credit window.
    assert sm.record_retrieval("parse JSON file") is True
    assert sm.record_retrieval("deploy kubernetes cluster") is True
    return sm


def _helpful(sm, trigger_key):
    for l in sm._load_playbook():
        if _normalize_trigger(l.get("trigger") or l.get("task") or "") == _normalize_trigger(trigger_key):
            return int(l.get("helpful_retrievals") or 0)
    raise AssertionError(f"lesson {trigger_key!r} not found")


def test_credit_only_relevant_lesson(tmp_path):
    sm = _seed_two_retrieved_lessons(tmp_path)

    credited = sm.credit_recent_retrievals(query="how to parse a JSON file in python")

    assert credited == 1
    assert _helpful(sm, "parse JSON file") == 1      # relevant -> credited
    assert _helpful(sm, "deploy kubernetes cluster") == 0  # irrelevant -> NOT credited


def test_credit_via_top_triggers(tmp_path):
    sm = _seed_two_retrieved_lessons(tmp_path)

    # No query token overlap, but the lesson was the top-ranked retrieval.
    credited = sm.credit_recent_retrievals(
        query="",
        top_triggers=["deploy kubernetes cluster"],
    )

    assert credited == 1
    assert _helpful(sm, "deploy kubernetes cluster") == 1
    assert _helpful(sm, "parse JSON file") == 0


def test_legacy_no_query_credits_everything(tmp_path):
    sm = _seed_two_retrieved_lessons(tmp_path)

    # Backward-compatible path: no query / no top_triggers -> credit all
    # recently-retrieved lessons (original behaviour preserved for old callers).
    credited = sm.credit_recent_retrievals()

    assert credited == 2
    assert _helpful(sm, "parse JSON file") == 1
    assert _helpful(sm, "deploy kubernetes cluster") == 1


def test_credit_is_idempotent_per_retrieval(tmp_path):
    sm = _seed_two_retrieved_lessons(tmp_path)

    first = sm.credit_recent_retrievals(query="parse JSON file")
    second = sm.credit_recent_retrievals(query="parse JSON file")

    assert first == 1
    assert second == 0  # same retrieval not double-counted
    assert _helpful(sm, "parse JSON file") == 1
