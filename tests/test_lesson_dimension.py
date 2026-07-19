"""Tests for the `dimension` field on lessons (2026-07-19).

The write chokepoint (`SkillMemory.learn_lesson`) attributes every
real-mistake lesson to a harness dimension so producers never have to.

These tests pin:
  * build_lesson / _normalize_lesson defaults (empty string, additive)
  * chokepoint auto-classification for real mistakes; no attribution for
    mistake-less rules; GHOST_FAILURE_DIM=0 kill switch
  * an explicit dimension kwarg wins over the heuristic
  * dedup merges back-fill dimension onto an existing unattributed row
  * the vector twin metadata carries the dimension
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.memory.skills import (
    SkillMemory, build_lesson, _normalize_lesson,
)

_REAL_TRIGGER = "code correction parser wrote stray markers"
_REAL_MISTAKE = ("SEARCH/REPLACE block failed to parse and markers were "
                 "written into the file")


@pytest.fixture
def sm(tmp_path):
    return SkillMemory(tmp_path)


def _lessons(sm):
    return sm.list_lessons(scope="all", limit=50)


# ------------------------------------------------------------- schema

class TestSchema:
    def test_build_lesson_defaults_empty(self):
        lesson = build_lesson(trigger="t", anti_pattern="a",
                              correct_pattern="c")
        assert lesson["dimension"] == ""

    def test_build_lesson_carries_dimension(self):
        lesson = build_lesson(trigger="t", anti_pattern="a",
                              correct_pattern="c", dimension="memory")
        assert lesson["dimension"] == "memory"

    def test_normalize_backfills_legacy(self):
        legacy = {"task": "t", "mistake": "m", "solution": "s"}
        assert _normalize_lesson(legacy)["dimension"] == ""


# ------------------------------------------------------------- chokepoint

class TestChokepoint:
    def test_real_mistake_gets_classified(self, sm):
        sm.learn_lesson(
            _REAL_TRIGGER, _REAL_MISTAKE,
            "Fail closed: reject the edit when markers remain after "
            "substitution.")
        (lesson,) = _lessons(sm)
        assert lesson["dimension"] == "output_processing"

    def test_mistake_less_rule_gets_no_dimension(self, sm):
        sm.learn_lesson(
            "Always use absolute paths in Docker.", "none",
            "Always use absolute paths in Docker.")
        (lesson,) = _lessons(sm)
        assert lesson["dimension"] == ""

    def test_kill_switch(self, sm, monkeypatch):
        monkeypatch.setenv("GHOST_FAILURE_DIM", "0")
        sm.learn_lesson(_REAL_TRIGGER, _REAL_MISTAKE, "Fail closed.")
        (lesson,) = _lessons(sm)
        assert lesson["dimension"] == ""

    def test_explicit_dimension_wins(self, sm):
        sm.learn_lesson(
            _REAL_TRIGGER, _REAL_MISTAKE, "Fail closed.",
            dimension="orchestration")
        (lesson,) = _lessons(sm)
        assert lesson["dimension"] == "orchestration"

    def test_unclassifiable_stays_empty(self, sm):
        sm.learn_lesson(
            "frobnicator assembly calibration",
            "the frobnicator produced wrong colours",
            "Calibrate the frobnicator before each batch.")
        (lesson,) = _lessons(sm)
        assert lesson["dimension"] == ""

    def test_dedup_backfills_dimension(self, sm, monkeypatch):
        # First write lands unattributed (kill switch on) — the shape of
        # every pre-feature legacy row.
        monkeypatch.setenv("GHOST_FAILURE_DIM", "0")
        sm.learn_lesson(_REAL_TRIGGER, _REAL_MISTAKE, "Fail closed.")
        assert _lessons(sm)[0]["dimension"] == ""
        # Re-learning the same trigger with the feature on must merge
        # (frequency bump), not duplicate, and back-fill the dimension.
        monkeypatch.delenv("GHOST_FAILURE_DIM")
        sm.learn_lesson(
            _REAL_TRIGGER, _REAL_MISTAKE,
            "Fail closed: reject the edit when markers remain after "
            "substitution and surface the parse failure.")
        lessons = _lessons(sm)
        assert len(lessons) == 1
        assert lessons[0]["dimension"] == "output_processing"
        assert lessons[0]["frequency"] >= 2


# ------------------------------------------------------------- vector twin

class TestVectorTwin:
    def test_meta_carries_dimension(self, sm):
        captured = {}

        class _MemStub:
            def add(self, text, meta):
                captured.update(meta)

            def search(self, *a, **kw):
                return []

        sm.learn_lesson(_REAL_TRIGGER, _REAL_MISTAKE, "Fail closed.",
                        memory_system=_MemStub())
        assert captured.get("dimension") == "output_processing"
