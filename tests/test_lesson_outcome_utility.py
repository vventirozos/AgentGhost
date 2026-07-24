"""Outcome-gated learning loop (2026-07-24).

The pre-existing retrieval-feedback loop only ever CREDITED lessons on
success (``helpful_retrievals`` via ``record_helpful_retrieval`` /
``credit_recent_retrievals``). It had no FAILURE arm, so a lesson surfaced
on a failing turn was indistinguishable from one surfaced on a success it
didn't get credited for — both just carried un-credited ``retrievals``.
That let a harmful lesson ride along on failures invisibly
(experience-following).

This suite covers the added failure arm:

  1. schema: ``succeeded_retrievals`` / ``failed_retrievals`` on build +
     legacy-safe normalize.
  2. ``record_surfaced_outcomes``: bulk, one-write, disk round-trip, and
     it only touches the correct arm.
  3. ``compute_lesson_utility``: the failure arm demotes; cold-start
     neutral below the min-observation gate; kill switch disables it.
  4. end-to-end: a lesson present on repeated FAILURES becomes
     prune-eligible and is removed by ``prune_low_utility`` while a
     never-failing peer survives — i.e. the loop actually closes.
"""
from __future__ import annotations

import json

import pytest

from ghost_agent.memory import skills as skills_mod
from ghost_agent.memory.skills import (
    SkillMemory,
    build_lesson,
    _normalize_lesson,
    compute_lesson_utility,
)


# --------------------------------------------------------------- schema


def test_build_lesson_has_outcome_arms_zeroed():
    lesson = build_lesson(task="x", anti_pattern="y", correct_pattern="z")
    assert lesson["succeeded_retrievals"] == 0
    assert lesson["failed_retrievals"] == 0


def test_normalize_backfills_outcome_arms_for_legacy_lessons():
    legacy = {"task": "fix x", "mistake": "did y", "solution": "do z"}
    out = _normalize_lesson(legacy)
    assert out["succeeded_retrievals"] == 0
    assert out["failed_retrievals"] == 0


def test_normalize_preserves_existing_outcome_arms():
    out = _normalize_lesson(
        {"trigger": "t", "succeeded_retrievals": 3, "failed_retrievals": 5})
    assert out["succeeded_retrievals"] == 3
    assert out["failed_retrievals"] == 5


# ------------------------------------------------ record_surfaced_outcomes


def _write_playbook(tmp_path, lessons):
    (tmp_path / "skills_playbook.json").write_text(json.dumps(lessons))


def test_record_surfaced_outcomes_failure_arm(tmp_path):
    _write_playbook(tmp_path, [
        {"trigger": "parse json", "correct_pattern": "use json.loads"},
    ])
    sm = SkillMemory(tmp_path)
    n = sm.record_surfaced_outcomes(["parse json"], success=False)
    assert n == 1
    pb = json.loads((tmp_path / "skills_playbook.json").read_text())
    assert pb[0]["failed_retrievals"] == 1
    assert pb[0]["succeeded_retrievals"] == 0


def test_record_surfaced_outcomes_success_arm(tmp_path):
    _write_playbook(tmp_path, [{"trigger": "parse json"}])
    sm = SkillMemory(tmp_path)
    sm.record_surfaced_outcomes(["parse json"], success=True)
    sm.record_surfaced_outcomes(["parse json"], success=True)
    pb = json.loads((tmp_path / "skills_playbook.json").read_text())
    assert pb[0]["succeeded_retrievals"] == 2
    assert pb[0]["failed_retrievals"] == 0


def test_record_surfaced_outcomes_is_bulk_and_case_insensitive(tmp_path):
    _write_playbook(tmp_path, [
        {"trigger": "Parse JSON"},
        {"trigger": "read file"},
        {"trigger": "unrelated"},
    ])
    sm = SkillMemory(tmp_path)
    n = sm.record_surfaced_outcomes(["parse json", "READ FILE"], success=False)
    assert n == 2
    pb = {p["trigger"]: p for p in
          json.loads((tmp_path / "skills_playbook.json").read_text())}
    assert pb["Parse JSON"]["failed_retrievals"] == 1
    assert pb["read file"]["failed_retrievals"] == 1
    # Unmatched lessons are left byte-for-byte untouched (same contract as
    # record_retrievals_bulk: only matched entries are normalized+rewritten).
    assert pb["unrelated"].get("failed_retrievals", 0) == 0


def test_record_surfaced_outcomes_ignores_empty(tmp_path):
    _write_playbook(tmp_path, [{"trigger": "t"}])
    sm = SkillMemory(tmp_path)
    assert sm.record_surfaced_outcomes([], success=False) == 0
    assert sm.record_surfaced_outcomes(["", "   "], success=True) == 0


# ---------------------------------------------- compute_lesson_utility arm


def _lesson(**over):
    base = {
        "trigger": "t", "retrievals": 6, "helpful_retrievals": 3,
        "confidence": 0.6, "verified": False, "frequency": 1,
        "succeeded_retrievals": 0, "failed_retrievals": 0,
    }
    base.update(over)
    return base


def test_failure_arm_demotes_relative_to_success_arm():
    """Two lessons identical except the outcome arms: the one present on
    failures must score strictly below the one present on successes."""
    harmful = compute_lesson_utility(_lesson(failed_retrievals=6))
    helpful = compute_lesson_utility(_lesson(succeeded_retrievals=6))
    assert harmful < helpful


def test_failure_arm_demotes_relative_to_no_outcome_data():
    neutral = compute_lesson_utility(_lesson())          # no decisive outcomes
    harmful = compute_lesson_utility(_lesson(failed_retrievals=6))
    assert harmful < neutral


def test_outcome_arm_is_cold_start_neutral_below_gate():
    """Below _OUTCOME_MIN_OBS decisive outcomes the arm must not move the
    score — one unlucky co-occurring failure can't punish a lesson."""
    assert skills_mod._OUTCOME_MIN_OBS >= 2
    below = skills_mod._OUTCOME_MIN_OBS - 1
    scored = compute_lesson_utility(_lesson(failed_retrievals=below))
    neutral = compute_lesson_utility(_lesson())
    assert scored == neutral


def test_kill_switch_disables_outcome_arm(monkeypatch):
    monkeypatch.setattr(skills_mod, "_OUTCOME_UTILITY_ENABLED", False)
    off = compute_lesson_utility(_lesson(failed_retrievals=6))
    neutral = compute_lesson_utility(_lesson())
    assert off == neutral


# ------------------------------------------------------- end-to-end prune


def test_present_on_failure_lesson_gets_pruned(tmp_path):
    """The loop closes: a lesson repeatedly surfaced on FAILING turns loses
    utility and is removed by prune_low_utility, while a never-failing peer
    (kept below the retrieval gate) survives untouched."""
    lessons = []
    # 10 benign peers, deliberately below min_retrievals so they can never be
    # prune-eligible regardless of score — isolates the harmful one.
    for i in range(10):
        lessons.append({
            "trigger": f"good-{i}", "retrievals": 2, "helpful_retrievals": 1,
            "confidence": 0.7, "succeeded_retrievals": 0,
            "failed_retrievals": 0, "timestamp": f"2026-07-24T00:00:{i:02d}",
        })
    # One harmful lesson: retrieved often, present on 8 failures.
    lessons.append({
        "trigger": "harmful", "retrievals": 8, "helpful_retrievals": 2,
        "confidence": 0.6, "succeeded_retrievals": 0, "failed_retrievals": 8,
        "timestamp": "2026-07-24T00:00:99",
    })
    _write_playbook(tmp_path, lessons)

    sm = SkillMemory(tmp_path)
    removed = sm.prune_low_utility(min_retrievals=5)
    assert removed == 1

    survivors = {p["trigger"] for p in
                 json.loads((tmp_path / "skills_playbook.json").read_text())}
    assert "harmful" not in survivors
    assert len(survivors) == 10


def test_verified_lesson_is_never_pruned_even_if_present_on_failures(tmp_path):
    """Verification is a stronger signal than co-occurrence statistics —
    prune_low_utility pins verified lessons (mirrors the existing contract)."""
    lessons = [{
        "trigger": f"good-{i}", "retrievals": 2, "confidence": 0.7,
        "timestamp": f"2026-07-24T00:00:{i:02d}",
    } for i in range(10)]
    lessons.append({
        "trigger": "verified-but-failing", "retrievals": 8,
        "helpful_retrievals": 1, "confidence": 0.6, "verified": True,
        "succeeded_retrievals": 0, "failed_retrievals": 8,
        "timestamp": "2026-07-24T00:00:99",
    })
    _write_playbook(tmp_path, lessons)

    sm = SkillMemory(tmp_path)
    sm.prune_low_utility(min_retrievals=5)
    survivors = {p["trigger"] for p in
                 json.loads((tmp_path / "skills_playbook.json").read_text())}
    assert "verified-but-failing" in survivors
