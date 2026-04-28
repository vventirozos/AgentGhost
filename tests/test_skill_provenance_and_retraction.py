"""Tests for SkillMemory provenance + retraction.

Two layers, both required for the user-correction loop to scrub
poisoned lessons:

  1. Every lesson written via ``learn_lesson`` carries a
     ``source_trajectory_id`` that round-trips through
     `build_lesson` / `_normalize_lesson` / the JSON playbook /
     the vector-store metadata.

  2. ``retract_lessons_from_trajectory(traj_id)`` removes every
     lesson whose source matches, atomically updates the playbook,
     and best-effort scrubs the vector store. Idempotent. Empty /
     missing ids are no-ops. Legacy lessons (no provenance) are
     never accidentally retracted.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ghost_agent.memory.skills import (
    SkillMemory,
    build_lesson,
    _normalize_lesson,
)


# --------------------------------------------------------------- schema


def test_build_lesson_records_source_trajectory_id():
    lesson = build_lesson(
        task="x",
        anti_pattern="y",
        correct_pattern="z",
        source_trajectory_id="abc123",
    )
    assert lesson.get("source_trajectory_id") == "abc123"


def test_build_lesson_default_provenance_is_empty_string():
    lesson = build_lesson(task="x", anti_pattern="y", correct_pattern="z")
    assert lesson.get("source_trajectory_id") == ""


def test_normalize_lesson_back_fills_missing_provenance_field():
    """A legacy on-disk lesson lacks the new field. The normalizer
    must fill it with an empty string so retrieval / retraction
    code paths can rely on the key being present."""
    legacy = {
        "task": "fix x",
        "mistake": "did y",
        "solution": "do z",
    }
    out = _normalize_lesson(legacy)
    assert "source_trajectory_id" in out
    assert out["source_trajectory_id"] == ""


# --------------------------------------------------------------- learn_lesson + persistence


def test_learn_lesson_persists_provenance_to_playbook(tmp_path):
    sm = SkillMemory(tmp_path)
    sm.learn_lesson(
        task="how to count lines",
        mistake="ran the count instead of giving code",
        solution="show the find/wc one-liner",
        source_trajectory_id="traj-A",
        source="perfection_protocol",
    )
    pb = json.loads((tmp_path / "skills_playbook.json").read_text())
    assert pb
    assert pb[0].get("source_trajectory_id") == "traj-A"
    assert pb[0].get("source") == "perfection_protocol"


def test_learn_lesson_passes_provenance_to_vector_metadata(tmp_path):
    sm = SkillMemory(tmp_path)
    captured = {}

    class _StubMem:
        def __init__(self):
            self.collection = MagicMock()

        def add(self, text, meta):
            captured["text"] = text
            captured["meta"] = meta

    sm.learn_lesson(
        task="a",
        mistake="b",
        solution="c",
        memory_system=_StubMem(),
        source_trajectory_id="traj-B",
    )
    assert captured.get("meta", {}).get("source_trajectory_id") == "traj-B"


def test_learn_lesson_without_provenance_writes_empty_string(tmp_path):
    """Back-compat: callers that don't pass `source_trajectory_id`
    must still work — the lesson lands with an empty string and
    retraction won't accidentally match it."""
    sm = SkillMemory(tmp_path)
    sm.learn_lesson(task="a", mistake="b", solution="c")
    pb = json.loads((tmp_path / "skills_playbook.json").read_text())
    assert pb[0].get("source_trajectory_id") == ""


# --------------------------------------------------------------- retraction


def test_retract_removes_matching_lessons(tmp_path):
    sm = SkillMemory(tmp_path)
    sm.learn_lesson(
        task="poisoned",
        mistake="x",
        solution="y",
        source_trajectory_id="traj-poison",
    )
    sm.learn_lesson(
        task="unrelated",
        mistake="x",
        solution="y",
        source_trajectory_id="traj-other",
    )
    sm.learn_lesson(  # legacy / no provenance
        task="legacy",
        mistake="x",
        solution="y",
    )

    removed = sm.retract_lessons_from_trajectory("traj-poison")
    assert removed == 1

    pb = json.loads((tmp_path / "skills_playbook.json").read_text())
    triggers = {p.get("trigger") for p in pb}
    assert "poisoned" not in triggers
    assert "unrelated" in triggers
    assert "legacy" in triggers


def test_retract_is_idempotent(tmp_path):
    sm = SkillMemory(tmp_path)
    sm.learn_lesson(
        task="a", mistake="b", solution="c",
        source_trajectory_id="traj-A",
    )
    assert sm.retract_lessons_from_trajectory("traj-A") == 1
    assert sm.retract_lessons_from_trajectory("traj-A") == 0
    assert sm.retract_lessons_from_trajectory("traj-A") == 0


def test_retract_never_removes_legacy_empty_provenance_lessons(tmp_path):
    """Empty source_trajectory_id (legacy lessons, lessons written by
    callers that don't yet thread provenance) must NEVER match a
    retraction call. Otherwise retracting trajectory '' would scrub
    every legacy lesson at once."""
    sm = SkillMemory(tmp_path)
    sm.learn_lesson(task="legacy1", mistake="x", solution="y")
    sm.learn_lesson(task="legacy2", mistake="x", solution="y")
    assert sm.retract_lessons_from_trajectory("") == 0
    pb = json.loads((tmp_path / "skills_playbook.json").read_text())
    assert len(pb) == 2


@pytest.mark.parametrize("bad", [None, 0, [], {}, 123])
def test_retract_rejects_non_string_ids(tmp_path, bad):
    sm = SkillMemory(tmp_path)
    sm.learn_lesson(task="x", mistake="y", solution="z",
                    source_trajectory_id="real")
    assert sm.retract_lessons_from_trajectory(bad) == 0
    pb = json.loads((tmp_path / "skills_playbook.json").read_text())
    assert len(pb) == 1


def test_retract_calls_vector_collection_delete_with_where_filter(tmp_path):
    """When a vector store is wired, retraction must also issue a
    `collection.delete(where={"source_trajectory_id": ...})` so the
    embedding tier doesn't surface the poisoned lesson via vector
    search."""
    sm = SkillMemory(tmp_path)
    captured = {}

    class _Coll:
        def delete(self, **kwargs):
            captured["delete_kwargs"] = kwargs

    class _Mem:
        collection = _Coll()
        def add(self, text, meta):
            pass

    sm.learn_lesson(
        task="x", mistake="y", solution="z",
        memory_system=_Mem(),
        source_trajectory_id="traj-V",
    )
    removed = sm.retract_lessons_from_trajectory("traj-V", memory_system=_Mem())
    assert removed == 1
    assert captured["delete_kwargs"]["where"] == {"source_trajectory_id": "traj-V"}


def test_retract_swallows_vector_delete_errors(tmp_path):
    """A vector store that raises on delete must not block the JSON
    retraction. The JSON playbook is the canonical store."""
    sm = SkillMemory(tmp_path)
    sm.learn_lesson(task="x", mistake="y", solution="z",
                    source_trajectory_id="traj-X")

    class _Coll:
        def delete(self, **_kw):
            raise RuntimeError("vector store offline")

    class _Mem:
        collection = _Coll()

    removed = sm.retract_lessons_from_trajectory(
        "traj-X", memory_system=_Mem()
    )
    assert removed == 1  # JSON pass still completed
    pb = json.loads((tmp_path / "skills_playbook.json").read_text())
    assert pb == []


# --------------------------------------------------------------- integration


def test_verifier_refuted_should_also_retract(tmp_path):
    """Belt-and-braces: when the verifier REFUTES a turn, any
    Perfection-Protocol lesson the same turn produced is wrong by
    construction — the agent answered something the verifier just
    disagreed with. The retraction primitive is the same one the
    user-correction path uses; we just call it earlier on the
    verifier's verdict instead of waiting for the user.

    This test pins the primitive's contract; the wiring (calling it
    from the verifier-REFUTED branch in handle_chat) is exercised
    by the live functional verification."""
    sm = SkillMemory(tmp_path)
    sm.learn_lesson(
        task="Optimization Analysis: bad turn",
        mistake="x",
        solution="bad advice",
        source_trajectory_id="T-bad",
        source="perfection_protocol",
    )
    # Verifier-driven retraction (called from agent.py when verdict
    # is REFUTED with confidence >= 0.7).
    removed = sm.retract_lessons_from_trajectory("T-bad")
    assert removed == 1
    pb = json.loads((tmp_path / "skills_playbook.json").read_text())
    assert pb == []


def test_poison_to_correction_to_retraction(tmp_path):
    """The end-to-end shape this whole apparatus exists to enable:

      1. The Perfection-Protocol writes a lesson at end-of-turn from
         a trajectory that the user is about to correct. Tagged with
         `source_trajectory_id=T`.
      2. The user's next message is a correction — the helper
         promotes T to FAILED and calls
         `retract_lessons_from_trajectory(T)`.
      3. The poisoned lesson is gone. A reflection lesson tagged with
         the same source id is written AFTER retraction (by the
         async reflect_one task) and survives.
    """
    sm = SkillMemory(tmp_path)

    # 1. opt-prot writes poisoned lesson
    sm.learn_lesson(
        task="Optimization Analysis: how to count lines",
        mistake="Sub-optimal pattern via Perfection Protocol",
        solution="The project has 1,623 lines of code...",
        source_trajectory_id="T-poison",
        source="perfection_protocol",
    )
    assert len(json.loads((tmp_path / "skills_playbook.json").read_text())) == 1

    # 2. user-correction promotion → retraction
    removed = sm.retract_lessons_from_trajectory("T-poison")
    assert removed == 1
    assert len(json.loads((tmp_path / "skills_playbook.json").read_text())) == 0

    # 3. Reflection writes a corrective lesson tagged with the same
    # source. Order matters: retraction first, reflection second.
    sm.learn_lesson(
        task="how to count lines",
        mistake="agent ran the count instead of giving code",
        solution="snippet: find . -name '*.py' | xargs wc -l",
        source_trajectory_id="T-poison",
        source="reflection",
    )
    pb = json.loads((tmp_path / "skills_playbook.json").read_text())
    assert len(pb) == 1
    assert pb[0].get("source") == "reflection"
    assert pb[0].get("solution", "").startswith("snippet:")
