"""Provenance links on abstractions (NapMem-inspired, 2026-07-14).

Abstractions previously kept no pointer to the evidence they were generalized
from — a stale or hallucinated abstraction was unfalsifiable after the fact.
Now (§4NM: the dream synthesis / episode-consolidation writers are gone;
lessons come only from ``learn_skill`` and the operator's repairs):

- syntheses carry ``{"provenance": [{"id", "excerpt"}, ...]}`` metadata;
- lessons carry ``source_refs`` (e.g. ``["ep:12", "ep:15"]``), unioned on
  dedup-merge, mirrored onto the vector twin as a comma-joined string;
- ``tool_recall`` surfaces both as EVIDENCE lines so the model (and the
  operator) can drill from an abstraction back to its sources.
"""

import json

import pytest
from unittest.mock import MagicMock

from ghost_agent.memory.skills import SkillMemory, build_lesson


# ----------------------------------------------------------------- lessons


class TestLessonSourceRefs:
    def test_build_lesson_stores_refs_capped(self):
        lesson = build_lesson(task="T", source_refs=[f"ep:{i}" for i in range(30)])
        assert lesson["source_refs"][0] == "ep:0"
        assert len(lesson["source_refs"]) == 20

    def test_learn_lesson_persists_refs(self, tmp_path):
        sm = SkillMemory(tmp_path)
        sm.learn_lesson("Always verify paths before writing.", "none",
                        "Always verify paths before writing.",
                        source="learn_skill", source_refs=["ep:1", "ep:2"])
        entry = sm._load_playbook()[0]
        assert entry["source_refs"] == ["ep:1", "ep:2"]

    def test_duplicate_merge_unions_refs(self, tmp_path):
        sm = SkillMemory(tmp_path)
        sm.learn_lesson("Always verify paths before writing.", "none",
                        "Always verify paths before writing.",
                        source="learn_skill", source_refs=["ep:1"])
        sm.learn_lesson("Always verify paths before writing.", "none",
                        "Always verify paths before writing.",
                        source="learn_skill", source_refs=["ep:2", "ep:1"])
        entry = sm._load_playbook()[0]
        assert entry["source_refs"] == ["ep:1", "ep:2"]
        assert entry["frequency"] == 2

    def test_vector_twin_carries_refs(self, tmp_path):
        sm = SkillMemory(tmp_path)
        vm = MagicMock()
        vm.search_advanced.return_value = []  # no vector duplicate
        sm.learn_lesson("Always verify paths before writing.", "none",
                        "Always verify paths before writing.",
                        memory_system=vm, source="learn_skill",
                        source_refs=["ep:1", "ep:2"])
        meta = vm.add.call_args.args[1]
        assert meta["type"] == "skill"
        assert meta["source_refs"] == "ep:1,ep:2"


# ------------------------------------------------------------ recall surface


class TestRecallSurfacesEvidence:
    async def test_recall_shows_provenance_and_refs(self):
        from ghost_agent.tools.memory import tool_recall
        vm = MagicMock()
        vm.search_advanced.return_value = [
            {
                "score": 0.2,
                "text": "The operator's deploys fail on permissions.",
                "metadata": {
                    "type": "synthesis",
                    "provenance": json.dumps([
                        {"id": "id1", "excerpt": "deploy v1 failed EACCES"},
                        {"id": "id2", "excerpt": "deploy v2 failed EACCES"},
                    ]),
                },
            },
            {
                "score": 0.25,
                "text": "Always verify the publish path before retrying.",
                "metadata": {"type": "skill", "source_refs": "ep:1,ep:2"},
            },
        ]

        out = await tool_recall(query="deploy failures", memory_system=vm)

        assert "EVIDENCE (synthesized from 2 fragments)" in out
        assert "deploy v1 failed EACCES" in out
        assert "EVIDENCE REFS: ep:1,ep:2" in out
