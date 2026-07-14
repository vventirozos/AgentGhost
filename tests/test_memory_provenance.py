"""Provenance links on abstractions (NapMem-inspired, 2026-07-14).

Dream syntheses and episode-derived skills previously kept no pointer to the
evidence they were generalized from — a stale or hallucinated abstraction was
unfalsifiable after the fact (dream even DELETES the merged source fragments).
Now:

- syntheses carry ``{"provenance": [{"id", "excerpt"}, ...]}`` metadata (the
  excerpt is the surviving evidence, captured before the source delete);
- lessons carry ``source_refs`` (e.g. ``["ep:12", "ep:15"]``), unioned on
  dedup-merge, mirrored onto the vector twin as a comma-joined string;
- ``tool_recall`` surfaces both as EVIDENCE lines so the model (and the
  operator) can drill from an abstraction back to its sources.
"""

import json

import pytest
from unittest.mock import AsyncMock, MagicMock

from ghost_agent.core.dream import Dreamer
from ghost_agent.memory.episodes import EpisodicMemory
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
                        source_refs=["ep:1", "ep:2"])
        entry = sm._load_playbook()[0]
        assert entry["source_refs"] == ["ep:1", "ep:2"]

    def test_duplicate_merge_unions_refs(self, tmp_path):
        sm = SkillMemory(tmp_path)
        sm.learn_lesson("Always verify paths before writing.", "none",
                        "Always verify paths before writing.",
                        source_refs=["ep:1"])
        sm.learn_lesson("Always verify paths before writing.", "none",
                        "Always verify paths before writing.",
                        source_refs=["ep:2", "ep:1"])
        entry = sm._load_playbook()[0]
        assert entry["source_refs"] == ["ep:1", "ep:2"]
        assert entry["frequency"] == 2

    def test_vector_twin_carries_refs(self, tmp_path):
        sm = SkillMemory(tmp_path)
        vm = MagicMock()
        vm.search_advanced.return_value = []  # no vector duplicate
        sm.learn_lesson("Always verify paths before writing.", "none",
                        "Always verify paths before writing.",
                        memory_system=vm,
                        source_refs=["ep:1", "ep:2"])
        meta = vm.add.call_args.args[1]
        assert meta["type"] == "skill"
        assert meta["source_refs"] == "ep:1,ep:2"


# ------------------------------------------------------- dream write paths


class TestDreamProvenance:
    async def test_episode_strategy_lessons_carry_ep_refs(self, tmp_path):
        epi = EpisodicMemory(tmp_path)
        ep_ids = [
            epi.record_episode(trigger=f"deploy v{i} failed", outcome="EACCES",
                               success=False, cluster_id="deploy")
            for i in range(3)
        ]
        ctx = MagicMock()
        ctx.episodic_memory = epi
        ctx.llm_client.chat_completion = AsyncMock(return_value={
            "choices": [{"message": {"content": json.dumps({
                "strategies": ["Always verify the publish path before retrying a deploy."],
            })}}]
        })

        await Dreamer(ctx)._consolidate_episodes("test-model")

        kwargs = ctx.skill_memory.learn_lesson.call_args.kwargs
        # get_unconsolidated drains newest-first; order is irrelevant here.
        assert sorted(kwargs["source_refs"]) == sorted(f"ep:{i}" for i in ep_ids)

    async def test_synthesis_metadata_carries_provenance(self):
        ctx = MagicMock()
        ctx.memory_system.collection.get.return_value = {
            "ids": ["id1", "id2", "id3"],
            "documents": ["A" * 200, "B" * 200, "C" * 100],
            "metadatas": [{"type": "auto"}] * 3,
            "embeddings": [[0.1]] * 3,
        }
        ctx.llm_client.chat_completion = AsyncMock(return_value={
            "choices": [{"message": {"content": json.dumps({
                "consolidations": [{
                    "synthesis": "Compressed summary",
                    "merged_ids": ["ID:id1", "ID:id2"],
                }],
                "heuristics": [],
            })}}]
        })

        result = await Dreamer(ctx).dream()

        assert "Dream Complete" in result
        syn_calls = [c for c in ctx.memory_system.add.call_args_list
                     if len(c.args) > 1 and isinstance(c.args[1], dict)
                     and c.args[1].get("type") == "synthesis"]
        assert syn_calls, "synthesis was not stored"
        prov = json.loads(syn_calls[0].args[1]["provenance"])
        assert [p["id"] for p in prov] == ["id1", "id2"]
        assert prov[0]["excerpt"] == "A" * 100  # evidence survives the source delete


# ------------------------------------------------------------ recall surface


class TestRecallSurfacesEvidence:
    async def test_recall_shows_provenance_and_refs(self):
        from ghost_agent.tools.memory import tool_recall
        vm = MagicMock()
        vm.search_advanced.return_value = [
            {
                "score": 0.4,
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
                "score": 0.5,
                "text": "Always verify the publish path before retrying.",
                "metadata": {"type": "skill", "source_refs": "ep:1,ep:2"},
            },
        ]

        out = await tool_recall(query="deploy failures", memory_system=vm)

        assert "EVIDENCE (synthesized from 2 fragments)" in out
        assert "deploy v1 failed EACCES" in out
        assert "EVIDENCE REFS: ep:1,ep:2" in out
