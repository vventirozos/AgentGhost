"""news_headlines went invisible; the prefix cache was primed twice.

Both from the 2026-07-27 evening log.

**Skill invisibility.** The registry held 3 active skills but the vector
store held 1 embedding, so every request logged
`injected 1: format_results_to_csv` no matter the query. A user asking for
`news_headlines` got a model that could not see the tool: it called
`manage_skills` twice, tried to shell out to an inline script (blocked at
219 chars), tripped the loop-breaker, and had its final answer eaten by the
stream scrub. The index could only ever LOSE entries — `save_skill` embeds
only when the content hash CHANGES, so nothing rebuilt a lost one.

**Double prefill.** The boot warmup primes the request head with
``query=""`` (→ advertise ALL skills) while every live request routes
(→ a subset), so the ~22k-token warmup never matched a real request and
each one re-prefilled from scratch.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ghost_agent.tools.acquired_skills import AcquiredSkillManager
from ghost_agent.tools import registry as reg


def _mgr(tmp, skills, embedded_names):
    """Manager over a temp registry plus a fake vector store."""
    d = Path(tmp) / "acquired_skills"
    d.mkdir(parents=True, exist_ok=True)
    (d / "skills_registry.json").write_text(json.dumps(skills))
    mem = MagicMock()
    mem.collection.get.return_value = {
        "ids": [f"id{i}" for i in range(len(embedded_names))],
        "metadatas": [{"type": "acquired_skill", "name": n}
                      for n in embedded_names],
        "documents": [None] * len(embedded_names),
    }
    m = AcquiredSkillManager(Path(tmp), mem)
    return m, mem


_SKILLS = {
    "news_headlines": {"name": "news_headlines", "description": "Fetch news.",
                       "status": "active", "content_hash": "a"},
    "generate_password": {"name": "generate_password", "description": "Make a password.",
                          "status": "active", "content_hash": "b"},
    "format_results_to_csv": {"name": "format_results_to_csv", "description": "To CSV.",
                              "status": "active", "content_hash": "c"},
}


# ──────────────────────────────────────────────────────────────────────
# Backfill — the index must be able to HEAL, not only shrink
# ──────────────────────────────────────────────────────────────────────

class TestEmbeddingBackfill:
    def test_missing_embeddings_are_rebuilt(self):
        """The live shape: 3 active skills, 1 embedding."""
        with tempfile.TemporaryDirectory() as t:
            m, mem = _mgr(t, _SKILLS, ["format_results_to_csv"])
            assert m.backfill_missing_skill_embeddings() == 2
            added = sorted(c.args[1]["name"] for c in mem.add.call_args_list)
            assert added == ["generate_password", "news_headlines"]

    def test_embeds_the_description_not_the_name(self):
        with tempfile.TemporaryDirectory() as t:
            m, mem = _mgr(t, _SKILLS, [])
            m.backfill_missing_skill_embeddings()
            docs = [c.args[0] for c in mem.add.call_args_list]
            assert "Fetch news." in docs

    def test_noop_when_everything_is_indexed(self):
        with tempfile.TemporaryDirectory() as t:
            m, mem = _mgr(t, _SKILLS, list(_SKILLS))
            assert m.backfill_missing_skill_embeddings() == 0
            mem.add.assert_not_called()

    def test_inactive_skills_are_not_embedded(self):
        skills = dict(_SKILLS)
        skills["news_headlines"] = dict(skills["news_headlines"], status="degraded")
        with tempfile.TemporaryDirectory() as t:
            m, mem = _mgr(t, skills, [])
            m.backfill_missing_skill_embeddings()
            names = {c.args[1]["name"] for c in mem.add.call_args_list}
            assert "news_headlines" not in names

    def test_never_raises_without_a_memory_system(self):
        with tempfile.TemporaryDirectory() as t:
            m = AcquiredSkillManager(Path(t), None)
            assert m.backfill_missing_skill_embeddings() == 0


# ──────────────────────────────────────────────────────────────────────
# Purge fail-safe — an unreadable registry must not wipe the index
# ──────────────────────────────────────────────────────────────────────

class TestPurgeFailSafe:
    def test_empty_registry_does_not_wipe_every_embedding(self):
        """Without the guard, one transient/mid-write registry read
        classifies EVERY embedding as an orphan and deletes the whole
        routing index — unrecoverable, since nothing re-embeds an
        unchanged skill."""
        with tempfile.TemporaryDirectory() as t:
            m, mem = _mgr(t, {}, ["news_headlines", "generate_password"])
            assert m.purge_orphaned_skill_embeddings() == 0
            mem.collection.delete.assert_not_called()

    def test_genuine_orphans_are_still_purged(self):
        with tempfile.TemporaryDirectory() as t:
            m, mem = _mgr(t, _SKILLS, list(_SKILLS) + ["deleted_skill"])
            assert m.purge_orphaned_skill_embeddings() == 1
            assert mem.collection.delete.called


# ──────────────────────────────────────────────────────────────────────
# Prefix stability — the advertised tool set must not vary per query
# ──────────────────────────────────────────────────────────────────────

class TestAdvertisedSetIsPrefixStable:
    def test_threshold_exists_and_is_meaningful(self):
        assert reg._SKILL_ROUTING_MIN_SKILLS >= 3

    def test_routing_is_skipped_below_the_threshold(self):
        """Below the threshold the tool block must be identical for ANY
        query — that is what lets the boot warmup's prefill match a live
        request instead of both paying the ~22k-token head."""
        src = (Path(__file__).resolve().parents[1] / "src" / "ghost_agent"
               / "tools" / "registry.py").read_text()
        assert "_active_count > _SKILL_ROUTING_MIN_SKILLS" in src
        # The old unconditional form must be gone.
        assert "target_skill_names = None\n            if query:\n" not in src

    def test_warmup_uses_the_neutral_query(self):
        """The warmup passes "" and therefore advertises everything; with
        routing disabled below the threshold a live request now resolves to
        the same set."""
        src = (Path(__file__).resolve().parents[1] / "src" / "ghost_agent"
               / "core" / "agent.py").read_text()
        assert 'get_active_tool_defs("")' in src
