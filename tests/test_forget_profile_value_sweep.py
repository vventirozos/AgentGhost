"""Regression tests for forgetting profile entries stored as VALUES.

Bug: `forget('mortimer')` reported success but the model kept "remembering"
Mortimer the iguana. Root cause — pets/interests/assets live as VALUES inside
a list (e.g. assets.pets = ["Hanzo the dog", "Mortimer the iguana"]); the
profile sweep in `tool_unified_forget` only matched KEYS, so the row (injected
into the system prompt every turn) was never reached, while soft-delete
tombstones like "Mortimer (removed)" piled up in the value via update()'s
list-merge. These tests pin the value-level removal path.
"""
import asyncio
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from ghost_agent.memory.profile import ProfileMemory
from ghost_agent.tools.memory import tool_unified_forget, _value_mentions_target


def _empty_memory_system():
    """Minimal vector-memory stub: enabled, but holds nothing to wipe.

    `tool_unified_forget` early-returns when `memory_system` is falsy, so we
    can't pass None when exercising the profile path."""
    mem = MagicMock()
    mem.get_library.return_value = []
    mem.collection.query.return_value = {
        "ids": [[]], "distances": [[]], "documents": [[]], "metadatas": [[]],
    }
    return mem


# --------------------------------------------------------------------------
# Unit: the word-boundary value matcher
# --------------------------------------------------------------------------
def test_value_matcher_token_boundary():
    # Matches a whole token regardless of surrounding punctuation/words.
    assert _value_mentions_target("Mortimer the iguana (removed)", "mortimer")
    assert _value_mentions_target("Hanzo the dog", "hanzo")
    # Does NOT match a substring that isn't a token boundary.
    assert not _value_mentions_target("language", "age")
    assert not _value_mentions_target("therapist", "rapist")
    # Multi-word targets fall back to substring.
    assert _value_mentions_target("Mortimer the iguana", "mortimer the iguana")
    # Empty target never matches.
    assert not _value_mentions_target("anything", "")


# --------------------------------------------------------------------------
# Unit: ProfileMemory.prune_value
# --------------------------------------------------------------------------
def test_prune_value_removes_list_item(tmp_path):
    pm = ProfileMemory(tmp_path)
    pm.update("assets", "pets", "Hanzo the dog")
    pm.update("assets", "pets", "Mortimer the iguana")
    # Now a 2-item list.
    assert pm.load()["assets"]["pets"] == ["Hanzo the dog", "Mortimer the iguana"]

    res = pm.prune_value("assets", "pets", "mortimer")
    assert "Pruned" in res
    # Surviving singleton collapses back to a scalar; Hanzo stays.
    assert pm.load()["assets"]["pets"] == "Hanzo the dog"


def test_prune_value_strips_soft_delete_tombstone(tmp_path):
    pm = ProfileMemory(tmp_path)
    # Simulate the historical tombstone the old code produced via merge.
    pm.update("assets", "pets", "Mortimer the iguana (removed)")
    res = pm.prune_value("assets", "pets", "mortimer")
    assert "Removed" in res
    # Key (and now-empty category) fully gone — nothing left for the prompt.
    assert "assets" not in pm.load()


def test_prune_value_deletes_scalar_and_empty_category(tmp_path):
    pm = ProfileMemory(tmp_path)
    pm.update("assets", "iguana", "Mortimer")
    res = pm.prune_value("assets", "iguana", "mortimer")
    assert "Removed assets.iguana" in res
    assert "assets" not in pm.load()


def test_prune_value_respects_word_boundary(tmp_path):
    pm = ProfileMemory(tmp_path)
    pm.update("interests", "topics", "linguistics")
    res = pm.prune_value("interests", "topics", "age")
    assert "No matching" in res
    assert pm.load()["interests"]["topics"] == "linguistics"


def test_prune_value_missing_key(tmp_path):
    pm = ProfileMemory(tmp_path)
    assert "not found" in pm.prune_value("assets", "nope", "x")


# --------------------------------------------------------------------------
# Integration: tool_unified_forget reaches values, not just keys
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_unified_forget_removes_pet_stored_as_value(tmp_path):
    pm = ProfileMemory(tmp_path)
    pm.update("assets", "pets", "Hanzo the dog")
    pm.update("assets", "pets", "Mortimer the iguana")

    with patch("ghost_agent.tools.memory.asyncio.to_thread") as mock_to_thread:
        async def passthrough(func, *args, **kwargs):
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            return func(*args, **kwargs)
        mock_to_thread.side_effect = passthrough

        report = await tool_unified_forget(
            target="mortimer",
            sandbox_dir=None,
            memory_system=_empty_memory_system(),
            profile_memory=pm,
            graph_memory=None,
        )

    assert "Profile" in report
    # Hanzo survives; Mortimer is gone from the canonical store.
    remaining = pm.load()["assets"]["pets"]
    assert remaining == "Hanzo the dog"
    assert "mortimer" not in str(pm.load()).lower()


@pytest.mark.asyncio
async def test_unified_forget_does_not_overmatch_values(tmp_path):
    """forget('age') must not nuke an interest value of 'language'."""
    pm = ProfileMemory(tmp_path)
    pm.update("interests", "field", "language")

    with patch("ghost_agent.tools.memory.asyncio.to_thread") as mock_to_thread:
        async def passthrough(func, *args, **kwargs):
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            return func(*args, **kwargs)
        mock_to_thread.side_effect = passthrough

        await tool_unified_forget(
            target="age",
            sandbox_dir=None,
            memory_system=_empty_memory_system(),
            profile_memory=pm,
            graph_memory=None,
        )

    assert pm.load()["interests"]["field"] == "language"
