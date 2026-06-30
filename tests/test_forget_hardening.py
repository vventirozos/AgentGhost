"""Hardening for entity forgetting — stop tombstones surviving / regenerating.

Symptom: `forget mortimer` / `forget iguana` reported success, but
"you previously had an iguana that was removed" kept coming back. Causes:
  1. Consolidation re-stored the removal statement as a positive fact + graph
     triplet every time the user asked about pets (self-perpetuating tombstone).
  2. The forget vector sweep relied on a distance threshold that missed facts
     literally naming the entity.
  3. The tombstone lived under a SYNONYM (iguana) the mortimer-scoped sweep
     never reached.

These tests pin the three fixes: removal-fact suppression, the literal-mention
vector override, and graph-neighbour entity expansion.
"""
import asyncio
import pytest
from unittest.mock import patch

from ghost_agent.utils.helpers import (
    is_removal_or_negation_text,
    is_removal_triplet,
)
from ghost_agent.memory.profile import ProfileMemory
from ghost_agent.tools.memory import tool_unified_forget


# Resolve the graph store class by name (avoid hard-coding the class identifier).
import ghost_agent.memory.graph as _graphmod
GraphCls = getattr(_graphmod, "KnowledgeGraph", None) or getattr(_graphmod, "GraphMemory", None)


# --------------------------------------------------------------------------
# Fakes for the vector store
# --------------------------------------------------------------------------
class FakeCollection:
    def __init__(self, docs):
        # docs: list of (id, text, meta)
        self.docs = {d[0]: (d[1], d[2]) for d in docs}

    def query(self, query_texts=None, n_results=10):
        ids = list(self.docs.keys())[:n_results]
        # Deliberately HIGH distance so only the literal-mention override can
        # trigger a delete — proves we're not just riding the threshold.
        return {
            "ids": [ids],
            "distances": [[1.5] * len(ids)],
            "documents": [[self.docs[i][0] for i in ids]],
            "metadatas": [[self.docs[i][1] for i in ids]],
        }

    def delete(self, ids=None):
        for i in (ids or []):
            self.docs.pop(i, None)

    def get(self, ids=None):
        return {"ids": [i for i in (ids or []) if i in self.docs]}


class FakeMemSys:
    def __init__(self, docs):
        self.collection = FakeCollection(docs)

    def get_library(self):
        return []

    def delete_document_by_name(self, name):
        pass


def _passthrough_to_thread():
    async def passthrough(func, *args, **kwargs):
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        return func(*args, **kwargs)
    return passthrough


# --------------------------------------------------------------------------
# Removal / negation detection
# --------------------------------------------------------------------------
def test_removal_text_detection():
    assert is_removal_or_negation_text("User previously had an iguana that was removed")
    assert is_removal_or_negation_text("The user no longer owns a car")
    assert is_removal_or_negation_text("user does not have a cat")
    # Positive facts are NOT flagged.
    assert not is_removal_or_negation_text("User owns a dog named Hanzo")
    assert not is_removal_or_negation_text("User lives in Athens")
    assert not is_removal_or_negation_text("")


def test_removal_triplet_detection():
    assert is_removal_triplet({"subject": "user", "predicate": "PREVIOUSLY_OWNED", "object": "iguana"})
    assert is_removal_triplet({"subject": "user", "predicate": "NO_LONGER_HAS", "object": "iguana"})
    assert is_removal_triplet({"subject": "user", "predicate": "HAS_PET", "object": "iguana (removed)"})
    # A clean ownership edge survives.
    assert not is_removal_triplet({"subject": "user", "predicate": "HAS_PET", "object": "hanzo"})
    assert not is_removal_triplet("not a dict")


# --------------------------------------------------------------------------
# Graph: connected-entity expansion
# --------------------------------------------------------------------------
def test_graph_connected_entities(tmp_path):
    g = GraphCls(tmp_path)
    g.add_triplets([
        {"subject": "user", "predicate": "HAS_PET", "object": "mortimer"},
        {"subject": "mortimer", "predicate": "IS_A", "object": "iguana"},
    ])
    related = g.get_connected_entities("mortimer")
    assert "iguana" in related
    # Hub node 'user' is filtered; the target itself is excluded.
    assert "user" not in related
    assert "mortimer" not in related


# --------------------------------------------------------------------------
# Vector sweep: literal-mention override
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_forget_vector_literal_override(tmp_path):
    mem = FakeMemSys([
        ("m1", "User previously had an iguana that was removed", {"type": "identity"}),
        ("d1", "User owns a dog named Hanzo", {"type": "auto"}),
    ])
    with patch("ghost_agent.tools.memory.asyncio.to_thread") as mt:
        mt.side_effect = _passthrough_to_thread()
        await tool_unified_forget(
            target="iguana", sandbox_dir=None, memory_system=mem,
            profile_memory=None, graph_memory=None,
        )
    # Literal mention of 'iguana' deleted despite the high distance...
    assert "m1" not in mem.collection.docs
    # ...and the unrelated dog fact (no 'iguana' token) is untouched.
    assert "d1" in mem.collection.docs


# --------------------------------------------------------------------------
# End-to-end: forget('mortimer') reaches the 'iguana' alias
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_forget_expands_to_alias(tmp_path):
    g = GraphCls(tmp_path)
    g.add_triplets([
        {"subject": "user", "predicate": "HAS_PET", "object": "mortimer"},
        {"subject": "mortimer", "predicate": "IS_A", "object": "iguana"},
        {"subject": "user", "predicate": "PREVIOUSLY_OWNED", "object": "iguana"},
    ])
    pm = ProfileMemory(tmp_path)
    pm.update("assets", "pets", "Hanzo the dog")
    pm.update("assets", "pets", "Mortimer the iguana")
    mem = FakeMemSys([
        ("t1", "User previously had an iguana that was removed", {"type": "identity"}),
        ("k1", "User owns a dog named Hanzo", {"type": "auto"}),
    ])

    with patch("ghost_agent.tools.memory.asyncio.to_thread") as mt:
        mt.side_effect = _passthrough_to_thread()
        report = await tool_unified_forget(
            target="mortimer", sandbox_dir=None, memory_system=mem,
            profile_memory=pm, graph_memory=g,
        )

    # Profile: Mortimer pruned from the list, Hanzo survives.
    assert pm.load()["assets"]["pets"] == "Hanzo the dog"
    # Vector: the iguana tombstone (an ALIAS of mortimer) is gone...
    assert "t1" not in mem.collection.docs
    # ...the dog fact stays.
    assert "k1" in mem.collection.docs
    # Graph: the orphan 'iguana' edge is severed via expansion.
    assert g.get_connected_entities("iguana") == [] or "iguana" not in str(g.get_recent_triplets()).lower()
    assert "iguana" in report.lower()
