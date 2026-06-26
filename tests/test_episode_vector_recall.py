"""Tests for episodic vector ingestion + semantic recall (feature 1C).

Previously ``EpisodicMemory.record_episode`` never wrote episodes into the
vector store, so ``search_similar``'s semantic path found no
``type=="episode"`` hits and silently fell back to substring matching. These
tests pin the now-wired behaviour: episodes are ingested with the right
metadata, recalled semantically, and their vector entries are removed when
the capacity cap evicts them.
"""

import hashlib

import pytest

from ghost_agent.memory.episodes import EpisodicMemory


class FakeVector:
    """Minimal stand-in for VectorMemory: dedups on md5(text) like the real
    ``add``, supports ``forget_episode`` by metadata, and ``search_advanced``
    returns every stored doc as a hit (the relevance ranking is the unit
    under test elsewhere — here we only verify the episode plumbing)."""

    def __init__(self):
        self.docs = {}  # id -> {"text", "metadata"}

    def add(self, text, meta=None):
        if len(text) < 5:
            return
        mem_id = hashlib.md5(text.encode("utf-8")).hexdigest()
        if mem_id in self.docs:
            return  # dedup
        self.docs[mem_id] = {"text": text, "metadata": meta or {"type": "auto"}}

    def forget_episode(self, episode_id):
        eid = int(episode_id)
        self.docs = {
            k: v for k, v in self.docs.items()
            if (v["metadata"] or {}).get("episode_id") != eid
        }

    def search_advanced(self, query, limit=5):
        return [
            {"id": k, "text": v["text"], "metadata": v["metadata"], "score": 0.1}
            for k, v in self.docs.items()
        ][:limit]


def test_record_ingests_episode_into_vector(tmp_path):
    em = EpisodicMemory(tmp_path)
    vec = FakeVector()
    ep_id = em.record_episode(
        trigger="deploy the staging service over tor",
        outcome="succeeded after fixing the proxy",
        success=True,
        lesson="set TOR_PROXY before booting",
        vector_memory=vec,
    )
    # Exactly one episode-typed vector entry, tagged with this episode id.
    episode_docs = [d for d in vec.docs.values()
                    if d["metadata"].get("type") == "episode"]
    assert len(episode_docs) == 1
    assert episode_docs[0]["metadata"]["episode_id"] == ep_id
    # Trigger and lesson are both in the indexed text.
    assert "staging service" in episode_docs[0]["text"]
    assert "TOR_PROXY" in episode_docs[0]["text"]


def test_search_similar_uses_vector_path(tmp_path):
    em = EpisodicMemory(tmp_path)
    vec = FakeVector()
    ep_id = em.record_episode(
        trigger="train a GRU on the pet dataset",
        outcome="loss converged",
        success=True,
        vector_memory=vec,
    )
    # A query that shares NO >3-char substring tokens with the trigger — the
    # substring fallback would miss it; the vector path maps the hit back.
    results = em.search_similar("recurrent network sequence model",
                                limit=5, vector_memory=vec)
    assert results, "expected the vector path to surface the episode"
    assert results[0]["id"] == ep_id
    assert results[0]["relevance_score"] == 1.0


def test_no_vector_memory_is_a_noop(tmp_path):
    """Recording without a vector store still works (no ingestion)."""
    em = EpisodicMemory(tmp_path)
    ep_id = em.record_episode(trigger="do a thing", success=True)
    assert ep_id > 0
    assert em.count() == 1


def test_eviction_forgets_vector_entries(tmp_path, monkeypatch):
    """When the capacity cap evicts old episodes, their vector entries are
    removed so the index doesn't accumulate orphans."""
    monkeypatch.setattr(EpisodicMemory, "MAX_EPISODES", 3)
    em = EpisodicMemory(tmp_path)
    vec = FakeVector()
    ids = []
    for i in range(5):
        ids.append(em.record_episode(
            trigger=f"distinct episode trigger number {i}",
            outcome="done", success=True, vector_memory=vec,
        ))
    # Table is capped at 3.
    assert em.count() == 3
    # The vector store holds only the surviving episodes — evicted ids gone.
    live_eids = {d["metadata"]["episode_id"] for d in vec.docs.values()
                 if d["metadata"].get("type") == "episode"}
    assert len(live_eids) == 3
    # The two oldest were evicted from both stores.
    assert ids[0] not in live_eids
    assert ids[1] not in live_eids


def test_short_trigger_not_indexed(tmp_path):
    em = EpisodicMemory(tmp_path)
    vec = FakeVector()
    em.record_episode(trigger="hi", success=True, vector_memory=vec)
    assert not [d for d in vec.docs.values()
                if d["metadata"].get("type") == "episode"]
