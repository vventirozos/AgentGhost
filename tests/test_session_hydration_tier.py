"""SessionStore as the raw-conversation hydration tier (2026-07-14).

Durable sessions (2026-07-11) were replay-only: the chat route merged them
into requests, but NO retrieval path could reach them — the lowest-
abstraction memory layer was invisible to recall. `SessionStore.
search_messages` adds a bounded keyword search (most-recent 50 sessions,
mtime-cached parses, ≥2-term floor for multi-term queries) and `MemoryBus`
gains a fifth fetcher + "session" RRF source feeding the standard fusion /
budget / relevance pipeline under a PAST CONVERSATIONS header.
"""

import pytest
from unittest.mock import MagicMock

from ghost_agent.core.bus import MemoryBus
from ghost_agent.core.sessions import SessionStore


@pytest.fixture
def store(tmp_path):
    s = SessionStore(tmp_path / "sessions")
    s.append_turn(
        "sessdeploy",
        [{"role": "user", "content": "let's plan the kubernetes deployment for the ghost node"}],
        "We agreed the kubernetes deployment happens on friday after the backup.",
    )
    s.append_turn(
        "sesschess",
        [{"role": "user", "content": "coach me through an alekhine defense line"}],
        "Sure — the alekhine defense invites the pawn chase.",
    )
    return s


class TestSearchMessages:
    def test_finds_matching_conversation(self, store):
        hits = store.search_messages("kubernetes deployment schedule")
        assert hits
        assert hits[0]["session_id"] == "sessdeploy"
        assert "kubernetes deployment" in hits[0]["text"]
        assert hits[0]["role"] in ("user", "assistant")

    def test_multi_term_floor_rejects_single_shared_word(self, store):
        # "deployment" alone matches, but a two-term query needs >=2 hits —
        # one shared word across topics is noise, not conversation recall.
        hits = store.search_messages("deployment sailboat")
        assert hits == []

    def test_stopwords_and_short_terms_ignored(self, store):
        assert store.search_messages("what about this that") == []
        assert store.search_messages("") == []

    def test_best_match_first(self, store):
        store.append_turn(
            "sessdeploy2",
            [{"role": "user", "content": "kubernetes deployment friday backup node checklist"}],
            "ack",
        )
        hits = store.search_messages("kubernetes deployment friday backup")
        assert hits[0]["session_id"] == "sessdeploy2"
        assert hits[0]["score"] >= hits[-1]["score"]

    def test_failsafe_on_absent_dir(self, tmp_path):
        s = SessionStore(tmp_path / "nonexistent")
        assert s.search_messages("kubernetes deployment") == []

    def test_mtime_cache_avoids_reparse(self, store, monkeypatch):
        store.search_messages("kubernetes deployment")  # warm the cache
        calls = []
        real_get = store.get

        def counting_get(sid):
            calls.append(sid)
            return real_get(sid)

        monkeypatch.setattr(store, "get", counting_get)
        store._list_memo = (0.0, [])  # force a fresh summaries scan
        hits = store.search_messages("kubernetes deployment")
        assert hits  # cache serves parsed messages...
        assert calls == []  # ...without re-reading unchanged session files


class TestBusSessionTier:
    async def test_fetch_session_formats_items(self, store):
        bus = MemoryBus(session_store=store)
        items = await bus._fetch_session("kubernetes deployment schedule")
        assert items
        assert items[0]["source"] == "session"
        assert items[0]["text"].startswith("[")  # title prefix
        assert "kubernetes deployment" in items[0]["text"]

    async def test_hydrate_includes_past_conversations(self, store):
        bus = MemoryBus(session_store=store)
        out = await bus.hydrate_context("kubernetes deployment schedule")
        assert "### PAST CONVERSATIONS" in out
        assert "kubernetes deployment" in out

    async def test_no_session_store_is_silent(self):
        bus = MemoryBus()
        assert await bus._fetch_session("anything at all") == []
        assert await bus.hydrate_context("kubernetes deployment schedule") == ""

    async def test_store_without_search_method_is_silent(self):
        legacy = MagicMock(spec=[])  # no search_messages attribute
        bus = MemoryBus(session_store=legacy)
        assert await bus._fetch_session("kubernetes deployment") == []

    def test_intent_weights_cover_session(self):
        for intent, weights in MemoryBus._INTENT_WEIGHTS.items():
            assert "session" in weights, f"missing session weight for {intent}"
