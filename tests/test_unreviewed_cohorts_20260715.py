"""Regression tests for the 2026-07-15 never-reviewed-cohorts audit.

Covers the CONFIRMED findings from the six-cohort multi-agent review of the
memory-upgrade read/write paths, the Tor search racing core, and their
supporting utilities. One class per defect; each test fails against the
pre-fix code and passes after.
"""
import asyncio
import json


# ─────────────────────────────────────────────────────────────────────
# sessions.py — eviction dead code, cap bypass, thin-client system dupes
# ─────────────────────────────────────────────────────────────────────
class TestSessionEviction:
    def _store(self, tmp_path):
        from ghost_agent.core.sessions import SessionStore
        return SessionStore(tmp_path)

    def test_evict_actually_deletes_past_cap(self, tmp_path, monkeypatch):
        import ghost_agent.core.sessions as S
        monkeypatch.setattr(S, "MAX_SESSIONS", 5)
        store = self._store(tmp_path)
        # 8 sessions, each written a beat apart so mtime ordering is stable.
        for i in range(8):
            sess = S.Session(id=f"sess{i:02d}")
            sess.messages = [{"role": "user", "content": f"m{i}"}]
            store._write(sess)
        store._evict()
        remaining = list(tmp_path.glob("*.json"))
        assert len(remaining) == 5, f"eviction left {len(remaining)} (cap 5)"

    def test_append_turn_enforces_message_cap_with_many_systems(self, tmp_path, monkeypatch):
        import ghost_agent.core.sessions as S
        monkeypatch.setattr(S, "MAX_MESSAGES_PER_SESSION", 10)
        store = self._store(tmp_path)
        sess = S.Session(id="big")
        # Pre-load more system messages than the cap — the old negative-keep
        # slice kept the whole tail.
        sess.messages = [{"role": "system", "content": f"s{i}"} for i in range(15)]
        store._write(sess)
        store.append_turn("big", [{"role": "user", "content": "hi"}], "yo")
        got = store.get("big")
        assert len(got.messages) <= 10, f"cap bypassed: {len(got.messages)} messages"


class TestMergeHistoryThinClient:
    def test_duplicate_system_prompt_not_accumulated(self):
        from ghost_agent.core.sessions import merge_history
        stored = [
            {"role": "system", "content": "You are Ghost."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        # Thin client re-sends its system prompt + one new user message.
        incoming = [
            {"role": "system", "content": "You are Ghost."},
            {"role": "user", "content": "next"},
        ]
        merged = merge_history(stored, incoming)
        systems = [m for m in merged if m.get("role") == "system"]
        assert len(systems) == 1, "duplicate system message accumulated"
        assert merged[-1]["content"] == "next"

    def test_fat_client_still_used_as_is(self):
        from ghost_agent.core.sessions import merge_history
        stored = [{"role": "user", "content": "a"},
                  {"role": "assistant", "content": "b"}]
        incoming = stored + [{"role": "user", "content": "c"}]
        assert merge_history(stored, incoming) == incoming


def _bus(sessions=None):
    from ghost_agent.core.bus import MemoryBus
    return MemoryBus(session_store=sessions)


# (TestHydrationStashIdentity went with `MemoryBus.judge_hydration_usefulness`,
# removed in §4NM; the `last_hydration` stash went in §4TD, it had no reader.)


class TestRRFConsensus:
    def test_cross_subquery_consensus_outranks_single(self):
        from ghost_agent.core.bus import MemoryBus
        # Two sub-queries; item CONSENSUS appears in both vector lists, item
        # SINGLE in only one. Per-(subquery,tier) fusion must rank CONSENSUS
        # strictly higher.
        sq1_vec = [{"source": "vector", "text": "CONSENSUS"},
                   {"source": "vector", "text": "SINGLE"}]
        sq2_vec = [{"source": "vector", "text": "CONSENSUS"}]
        fused = MemoryBus._reciprocal_rank_fusion([sq1_vec, sq2_vec], k=60)
        order = [it["text"] for it, _ in fused]
        assert order.index("CONSENSUS") < order.index("SINGLE")
        # And CONSENSUS's score is strictly greater (it got two contributions).
        scores = {it["text"]: s for it, s in fused}
        assert scores["CONSENSUS"] > scores["SINGLE"]


# ─────────────────────────────────────────────────────────────────────
# rrf_weights.py — default merge (the fit itself was removed in §4NM;
# the module is loader-only now)
# ─────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────
# search.py — anchored circuit tags, reformulation no-op guard
# ─────────────────────────────────────────────────────────────────────
class TestSearchCircuitTag:
    def test_tag_varies_over_time_for_same_query(self, monkeypatch):
        import ghost_agent.tools.search as S
        seen = {}

        def fake_identity(bare, tag):
            seen["tag"] = tag
            return f"{bare}#{tag}"

        monkeypatch.setattr("ghost_agent.utils.helpers.socks_url_with_identity",
                            fake_identity)
        monkeypatch.setattr(S.time, "monotonic", lambda: 0.0)
        t0 = S._proxy_for_attempt("socks5://127.0.0.1:9050", "q", 0, salt="moje")
        tag0 = seen["tag"]
        monkeypatch.setattr(S.time, "monotonic", lambda: 600.0)  # +10 min
        S._proxy_for_attempt("socks5://127.0.0.1:9050", "q", 0, salt="moje")
        tag1 = seen["tag"]
        assert tag0 != tag1, "identical circuit tag across the dirtiness window"
        # Query hash still present → per-query isolation preserved.
        assert tag0.startswith(tag1.split("n")[0])


class TestReformulationNoOpGuard:
    def test_short_question_does_not_reformulate_to_itself(self):
        from ghost_agent.tools.search import _reformulate_query
        q = "how does postgres vacuum work"   # 5 words, question form, no digits
        reforms = _reformulate_query(q)
        assert q not in reforms, "reformulation echoed the original query"
        assert q.strip() not in [r.strip() for r in reforms]
