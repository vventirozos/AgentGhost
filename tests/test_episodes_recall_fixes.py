"""Regression tests for the episodic-recall fixes (2026-07-22).

Every case here pins a behaviour that was verified BROKEN against the live
episode store (145 episodes):

1. Action truncation kept ``actions[:20]`` — the HEAD — and dropped the tail,
   i.e. the recovery. 16 of 139 live episodes with actions sat at exactly 20,
   silently amputated, while their row still asserted ``outcome_success = 1``.
2. ``search_recoveries`` required a non-empty ``lesson``; the only production
   writer never sets one (0 of 145 live episodes have a lesson), so every
   System-3 crisis pivot got ``[]``.
3. ``_vector_search`` retrieved via ``search_advanced`` (no type filter),
   which bumps retrieval stats on EVERY hit before episodes are filtered out —
   phantom reinforcement on document/identity rows the model never sees.
4. Episode vector twins stored a float epoch timestamp while every other
   writer stores ISO-8601, so the prompt showed ``[1783488280.73]``.
5. A failed vector ingest was logged at ``debug`` — an episode invisible to
   semantic recall, silently.
6. The two recall paths used incompatible ``relevance_score`` scales (int word
   count vs float in [0,1]) and neither carried any recency signal.
7. The substring fallback scanned only the newest 100 rows of a 500-row store.
"""

import logging
import re
import time

import pytest

from ghost_agent.memory.episodes import EpisodicMemory


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class FakeCollection:
    """Chroma-shaped collection: honours ``where`` metadata equality."""

    def __init__(self, rows=None):
        # rows: list of (vector_id, metadata, distance)
        self.rows = list(rows or [])
        self.queries = []
        self.gets = []

    def _matching(self, where):
        if not where:
            return list(self.rows)
        return [r for r in self.rows
                if all((r[1] or {}).get(k) == v for k, v in where.items())]

    def query(self, query_texts=None, n_results=5, where=None):
        self.queries.append({"query_texts": query_texts,
                             "n_results": n_results, "where": where})
        rows = self._matching(where)[:n_results]
        return {
            "ids": [[r[0] for r in rows]],
            "metadatas": [[r[1] for r in rows]],
            "distances": [[r[2] for r in rows]],
            "documents": [["doc"] * len(rows)],
        }

    def get(self, where=None, include=None):
        self.gets.append({"where": where, "include": include})
        rows = self._matching(where)
        return {"ids": [r[0] for r in rows], "metadatas": [r[1] for r in rows]}


class VectorStub:
    """Minimal VectorMemory stand-in.

    ``collection`` present  → the scoped-query path is available.
    ``collection is None``  → episodes.py must fall back to search_advanced.
    """

    def __init__(self, collection=None, advanced_hits=None, fail_add=False):
        self.collection = collection
        self.advanced_hits = list(advanced_hits or [])
        self.fail_add = fail_add
        self.added = []
        self.bumped = []
        self.advanced_calls = 0

    def add(self, text, meta=None):
        if self.fail_add:
            raise RuntimeError("embedder down")
        self.added.append({"text": text, "meta": meta or {}})

    def search_advanced(self, query, limit=5):
        self.advanced_calls += 1
        return list(self.advanced_hits)

    def bump_retrievals(self, ids):
        self.bumped.extend(ids)


@pytest.fixture
def em(tmp_path):
    return EpisodicMemory(tmp_path)


def _actions(n, fail_at=None):
    return [
        {"tool": f"tool_{i}", "args": {"i": i}, "result": f"result {i}",
         "success": (i != fail_at)}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# 1. Action truncation keeps the RESOLUTION
# ---------------------------------------------------------------------------

def test_truncation_keeps_the_tail_not_just_the_head(em):
    actions = _actions(40)
    actions[-1] = {"tool": "final_fix", "args": {}, "result": "RESOLVED",
                   "success": True}
    ep_id = em.record_episode(trigger="long task", actions=actions,
                              outcome="done", success=True)
    stored = em.get_episode(ep_id)["actions"]

    assert len(stored) == EpisodicMemory.MAX_ACTIONS_PER_EPISODE
    names = [a["tool_name"] for a in stored]
    # The resolution — previously discarded — is present.
    assert names[-1] == "final_fix"
    assert stored[-1]["result"] == "RESOLVED"
    # The opening approach is still present.
    assert names[0] == "tool_0"
    # And the gap is explicit, not silent.
    assert EpisodicMemory.TRUNCATION_MARKER_TOOL in names


def test_truncation_marker_records_how_much_was_dropped(em):
    ep_id = em.record_episode(trigger="long task", actions=_actions(40),
                              success=True)
    stored = em.get_episode(ep_id)["actions"]
    marker = [a for a in stored
              if a["tool_name"] == EpisodicMemory.TRUNCATION_MARKER_TOOL][0]
    assert "21" in marker["tool_args"]  # 40 - 5 head - 14 tail = 21 elided
    assert marker["success"] == 1       # never counts as a failed action


def test_truncation_warns_on_the_live_stream(em, caplog):
    with caplog.at_level(logging.WARNING, logger="GhostAgent"):
        em.record_episode(trigger="long task", actions=_actions(40), success=True)
    assert any("exceed the cap" in r.getMessage()
               for r in caplog.records if r.levelno >= logging.WARNING)


def test_short_action_list_is_untouched(em):
    """Happy path (the overwhelming majority of episodes) is unchanged."""
    ep_id = em.record_episode(trigger="short task", actions=_actions(4),
                              success=True)
    stored = em.get_episode(ep_id)["actions"]
    assert [a["tool_name"] for a in stored] == [
        "tool_0", "tool_1", "tool_2", "tool_3"]


# ---------------------------------------------------------------------------
# 2. search_recoveries works on the LIVE schema (no lesson field)
# ---------------------------------------------------------------------------

def test_recoveries_found_without_any_lesson(em):
    """The live shape: outcome_success=1, lesson='', one failed tool call."""
    em.record_episode(
        trigger="sandbox execution timeout while processing data",
        actions=_actions(5, fail_at=1),
        outcome="chunked the input and it went through",
        success=True,
    )
    recoveries = em.search_recoveries("execution timeout during processing")
    assert recoveries, "structurally dead filter would return [] here"
    assert recoveries[0]["lesson"] == ""
    assert recoveries[0]["recovery_evidence"] == "failed_action"
    # The recovery chain is attached even though the substring path
    # returns bare episode rows.
    assert recoveries[0]["actions"]


def test_recoveries_use_outcome_text_when_no_action_rows(em):
    em.record_episode(
        trigger="database connection pool exhausted",
        outcome="Found the bug — retried with a larger pool and it recovered",
        success=True,
    )
    recoveries = em.search_recoveries("database connection problem")
    assert len(recoveries) == 1
    assert recoveries[0]["recovery_evidence"] == "outcome_text"


def test_recoveries_still_honour_an_explicit_lesson(em):
    em.record_episode(
        trigger="rate limit exceeded calling the api",
        outcome="added backoff",
        success=True,
        lesson="use exponential backoff",
    )
    recoveries = em.search_recoveries("rate limit exceeded")
    assert recoveries[0]["recovery_evidence"] == "lesson"


def test_recoveries_exclude_unrecovered_failures(em):
    em.record_episode(
        trigger="sandbox execution timeout",
        actions=_actions(3, fail_at=0),
        outcome="could not recover",
        success=False,
    )
    assert em.search_recoveries("sandbox execution timeout") == []


def test_recoveries_prefer_strong_evidence_over_plain_success(em):
    em.record_episode(
        trigger="deployment pipeline stalled midway",
        outcome="ok",
        success=True,
    )
    em.record_episode(
        trigger="deployment pipeline stalled again",
        actions=_actions(4, fail_at=2),
        outcome="ok",
        success=True,
    )
    recoveries = em.search_recoveries("deployment pipeline stalled")
    assert [r["recovery_evidence"] for r in recoveries] == ["failed_action"]


def test_recoveries_fall_back_to_relevant_successes(em):
    em.record_episode(trigger="rebuild the search index nightly",
                      actions=_actions(3), outcome="ok", success=True)
    recoveries = em.search_recoveries("rebuild the search index")
    assert len(recoveries) == 1
    assert recoveries[0]["recovery_evidence"] == "success_only"


def test_recoveries_empty_when_nothing_is_relevant(em):
    em.record_episode(trigger="unrelated", outcome="done", success=True)
    assert em.search_recoveries("completely different problem") == []


def test_truncation_marker_is_not_a_failed_action(em):
    """A truncated-but-clean episode must not be mislabelled a recovery."""
    em.record_episode(trigger="very long clean run of the exporter",
                      actions=_actions(40), outcome="ok", success=True)
    recoveries = em.search_recoveries("long clean exporter run")
    assert recoveries[0]["recovery_evidence"] != "failed_action"


# ---------------------------------------------------------------------------
# 3. Vector retrieval is TYPE-SCOPED (no phantom stat bumps)
# ---------------------------------------------------------------------------

def test_vector_search_scopes_the_query_to_episodes(em):
    ep_id = em.record_episode(trigger="train a GRU on the pet dataset",
                              outcome="converged", success=True)
    col = FakeCollection([
        ("v-doc", {"type": "document", "source": "manual.pdf"}, 0.05),
        ("v-ident", {"type": "identity"}, 0.06),
        ("v-ep", {"type": "episode", "episode_id": ep_id}, 0.1),
    ])
    vec = VectorStub(collection=col)

    results = em.search_similar("recurrent sequence model", limit=5,
                                vector_memory=vec)

    assert [r["id"] for r in results] == [ep_id]
    # The query itself was scoped — the document/identity rows were never
    # retrieved, so nothing could bump their retrieval stats.
    assert col.queries and col.queries[0]["where"] == {"type": "episode"}
    assert vec.advanced_calls == 0
    # Reinforcement lands ONLY on the episode row actually surfaced.
    assert vec.bumped == ["v-ep"]


def test_vector_search_falls_back_when_no_scoped_path(em):
    ep_id = em.record_episode(trigger="css layout broken", success=True)
    vec = VectorStub(collection=None, advanced_hits=[
        {"id": "v-ep", "metadata": {"type": "episode", "episode_id": ep_id},
         "score": 0.1},
        {"id": "v-doc", "metadata": {"type": "document"}, "score": 0.0},
    ])
    results = em.search_similar("styling issue", limit=5, vector_memory=vec)
    assert [r["id"] for r in results] == [ep_id]
    assert vec.advanced_calls == 1
    # search_advanced already bumped internally — don't double-credit.
    assert vec.bumped == []


def test_scoped_path_with_no_episode_hits_falls_back_to_substring(em):
    em.record_episode(trigger="memory leak detected in worker", success=True)
    vec = VectorStub(collection=FakeCollection([]))
    results = em.search_similar("memory leak", limit=5, vector_memory=vec)
    assert len(results) == 1
    assert results[0]["trigger"].startswith("memory leak")


# ---------------------------------------------------------------------------
# 4. Vector twin carries an ISO-8601 timestamp
# ---------------------------------------------------------------------------

def test_episode_vector_twin_uses_iso_timestamp(em):
    vec = VectorStub()
    em.record_episode(trigger="index the postgres manual", success=True,
                      vector_memory=vec)
    meta = vec.added[0]["meta"]
    assert meta["type"] == "episode"
    assert isinstance(meta["timestamp"], str)
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z$",
                    meta["timestamp"]), meta["timestamp"]


# ---------------------------------------------------------------------------
# 5. A failed ingest is LOUD, and reconcilable
# ---------------------------------------------------------------------------

def test_failed_vector_ingest_warns(em, caplog):
    vec = VectorStub(fail_add=True)
    with caplog.at_level(logging.WARNING, logger="GhostAgent"):
        ep_id = em.record_episode(trigger="episode with a dead embedder",
                                  success=True, vector_memory=vec)
    assert ep_id  # the episode write itself still succeeds
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("vector ingest FAILED" in (r.msg or "") for r in warnings)


def test_reconcile_reingests_episodes_missing_a_vector_twin(em):
    ep1 = em.record_episode(trigger="first episode about tor circuits",
                            success=True)
    ep2 = em.record_episode(trigger="second episode about llama restarts",
                            success=True)
    # Only ep1 made it into the index.
    col = FakeCollection([("v1", {"type": "episode", "episode_id": ep1}, 0.1)])
    vec = VectorStub(collection=col)

    attempted = em.reconcile_vector_index(vec)

    assert attempted == 1
    assert len(vec.added) == 1
    assert vec.added[0]["meta"]["episode_id"] == ep2


def test_reconcile_is_a_noop_when_everything_is_indexed(em):
    ep1 = em.record_episode(trigger="only episode here", success=True)
    col = FakeCollection([("v1", {"type": "episode", "episode_id": ep1}, 0.1)])
    vec = VectorStub(collection=col)
    assert em.reconcile_vector_index(vec) == 0
    assert vec.added == []


# ---------------------------------------------------------------------------
# 6. One comparable relevance scale + a recency signal
# ---------------------------------------------------------------------------

def test_substring_relevance_is_a_fraction_not_a_word_count(em):
    em.record_episode(trigger="parse spreadsheet with pandas dataframe",
                      success=True)
    results = em.search_similar("parse spreadsheet pandas")
    assert results
    score = results[0]["relevance_score"]
    assert isinstance(score, float)
    assert 0.0 < score <= 1.0
    assert results[0]["match_count"] == 3  # raw overlap still available


def test_both_paths_produce_comparable_scores(em):
    ep_id = em.record_episode(trigger="parse spreadsheet with pandas",
                              success=True)
    substring = em.search_similar("parse spreadsheet pandas")[0]
    col = FakeCollection([
        ("v-ep", {"type": "episode", "episode_id": ep_id}, 0.1)])
    vector = em.search_similar("tabular file reader",
                               vector_memory=VectorStub(collection=col))[0]
    for score in (substring["relevance_score"], vector["relevance_score"]):
        assert isinstance(score, float) and 0.0 <= score <= 1.0
    # A full-coverage substring match and a distance-0.1 vector hit now land
    # within the same band; before, one was `3` and the other `0.9`.
    assert abs(substring["relevance_score"] - vector["relevance_score"]) < 0.3


def test_recency_breaks_ties_between_equal_matches(em):
    old_id = em.record_episode(trigger="rotate the backup archive", success=True)
    new_id = em.record_episode(trigger="rotate the backup archive", success=True)
    # Age the first episode by 90 days.
    import sqlite3
    from contextlib import closing
    with closing(sqlite3.connect(em.db_path)) as conn:
        conn.execute("UPDATE episodes SET timestamp = ? WHERE id = ?",
                     (time.time() - 90 * 86400, old_id))
        conn.commit()
    results = em.search_similar("rotate the backup archive", limit=5)
    assert [r["id"] for r in results][0] == new_id
    scores = {r["id"]: r["relevance_score"] for r in results}
    assert scores[new_id] > scores[old_id]


def test_format_episode_carries_a_relative_date(em):
    ep = {"cluster_id": "ops", "trigger": "restart llama-server",
          "outcome": "back up", "outcome_success": 1,
          "timestamp": time.time() - 3 * 86400}
    line = EpisodicMemory.format_episode(ep)
    assert "[3d ago]" in line
    assert "restart llama-server" in line


def test_format_episode_survives_a_missing_timestamp(em):
    line = EpisodicMemory.format_episode(
        {"trigger": "no timestamp here", "outcome": "ok", "outcome_success": 0})
    assert "no timestamp here" in line
    assert "ago" not in line


# ---------------------------------------------------------------------------
# 7. Reachability: the fallback scans the whole store, not the newest 100
# ---------------------------------------------------------------------------

def test_old_episodes_stay_reachable_by_the_substring_fallback(em):
    em.record_episode(trigger="calibrate the spectrometer harness", success=True)
    for i in range(130):
        em.record_episode(trigger=f"routine filler episode number {i}",
                          success=True)
    results = em.search_similar("calibrate spectrometer")
    assert results, "episode older than the newest 100 was unreachable"
    assert results[0]["trigger"].startswith("calibrate the spectrometer")


def test_fallback_scan_limit_matches_the_capacity_cap():
    assert EpisodicMemory.FALLBACK_SCAN_LIMIT >= EpisodicMemory.MAX_EPISODES
