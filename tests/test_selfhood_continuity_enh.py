"""Tests for the selfhood continuity enhancements.

Covers proposal items:
  #2 semantic (IDF-weighted) autobiographical recall + query-aware wake-up
  #3 outcome backfill into autobiographical entries
  #4 meta-cognitive narrative
  #5 wired Experience.cluster field
"""

from pathlib import Path

from ghost_agent.selfhood import SelfModel
from ghost_agent.selfhood.autobiographical import (
    AutobiographicalMemory,
    _derive_cluster,
    _outcome_phrase,
)
from ghost_agent.selfhood.recognition import build_wakeup_prefix
from ghost_agent.selfhood.schema import Experience


# --------------------------------------------------------------------------
# #5 — cluster derivation
# --------------------------------------------------------------------------

def test_derive_cluster_picks_dominant_topic():
    assert _derive_cluster("fix the python bug, error in traceback") == "debugging"
    assert _derive_cluster("write a python function to refactor code") == "coding"
    assert _derive_cluster("run a sql query against the database") == "data"
    assert _derive_cluster("") is None
    assert _derive_cluster("xyzzy plugh frobnicate") is None


def test_capture_turn_derives_cluster(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    exp = sm.capture_turn(
        trajectory_id="t-1",
        user_request="write a python function to sort a list",
        tool_names=["execute"],
        outcome="passed",
        final_response="done",
    )
    assert exp is not None
    assert exp.cluster == "coding"


def test_capture_turn_respects_explicit_cluster(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    exp = sm.capture_turn(
        trajectory_id="t-1",
        user_request="anything",
        tool_names=[],
        outcome="passed",
        final_response="done",
        cluster="sql",
    )
    assert exp.cluster == "sql"


def test_cluster_counts(tmp_path: Path):
    mem = AutobiographicalMemory(tmp_path)
    mem.append(Experience(summary="a", cluster="coding"))
    mem.append(Experience(summary="b", cluster="coding"))
    mem.append(Experience(summary="c", cluster="debugging"))
    mem.append(Experience(summary="d", cluster=None))
    counts = mem.cluster_counts()
    assert counts == {"coding": 2, "debugging": 1}


# --------------------------------------------------------------------------
# #3 — outcome backfill
# --------------------------------------------------------------------------

def test_outcome_phrase_helper():
    assert _outcome_phrase("passed") == "and the answer landed"
    assert "didn't land" in _outcome_phrase("failed", "boom")
    assert _outcome_phrase("unknown") == "without a verdict either way"


def test_update_outcome_backfills_verdict(tmp_path: Path):
    mem = AutobiographicalMemory(tmp_path)
    mem.append(Experience(
        trajectory_id="traj-1",
        summary='I worked on "x". I reached for execute without a verdict either way.',
        outcome="unknown",
    ))
    assert mem.update_outcome("traj-1", "passed") is True
    exp = list(mem.iter_experiences())[0]
    assert exp.outcome == "passed"
    assert "and the answer landed" in exp.summary
    assert "without a verdict" not in exp.summary


def test_update_outcome_idempotent_and_guards(tmp_path: Path):
    mem = AutobiographicalMemory(tmp_path)
    mem.append(Experience(trajectory_id="t1", summary="I did a thing.", outcome="passed"))
    # already that outcome → no-op
    assert mem.update_outcome("t1", "passed") is False
    # unknown trajectory id → no-op
    assert mem.update_outcome("nope", "failed") is False
    # invalid outcome → no-op
    assert mem.update_outcome("t1", "weird") is False


def test_update_outcome_targets_most_recent_match(tmp_path: Path):
    mem = AutobiographicalMemory(tmp_path)
    mem.append(Experience(trajectory_id="dup", summary="first", outcome="unknown"))
    mem.append(Experience(trajectory_id="dup", summary="second", outcome="unknown"))
    assert mem.update_outcome("dup", "failed") is True
    exps = list(mem.iter_experiences())
    assert exps[0].outcome == "unknown"   # older untouched
    assert exps[1].outcome == "failed"    # most recent updated


def test_record_outcome_facade(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    sm.capture_turn(
        trajectory_id="t-9", user_request="do a thing",
        tool_names=[], outcome="unknown", final_response="ok",
    )
    assert sm.record_outcome("t-9", "passed") is True
    assert list(sm.autobio.iter_experiences())[0].outcome == "passed"


# --------------------------------------------------------------------------
# #2 — IDF-weighted recall
# --------------------------------------------------------------------------

def test_search_my_past_idf_prefers_rare_terms(tmp_path: Path):
    mem = AutobiographicalMemory(tmp_path)
    # "problem" is common (all three); "trapdoor" is rare (one).
    mem.append(Experience(summary="I solved a trapdoor cryptography problem."))
    mem.append(Experience(summary="I solved a routing problem."))
    mem.append(Experience(summary="I solved a scheduling problem."))
    hits = mem.search_my_past("trapdoor problem", limit=3)
    assert hits
    # The rare-term match must rank first.
    assert "trapdoor" in hits[0].summary


def test_search_my_past_empty_and_short_queries(tmp_path: Path):
    mem = AutobiographicalMemory(tmp_path)
    mem.append(Experience(summary="I did something."))
    assert mem.search_my_past("", limit=5) == []
    assert mem.search_my_past("ab", limit=5) == []  # tokens <= 2 chars filtered


def test_recall_relevant_facade(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    sm.capture_turn(trajectory_id="t1", user_request="optimize a postgres index",
                    tool_names=[], outcome="passed", final_response="ok")
    sm.capture_turn(trajectory_id="t2", user_request="bake a sourdough loaf",
                    tool_names=[], outcome="passed", final_response="ok")
    hits = sm.recall_relevant("postgres index tuning", limit=3)
    assert any("postgres" in h.summary for h in hits)


def test_wakeup_prefix_query_surfaces_relevant_past(tmp_path: Path):
    mem = AutobiographicalMemory(tmp_path)
    # Many recent unrelated turns, plus one old, distinctive, relevant one.
    mem.append(Experience(summary="I debugged a kafka consumer lag issue."))
    for i in range(6):
        mem.append(Experience(summary=f"I answered trivia question {i}."))
    prefix = build_wakeup_prefix(
        autobio=mem, state=None, narrative="",
        recent_experiences_n=3, query="kafka consumer lag",
    )
    assert "kafka" in prefix
    assert "connects to things I've done before" in prefix


def test_wakeup_prefix_no_query_has_no_relevant_block(tmp_path: Path):
    mem = AutobiographicalMemory(tmp_path)
    mem.append(Experience(summary="I did a thing."))
    prefix = build_wakeup_prefix(
        autobio=mem, state=None, narrative="", recent_experiences_n=3,
    )
    assert "connects to things I've done before" not in prefix


# --------------------------------------------------------------------------
# #4 — meta-cognitive narrative
# --------------------------------------------------------------------------

async def test_narrative_folds_in_meta_insights(tmp_path: Path):
    captured = {}

    async def fake_llm(prompt: str) -> str:
        captured["prompt"] = prompt
        return "Today I noticed a pattern in my work."

    sm = SelfModel(root=tmp_path, narrative_critique_fn=fake_llm)
    sm.capture_turn(trajectory_id="t1", user_request="debug a python error",
                    tool_names=["execute"], outcome="failed",
                    final_response="", failure_reason="x")
    out = await sm.consolidate_narrative(
        meta_insights="I keep slipping on type-coercion in SQL.")
    assert out
    # The meta-insight reached the LLM prompt...
    assert "type-coercion in SQL" in captured["prompt"]
    # ...under the dedicated patterns section...
    assert "PATTERNS" in captured["prompt"]
    # ...alongside the self-derived cluster patterns.
    assert "Recurring kinds of work" in captured["prompt"]


async def test_narrative_template_fallback_includes_meta_insights(tmp_path: Path):
    # No critique_fn → template fallback path.
    sm = SelfModel(root=tmp_path)
    sm.capture_turn(trajectory_id="t1", user_request="a task",
                    tool_names=[], outcome="passed", final_response="ok")
    out = await sm.consolidate_narrative(meta_insights="I keep doing X.")
    assert "What I've noticed about myself: I keep doing X." in out


def test_stats_reports_clusters(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    sm.capture_turn(trajectory_id="t1", user_request="write python code",
                    tool_names=[], outcome="passed", final_response="ok")
    stats = sm.stats()
    assert "clusters" in stats
    assert stats["clusters"].get("coding") == 1
