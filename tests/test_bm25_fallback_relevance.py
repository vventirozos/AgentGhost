"""The BM25 lesson-retrieval fallback must not match unrelated queries.

THE DEFECT (found 2026-08-11 while scoping the hydration arm, #62). The
fallback branch — taken whenever `memory_system` is None, which `main.py`
reaches by assigning `context.memory_system` inside a try/except that logs
and continues, so a VectorMemory init failure routes EVERY turn through it —
admitted a lesson on `score > 0`, where the score was the raw fraction of
query tokens present in the trigger with no IDF and no stopword filter.

One shared `the` scored 1/9 > 0. Measured on the live 50-lesson playbook:

    "what year did the treaty of westphalia end the thirty years war"  -> 5 lessons
    "describe the mating rituals of the emperor penguin in antarctica" -> 5 lessons
    "how do i bake a sourdough loaf with a long cold ferment"          -> 5 lessons

Only literal gibberish scored 0. In the degraded mode this branch exists to
serve, every turn was getting 5 arbitrary tool-use lessons in its prompt.

⚠ THESE TESTS ARE TWO-SIDED ON PURPOSE. Suppressing everything would also
make the unrelated queries return 0, so each "must reject" case is paired
with a "must still retrieve" case. A filter that fires on everything is not
a fix, it is a different outage.
"""

import json
from pathlib import Path

import pytest

from ghost_agent.memory import skills as sk

UNRELATED = [
    ("penguins", "describe the mating rituals of the emperor penguin in antarctica"),
    ("westphalia", "what year did the treaty of westphalia end the thirty years war"),
    ("sourdough", "how do i bake a sourdough loaf with a long cold ferment"),
    ("worldcup", "who won the world cup final in 1998 and what was the score"),
    ("biography", "I am a PostgreSQL engineer with 20+ years of experience."),
]

TRIGGERS = [
    "use the news_headlines skill to get me 3 headlines",
    "when using the introspect tool, ensure the requested action is available",
    "using the file system tool, count the lines in the log file",
    "stop the chess service.",
    "how to see my current role in postgres ?",
    "update the information about this project, including its status",
    "when asked to perform a health probe, reply with the service status",
    "log parsing and iso timestamp handling",
    "data aggregation and ratio calculation from csv files",
    "parsing structured log files using regular expressions",
]


def _playbook(triggers=TRIGGERS):
    return [{"trigger": t, "task": t, "solution": "s", "confidence": 0.5,
             "retrievals": 0, "helpful_retrievals": 0, "frequency": 1,
             "verified": False, "schema_version": 2} for t in triggers]


@pytest.fixture()
def sm(tmp_path):
    md = tmp_path / "memory"
    md.mkdir()
    (md / "skills_playbook.json").write_text(json.dumps(_playbook()))
    return sk.SkillMemory(md)


# ── the regression ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,query", UNRELATED)
def test_an_unrelated_query_retrieves_NOTHING_on_the_fallback(sm, name, query):
    items, branch = sm._playbook_items_and_branch(query, None)
    assert not items, (
        f"{name!r} pulled {len(items)} lesson(s) — the stopword-overlap "
        f"regression is back: {[i.get('trigger') for i in items]}")
    assert branch == "bm25_empty"


@pytest.mark.parametrize("query,expect_in_trigger", [
    ("Use the news_headlines skill to get me 3 headlines", "news_headlines"),
    ("Stop the chess service.", "chess"),
    ("Using the file system tool, count the lines in the log file", "count the lines"),
    ("how to see my current role in postgres ?", "postgres"),
])
def test_a_REAL_task_query_still_retrieves_the_right_lesson(sm, query, expect_in_trigger):
    """⚠ THE OVER-SUPPRESSION GUARD. Without this, `return []` passes every
    rejection test above."""
    items, branch = sm._playbook_items_and_branch(query, None)
    assert items, f"{query!r} retrieved nothing — the floor is too high"
    assert branch == "bm25"
    assert any(expect_in_trigger in (i.get("trigger") or "") for i in items), (
        f"retrieved the wrong lessons: {[i.get('trigger') for i in items]}")


def test_a_single_shared_stopword_is_not_a_match():
    """The exact mechanism of the bug, isolated."""
    assert sk._bm25_like_score("the treaty of westphalia", "the chess service") == 0.0
    assert sk._bm25_like_score("what is that", "that file system") == 0.0


def test_no_query_still_falls_back_to_RECENCY(sm):
    """⚠ SCOPE GUARD. The floor applies to the keyword branch only. A caller
    with NO query wants generic system-prompt injection, and that contract
    must not change."""
    items, branch = sm._playbook_items_and_branch(None, None)
    assert branch == "recency" and items


# ── IDF ────────────────────────────────────────────────────────────────────

def test_idf_sinks_boilerplate_and_lifts_distinctive_terms():
    """This is what makes the score self-calibrating: no hand-written
    stopword list can track which words are boilerplate IN THIS playbook,
    but the corpus always knows."""
    triggers = ["output formatting for reports", "output formatting for logs",
                "output formatting for tables", "quantum entanglement analysis"]
    idf = sk._bm25_idf(triggers)
    assert idf["formatting"] < idf["quantum"]
    assert idf["output"] < idf["entanglement"]


def test_idf_weighted_coverage_beats_raw_token_count():
    """A trigger matching one RARE query term should outrank one matching two
    boilerplate terms — the ordering raw counting gets backwards."""
    triggers = ["output formatting alpha", "output formatting beta",
                "output formatting gamma", "postgres role inspection"]
    idf = sk._bm25_idf(triggers)
    q = "postgres output formatting"
    rare = sk._bm25_like_score(q, "postgres role inspection", idf)
    common = sk._bm25_like_score(q, "output formatting alpha", idf)
    assert rare > common


def test_query_terms_absent_from_the_corpus_are_not_free():
    """⚠ EDGE CASE THAT WOULD RE-OPEN THE BUG. Unseen terms have no IDF
    entry. Scoring them 0 would let a trigger claim FULL coverage of a query
    made entirely of words it does not contain — every unrelated query would
    match again, by a different route."""
    idf = sk._bm25_idf(["output formatting for reports", "chess service control"])
    score = sk._bm25_like_score("penguin antarctica mating rituals", "chess service control", idf)
    assert score == 0.0
    partial = sk._bm25_like_score("penguin antarctica chess", "chess service control", idf)
    assert partial < 0.5, "unseen terms were treated as worthless, inflating coverage"


# ── the calibrated floor ───────────────────────────────────────────────────

_BOILERPLATE = ["data aggregation for reports", "data formatting for logs",
                "data parsing for tables", "data export for dashboards",
                "chess service control", "postgres role inspection"]


@pytest.mark.parametrize("query", [
    "show me the data about penguin migration patterns across antarctica during winter",
    "what data did the treaty of westphalia record about european borders",
    "the data on world cup finals since 1930 by host nation",
])
def test_the_FLOOR_rejects_scores_that_survive_the_stopword_filter(tmp_path, query):
    """⚠ THE PIN THAT WAS MISSING, AND MUTATION-TESTING FOUND IT.

    The first version of these tests never exercised the floor at all: with
    IDF and stopword filtering, the unrelated queries in the main fixture
    score EXACTLY 0.0, so reverting the gate to the old `score > 0` left
    every test green. The threshold was untested.

    The gap the floor covers is `0 < score < _BM25_MIN_SCORE` — a real,
    non-stopword term shared with a trigger, but carrying too little of the
    query's information mass to mean anything. On the live playbook that band
    held "westphalia" (0.14) and "world cup" (0.20). Here `data` is
    boilerplate (low IDF) and the queries are long, so each scores ~0.04–0.05:
    non-zero, and rejected only because of the floor.
    """
    md = tmp_path / "memory"
    md.mkdir()
    (md / "skills_playbook.json").write_text(json.dumps(_playbook(_BOILERPLATE)))
    memory = sk.SkillMemory(md)

    idf = sk._bm25_idf(_BOILERPLATE)
    raw = max(sk._bm25_like_score(query, t, idf) for t in _BOILERPLATE)
    assert 0.0 < raw < sk._BM25_MIN_SCORE, (
        f"fixture no longer lands in the band the floor guards (raw={raw:.4f}); "
        f"this test would pass with the floor removed")

    items, branch = memory._playbook_items_and_branch(query, None)
    assert not items and branch == "bm25_empty"


def test_the_floor_separates_the_measured_populations():
    """CALIBRATED, not chosen (§4AP): unrelated 0.00–0.20, paraphrases
    0.26–0.54, near-verbatim 0.66–1.00 on the live playbook."""
    assert 0.20 < sk._BM25_MIN_SCORE <= 0.66


def test_the_floor_is_operator_overridable(monkeypatch):
    monkeypatch.setenv("GHOST_BM25_MIN_SCORE", "0.9")
    import importlib
    reloaded = importlib.reload(sk)
    try:
        assert reloaded._BM25_MIN_SCORE == 0.9
    finally:
        monkeypatch.delenv("GHOST_BM25_MIN_SCORE", raising=False)
        importlib.reload(sk)


def test_a_malformed_override_does_not_crash_import(monkeypatch):
    monkeypatch.setenv("GHOST_BM25_MIN_SCORE", "not-a-number")
    import importlib
    try:
        with pytest.raises(ValueError):
            importlib.reload(sk)
    finally:
        monkeypatch.delenv("GHOST_BM25_MIN_SCORE", raising=False)
        importlib.reload(sk)


# ── degenerate inputs ──────────────────────────────────────────────────────

@pytest.mark.parametrize("q,t", [
    ("", "chess service"), ("chess", ""), (None, "chess"), ("chess", None),
    ("the and for", "chess service"),          # query is ALL stopwords
])
def test_degenerate_inputs_score_zero_without_raising(q, t):
    assert sk._bm25_like_score(q, t) == 0.0


def test_empty_idf_falls_back_to_unweighted_coverage_still_stopword_filtered():
    """`idf={}` must not resurrect the old behaviour."""
    assert sk._bm25_like_score("the treaty of westphalia", "the chess service", {}) == 0.0
    assert sk._bm25_like_score("chess service status", "chess service", {}) > 0.0
