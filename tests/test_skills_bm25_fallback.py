"""Regression test for skill-playbook retrieval when vector store is offline.

Before the fix, `SkillMemory.get_playbook_context(query, memory_system=None)`
dumped the N most recent lessons regardless of the query's relevance.
Confirmed live: asking *"what's the capital of France?"* surfaced a
lesson about fixing a Python SyntaxError, just because the SyntaxError
lesson was the most recent entry.

The bug path fires whenever the caller has a real query but no vector
store to query against. That happens:
  * when main.py's VectorMemory init raises (sentence-transformers not
    installable, no disk space, etc.) and the fallback lands on
    `context.memory_system = None` with the skill store still alive;
  * in any test / embedded usage that wires a `MemoryBus` without a
    vector tier;
  * on the MemoryBus's `_fetch_skill` path when `self.vector is None`.

The fix adds a BM25-keyword-overlap fallback between the vector path
and the legacy recency fallback. With a query + no vector, we return
only lessons whose trigger shares tokens with the query. With NO query
(system-prompt injection style), recency still applies because "some
recent lesson is better than nothing" is the right default there.
"""

import tempfile
from pathlib import Path

from ghost_agent.memory.skills import SkillMemory


def _seed(tmp: Path) -> SkillMemory:
    sm = SkillMemory(tmp)
    sm.learn_lesson(
        task="fix a SyntaxError in a Python function",
        mistake="missing colon after def",
        solution="add the missing colon",
    )
    sm.learn_lesson(
        task="read a logfile that may not exist",
        mistake="tried to read hardcoded path without checking",
        solution="list the directory first, then read",
    )
    return sm


# ------------------------------------------------------------------
# Happy path: query unrelated to any lesson → empty (no pollution)
# ------------------------------------------------------------------

def test_unrelated_query_no_memory_system_returns_empty():
    with tempfile.TemporaryDirectory() as td:
        sm = _seed(Path(td))
        ctx = sm.get_playbook_context(
            query="what's the capital of France?",
            memory_system=None,
        )
        assert ctx == "", (
            f"unrelated query must not surface lessons when vector is off; "
            f"got: {ctx!r}"
        )


def test_unrelated_technical_query_still_returns_empty():
    with tempfile.TemporaryDirectory() as td:
        sm = _seed(Path(td))
        ctx = sm.get_playbook_context(
            query="can you deploy an nginx reverse proxy for me?",
            memory_system=None,
        )
        assert ctx == ""


# ------------------------------------------------------------------
# Happy path: query semantically related by keyword → surfaced
# ------------------------------------------------------------------

def test_keyword_match_surfaces_lesson():
    with tempfile.TemporaryDirectory() as td:
        sm = _seed(Path(td))
        ctx = sm.get_playbook_context(
            query="I hit a SyntaxError in my python script",
            memory_system=None,
        )
        assert ctx, "matching lesson should be surfaced"
        assert "SyntaxError" in ctx
        assert "RELEVANT LESSONS" in ctx
        assert "RECENT LESSONS" not in ctx, (
            "must not use the recency header when we've done a real filter"
        )


def test_logfile_query_surfaces_discover_first_lesson():
    with tempfile.TemporaryDirectory() as td:
        sm = _seed(Path(td))
        ctx = sm.get_playbook_context(
            query="how should I read the logfile?",
            memory_system=None,
        )
        assert "logfile" in ctx or "directory" in ctx.lower()


def test_multiple_lessons_ranked_by_overlap():
    """Query that overlaps BOTH lessons should surface the best-
    matching one first (not just insertion order)."""
    with tempfile.TemporaryDirectory() as td:
        sm = _seed(Path(td))
        # "read a python logfile" hits the 'read a logfile' lesson
        # (overlap: 'read', 'logfile') more than the SyntaxError one
        # (overlap: 'python'). So the logfile lesson should be first.
        ctx = sm.get_playbook_context(
            query="how should I read a python logfile?",
            memory_system=None,
        )
        assert ctx
        logfile_pos = ctx.find("logfile")
        syntax_pos = ctx.find("SyntaxError")
        assert logfile_pos != -1
        if syntax_pos != -1:
            assert logfile_pos < syntax_pos, (
                f"logfile lesson should rank higher than syntax: "
                f"logfile@{logfile_pos}, syntax@{syntax_pos}"
            )


# ------------------------------------------------------------------
# Recency fallback PRESERVED when query is missing
# ------------------------------------------------------------------

def test_no_query_uses_recency_fallback():
    """System-prompt injection style: no query → give me any recent
    lessons to put in the prompt. This path MUST keep working."""
    with tempfile.TemporaryDirectory() as td:
        sm = _seed(Path(td))
        ctx = sm.get_playbook_context(query=None, memory_system=None)
        assert "RECENT LESSONS" in ctx, (
            "no-query path must still return recent lessons"
        )


def test_empty_string_query_uses_recency_fallback():
    with tempfile.TemporaryDirectory() as td:
        sm = _seed(Path(td))
        ctx = sm.get_playbook_context(query="", memory_system=None)
        assert "RECENT LESSONS" in ctx


def test_whitespace_only_query_uses_recency_fallback():
    with tempfile.TemporaryDirectory() as td:
        sm = _seed(Path(td))
        ctx = sm.get_playbook_context(query="   \n\t  ", memory_system=None)
        assert "RECENT LESSONS" in ctx


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------

def test_empty_playbook_no_query():
    with tempfile.TemporaryDirectory() as td:
        sm = SkillMemory(Path(td))
        ctx = sm.get_playbook_context(query=None, memory_system=None)
        assert ctx == "No lessons learned yet."


def test_empty_playbook_with_query():
    with tempfile.TemporaryDirectory() as td:
        sm = SkillMemory(Path(td))
        ctx = sm.get_playbook_context(
            query="anything at all",
            memory_system=None,
        )
        # Empty playbook + query → same "no lessons" sentinel
        assert ctx == "No lessons learned yet."


def test_query_with_only_stopwords_returns_empty():
    """Query that tokenizes to nothing useful ('the', 'a', 'is' are
    all filtered by the BM25 token filter which drops ≤3-char words).
    Expected: no false-positive match, return empty."""
    with tempfile.TemporaryDirectory() as td:
        sm = _seed(Path(td))
        ctx = sm.get_playbook_context(query="is a the it", memory_system=None)
        assert ctx == ""


def test_record_retrievals_bumps_counter_on_bm25_hit():
    """When BM25 surfaces a lesson, the retrieval counter on that
    lesson should increment — same contract as the vector path."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        sm = _seed(tmp)
        # Read playbook, check initial retrieval counts
        import json as js
        before = js.loads((tmp / "skills_playbook.json").read_text())
        initial_counts = {p.get("task"): p.get("retrievals", 0) for p in before}

        sm.get_playbook_context(
            query="fix a python SyntaxError issue",
            memory_system=None,
        )

        after = js.loads((tmp / "skills_playbook.json").read_text())
        for p in after:
            task = p.get("task")
            delta = p.get("retrievals", 0) - initial_counts.get(task, 0)
            if "SyntaxError" in (task or ""):
                assert delta >= 1, (
                    f"retrieval counter not bumped for matching lesson: "
                    f"initial={initial_counts.get(task)} after={p.get('retrievals')}"
                )


def test_record_retrievals_false_does_not_bump():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        sm = _seed(tmp)
        import json as js
        before = js.loads((tmp / "skills_playbook.json").read_text())
        initial = {p.get("task"): p.get("retrievals", 0) for p in before}

        sm.get_playbook_context(
            query="fix a python SyntaxError issue",
            memory_system=None,
            record_retrievals=False,
        )

        after = js.loads((tmp / "skills_playbook.json").read_text())
        for p in after:
            task = p.get("task")
            if "SyntaxError" in (task or ""):
                assert p.get("retrievals", 0) == initial.get(task, 0), (
                    "record_retrievals=False must not bump the counter"
                )


def test_limit_respected():
    """Limit parameter caps the number of lessons returned even when
    many match."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        sm = SkillMemory(tmp)
        for i in range(10):
            sm.learn_lesson(
                task=f"fix python syntax error case {i}",
                mistake=f"mistake {i}",
                solution=f"solution {i}",
            )
        ctx = sm.get_playbook_context(
            query="fix python syntax",
            memory_system=None,
            limit=3,
        )
        # Count how many "TRIGGER" entries appear
        n = ctx.count("TRIGGER")
        assert 1 <= n <= 3, f"expected at most 3 lessons, got {n}"
