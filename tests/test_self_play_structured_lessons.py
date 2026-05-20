"""Tests for the structured lesson schema redesign.

Covers the five substantive changes made in the same pass:

  1. Structured lesson schema (trigger / anti_pattern / correct_pattern /
     domains / confidence / source_challenge_hash / verified /
     retrievals / helpful_retrievals).
  2. Verification-grounded lessons — re-run the solver with the lesson
     prepended, keep only if outcome strictly improves.
  3. Retrieval feedback loop — record_retrieval,
     record_helpful_retrieval, credit_recent_retrievals, prune_low_utility.
  4. Journal-mined challenge generation — pick_journal_challenge.
  5. Correctness-weighted scoring — correctness_weighted_score,
     count_tool_errors.

Every test is hermetic: no network, no Docker, no real LLM.
"""

import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.memory.skills import (
    SkillMemory,
    build_lesson,
    compute_lesson_utility,
    render_lesson_for_prompt,
    lesson_embedding_text,
    _bm25_like_score,
    DEFAULT_RETRIEVAL_DISTANCE,
)
from ghost_agent.core.self_play_scoring import (
    correctness_weighted_score,
    count_tool_errors,
)
from ghost_agent.core.journal_challenges import (
    MinedChallenge,
    mine_challenges,
    pick_journal_challenge,
)


# ---------------------------------------------------------------------------
# 1. Structured lesson schema
# ---------------------------------------------------------------------------


class TestStructuredLessonSchema:
    def test_build_lesson_fills_both_schemas(self):
        lesson = build_lesson(
            trigger="parse apache access log",
            anti_pattern="used int() on status codes before stripping",
            correct_pattern="for line in f: ...\n```python\nint(line.strip())\n```",
            domains=["regex_parse"],
            confidence=0.7,
            source_challenge_hash="abc1234",
        )
        # Legacy fields populated for old readers.
        assert lesson["task"] == "parse apache access log"
        assert lesson["mistake"].startswith("used int()")
        assert "int(line.strip())" in lesson["solution"]
        # Structured fields populated.
        assert lesson["trigger"] == "parse apache access log"
        assert lesson["domains"] == ["regex_parse"]
        assert lesson["confidence"] == 0.7
        assert lesson["source_challenge_hash"] == "abc1234"
        assert lesson["verified"] is False
        assert lesson["retrievals"] == 0
        assert lesson["helpful_retrievals"] == 0
        # Code snippet extracted from the correct_pattern.
        assert "int(line.strip())" in lesson["code_example"]

    def test_render_lesson_structured_vs_legacy(self):
        structured = build_lesson(trigger="x", correct_pattern="y", verified=True)
        out = render_lesson_for_prompt(structured)
        assert "TRIGGER (✓): x" in out  # verified checkmark
        assert "CORRECT-PATTERN: y" in out

        legacy = {"task": "L", "mistake": "M", "solution": "S"}
        out2 = render_lesson_for_prompt(legacy)
        assert "SITUATION: L" in out2
        assert "PREVIOUS MISTAKE: M" in out2

    def test_learn_lesson_writes_structured_fields(self, tmp_path):
        sm = SkillMemory(tmp_path)
        sm.learn_lesson(
            "t",
            "m",
            "s",
            trigger="T",
            anti_pattern="M",
            correct_pattern="def f():\n    return 1",
            domains=["data_analysis"],
            confidence=0.8,
            source_challenge_hash="deadbeef",
            verified=True,
        )
        playbook = json.loads(sm.file_path.read_text())
        assert len(playbook) == 1
        l = playbook[0]
        # Both schemas present.
        assert l["task"] == "t"  # from positional task
        assert l["trigger"] == "T"
        assert l["domains"] == ["data_analysis"]
        assert l["confidence"] == 0.8
        assert l["source_challenge_hash"] == "deadbeef"
        assert l["verified"] is True
        assert "def f" in l["code_example"]

    def test_legacy_lesson_load_still_renders(self, tmp_path):
        """A playbook written by the old code (no structured fields) must
        still render and still be retrievable."""
        sm = SkillMemory(tmp_path)
        sm.save_playbook([{
            "task": "legacy task",
            "mistake": "legacy miss",
            "solution": "legacy fix",
            "frequency": 1,
        }])
        ctx = sm.get_playbook_context()
        assert "SITUATION: legacy task" in ctx
        assert "legacy fix" in ctx


# ---------------------------------------------------------------------------
# 2. Verification-grounded lessons
# ---------------------------------------------------------------------------


class TestLessonVerification:
    @pytest.mark.asyncio
    async def test_verify_helpful_returns_true_when_original_failed(self, tmp_path):
        from ghost_agent.core.dream import Dreamer

        ctx = MagicMock()
        ctx.sandbox_dir = tmp_path
        dreamer = Dreamer.__new__(Dreamer)
        dreamer.context = ctx
        dreamer.memory = MagicMock()

        # A fake temp_agent whose handle_chat is a no-op.
        temp_agent = MagicMock()
        temp_agent.handle_chat = AsyncMock(return_value=("ok", None, None))

        # Isolated context with sandbox that exits 0 on the validator.
        isolated_ctx = MagicMock()
        isolated_ctx.sandbox_manager = MagicMock()
        isolated_ctx.sandbox_manager.execute.return_value = ("OK", 0)

        lesson = build_lesson(
            trigger="do X properly",
            correct_pattern="def fix(): return 1",
        )
        result = await dreamer._verify_lesson_helpful(
            temp_agent=temp_agent,
            isolated_context=isolated_ctx,
            sandbox_path=tmp_path,
            setup_snapshot={},
            challenge_msg={"role": "user", "content": "solve this"},
            lesson=lesson,
            validation_script="",
            model_name="test",
            original_attempts_used=3,
            original_passed=False,
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_verify_helpful_returns_false_if_lesson_still_fails(self, tmp_path):
        from ghost_agent.core.dream import Dreamer
        dreamer = Dreamer.__new__(Dreamer)
        dreamer.context = MagicMock()
        dreamer.memory = MagicMock()

        temp_agent = MagicMock()
        temp_agent.handle_chat = AsyncMock(return_value=("ok", None, None))

        isolated_ctx = MagicMock()
        isolated_ctx.sandbox_manager = MagicMock()
        # Validator still exits non-zero even with the lesson.
        isolated_ctx.sandbox_manager.execute.return_value = ("FAIL", 1)

        lesson = build_lesson(
            trigger="still broken", correct_pattern="def bad(): raise"
        )
        result = await dreamer._verify_lesson_helpful(
            temp_agent=temp_agent,
            isolated_context=isolated_ctx,
            sandbox_path=tmp_path,
            setup_snapshot={},
            challenge_msg={"role": "user", "content": "solve"},
            lesson=lesson,
            validation_script="",
            model_name="test",
            original_attempts_used=1,
            original_passed=False,
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_verify_rejects_first_try_passes(self, tmp_path):
        """No improvement to prove when the original pass was first-try —
        must return False even if the verify run also passes."""
        from ghost_agent.core.dream import Dreamer
        dreamer = Dreamer.__new__(Dreamer)
        dreamer.context = MagicMock()
        dreamer.memory = MagicMock()

        temp_agent = MagicMock()
        temp_agent.handle_chat = AsyncMock(return_value=("ok", None, None))
        isolated_ctx = MagicMock()
        isolated_ctx.sandbox_manager = MagicMock()
        isolated_ctx.sandbox_manager.execute.return_value = ("OK", 0)

        lesson = build_lesson(
            trigger="already easy", correct_pattern="def trivial(): return 1"
        )
        result = await dreamer._verify_lesson_helpful(
            temp_agent=temp_agent,
            isolated_context=isolated_ctx,
            sandbox_path=tmp_path,
            setup_snapshot={},
            challenge_msg={"role": "user", "content": "solve"},
            lesson=lesson,
            validation_script="",
            model_name="test",
            original_attempts_used=1,
            original_passed=True,  # first-try pass
        )
        assert result is False


# ---------------------------------------------------------------------------
# 3. Retrieval feedback loop
# ---------------------------------------------------------------------------


class TestRetrievalFeedback:
    def test_record_retrieval_bumps_counter(self, tmp_path):
        sm = SkillMemory(tmp_path)
        sm.learn_lesson("t1", "m", "s", trigger="trig1", correct_pattern="fix")
        sm.record_retrieval("trig1")
        sm.record_retrieval("trig1")
        playbook = json.loads(sm.file_path.read_text())
        assert playbook[0]["retrievals"] == 2
        assert playbook[0]["last_retrieved_at"] != ""

    def test_record_helpful_retrieval_bumps_confidence(self, tmp_path):
        sm = SkillMemory(tmp_path)
        sm.learn_lesson("t1", "m", "s", trigger="trig1", correct_pattern="fix", confidence=0.5)
        sm.record_helpful_retrieval("trig1")
        sm.record_helpful_retrieval("trig1")
        playbook = json.loads(sm.file_path.read_text())
        assert playbook[0]["helpful_retrievals"] == 2
        assert playbook[0]["confidence"] >= 0.55

    def test_credit_recent_retrievals_only_credits_recent(self, tmp_path):
        sm = SkillMemory(tmp_path)
        sm.learn_lesson("old", "m", "s", trigger="old-trig", correct_pattern="fix")
        sm.learn_lesson("new", "m", "s", trigger="new-trig", correct_pattern="fix")
        # Retrieve only the new one — bumps last_retrieved_at.
        sm.record_retrieval("new-trig")
        credited = sm.credit_recent_retrievals(window_seconds=60)
        assert credited == 1
        playbook = json.loads(sm.file_path.read_text())
        new_entry = [p for p in playbook if p.get("trigger") == "new-trig"][0]
        old_entry = [p for p in playbook if p.get("trigger") == "old-trig"][0]
        assert new_entry["helpful_retrievals"] == 1
        assert old_entry["helpful_retrievals"] == 0

    def test_credit_recent_retrievals_is_idempotent(self, tmp_path):
        sm = SkillMemory(tmp_path)
        sm.learn_lesson("t", "m", "s", trigger="t")
        sm.record_retrieval("t")
        sm.credit_recent_retrievals(300)
        sm.credit_recent_retrievals(300)  # second call in same window
        playbook = json.loads(sm.file_path.read_text())
        # Only the first credit call counted.
        assert playbook[0]["helpful_retrievals"] == 1

    def test_get_playbook_context_increments_retrievals(self, tmp_path):
        sm = SkillMemory(tmp_path)
        sm.learn_lesson("task a", "m", "s", trigger="task a", correct_pattern="fix")
        sm.learn_lesson("task b", "m", "s", trigger="task b", correct_pattern="fix")
        # JSON fallback path (no memory_system)
        ctx = sm.get_playbook_context()
        assert "task a" in ctx
        playbook = json.loads(sm.file_path.read_text())
        for entry in playbook[:2]:
            assert entry["retrievals"] == 1

    def test_prune_low_utility_drops_bottom_quartile(self, tmp_path):
        sm = SkillMemory(tmp_path)
        # Populate 12 lessons. Half are retrieved a lot but never credited
        # (low hit rate), the other half are verified (should be kept).
        for i in range(6):
            sm.learn_lesson(
                f"useless {i}", "m", "s",
                trigger=f"useless-{i}", correct_pattern="fix", confidence=0.3,
            )
            for _ in range(10):
                sm.record_retrieval(f"useless-{i}")
        for i in range(6):
            sm.learn_lesson(
                f"keeper {i}", "m", "s",
                trigger=f"keeper-{i}", correct_pattern="fix",
                confidence=0.7, verified=True,
            )
        removed = sm.prune_low_utility(min_retrievals=5)
        assert removed > 0
        playbook = json.loads(sm.file_path.read_text())
        # Verified lessons must survive.
        triggers = [p.get("trigger") for p in playbook]
        for i in range(6):
            assert f"keeper-{i}" in triggers

    def test_prune_respects_max_drop_fraction(self, tmp_path):
        sm = SkillMemory(tmp_path)
        # 20 lessons, all eligible — but the prune must cap at 25%.
        for i in range(20):
            sm.learn_lesson(
                f"l{i}", "m", "s",
                trigger=f"trg-{i}", correct_pattern="fix", confidence=0.2,
            )
            for _ in range(10):
                sm.record_retrieval(f"trg-{i}")
        removed = sm.prune_low_utility(min_retrievals=5, max_drop_fraction=0.25)
        assert removed <= 5  # 25% of 20

    def test_prune_noop_on_small_playbook(self, tmp_path):
        sm = SkillMemory(tmp_path)
        for i in range(5):
            sm.learn_lesson(f"l{i}", "m", "s", trigger=f"t-{i}", correct_pattern="fix")
        assert sm.prune_low_utility() == 0

    def test_compute_lesson_utility_prefers_verified_and_helpful(self):
        high = build_lesson(trigger="a", correct_pattern="f", confidence=0.9, verified=True)
        high["retrievals"] = 10
        high["helpful_retrievals"] = 9
        low = build_lesson(trigger="b", correct_pattern="g", confidence=0.2)
        low["retrievals"] = 10
        low["helpful_retrievals"] = 1
        assert compute_lesson_utility(high) > compute_lesson_utility(low)


# ---------------------------------------------------------------------------
# 4. Journal-mined challenge generation
# ---------------------------------------------------------------------------


class TestJournalChallengeMining:
    def test_mine_produces_valid_python(self):
        import ast
        entries = [{
            "type": "post_mortem",
            "data": {
                "user": "Parse this access log and count 5xx errors by IP",
                "ai": "My regex did not match 5xx codes — Traceback",
                "tools": ["execute"],
            }
        }]
        out = mine_challenges(entries)
        assert len(out) == 1
        ast.parse(out[0].setup_script)
        ast.parse(out[0].validation_script)
        # Post-2026-05-17: the journal miner now materialises a shape-
        # specific fixture (log / csv / json / sql / text). An "access
        # log" entry routes to `input.log`. The prompt should reference
        # the same filename as the setup script writes.
        fixture_name = "input.log"
        assert fixture_name in out[0].setup_script
        assert fixture_name in out[0].challenge
        assert "regex_parse" in out[0].domains

    def test_mine_skips_successful_entries(self):
        entries = [{
            "type": "post_mortem",
            "data": {"user": "Write a greeting", "ai": "Hello world", "tools": []}
        }]
        assert mine_challenges(entries) == []

    def test_mine_dedupes_identical_entries(self):
        payload = {
            "type": "post_mortem",
            "data": {
                "user": "Analyse CSV and compute ERROR rates per product",
                "ai": "I had an Error with the pandas import",
            }
        }
        entries = [payload, payload, payload]
        out = mine_challenges(entries, max_out=5)
        assert len(out) == 1

    def test_mine_skips_short_entries(self):
        entries = [{
            "type": "post_mortem",
            "data": {"user": "bad", "ai": "error"}
        }]
        assert mine_challenges(entries) == []

    def test_pick_journal_challenge_reads_journal(self, tmp_path):
        from ghost_agent.memory.journal import MemoryJournal
        j = MemoryJournal(tmp_path)
        j.append("post_mortem", {
            "user": "Investigate why the regex for parsing Apache logs is failing to count errors",
            "ai": "got a Traceback — my regex was off",
        })
        mined = pick_journal_challenge(j)
        assert mined is not None
        assert isinstance(mined, MinedChallenge)
        assert mined.journal_hash != ""

    def test_pick_journal_challenge_returns_none_on_empty(self, tmp_path):
        from ghost_agent.memory.journal import MemoryJournal
        j = MemoryJournal(tmp_path)
        assert pick_journal_challenge(j) is None

    def test_pick_journal_challenge_tolerates_no_journal(self):
        assert pick_journal_challenge(None) is None

    def test_mine_end_to_end_validator_passes_compliant_solution(self, tmp_path):
        """The mined validator must pass a reasonable solution.py —
        prove the validator is runnable and not unwinnable by construction."""
        import subprocess, os
        entries = [{
            "type": "post_mortem",
            "data": {
                "user": "Filter the lines in input.txt and print the ones mentioning alpha",
                "ai": "I had an Error",
            }
        }]
        mined = mine_challenges(entries)[0]
        # Write setup + validator + a compliant solution.
        setup = tmp_path / "setup.py"
        validator = tmp_path / ".validator.py"
        solution = tmp_path / "solution.py"
        setup.write_text(mined.setup_script)
        validator.write_text(mined.validation_script)
        solution.write_text(
            "with open('input.txt') as f:\n"
            "    for line in f:\n"
            "        if 'alpha' in line:\n"
            "            print(line.strip())\n"
        )
        # Run setup to create input.txt
        r1 = subprocess.run(["python3", "setup.py"], cwd=tmp_path, capture_output=True, timeout=10)
        assert r1.returncode == 0
        r2 = subprocess.run(["python3", ".validator.py"], cwd=tmp_path, capture_output=True, timeout=10)
        assert r2.returncode == 0, r2.stderr.decode()


# ---------------------------------------------------------------------------
# 5. Correctness-weighted scoring
# ---------------------------------------------------------------------------


class TestCorrectnessWeightedScore:
    def test_passed_run_with_positive_delta_scores_above_1(self):
        s = correctness_weighted_score(passed=True, compression_delta=0.2, tool_errors=0)
        assert s > 1.0

    def test_failed_run_scores_at_or_below_zero(self):
        s = correctness_weighted_score(passed=False, compression_delta=0.5, tool_errors=0)
        assert s <= 0.0

    def test_tool_errors_penalize_score(self):
        clean = correctness_weighted_score(passed=True, compression_delta=0.0, tool_errors=0)
        dirty = correctness_weighted_score(passed=True, compression_delta=0.0, tool_errors=5)
        assert dirty < clean

    def test_count_tool_errors_detects_markers(self):
        msgs = [
            {"role": "tool", "content": "ok"},
            {"role": "tool", "content": "Error: file not found"},
            {"role": "tool", "content": "Traceback (most recent call last)"},
            {"role": "user", "content": "error-like but user msg"},
        ]
        assert count_tool_errors(msgs) == 2

    def test_count_tool_errors_handles_empty(self):
        assert count_tool_errors([]) == 0
        assert count_tool_errors(None) == 0

    def test_alpha_beta_are_adjustable(self):
        s1 = correctness_weighted_score(
            passed=True, compression_delta=0.5, tool_errors=0, alpha=0.0
        )
        s2 = correctness_weighted_score(
            passed=True, compression_delta=0.5, tool_errors=0, alpha=1.0
        )
        assert s2 > s1


# ---------------------------------------------------------------------------
# BM25-lite re-ranker
# ---------------------------------------------------------------------------


class TestBM25LikeRanking:
    def test_overlap_returns_fraction(self):
        assert _bm25_like_score("parse apache log", "parse apache") == pytest.approx(2 / 3, abs=0.01)

    def test_no_overlap_is_zero(self):
        assert _bm25_like_score("completely different query", "apache log") == 0.0

    def test_empty_inputs(self):
        assert _bm25_like_score("", "trigger") == 0.0
        assert _bm25_like_score("query", "") == 0.0

    def test_short_tokens_are_ignored(self):
        # `is`, `a`, `of` are length < 3 and filtered.
        assert _bm25_like_score("is a of", "is a of") == 0.0


# ---------------------------------------------------------------------------
# Dedup across new-schema lessons
# ---------------------------------------------------------------------------


class TestStructuredDedup:
    def test_structured_dedup_bumps_frequency(self, tmp_path):
        sm = SkillMemory(tmp_path)
        sm.learn_lesson(
            "csv parse", "wrong delim", "use csv.reader",
            trigger="csv parse", correct_pattern="import csv",
        )
        sm.learn_lesson(
            "csv parse", "wrong delim", "use csv.reader with bigger example",
            trigger="csv parse", correct_pattern="import csv\nreader = csv.reader(f)",
        )
        playbook = json.loads(sm.file_path.read_text())
        assert len(playbook) == 1
        assert playbook[0]["frequency"] == 2
        # Richer pattern should replace the shorter one.
        assert "reader = csv.reader" in playbook[0]["correct_pattern"]

    def test_verified_flag_upgrades_existing_lesson(self, tmp_path):
        sm = SkillMemory(tmp_path)
        sm.learn_lesson(
            "trigger x", "m", "s",
            trigger="trigger x", correct_pattern="p", confidence=0.4,
        )
        sm.learn_lesson(
            "trigger x", "m", "s",
            trigger="trigger x", correct_pattern="p", confidence=0.5,
            verified=True,
        )
        playbook = json.loads(sm.file_path.read_text())
        assert playbook[0]["verified"] is True
        assert playbook[0]["confidence"] >= 0.5


# ---------------------------------------------------------------------------
# Dreamer-level hooks (journal challenge probability, report format)
# ---------------------------------------------------------------------------


class TestDreamerHooks:
    def test_try_journal_challenge_probability_zero_returns_none(self, tmp_path):
        from ghost_agent.core.dream import Dreamer
        dreamer = Dreamer.__new__(Dreamer)
        dreamer.context = MagicMock()
        dreamer.context.journal = MagicMock()
        dreamer.context.journal.load.return_value = [
            {"type": "post_mortem", "data": {"user": "x" * 50, "ai": "error"}}
        ]
        assert dreamer._try_journal_challenge(probability=0.0) is None

    def test_try_journal_challenge_returns_triple_when_hit(self, tmp_path):
        from ghost_agent.core.dream import Dreamer
        dreamer = Dreamer.__new__(Dreamer)
        dreamer.context = MagicMock()
        dreamer.context.journal = MagicMock()
        dreamer.context.journal.load.return_value = [{
            "type": "post_mortem",
            "data": {
                "user": "Regex over apache access log to count 5xx by IP — this should be doable",
                "ai": "Traceback (most recent call last)",
            }
        }]
        # Force the probability to always fire.
        with patch("random.random", return_value=0.0):
            result = dreamer._try_journal_challenge(probability=1.0)
        assert result is not None
        challenge, setup, validator, tag, domains = result
        assert "solution.py" in validator
        assert tag == "journal_replay"
        assert "regex_parse" in domains

    def test_try_journal_challenge_skips_when_no_journal(self):
        from ghost_agent.core.dream import Dreamer
        dreamer = Dreamer.__new__(Dreamer)
        dreamer.context = MagicMock()
        dreamer.context.journal = None
        assert dreamer._try_journal_challenge(probability=1.0) is None


# ---------------------------------------------------------------------------
# Regression: dedup still works with memory_system mock
# ---------------------------------------------------------------------------


class TestVectorDedupRegression:
    def test_vector_duplicate_still_detected(self, tmp_path):
        sm = SkillMemory(tmp_path)
        mock_mem = MagicMock()
        mock_mem.collection = MagicMock()
        mock_mem.collection.query.return_value = {
            "ids": [["id1"]],
            "documents": [["SITUATION: x\nMISTAKE: y\nSOLUTION: z"]],
            "distances": [[0.05]],
        }
        sm.learn_lesson(
            "same task", "m", "s",
            memory_system=mock_mem,
            trigger="same task", correct_pattern="fix",
        )
        # Playbook should be empty — vector dedup skipped the insert.
        playbook = json.loads(sm.file_path.read_text())
        assert len(playbook) == 0


# ---------------------------------------------------------------------------
# Retrieval uses the tightened distance threshold
# ---------------------------------------------------------------------------


class TestTightenedRetrievalThreshold:
    def test_default_threshold_is_tighter_than_old_065(self):
        assert DEFAULT_RETRIEVAL_DISTANCE < 0.65

    def test_retrieval_respects_custom_threshold(self, tmp_path):
        sm = SkillMemory(tmp_path)
        sm.learn_lesson("t", "m", "s", trigger="t", correct_pattern="fix")
        mock_mem = MagicMock()
        mock_mem.collection = MagicMock()
        mock_mem.collection.query.return_value = {
            "ids": [["id1"]],
            "documents": [["SITUATION: t\nMISTAKE: m\nSOLUTION: fix"]],
            "distances": [[0.50]],  # between tight (0.45) and old (0.65)
            "metadatas": [[{"trigger": "t"}]],
        }
        # At default threshold (0.45), this gets dropped.
        ctx_tight = sm.get_playbook_context(query="t", memory_system=mock_mem)
        # At 0.60, it's included.
        ctx_loose = sm.get_playbook_context(
            query="t", memory_system=mock_mem, distance_threshold=0.60,
        )
        assert "RELEVANT LESSONS" not in ctx_tight
        assert "RELEVANT LESSONS" in ctx_loose
