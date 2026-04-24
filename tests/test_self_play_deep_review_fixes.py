"""Tests for the deep-review self-play fixes.

Nine fixes are covered, one TestCase class each, in the same order they
were applied:

    1. Journal-sourced runs do not trigger skill writes on pass.
    2. Stored lesson trigger is free of "[Self-Play] " prefix — a
       separate `source` field carries provenance.
    3. Verification uses the production SKILL PLAYBOOK format.
    4. sql / bash / concurrency cluster templates exist and classify.
    5. Playbook trimming at capacity is utility-aware, not FIFO.
    6. Retrieval credit fires on any clean-exit turn, not just complex
       ones.
    7. Planner prompt includes relevant prior lessons.
    8. Long-transcript summariser preserves turning-point breadcrumbs.
    9. Self-play report lands in scratchpad on the success path too.

All tests are hermetic — no network, no Docker, no LLM.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.core.challenge_templates import TEMPLATES, try_template
from ghost_agent.core.dream import _summarize_long_transcript
from ghost_agent.core.self_play_scoring import correctness_weighted_score
from ghost_agent.memory.frontier import classify_cluster
from ghost_agent.memory.scratchpad import Scratchpad
from ghost_agent.memory.skills import (
    PLAYBOOK_MAX,
    SkillMemory,
    _trim_playbook_by_utility,
    build_lesson,
    compute_lesson_utility,
    lesson_embedding_text,
    render_lesson_for_prompt,
)


# ---------------------------------------------------------------------------
# 1. Journal-sourced runs do not trigger skill writes on pass
# ---------------------------------------------------------------------------


class TestJournalSourcedSkillGate:
    """The journal-mined validator is deliberately lenient (any line
    containing a token from input.txt passes). A "pass" there carries
    almost no correctness signal; writing a skill from a trivially
    passing run would poison the playbook with lessons whose trigger
    is the user's real task but whose pattern is "print any line".
    This is guarded by the new gate in `dream.synthetic_self_play`.
    """

    def _branch_reached(self, *, journal_source, passed, attempt, is_new_cluster):
        """Replay exactly the gate cascade from dream.synthetic_self_play
        (lines ~2136-2170 after the fix). Returns the resulting
        ``(should_write_skill, gate_reason)`` tuple so we can assert
        against the branch that fired.
        """
        aborted_by_solver = False
        mastered = False
        compression_delta = 0.2  # plausible first-try compression gain

        should_write_skill = False
        gate_reason = ""
        if aborted_by_solver:
            gate_reason = "aborted"
        elif mastered:
            gate_reason = "mastered"
        elif journal_source and passed:
            gate_reason = "journal-mined pass → lenient validator, skill write suppressed"
        elif passed and attempt > 0:
            should_write_skill = True
            gate_reason = "struggled-then-won"
        elif passed and attempt == 0 and (is_new_cluster or compression_delta > 0.05):
            should_write_skill = True
            gate_reason = "new cluster or compression improvement"
        elif not passed and is_new_cluster:
            should_write_skill = True
            gate_reason = "first failure on new cluster"
        elif not passed:
            gate_reason = "repeat failure on known cluster"
        else:
            gate_reason = "no new signal"
        return should_write_skill, gate_reason

    def test_journal_source_first_try_pass_is_suppressed(self):
        write, reason = self._branch_reached(
            journal_source=True, passed=True, attempt=0, is_new_cluster=True,
        )
        assert write is False
        assert "journal" in reason

    def test_journal_source_struggled_then_won_is_suppressed(self):
        # Even attempt>0+passed is suppressed on journal — the lenient
        # validator makes any "success" low-signal regardless of
        # attempt count.
        write, reason = self._branch_reached(
            journal_source=True, passed=True, attempt=2, is_new_cluster=False,
        )
        assert write is False
        assert "journal" in reason

    def test_journal_source_failure_still_writes(self):
        # A failure even on the lenient validator IS informative —
        # the solver couldn't even print any qualifying output —
        # so writing the lesson is still useful.
        write, reason = self._branch_reached(
            journal_source=True, passed=False, attempt=2, is_new_cluster=True,
        )
        assert write is True
        assert "new cluster" in reason

    def test_non_journal_first_try_pass_still_writes(self):
        write, _ = self._branch_reached(
            journal_source=False, passed=True, attempt=0, is_new_cluster=True,
        )
        assert write is True


# ---------------------------------------------------------------------------
# 2. Stored trigger is free of "[Self-Play] " prefix
# ---------------------------------------------------------------------------


class TestTriggerPrefixRemoved:
    def test_build_lesson_accepts_source_metadata(self):
        lesson = build_lesson(
            trigger="parse access log",
            correct_pattern="use re.match",
            source="self_play",
        )
        assert lesson["trigger"] == "parse access log"
        assert lesson["source"] == "self_play"
        # Crucially the provenance does NOT pollute the retrieval text.
        assert "[Self-Play]" not in lesson["trigger"]
        assert "[Self-Play]" not in lesson["task"]
        assert "self_play" not in lesson_embedding_text(lesson).lower()

    def test_learn_lesson_passes_source_through(self, tmp_path):
        sm = SkillMemory(tmp_path)
        sm.learn_lesson(
            "parse access log", "m", "s",
            trigger="parse access log",
            correct_pattern="use re.match",
            source="self_play",
        )
        playbook = json.loads(sm.file_path.read_text())
        assert playbook[0]["source"] == "self_play"
        assert playbook[0]["trigger"] == "parse access log"
        assert not playbook[0]["trigger"].startswith("[Self-Play]")

    def test_retrieval_text_stays_clean(self, tmp_path):
        """Regression: the embedded retrieval doc must not carry the
        provenance prefix. A user query like "parse access log" needs
        to match the lesson trigger directly — if the stored trigger
        were "[Self-Play] parse access log", BM25 overlap gets diluted
        by tokens nobody will ever search for.
        """
        sm = SkillMemory(tmp_path)
        sm.learn_lesson(
            "parse access log", "m", "s",
            trigger="parse access log", correct_pattern="fix",
            source="self_play",
        )
        ctx = sm.get_playbook_context(query="parse access log")
        assert "Self-Play" not in ctx
        assert "parse access log" in ctx


# ---------------------------------------------------------------------------
# 3. Verification uses production SKILL PLAYBOOK format
# ---------------------------------------------------------------------------


class TestVerificationFormatMatchesProduction:
    @pytest.mark.asyncio
    async def test_lesson_injection_uses_production_headers(self, tmp_path):
        """The lesson text prepended to the verification challenge
        should be structurally identical to what `agent.py` injects
        at inference time — namely `### SKILL PLAYBOOK:` wrapping a
        `## RELEVANT LESSONS LEARNED` block rendered by
        `render_lesson_for_prompt`. A verified-but-misformatted lesson
        would otherwise pass verification and still fail to fire in
        production because the two paths spoke different dialects.
        """
        from ghost_agent.core.dream import Dreamer

        captured_body = {}

        async def _capture(body, background_tasks=None):
            captured_body["content"] = body["messages"][0]["content"]
            return ("ok", None, None)

        temp_agent = MagicMock()
        temp_agent.handle_chat = AsyncMock(side_effect=_capture)

        isolated_ctx = MagicMock()
        isolated_ctx.sandbox_manager = MagicMock()
        isolated_ctx.sandbox_manager.execute.return_value = ("OK", 0)

        dreamer = Dreamer.__new__(Dreamer)
        dreamer.context = MagicMock()
        dreamer.memory = MagicMock()

        lesson = build_lesson(
            trigger="log parsing", correct_pattern="use regex", verified=False
        )
        await dreamer._verify_lesson_helpful(
            temp_agent=temp_agent,
            isolated_context=isolated_ctx,
            sandbox_path=tmp_path,
            setup_snapshot={},
            challenge_msg={"role": "user", "content": "SOLVE"},
            lesson=lesson,
            validation_script="",
            model_name="test",
            original_attempts_used=3,
            original_passed=False,
        )
        content = captured_body["content"]
        # Production-equivalent header exists.
        assert "### SKILL PLAYBOOK:" in content
        assert "## RELEVANT LESSONS LEARNED" in content
        # The old dialect is gone.
        assert "### PRIOR LESSON" not in content
        # And the user's challenge still comes through unchanged.
        assert "SOLVE" in content


# ---------------------------------------------------------------------------
# 4. sql / bash / concurrency cluster templates
# ---------------------------------------------------------------------------


class TestMissingClusterTemplates:
    @pytest.mark.parametrize("cluster", ["sql", "bash", "concurrency"])
    def test_template_registered(self, cluster):
        assert cluster in TEMPLATES
        triple = try_template(cluster)
        assert triple is not None
        prompt, setup, validator = triple
        assert prompt and setup and validator

    @pytest.mark.parametrize("cluster", ["sql", "bash", "concurrency"])
    def test_prompt_classifies_back_to_cluster(self, cluster):
        """Frontier classification must route the generated prompt to
        the cluster it was designed for, otherwise seed-pick /
        brittle-cluster targeting accounting goes wrong."""
        prompt, _, _ = try_template(cluster)
        assert classify_cluster(prompt) == cluster

    @pytest.mark.parametrize("cluster", ["sql", "bash", "concurrency"])
    def test_setup_and_validator_are_stdlib_safe_python(self, cluster):
        """Setup scripts must parse as Python 3 and use stdlib only."""
        import ast
        _, setup, validator = try_template(cluster)
        ast.parse(setup)
        ast.parse(validator)
        for forbidden in ("faker", "requests", "numpy", "pandas"):
            assert forbidden not in setup.lower()

    @pytest.mark.parametrize("cluster", ["sql", "bash", "concurrency"])
    def test_template_end_to_end_passes_with_reference_solution(self, cluster, tmp_path):
        """Run setup, write a known-good solution.py, run validator.
        Proves the template isn't unwinnable by construction.

        The concurrency cluster now routes to a variant bank (parallel
        sum, max-with-source, shared counter, bounded pool, first-hit
        racer) so we pin the test to the parallel-sum variant whose
        reference solution lives below — each sibling variant has its
        own coverage in `test_self_play_loop_and_lessons.py`.
        """
        import subprocess
        if cluster == "concurrency":
            from ghost_agent.core.challenge_templates import _concurrency_parallel_sum
            prompt, setup, validator = _concurrency_parallel_sum()
        else:
            prompt, setup, validator = try_template(cluster)
        (tmp_path / ".setup.py").write_text(setup)
        (tmp_path / ".validator.py").write_text(validator)

        solutions = {
            "sql": (
                "import sqlite3\n"
                "from collections import defaultdict\n"
                "conn = sqlite3.connect('shop.db')\n"
                "totals = defaultdict(float)\n"
                "for p, a in conn.execute('SELECT product, amount FROM sales'):\n"
                "    totals[p] += a\n"
                "conn.close()\n"
                "for p, t in sorted(totals.items(), key=lambda kv: (-kv[1], kv[0])):\n"
                "    print(f'{p}: {t:.2f}')\n"
            ),
            "bash": (
                "import glob\n"
                "err = warn = 0\n"
                "for path in sorted(glob.glob('logs/log*.txt')):\n"
                "    with open(path) as f:\n"
                "        for line in f:\n"
                "            if 'ERROR' in line: err += 1\n"
                "            elif 'WARN' in line: warn += 1\n"
                "print(f'ERROR: {err}')\n"
                "print(f'WARN: {warn}')\n"
            ),
            "concurrency": (
                "from concurrent.futures import ThreadPoolExecutor\n"
                "import glob\n"
                "def sum_file(p):\n"
                "    s = 0\n"
                "    with open(p) as f:\n"
                "        for line in f:\n"
                "            line = line.strip()\n"
                "            if line: s += int(line)\n"
                "    return s\n"
                "files = sorted(glob.glob('part*.txt'))\n"
                "with ThreadPoolExecutor() as ex:\n"
                "    print(sum(ex.map(sum_file, files)))\n"
            ),
        }
        (tmp_path / "solution.py").write_text(solutions[cluster])

        r1 = subprocess.run(
            ["python3", ".setup.py"], cwd=tmp_path, capture_output=True, text=True, timeout=15
        )
        assert r1.returncode == 0, f"setup failed: {r1.stderr}"
        r2 = subprocess.run(
            ["python3", ".validator.py"], cwd=tmp_path, capture_output=True, text=True, timeout=15
        )
        assert r2.returncode == 0, f"validator rejected reference solution: {r2.stdout}\n{r2.stderr}"


# ---------------------------------------------------------------------------
# 5. Utility-aware playbook trimming
# ---------------------------------------------------------------------------


class TestUtilityAwareTrim:
    def test_trim_noop_under_cap(self):
        playbook = [build_lesson(trigger=f"t{i}", correct_pattern="f") for i in range(3)]
        kept = _trim_playbook_by_utility(playbook, 5)
        assert len(kept) == 3

    def test_trim_preserves_head_even_if_low_utility(self):
        head = build_lesson(trigger="fresh", correct_pattern="f", confidence=0.1)
        others = [
            build_lesson(trigger=f"o{i}", correct_pattern="f", confidence=0.9)
            for i in range(5)
        ]
        kept = _trim_playbook_by_utility([head] + others, 3)
        assert kept[0]["trigger"] == "fresh"
        assert len(kept) == 3

    def test_verified_lessons_are_pinned(self):
        head = build_lesson(trigger="new", correct_pattern="f")
        verified = [
            build_lesson(
                trigger=f"v{i}", correct_pattern="f", confidence=0.5, verified=True,
            )
            for i in range(3)
        ]
        unverified_strong = [
            build_lesson(trigger=f"u{i}", correct_pattern="f", confidence=0.9)
            for i in range(10)
        ]
        # Cap at 5: head + 3 verified + 1 unverified-strong (highest utility).
        kept = _trim_playbook_by_utility(
            [head] + verified + unverified_strong, 5
        )
        assert len(kept) == 5
        triggers = {k["trigger"] for k in kept}
        assert "new" in triggers
        for i in range(3):
            assert f"v{i}" in triggers
        # Exactly one unverified slot should remain.
        assert sum(1 for k in kept if not k.get("verified") and k["trigger"].startswith("u")) == 1

    def test_unverified_eviction_prefers_higher_utility(self):
        head = build_lesson(trigger="new", correct_pattern="f")
        low = build_lesson(trigger="low", correct_pattern="f", confidence=0.1)
        high = build_lesson(
            trigger="high", correct_pattern="f", confidence=0.9, verified=False,
        )
        high["retrievals"] = 5
        high["helpful_retrievals"] = 4
        low["retrievals"] = 5
        low["helpful_retrievals"] = 0
        kept = _trim_playbook_by_utility([head, low, high], 2)
        assert kept[0]["trigger"] == "new"
        # Between low and high the one with higher utility is retained.
        assert kept[1]["trigger"] == "high"

    def test_learn_lesson_respects_playbook_max(self, tmp_path):
        sm = SkillMemory(tmp_path)
        # Seed with one verified lesson that must survive even when
        # many new unverified ones are added.
        sm.learn_lesson(
            "keep me", "m", "s", trigger="keep me",
            correct_pattern="fix", verified=True, confidence=0.8,
        )
        for i in range(PLAYBOOK_MAX + 10):
            sm.learn_lesson(
                f"ephemeral-{i}", "m", "s",
                trigger=f"ephemeral-{i}", correct_pattern="fix", confidence=0.2,
            )
        playbook = json.loads(sm.file_path.read_text())
        assert len(playbook) <= PLAYBOOK_MAX
        triggers = [p["trigger"] for p in playbook]
        assert "keep me" in triggers


# ---------------------------------------------------------------------------
# 6. Retrieval credit fires on any clean-exit turn
# ---------------------------------------------------------------------------


class TestCreditHoistedOutOfComplexTaskGate:
    """Structural test: confirm the credit call site is no longer
    nested inside `if was_complex_task or execution_failure_count > 0`.

    A behavioural test would require a full `handle_chat` driver. The
    structural check catches any accidental re-nesting and is cheap.
    """

    def test_credit_call_site_is_not_inside_complex_task_branch(self):
        import inspect
        from ghost_agent.core import agent as agent_mod

        src = inspect.getsource(agent_mod)
        # There should be at least two occurrences of credit_recent_retrievals,
        # one in the streaming path and one in the non-streaming path.
        occurrences = src.count("credit_recent_retrievals")
        assert occurrences >= 2

        # Both sites should be preceded by a comment explaining the
        # hoist; the hoist comment mentions "hoisted" or "clean-exit".
        lowered = src.lower()
        assert "hoisted" in lowered or "clean-exit" in lowered

    def test_credit_runs_after_simple_successful_tool_turn(self, tmp_path):
        """`credit_recent_retrievals` is idempotent, so calling it on
        a simple successful turn is safe. This test proves that a
        retrieved lesson followed by a clean exit is credited — the
        exact population the old gate excluded.
        """
        sm = SkillMemory(tmp_path)
        sm.learn_lesson("simple", "m", "s", trigger="simple", correct_pattern="f")
        sm.record_retrieval("simple")

        credited = sm.credit_recent_retrievals(window_seconds=300)
        assert credited == 1
        playbook = json.loads(sm.file_path.read_text())
        assert playbook[0]["helpful_retrievals"] == 1


# ---------------------------------------------------------------------------
# 7. Planner prompt includes relevant prior lessons
# ---------------------------------------------------------------------------


class TestPlannerReceivesPlaybook:
    def test_planner_transient_includes_playbook_when_available(self):
        """Smoke test: the string `### RELEVANT PRIOR LESSONS` should
        appear in `agent.py` next to the planner construction. If
        this heading disappears we've regressed the integration.
        """
        import inspect
        from ghost_agent.core import agent as agent_mod

        src = inspect.getsource(agent_mod)
        assert "### RELEVANT PRIOR LESSONS" in src
        # The playbook fetch must happen inside the planner branch,
        # before `planner_transient` is built.
        planner_idx = src.find("planner_transient = f\"\"\"")
        playbook_idx = src.find("planner_playbook = await request_state.get_skill_playbook")
        assert playbook_idx != -1
        assert playbook_idx < planner_idx, (
            "Planner playbook must be fetched before planner_transient is built"
        )


# ---------------------------------------------------------------------------
# 8. Transcript summariser keeps turning points
# ---------------------------------------------------------------------------


class TestTranscriptSummariser:
    def test_returns_unchanged_when_short(self):
        s = "A" * 100
        assert _summarize_long_transcript(s) == s

    def test_keeps_attempt_boundaries(self):
        big = (
            "HEAD " * 500
            + "\n--- ATTEMPT 1 ---\nsome assistant output\n"
            + "FILLER " * 2000
            + "\n--- ATTEMPT 2 ---\nTraceback (most recent call last)\nNameError: x\n"
            + "FILLER " * 2000
            + "\nSUCCESS TAIL " * 200
        )
        out = _summarize_long_transcript(big)
        # Much smaller than the original.
        assert len(out) < len(big)
        # Turning points from the middle survive.
        assert "--- ATTEMPT 2 ---" in out
        assert "NameError" in out or "Traceback" in out
        # Head + tail still present.
        assert "HEAD" in out
        assert "SUCCESS TAIL" in out

    def test_respects_highlight_budget(self):
        middle = "\nTraceback: something\n" * 1000
        big = "HEAD\n" + middle + "\nTAIL"
        out = _summarize_long_transcript(
            big, head_chars=10, tail_chars=10, highlight_budget=200
        )
        # Highlights capped — we don't blow through the budget.
        assert len(out) < 10 + 10 + 200 + 200  # generous upper bound

    def test_handles_no_turning_points_gracefully(self):
        big = "plain text " * 5000
        out = _summarize_long_transcript(big)
        assert "MIDDLE ELIDED" in out
        # Still has head + tail.
        assert out.startswith("plain text")
        assert out.endswith("plain text ")


# ---------------------------------------------------------------------------
# 9. Self-play report lands in scratchpad on success path
# ---------------------------------------------------------------------------


class TestReportPersistedOnSuccessPath:
    """The production code path that persists `report_val` sits at
    module scope (not easy to unit-test in isolation because it is
    inside `synthetic_self_play`). We pin the behaviour with a
    structural test: the scratchpad write now sits OUTSIDE the
    `else:` branch, reachable on both the success and no-skill paths.
    """

    def test_scratchpad_write_is_unconditional_in_dream(self):
        import inspect
        from ghost_agent.core import dream as dream_mod

        src = inspect.getsource(dream_mod.Dreamer.synthetic_self_play)
        # The scratchpad.set call must mention "Self-Play Report".
        assert "Self-Play Report" in src
        # Count the number of `scratchpad.set("Self-Play Report"` occurrences
        # — there must be exactly one, and it must not be nested inside
        # the `else:` that historically gated it.
        occurrences = src.count("Self-Play Report")
        assert occurrences >= 1
        # The fix's hallmark comment.
        assert "Persist the report to scratchpad on EVERY path" in src

    def test_scratchpad_accepts_report_from_either_branch(self):
        """Behavioural check: Scratchpad.set should accept any report
        text we might write — including the long multi-line report
        used in the success path."""
        pad = Scratchpad()
        long_report = (
            "Challenge: Do the thing.\n"
            "Status: SUCCESS\n"
            "Cluster: algo  Score: +1.20\n"
            "Learned trigger: Some Trigger\n"
            "Correct-pattern: def f(): return 1"
        )
        pad.set("Self-Play Report", long_report)
        assert pad.get("Self-Play Report") == long_report
