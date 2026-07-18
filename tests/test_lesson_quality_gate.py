"""Lesson-quality gate (2026-07-16).

The autonomous loops (dream, self-play) were writing OBSERVATIONS ("On a
regex_parse task that has a familiar shape…", "The user prefers ripgrep") as
mistake-less pseudo-lessons that were 28% of playbook retrievals. The gate at
the learn_lesson write chokepoint drops mistake-less entries whose solution
isn't an actionable rule, while keeping real mistake-and-fix lessons and legit
imperative rules.
"""
from pathlib import Path

import pytest

from ghost_agent.memory.lesson_quality import (
    is_actionable_lesson, _is_actionable_heuristic, _is_mistake_less,
)
from ghost_agent.memory.skills import SkillMemory


class TestIsActionableHeuristic:
    @pytest.mark.parametrize("text", [
        "Always use absolute paths in Docker.",
        "When executing code, always handle runtime errors.",
        "Verify the publish path ownership before retrying.",
        "Never double-signal llama-server on shutdown.",
    ])
    def test_imperative_rules_pass(self, text):
        assert _is_actionable_heuristic(text)

    @pytest.mark.parametrize("text", [
        "The user prefers ripgrep over grep for searches.",
        "The agent is capable of running Python code.",
        "On a regex_parse task that has a familiar shape, a structurally different one appears.",
        "When playing live chess against Vasilis, provide continuous coaching commentary.",  # 'when' but no modal
        "System service management tasks frequently involve restarting services.",
    ])
    def test_observations_rejected(self, text):
        assert not _is_actionable_heuristic(text)


class TestIsActionableLesson:
    def test_real_mistake_and_fix_always_kept(self):
        # A genuine correction passes regardless of solution phrasing.
        assert is_actionable_lesson(
            mistake="retried the copy blindly",
            solution="The fix is to check ownership.",  # not imperative-shaped
            task="deploy fails with EACCES")

    def test_mistake_less_observation_dropped(self):
        assert not is_actionable_lesson(
            mistake="none",
            solution="The user prefers ripgrep over grep",
            task="search preference")

    def test_mistake_less_actionable_rule_kept(self):
        assert is_actionable_lesson(
            mistake="none",
            solution="Always use absolute paths in Docker.",
            task="Always use absolute paths in Docker.")   # solution==task is fine for a rule

    def test_short_rule_with_solution_equal_task_kept(self):
        # The dream loop stores task=solution[:80]; equality must NOT reject.
        assert is_actionable_lesson("none", "Use a lock around the write.",
                                    "Use a lock around the write.")

    def test_mistake_less_helpers(self):
        assert _is_mistake_less("none") and _is_mistake_less("") and _is_mistake_less(None)
        assert not _is_mistake_less("retried blindly")


class TestGateAtWriteChokepoint:
    def _sm(self, tmp_path):
        return SkillMemory(Path(tmp_path))

    def test_non_actionable_lesson_not_written(self, tmp_path):
        sm = self._sm(tmp_path)
        sm.learn_lesson(
            task="familiar regex shape",
            mistake="none",
            solution="On a regex_parse task that has a familiar shape, a different one appears",
            source="self_play")
        pb = sm._load_playbook()
        assert pb == [], f"non-actionable observation was written: {pb}"

    def test_user_preference_observation_not_written(self, tmp_path):
        sm = self._sm(tmp_path)
        sm.learn_lesson(task="pref", mistake="none",
                        solution="The user prefers ripgrep over grep", source="dream")
        assert sm._load_playbook() == []

    def test_real_lesson_is_written(self, tmp_path):
        sm = self._sm(tmp_path)
        sm.learn_lesson(
            task="deploy fails with EACCES",
            mistake="retrying the copy blindly",
            solution="verify the publish path ownership first",
            source="reflection")
        pb = sm._load_playbook()
        assert len(pb) == 1
        assert "publish path" in (pb[0].get("solution") or "")

    def test_actionable_dream_rule_is_written(self, tmp_path):
        sm = self._sm(tmp_path)
        sm.learn_lesson(
            task="When executing code, always anticipate runtime errors",
            mistake="none",
            solution="When executing code, always anticipate runtime errors",
            trigger="When executing code, always anticipate runtime errors",
            source="dream")
        assert len(sm._load_playbook()) == 1


def test_gate_shared_by_dream_import():
    # dream.py re-exports _is_actionable_heuristic from the shared leaf module —
    # its two call sites (heuristics loop, episode consolidation) still work.
    from ghost_agent.core.dream import _is_actionable_heuristic as dh
    from ghost_agent.memory.lesson_quality import _is_actionable_heuristic as lq
    assert dh is lq


class TestConversationalTrigger:
    """2026-07-18: mistake-bearing lessons used to bypass the gate entirely,
    letting the overnight REM cycle write corrections keyed on raw user chat
    ("proceed with the next task", "it still does the same…"). A trigger is
    the lesson's retrieval key — a chat fragment matches no future query."""

    @pytest.mark.parametrize("trigger", [
        "proceed with the next task.",
        "it still does the same. the game never starts, notify me in slack when it's fixed",
        "ok now try again",
        "continue",
        "that didn't work, do it again please",
    ])
    def test_chat_fragment_triggers_reject_even_with_real_mistake(self, trigger):
        assert not is_actionable_lesson(
            mistake="verifier refuted (late): claim not aligned with request",
            solution="Confirm the concrete next step before advancing.",
            task=trigger)

    @pytest.mark.parametrize("trigger", [
        "deploy fails with EACCES",
        "Uncaught TypeError: enemyManager.loadLevel is not a function",
        "When fixing a game, verify the existence of core project files",
        "Complex SQL aggregation and string formatting",
        # "you"-phrasing is legitimate rule language addressed to the agent
        "verify core files exist before you edit them",
        # user-QUESTION triggers are legitimate recurring retrieval keys —
        # the paraphrase dedup counts on them (bare pronouns/please must
        # NOT reject)
        "How do I parse JSON?",
        "Please parse JSON!",
    ])
    def test_situation_keys_still_pass_with_real_mistake(self, trigger):
        assert is_actionable_lesson(
            mistake="edited a file that did not exist",
            solution="List the project directory first.",
            task=trigger)

    def test_empty_trigger_still_passes_mistake_bearing_lesson(self):
        # pre-2026-07-18 behaviour preserved: no trigger ≠ conversational
        assert is_actionable_lesson(
            mistake="real mistake", solution="real fix", task="")

    def test_long_resume_scenario_trigger_not_rejected_as_turn_command(self):
        # turn-command rejection only applies to SHORT bare commands
        trigger = ("resume of a long-running batch job after a crash requires "
                   "replaying the journal before accepting new work")
        assert is_actionable_lesson(
            mistake="accepted new work before replaying the journal",
            solution="Replay the journal first.",
            task=trigger)
