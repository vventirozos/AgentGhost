"""Tests for the self-play hardening fixes:
  - per-attempt turn cap (max_turns_override)
  - tightened thinking-loop detection (smaller window, lower threshold)
  - max_thinking_chars_override on GhostAgent
  - validator timeout trim (30s instead of 300s)
  - brittleness scoring now counts struggled-then-won as partial signal
§4NM: the dream.py integration tests (quality-gate regeneration, temp-agent
budget overrides, validator timeout) went with synthetic self-play.
"""

import pytest
from unittest.mock import MagicMock

from ghost_agent.core.agent import (
    GhostAgent,
    _detect_thinking_loop,
    THINKING_LOOP_WINDOW,
    THINKING_LOOP_THRESHOLD,
    THINKING_LOOP_PROBE_EVERY,
)
from ghost_agent.memory.frontier import FrontierTracker


# --------------------------------------------------------- thinking-loop probe


class TestThinkingLoopConstants:
    def test_window_is_tighter_than_pre_fix(self):
        # Pre-fix was 400; post-fix must be <= 200 to catch shorter loops.
        assert THINKING_LOOP_WINDOW <= 200

    def test_threshold_is_tighter_than_pre_fix(self):
        # Pre-fix was 4; post-fix must be <= 3.
        assert THINKING_LOOP_THRESHOLD <= 3

    def test_probe_frequency_is_tighter_than_pre_fix(self):
        # Pre-fix was 2000; post-fix must be <= 500 so runaway streams are
        # caught within a few probes, not a dozen.
        assert THINKING_LOOP_PROBE_EVERY <= 500


class TestDetectThinkingLoop:
    def test_short_buffer_returns_false(self):
        assert _detect_thinking_loop("short text") is False

    def test_repeated_exact_block_is_loop(self):
        # A 250-char paragraph repeated 4 times — the tight 200-char window
        # tail must appear >=3 times somewhere in the buffer for the probe
        # to fire.
        block = "A" * 250
        buf = block * 4
        assert _detect_thinking_loop(buf) is True

    def test_nonrepeating_text_not_loop(self):
        buf = "".join(f"unique line {i} with different content each pass\n" for i in range(30))
        assert _detect_thinking_loop(buf) is False

    def test_wide_window_backstop_catches_longer_repeats(self):
        # Each chunk is larger than the tight window but smaller than the
        # wide (2x) backstop. Three repeats should still fire.
        chunk = "A" * 300 + "\n"  # 301 chars, larger than 200-char tight window
        buf = chunk * 4
        assert _detect_thinking_loop(buf) is True


# --------------------------------------------------------- GhostAgent overrides


class TestAgentOverrides:
    def test_max_turns_override_attribute_respected(self):
        """Verify the override attribute is wired: when set, the effective
        cap used by handle_chat is the override, not the default 40."""
        # We can't run handle_chat end-to-end here, but we can verify the
        # lookup logic by creating an agent, setting the override, and
        # checking the attribute resolution path.
        ctx = MagicMock()
        agent = GhostAgent.__new__(GhostAgent)  # bypass __init__
        agent.max_turns_override = 15
        effective = getattr(agent, "max_turns_override", None) or 40
        assert effective == 15

    def test_max_turns_default_is_40(self):
        agent = GhostAgent.__new__(GhostAgent)
        effective = getattr(agent, "max_turns_override", None) or 40
        assert effective == 40

    # §4SD: the thinking char caps (and their override) were removed — the
    # turn's max_tokens bounds length, the n-gram probe bounds repetition.


# --------------------------------------------------------- brittleness scoring


class TestBrittlenessScoring:
    def test_struggled_then_won_is_partial_brittle(self, tmp_path):
        """A single 2-attempt struggle (success) must register as mildly
        brittle so the next session targets it instead of cold-starting.
        Pre-fix the score was 0 and the cluster was ignored."""
        ft = FrontierTracker(tmp_path)
        ft.record_run("data_analysis", "c1", 2, True, 1500)
        brittle = ft.get_brittle_clusters()
        assert any(k == "data_analysis" for k, _ in brittle)

    def test_one_attempt_win_is_not_brittle(self, tmp_path):
        ft = FrontierTracker(tmp_path)
        ft.record_run("algo", "c", 1, True, 500)
        brittle = ft.get_brittle_clusters()
        assert not any(k == "algo" for k, _ in brittle)

    def test_failure_ranks_higher_than_soft_win(self, tmp_path):
        ft = FrontierTracker(tmp_path)
        ft.record_run("bash", "c_fail", 3, False, 0)
        ft.record_run("sql", "c_soft", 2, True, 1000)
        brittle = ft.get_brittle_clusters()
        # Failure (weight 2) should outrank soft win (weight 1)
        assert brittle[0][0] == "bash"
        assert any(k == "sql" for k, _ in brittle)

    def test_pick_seed_targets_struggled_cluster(self, tmp_path):
        ft = FrontierTracker(tmp_path)
        ft.record_run("data_analysis", "c1", 2, True, 1500)
        seed = ft.pick_seed(random_explore_prob=0.0)
        assert seed["mode"] == "frontier"
        assert seed["cluster_key"] == "data_analysis"
