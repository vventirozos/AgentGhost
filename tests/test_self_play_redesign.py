"""Tests for the self-play redesign bundle — the parts that outlived it.

Originally covered 8 items; §4NM removed synthetic self-play, so the
cycle timeout (#2), worker prompt (#4), challenge-gen speedup (#6),
native-tools opt-in (#7) and the challenge template library (#8) tests
are gone. What remains pins the main agent:

  #1  `_detect_tool_call_loop` — fail-fast detector for decoder-collapse
      (>N unclosed `<tool_call>` openings).
  #3  `DEFAULT_TOOL_TURN_MAX_TOKENS` paired with the detector — the
      raise to 16384 is only safe because the detector bounds collapse.
  #5  `stop=[...]` sequences in the tool-turn payload — hard stop on
      back-to-back tool_call.
  plus the guard that the main tool turn never disables thinking. (The
  SYNTHETIC_CHALLENGE_PROMPT worked-example pins and the unused
  NON_THINKING_REASONING_PARAMS went with those constants in the post-§4NN
  tidy-up; NON_THINKING_GENERAL_PARAMS went with its last caller, the
  greeting fast path, §4RP-c.)
"""

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
AGENT_SRC = REPO_ROOT / "src" / "ghost_agent" / "core" / "agent.py"


# ---------------------------------------------------------------------------
# #1  _detect_tool_call_loop
# ---------------------------------------------------------------------------


class TestToolCallLoopDetector:
    def test_healthy_single_tool_call_not_flagged(self):
        from ghost_agent.core.agent import _detect_tool_call_loop
        s = "<tool_call><function name=\"x\"></function></tool_call>"
        assert _detect_tool_call_loop(s) is False

    def test_healthy_three_balanced_tool_calls_not_flagged(self):
        from ghost_agent.core.agent import _detect_tool_call_loop
        s = ("<tool_call></tool_call>" * 3) + "some prose"
        assert _detect_tool_call_loop(s) is False

    def test_one_open_in_progress_not_flagged(self):
        """Streaming mid-generation — one open, no close yet. That's
        normal during a legitimate long tool_call body."""
        from ghost_agent.core.agent import _detect_tool_call_loop
        s = "<tool_call><function name=\"execute\"><parameter name=\"content\">still writing..."
        assert _detect_tool_call_loop(s) is False

    def test_eleven_unclosed_opens_is_flagged(self):
        """The collapse signature: N opens, zero closes. 11 > threshold 10."""
        from ghost_agent.core.agent import _detect_tool_call_loop
        s = "<tool_call>" * 11
        assert _detect_tool_call_loop(s) is True

    def test_observed_8135_opens_is_flagged(self):
        """The exact shape from the production log."""
        from ghost_agent.core.agent import _detect_tool_call_loop
        s = "<tool_call>" * 8135
        assert _detect_tool_call_loop(s) is True

    def test_opens_equal_to_closes_even_when_many_is_not_flagged(self):
        from ghost_agent.core.agent import _detect_tool_call_loop
        s = "<tool_call></tool_call>" * 100
        assert _detect_tool_call_loop(s) is False

    def test_empty_string_is_safe(self):
        from ghost_agent.core.agent import _detect_tool_call_loop
        assert _detect_tool_call_loop("") is False

    def test_threshold_constant_is_small(self):
        """Guard: the threshold must be small enough to fire in seconds,
        not minutes. Anything over ~30 defeats the purpose."""
        from ghost_agent.core.agent import TOOL_CALL_LOOP_THRESHOLD
        assert 5 <= TOOL_CALL_LOOP_THRESHOLD <= 30

    def test_stream_path_wires_the_detector(self):
        src = AGENT_SRC.read_text()
        assert "_detect_tool_call_loop(full_content)" in src
        assert '"Tool-Call Loop"' in src


class TestMaxTokensStillReasonable:
    def test_default_is_16384(self):
        from ghost_agent.core.agent import DEFAULT_TOOL_TURN_MAX_TOKENS
        assert DEFAULT_TOOL_TURN_MAX_TOKENS == 16384

    def test_detector_fires_well_before_cap(self):
        """The whole point of tiering: the detector must bound the
        generation well short of the max_tokens cap, so a collapse at
        cap = 16384 tokens can't actually run to 16384 tokens of garbage."""
        from ghost_agent.core.agent import (
            TOOL_CALL_LOOP_THRESHOLD,
            DEFAULT_TOOL_TURN_MAX_TOKENS,
        )
        # Each <tool_call> is ~4 tokens. 10 unclosed opens × 4 = ~40
        # tokens of collapse before detector fires. Cap is 16384.
        approx_tokens_before_fire = TOOL_CALL_LOOP_THRESHOLD * 4
        assert approx_tokens_before_fire * 100 < DEFAULT_TOOL_TURN_MAX_TOKENS


class TestStopSequences:
    def test_collapse_detector_still_present(self):
        src = AGENT_SRC.read_text()
        assert "_detect_tool_call_loop" in src
        assert "TOOL_CALL_LOOP_THRESHOLD" in src

    def test_back_to_back_tool_call_stops_removed(self):
        """These stops must NOT be in the main tool-turn payload — they
        would silently truncate parallel tool calls. (Occurrences are
        fine in the decoder-collapse detector regex, but not as a
        payload `stop` sequence.)"""
        src = AGENT_SRC.read_text()
        # The exact payload `stop` block that previously carried these
        # must be gone. Match the `"stop": [...]` container specifically
        # to avoid false-positives against the detector comments.
        import re as _re
        payload_stops = _re.findall(
            r'"stop":\s*\[\s*"</tool_call>\\n<tool_call>"',
            src,
        )
        assert not payload_stops, (
            "Payload `stop` sequence for back-to-back tool_calls must be "
            "removed — it truncates legitimate parallel calls."
        )


# ---------------------------------------------------------------------------
# Thinking-mode vs non-thinking-mode plumbing for Qwen3.6
#
# The model card for HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
# publishes four distinct sampling profiles — two for thinking mode, two
# for non-thinking mode — and exposes both a portable `/no_think`
# soft-switch and a chat_template_kwargs hard-switch. Before this pass
# the codebase only wired the thinking-mode profiles, so callers who
# didn't actually need reasoning (structured-XML emission) were burning
# 100+ seconds of <think> preamble per call.
# ---------------------------------------------------------------------------


class TestNonThinkingSamplingProfiles:
    def test_thinking_mode_profiles_unchanged(self):
        """Regression guard: adding non-thinking profiles must not
        perturb the existing thinking-mode profiles that the main
        agent and self-play worker rely on."""
        from ghost_agent.core.agent import (
            CODING_SAMPLING_PARAMS,
            GENERAL_SAMPLING_PARAMS,
        )
        assert CODING_SAMPLING_PARAMS == {
            "temperature": 1.0,     # §4RG: Qwen3.8 card thinking-mode value
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0,
            "presence_penalty": 0,
        }
        assert GENERAL_SAMPLING_PARAMS == {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0,
            "presence_penalty": 0,     # §4RG-b: card thinking-mode value (was the instruct 1.5)
        }


class TestMainAgentKeepsThinking:
    """§4NM: the challenge-generation non-thinking switches went with
    synthetic self-play; only the negative guard on the MAIN agent path
    remains."""

    def test_main_agent_tool_turn_payload_does_NOT_disable_thinking(self):
        """Negative guard: the main agent's tool-turn payload MUST
        keep thinking mode on — that's where reasoning actually helps.
        Only challenge generation opts out. A future change that
        copies the non-thinking switches into the main agent path
        would silently degrade the agent's reasoning quality."""
        src = AGENT_SRC.read_text()
        # The disable-thinking hard-switch is legitimately present in
        # agent.py only in BOUNDED side-call payloads: the trivial
        # fast-path (greeting) and, since 2026-08-01 (req 50855398), the
        # two System 3 crisis-pivot calls (generator/evaluator) — a pivot
        # that thinks blows its own 120s wall and ReadTimeouts during the
        # crisis. The guard is that the switch never gets copied into the
        # MAIN TOOL-TURN loop: cap the count at those known sites and
        # require every occurrence to sit in a /no_think-marked or
        # trivial-path block.
        needle = '"chat_template_kwargs": {"enable_thinking": False}'
        occurrences = src.count(needle)
        assert occurrences <= 3, (
            f"disable-thinking switch appears {occurrences}x — it must stay "
            "confined to the trivial fast-path and the System 3 pivot, "
            "never the main tool turn."
        )
        start = 0
        while True:
            idx = src.find(needle, start)
            if idx == -1:
                break
            preceding = src[max(0, idx - 1200):idx]
            assert ("trivial" in preceding.lower()
                    or "/no_think" in preceding), (
                "disable-thinking switch found outside the known bounded "
                "side-call blocks")
            start = idx + 1
