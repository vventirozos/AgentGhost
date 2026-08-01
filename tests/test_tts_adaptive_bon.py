"""Tests for trajectory-level TTS — adaptive best-of-N (§4F Phase 3b).

Contracts under test: env-gated OFF by default; fires only on the
verifier wobble band; the ORIGINAL answer is always candidate 1 and every
failure mode (no alternatives, judge error, unparseable/out-of-range
verdict) resolves to the original — the pass can never make things worse
by construction.
"""

import asyncio
import json

import pytest

from ghost_agent.core import tts


class _VR:
    def __init__(self, verdict, confidence):
        self.verdict = verdict
        self.confidence = confidence


class TestGates:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("GHOST_TTS_ADAPTIVE_BON", raising=False)
        assert not tts.adaptive_bon_enabled()

    def test_enabled(self, monkeypatch):
        monkeypatch.setenv("GHOST_TTS_ADAPTIVE_BON", "1")
        assert tts.adaptive_bon_enabled()

    def test_k_clamped(self, monkeypatch):
        monkeypatch.setenv("GHOST_TTS_BON_K", "99")
        assert tts.bon_k() == 4
        monkeypatch.setenv("GHOST_TTS_BON_K", "junk")
        assert tts.bon_k() == 2


class TestWobbleBand:
    def test_uncertain_triggers(self):
        assert tts.wobble_band(_VR("UNCERTAIN", 0.9))

    def test_soft_refuted_triggers(self):
        assert tts.wobble_band(_VR("REFUTED", 0.5))

    def test_hard_refuted_does_not(self):
        # ≥0.7 REFUTED belongs to the repair path — never both mechanisms.
        assert not tts.wobble_band(_VR("REFUTED", 0.85))

    def test_confirmed_does_not(self):
        assert not tts.wobble_band(_VR("CONFIRMED", 0.4))

    def test_no_verdict_is_not_a_signal(self):
        assert not tts.wobble_band(None)

    def test_enum_style_verdicts(self):
        # Live VerifyResult carries an Enum whose str() endswith the name.
        from ghost_agent.core.verifier import VerifyVerdict
        assert tts.wobble_band(_VR(VerifyVerdict.UNCERTAIN, 0.9))
        assert not tts.wobble_band(_VR(VerifyVerdict.CONFIRMED, 0.9))


class TestJudgeParsing:
    def test_valid_verdict(self):
        raw = '{"winner": 2, "why": "candidate 2 cites the evidence"}'
        assert tts.parse_judge_verdict(raw, 3) == 1

    def test_prose_wrapped_json(self):
        raw = 'Sure! {"winner": 1, "why": "x"} hope that helps'
        assert tts.parse_judge_verdict(raw, 2) == 0

    def test_out_of_range_rejected(self):
        assert tts.parse_judge_verdict('{"winner": 5, "why": ""}', 3) is None
        assert tts.parse_judge_verdict('{"winner": 0, "why": ""}', 3) is None

    def test_garbage_rejected(self):
        assert tts.parse_judge_verdict("I like candidate two", 3) is None
        assert tts.parse_judge_verdict("", 3) is None

    def test_prompt_contains_all_candidates_and_request(self):
        p = tts.build_judge_prompt("count the files", ["answer A", "answer B"])
        assert "CANDIDATE 1" in p and "CANDIDATE 2" in p
        assert "count the files" in p
        assert "1-2" in p

    def test_excerpt_collapses_code_and_truncates(self):
        long = "x" * 5000 + "```\ncode\n```" + "y" * 5000
        e = tts.candidate_excerpt(long)
        assert len(e) < 2000
        assert "[code block]" not in e or "```" not in e
        assert "truncated" in e


class TestJudgePayload:
    def test_shape_is_nothink_bounded_single_line(self):
        p = tts.judge_payload("PROMPT")
        assert p["chat_template_kwargs"] == {"enable_thinking": False}
        assert p["messages"][0]["content"].endswith("/no_think")
        assert p["stop"] == ["\n"]
        assert p["max_tokens"] <= 512
        assert p["stream"] is False

    def test_agent_source_carries_no_disable_thinking_switch_for_bon(self):
        """The guard in test_self_play_redesign allows the disable-thinking
        switch in agent.py only at the known bounded side-call sites
        (trivial fast-path + the two System 3 pivot calls, 2026-08-01).
        The BoN judge's switch must live in tts.py — never migrate it
        back."""
        from pathlib import Path
        import ghost_agent.core.agent as agent_mod
        src = Path(agent_mod.__file__).read_text()
        assert src.count(
            '"chat_template_kwargs": {"enable_thinking": False}') <= 3


def _run(coro):
    return asyncio.run(coro)


class TestAdaptiveBon:
    def _gen(self, texts):
        async def gen(i):
            return texts[i - 1] if i - 1 < len(texts) else None
        return gen

    def _judge(self, winner, why="better"):
        async def judge(prompt):
            return json.dumps({"winner": winner, "why": why})
        return judge

    def test_winner_substituted(self):
        text, meta = _run(tts.adaptive_bon(
            self._gen(["alt one", "alt two"]), self._judge(2),
            "req", "original", k=2))
        assert text == "alt one"
        assert meta["substituted"] is True
        assert meta["candidates"] == 3

    def test_original_kept_when_judge_picks_it(self):
        text, meta = _run(tts.adaptive_bon(
            self._gen(["alt one"]), self._judge(1), "req", "original", k=1))
        assert text == "original"
        assert meta["substituted"] is False

    def test_no_distinct_alternatives_keeps_original(self):
        text, meta = _run(tts.adaptive_bon(
            self._gen(["original", None]), self._judge(2),
            "req", "original", k=2))
        assert text == "original"
        assert meta["candidates"] == 1
        assert "no distinct" in meta["why"]

    def test_generation_failure_shrinks_pool(self):
        async def gen(i):
            if i == 1:
                raise RuntimeError("upstream hiccup")
            return "alt"
        text, meta = _run(tts.adaptive_bon(
            gen, self._judge(2), "req", "original", k=2))
        assert text == "alt"
        assert meta["candidates"] == 2

    def test_judge_failure_keeps_original(self):
        async def judge(prompt):
            raise RuntimeError("worker down")
        text, meta = _run(tts.adaptive_bon(
            self._gen(["alt"]), judge, "req", "original", k=1))
        assert text == "original"
        assert meta["why"] == "judge call failed"

    def test_unparseable_verdict_keeps_original(self):
        async def judge(prompt):
            return "candidate two looks nice"
        text, meta = _run(tts.adaptive_bon(
            self._gen(["alt"]), judge, "req", "original", k=1))
        assert text == "original"
        assert meta["why"] == "judge verdict unparseable"
