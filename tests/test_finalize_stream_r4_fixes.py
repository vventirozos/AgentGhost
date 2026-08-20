"""§ finalize/stream slice R4 (2026-08-19) — pins for the round's fixes.

R4's differential fuzzer (54,941 chunks) proved two byte-divergence classes
inside R3's incremental scrub — the name-blind close-end check (eaten
tool-call internals leaked to the client on the CANONICAL native shape) and
the flip-iteration double-append (ordinary " < " math prose duplicated) —
and exposed that the A-D3 pins were vacuous against the full-mechanism
revert, with the in-code comment citing a fuzz pin that did not exist. These
are the honest pins: the two repros, a sub-call-count discriminator the
naive revert cannot pass, and a REAL (bounded, seeded) differential fuzzer.
"""

import json
import random

import pytest

import ghost_agent.core.agent as agent_mod
from ghost_agent.core.agent import _scrub_tail_is_open, _MODULE_SCRUB_RE
from unittest.mock import AsyncMock

from tests.test_finalize_stream_pins import make_stream_agent
from tests.test_finalize_stream_r1_fixes import _client_text, _drive


# ── R4 D1: name-aware close-end (frozen-view invariant) ──────────────────────

class TestTailOpenNameAware:
    def test_inner_function_close_does_not_close_tool_call(self):
        buf = "pre <tool_call> internal args </function>"
        assert _scrub_tail_is_open(buf) is True

    def test_own_close_tag_closes(self):
        assert _scrub_tail_is_open("pre <tool_call>x</tool_call>") is False

    @pytest.mark.asyncio
    async def test_canonical_native_shape_no_leak(self):
        # the R4 reproduction: eaten internals must never reach the client.
        a = make_stream_agent()
        a._record_calibration_safe = AsyncMock()
        chunks = await _drive(a, [
            "pre ", "<tool_call>", " internal args </function>",
            " SECRET1 ", "SECRET2 ", "SECRET3", "</tool_call>",
            " visible tail."])
        text = _client_text(chunks)
        assert "SECRET" not in text
        assert "visible tail." in text
        assert text.startswith("pre ")


# ── R4 D2: flip iteration must not double-append ─────────────────────────────

class TestFlipNoDoubleAppend:
    @pytest.mark.asyncio
    async def test_math_less_than_not_duplicated(self):
        a = make_stream_agent()
        a._record_calibration_safe = AsyncMock()
        chunks = await _drive(a, ["Compare: 5 ", "<", " 7 holds. ", "Done."])
        text = _client_text(chunks)
        assert "<<" not in text
        assert text.count("7 holds.") == 1

    @pytest.mark.asyncio
    async def test_toolbox_word_not_duplicated(self):
        a = make_stream_agent()
        a._record_calibration_safe = AsyncMock()
        chunks = await _drive(a, ["Answer ", "<toolbox",
                                  " of parts. ", "Done."])
        text = _client_text(chunks)
        assert text.count("<toolbox") == 1
        assert "of parts." in text


# ── V1: mechanism pins the naive revert cannot pass ──────────────────────────

class _CountingPattern:
    """Wraps the scrub pattern; counts .sub calls (finditer left free)."""

    def __init__(self, real):
        self._real = real
        self.sub_calls = 0

    def sub(self, repl, s):
        self.sub_calls += 1
        return self._real.sub(repl, s)

    def finditer(self, s):
        return self._real.finditer(s)

    def search(self, s):
        return self._real.search(s)


class TestIncrementalMechanism:
    @pytest.mark.asyncio
    async def test_sub_runs_only_on_gt_deltas(self, monkeypatch):
        # 40 '>'-less deltas after one '>' delta: the incremental mechanism
        # re-subs ONLY on '>' deltas (2 here). The naive per-chunk revert
        # subs ~42 times — this pin is the structural discriminator the
        # vacuous R3 pins lacked.
        counter = _CountingPattern(_MODULE_SCRUB_RE)
        monkeypatch.setattr(agent_mod, "_MODULE_SCRUB_RE", counter)
        a = make_stream_agent()
        a._record_calibration_safe = AsyncMock()
        deltas = (["intro <tool_call>opened>"]      # '>' → recompute
                  + [f"body {i} " for i in range(40)]  # frozen, no sub
                  + ["</tool_call> tail>"])         # '>' → recompute
        await _drive(a, deltas)
        # § R5: lower bound too — an import-time alias that dodges the
        # monkeypatch would leave the counter at 0 while a naive revert runs
        # elsewhere; the pin must prove the counted pattern IS the live one.
        assert 1 <= counter.sub_calls <= 4, (
            f"sub ran {counter.sub_calls}x — incremental mechanism reverted "
            f"or the counted pattern is not the live one?")

    @pytest.mark.asyncio
    async def test_differential_fuzz_matches_naive(self):
        # THE fuzz pin the R3 comment claimed existed. Seeded, bounded:
        # the client text of the real incremental generator must equal the
        # naive always-recompute reference for every stream.
        rng = random.Random(4242)
        alphabet = [
            "plain words ", "more text. ", "<", ">", "`",
            "<tool_call>", "</tool_call>", "<function name=\"x\">",
            "</function>", "<tool_response>", "</tool_response>",
            "<tool", "_call>", " <think>", "</think> ", "5 < 7 ",
            "internal </function> data ", "`<tool_call>` mention ",
            # § R5 killer class: a malformed opener with no '>' of its own —
            # its close tag gets swallowed by [^>]*> and the match ends via
            # the \Z arm; R4's alphabet could not generate this shape.
            '<tool_call id="1"', "</tool_call>", "<function name=",
        ]

        def naive_client(deltas):
            full = ""
            emitted = 0
            out = []
            seen_lt = False
            for d in deltas:
                full += d
                if not seen_lt and "<" in full:
                    seen_lt = True
                view = (_MODULE_SCRUB_RE.sub("", full) if seen_lt else full)
                safe = agent_mod._emit_safe_end(view, emitted)
                piece = view[emitted:safe]
                if piece:
                    out.append(piece)
                    emitted = safe
            # end-flush
            view = _MODULE_SCRUB_RE.sub("", full)
            safe = agent_mod._emit_safe_end(view, emitted)
            piece = view[emitted:safe]
            if piece:
                out.append(piece)
            return "".join(out)

        for stream_no in range(60):
            n = rng.randint(3, 14)
            deltas = [rng.choice(alphabet) for _ in range(n)]
            # random re-splits across chunk boundaries
            joined = "".join(deltas)
            cuts = sorted(rng.sample(range(1, max(2, len(joined))),
                                     min(len(joined) - 1, rng.randint(1, 9))))
            deltas = [joined[a:b] for a, b in
                      zip([0] + cuts, cuts + [len(joined)])]
            deltas = [d for d in deltas if d]

            a = make_stream_agent()
            a._record_calibration_safe = AsyncMock()
            chunks = await _drive(a, deltas)
            got = _client_text(chunks)
            want = naive_client(deltas)
            if not want.strip():
                # the scrub consumed everything → the generator's deliberate
                # EMPTY-OUTPUT FALLBACK fires (a feature, not a divergence).
                assert (not got.strip()
                        or got.startswith("I prepared a tool call")), (
                    f"stream {stream_no}: empty-want got {got!r}")
            else:
                assert got == want, (
                    f"stream {stream_no} diverged\ndeltas={deltas!r}\n"
                    f"got ={got!r}\nwant={want!r}")


# ── R5: arm identity, not text property ──────────────────────────────────────

class TestTailOpenArmIdentity:
    def test_opener_swallowing_its_own_close_tag_is_still_open(self):
        # the R5 killer: no '>' before the close tag → the opener's [^>]*>
        # consumes it, the match ends via \Z (eating) — a text-suffix check
        # read this as "closed".
        assert _scrub_tail_is_open('<tool_call id="1"</tool_call>') is True

    def test_properly_closed_block_reads_closed(self):
        assert _scrub_tail_is_open(
            '<tool_call id="1">body</tool_call>') is False

    @pytest.mark.asyncio
    async def test_swallowed_close_shape_no_client_leak(self):
        from unittest.mock import AsyncMock as _AM
        a = make_stream_agent()
        a._record_calibration_safe = _AM()
        chunks = await _drive(a, [
            "pre ", '<tool_call id="1"', "</tool_call>",
            " SECRET_A ", "SECRET_B", " more>", " tail."])
        text = _client_text(chunks)
        assert "SECRET" not in text
        assert text.startswith("pre ")
