"""§4EZ (2026-09-06) — origin colour families on the operator stream.

Before: the tag / frame colour was one hash over twelve 256-colour codes,
so a self-play turn, a bench attempt and the operator's own request drew
from the same palette, and SYSTEM (`**`) lines had no colour at all. The
only origin signal was the frame's trailing `· sim` word.

After: the HUE of the tag is the population — blues for user, violets for
sim, ambers for bench, one green for SYSTEM, greys for a request that
declared no origin — and the shade within a family is the per-request hash
that keeps concurrent same-population requests separable. The origin
reaches non-frame lines through `request_origin_context`, set by
`handle_chat` beside the request id from the SAME `turn_origin` value the
BEGIN frame stamps.

Every pin here names the world where it fails: the pre-§4EZ module (no
`request_origin_context`, no `origin=` on `_req_color`, "" for SYSTEM, one
shared palette) fails all of the family pins; a resolver that ignores the
contextvar fails the before-BEGIN / after-END pin; a `handle_chat` that
does not set or reset the contextvar fails the executed turn pin.
"""
from __future__ import annotations

import re
from unittest.mock import AsyncMock

import pytest

from ghost_agent.utils import logging as glog
from ghost_agent.utils.logging import (
    pretty_log, request_id_context, request_origin_context, _req_color,
)

_CODE_RE = re.compile(r"\x1b\[38;5;(\d+)m")


def _codes(line: str) -> set[int]:
    return {int(c) for c in _CODE_RE.findall(line)}


def _family(origin: str) -> set[int]:
    return set(glog._ORIGIN_PALETTES[origin])


_ALL_REQUEST_FAMILIES = (
    set(glog._ORIGIN_PALETTES["user"])
    | set(glog._ORIGIN_PALETTES["sim"])
    | set(glog._ORIGIN_PALETTES["bench"])
)


@pytest.fixture
def colour_on(monkeypatch):
    """Force colour regardless of the sink (tests run captured, and the
    operator's FORCE_COLOR export is exactly the env this must not depend
    on)."""
    monkeypatch.setattr(glog, "_USE_COLOR", True)
    monkeypatch.setattr(glog, "_COLLAPSE_STATE", None)
    yield
    glog._COLLAPSE_STATE = None


@pytest.fixture
def quiet_mirror(monkeypatch):
    class _Spy:
        def log(self, *a, **k):
            pass
    monkeypatch.setattr(glog, "_MIRROR_LOGGER", _Spy())


@pytest.fixture
def _clean_req_state():
    yield
    with glog._REQ_STATE_LOCK:
        glog._REQ_STATE.clear()


# ── the table itself ────────────────────────────────────────────────────────

class TestPaletteTable:
    def test_families_are_pairwise_disjoint(self):
        fams = {k: set(v) for k, v in glog._ORIGIN_PALETTES.items()}
        names = sorted(fams)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                assert not (fams[a] & fams[b]), (
                    f"{a} and {b} share a colour code — a shared code is a "
                    "line the operator cannot classify by hue")

    def test_system_and_unclassified_sit_outside_every_request_family(self):
        assert glog._SYSTEM_COLOR not in _ALL_REQUEST_FAMILIES
        assert not (set(glog._UNCLASSIFIED_PALETTE) & _ALL_REQUEST_FAMILIES)
        assert glog._SYSTEM_COLOR not in set(glog._UNCLASSIFIED_PALETTE)

    def test_the_populations_the_frames_stamp_all_have_a_family(self):
        # The frame vocabulary (LOG-4) is user / sim / bench, plus probe
        # since §4FB (a diagnostic sent through the user path); a population
        # the frame can name but the palette cannot colour would render
        # grey and read as "unclassified" — a lie.
        assert set(glog._ORIGIN_PALETTES) == {"user", "sim", "bench", "probe"}

    def test_every_family_has_more_than_one_shade(self):
        # Shade-within-family is what keeps two concurrent sim turns
        # separable; a one-shade family silently drops that grouping.
        for name, fam in glog._ORIGIN_PALETTES.items():
            assert len(set(fam)) >= 2, name


# ── _req_color: the resolver ────────────────────────────────────────────────

class TestReqColor:
    def test_system_lines_get_their_own_colour(self, colour_on):
        # Pre-§4EZ: "" — the `**` stream was the only uncoloured one.
        assert _codes(_req_color("SYSTEM")) == {glog._SYSTEM_COLOR}

    @pytest.mark.parametrize("origin", ["user", "sim", "bench"])
    def test_explicit_origin_maps_every_request_into_its_family(
            self, colour_on, origin):
        ids = [f"{origin[:1]}{i:07x}" for i in range(40)] + [
            "a8a93a27", "eb30aa56", "bench-0123456789", "sub-77aa", "SYS"]
        seen = set()
        for rid in ids:
            code = _codes(_req_color(rid, origin=origin))
            assert len(code) == 1 and code <= _family(origin), (rid, code)
            seen |= code
        # The hash still spreads requests across the family's shades.
        assert len(seen) >= 2, f"{origin}: every request got one shade"

    def test_same_request_same_shade_across_calls(self, colour_on):
        assert (_req_color("deadbeef", origin="user")
                == _req_color("deadbeef", origin="user"))

    def test_unclassified_request_is_grey_not_laundered_into_user(
            self, colour_on, _clean_req_state):
        tok = request_id_context.set("noorigin1")
        try:
            code = _codes(_req_color("noorigin1"))
        finally:
            request_id_context.reset(tok)
        assert code <= set(glog._UNCLASSIFIED_PALETTE), code
        assert not (code & _family("user"))

    def test_no_colour_at_all_when_colour_is_off(self, monkeypatch):
        monkeypatch.setattr(glog, "_USE_COLOR", False)
        assert _req_color("SYSTEM") == ""
        assert _req_color("abc", origin="sim") == ""
        assert _req_color("abc") == ""


# ── pretty_log: the lines the operator sees ─────────────────────────────────

class TestStreamLines:
    def _run_framed(self, capsys, rid: str, origin: str) -> list[str]:
        tok = request_id_context.set(rid)
        try:
            pretty_log("Before Frame", "x")
            pretty_log("Request Initialized", special_marker="BEGIN",
                       origin=origin)
            pretty_log("Inside", "payload")
            pretty_log("Section", special_marker="SECTION_START")
            pretty_log("Section", special_marker="SECTION_END")
            pretty_log("Request Finished", special_marker="END")
        finally:
            request_id_context.reset(tok)
        return [l for l in capsys.readouterr().out.splitlines() if l.strip()]

    @pytest.mark.parametrize("origin", ["user", "sim", "bench"])
    def test_every_line_inside_a_frame_wears_the_frames_family(
            self, colour_on, quiet_mirror, _clean_req_state, capsys, origin):
        # Several ids: under the pre-§4EZ single hash at least one of them
        # lands outside any given family, so this cannot pass by luck.
        for rid in ("00aa11bb", "ff00ff00", "simreq99", "benchr42", "zz9"):
            lines = self._run_framed(capsys, rid, origin)
            assert len(lines) == 6, lines
            for line in lines[1:]:          # the pre-BEGIN line is pinned below
                assert _codes(line) & _family(origin), (rid, origin, line)
                assert not (_codes(line) & (_ALL_REQUEST_FAMILIES
                                             - _family(origin))), (
                    rid, origin, line)

    def test_system_lines_render_green_and_request_lines_do_not(
            self, colour_on, quiet_mirror, capsys):
        pretty_log("Idle Cycle", "ran dream")          # default context = SYSTEM
        sys_line = capsys.readouterr().out
        assert "**" in sys_line
        assert glog._SYSTEM_COLOR in _codes(sys_line)
        tok = request_id_context.set("user0001")
        try:
            pretty_log("Request Initialized", special_marker="BEGIN",
                       origin="user")
            pretty_log("Inside", "x")
            pretty_log("Request Finished", special_marker="END")
        finally:
            request_id_context.reset(tok)
        for line in capsys.readouterr().out.splitlines():
            assert glog._SYSTEM_COLOR not in _codes(line), line

    def test_context_origin_colours_lines_before_begin_and_after_end(
            self, colour_on, quiet_mirror, _clean_req_state, capsys):
        # The lines handle_chat logs before its BEGIN frame, and the
        # background writes that keep logging after END popped the frame
        # state, carry no stash — the contextvar is their only source.
        tok = request_id_context.set("ctxonly1")
        otok = request_origin_context.set("bench")
        try:
            pretty_log("Pre Begin", "x")
            pretty_log("Request Initialized", special_marker="BEGIN",
                       origin="bench")
            pretty_log("Request Finished", special_marker="END")
            pretty_log("Post End", "background write")
        finally:
            request_origin_context.reset(otok)
            request_id_context.reset(tok)
        lines = [l for l in capsys.readouterr().out.splitlines() if l.strip()]
        assert len(lines) == 4
        for line in (lines[0], lines[-1]):
            assert _codes(line) & _family("bench"), line
            assert not (_codes(line) & set(glog._UNCLASSIFIED_PALETTE)), line

    def test_stash_outranks_contextvar_so_a_line_matches_its_frame(
            self, colour_on, quiet_mirror, _clean_req_state, capsys):
        # One resolver, fixed order: the frame the operator saw open wins.
        tok = request_id_context.set("mixed001")
        otok = request_origin_context.set("user")
        try:
            pretty_log("Request Initialized", special_marker="BEGIN",
                       origin="sim")
            pretty_log("Inside", "x")
            pretty_log("Request Finished", special_marker="END")
        finally:
            request_origin_context.reset(otok)
            request_id_context.reset(tok)
        lines = [l for l in capsys.readouterr().out.splitlines() if l.strip()]
        for line in lines:
            assert _codes(line) & _family("sim"), line
            assert not (_codes(line) & _family("user")), line

    def test_frame_origin_word_carries_the_family_colour(
            self, colour_on, quiet_mirror, _clean_req_state, capsys):
        tok = request_id_context.set("wordcol1")
        try:
            pretty_log("Request Initialized", special_marker="BEGIN",
                       origin="sim")
            pretty_log("Request Finished", special_marker="END")
        finally:
            request_id_context.reset(tok)
        for line in capsys.readouterr().out.splitlines():
            if not line.strip():
                continue
            tail = line[line.rindex("·"):]
            assert _codes(tail) & _family("sim"), (
                "the origin word is the one place the population is spelled "
                "out; it must wear the family it names", line)

    def test_frames_stay_parseable_by_the_client_regexes(
            self, colour_on, quiet_mirror, _clean_req_state, capsys):
        # The colour lives in ANSI codes only — stripped, the head fields and
        # the "request started/finished" substrings are byte-identical.
        tok = request_id_context.set("parse001")
        try:
            pretty_log("Request Initialized", special_marker="BEGIN",
                       origin="bench")
            pretty_log("Request Finished", special_marker="END")
        finally:
            request_id_context.reset(tok)
        out = re.sub(r"\x1b\[[0-9;]*m", "", capsys.readouterr().out)
        begin, end = [l for l in out.splitlines() if l.strip()]
        assert begin.startswith("┌─ PA parse001  request started  ")
        assert begin.rstrip().endswith("· bench")
        assert end.startswith("└─ PA  request finished  ")
        assert end.rstrip().endswith("· bench")

    def test_collapse_summary_wears_the_family_too(
            self, colour_on, quiet_mirror, _clean_req_state, capsys,
            monkeypatch):
        monkeypatch.delenv("GHOST_LOG_COLLAPSE", raising=False)
        tok = request_id_context.set("collapse1")
        try:
            pretty_log("Request Initialized", special_marker="BEGIN",
                       origin="sim")
            for _ in range(3):
                pretty_log("Critic Compute", "Routing verification to X")
            pretty_log("Different", "y")     # flushes the ×3 summary
            pretty_log("Request Finished", special_marker="END")
        finally:
            request_id_context.reset(tok)
        summary = [l for l in capsys.readouterr().out.splitlines()
                   if "repeated ×3" in l]
        assert summary, "collapse summary did not print"
        assert _codes(summary[0]) & _family("sim"), summary[0]


# ── handle_chat: the producer, executed ─────────────────────────────────────

class TestHandleChatSetsTheOrigin:
    async def _turn(self, monkeypatch, **ctx_overrides):
        from tests.helpers import make_agent
        import ghost_agent.core.agent as agent_mod
        agent = make_agent(**ctx_overrides)
        agent.context.llm_client.chat_completion = AsyncMock(return_value={
            "choices": [{"message": {"content": "Hello", "tool_calls": []}}]
        })
        seen = {}
        real = agent_mod.pretty_log

        def spy(*a, **k):
            if k.get("special_marker") == "BEGIN":
                seen["begin_origin_kw"] = k.get("origin")
                seen["ctx_at_begin"] = request_origin_context.get()
            return real(*a, **k)

        monkeypatch.setattr(agent_mod, "pretty_log", spy)
        body = {"messages": [{"role": "user", "content": "hi"}],
                "model": "Qwen-Test"}
        assert request_origin_context.get() == ""
        await agent.handle_chat(body, background_tasks=None)
        seen["ctx_after"] = request_origin_context.get()
        return seen

    async def test_user_turn_sets_and_resets_the_origin_context(
            self, monkeypatch):
        seen = await self._turn(monkeypatch)
        assert seen.get("begin_origin_kw") == "user"
        assert seen.get("ctx_at_begin") == "user", (
            "the contextvar must already carry the origin when BEGIN fires — "
            "the lines before the frame are the ones with no stash")
        assert seen.get("ctx_after") == "", (
            "an origin leaking past the turn's finally would colour the next "
            "SYSTEM-context line of this task as a user line")

    async def test_bench_label_reaches_the_context(self, monkeypatch):
        seen = await self._turn(monkeypatch, turn_origin_label="bench")
        assert seen.get("begin_origin_kw") == "bench"
        assert seen.get("ctx_at_begin") == "bench"
        assert seen.get("ctx_after") == ""

    async def test_context_and_frame_come_from_one_derivation(self,
                                                              monkeypatch):
        # The value on the contextvar IS the value on the frame — no second
        # derivation that could drift from what the operator reads.
        seen = await self._turn(monkeypatch, turn_origin_label="sim")
        assert seen["ctx_at_begin"] == seen["begin_origin_kw"] == "sim"
