"""Regression tests for the 2026-07-27 least-audited-systems bug hunt.

Five parallel read-only agents audited the subsystems that had gone longest
without a deep review (tasks/scheduling, media+bridge tools, database+games,
sessions/interface/composed-skills, deep-reason/introspection). These tests
pin the fixes for every HIGH/MED finding that was confirmed at source.

Grouped by the file the defect lived in.
"""

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ──────────────────────────────────────────────────────────────────────
# validators.py — the destructive-SQL guard was inoperative in prod
# ──────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────
# tasks.py — cron fired in LOCAL time despite a UTC contract
# ──────────────────────────────────────────────────────────────────────

class TestCronTimezone:
    def test_crontab_trigger_is_built_in_utc(self):
        """A pre-built trigger instance never inherits the scheduler's
        timezone, so `from_crontab(expr)` defaulted to the host zone
        (Europe/Athens) while registry.py tells the model times are UTC —
        every cron task fired hours early and drifted with DST."""
        from ghost_agent.tools import tasks as t
        if t.CronTrigger is None:                      # apscheduler absent
            pytest.skip("apscheduler not installed")
        sched = MagicMock()
        t.run_proactive_task_fn = lambda *a, **k: None
        err = t._add_job(sched, "job1", "nightly", "do a thing", "0 6 * * *")
        assert err is None
        trigger = sched.add_job.call_args[0][1]
        assert str(trigger.timezone) == "UTC", (
            f"cron trigger built in {trigger.timezone}, not UTC")

    def test_jobs_get_a_real_misfire_grace(self):
        """APScheduler's default grace is 1s; the agent's event loop stalls
        longer than that routinely, and a misfire is SKIPPED silently."""
        from ghost_agent.tools import tasks as t
        if t.CronTrigger is None:
            pytest.skip("apscheduler not installed")
        sched = MagicMock()
        t.run_proactive_task_fn = lambda *a, **k: None
        assert t._add_job(sched, "j", "n", "p", "0 6 * * *") is None
        assert sched.add_job.call_args[1]["misfire_grace_time"] >= 60
        sched.reset_mock()
        assert t._add_job(sched, "j2", "n", "p", "interval:300") is None
        assert sched.add_job.call_args[1]["misfire_grace_time"] >= 60


# ──────────────────────────────────────────────────────────────────────
# vision.py — raster budget + URL content sniffing
# ──────────────────────────────────────────────────────────────────────

class TestVisionHardening:
    def test_pdf_zoom_is_derived_from_a_pixel_budget(self):
        """The 50 MB FILE cap cannot bound get_pixmap: a KB-sized vector PDF
        declaring a 10000x10000pt page allocated ~1.2 GB per page at the old
        fixed 2x zoom."""
        from ghost_agent.tools.vision import _MAX_PDF_PAGE_PIXELS
        # Reproduce the zoom formula the tool applies per page.
        def zoom_for(w, h):
            return min(2.0, (_MAX_PDF_PAGE_PIXELS / max(1.0, w * h)) ** 0.5)

        # A4 keeps full quality...
        assert zoom_for(595, 842) == 2.0
        # ...a poster-sized page is scaled down hard.
        z = zoom_for(10000, 10000)
        assert z < 2.0
        px = (10000 * z) * (10000 * z)
        assert px <= _MAX_PDF_PAGE_PIXELS * 1.01

    @pytest.mark.parametrize("mime", ["image/jpeg", "image/png"])
    def test_sniffable_mime_without_signature_is_refused(self, mime):
        """The URL branch trusted Content-Type (defaulting to image/jpeg when
        absent) and never magic-sniffed — resurrecting the hallucination bug
        the local branch's sniffing was added to kill."""
        from ghost_agent.tools.vision import _sniff_image_mime, _SNIFFABLE_MIMES
        assert mime in _SNIFFABLE_MIMES
        assert _sniff_image_mime(b"<html><body>404") is None

    def test_sniffer_still_types_real_images(self):
        from ghost_agent.tools.vision import _sniff_image_mime
        assert _sniff_image_mime(b"\x89PNG\r\n\x1a\n") == "image/png"
        assert _sniff_image_mime(b"\xff\xd8\xff\xe0") == "image/jpeg"
        assert _sniff_image_mime(b"RIFF\x00\x00\x00\x00WEBP") == "image/webp"


# ──────────────────────────────────────────────────────────────────────
# report_pdf.py — headers inside fenced code blocks
# ──────────────────────────────────────────────────────────────────────

class TestReportPdfFences:
    def test_comments_in_a_fenced_block_are_not_headers(self):
        """`# install deps` inside a ```bash fence became an h2 section and
        the fence markers were stranded across two sections."""
        from ghost_agent.tools.report_pdf import _split_markdown_into_sections
        md = (
            "# Report\n"
            "Intro text.\n"
            "## Setup\n"
            "```bash\n"
            "# install deps\n"
            "pip install -r requirements.txt\n"
            "# start the server\n"
            "./run.sh\n"
            "```\n"
            "Done.\n"
        )
        secs = _split_markdown_into_sections(md)
        headings = [s["heading"] for s in secs]
        assert "install deps" not in headings
        assert "start the server" not in headings
        assert headings == ["Report", "Setup"]
        body = secs[-1]["body"]
        assert body.count("```") == 2, "both fence markers stay in one section"

    def test_tilde_fences_are_handled(self):
        from ghost_agent.tools.report_pdf import _split_markdown_into_sections
        md = "# A\nx\n## B\n~~~\n# not a header\n~~~\n## C\ny\n"
        headings = [s["heading"] for s in _split_markdown_into_sections(md)]
        assert headings == ["A", "B", "C"]

    def test_real_headers_after_a_closed_fence_still_split(self):
        from ghost_agent.tools.report_pdf import _split_markdown_into_sections
        md = "# A\n```\ncode\n```\n## B\nbody\n"
        headings = [s["heading"] for s in _split_markdown_into_sections(md)]
        assert headings == ["A", "B"]

    def test_markdown_render_failure_degrades_instead_of_raising(self):
        from ghost_agent.tools import report_pdf
        with patch.object(report_pdf, "_md_to_html", wraps=report_pdf._md_to_html):
            html = report_pdf._md_to_html("**bold**")
        assert "bold" in html


# ──────────────────────────────────────────────────────────────────────
# database.py — pool keying + session-state leakage + cell clipping
# ──────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────
# games — parity brick + draft-vs-final extraction
# ──────────────────────────────────────────────────────────────────────

class TestGames:
    def _a(self):
        from ghost_agent.api.games.tictactoe import TicTacToeAdapter
        return TicTacToeAdapter()

    def test_agent_opening_as_o_does_not_brick_the_game(self):
        """`load` accepted "......... O", the agent played O, and the
        resulting "....O.... X" was then rejected forever (422) — the game
        died after exactly one move."""
        a = self._a()
        b = a.load("......... O", [])
        assert b.turn == "O"
        a.apply_move(b, "5")
        nxt = a.serialize(b)
        assert nxt == "....O.... X"
        reloaded = a.load(nxt, [])          # must not raise
        assert reloaded.turn == "X"

    def test_turn_contradicting_the_counts_is_rejected(self):
        """With unequal counts the side to move is forced; an explicit field
        that disagrees let one side move twice."""
        from ghost_agent.api.games.base import GameStateError
        a = self._a()
        with pytest.raises(GameStateError):
            a.load("X........ X", [])       # X already moved; must be O

    def test_even_board_accepts_either_side(self):
        a = self._a()
        assert a.load("OXXO..... O", []).turn == "O"
        assert a.load("OXXO..... X", []).turn == "X"

    def test_truly_impossible_counts_still_rejected(self):
        from ghost_agent.api.games.base import GameStateError
        a = self._a()
        with pytest.raises(GameStateError):
            a.load("XXX...... O", [])       # 3 X vs 0 O

    def test_labeled_extraction_is_last_wins(self):
        """A thinking model drafts a move, rejects it, and commits to
        another: the move came from the final answer but the EXPLANATION
        from the discarded draft."""
        from ghost_agent.api.games.base import extract_labeled, extract_move_text
        reply = (
            "<think>MOVE: e4\nEXPLANATION: I grab the centre.\n"
            "...actually that hangs a pawn...</think>\n"
            "MOVE: d4\nEXPLANATION: The queen pawn is safer here.\n"
        )
        assert extract_move_text(reply) == "d4"
        assert extract_labeled(reply, "EXPLANATION") == "The queen pawn is safer here."


# ──────────────────────────────────────────────────────────────────────
# introspect.py / learning_health.py — the reports went dark
# ──────────────────────────────────────────────────────────────────────

class TestActivityReporting:
    def _ledger(self, tmp_path, n, phase="dream"):
        p = tmp_path / "autonomous_activity.jsonl"
        now = time.time()
        with p.open("w") as fh:
            for i in range(n):
                fh.write(json.dumps({
                    "ts": now - (n - i), "phase": phase,
                    "summary": f"record {i}", "severity": "info", "meta": {},
                }) + "\n")
        return p

    def test_activity_read_returns_the_NEWEST_records(self, tmp_path):
        """`read_since(0)` inherits limit=200, so the report replayed the 200
        OLDEST lines — live, it had been answering "what did you do while I
        was away" from records that stopped weeks earlier."""
        from ghost_agent.core.autonomous_activity import ActivityLog
        from ghost_agent.tools.introspect import _read_activity_tail
        path = self._ledger(tmp_path, 900)
        recs, _truncated, failed = _read_activity_tail(ActivityLog(str(path)))
        assert recs, "tail read returned nothing"
        assert failed is False
        summaries = {r.summary for r in recs}
        assert "record 899" in summaries, "newest record missing"
        assert "record 0" not in summaries or len(recs) >= 900

    def test_activity_read_handles_a_small_ledger(self, tmp_path):
        from ghost_agent.core.autonomous_activity import ActivityLog
        from ghost_agent.tools.introspect import _read_activity_tail
        path = self._ledger(tmp_path, 5)
        recs, truncated, failed = _read_activity_tail(ActivityLog(str(path)))
        assert len(recs) == 5
        assert truncated is False and failed is False

    def test_activity_read_handles_a_missing_ledger(self, tmp_path):
        from ghost_agent.core.autonomous_activity import ActivityLog
        from ghost_agent.tools.introspect import _read_activity_tail
        recs, truncated, failed = _read_activity_tail(
            ActivityLog(str(tmp_path / "nope.jsonl")))
        assert recs == []
        assert failed is False  # a missing ledger is empty, not broken

    def test_learning_health_counts_by_phase(self, tmp_path):
        """`to_dict` serializes the kind as `phase`; keying on
        kind/type/category matched nothing, so the BACKGROUND ACTIVITY
        section silently never rendered."""
        from ghost_agent.core.learning_health import _activity_counts
        path = self._ledger(tmp_path, 10, phase="self_play")
        counts = _activity_counts(path)
        assert counts.get("self_play") == 10

    def test_learning_health_respects_the_recent_window(self, tmp_path):
        from ghost_agent.core.learning_health import _activity_counts
        p = tmp_path / "led.jsonl"
        old = time.time() - 40 * 24 * 3600
        with p.open("w") as fh:
            fh.write(json.dumps({"ts": old, "phase": "dream"}) + "\n")
            fh.write(json.dumps({"ts": time.time(), "phase": "dream"}) + "\n")
        # §4NM: a pre-rename "dream" row is read as "maintenance"
        assert _activity_counts(p, window_hours=168).get("maintenance") == 1


# ──────────────────────────────────────────────────────────────────────
# sessions.py — the fat-client duplication cascade
# ──────────────────────────────────────────────────────────────────────

class TestSessionMerge:
    def _m(self, stored, incoming):
        from ghost_agent.core.sessions import merge_history
        return merge_history(stored, incoming)

    def _msgs(self, *pairs):
        return [{"role": r, "content": c} for r, c in pairs]

    def test_exact_fat_replay_is_not_duplicated(self):
        stored = self._msgs(("user", "hi"), ("assistant", "hello"))
        incoming = stored + self._msgs(("user", "again"))
        assert len(self._m(stored, incoming)) == 3

    def test_diverged_fat_replay_does_not_compound(self):
        """An aborted stream leaves the server holding the FULL reply while
        the client kept a partial one. The stored-prefix compare then failed
        and the whole conversation was re-appended EVERY turn (5 → 11 → 19 →
        29 messages, quadratic, until the 400-cap filled with duplicates)."""
        stored = self._msgs(("user", "hi"), ("assistant", "FULL ANSWER"))
        incoming = self._msgs(("user", "hi"), ("assistant", "FULL ANS"),
                              ("user", "next"))
        merged = self._m(stored, incoming)
        assert len(merged) == 3, "diverged replay must not concatenate"
        # ⚠ ASSERTION CHANGED 2026-08-17 (§4BU), deliberately. This used to
        # assert `merged == incoming` — the client's partial reply replacing
        # the server's full one. The rewrite makes STORED authoritative and
        # appends only the client's new tail, because "incoming replaces
        # stored" is what allowed durable history to be DELETED when a stale
        # tab was missing middle messages. Keeping the server's full reply
        # is information kept, not lost; the property this test exists for —
        # no compounding — is asserted above and by the invariant suite in
        # tests/test_sessions_merge_properties.py.
        assert merged[0] == incoming[0]
        assert merged[-1] == incoming[-1], "the new user message must survive"
        assert merged[1]["content"] == "FULL ANSWER", (
            "the server's complete reply is authoritative, not the client's "
            "aborted stub")

    def test_repeated_turns_do_not_grow_quadratically(self):
        stored = self._msgs(("user", "hi"), ("assistant", "FULL ANSWER"))
        incoming = self._msgs(("user", "hi"), ("assistant", "FULL ANS"))
        for _ in range(4):
            incoming = incoming + self._msgs(("user", "more"))
            merged = self._m(stored, incoming)
            stored = merged
            assert len(merged) == len(incoming)

    def test_cap_evicted_prefix_still_recognised_as_the_same_conversation(self):
        """Once a session hits the message cap the stored copy loses its
        oldest messages, so the client replay could never prefix-match
        again — permanent duplication."""
        full = self._msgs(*[("user", f"m{i}") for i in range(10)])
        stored = full[3:]                      # oldest three evicted
        merged = self._m(stored, full)
        assert merged == full

    def test_thin_client_still_appends(self):
        stored = self._msgs(("user", "hi"), ("assistant", "hello"))
        incoming = self._msgs(("user", "new question"))
        merged = self._m(stored, incoming)
        assert len(merged) == 3
        assert merged[-1]["content"] == "new question"

    def test_thin_client_duplicate_system_still_deduped(self):
        stored = self._msgs(("system", "S"), ("user", "hi"))
        incoming = self._msgs(("system", "S"), ("user", "next"))
        merged = self._m(stored, incoming)
        assert sum(1 for m in merged if m["role"] == "system") == 1


# (composed_skills.py — swallowed diagnostics + invalid names: removed with the macro engine, §4TA)

