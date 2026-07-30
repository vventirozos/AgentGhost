"""Tests for scripts/ghost_kill_sessions.py — the in-flight-turn TUI's
pure helpers (API client, key precedence, formatting). The curses loop is
deliberately untested; everything it renders/acts on comes through these."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

import io
import json
import urllib.error

import pytest

import ghost_kill_sessions as gks


# ──────────────────────────────────────────────────────────────────────
# Key precedence
# ──────────────────────────────────────────────────────────────────────

class TestLoadApiKey:
    def test_nonempty_env_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GHOST_API_KEY", "env-key")
        f = tmp_path / "k"
        f.write_text("file-key")
        assert gks.load_api_key(f) == "env-key"

    def test_empty_env_falls_through_to_file(self, monkeypatch, tmp_path):
        # Observed live: an interactive shell exported GHOST_API_KEY=""
        # and the script sent no key → 403. Set-but-empty must not win.
        monkeypatch.setenv("GHOST_API_KEY", "")
        f = tmp_path / "k"
        f.write_text("file-key\n")
        assert gks.load_api_key(f) == "file-key"

    def test_missing_everything_is_empty(self, monkeypatch, tmp_path):
        monkeypatch.delenv("GHOST_API_KEY", raising=False)
        assert gks.load_api_key(tmp_path / "absent") == ""


# ──────────────────────────────────────────────────────────────────────
# Formatting
# ──────────────────────────────────────────────────────────────────────

class TestFormatting:
    def test_fmt_age(self):
        assert gks.fmt_age(42) == "42s"
        assert gks.fmt_age(125) == "2m05s"
        assert gks.fmt_age(3900) == "1h05m"
        assert gks.fmt_age(None) == "?"

    def test_turn_state(self):
        assert gks.turn_state({"running": True}) == "RUNNING"
        assert gks.turn_state({"running": False}) == "QUEUED"
        assert gks.turn_state({"running": True, "cancelled": True}) == \
            "CANCELLING"

    def test_age_severity_escalates(self):
        assert gks.age_severity(10) == ""
        assert gks.age_severity(gks.AGE_WARN_S) == "age_warn"
        assert gks.age_severity(gks.AGE_BAD_S) == "age_bad"
        assert gks.age_severity(None) == ""     # unknown age → no alarm

    def test_insane_age_never_raises(self):
        # json.loads accepts bare Infinity; int(inf) and localtime(1e20)
        # both throw OverflowError — the draw loop must survive them.
        assert gks.fmt_age(float("inf")) == "?"
        t = {"request_id": "x", "running": True, "age_s": float("inf"),
             "session_id": "", "preview": "p"}
        assert gks.render_rows([t], 120)          # no raise
        t["age_s"] = 1e20
        lines = gks.detail_lines(t, {}, 120, now=100.0)
        assert "started ?" in "".join(s for s, _ in lines[0])

    def test_render_rows_truncates_to_width(self):
        turns = [{"request_id": "e7e55cb8aabbccdd", "running": True,
                  "age_s": 5, "session_id": "sess-1234567",
                  "preview": "x" * 500}]
        rows = gks.render_rows(turns, width=80)
        assert len(rows) == 1
        assert rows[0].startswith("e7e55cb8aabb ")
        assert len(rows[0]) <= 80

    def test_render_rows_collapses_preview_whitespace(self):
        turns = [{"request_id": "a" * 12, "running": False, "age_s": 1,
                  "session_id": None, "preview": "line1\nline2\t end"}]
        assert "line1 line2 end" in gks.render_rows(turns, 120)[0]

    def test_rows_align_with_column_header(self):
        # PREVIEW must start exactly where the header says it does.
        turns = [{"request_id": "r" * 12, "running": True, "age_s": 1,
                  "session_id": "s1", "preview": "PVW"}]
        row = gks.render_rows(turns, 120)[0]
        assert row.index("PVW") == gks.column_header().index("PREVIEW")


# ──────────────────────────────────────────────────────────────────────
# Session join (/api/sessions enrichment)
# ──────────────────────────────────────────────────────────────────────

class TestSessionJoin:
    SESSIONS = {"s1": {"id": "s1", "title": "postgres  upgrade",
                       "message_count": 42, "updated_at": 1000.0}}

    def test_label_prefers_title_and_carries_msg_count(self):
        label, msgs = gks.session_label("s1", self.SESSIONS)
        assert label == "postgres upgrade"   # whitespace collapsed
        assert msgs == "42"

    def test_label_falls_back_to_id_prefix_when_unjoined(self):
        label, msgs = gks.session_label("deadbeefcafe", {})
        assert label == "deadbeef"
        assert msgs == ""

    def test_label_untitled_session_uses_id_prefix(self):
        label, _ = gks.session_label(
            "s1", {"s1": {"id": "s1", "title": "", "message_count": 1}})
        assert label == "s1"

    def test_label_no_session_id_is_dash(self):
        assert gks.session_label(None, self.SESSIONS) == ("-", "")

    def test_row_cells_show_title_msgs_and_state_role(self):
        t = {"request_id": "r" * 16, "running": True, "age_s": 700,
             "session_id": "s1", "preview": "hi"}
        cells = gks.row_cells(t, 120, self.SESSIONS)
        text = "".join(s for s, _ in cells)
        roles = dict((role, s) for s, role in cells)
        assert "postgres upgrade" in text and "  42  " in text
        assert "running" in roles          # state colored by state
        assert "age_bad" in roles          # 700s ≥ AGE_BAD_S escalates
        assert "session" in roles

    def test_fetch_sessions_maps_by_id(self, monkeypatch):
        cap = []
        _mock_urlopen(monkeypatch,
                      {"enabled": True,
                       "sessions": [{"id": "a", "title": "T"}, "junk"]},
                      capture=cap)
        out = gks.fetch_sessions("http://x", "k")
        assert out == {"a": {"id": "a", "title": "T"}}
        assert cap[0][0].full_url.endswith("/api/sessions?limit=200")

    def test_fetch_sessions_failure_is_empty_enrichment(self, monkeypatch):
        _mock_urlopen(monkeypatch,
                      error=urllib.error.URLError("conn refused"))
        assert gks.fetch_sessions("http://x", "k") == {}
        _mock_urlopen(monkeypatch, {"enabled": False, "sessions": []})
        assert gks.fetch_sessions("http://x", "k") == {}


# ──────────────────────────────────────────────────────────────────────
# Detail pane (selected turn)
# ──────────────────────────────────────────────────────────────────────

class TestDetailLines:
    def test_full_id_session_join_and_preview(self):
        now = 2000.0
        t = {"request_id": "e7e55cb8-full-id", "running": True, "age_s": 65,
             "session_id": "s1", "preview": "fix the flaky test"}
        sessions = {"s1": {"id": "s1", "title": "My project",
                           "message_count": 3, "updated_at": now - 60}}
        lines = gks.detail_lines(t, sessions, width=120, now=now)
        assert len(lines) == 3
        flat = ["".join(s for s, _ in segs) for segs in lines]
        assert "e7e55cb8-full-id" in flat[0]          # UNtruncated id
        assert "My project" in flat[1]
        assert "3 msgs" in flat[1]
        assert "active 1m00s ago" in flat[1]
        assert "fix the flaky test" in flat[2]

    def test_no_session_and_cancelled_are_stated(self):
        t = {"request_id": "x", "running": True, "age_s": 5,
             "cancelled": True, "session_id": "", "preview": ""}
        lines = gks.detail_lines(t, {}, width=80, now=100.0)
        flat = ["".join(s for s, _ in segs) for segs in lines]
        assert "cancel requested" in flat[0]
        assert "no session id" in flat[1]
        assert "(no preview)" in flat[2]

    def test_unjoined_session_says_so(self):
        t = {"request_id": "x", "running": True, "age_s": 5,
             "session_id": "mystery", "preview": "p"}
        flat = "".join(s for s, _ in gks.detail_lines(t, {}, 80, now=1.0)[1])
        assert "mystery" in flat and "not in /api/sessions" in flat


# ──────────────────────────────────────────────────────────────────────
# Selection re-binding across refreshes
# ──────────────────────────────────────────────────────────────────────

class TestFindTurnIndex:
    TURNS = [{"request_id": "aaa"}, {"request_id": "bbb"},
             {"request_id": "ccc"}]

    def test_selection_follows_the_turn_not_the_row(self):
        # The running turn finishing shifts every queued turn up one —
        # a cancel keyed off the old index would kill the WRONG request.
        assert gks.find_turn_index(self.TURNS[1:], "bbb", fallback=1) == 0

    def test_vanished_turn_uses_fallback(self):
        assert gks.find_turn_index(self.TURNS, "gone", fallback=2) == 2
        assert gks.find_turn_index([], "x", fallback=0) == 0


# ──────────────────────────────────────────────────────────────────────
# ANSI painting (--list mode)
# ──────────────────────────────────────────────────────────────────────

class TestPaint:
    def test_color_off_is_plain_join(self):
        segs = [("RUNNING", "running"), (" x", "plain")]
        assert gks.paint(segs, color=False) == "RUNNING x"

    def test_color_on_wraps_known_roles_only(self):
        out = gks.paint([("RUNNING", "running"), (" x", "plain")], color=True)
        assert out.startswith("\x1b[1;32mRUNNING\x1b[0m")
        assert out.endswith(" x")                      # plain left untouched

    def test_ansi_enabled_gates(self, monkeypatch):
        class Tty:
            def isatty(self):
                return True
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("TERM", "xterm-256color")
        assert gks.ansi_enabled(Tty()) is True
        monkeypatch.setenv("NO_COLOR", "1")
        assert gks.ansi_enabled(Tty()) is False        # NO_COLOR wins
        monkeypatch.delenv("NO_COLOR", raising=False)
        assert gks.ansi_enabled(io.StringIO()) is False  # pipe → plain


# ──────────────────────────────────────────────────────────────────────
# API client (urlopen mocked)
# ──────────────────────────────────────────────────────────────────────

class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _mock_urlopen(monkeypatch, payload=None, error=None, capture=None):
    def fake(req, data=None, timeout=None):
        if capture is not None:
            capture.append((req, data))
        if error is not None:
            raise error
        return _Resp(json.dumps(payload).encode())
    monkeypatch.setattr(gks.urllib.request, "urlopen", fake)


class TestApiClient:
    def test_fetch_turns_accepts_dict_and_list_shapes(self, monkeypatch):
        _mock_urlopen(monkeypatch, {"turns": [{"request_id": "x"}]})
        turns, err = gks.fetch_turns("http://a", "k")
        assert err is None and turns == [{"request_id": "x"}]
        _mock_urlopen(monkeypatch, [{"request_id": "y"}])
        turns, err = gks.fetch_turns("http://a", "k")
        assert err is None and turns == [{"request_id": "y"}]

    def test_fetch_turns_drops_non_dict_entries(self, monkeypatch):
        # A lying agent must not crash the TUI (t.get on None/str raises).
        _mock_urlopen(monkeypatch,
                      {"turns": [None, "junk", {"request_id": "x"}]})
        turns, err = gks.fetch_turns("http://a", "k")
        assert err is None and turns == [{"request_id": "x"}]

    def test_key_header_attached(self, monkeypatch):
        cap = []
        _mock_urlopen(monkeypatch, {"turns": []}, capture=cap)
        gks.fetch_turns("http://a", "sekret")
        req, _ = cap[0]
        assert req.get_header("X-ghost-key") == "sekret"

    def test_403_is_a_readable_error(self, monkeypatch):
        err403 = urllib.error.HTTPError("u", 403, "F", {}, io.BytesIO(b""))
        _mock_urlopen(monkeypatch, error=err403)
        turns, err = gks.fetch_turns("http://a", "bad")
        assert turns == [] and "auth rejected" in err

    def test_unreachable_is_a_readable_error(self, monkeypatch):
        _mock_urlopen(monkeypatch,
                      error=urllib.error.URLError("conn refused"))
        turns, err = gks.fetch_turns("http://a", "k")
        assert turns == [] and "unreachable" in err

    def test_cancel_posts_id_and_hard_flag(self, monkeypatch):
        cap = []
        _mock_urlopen(monkeypatch, {"cancelled": True, "mode": "hard"},
                      capture=cap)
        result, err = gks.cancel_turn("http://a", "k", "req-1", hard=True)
        assert err is None and result["mode"] == "hard"
        _, data = cap[0]
        body = json.loads(data.decode())
        assert body == {"request_id": "req-1", "hard": True}

    def test_cancel_404_surfaces_server_message(self, monkeypatch):
        # /api/turn/cancel answers 404 with {"error": "no active turn …"}.
        body = io.BytesIO(json.dumps(
            {"cancelled": False, "error": "no active turn 'x'"}).encode())
        err404 = urllib.error.HTTPError("u", 404, "NF", {}, body)
        _mock_urlopen(monkeypatch, error=err404)
        result, err = gks.cancel_turn("http://a", "k", "x", hard=False)
        assert result is None
        assert "no active turn" in err
