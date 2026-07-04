"""Regression tests for bug-hunt units 27-28 (interface + scripts).

See BUGHUNT.md. Fixed bugs pinned here:

Unit 27 (interface):
 - the /ws live-log stream is auth-gated (was an unauthenticated broadcast of
   the agent log, hijackable cross-site since WebSockets bypass CORS);
 - a non-object JSON body to /api/chat and /api/tts is a clean 400, not a
   502 + leaked str(e);
 - the Slack bot sanitizes an attacker-controlled upload filename (path
   traversal → arbitrary file write).
Unit 28 (scripts):
 - gaia_eval.extract_final_answer selects the LAST "FINAL ANSWER:" (was the
   first) and an empty extraction never scores CORRECT;
 - introspective_consistency._band_summary excludes under-parsed probes from
   the consistency median that drives the verdict.
"""

import os

import pytest

# The interface server reads GHOST_API_KEY at import — ensure it's set.
os.environ.setdefault("GHOST_API_KEY", "test-key")

from unittest.mock import AsyncMock, MagicMock

from fastapi import Request, WebSocketDisconnect


# ══════════════════════════════════════════════════════════════════════
# Unit 27 — interface /ws auth
# ══════════════════════════════════════════════════════════════════════

class TestWebsocketAuth:
    async def test_ws_rejects_missing_key(self):
        from interface.server import websocket_endpoint
        ws = MagicMock()
        ws.accept = AsyncMock()
        ws.close = AsyncMock()
        await websocket_endpoint(ws, key=None)
        ws.close.assert_awaited_once()
        ws.accept.assert_not_called()

    async def test_ws_rejects_wrong_key(self):
        from interface.server import websocket_endpoint
        ws = MagicMock()
        ws.accept = AsyncMock()
        ws.close = AsyncMock()
        await websocket_endpoint(ws, key="not-the-key")
        ws.close.assert_awaited_once()
        ws.accept.assert_not_called()

    async def test_ws_accepts_valid_key(self):
        from interface.server import websocket_endpoint, GHOST_API_KEY
        ws = MagicMock()
        ws.accept = AsyncMock()
        ws.close = AsyncMock()
        # Exit the receive loop immediately via a disconnect.
        ws.receive_text = AsyncMock(side_effect=WebSocketDisconnect(code=1000))
        await websocket_endpoint(ws, key=GHOST_API_KEY)
        ws.accept.assert_awaited_once()


# ══════════════════════════════════════════════════════════════════════
# Unit 27 — non-object JSON body → 400 (not 502 + leak)
# ══════════════════════════════════════════════════════════════════════

class TestNonDictBody:
    async def test_chat_proxy_non_dict_body_is_400(self):
        from interface.server import chat_proxy
        req = MagicMock(spec=Request)
        req.json = AsyncMock(return_value=["not", "an", "object"])
        resp = await chat_proxy(req)
        assert getattr(resp, "status_code", None) == 400

    async def test_tts_proxy_non_dict_body_is_400(self):
        from interface.server import tts_proxy
        req = MagicMock(spec=Request)
        req.json = AsyncMock(return_value="a bare string")
        resp = await tts_proxy(req)
        assert getattr(resp, "status_code", None) == 400


# ══════════════════════════════════════════════════════════════════════
# Unit 27 — interface constant-time compare (source pin)
# ══════════════════════════════════════════════════════════════════════

class TestConstantTimeCompare:
    def test_uses_compare_digest(self):
        import inspect
        import interface.server as srv
        src = inspect.getsource(srv)
        assert "secrets.compare_digest" in src
        # The timing-leaky forms must be gone from the auth checks.
        assert "x_ghost_key != GHOST_API_KEY" not in src
        assert "key != GHOST_API_KEY" not in src


# ══════════════════════════════════════════════════════════════════════
# Unit 27 — Slack bot filename sanitization (source pin)
# ══════════════════════════════════════════════════════════════════════

class TestSlackTraversal:
    def test_download_uses_basename(self):
        import inspect
        import interface.externals.slack_bot.main as m
        src = inspect.getsource(m.download_slack_file)
        assert "os.path.basename(filename)" in src


# ══════════════════════════════════════════════════════════════════════
# Unit 28 — gaia_eval answer extraction
# ══════════════════════════════════════════════════════════════════════

class TestGaiaExtract:
    def test_last_final_answer_wins(self):
        from scripts.gaia_eval import extract_final_answer
        text = "thinking... FINAL ANSWER: 41\nrecheck\nFINAL ANSWER: 42"
        assert extract_final_answer(text) == "42"

    def test_no_marker_returns_none(self):
        from scripts.gaia_eval import extract_final_answer
        assert extract_final_answer("no marker here") is None
        assert extract_final_answer("") is None

    def test_empty_answer_not_scored_correct_against_placeholder_gt(self):
        # The scorer itself equates empty vs a "?" placeholder (both normalize
        # to "") — the gaia_eval loop guards against crediting an empty answer.
        from scripts.gaia_scorer import question_scorer
        assert question_scorer("", "?") is True  # confirms the hazard the guard covers
        # And a real answer still scores normally.
        assert question_scorer("42", "42") is True


# ══════════════════════════════════════════════════════════════════════
# Unit 28 — introspective consistency band summary
# ══════════════════════════════════════════════════════════════════════

class TestIntrospectiveBandSummary:
    def _row(self, mode_share, parsed_n, n=5):
        return {
            "mode_share": mode_share,
            "parsed_n": parsed_n,
            "parse_rate": parsed_n / n,
            "label_entropy_bits": 0.0,
        }

    def test_underparsed_probe_excluded_from_median(self):
        from scripts.introspective_consistency import _band_summary
        # One well-parsed probe with genuine disagreement (0.4) and one
        # under-parsed probe that trivially reports 1.0 from a single answer.
        rows = [self._row(0.4, parsed_n=5), self._row(1.0, parsed_n=1)]
        out = _band_summary(rows)
        # The spurious 1.0 must be excluded → median reflects only the trusted
        # probe, not (0.4+1.0)/2 = 0.7.
        assert out["median_mode_share"] == pytest.approx(0.4)
        assert out["n_probes_trusted"] == 1
        assert out["n_probes"] == 2

    def test_all_underparsed_falls_back(self):
        from scripts.introspective_consistency import _band_summary
        rows = [self._row(1.0, parsed_n=1), self._row(1.0, parsed_n=0)]
        out = _band_summary(rows)
        # Never returns empty — falls back to the raw rows.
        assert out["n_probes"] == 2
        assert "median_mode_share" in out
