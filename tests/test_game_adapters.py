"""Tests for the GAME-AGNOSTIC POST /api/game/move: game dispatch (the
`game` field), the second reference game (tic-tac-toe), and the adapter
registry. The chess-specific behaviour lives in test_game_move_api.py; this
file proves the endpoint is genuinely general — the same protocol
(stateless, code-enforced legality, LLM-chooses, no engine fallback) drives
a non-chess game with no endpoint changes."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ghost_agent.api.game_routes import game_router
from ghost_agent.api.routes import router as main_router
from ghost_agent.api import games


def _llm_reply(text):
    return {"choices": [{"message": {"content": text}}]}


def _make_app(replies, api_key=""):
    llm = SimpleNamespace(
        chat_completion=AsyncMock(side_effect=[_llm_reply(r) for r in replies]))
    context = SimpleNamespace(args=SimpleNamespace(api_key=api_key),
                              llm_client=llm, memory_system=MagicMock())
    agent = SimpleNamespace(context=context)
    app = FastAPI()
    app.state.agent = agent
    app.include_router(game_router)
    app.include_router(main_router)
    return TestClient(app), llm


class TestGameDispatch:
    def test_lists_available_games(self):
        client, _ = _make_app([])
        res = client.get("/api/game/games")
        assert res.status_code == 200
        body = res.json()
        assert body["default"] == "chess"
        assert "chess" in body["games"]
        assert "tictactoe" in body["games"]

    def test_unknown_game_is_400_with_available_list(self):
        client, llm = _make_app([])
        res = client.post("/api/game/move",
                          json={"game": "backgammon", "state": "x"})
        assert res.status_code == 400
        assert "Unknown game" in res.json()["detail"]
        assert "tictactoe" in res.json()["detail"]
        assert not llm.chat_completion.called

    def test_game_field_defaults_to_chess(self):
        # No `game` field + a FEN behaves exactly like the old chess-only
        # endpoint (backward compatibility for existing clients).
        client, _ = _make_app(["MOVE: e4"])
        res = client.post("/api/game/move",
                          json={"fen": "rnbqkbnr/pppppppp/8/8/8/8/"
                                       "PPPPPPPP/RNBQKBNR w KQkq - 0 1"})
        assert res.status_code == 200
        assert res.json()["game"] == "chess"
        assert res.json()["move"] == "e4"


class TestTicTacToe:
    G = "tictactoe"

    def test_agent_moves_from_initial_state_when_omitted(self):
        client, _ = _make_app(["MOVE: 5\nCOMMENT: I'll take the center."])
        res = client.post("/api/game/move", json={"game": self.G})
        assert res.status_code == 200
        body = res.json()
        assert body["game"] == self.G
        assert body["move"] == "5"
        assert body["comment"] == "I'll take the center."
        # X played cell 5 (index 4); now O to move.
        assert body["state"] == "....X.... O"
        assert body["history"] == ["5"]
        assert body["game_over"] is False

    def test_user_move_applied_before_agent_reply(self):
        client, _ = _make_app(["MOVE: 5"])
        res = client.post("/api/game/move",
                          json={"game": self.G, "user_move": "1"})
        body = res.json()
        assert body["user_move"] == "1"
        assert body["move"] == "5"
        assert body["history"] == ["1", "5"]
        assert body["state"] == "X...O.... X"

    def test_coordinate_move_accepted(self):
        client, _ = _make_app(["MOVE: 9"])
        res = client.post("/api/game/move",
                          json={"game": self.G, "user_move": "a1"})
        assert res.status_code == 200
        assert res.json()["user_move"] == "1"      # a1 == top-left == cell 1

    def test_illegal_move_occupied_cell_is_422(self):
        client, llm = _make_app([])
        res = client.post("/api/game/move",
                          json={"game": self.G, "state": "X........ O",
                                "user_move": "1"})
        assert res.status_code == 422
        assert "Illegal move '1'" in res.json()["detail"]
        assert not llm.chat_completion.called

    def test_illegal_move_out_of_range_is_422(self):
        client, _ = _make_app([])
        res = client.post("/api/game/move",
                          json={"game": self.G, "user_move": "10"})
        assert res.status_code == 422

    def test_empty_user_move_is_422(self):
        client, llm = _make_app(["MOVE: 5"])
        res = client.post("/api/game/move",
                          json={"game": self.G, "user_move": "  "})
        assert res.status_code == 422
        assert "Empty user_move" in res.json()["detail"]
        assert not llm.chat_completion.called

    def test_bad_state_is_422(self):
        client, _ = _make_app([])
        res = client.post("/api/game/move",
                          json={"game": self.G, "state": "not-a-board"})
        assert res.status_code == 422

    def test_user_winning_move_ends_game_skips_llm(self):
        # X to move, completing the top row 1-2-3.
        client, llm = _make_app([])
        res = client.post("/api/game/move",
                          json={"game": self.G, "state": "XX.OO.... X",
                                "user_move": "3"})
        body = res.json()
        assert body["game_over"] is True
        assert body["result"] == "1-0"
        assert body["termination"] == "X_WINS"
        assert body["move"] is None
        assert llm.chat_completion.await_count == 0

    def test_agent_winning_move_reports_ending(self):
        # O to move with two-in-a-column; the LLM completes it.
        client, _ = _make_app(["MOVE: 7"])
        res = client.post("/api/game/move",
                          json={"game": self.G, "state": "OXXO..... O"})
        body = res.json()
        assert body["move"] == "7"
        assert body["game_over"] is True
        assert body["result"] == "0-1"
        assert body["termination"] == "O_WINS"

    def test_board_full_is_draw(self):
        # Last empty cell (9); filling it makes a full board with no line.
        client, llm = _make_app([])
        res = client.post("/api/game/move",
                          json={"game": self.G, "state": "XOXXOOOX. X",
                                "user_move": "9"})
        body = res.json()
        assert body["game_over"] is True
        assert body["result"] == "1/2-1/2"
        assert body["termination"] == "DRAW"
        assert llm.chat_completion.await_count == 0

    def test_no_engine_fallback_after_exhausted_retries(self):
        client, llm = _make_app(["nope"] * 3)
        res = client.post("/api/game/move", json={"game": self.G})
        assert res.status_code == 502
        assert llm.chat_completion.await_count == 3

    def test_illegal_llm_reply_retries_then_succeeds(self):
        client, llm = _make_app(["MOVE: 1", "MOVE: 5"])
        res = client.post("/api/game/move",
                          json={"game": self.G, "state": "X........ O"})
        assert res.status_code == 200
        assert res.json()["move"] == "5"          # cell 1 taken → retry
        assert res.json()["attempts"] == 2

    def test_auth_enforced(self):
        client, _ = _make_app(["MOVE: 5"], api_key="sekrit")
        res = client.post("/api/game/move", json={"game": self.G})
        assert res.status_code == 403
        res = client.post("/api/game/move", json={"game": self.G},
                          headers={"X-Ghost-Key": "sekrit"})
        assert res.status_code == 200

    def test_coaching_fields_are_game_agnostic(self):
        # critique + move_explanation work for tic-tac-toe too, proving the
        # learning surface lives in the generic layer, not just chess.
        reply = ("MOVE: 5\n"
                 "CRITIQUE: Taking a corner first was fine, but the centre "
                 "is the strongest opening cell.\n"
                 "EXPLANATION: I take the centre so I sit on four lines.")
        client, _ = _make_app([reply])
        res = client.post("/api/game/move",
                          json={"game": self.G, "user_move": "1"})
        body = res.json()
        assert body["move"] == "5"
        assert "strongest opening cell" in body["critique"]
        assert "four lines" in body["move_explanation"]


class TestTicTacToeAdapterUnit:
    def _a(self):
        return games.get_adapter("tictactoe")

    def test_registered(self):
        assert self._a() is not None
        assert "tictactoe" in games.available_games()

    def test_row_win_detected(self):
        a = self._a()
        b = a.load("XXXOO.... O", [])
        st = a.status(b)
        assert st["game_over"] and st["termination"] == "X_WINS"

    def test_diagonal_win_detected(self):
        a = self._a()
        b = a.load("OXX.O.X.O X", [])   # O on the 1-5-9 diagonal (3 X, 3 O)
        assert a.status(b)["termination"] == "O_WINS"

    def test_turn_inferred_from_parity(self):
        a = self._a()
        assert a.load("X........", []).turn == "O"   # X moved once → O next
        assert a.load(".........", []).turn == "X"

    def test_impossible_position_rejected(self):
        a = self._a()
        with pytest.raises(games.GameStateError):
            a.load("XXX...... X", [])   # 3 X, 0 O — impossible

    def test_serialize_round_trips(self):
        a = self._a()
        assert a.serialize(a.load("XOX.O.... O", [])) == "XOX.O.... O"

    def test_apply_move_flips_turn_and_returns_notation(self):
        a = self._a()
        b = a.load("......... X", [])
        applied = a.apply_move(b, "5")
        assert applied.notation == "5"
        assert b.turn == "O"
        assert a.serialize(b) == "....X.... O"
        assert a.apply_move(b, "5") is None          # now occupied
