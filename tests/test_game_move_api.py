"""Tests for POST /api/game/move (participant-mode chess: the AGENT's LLM
chooses the move; python-chess only validates) and for the surgical
memory-correction path (VectorMemory.correct_fragment +
POST /api/memory/correct)."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("chess")

import chess
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ghost_agent.api.game_routes import game_router
from ghost_agent.api.routes import router as main_router

START_FEN = chess.STARTING_FEN


def _llm_reply(text):
    return {"choices": [{"message": {"content": text}}]}


def _make_app(replies, api_key=""):
    """App with game_router + a mock agent whose LLM replies in order."""
    llm = SimpleNamespace(
        chat_completion=AsyncMock(side_effect=[_llm_reply(r) for r in replies]))
    memory = MagicMock()
    context = SimpleNamespace(args=SimpleNamespace(api_key=api_key),
                              llm_client=llm, memory_system=memory)
    agent = SimpleNamespace(context=context)
    app = FastAPI()
    app.state.agent = agent
    app.include_router(game_router)
    app.include_router(main_router)
    return TestClient(app), llm, memory


class TestGameMove:
    def test_agent_replies_with_legal_move(self):
        client, llm, _ = _make_app(["MOVE: e4\nCOMMENT: Good luck!"])
        res = client.post("/api/game/move", json={"fen": START_FEN})
        assert res.status_code == 200
        body = res.json()
        assert body["move"] == "e4"
        assert body["comment"] == "Good luck!"
        assert body["history"][-1] == "e4"
        assert chess.Board(body["fen"]).turn == chess.BLACK

    def test_user_move_applied_before_agent_reply(self):
        client, llm, _ = _make_app(["MOVE: e5"])
        res = client.post("/api/game/move",
                          json={"fen": START_FEN, "user_move": "e4"})
        assert res.status_code == 200
        body = res.json()
        assert body["user_move"] == "e4"
        assert body["move"] == "e5"
        assert body["history"] == ["e4", "e5"]

    def test_illegal_user_move_is_422_with_examples(self):
        client, _, _ = _make_app([])
        res = client.post("/api/game/move",
                          json={"fen": START_FEN, "user_move": "Ke2"})
        assert res.status_code == 422
        assert "Illegal move" in res.json()["detail"]

    def test_empty_user_move_is_422_not_agent_turn(self):
        # 2026-07-05: a client forwarded a bare Enter keypress as "" — the
        # old falsy check treated it as "field omitted" and the agent
        # silently moved for whichever side was to move (it ended up
        # playing BOTH sides). Empty string must be rejected, not
        # reinterpreted; omission stays the explicit way to request an
        # agent move.
        client, llm, _ = _make_app(["MOVE: e4"])
        for empty in ("", "   "):
            res = client.post("/api/game/move",
                              json={"fen": START_FEN, "user_move": empty})
            assert res.status_code == 422
            assert "Empty user_move" in res.json()["detail"]
        assert not llm.chat_completion.called

    def test_bad_fen_is_422(self):
        client, _, _ = _make_app([])
        res = client.post("/api/game/move", json={"fen": "not a fen"})
        assert res.status_code == 422

    def test_illegal_llm_reply_retries_then_succeeds(self):
        client, llm, _ = _make_app(["MOVE: Ke4", "MOVE: Nf3"])
        res = client.post("/api/game/move", json={"fen": START_FEN})
        assert res.status_code == 200
        assert res.json()["move"] == "Nf3"
        assert res.json()["attempts"] == 2
        # The retry prompt carries feedback about the illegal attempt.
        second_prompt = llm.chat_completion.call_args_list[1][0][0][
            "messages"][0]["content"]
        assert "was not a legal move" in second_prompt

    def test_no_engine_fallback_after_exhausted_retries(self):
        client, llm, _ = _make_app(["nonsense"] * 3)
        res = client.post("/api/game/move", json={"fen": START_FEN})
        assert res.status_code == 502
        assert llm.chat_completion.await_count == 3

    def test_game_over_after_user_move_skips_llm(self):
        # Fool's mate position: black mates with Qh4.
        fen = "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2"
        client, llm, _ = _make_app([])
        res = client.post("/api/game/move",
                          json={"fen": fen, "user_move": "Qh4#"})
        assert res.status_code == 200
        body = res.json()
        assert body["game_over"] is True
        assert body["result"] == "0-1"
        assert body["move"] is None
        assert llm.chat_completion.await_count == 0

    def test_uci_user_move_accepted(self):
        client, _, _ = _make_app(["MOVE: e5"])
        res = client.post("/api/game/move",
                          json={"fen": START_FEN, "user_move": "e2e4"})
        assert res.status_code == 200
        assert res.json()["user_move"] == "e4"

    def test_auth_enforced(self):
        client, _, _ = _make_app(["MOVE: e4"], api_key="sekrit")
        res = client.post("/api/game/move", json={"fen": START_FEN})
        assert res.status_code == 403
        res = client.post("/api/game/move", json={"fen": START_FEN},
                          headers={"X-Ghost-Key": "sekrit"})
        assert res.status_code == 200


class TestGameCoaching:
    """The learning-app surface: `critique` (the agent's read on the USER's
    move) and `move_explanation` (the reasoning behind the agent's OWN move),
    with `comment` kept as a backward-compatible friendly line."""

    def test_all_three_fields_surface(self):
        reply = ("MOVE: e5\n"
                 "CRITIQUE: 1.e4 grabs the centre and frees the bishop — "
                 "a principled opening move.\n"
                 "EXPLANATION: I mirror with ...e5 to contest the centre "
                 "immediately.\n"
                 "COMMENT: Your move!")
        client, _, _ = _make_app([reply])
        res = client.post("/api/game/move",
                          json={"fen": START_FEN, "user_move": "e4"})
        body = res.json()
        assert body["move"] == "e5"
        assert "grabs the centre" in body["critique"]
        assert "contest the centre" in body["move_explanation"]
        assert body["comment"] == "Your move!"

    def test_critique_empty_when_user_did_not_move(self):
        # Agent moves from the position (no user_move) → nothing to critique.
        reply = ("MOVE: e4\n"
                 "EXPLANATION: I open with 1.e4 for central space.")
        client, llm, _ = _make_app([reply])
        res = client.post("/api/game/move", json={"fen": START_FEN})
        body = res.json()
        assert body["critique"] == ""
        assert "central space" in body["move_explanation"]
        # The prompt must NOT solicit a critique when there was no user move.
        prompt = llm.chat_completion.call_args_list[0][0][0][
            "messages"][0]["content"]
        assert "CRITIQUE" not in prompt

    def test_prompt_requests_critique_after_user_move(self):
        client, llm, _ = _make_app(["MOVE: e5\nEXPLANATION: symmetry."])
        client.post("/api/game/move",
                    json={"fen": START_FEN, "user_move": "e4"})
        prompt = llm.chat_completion.call_args_list[0][0][0][
            "messages"][0]["content"]
        assert "CRITIQUE" in prompt
        assert "Your opponent just played: e4" in prompt

    def test_legacy_comment_only_reply_still_works(self):
        # Older model output with just a COMMENT line: comment populated,
        # the new fields empty (backward compatibility).
        client, _, _ = _make_app(["MOVE: e4\nCOMMENT: Good luck!"])
        res = client.post("/api/game/move", json={"fen": START_FEN})
        body = res.json()
        assert body["comment"] == "Good luck!"
        assert body["move_explanation"] == ""
        assert body["critique"] == ""


class TestMemoryCorrectEndpoint:
    def test_correct_ok(self):
        client, _, memory = _make_app([])
        memory.correct_fragment = MagicMock(
            return_value=(True, {"old_id": "a", "new_id": "b",
                                 "old_text": "old", "new_text": "new"}))
        res = client.post("/api/memory/correct",
                          json={"match": "old", "replacement": "new text here"})
        assert res.status_code == 200
        assert res.json()["ok"] is True
        memory.correct_fragment.assert_called_once_with("old", "new text here")

    def test_correct_refusal_is_409(self):
        client, _, memory = _make_app([])
        memory.correct_fragment = MagicMock(
            return_value=(False, "no stored fragment matches"))
        res = client.post("/api/memory/correct",
                          json={"match": "x", "replacement": "y" * 10})
        assert res.status_code == 409


class TestCorrectFragmentUnit:
    def _vm(self):
        from ghost_agent.memory.vector import VectorMemory

        class FakeCollection:
            def __init__(self):
                self.rows = {}

            def get(self, ids=None, include=None):
                if ids is not None:
                    hit = [(i,) + self.rows[i] for i in ids if i in self.rows]
                else:
                    hit = [(i,) + v for i, v in self.rows.items()]
                return {"ids": [h[0] for h in hit],
                        "documents": [h[1] for h in hit],
                        "metadatas": [h[2] for h in hit]}

            def delete(self, ids):
                for i in ids:
                    self.rows.pop(i, None)

            def add(self, documents, metadatas, ids):
                self.rows[ids[0]] = (documents[0], metadatas[0])

        vm = object.__new__(VectorMemory)
        vm.collection = FakeCollection()
        return vm

    def _seed(self, vm, text, meta=None):
        import hashlib
        mid = hashlib.md5(text.encode()).hexdigest()
        vm.collection.rows[mid] = (text, meta or {"type": "auto"})
        return mid

    def test_exact_text_correction_preserves_metadata(self):
        vm = self._vm()
        old = ("User wants to build a single-file, turn-based chess game "
               "where they play directly against the assistant rather than "
               "a random AI.")
        self._seed(vm, old, {"type": "auto", "timestamp": "t0"})
        ok, detail = vm.correct_fragment(old, "corrected text of the memory")
        assert ok
        assert len(vm.collection.rows) == 1
        (doc, meta), = vm.collection.rows.values()
        assert doc == "corrected text of the memory"
        assert meta["type"] == "auto"

    def test_substring_match_unique(self):
        vm = self._vm()
        self._seed(vm, "the quick brown fox memory")
        ok, _ = vm.correct_fragment("brown fox", "corrected fox memory")
        assert ok

    def test_ambiguous_substring_refused(self):
        vm = self._vm()
        self._seed(vm, "chess note one")
        self._seed(vm, "chess note two")
        ok, err = vm.correct_fragment("chess note", "replacement text")
        assert not ok and "2 fragments match" in err

    def test_no_match_refused(self):
        vm = self._vm()
        ok, err = vm.correct_fragment("nothing like this", "replacement text")
        assert not ok

    def test_document_type_excluded_from_substring_scan(self):
        vm = self._vm()
        self._seed(vm, "ingested chess document chunk", {"type": "document"})
        ok, _ = vm.correct_fragment("chess document", "replacement text")
        assert not ok


class TestMemoryDeleteEndpoint:
    def test_delete_ok(self):
        client, _, memory = _make_app([])
        memory.delete_fragment = MagicMock(
            return_value=(True, {"deleted_id": "abc",
                                 "deleted_text": "poisoned note"}))
        res = client.post("/api/memory/delete",
                          json={"match": "poisoned note"})
        assert res.status_code == 200
        assert res.json()["ok"] is True
        assert res.json()["deleted_text"] == "poisoned note"
        memory.delete_fragment.assert_called_once_with("poisoned note")

    def test_delete_refusal_is_409(self):
        client, _, memory = _make_app([])
        memory.delete_fragment = MagicMock(
            return_value=(False, "no stored fragment matches"))
        res = client.post("/api/memory/delete", json={"match": "nope"})
        assert res.status_code == 409
        assert res.json()["ok"] is False

    def test_delete_bad_json_is_400(self):
        client, _, memory = _make_app([])
        memory.delete_fragment = MagicMock()
        res = client.post("/api/memory/delete",
                          content=b"{not json",
                          headers={"Content-Type": "application/json"})
        assert res.status_code == 400
        memory.delete_fragment.assert_not_called()


class TestDeleteFragmentUnit:
    """VectorMemory.delete_fragment — the surgical-delete companion to
    correct_fragment, for fragments that are WHOLLY false (the 2026-07-04
    dream synthesis 'user prefers a random AI move selection')."""

    _vm = TestCorrectFragmentUnit._vm
    _seed = TestCorrectFragmentUnit._seed

    def test_exact_text_delete(self):
        vm = self._vm()
        poisoned = ("User is developing a terminal chess game and explicitly "
                    "prefers a random AI move selection strategy over "
                    "positional evaluation for the ghost piece.")
        self._seed(vm, poisoned)
        ok, detail = vm.delete_fragment(poisoned)
        assert ok
        assert detail["deleted_text"] == poisoned
        assert len(vm.collection.rows) == 0

    def test_substring_match_unique_delete(self):
        vm = self._vm()
        self._seed(vm, "the Marlin dream synthesis memory")
        ok, detail = vm.delete_fragment("Marlin dream")
        assert ok
        assert len(vm.collection.rows) == 0

    def test_ambiguous_substring_refused(self):
        vm = self._vm()
        self._seed(vm, "chess memory one")
        self._seed(vm, "chess memory two")
        ok, err = vm.delete_fragment("chess memory")
        assert not ok and "2 fragments match" in err
        assert len(vm.collection.rows) == 2  # nothing deleted

    def test_no_match_refused(self):
        vm = self._vm()
        self._seed(vm, "an unrelated fact")
        ok, err = vm.delete_fragment("nothing like this")
        assert not ok
        assert len(vm.collection.rows) == 1

    def test_empty_match_refused(self):
        vm = self._vm()
        ok, err = vm.delete_fragment("   ")
        assert not ok

    def test_document_type_excluded_from_substring_scan(self):
        vm = self._vm()
        self._seed(vm, "ingested chess document chunk", {"type": "document"})
        ok, _ = vm.delete_fragment("chess document")
        assert not ok
        assert len(vm.collection.rows) == 1


# --------------------------------------------------------------------------
# Full-rules coverage (2026-07-05): en passant, promotion, mate/stalemate/
# material endings, and the repetition rules — the last only work because
# _restore_move_stack replays the SAN history (a bare FEN has no memory,
# so "fivefold" could never trigger across stateless requests).
# --------------------------------------------------------------------------

def _fen_after(moves):
    b = chess.Board()
    for san in moves:
        b.push_san(san)
    return b.fen()


class TestChessRules:
    def test_en_passant_capture_applied(self):
        moves = ["e4", "a6", "e5", "d5"]          # d5 makes exd6 e.p. legal
        client, _, _ = _make_app(["MOVE: Nc6"])
        res = client.post("/api/game/move",
                          json={"fen": _fen_after(moves), "history": moves,
                                "user_move": "exd6"})
        assert res.status_code == 200
        body = res.json()
        assert body["user_move"] == "exd6"
        assert "d6" in body["fen"].split()[0] or "P" in body["fen"]
        # the captured d-pawn is gone: black has 7 pawns left
        assert body["fen"].split()[0].count("p") == 7

    def test_promotion_san_and_uci(self):
        fen = "8/P6k/8/8/8/4K3/8/8 w - - 0 1"
        for promo in ("a8=Q", "a7a8q"):
            client, _, _ = _make_app(["MOVE: Kh6"])
            res = client.post("/api/game/move",
                              json={"fen": fen, "user_move": promo})
            assert res.status_code == 200
            body = res.json()
            assert body["user_move"] == "a8=Q"
            assert "Q" in body["fen"].split()[0]

    def test_user_checkmate_ends_game_no_llm(self):
        fen = ("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR "
               "w kq - 0 4")
        client, llm, _ = _make_app([])
        res = client.post("/api/game/move",
                          json={"fen": fen, "user_move": "Qxf7#"})
        assert res.status_code == 200
        body = res.json()
        assert body["game_over"] is True
        assert body["result"] == "1-0"
        assert body["termination"] == "CHECKMATE"
        assert body["move"] is None
        assert not llm.chat_completion.called

    def test_stalemate_is_draw(self):
        fen = "k7/8/8/1Q6/8/8/8/7K w - - 0 1"
        client, llm, _ = _make_app([])
        res = client.post("/api/game/move",
                          json={"fen": fen, "user_move": "Qb6"})
        body = res.json()
        assert body["game_over"] is True
        assert body["result"] == "1/2-1/2"
        assert body["termination"] == "STALEMATE"
        assert not llm.chat_completion.called

    def test_insufficient_material_is_draw(self):
        fen = "7k/8/8/8/8/8/1q6/K7 w - - 0 1"
        client, _, _ = _make_app([])
        res = client.post("/api/game/move",
                          json={"fen": fen, "user_move": "Kxb2"})
        body = res.json()
        assert body["game_over"] is True
        assert body["termination"] == "INSUFFICIENT_MATERIAL"

    def test_fivefold_repetition_auto_draw_via_history(self):
        # 4 knight-shuffle cycles = start position occurs 5 times.
        moves = ["Nf3", "Nf6", "Ng1", "Ng8"] * 4
        client, llm, _ = _make_app([])
        res = client.post("/api/game/move",
                          json={"fen": _fen_after(moves), "history": moves})
        body = res.json()
        assert body["game_over"] is True
        assert body["result"] == "1/2-1/2"
        assert body["termination"] == "FIVEFOLD_REPETITION"
        assert body["move"] is None
        assert not llm.chat_completion.called

    def test_arrival_game_over_beats_user_move(self):
        # Same fivefold position + a user_move: report the ending, not
        # a confusing "Illegal move" against an empty legal list.
        moves = ["Nf3", "Nf6", "Ng1", "Ng8"] * 4
        client, llm, _ = _make_app([])
        res = client.post("/api/game/move",
                          json={"fen": _fen_after(moves), "history": moves,
                                "user_move": "e4"})
        body = res.json()
        assert res.status_code == 200
        assert body["game_over"] is True
        assert body["user_move"] is None
        assert not llm.chat_completion.called

    def test_fifty_move_rule_surfaces_claim(self):
        # Halfmove clock ≥ 100 → claimable, but NOT auto-ended.
        fen = "7k/8/8/4Q3/8/8/8/K7 w - - 100 80"
        client, _, _ = _make_app(["MOVE: Qe4"])
        res = client.post("/api/game/move", json={"fen": fen})
        body = res.json()
        assert body["game_over"] is False
        assert body["can_claim_draw"] is True

    def test_inconsistent_history_falls_back_to_fen(self):
        # History that doesn't reproduce the FEN must not break anything.
        fen = _fen_after(["e4"])
        client, _, _ = _make_app(["MOVE: Nf3"])
        res = client.post("/api/game/move",
                          json={"fen": fen, "history": ["d4"],
                                "user_move": "e5"})
        assert res.status_code == 200
        assert res.json()["user_move"] == "e5"

    def test_castling_through_attacked_square_rejected(self):
        # Black rook on g-file attacks g1 — O-O must be illegal.
        fen = "4k3/8/8/8/8/8/6r1/R3K2R w KQ - 0 1"
        client, _, _ = _make_app([])
        res = client.post("/api/game/move",
                          json={"fen": fen, "user_move": "O-O"})
        assert res.status_code == 422
        assert "Illegal move" in res.json()["detail"]
