"""Chess adapter — the original participant-mode game.

python-chess is used strictly for legality: validating the user's move,
listing legal replies, applying the chosen one, and detecting endings. It
never picks the move (the LLM does). The library is imported lazily so the
rest of the game system — and other games — work even when python-chess is
absent; a chess request then fails with 501, not an import-time crash.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import AppliedMove, GameAdapter, GameDependencyError, GameStateError


class ChessAdapter(GameAdapter):
    name = "chess"
    display_name = "chess"

    def _chess(self):
        try:
            import chess
            return chess
        except ImportError as e:
            raise GameDependencyError("python-chess is not installed") from e

    def initial_state(self) -> str:
        return self._chess().STARTING_FEN

    def load(self, state: str, history: List[str]) -> Any:
        chess = self._chess()
        try:
            board = chess.Board(state)
        except ValueError as e:
            raise GameStateError(f"Bad FEN: {e}") from e
        return self._restore_move_stack(board, list(history or []))

    def _restore_move_stack(self, board, history: List[str]):
        """Return a board equal to ``board`` but WITH its move stack, by
        replaying ``history`` (SAN, from the standard start) when the replay
        reproduces the position exactly.

        A bare FEN has no memory, so repetition-based rules (mandatory
        fivefold, claimable threefold) could never trigger across the
        stateless request boundary — the position "occurred once" every
        request. Replaying the client-supplied SAN history is what makes
        those rules real. Falls back to the FEN-only board for custom
        positions or an inconsistent history (never raises)."""
        import chess
        if not history:
            return board
        try:
            replay = chess.Board()
            for san in history:
                replay.push_san(str(san))
            if replay.fen() == board.fen():
                return replay
        except (ValueError, AssertionError):
            pass
        return board

    def serialize(self, board) -> str:
        return board.fen()

    def is_over(self, board) -> bool:
        return board.is_game_over()

    def status(self, board) -> Dict[str, Any]:
        # is_game_over() covers every MANDATORY ending: checkmate,
        # stalemate, insufficient material, seventy-five moves, fivefold
        # repetition (the last only detectable with the move stack
        # restored). CLAIMABLE draws (threefold, fifty-move) don't end the
        # game; they surface via can_claim_draw so the client can offer the
        # claim. ``fen`` is a back-compat alias for the generic ``state``.
        over = board.is_game_over()
        outcome = board.outcome() if over else None
        return {
            "game_over": over,
            "result": board.result() if over else None,
            "termination": outcome.termination.name if outcome else None,
            "can_claim_draw": bool(not over and board.can_claim_draw()),
            "check": board.is_check(),
            "turn": "white" if board.turn else "black",
            "fen": board.fen(),
        }

    def apply_move(self, board, move_text: str) -> Optional[AppliedMove]:
        """Parse SAN first, then UCI. Applies and returns the move, or None
        if illegal. Chooses nothing — only accepts the given text."""
        chess = self._chess()
        text = (move_text or "").strip()
        move = None
        for parser in (board.parse_san, chess.Move.from_uci):
            try:
                cand = parser(text)
                if cand in board.legal_moves:
                    move = cand
                    break
            except (ValueError, AssertionError):
                continue
        if move is None:
            return None
        san = board.san(move)
        uci = move.uci()
        board.push(move)
        return AppliedMove(notation=san, extras={"move_uci": uci})

    def legal_examples(self, board, limit: int = 12) -> List[str]:
        return [board.san(m) for m in list(board.legal_moves)[:limit]]

    def prompt(self, board, history: List[str], feedback: str = "",
               user_move=None) -> str:
        color = "White" if board.turn else "Black"
        legal_san = [board.san(m) for m in board.legal_moves]
        hist = " ".join(history[-40:]) if history else "(game start)"
        opp = f"Your opponent just played: {user_move}\n" if user_move else ""
        critique_line = (
            "- 'CRITIQUE: <assess your opponent's last move — is it sound, "
            "what it threatens or overlooks, and the stronger idea if there "
            "was one; teach, do not just label it good/bad>'\n"
            if user_move else "")
        return (
            f"You are playing a chess game as {color} against your user. YOU "
            f"are the player — choose the move yourself, by judgement; there "
            f"is no engine behind you.\n"
            f"Position (FEN): {board.fen()}\n"
            f"Board (uppercase = White):\n{board}\n"
            f"Moves so far (SAN): {hist}\n"
            f"{opp}"
            f"Your legal moves: {', '.join(legal_san)}\n"
            f"{feedback}"
            f"Reply with these lines, each label on its own line:\n"
            f"- 'MOVE: <san>' choosing one move EXACTLY as written in the "
            f"legal list\n"
            f"{critique_line}"
            f"- 'EXPLANATION: <the idea behind YOUR move in one or two "
            f"sentences — the plan, threat, or defence it serves>'\n"
            f"- optionally 'COMMENT: <one short friendly sentence to your "
            f"opponent>'."
        )
