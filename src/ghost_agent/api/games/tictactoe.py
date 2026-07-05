"""Tic-tac-toe adapter — a small, dependency-free second game that proves
the move endpoint is genuinely game-agnostic, not chess-with-extra-steps.

State string: nine cells in reading order (``X``/``O``/``.``) followed by
the side to move, e.g. ``"......... X"``. Cells are numbered 1-9 in reading
order (top-left = 1, bottom-right = 9); a move is that number (``"a1".."c3"``
coordinates are also accepted). Same contract as every adapter: code
enforces legality and detects the ending, the LLM only chooses.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import AppliedMove, GameAdapter, GameStateError

_LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
          (0, 3, 6), (1, 4, 7), (2, 5, 8),   # columns
          (0, 4, 8), (2, 4, 6)]              # diagonals


class _Board:
    __slots__ = ("cells", "turn")

    def __init__(self, cells: List[str], turn: str):
        self.cells = cells      # 9 chars, each 'X' | 'O' | '.'
        self.turn = turn        # 'X' | 'O' (X moves first)


class TicTacToeAdapter(GameAdapter):
    name = "tictactoe"
    display_name = "tic-tac-toe"

    def initial_state(self) -> str:
        return "......... X"

    def load(self, state: str, history: List[str]) -> Any:
        s = (state or "").strip() or self.initial_state()
        parts = s.split()
        grid = parts[0] if parts else ""
        turn = parts[1].upper() if len(parts) > 1 else None
        if len(grid) != 9 or any(c not in "XO." for c in grid):
            raise GameStateError(
                "Bad tic-tac-toe state: expected 9 cells of X/O/. optionally "
                "followed by the side to move (X or O), e.g. '......... X'. "
                f"Got: {state!r}")
        cells = list(grid)
        nx, no = cells.count("X"), cells.count("O")
        # X moves first, so X's count is equal to or one more than O's.
        if nx - no not in (0, 1):
            raise GameStateError(
                f"Impossible tic-tac-toe position: {nx} X vs {no} O "
                "(X moves first, so counts differ by at most one).")
        if turn is None:
            turn = "X" if nx == no else "O"      # infer from parity
        elif turn not in ("X", "O"):
            raise GameStateError(
                f"Bad side to move {turn!r}: expected 'X' or 'O'.")
        return _Board(cells, turn)

    def serialize(self, b: _Board) -> str:
        return "".join(b.cells) + " " + b.turn

    def _winner(self, cells: List[str]) -> Optional[str]:
        for a, c, d in _LINES:
            if cells[a] != "." and cells[a] == cells[c] == cells[d]:
                return cells[a]
        return None

    def is_over(self, b: _Board) -> bool:
        return self._winner(b.cells) is not None or "." not in b.cells

    def status(self, b: _Board) -> Dict[str, Any]:
        w = self._winner(b.cells)
        full = "." not in b.cells
        if w == "X":
            result, term = "1-0", "X_WINS"
        elif w == "O":
            result, term = "0-1", "O_WINS"
        elif full:
            result, term = "1/2-1/2", "DRAW"
        else:
            result, term = None, None
        return {
            "game_over": w is not None or full,
            "result": result,
            "termination": term,
            "can_claim_draw": False,
            "turn": b.turn,
        }

    def _parse_cell(self, move_text: str) -> Optional[int]:
        t = (move_text or "").strip().lower()
        if t.isdigit() and 1 <= int(t) <= 9:
            return int(t) - 1
        if len(t) == 2 and t[0] in "abc" and t[1] in "123":
            return (int(t[1]) - 1) * 3 + "abc".index(t[0])
        return None

    def apply_move(self, b: _Board, move_text: str) -> Optional[AppliedMove]:
        idx = self._parse_cell(move_text)
        if idx is None or b.cells[idx] != ".":
            return None
        b.cells[idx] = b.turn
        b.turn = "O" if b.turn == "X" else "X"
        return AppliedMove(notation=str(idx + 1))

    def legal_examples(self, b: _Board, limit: int = 12) -> List[str]:
        return [str(i + 1) for i, c in enumerate(b.cells) if c == "."][:limit]

    def _render(self, cells: List[str]) -> str:
        rows = []
        for r in range(3):
            cells_r = [cells[r * 3 + c] if cells[r * 3 + c] != "."
                       else str(r * 3 + c + 1) for c in range(3)]
            rows.append(" " + " | ".join(cells_r) + " ")
        return "\n---+---+---\n".join(rows)

    def prompt(self, b: _Board, history: List[str], feedback: str = "",
               user_move=None) -> str:
        legal = self.legal_examples(b, limit=9)
        opp = (f"Your opponent just played cell {user_move}.\n"
               if user_move else "")
        critique_line = (
            "- 'CRITIQUE: <assess your opponent's last move — did it block "
            "you, build a threat, or miss a better cell; teach, do not just "
            "label it>'\n" if user_move else "")
        return (
            f"You are playing tic-tac-toe as '{b.turn}' against your user. "
            f"YOU are the player — choose the move yourself; there is no "
            f"engine behind you.\n"
            f"Cells are numbered 1-9 in reading order (top-left = 1, "
            f"bottom-right = 9). Current board (a number = that empty "
            f"cell):\n{self._render(b.cells)}\n"
            f"{opp}"
            f"Your legal moves (empty cells): {', '.join(legal)}\n"
            f"{feedback}"
            f"Reply with these lines, each label on its own line:\n"
            f"- 'MOVE: <cell-number>' choosing one empty cell\n"
            f"{critique_line}"
            f"- 'EXPLANATION: <the idea behind YOUR move in one sentence — "
            f"the line you are making or blocking>'\n"
            f"- optionally 'COMMENT: <one short friendly sentence to your "
            f"opponent>'."
        )
