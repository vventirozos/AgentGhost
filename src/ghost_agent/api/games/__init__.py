"""Registry of turn-based game adapters for POST /api/game/move.

Adding a game = write a :class:`GameAdapter` and ``register`` it here. The
endpoint dispatches on the request's ``game`` field (default ``"chess"`` for
backward compatibility) and needs no change per game.

Adapters are constructed lazily-safe: a chess adapter can be registered even
when python-chess is absent — the missing dependency only surfaces (as 501)
when a chess request is actually served.
"""

from typing import Dict, List, Optional

from .base import (
    AppliedMove,
    GameAdapter,
    GameDependencyError,
    GameStateError,
    extract_comment,
    extract_critique,
    extract_explanation,
    extract_labeled,
    extract_move_text,
)
from .chess_game import ChessAdapter
from .tictactoe import TicTacToeAdapter

_ADAPTERS: Dict[str, GameAdapter] = {}


def register(adapter: GameAdapter) -> None:
    _ADAPTERS[adapter.name] = adapter


def get_adapter(name: Optional[str]) -> Optional[GameAdapter]:
    return _ADAPTERS.get((name or "").strip().lower())


def available_games() -> List[str]:
    return sorted(_ADAPTERS)


register(ChessAdapter())
register(TicTacToeAdapter())

__all__ = [
    "AppliedMove",
    "GameAdapter",
    "GameDependencyError",
    "GameStateError",
    "extract_comment",
    "extract_critique",
    "extract_explanation",
    "extract_labeled",
    "extract_move_text",
    "get_adapter",
    "available_games",
    "register",
    "ChessAdapter",
    "TicTacToeAdapter",
]
