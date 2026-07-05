"""Turn-based game adapters for POST /api/game/move.

The move endpoint is game-agnostic: it owns the *protocol* (stateless,
client-owns-the-state, code-enforced legality, the AGENT's LLM chooses the
move, bounded retries, NO engine fallback) and delegates every rule of a
particular game to a ``GameAdapter``. Adding a game is adding an adapter and
registering it — the endpoint never changes.

The one invariant every adapter must honour is the reason this endpoint
exists (chess incident, 2026-07-02): the adapter validates legality and
detects endings with CODE, but it NEVER chooses a move. Move selection is
the LLM's job at inference time; ``apply_move`` only accepts or rejects the
text the model produced.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, NamedTuple, Optional

# ``MOVE:`` line wins; the last one wins because thinking models restate
# their final choice at the end. We capture the first whitespace-delimited
# token after ``MOVE:`` — every notation this serves (chess SAN/UCI,
# castling ``O-O``, a tic-tac-toe cell number) is space-free, so this
# tolerates a trailing parenthetical the model may add.
_MOVE_LINE_RE = re.compile(r"MOVE:\s*(\S+)")


class GameStateError(ValueError):
    """The client sent an unparseable/impossible state (→ HTTP 422)."""


class GameDependencyError(RuntimeError):
    """A library the adapter needs is not installed (→ HTTP 501)."""


class AppliedMove(NamedTuple):
    """Result of applying a legal move: ``notation`` is appended to the
    history; ``extras`` are merged into the response envelope (e.g. chess
    adds ``move_uci``)."""
    notation: str
    extras: Dict[str, Any] = {}


def extract_move_text(reply: str) -> Optional[str]:
    """Pull the chosen move token out of an LLM reply. Last ``MOVE:`` line
    wins; failing that, tolerate a bare short token as the whole reply."""
    matches = _MOVE_LINE_RE.findall(reply or "")
    if matches:
        return matches[-1]
    bare = (reply or "").strip().splitlines()[-1].strip() if reply else ""
    if bare and len(bare) <= 12 and " " not in bare:
        return bare
    return None


def extract_labeled(reply: str, label: str) -> str:
    """First ``LABEL: <rest of line>`` value in ``reply`` (stripped), or ""."""
    m = re.search(re.escape(label) + r":\s*(.+)", reply or "")
    return m.group(1).strip() if m else ""


def extract_comment(reply: str) -> str:
    """A short friendly one-liner (``COMMENT:``). Kept for backward
    compatibility; the pedagogical fields below are the learning surface."""
    return extract_labeled(reply, "COMMENT")


def extract_critique(reply: str) -> str:
    """The agent's assessment of the OPPONENT's last move (``CRITIQUE:``) —
    was it sound, what it threatens/misses, a stronger idea."""
    return extract_labeled(reply, "CRITIQUE")


def extract_explanation(reply: str) -> str:
    """The agent's reasoning about ITS OWN move (``EXPLANATION:``) — the plan
    or threat behind it. Surfaced as ``move_explanation``."""
    return extract_labeled(reply, "EXPLANATION")


class GameAdapter(ABC):
    """One turn-based game. All methods operate on an opaque *game-state*
    object the adapter itself produces from :meth:`load` — the endpoint
    never inspects it, only passes it back."""

    #: registry key used in the request's ``game`` field (lower-case).
    name: str = ""
    #: human-facing name used in prompts / errors.
    display_name: str = ""

    @abstractmethod
    def initial_state(self) -> str:
        """Serialized starting position, used when the client omits state."""

    @abstractmethod
    def load(self, state: str, history: List[str]) -> Any:
        """Build a game-state object from a serialized state (+ optional
        move history for rules that need it, e.g. chess repetition).

        Raise :class:`GameStateError` on an unparseable/impossible state and
        :class:`GameDependencyError` if a required library is missing."""

    @abstractmethod
    def serialize(self, game_state: Any) -> str:
        """Serialize the state object back to the string the client stores."""

    @abstractmethod
    def is_over(self, game_state: Any) -> bool:
        """True if the game has reached a MANDATORY ending."""

    @abstractmethod
    def status(self, game_state: Any) -> Dict[str, Any]:
        """Status dict merged into the response: must include ``game_over``,
        ``result``, ``termination``, ``turn``; may add game-specific keys."""

    @abstractmethod
    def apply_move(self, game_state: Any, move_text: str) -> Optional[AppliedMove]:
        """Validate ``move_text`` for the side to move; if legal, MUTATE
        ``game_state`` to apply it and return the :class:`AppliedMove`.
        Return ``None`` if the move is illegal/unparseable. NEVER choose a
        move here — only accept or reject the given one."""

    @abstractmethod
    def legal_examples(self, game_state: Any, limit: int = 12) -> List[str]:
        """A few legal moves, for the illegal-move error message."""

    @abstractmethod
    def prompt(self, game_state: Any, history: List[str], feedback: str = "",
               user_move: Optional[str] = None) -> str:
        """The instruction shown to the LLM to elicit its reply. Must ask for
        a ``MOVE:`` line and an ``EXPLANATION:`` line (the reasoning behind
        the agent's own move); when ``user_move`` is given (the opponent just
        moved), also ask for a ``CRITIQUE:`` line assessing that move. These
        power the learning-app fields ``move_explanation`` and ``critique``."""
