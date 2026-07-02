"""Participant-mode game endpoint: the agent PLAYS, it does not delegate.

POST /api/game/move exists so a browser chess board can make "play against
the agent" literal: the page sends the position, the AGENT (its LLM, at
inference time) chooses the reply move. python-chess is used strictly for
legality — validating the user's move, listing legal replies, applying the
chosen one. It never picks the move; if the model cannot produce a legal
move after bounded retries the endpoint returns 502 rather than falling
back to an engine, because a silent engine fallback is exactly the
"random AI" failure this endpoint was built to prevent (chess incident,
2026-07-02).

Stateless by design: the client owns the game (sends FEN + optional
history), so any UI — or the chat loop itself — can drive a game without
server-side session storage.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, Security
from pydantic import BaseModel, Field

from .routes import get_agent, verify_api_key
from ..utils.logging import Icons, pretty_log

logger = logging.getLogger("GhostAgent")

game_router = APIRouter(prefix="/api/game",
                        dependencies=[Security(verify_api_key)])

_MAX_MOVE_ATTEMPTS = 3
_MOVE_LINE_RE = re.compile(r"MOVE:\s*([A-Za-z0-9+#=\-]+)")
_COMMENT_LINE_RE = re.compile(r"COMMENT:\s*(.+)")


class GameMoveRequest(BaseModel):
    fen: str = Field(..., description="Position BEFORE user_move is applied "
                                      "(or the position to move from).")
    user_move: Optional[str] = Field(
        None, description="The user's move in SAN or UCI, applied to fen "
                          "before the agent replies. Omit if fen is already "
                          "the agent-to-move position.")
    history: List[str] = Field(
        default_factory=list,
        description="SAN move list so far (context for the agent).")


def _apply_move(board, move_text: str):
    """Parse SAN first, then UCI. Returns the Move or None."""
    import chess
    for parser in (board.parse_san, chess.Move.from_uci):
        try:
            move = parser(move_text.strip())
            if move in board.legal_moves:
                return move
        except (ValueError, AssertionError):
            continue
    return None


def _extract_move_text(reply: str) -> Optional[str]:
    """Last MOVE: line wins (thinking models restate their choice at the
    end); tolerate a bare move as the whole reply."""
    matches = _MOVE_LINE_RE.findall(reply or "")
    if matches:
        return matches[-1]
    bare = (reply or "").strip().splitlines()[-1].strip() if reply else ""
    if bare and len(bare) <= 8 and " " not in bare:
        return bare
    return None


def _game_status(board) -> Dict[str, Any]:
    return {
        "game_over": board.is_game_over(),
        "result": board.result() if board.is_game_over() else None,
        "check": board.is_check(),
        "turn": "white" if board.turn else "black",
    }


def _move_prompt(board, history: List[str], feedback: str = "") -> str:
    color = "White" if board.turn else "Black"
    legal_san = [board.san(m) for m in board.legal_moves]
    hist = " ".join(history[-40:]) if history else "(game start)"
    prompt = (
        f"You are playing a chess game as {color} against your user. YOU "
        f"are the player — choose the move yourself, by judgement; there "
        f"is no engine behind you.\n"
        f"Position (FEN): {board.fen()}\n"
        f"Board (uppercase = White):\n{board}\n"
        f"Moves so far (SAN): {hist}\n"
        f"Your legal moves: {', '.join(legal_san)}\n"
        f"{feedback}"
        f"Reply with EXACTLY one line 'MOVE: <san>' choosing one move from "
        f"the legal list, optionally followed by one line "
        f"'COMMENT: <one short friendly sentence to your opponent>'."
    )
    return prompt


@game_router.post("/move")
async def game_move(req: GameMoveRequest, request: Request):
    try:
        import chess
    except ImportError:
        raise HTTPException(status_code=501,
                            detail="python-chess is not installed")
    agent = get_agent(request)

    try:
        board = chess.Board(req.fen)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Bad FEN: {e}")

    history = list(req.history or [])
    user_move_san = None
    if req.user_move:
        move = _apply_move(board, req.user_move)
        if move is None:
            sample = [board.san(m) for m in list(board.legal_moves)[:12]]
            raise HTTPException(
                status_code=422,
                detail=f"Illegal move '{req.user_move}'. Examples of legal "
                       f"moves here: {sample}")
        user_move_san = board.san(move)
        board.push(move)
        history.append(user_move_san)

    if board.is_game_over():
        return {"ok": True, "user_move": user_move_san, "move": None,
                "fen": board.fen(), "history": history,
                **_game_status(board)}

    llm = getattr(getattr(agent, "context", None), "llm_client", None)
    if llm is None:
        raise HTTPException(status_code=503, detail="LLM client unavailable")

    feedback = ""
    reply_text = ""
    for attempt in range(1, _MAX_MOVE_ATTEMPTS + 1):
        payload = {
            "model": "default",
            "messages": [{"role": "user",
                          "content": _move_prompt(board, history, feedback)}],
            "stream": False,
            "temperature": 0.6,
        }
        try:
            data = await llm.chat_completion(payload)
            reply_text = str(
                (data.get("choices") or [{}])[0]
                .get("message", {}).get("content") or "")
        except Exception as e:
            logger.warning("game_move LLM call failed: %s", e)
            raise HTTPException(status_code=502,
                                detail=f"LLM request failed: {e}")
        move_text = _extract_move_text(reply_text)
        move = _apply_move(board, move_text) if move_text else None
        if move is not None:
            move_san = board.san(move)
            move_uci = move.uci()
            board.push(move)
            history.append(move_san)
            comment_match = _COMMENT_LINE_RE.search(reply_text)
            comment = comment_match.group(1).strip() if comment_match else ""
            pretty_log("Game Move",
                       f"{move_san} (attempt {attempt}) fen={board.fen()}",
                       icon=Icons.GAME_MOVE)
            return {"ok": True, "user_move": user_move_san,
                    "move": move_san, "move_uci": move_uci,
                    "comment": comment, "fen": board.fen(),
                    "history": history, "attempts": attempt,
                    **_game_status(board)}
        feedback = (f"Your previous reply ('{(move_text or reply_text)[:60]}') "
                    f"was not a legal move. Pick one move EXACTLY as written "
                    f"in the legal list.\n")

    # No legal move after bounded retries. Refuse honestly — an engine
    # fallback here would silently reintroduce the "random AI" the user
    # explicitly forbade.
    pretty_log("Game Move",
               f"no legal move after {_MAX_MOVE_ATTEMPTS} attempts",
               icon=Icons.GAME_MOVE, level="WARNING")
    raise HTTPException(
        status_code=502,
        detail=f"The agent could not produce a legal move in "
               f"{_MAX_MOVE_ATTEMPTS} attempts (last reply: "
               f"{reply_text[:200]!r}). Retry the request.")
