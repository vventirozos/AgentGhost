"""Participant-mode game endpoint: the agent PLAYS, it does not delegate.

POST /api/game/move makes "play against the agent" literal: the client
sends a position, the AGENT (its LLM, at inference time) chooses the reply
move. It is game-agnostic — the endpoint owns the protocol and each game's
rules live in a :class:`~ghost_agent.api.games.GameAdapter` (chess,
tic-tac-toe, …); ``game`` in the request selects one (default ``"chess"``).

The adapter validates legality with CODE — it never picks the move. If the
model cannot produce a legal move after bounded retries the endpoint returns
502 rather than falling back to an engine, because a silent engine fallback
is exactly the "random AI" failure this endpoint was built to prevent (chess
incident, 2026-07-02).

Stateless by design: the client owns the game (sends the serialized state +
optional history), so any UI — or the chat loop itself — can drive a game
without server-side session storage. Chess replays the SAN history to make
repetition rules detectable across the stateless boundary; other games that
need history do the same in their adapter's ``load``.
"""

import logging
import secrets
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from .routes import _mark_foreground, api_key_header, get_agent, verify_api_key
from .games import (
    GameDependencyError,
    GameStateError,
    available_games,
    extract_comment,
    extract_critique,
    extract_explanation,
    extract_move_text,
    get_adapter,
)
from ..utils.logging import Icons, pretty_log

logger = logging.getLogger("GhostAgent")

_service_token_header = APIKeyHeader(name="X-Ghost-Service-Token",
                                     auto_error=False)


async def verify_game_access(
        request: Request,
        api_key: str = Security(api_key_header),
        service_token: str = Security(_service_token_header)):
    """Master API key OR a live supervisor-minted service token.

    Sandbox-hosted apps (the chess coach) are participant clients by
    design — they call /api/game/move for the agent's moves/coaching —
    but they must never hold the master key inside the sandbox. The
    ServiceSupervisor mints $GHOST_SERVICE_TOKEN at service start
    (2026-07-30; the 2026-07-13 auth rollout had silently orphaned this
    app class), and this dependency accepts it for the GAME routes only:
    a token grants participant-game compute, nothing else. Validity is
    registry-driven — stop revokes. Failures fall through to
    verify_api_key so the standard auth-rejected logging + 403 applies.
    """
    agent = get_agent(request)
    configured = agent.context.args.api_key
    if not configured:
        return None
    if api_key and secrets.compare_digest(
            str(api_key).encode("utf-8", "ignore"),
            str(configured).encode("utf-8", "ignore")):
        return api_key
    if service_token:
        from ..sandbox.services import valid_service_token
        if valid_service_token(
                getattr(agent.context, "sandbox_manager", None),
                service_token):
            return service_token
    return await verify_api_key(request, api_key)


game_router = APIRouter(prefix="/api/game",
                        dependencies=[Security(verify_game_access)])

_MAX_MOVE_ATTEMPTS = 3


class GameMoveRequest(BaseModel):
    game: str = Field("chess",
                      description="Which turn-based game to play (default "
                                  "chess). See GET /api/game/games.")
    state: Optional[str] = Field(
        None, description="Serialized game state to move FROM (chess: a FEN; "
                          "tic-tac-toe: 9 cells + side to move). Omit to "
                          "start from the game's initial position.")
    fen: Optional[str] = Field(
        None, description="[chess] Backward-compatible alias for `state`.")
    user_move: Optional[str] = Field(
        None, description="The user's move, applied to the state before the "
                          "agent replies. Omit if the state is already the "
                          "agent-to-move position.")
    history: List[str] = Field(
        default_factory=list,
        description="Move list so far (context for the agent; some games "
                    "also need it to detect repetition endings).")


@game_router.get("/games")
async def list_games():
    """The registered turn-based games this endpoint can play."""
    return {"games": available_games(), "default": "chess"}


@game_router.post("/move")
async def game_move(req: GameMoveRequest, request: Request):
    adapter = get_adapter(req.game)
    if adapter is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown game {req.game!r}. Available: "
                   f"{available_games()}.")
    agent = get_agent(request)

    # `state` is canonical; `fen` is the chess back-compat alias; missing
    # state means "start a new game from the initial position".
    state = req.state if req.state is not None else req.fen
    if state is not None and not str(state).strip():
        # Same failure class as the empty `user_move` guard below: a client
        # that cleared its state field sent "", which tic-tac-toe's `load`
        # silently reinterpreted as "new game" — discarding the user's
        # in-progress board and returning 200. Chess 422s on the same input,
        # so the two games disagreed. Omission must be EXPLICIT.
        raise HTTPException(
            status_code=422,
            detail="Empty 'state'. Send the current position, or OMIT the "
                   "'state' field entirely to start a new game.")
    try:
        if state is None:
            state = adapter.initial_state()
        game_state = adapter.load(state, list(req.history or []))
    except GameDependencyError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except GameStateError as e:
        raise HTTPException(status_code=422, detail=str(e))

    history = list(req.history or [])

    if adapter.is_over(game_state):
        # Already over on ARRIVAL (e.g. a repetition/mate completed with the
        # previous reply) — report the ending; applying user_move here would
        # produce a confusing "Illegal move" against an empty legal list.
        return {"ok": True, "game": adapter.name, "user_move": None,
                "move": None, "state": adapter.serialize(game_state),
                "history": history, **adapter.status(game_state)}

    if req.user_move is not None and not req.user_move.strip():
        # An empty user_move is always a client bug. Observed live
        # (2026-07-05, chess): a client forwarded a bare Enter keypress as
        # "", a falsy check treated it as "field omitted", and the agent
        # silently moved for whichever side was to move — playing both
        # sides. Omission must be EXPLICIT (send no field), not "".
        raise HTTPException(
            status_code=422,
            detail="Empty user_move. Send your move, or OMIT the 'user_move' "
                   "field entirely if you want the agent to move from this "
                   "position.")

    user_move_norm = None
    if req.user_move:
        applied = adapter.apply_move(game_state, req.user_move)
        if applied is None:
            sample = adapter.legal_examples(game_state)
            raise HTTPException(
                status_code=422,
                detail=f"Illegal move '{req.user_move}'. Examples of legal "
                       f"moves here: {sample}")
        user_move_norm = applied.notation
        history.append(user_move_norm)

    if adapter.is_over(game_state):
        return {"ok": True, "game": adapter.name, "user_move": user_move_norm,
                "move": None, "state": adapter.serialize(game_state),
                "history": history, **adapter.status(game_state)}

    llm = getattr(getattr(agent, "context", None), "llm_client", None)
    if llm is None:
        raise HTTPException(status_code=503, detail="LLM client unavailable")

    feedback = ""
    reply_text = ""
    for attempt in range(1, _MAX_MOVE_ATTEMPTS + 1):
        payload = {
            "model": "default",
            "messages": [{"role": "user",
                          "content": adapter.prompt(game_state, history,
                                                    feedback,
                                                    user_move=user_move_norm)}],
            "stream": False,
            "temperature": 0.6,
        }
        # Mark this as an ACTIVE USER REQUEST for its whole duration (§4Q
        # Lens-D). Without it, `foreground_requests` stayed 0 on this path, so
        # a parked background call in `_wait_for_foreground_clear` saw
        # "no request active", gave up after its 30s budget and took the main
        # model slot — the user's next move then queued behind a full
        # background generation. That is exactly the starvation the
        # foreground_requests counter was introduced to prevent, and this was
        # the one live-user endpoint that never set it.
        _mark_foreground(agent, +1)
        try:
            data = await llm.chat_completion(payload)
            reply_text = str(
                (data.get("choices") or [{}])[0]
                .get("message", {}).get("content") or "")
        except Exception as e:
            logger.warning("game_move LLM call failed: %s", e)
            raise HTTPException(status_code=502,
                                detail=f"LLM request failed: {e}")
        finally:
            _mark_foreground(agent, -1)
        move_text = extract_move_text(reply_text)
        applied = adapter.apply_move(game_state, move_text) if move_text else None
        if applied is not None:
            history.append(applied.notation)
            # Pedagogical text for a learning client: `critique` is the
            # agent's read on the OPPONENT's move (only when they just
            # moved); `move_explanation` is the reasoning behind the agent's
            # OWN move. `comment` (the short friendly line) is kept for
            # backward compatibility with older clients.
            comment = extract_comment(reply_text)
            explanation = extract_explanation(reply_text)
            critique = extract_critique(reply_text) if user_move_norm else ""
            pretty_log("Game Move",
                       f"[{adapter.name}] {applied.notation} "
                       f"(attempt {attempt}) state={adapter.serialize(game_state)}",
                       icon=Icons.GAME_MOVE)
            return {"ok": True, "game": adapter.name,
                    "user_move": user_move_norm, "move": applied.notation,
                    "comment": comment, "move_explanation": explanation,
                    "critique": critique,
                    "state": adapter.serialize(game_state),
                    "history": history, "attempts": attempt,
                    **applied.extras, **adapter.status(game_state)}
        feedback = (f"Your previous reply ('{(move_text or reply_text)[:60]}') "
                    f"was not a legal move. Pick one move EXACTLY as written "
                    f"in the legal list.\n")

    # No legal move after bounded retries. Refuse honestly — an engine
    # fallback here would silently reintroduce the "random AI" the user
    # explicitly forbade.
    pretty_log("Game Move",
               f"[{adapter.name}] no legal move after "
               f"{_MAX_MOVE_ATTEMPTS} attempts",
               icon=Icons.GAME_MOVE, level="WARNING")
    raise HTTPException(
        status_code=502,
        detail=f"The agent could not produce a legal move in "
               f"{_MAX_MOVE_ATTEMPTS} attempts (last reply: "
               f"{reply_text[:200]!r}). Retry the request.")
