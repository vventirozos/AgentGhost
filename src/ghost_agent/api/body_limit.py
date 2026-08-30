"""ASGI request-body size cap for the AGENT app (§4DW, 2026-08-29).

The interface server has had `BodySizeLimitMiddleware` for a while; the
agent — the process that binds 0.0.0.0 when a key is configured, and the
one holding every model, index and sandbox handle — had NOTHING. Measured
on 2026-08-29: a single 150 MB POST to `/api/upload` took the live daemon's
RSS from 509 MB to 960 MB, and the body was fully received before any
handler could look at it.

A handler-level cap cannot fix that. Starlette parses the ENTIRE multipart
body (spooling to disk) before the endpoint function is entered, and
`request.json()` reads the whole body into RAM first. By the time
`_read_capped` runs, the bytes are already here. The cap has to sit at the
ASGI layer, in front of parsing:

  * declared `Content-Length` over the cap  -> 413 immediately, body never
    read;
  * chunked / undeclared                    -> count bytes as they arrive
    and abort the moment the running total crosses the cap.

Deliberately modelled on the interface's proven implementation rather than
invented fresh, including the two non-obvious parts: `_BodyTooLarge` is a
BaseException so it survives FastAPI's body-parsing `except Exception` and
the handlers' own broad excepts, and DELETE is capped because the agent
reads DELETE bodies (`/api/delete`).

Deliberately NOT shared with the interface's copy: the two servers are
separate processes with separate launchers, the interface runs as a plain
module with `cwd=interface/` and cannot import `ghost_agent`, and their
path tables differ (the interface has `/api/stt`, which the agent has no
route for). `tests/test_http_surface_hardening_4dw.py` pins the two against
one shared corpus so they cannot drift apart silently.
"""

from __future__ import annotations

import os

from fastapi.responses import JSONResponse


def _env_num(name: str, default: int) -> int:
    """Env override that cannot crash the process at IMPORT.

    This module is imported by `create_app()` inside a LaunchDaemon with
    KeepAlive. A typo'd or EMPTY value (`export X="$UNSET"`) raising here
    would be an import-time crash -> exit -> relaunch -> same crash,
    forever, with no endpoint left to report the outage. Same reasoning,
    and same idiom, as `interface/server.py::_env_num`.
    """
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        val = int(str(raw).strip())
    except (TypeError, ValueError):
        return int(default)
    return val if val > 0 else int(default)


def max_upload_bytes() -> int:
    return _env_num("GHOST_AGENT_MAX_UPLOAD_BYTES", 100 * 1024 * 1024)


def max_json_bytes() -> int:
    return _env_num("GHOST_AGENT_MAX_JSON_BYTES", 10 * 1024 * 1024)


def max_workspace_json_bytes() -> int:
    """A workspace save carries a whole conversation, so it legitimately
    needs more than a chat turn — but bounded, and parsed as JSON."""
    return _env_num("GHOST_AGENT_MAX_WORKSPACE_JSON_BYTES", 64 * 1024 * 1024)


# Multipart framing (boundaries, part headers) rides on top of the file
# bytes, so upload paths get slack above the file cap — otherwise a file at
# exactly the cap is rejected for its envelope.
UPLOAD_CAP_SLACK_BYTES = 1024 * 1024

# Paths that carry a FILE, not JSON.
UPLOAD_PATHS = frozenset({"/api/upload", "/api/workspace/load"})

# Paths whose JSON is legitimately larger than a chat turn.
JSON_CAP_OVERRIDES = {"/api/workspace/save": max_workspace_json_bytes}

# The methods that can carry a body here. DELETE is included because
# `/api/delete` reads one.
_CAPPED_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


class BodyTooLarge(BaseException):
    """Raised by the counting receive when a body overflows.

    A BaseException on purpose: it must sail through FastAPI's body-parsing
    `except Exception` (which would remap it to a generic 400) and through
    the handlers' own broad `except Exception` blocks (which would remap it
    to a 502) so the middleware can convert it into a proper 413.
    """

    def __init__(self, cap: int):
        self.cap = cap


def _normalise(path) -> str:
    """Collapse the spellings that name the same route.

    `UPLOAD_PATHS` is an exact-match set, so `/api/upload/` and
    `//api/upload` fell through to the 10 MB JSON cap and 413'd a
    legitimate upload — measured, both spellings, against a live server.
    Starlette would have redirected the first to the real route; the
    middleware runs before the router and never got there.

    Only the two spellings a client actually produces are collapsed
    (duplicate separators and one trailing slash). Deliberately NOT a
    `..`-resolving normaliser: this function hands out a LARGER cap for
    upload paths, so a normaliser more permissive than the router's would
    let a JSON route claim the 100 MB ceiling.
    """
    p = str(path or "")
    while "//" in p:
        p = p.replace("//", "/")
    if len(p) > 1 and p.endswith("/"):
        p = p[:-1]
    return p


def cap_for_path(path) -> int:
    """The byte ceiling for one request path."""
    path = _normalise(path)
    if path in UPLOAD_PATHS:
        return max_upload_bytes() + UPLOAD_CAP_SLACK_BYTES
    override = JSON_CAP_OVERRIDES.get(path)
    if override is not None:
        return override()
    return max_json_bytes()


class BodySizeLimitMiddleware:
    """Reject oversized request bodies BEFORE any parsing happens."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (scope.get("type") != "http"
                or scope.get("method") not in _CAPPED_METHODS):
            await self.app(scope, receive, send)
            return

        cap = cap_for_path(scope.get("path"))

        declared = None
        for name, value in scope.get("headers") or ():
            if name == b"content-length":
                try:
                    declared = int(value)
                except (TypeError, ValueError):
                    declared = None
                break
        if declared is not None and declared > cap:
            await JSONResponse(
                {"error": f"Request body too large "
                          f"({declared} > {cap} byte cap)"},
                status_code=413,
            )(scope, receive, send)
            return

        received = 0
        response_started = False

        async def counting_receive():
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body") or b"")
                if received > cap:
                    raise BodyTooLarge(cap)
            return message

        async def tracking_send(message):
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, counting_receive, tracking_send)
        except BodyTooLarge as exc:
            # A clean 413 is only possible while the app has not started
            # responding; otherwise re-raise and let the connection drop —
            # the alternative is a corrupt half-response.
            if response_started:
                raise
            await JSONResponse(
                {"error": f"Request body too large (> {exc.cap} byte cap)"},
                status_code=413,
            )(scope, receive, send)
