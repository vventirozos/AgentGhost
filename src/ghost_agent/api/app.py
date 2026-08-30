from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .body_limit import BodySizeLimitMiddleware
from .routes import router, verify_api_key
from .projects_routes import projects_router
from .game_routes import game_router

def create_app():
    # ⚠ The interactive docs are OFF and the schema is AUTHENTICATED.
    #
    # FastAPI's defaults publish `/docs`, `/redoc` and `/openapi.json` to
    # anyone who can open the port — and with a key configured this app
    # binds 0.0.0.0, so that is the whole tailnet. The schema is a complete
    # map of the attack surface: every route, method, path parameter and
    # body model, including the ones that write files, execute code and
    # reach the sandbox. Authentication on the endpoints is not a reason to
    # hand out the map; it is the reason the map is worth taking.
    #
    # The UIs are dropped rather than gated because this app authenticates
    # by the `X-Ghost-Key` HEADER, which a browser cannot set on a plain
    # navigation — a gated Swagger page could never load its own schema.
    # `/openapi.json` stays, behind the same dependency as everything else,
    # so a tool holding the key can still generate a client. (§4DW)
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    # CORS: `allow_origins=["*"]` and `allow_credentials=True` are mutually
    # exclusive per the CORS spec — every modern browser rejects the request.
    # We don't use cookies for auth (the API key rides in `X-Ghost-Key`), so
    # `allow_credentials=False` is correct AND keeps the wildcard origin
    # working for browser clients.
    # ⚠ Added BEFORE CORS. Starlette applies middleware in REVERSE order of
    # registration, so the one added last ends up outermost — registering
    # the body cap first keeps CORS outside it, and CORS headers therefore
    # still decorate the 413. (Same ordering, and the same reason, as
    # `interface/server.py`.)
    #
    # Until §4DW the agent had no cap at all: a single 150 MB POST to
    # `/api/upload` moved the live daemon's RSS from 509 MB to 960 MB
    # because Starlette parses the whole multipart body before any handler
    # runs.
    app.add_middleware(BodySizeLimitMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Re-added by hand so it carries the auth dependency (the built-in was
    # turned off above). Registered BEFORE every router: Starlette matches
    # in REGISTRATION order and `router` ends with a `/{path:path}`
    # catch-all proxy, so a route declared after it is unreachable — the
    # first attempt at this landed below `include_router(router)` and every
    # request for the schema went to the upstream proxy instead.
    # `app.openapi()` is evaluated per-request, so the schema still
    # describes the routers included after this point.
    @app.get("/openapi.json", include_in_schema=False,
             dependencies=[Depends(verify_api_key)])
    async def _authenticated_openapi():
        return JSONResponse(app.openapi())

    # Turning the built-in UIs off does not make them 404 here: `router`
    # ends in a `/{path:path}` catch-all PROXY, so an authenticated operator
    # asking for `/docs` got whatever llama-server answers. Answer honestly
    # instead. Still auth-gated — an anonymous caller must not learn which
    # paths exist.
    @app.get("/docs", include_in_schema=False,
             dependencies=[Depends(verify_api_key)])
    @app.get("/redoc", include_in_schema=False,
             dependencies=[Depends(verify_api_key)])
    async def _docs_disabled():
        return JSONResponse(
            {"detail": "Interactive docs are disabled. The schema is at "
                       "/openapi.json with the X-Ghost-Key header."},
            status_code=404)

    # Order matters: `projects_router` and `game_router` must be included
    # BEFORE `router`, because `router` ends with a catch-all
    # `/{path:path}` proxy that would otherwise match every
    # `/api/projects*` / `/api/game*` request before reaching the
    # dedicated routes. (Previously every project endpoint silently 404'd
    # because of this — see test_router_order_projects.)
    app.include_router(projects_router)
    app.include_router(game_router)
    app.include_router(router)
    return app
