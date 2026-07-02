from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import router
from .projects_routes import projects_router
from .game_routes import game_router

def create_app():
    app = FastAPI()
    # CORS: `allow_origins=["*"]` and `allow_credentials=True` are mutually
    # exclusive per the CORS spec — every modern browser rejects the request.
    # We don't use cookies for auth (the API key rides in `X-Ghost-Key`), so
    # `allow_credentials=False` is correct AND keeps the wildcard origin
    # working for browser clients.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
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
