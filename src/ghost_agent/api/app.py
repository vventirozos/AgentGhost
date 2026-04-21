from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import router
from .projects_routes import projects_router

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
    app.include_router(router)
    app.include_router(projects_router)
    return app
