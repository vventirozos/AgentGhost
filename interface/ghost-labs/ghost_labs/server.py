"""`python -m ghost_labs.server`: the API on <tailnet IP>:8700 and the Apps page
on 127.0.0.1:8701 (published with `tailscale serve --https=443`)."""
from __future__ import annotations

import asyncio
import logging
import sys

import uvicorn

from . import config as C

APPS_PAGE_PORT = 8701


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s", stream=sys.stdout)
    log = logging.getLogger("ghost_labs")
    cfg = C.load()
    try:
        C.resolve_bind(cfg)
    except RuntimeError as e:
        log.error("refusing to start: %s", e)
        return 2
    import docker

    from .api import create_app, create_apps_page
    from .service import Service
    try:
        from .templates import SHIPPED_DIR
        svc = Service(cfg, docker.from_env(timeout=300), SHIPPED_DIR, C.STATE_DIR,
                      extra_templates_dir=C.CONFIG_DIR / "templates")
    except RuntimeError as e:
        log.error("refusing to start: %s", e)
        return 2
    try:
        app = create_app(svc, C.api_key())
    except (RuntimeError, ValueError, OSError) as e:
        log.error("refusing to start: %s", e)
        return 2
    svc.mgr.ports.serve_page(APPS_PAGE_PORT)
    api = uvicorn.Server(uvicorn.Config(app, host=cfg.resolved_bind_host,
                                        port=int(cfg.bind_port), log_level="info", access_log=False))
    page = uvicorn.Server(uvicorn.Config(create_apps_page(svc), host="127.0.0.1", port=APPS_PAGE_PORT,
                                         log_level="warning", access_log=False))
    log.info("ghost-labs listening on http://%s:%s (apps page https://%s/)", cfg.resolved_bind_host, cfg.bind_port,
             cfg.resolved_dns_name)

    async def run():
        await asyncio.gather(api.serve(), page.serve())
    asyncio.run(run())
    svc.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
