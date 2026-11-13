"""HTTP API (contract v1 §9 + v2). Every handler is a plain `def`, so FastAPI
runs it in its thread pool and Docker calls never block the event loop."""
from __future__ import annotations

import hmac
import html
import time
from typing import Any, Callable, Dict, Optional

from fastapi import Body, FastAPI, Header, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from . import redact
from .errors import LabError
from .labs import ORIGINS
from .service import Service


def _origin(value: Optional[str]) -> str:
    """X-Labs-Origin; missing or unknown counts as autopilot (fail closed, v2 §11.6)."""
    v = (value or "").strip().lower()
    return v if v in ORIGINS else "autopilot"


MIN_KEY_LEN = 16


def create_app(svc: Service, api_key: str) -> FastAPI:
    # fail closed (review 11-05 #11): an empty key file made compare_digest(b"", b"") admit every request
    if not isinstance(api_key, str) or len(api_key.strip()) < MIN_KEY_LEN:
        raise ValueError(f"the API key is empty or shorter than {MIN_KEY_LEN} characters; refusing to serve")
    api_key = api_key.strip()
    app = FastAPI(title="ghost-labs", docs_url=None, redoc_url=None, openapi_url=None)
    mgr = svc.mgr

    @app.middleware("http")
    async def auth(request: Request, call_next):
        given = request.headers.get("x-labs-key", "")
        if not given or not hmac.compare_digest(given.encode(), api_key.encode()):
            return JSONResponse(status_code=401, content={"error": "unauthorized", "message": "missing or wrong X-Labs-Key",
                                                          "hint": "send the key from /etc/ghost-labs/api_key"})
        return await call_next(request)

    @app.exception_handler(LabError)
    async def lab_error(_req: Request, e: LabError):
        return JSONResponse(status_code=e.status, content=e.body())

    @app.exception_handler(Exception)
    async def internal(_req: Request, e: Exception):
        return JSONResponse(status_code=500, content={"error": "internal", "message": f"{type(e).__name__}: {e}"[:500],
                                                      "hint": "check journalctl -u ghost-labs on the labs host"})

    def act(lab_id: str, action: str, origin: str, fn: Callable[[], Any], member: str = "", **extra: Any):
        t0 = time.time()
        try:
            out = fn()
        except LabError as e:
            svc.actions.write(lab_id, action, e.code, int((time.time() - t0) * 1000), member, origin, **extra)
            raise
        lid = lab_id or (out.get("lab_id", "") if isinstance(out, dict) else "")
        svc.actions.write(lid, action, "ok", int((time.time() - t0) * 1000), member, origin, **extra)
        return out

    O = Header(default=None, alias="X-Labs-Origin")

    # ---------------------------------------------------------------- service
    @app.get("/v1/health")
    def health():
        return svc.health()

    @app.get("/v1/templates")
    def templates():
        out = []
        for t in svc.templates.values():
            out.append({"name": t["name"], "description": t["description"], "egress": t["egress"],
                        "members": [{"name": m["name"], "image": m["image"], "mem_mb": m["mem_mb"]} for m in t["members"]],
                        "total_mem_mb": sum(m["mem_mb"] for m in t["members"]),
                        "ports": sorted({p for m in t["members"] for p in m["ports"]}), "params": t["params"],
                        **({"app": t["app"]} if t.get("app") else {})})
        return {"templates": out, "images": sorted(mgr.catalog_images())}

    @app.post("/v1/gc")
    def gc(x_origin: Optional[str] = O):
        origin = _origin(x_origin)
        if origin != "owner":
            raise LabError("owner_only", "gc is an operator action", hint="run `ghost-labs gc` on the labs host")
        return act("", "gc", origin, mgr.gc)

    # ---------------------------------------------------------------- labs
    @app.post("/v1/labs", status_code=201)
    def create(body: Dict[str, Any] = Body(...), x_origin: Optional[str] = O):
        origin = _origin(x_origin)
        body.pop("origin", None)  # the header is the origin (v2 §11.6)
        return act("", "create", origin, lambda: mgr.create(body, origin), template=str(body.get("template") or "members"))

    @app.get("/v1/labs")
    def list_labs(owner: Optional[str] = None):
        rows = [mgr.summary(r) for r in svc.store.labs() if owner is None or r["owner"] == owner]
        return {"labs": rows}

    @app.delete("/v1/labs")
    def destroy_owner(owner: str = Query(...), include_kept: int = 0, x_origin: Optional[str] = O):
        origin = _origin(x_origin)
        if include_kept and origin != "owner":
            raise LabError("owner_only", "only the owner may remove kept labs", hint="ask the owner")
        return act("", "destroy_owner", origin, lambda: mgr.destroy_owner(owner, origin, bool(include_kept)), owner=owner)

    @app.get("/v1/labs/{lab_id}")
    def detail(lab_id: str, x_origin: Optional[str] = O):
        mgr.get(lab_id)
        mgr.touch(lab_id, _origin(x_origin))
        return mgr.detail(lab_id)

    @app.delete("/v1/labs/{lab_id}")
    def destroy(lab_id: str, include_kept: int = 0, drop_data: int = 0, x_origin: Optional[str] = O):
        origin = _origin(x_origin)
        if drop_data and not include_kept:
            raise LabError("bad_request", "drop_data needs include_kept=1", field="include_kept")
        if drop_data and svc.kept:
            return act(lab_id, "destroy_kept_data", origin, lambda: svc.kept.destroy_with_data(lab_id, origin))
        return act(lab_id, "destroy", origin, lambda: mgr.destroy(lab_id, reason="deleted", by=origin, origin=origin,
                                                                  include_kept=bool(include_kept)))

    @app.post("/v1/labs/{lab_id}/exec")
    def exec_(lab_id: str, body: Dict[str, Any] = Body(...), x_origin: Optional[str] = O):
        origin = _origin(x_origin)
        out = act(lab_id, "exec", origin, lambda: mgr.exec(lab_id, body, origin), member=str(body.get("member") or ""),
                  cmd=str(body.get("cmd") or ""), detach=bool(body.get("detach")))
        return JSONResponse(status_code=202, content=out) if body.get("detach") else out

    @app.post("/v1/labs/{lab_id}/exec/cancel")
    def exec_cancel(lab_id: str, body: Dict[str, Any] = Body(default={}), x_origin: Optional[str] = O):
        origin = _origin(x_origin)
        return act(lab_id, "exec_cancel", origin, lambda: mgr.cancel_exec(lab_id, body or {}, origin),
                   member=str((body or {}).get("member") or ""))

    @app.get("/v1/labs/{lab_id}/jobs")
    def jobs(lab_id: str, x_origin: Optional[str] = O):
        mgr.get(lab_id)
        mgr.touch(lab_id, _origin(x_origin))
        rows = svc.store.q("SELECT job_id, member, cmd, state, exit_code, started_at, ended_at "
                           "FROM jobs WHERE lab_id=? ORDER BY started_at", (lab_id,))
        rules = mgr.redactor()      # review 11-05 #17: a job's cmd is echoed redacted
        return {"lab_id": lab_id, "jobs": [{**r, "cmd": redact.apply(r["cmd"] or "", rules)} for r in rows]}

    @app.get("/v1/labs/{lab_id}/jobs/{job_id}")
    def job(lab_id: str, job_id: str, tail_bytes: int = 65536, x_origin: Optional[str] = O):
        mgr.touch(lab_id, _origin(x_origin)) if svc.store.lab(lab_id) else None
        return mgr.job(lab_id, job_id, tail_bytes, reader=_origin(x_origin))

    @app.delete("/v1/labs/{lab_id}/jobs/{job_id}")
    def kill_job(lab_id: str, job_id: str, x_origin: Optional[str] = O):
        origin = _origin(x_origin)
        return act(lab_id, "job_kill", origin, lambda: mgr.job(lab_id, job_id, 65536, kill=True, origin=origin),
                   job_id=job_id)

    @app.put("/v1/labs/{lab_id}/files")
    def put_file(lab_id: str, body: Dict[str, Any] = Body(...), x_origin: Optional[str] = O):
        origin = _origin(x_origin)
        return act(lab_id, "put_file", origin, lambda: mgr.put_file(lab_id, body, origin),
                   member=str(body.get("member") or ""), path=str(body.get("path") or ""))

    @app.get("/v1/labs/{lab_id}/files")
    def get_file(lab_id: str, member: str, path: str, x_origin: Optional[str] = O):
        return mgr.get_file(lab_id, member, path, _origin(x_origin))

    @app.put("/v1/labs/{lab_id}/tree")
    def put_tree(lab_id: str, body: Dict[str, Any] = Body(...), x_origin: Optional[str] = O):
        origin = _origin(x_origin)
        return act(lab_id, "put_tree", origin, lambda: mgr.put_tree(lab_id, body, origin),
                   member=str(body.get("member") or ""), path=str(body.get("path") or ""))

    @app.get("/v1/labs/{lab_id}/tree")
    def get_tree(lab_id: str, member: str, path: str, x_origin: Optional[str] = O):
        return mgr.get_tree(lab_id, member, path, _origin(x_origin))

    @app.post("/v1/labs/{lab_id}/members/{member}/{action}")
    def member_action(lab_id: str, member: str, action: str, x_origin: Optional[str] = O):
        origin = _origin(x_origin)
        return act(lab_id, f"member_{action}", origin, lambda: mgr.member_action(lab_id, member, action, origin), member=member)

    @app.get("/v1/labs/{lab_id}/logs")
    def logs(lab_id: str, member: str, tail: int = 200, x_origin: Optional[str] = O):
        return mgr.logs(lab_id, member, tail, _origin(x_origin))

    @app.post("/v1/labs/{lab_id}/extend")
    def extend(lab_id: str, body: Dict[str, Any] = Body(...), x_origin: Optional[str] = O):
        origin = _origin(x_origin)
        return act(lab_id, "extend", origin, lambda: mgr.extend(lab_id, body, origin))

    @app.post("/v1/labs/{lab_id}/http")
    def http_probe(lab_id: str, body: Dict[str, Any] = Body(...), x_origin: Optional[str] = O):
        return mgr.http(lab_id, body, _origin(x_origin))

    @app.post("/v1/labs/{lab_id}/screenshot")
    def screenshot(lab_id: str, body: Dict[str, Any] = Body(...), x_origin: Optional[str] = O):
        origin = _origin(x_origin)
        return act(lab_id, "screenshot", origin, lambda: mgr.screenshot(lab_id, body, origin))

    @app.get("/v1/labs/{lab_id}/spec")
    def spec(lab_id: str):
        return mgr.spec(lab_id)

    # ---------------------------------------------------------------- kept machines (v2 §5)
    def kept_call(lab_id: str, action: str, x_origin: Optional[str], fn: Callable[[str], Any], **extra):
        origin = _origin(x_origin)
        if svc.kept is None:
            raise LabError("not_supported", "kept machines are not available in this build")
        if origin != "owner" and not mgr.may_touch_kept(mgr.get(lab_id), origin):
            raise LabError("owner_only", f"{action} is an owner action", hint="ask the owner to do this")
        return act(lab_id, action, origin, lambda: fn(origin), **extra)

    @app.post("/v1/labs/{lab_id}/keep")
    def keep(lab_id: str, body: Dict[str, Any] = Body(...), x_origin: Optional[str] = O):
        return kept_call(lab_id, "keep", x_origin, lambda o: svc.kept.keep(lab_id, body, o))

    @app.post("/v1/labs/{lab_id}/deploy")
    def deploy(lab_id: str, body: Dict[str, Any] = Body(...), x_origin: Optional[str] = O):
        return kept_call(lab_id, "deploy", x_origin, lambda o: svc.kept.deploy(lab_id, body))

    @app.post("/v1/labs/{lab_id}/rehearse")
    def rehearse(lab_id: str, x_origin: Optional[str] = O):
        return kept_call(lab_id, "rehearse", x_origin, lambda o: svc.kept.rehearse(lab_id))

    @app.post("/v1/labs/{lab_id}/switch")
    def switch(lab_id: str, body: Dict[str, Any] = Body(...), x_origin: Optional[str] = O):
        def _sw(o):
            fl = str(body.get("from_lab") or "")
            # agent §4SX D3 code r1 #1: the version switched OUT is touched too
            if o != "owner" and fl and not mgr.may_touch_kept(mgr.get(fl), o):
                raise LabError("owner_only", "switch away from a lab this caller may not touch", field="from_lab")
            return svc.kept.switch(lab_id, fl)
        return kept_call(lab_id, "switch", x_origin, _sw)

    @app.post("/v1/labs/{lab_id}/rollback")
    def rollback(lab_id: str, body: Dict[str, Any] = Body(default={}), x_origin: Optional[str] = O):
        def _rb(o):
            prev = str((svc.kept._app(lab_id) or {}).get("switched_from") or "")
            # agent §4SX D3 code r1 #1: the version rolled back TO is touched too
            if o != "owner" and prev and mgr.store.lab(prev) and not mgr.may_touch_kept(mgr.get(prev), o):
                raise LabError("owner_only", "roll back to a lab this caller may not touch", field="lab_id")
            return svc.kept.rollback(lab_id, bool(body.get("restore_snapshot")))
        return kept_call(lab_id, "rollback", x_origin, _rb)

    @app.get("/v1/labs/{lab_id}/backups")
    def backups(lab_id: str):
        if svc.kept is None:
            raise LabError("not_supported", "kept machines are not available in this build")
        return svc.kept.backups(lab_id)

    @app.post("/v1/labs/{lab_id}/backups")
    def backup_now(lab_id: str, x_origin: Optional[str] = O):
        return kept_call(lab_id, "backup", x_origin, lambda o: svc.kept.backup(lab_id, kind="manual"))

    @app.post("/v1/labs/{lab_id}/restore")
    def restore(lab_id: str, body: Dict[str, Any] = Body(...), x_origin: Optional[str] = O):
        return kept_call(lab_id, "restore", x_origin, lambda o: svc.kept.restore(lab_id, str(body.get("backup_id") or "")))

    @app.post("/v1/labs/{lab_id}/app/{action}")
    def app_action(lab_id: str, action: str, reinit: bool = False, x_origin: Optional[str] = O):
        # §4UL round 5: reinit=true accepts /data after a restore that did not complete (the owner's choice)
        return kept_call(lab_id, f"app_{action}", x_origin, lambda o: svc.kept.app_action(lab_id, action, reinit))

    @app.put("/v1/labs/{lab_id}/app/config")
    def app_config(lab_id: str, body: Dict[str, Any] = Body(...), x_origin: Optional[str] = O):
        # §4UL: a kept app's cmd/health/port/workdir/build change in place (a second keep used to be refused)
        return kept_call(lab_id, "app_config", x_origin, lambda o: svc.kept.update_app(lab_id, body))

    @app.put("/v1/labs/{lab_id}/secrets")
    def put_secrets(lab_id: str, body: Dict[str, Any] = Body(...), x_origin: Optional[str] = O):
        return kept_call(lab_id, "secrets", x_origin,
                         lambda o: svc.kept.rotate_secrets(lab_id, dict(body.get("secret_values") or {})))

    # ---------------------------------------------------------------- previews (v2 §12)
    @app.put("/v1/previews")
    def put_previews(body: Dict[str, Any] = Body(...)):
        items = body.get("previews") or []
        if not isinstance(items, list):
            raise LabError("bad_request", "previews must be a list", field="previews")
        rows = []
        for p in items[:200]:
            if not isinstance(p, dict):
                raise LabError("bad_request", "each preview is {name, url, project}", field="previews")
            name, url = str(p.get("name") or "")[:80], str(p.get("url") or "")[:300]
            if not name or not url.startswith(("http://", "https://")):
                raise LabError("bad_request", "each preview needs a name and an http(s) url", field="previews")
            rows.append((name, url, str(p.get("project") or "")[:80], time.time()))
        with svc.store._lock:   # validated first: a bad list never wipes the current one
            svc.store.x("DELETE FROM previews")
            for r in rows:
                svc.store.x("INSERT OR REPLACE INTO previews VALUES (?,?,?,?)", r)
        return {"previews": svc.store.q("SELECT name, url, project, at FROM previews ORDER BY name")}

    @app.get("/v1/previews")
    def get_previews():
        return {"previews": svc.store.q("SELECT name, url, project, at FROM previews ORDER BY name")}

    return app


def apps_page_rows(svc: Service) -> Dict[str, Any]:
    apps = svc.kept.apps_health() if svc.kept else []
    previews = svc.store.q("SELECT name, url, project, at FROM previews ORDER BY name")
    return {"apps": apps, "previews": previews}


def create_apps_page(svc: Service) -> FastAPI:
    """The read-only Apps page (v2 §12). Served on loopback and published by
    `tailscale serve --https=443`; no auth because it is tailnet-only and holds
    names, states and links only."""
    page = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @page.get("/", response_class=HTMLResponse)
    def index():
        rows = apps_page_rows(svc)

        def link(u: Optional[str]) -> str:
            return f'<a href="{html.escape(u)}">{html.escape(u)}</a>' if u else ""
        app_rows = "".join(
            f"<tr><td>{html.escape(a.get('app', ''))}</td><td class='s-{html.escape(a.get('state', ''))}'>"
            f"{html.escape(a.get('state', ''))}</td><td>{link(a.get('name_url'))}<br>{link(a.get('url'))}"
            # §4UL: a TCP app (a database) has a host:port, not a link
            + (f"<code>{html.escape(a['address'])}</code> <small>TCP</small>" if a.get('address') else "")
            + (f"<br>{link(a.get('lan_url')) if str(a['lan_url']).startswith('http') else html.escape(a['lan_url'])}"
               " <small>no login: home network only</small>" if a.get('lan_url') else "")
            + f"</td><td><code>{html.escape(a.get('lab_id', ''))}</code></td></tr>" for a in rows["apps"]) \
            or "<tr><td colspan=4>No apps yet.</td></tr>"
        prev_rows = "".join(
            f"<tr><td>{html.escape(p['name'])}</td><td>{link(p['url'])}</td><td>{html.escape(p['project'] or '')}</td></tr>"
            for p in rows["previews"]) or "<tr><td colspan=3>No previews.</td></tr>"
        return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lab apps</title><meta http-equiv="refresh" content="30">
<style>
:root{{--bg:#fafaf9;--fg:#1c1917;--mut:#78716c;--line:#e7e5e4;--ok:#15803d;--bad:#b91c1c;--warn:#a16207}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1c1917;--fg:#f5f5f4;--mut:#a8a29e;--line:#44403c;--ok:#4ade80;--bad:#f87171;--warn:#facc15}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0 auto;max-width:900px;padding:16px}}
table{{border-collapse:collapse;width:100%;margin-bottom:28px}}td,th{{border-bottom:1px solid var(--line);padding:8px 6px;text-align:left;vertical-align:top;word-break:break-all}}
th{{color:var(--mut);font-weight:500}}a{{color:inherit}}small{{color:var(--mut)}}.s-up{{color:var(--ok)}}.s-crashloop,.s-stopped{{color:var(--bad)}}.s-restarting,.s-starting,.s-staged{{color:var(--warn)}}
</style></head><body><h1>Lab apps</h1>
<table><tr><th>App</th><th>State</th><th>Open</th><th>Lab</th></tr>{app_rows}</table>
<h2>Previews</h2><table><tr><th>Name</th><th>Link</th><th>Project</th></tr>{prev_rows}</table>
</body></html>"""

    return page
