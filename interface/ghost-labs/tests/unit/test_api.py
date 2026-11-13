import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from ghost_labs.api import _origin, create_app, create_apps_page  # noqa: E402

from .test_core import add_lab, env  # noqa: F401,E402


class Svc:
    def __init__(self, m):
        from ghost_labs.service import ActionLog
        self.mgr, self.store, self.templates, self.kept = m, m.store, m.templates, None
        self.actions = ActionLog(m.store_path.parent / "actions.log") if hasattr(m, "store_path") else _NoLog()

    def health(self):
        return {"ok": True}


class _NoLog:
    rows = []

    def write(self, *a, **k):
        self.rows.append((a, k))


@pytest.fixture
def client(env):  # noqa: F811
    return TestClient(create_app(Svc(env), "k" * 64)), env


def test_origin_fail_closed():
    assert _origin(None) == "autopilot" and _origin("Owner") == "owner" and _origin("root") == "autopilot"


def test_auth_required(client):
    c, _ = client
    assert c.get("/v1/health").status_code == 401
    assert c.get("/v1/health", headers={"X-Labs-Key": "wrong"}).json()["error"] == "unauthorized"
    assert c.get("/v1/health", headers={"X-Labs-Key": "k" * 64}).status_code == 200


def test_kept_actions_owner_only(client):
    c, m = client
    add_lab(m, "lab_1", owner="project:p")
    h = {"X-Labs-Key": "k" * 64}
    for path in ("keep", "deploy", "rehearse", "switch", "rollback", "restore"):
        r = c.post(f"/v1/labs/lab_1/{path}", json={}, headers=h)      # no origin header = autopilot
        assert r.status_code in (403, 501), path
    r = c.post("/v1/gc", headers=h)
    assert r.status_code == 403 and r.json()["error"] == "owner_only"


def test_errors_are_json_with_hint(client):
    c, _ = client
    r = c.get("/v1/labs/lab_nope", headers={"X-Labs-Key": "k" * 64})
    assert r.status_code == 404 and r.json()["error"] == "unknown_lab" and "hint" in r.json()


def test_previews_and_page(client, env):  # noqa: F811
    c, m = client
    h = {"X-Labs-Key": "k" * 64}
    r = c.put("/v1/previews", json={"previews": [{"name": "chess", "url": "https://x.ts.net:20001/", "project": "p"}]},
              headers=h)
    assert r.status_code == 200 and len(r.json()["previews"]) == 1
    assert c.put("/v1/previews", json={"previews": [{"name": "x", "url": "javascript:alert(1)"}]}, headers=h).status_code == 400
    page = TestClient(create_apps_page(Svc(m)))
    html = page.get("/").text
    assert "chess" in html and "https://x.ts.net:20001/" in html


def test_destroy_owner_with_kept_requires_owner(client):
    c, m = client
    add_lab(m, "lab_k", owner="project:p", lifetime="kept")
    r = c.delete("/v1/labs?owner=project:p&include_kept=1", headers={"X-Labs-Key": "k" * 64, "X-Labs-Origin": "autopilot"})
    assert r.status_code == 403
    assert m.store.lab("lab_k")


def test_a_probe_reaches_kept_actions_only_on_a_lab_it_made(env):  # noqa: F811
    """agent §4SX D3 r1 H2."""
    m = env
    svc = Svc(m)

    class _K:
        def app_action(self, lab_id, action, reinit=False):
            return {"lab_id": lab_id, "acted": action}
    svc.kept = _K()
    c = TestClient(create_app(svc, "k" * 64))
    add_lab(m, "lab_own", owner="project:p", origin="owner")
    add_lab(m, "lab_prb", owner="project:q", origin="probe")
    h = {"X-Labs-Key": "k" * 64, "X-Labs-Origin": "probe"}
    assert c.post("/v1/labs/lab_own/app/stop", headers=h).json()["error"] == "owner_only"
    assert c.post("/v1/labs/lab_prb/app/stop", headers=h).json() == {"lab_id": "lab_prb", "acted": "stop"}
    assert c.post("/v1/labs/lab_prb/app/stop", headers={"X-Labs-Key": "k" * 64}).json()["error"] == "owner_only"
