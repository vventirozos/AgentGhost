"""req a3514318 (2026-11-12): a failed probe names its cause.

The agent read {'alive': False, 'ok': False, 'http_status': 200} and guessed for eight turns —
the app had no cmd, so nothing could be alive while nginx answered the port."""
import pytest

from ghost_labs.kept import Kept


@pytest.mark.parametrize("app,res,want", [
    ({"cmd": ""}, {"alive": False, "ok": False, "http_status": 200}, "has no cmd"),
    ({"cmd": "./run.sh"}, {"alive": False, "ok": False}, "process is not running"),
    ({"cmd": "./run.sh"}, {"alive": True, "ok": False, "error": "ConnectionRefusedError: x"}, "port does not answer"),
    ({"cmd": "./run.sh"}, {"alive": True, "ok": False, "http_status": 503}, "HTTP 503"),
])
def test_the_cause_line(app, res, want):
    assert want in Kept._why(app, res)


def test_the_probe_carries_it_and_wait_up_stores_it(monkeypatch):
    k = Kept.__new__(Kept)
    monkeypatch.setattr(Kept, "_alive", lambda self, lab_id, app: False)

    class _M:
        def member(self, lab_id, m):
            return {"ip": "127.0.0.1"}
    k.mgr = _M()
    app = {"cmd": "", "member": "web1", "port": 9, "health": {"http": None, "tcp": False}}
    res = k.probe("lab_x", app)
    assert res["ok"] is False and "has no cmd" in res["why"]
    monkeypatch.setattr(Kept, "probe", lambda self, lab_id, a: res)
    k.wait_up("lab_x", app, timeout_s=1)
    assert "has no cmd" in app["last_error"]
