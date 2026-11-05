import time

import pytest

from ghost_labs.errors import LabError
from ghost_labs.kept import Kept, _backoff, data_volume_name

from .test_core import add_lab, env  # noqa: F401


@pytest.fixture
def kept(env):  # noqa: F811
    k = Kept(env)
    env.kept = k
    return k


def test_backoff_and_names():
    assert [_backoff(n) for n in (1, 2, 3, 4, 5, 9)] == [5, 15, 45, 135, 300, 300]
    assert data_volume_name("p", "a", "m") == "ghost-labdata-p-a-m"
    assert data_volume_name("p", "a", "m", "lab_1") == "ghost-labdata-p-a-m-lab_1"


def test_crash_accounting_reaches_crashloop(kept, monkeypatch):
    monkeypatch.setattr(kept, "_stop_proc", lambda *a: None)
    app = {"name": "x", "crash_times": []}
    for i in range(4):
        kept._record_crash("lab_1", app, "exit")
        assert app["state"] == "restarting"
    assert app["next_restart_at"] - time.time() == pytest.approx(135, abs=2)
    kept._record_crash("lab_1", app, "exit")
    assert app["state"] == "crashloop" and app["next_restart_at"] is None


def test_crashes_outside_window_forgotten(kept, monkeypatch):
    monkeypatch.setattr(kept, "_stop_proc", lambda *a: None)
    old = time.time() - 10_000
    app = {"name": "x", "crash_times": [old] * 10}
    kept._record_crash("lab_1", app, "exit")
    assert app["state"] == "restarting" and len(app["crash_times"]) == 1


@pytest.mark.parametrize("url", ["http://192.168.1.2:80", "http://example.com:80", "https://100.90.1.1:1",
                                 "http://100.90.1.1", "http://100.200.0.1:80", "http://100.63.255.255:80"])
def test_agent_api_url_rejected(kept, url):
    with pytest.raises(LabError):
        kept._validate_agent_api(url)


def test_agent_api_url_ok(kept):
    kept._validate_agent_api("http://100.90.234.23:8000")


def test_keep_requires_project_and_valid_app(kept, env):  # noqa: F811
    add_lab(env, "lab_c", owner="conversation:x")
    with pytest.raises(LabError) as e:
        kept.keep("lab_c", {"app": {"name": "a", "member": "m1", "port": 80}})
    assert e.value.extra["field"] == "owner"
    add_lab(env, "lab_p", owner="project:p1")
    for app, field in (({"name": "Bad!", "member": "m1", "port": 80}, "app.name"),
                       ({"name": "a", "member": "nope", "port": 80}, "app.member"),
                       ({"name": "a", "member": "m1"}, "app.port"),
                       ({"name": "a", "member": "m1", "port": 80, "health": {"http": "x"}}, "app.health")):
        with pytest.raises(LabError) as e:
            kept.keep("lab_p", {"app": app})
        assert e.value.extra["field"] == field
    with pytest.raises(LabError) as e:
        kept.keep("lab_p", {"app": {"name": "a", "member": "m1", "port": 80}, "secrets": ["lower"]})
    assert e.value.extra["field"] == "secrets"
    with pytest.raises(LabError) as e:
        kept.keep("lab_p", {"app": {"name": "a", "member": "m1", "port": 80}, "cron": ["* * * *"]})
    assert e.value.extra["field"] == "cron"


def test_not_kept_actions(kept, env):  # noqa: F811
    add_lab(env, "lab_d", owner="project:p")
    for fn in (lambda: kept.backups("lab_d"), lambda: kept.app_action("lab_d", "start"),
               lambda: kept.deploy("lab_d", {"archive_b64": ""})):
        with pytest.raises(LabError) as e:
            fn()
        assert e.value.code == "not_kept"
    with pytest.raises(LabError):
        kept.app_action("lab_d", "explode")


def test_apps_health_lists_kept_only(kept, env):  # noqa: F811
    from ghost_labs.store import dumps
    add_lab(env, "lab_k", lifetime="kept", owner="project:p")
    env.store.update_lab("lab_k", app_json=dumps({"name": "chess", "state": "up", "url": "https://h:20000/"}))
    add_lab(env, "lab_d")
    assert [a["lab_id"] for a in kept.apps_health()] == ["lab_k"]
