"""agent §4SX D3 design r1 H1/H2/H3: the probe tier on kept machines, the lineage guard, kept-member mutations."""
import time

import pytest

from ghost_labs.errors import LabError
from ghost_labs.kept import Kept

from .test_core import add_lab, env  # noqa: F401
from .test_kept_fixes import make_kept
from ghost_labs.store import dumps


def _spec(m, lab_id):
    m.store.update_lab(lab_id, spec_json=dumps({"members": [{"name": "m1", "image": "ghost-lab/base:1", "mem_mb": 512,
                                                            "cpus": 1.0, "cap_add": []}], "egress": "none",
                                                "readiness": None}))


@pytest.fixture
def kept(env):  # noqa: F811
    k = Kept(env)
    env.kept = k
    return k


def test_may_touch_kept_is_owner_or_a_probe_on_its_own_lab(env):  # noqa: F811
    m = env
    assert m.may_touch_kept({"origin": "autopilot"}, "owner")
    assert m.may_touch_kept({"origin": "probe"}, "probe")
    assert not m.may_touch_kept({"origin": "owner"}, "probe")
    assert not m.may_touch_kept({"origin": "probe"}, "autopilot")
    assert not m.may_touch_kept({"origin": "autopilot"}, "autopilot")


def test_mutating_a_kept_member_needs_the_owner(env):  # noqa: F811
    m = env
    add_lab(m, "lab_k", lifetime="kept", owner="project:p", origin="owner")
    m.docker.containers.run("img", name=m.container_name("lab_k", "m1"), labels={"ghost.labs": "1", "ghost.lab.id": "lab_k"})
    for call in (lambda o: m.exec("lab_k", {"member": "m1", "cmd": "true"}, o),
                 lambda o: m.put_file("lab_k", {"member": "m1", "path": "/x", "content_b64": ""}, o),
                 lambda o: m.member_action("lab_k", "m1", "kill", o)):
        for o in ("autopilot", "probe"):
            with pytest.raises(LabError) as e:
                call(o)
            assert e.value.code == "owner_only"


def test_a_disposable_lab_is_not_guarded(env):  # noqa: F811
    m = env
    add_lab(m, "lab_d", owner="project:p", origin="owner")
    m.guard_kept_mutation(m.store.lab("lab_d"), "autopilot", "exec")      # no raise


def test_a_probe_kept_lab_is_its_probes_and_retires(env):  # noqa: F811
    m = env
    add_lab(m, "lab_q", lifetime="kept", owner="project:q", origin="probe")
    m.guard_kept_mutation(m.store.lab("lab_q"), "probe", "exec")          # allowed
    with pytest.raises(LabError):
        m.guard_kept_mutation(m.store.lab("lab_q"), "autopilot", "exec")
    m._destroy_guard(m.store.lab("lab_q"), "deleted", "probe", True, False)   # probe may destroy it


def test_a_mounted_probe_lab_retires_but_an_owners_does_not(env, kept):  # noqa: F811
    m = env
    make_kept(m, "lab_o", lineage="po")
    m.store.update_lab("lab_o", origin="owner")
    with pytest.raises(LabError):
        m._destroy_guard(m.store.lab("lab_o"), "retired", "autopilot", False, False)
    make_kept(m, "lab_r", lineage="pr", name="other")
    m.store.update_lab("lab_r", origin="probe")
    m._destroy_guard(m.store.lab("lab_r"), "retired", "autopilot", False, False)      # no raise


def test_a_non_owner_cannot_borrow_another_lineage(kept, env):  # noqa: F811
    m = env
    make_kept(m, "lab_v3", lineage="chess")
    m.store.update_lab("lab_v3", origin="owner")
    add_lab(m, "lab_x", owner="project:probeproj", origin="probe")
    for lineage in ("chess", "someother"):
        with pytest.raises(LabError) as e:
            kept.keep("lab_x", {"app": {"name": "chess", "member": "m1", "port": 80}, "lineage": lineage}, "probe")
        assert e.value.code == "owner_only"


def test_the_owner_may_name_its_own_lineage_and_a_probe_its_project(kept, env):  # noqa: F811
    m = env
    add_lab(m, "lab_y", owner="project:probeproj", origin="probe")
    _spec(m, "lab_y")
    out = kept.keep("lab_y", {"app": {"name": "web", "member": "m1", "port": 80}}, "probe")
    row = m.store.lab("lab_y")
    assert row["lineage"] == "probeproj" and row["lifetime"] == "kept"
    assert row["retire_at"] and float(row["retire_at"]) > time.time() + 86000
    assert out["lab_id"] == "lab_y"
    add_lab(m, "lab_z", owner="project:own", origin="owner")
    _spec(m, "lab_z")
    kept.keep("lab_z", {"app": {"name": "site", "member": "m1", "port": 80}}, "owner")
    assert m.store.lab("lab_z")["retire_at"] is None


def test_a_probe_cannot_join_a_lineage_where_the_owner_keeps_a_lab_of_the_same_project(kept, env):  # noqa: F811
    """agent §4SX D3 code r1 #1: same owner key, same lineage — the owner's production is there."""
    m = env
    make_kept(m, "lab_live", lineage="shop")
    m.store.update_lab("lab_live", origin="owner", owner="project:shop")
    add_lab(m, "lab_pr", owner="project:shop", origin="probe")
    _spec(m, "lab_pr")
    with pytest.raises(LabError) as e:
        kept.keep("lab_pr", {"app": {"name": "chess", "member": "m1", "port": 80}}, "probe")
    assert e.value.code == "owner_only"
    kept.keep("lab_pr", {"app": {"name": "chess", "member": "m1", "port": 80}}, "owner")   # the owner may


def test_a_probe_cannot_switch_away_from_or_roll_back_to_an_owners_lab(env):  # noqa: F811
    from fastapi.testclient import TestClient
    from ghost_labs.api import create_app
    from .test_api import Svc
    m = env
    svc = Svc(m)
    seen = []

    class _K:
        def switch(self, lab_id, from_lab):
            seen.append(("switch", lab_id, from_lab))
            return {"switched": True}

        def rollback(self, lab_id, snap):
            seen.append(("rollback", lab_id))
            return {"rolled_back": True}

        def _app(self, lab_id):
            return {"switched_from": "lab_own"}
    svc.kept = _K()
    c = TestClient(create_app(svc, "k" * 64))
    add_lab(m, "lab_own", owner="project:p", origin="owner", lifetime="kept")
    add_lab(m, "lab_prb", owner="project:p", origin="probe", lifetime="kept")
    add_lab(m, "lab_prb2", owner="project:p", origin="probe", lifetime="kept")
    h = {"X-Labs-Key": "k" * 64, "X-Labs-Origin": "probe"}
    assert c.post("/v1/labs/lab_prb/switch", json={"from_lab": "lab_own"}, headers=h).json()["error"] == "owner_only"
    assert c.post("/v1/labs/lab_prb/rollback", json={}, headers=h).json()["error"] == "owner_only"
    assert seen == []
    assert c.post("/v1/labs/lab_prb/switch", json={"from_lab": "lab_prb2"}, headers=h).json() == {"switched": True}
    ho = {"X-Labs-Key": "k" * 64, "X-Labs-Origin": "owner"}
    assert c.post("/v1/labs/lab_prb/rollback", json={}, headers=ho).json() == {"rolled_back": True}


def test_a_served_probe_version_keeps_its_retire_date(kept, env):  # noqa: F811
    """agent §4SX D3 code r1 #2."""
    m = env
    add_lab(m, "lab_p1", owner="project:q", origin="probe", lifetime="kept")
    add_lab(m, "lab_o1", owner="project:o", origin="owner", lifetime="kept")
    assert kept._live_retire("lab_p1") > time.time() + 86000
    assert kept._live_retire("lab_o1") is None



def test_the_reaper_retires_a_probe_kept_lab_with_its_lease_and_secrets_and_no_backup(kept, env, monkeypatch):  # noqa: F811
    """agent §4SX D3 code r1 #6 + D4 r2 #2 (no final backup: a failing one wedged the reaper)."""
    m = env
    make_kept(m, "lab_pq", lineage="pq")
    m.store.update_lab("lab_pq", origin="probe", retire_at=time.time() - 1)
    m.store.set_secret("pq", "web", "API", "x")
    _probe_made(kept, "pq", "chess")
    _probe_made(kept, "pq", "web")
    monkeypatch.setattr(kept, "_backup_locked", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no backup")))
    acted = m.reap_once()
    assert acted == [{"lab_id": "lab_pq", "reason": "retired"}]
    assert m.store.lab("lab_pq") is None and m.store.secrets("pq", "web") == {}


def test_one_lab_that_raises_never_stops_the_reaper_tick(kept, env, monkeypatch):  # noqa: F811
    m = env
    add_lab(m, "lab_a1", age=10**6)
    add_lab(m, "lab_a2", age=10**6)
    real = m.destroy

    def flaky(lid, **kw):
        if lid == "lab_a1":
            raise RuntimeError("docker NotFound")
        return real(lid, **kw)
    monkeypatch.setattr(m, "destroy", flaky)
    acted = m.reap_once()
    assert {"lab_id": "lab_a2", "reason": "max_lifetime"} in acted


def _vol(m, name, lineage="", app="chess", member="m1"):
    m.docker.volumes.create(name, labels={"ghost.labs": "1", "ghost.data.lineage": lineage, "ghost.data.app": app,
                                          "ghost.data.member": member})


def _probe_made(kept, lineage, app, lab_id="lab_x"):
    """What a probe's /keep records (review 11-05 CRIT 2): retirement removes only a namespace a probe made."""
    kept._claim("app", f"{lineage}/{app}", lab_id)
    kept._claim("svc", app, lab_id)


def _retiring(m, *lids):
    for lid in lids:
        m.store.update_lab(lid, origin="probe", retire_at=time.time() - 1)


def test_a_retired_probe_lab_drops_its_lease_and_its_shared_volume_as_the_last_version(kept, env):  # noqa: F811
    """agent §4SX D4 r3 #5: the last version — mounted or not — takes the lease and the shared data with it
    (through the reaper)."""
    from ghost_labs.kept import data_volume_name
    m = env
    make_kept(m, "lab_s1", lineage="pz", mounted=False)            # a staged last version
    _retiring(m, "lab_s1")
    _probe_made(kept, "pz", "chess", "lab_s1")
    m.store.x("INSERT INTO ports VALUES (?,?,?,?,?,?)", (18999, "lab_s1", "m1", 8080, "pz/chess", 0))
    v = data_volume_name("pz", "chess", "m1", "")
    _vol(m, v, "pz")
    _vol(m, "ghost-labdata-other-chess-m1", "other")                 # another lineage's data stays
    assert {"lab_id": "lab_s1", "reason": "retired"} in m.reap_once()
    assert v not in m.docker._volumes and "ghost-labdata-other-chess-m1" in m.docker._volumes
    assert m.store.q("SELECT port FROM ports WHERE lease_key=?", ("pz/chess",)) == []


def test_a_retired_probe_lab_keeps_shared_data_another_version_uses_but_drops_its_fresh_volume(kept, env):  # noqa: F811
    from ghost_labs.kept import data_volume_name
    m = env
    make_kept(m, "lab_f1", lineage="pf", data="fresh")
    make_kept(m, "lab_f2", lineage="pf", mounted=False)
    _retiring(m, "lab_f1")
    m.store.update_lab("lab_f2", origin="probe")
    m.store.x("INSERT INTO ports VALUES (?,?,?,?,?,?)", (18998, "lab_f2", "m1", 8080, "pf/chess", 0))
    own, shared = data_volume_name("pf", "chess", "m1", "lab_f1"), data_volume_name("pf", "chess", "m1", "")
    _vol(m, own, "pf")
    _vol(m, shared, "pf")
    out = kept.retire_probe_lab("lab_f1")
    assert out["data_volumes_removed"] == [own]
    assert shared in m.docker._volumes                              # lab_f2 is still a version of the app
    assert m.store.q("SELECT port FROM ports WHERE lease_key=?", ("pf/chess",)) != []


def test_a_fresh_last_version_takes_every_data_volume_of_the_app(kept, env):  # noqa: F811
    """agent §4SX D4 r4 #2: v1 shared retired first (v2 still there) — v2, fresh and last, takes the shared one
    and a member only v1 had."""
    from ghost_labs.kept import data_volume_name
    m = env
    make_kept(m, "lab_g2", lineage="pg", data="fresh")
    _retiring(m, "lab_g2")
    _probe_made(kept, "pg", "chess", "lab_g2")
    for v, mem in ((data_volume_name("pg", "chess", "m1", ""), "m1"), (data_volume_name("pg", "chess", "db", ""), "db"),
                   (data_volume_name("pg", "chess", "m1", "lab_g2"), "m1")):
        _vol(m, v, "pg", member=mem)
    _vol(m, "ghost-labdata-pg-chessx-m1", "pg", app="chessx")        # another app of the lineage stays
    out = kept.retire_probe_lab("lab_g2")
    assert sorted(out["data_volumes_removed"]) == sorted([data_volume_name("pg", "chess", "m1", "lab_g2"),
                                                          data_volume_name("pg", "chess", "m1", ""),
                                                          data_volume_name("pg", "chess", "db", "")])
    assert "ghost-labdata-pg-chessx-m1" in m.docker._volumes


def test_a_probe_lab_that_serves_again_is_not_retired(kept, env):  # noqa: F811
    """agent §4SX D4 r4 minor 1: a switch made it serve again (retire_at moved) while the reaper waited."""
    m = env
    make_kept(m, "lab_h1", lineage="ph")
    m.store.update_lab("lab_h1", origin="probe", retire_at=time.time() + 3600)
    assert kept.retire_probe_lab("lab_h1")["skipped"]
    assert m.store.lab("lab_h1") is not None


def test_a_busy_app_lock_makes_the_reaper_skip_the_lab_this_tick(kept, env, monkeypatch):  # noqa: F811
    import contextlib
    from ghost_labs.errors import LabError as _LE
    m = env
    make_kept(m, "lab_b1", lineage="pb")
    _retiring(m, "lab_b1")

    waits = []

    @contextlib.contextmanager
    def busy(name, timeout=120.0):
        waits.append(timeout)
        raise _LE("busy", "x")
        yield
    monkeypatch.setattr(kept, "_app_locked", busy)
    assert m.reap_once() == [] and m.store.lab("lab_b1") is not None
    assert waits == [5.0]                                            # a brief wait, never the 120 s default


def test_a_retire_the_lock_found_serving_again_is_not_reported_as_reaped(kept, env, monkeypatch):  # noqa: F811
    m = env
    make_kept(m, "lab_c1", lineage="pc")
    _retiring(m, "lab_c1")
    monkeypatch.setattr(kept, "retire_probe_lab", lambda lid: {"lab_id": lid, "skipped": "serves again"})
    assert m.reap_once() == []


def test_a_probe_cannot_join_a_lineage_whose_kept_lab_has_another_owner(kept, env):  # noqa: F811
    """agent §4SX D3 code r1 #1: another owner key in the lineage — even another probe's lab — is not joined."""
    m = env
    make_kept(m, "lab_x1", lineage="shop")
    m.store.update_lab("lab_x1", origin="probe", owner="conversation:other")
    add_lab(m, "lab_x2", owner="project:shop", origin="probe")
    _spec(m, "lab_x2")
    with pytest.raises(LabError) as e:
        kept.keep("lab_x2", {"app": {"name": "chess", "member": "m1", "port": 80}}, "probe")
    assert e.value.code == "owner_only"


def test_the_summary_says_whether_a_kept_lab_has_its_app_mounted(kept, env):  # noqa: F811
    """agent §4SX D3 code r1 #8: the agent's project delete destroys the mounted version last."""
    m = env
    make_kept(m, "lab_m1", lineage="sm", mounted=True)
    make_kept(m, "lab_m2", lineage="sm", mounted=False)
    assert m.summary(m.store.lab("lab_m1"))["app_mounted"] is True
    assert m.summary(m.store.lab("lab_m2"))["app_mounted"] is False
    add_lab(m, "lab_d1")
    assert "app_mounted" not in m.summary(m.store.lab("lab_d1"))


def test_health_reports_the_disk_floor_a_create_needs(env, monkeypatch):  # noqa: F811
    """agent §4SX D4 r3 #1: the agent's capacity hold lifts on the host's own floor, not a guess."""
    import types
    from ghost_labs import service as S
    monkeypatch.setattr(S.C, "meminfo", lambda: {"MemAvailable": 4000})
    monkeypatch.setattr(S.C, "host_id", lambda: "h")
    env.check_firewall = lambda: None
    env.firewall_state = "ok"
    env.tor_state = lambda: "ok"
    env.disk_free_mb = lambda: 50000
    fake = types.SimpleNamespace(mgr=env, cfg=env.cfg, store=env.store, kept=None)
    h = S.Service.health(fake)
    assert h["disk_floor_mb"] == env.cfg.disk_floor_mb and h["disk_free_mb"] == 50000


def test_a_lab_the_reaper_cannot_retire_is_said_once(kept, env, monkeypatch, caplog):  # noqa: F811
    """agent D4 r5 #2: a non-busy refusal is said, not swallowed."""
    import logging
    from ghost_labs.errors import LabError as _LE
    m = env
    make_kept(m, "lab_r1", lineage="pr")
    _retiring(m, "lab_r1")
    monkeypatch.setattr(kept, "retire_probe_lab", lambda lid: (_ for _ in ()).throw(_LE("bad_request", "serves")))
    with caplog.at_level(logging.WARNING):
        assert m.reap_once() == []
        assert m.reap_once() == []
    said = [r for r in caplog.records if "lab_r1" in r.getMessage() and "bad_request" in r.getMessage()]
    assert len(said) == 1                                            # agent D4 r6: once, not 1440 a day
