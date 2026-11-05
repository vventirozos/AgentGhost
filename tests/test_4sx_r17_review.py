"""Pins for review 11-05 r17 (found at deploy): the labs host is SHARED — its own integration tests make kept
`project:itest…` labs — and the kept-app adopt step (`runs._adopt_app_rows`) filed every kept project lab it listed as
the owner's app run, so the down sweep would notify the owner about apps that are not theirs. Adoption is now for
projects in THIS agent's store only (`runs._is_own_project`, fails closed)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_4sx_d3_kept import LAB, LAB2, OWN, PID, _app, _det, ctx, fake  # noqa: E402,F401
from ghost_agent.memory.run_registry import registry_for  # noqa: E402
from ghost_agent.tools import runs as R  # noqa: E402

pytestmark = pytest.mark.asyncio


class _Store:
    def __init__(self, known, boom=False):
        self.known, self.boom = set(known), boom

    def get_project(self, pid):
        if self.boom:
            raise RuntimeError("db locked")
        return {"id": pid, "metadata": {}} if pid in self.known else None


def _host(fake, foreign="project:itest0978b2"):
    fake.on("GET", "/v1/labs", (200, {"labs": [{"lab_id": LAB, "lifetime": "kept", "owner": OWN},
                                               {"lab_id": LAB2, "lifetime": "kept", "owner": foreign}]}))
    fake.on("GET", f"/v1/labs/{LAB}", (200, _det(app=_app(name="shop"))))
    fake.on("GET", f"/v1/labs/{LAB2}", (200, _det(lab=LAB2, owner=foreign, app=_app(name="labs-test"))))


@pytest.fixture(autouse=True)
def _fresh_seen(monkeypatch):
    monkeypatch.setattr(R, "_ADOPT_SEEN", set())


async def test_only_a_kept_lab_of_this_agents_own_project_is_adopted(ctx, fake):
    ctx.project_store = _Store({PID})
    _host(fake)
    reg = registry_for(ctx)
    assert await R._adopt_app_rows(ctx, reg) == 1
    assert reg.live_by_name(OWN, "shop") is not None
    assert reg.live_by_name("project:itest0978b2", "labs-test") is None
    assert not any(c.path == f"/v1/labs/{LAB2}" for c in fake.calls)       # a foreign lab is not even read


@pytest.mark.parametrize("store", [None, _Store(set(), boom=True), _Store(set())])
async def test_an_unknown_project_or_an_unreadable_store_adopts_nothing(ctx, fake, store):
    ctx.project_store = store
    _host(fake)
    reg = registry_for(ctx)
    assert await R._adopt_app_rows(ctx, reg) == 0
    assert reg.live_by_name(OWN, "shop") is None


async def test_a_skipped_lab_is_looked_at_again_once_its_project_exists(ctx, fake):
    ctx.project_store = _Store(set())
    _host(fake)
    reg = registry_for(ctx)
    assert await R._adopt_app_rows(ctx, reg) == 0
    ctx.project_store = _Store({PID})
    assert await R._adopt_app_rows(ctx, reg) == 1


def _foreign_row(reg, owner="project:itestd457c9"):
    return reg.add(name="labs-test", place=LAB2, lifetime="app", command="x", port=8080, owner=owner,
                   origin="owner", backend="labs_app", backend_ref=f"{owner.partition(':')[2]}/labs-test")


async def test_the_sweep_ends_a_row_adopted_for_a_project_this_agent_does_not_have(ctx, fake, monkeypatch):
    """r17 review MAJOR: rows adopted BEFORE the adopt check stayed live, were probed, and paged the owner."""
    sent = []
    monkeypatch.setattr(R, "_notify_owner", lambda c, msg, ids: sent.append(msg) or True)
    monkeypatch.setattr(R, "_ADOPT_SEEN", {LAB, LAB2})
    ctx.project_store = _Store({PID})
    reg = registry_for(ctx)
    rid = _foreign_row(reg)
    fake.on("GET", "/v1/labs", (200, {"labs": []}))
    fake.on("GET", f"/v1/labs/{LAB2}", (200, _det(lab=LAB2, owner="project:itestd457c9",
                                                   app=_app(name="labs-test", state="crashloop"))))
    await R.app_down_sweep(ctx)
    await R.app_down_sweep(ctx)
    row = reg.get(rid)
    assert sent == [] and row["ended_at"] is not None and "not a project of this agent" in row["state_note"]
    assert not any(c.path == f"/v1/labs/{LAB2}" for c in fake.calls)


async def test_a_store_that_cannot_answer_never_ends_a_row(ctx, fake, monkeypatch):
    monkeypatch.setattr(R, "_notify_owner", lambda c, msg, ids: True)
    monkeypatch.setattr(R, "_ADOPT_SEEN", {LAB, LAB2})
    ctx.project_store = _Store(set(), boom=True)
    reg = registry_for(ctx)
    rid = _foreign_row(reg, owner=OWN)
    fake.on("GET", "/v1/labs", (200, {"labs": []}))
    await R.app_down_sweep(ctx)
    assert "not a project of this agent" not in (reg.get(rid)["state_note"] or "")
