"""`POST /api/lessons/quarantine` — in-process lesson quarantine (§4FB).

Why a route: the playbook has a single-writer contract (an external script
must not write it while the agent runs) and launchd respawns the agent
immediately on exit, so a process-free window does not exist. The endpoint
calls the designed, reversible `SkillMemory.quarantine_lesson` inside the
live process.

Worlds where these fail: the route stops forwarding trigger/reason; it
accepts an empty reason (nothing on the record); it writes through a
read-only (dream/self-play) skill memory; it reports ok when nothing
matched.
"""

from unittest.mock import MagicMock

import pytest


def _client(skill_memory):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from ghost_agent.api import routes as routes_module

    app = FastAPI()
    fake = MagicMock()
    fake.context.args.api_key = None
    fake.context.skill_memory = skill_memory
    app.state.agent = fake
    app.state.args = MagicMock()
    app.include_router(routes_module.router)
    return TestClient(app)


TRIGGER = ("Use deep_research ONCE on the query: llama.cpp prompt prefill speed "
           "apple silicon. Then reply with one short sentence about what you found.")


def test_forwards_trigger_and_reason_to_the_store():
    sm = MagicMock()
    sm.is_read_only = False
    sm.quarantine_lesson.return_value = 1
    resp = _client(sm).post("/api/lessons/quarantine",
                            json={"trigger": TRIGGER, "reason": "bench probe minted as a lesson"})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True and resp.json()["quarantined"] == 1
    sm.quarantine_lesson.assert_called_once_with(TRIGGER, "bench probe minted as a lesson")


def test_no_match_is_reported_not_ok():
    sm = MagicMock()
    sm.is_read_only = False
    sm.quarantine_lesson.return_value = 0
    resp = _client(sm).post("/api/lessons/quarantine",
                            json={"trigger": "nothing like this", "reason": "r"})
    assert resp.status_code == 200
    assert resp.json() == {"ok": False, "quarantined": 0, "trigger": "nothing like this"}


@pytest.mark.parametrize("body", [
    {"reason": "r"},                       # no trigger
    {"trigger": "   ", "reason": "r"},     # blank trigger
    {"trigger": TRIGGER},                  # no reason — nothing on the record
    {"trigger": TRIGGER, "reason": ""},
])
def test_rejects_incomplete_bodies_without_touching_the_store(body):
    sm = MagicMock()
    sm.is_read_only = False
    resp = _client(sm).post("/api/lessons/quarantine", json=body)
    assert resp.status_code == 400
    assert not sm.quarantine_lesson.called


def test_rejects_malformed_json():
    sm = MagicMock()
    resp = _client(sm).post("/api/lessons/quarantine", data=b"not json",
                            headers={"Content-Type": "application/json"})
    assert resp.status_code == 400
    assert not sm.quarantine_lesson.called


def test_read_only_or_missing_store_is_503():
    ro = MagicMock()
    ro.is_read_only = True
    resp = _client(ro).post("/api/lessons/quarantine",
                            json={"trigger": TRIGGER, "reason": "r"})
    assert resp.status_code == 503
    assert not ro.quarantine_lesson.called
    resp = _client(None).post("/api/lessons/quarantine",
                              json={"trigger": TRIGGER, "reason": "r"})
    assert resp.status_code == 503
