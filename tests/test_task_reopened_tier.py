"""§4E Tier 3 — retroactive calibration negatives from reopened work.

A task a turn closed DONE being later REOPENED is a delayed negative on
that turn. The store stamps the closing turn's req_id at close time
(`tasks.closed_req_id`), fires `on_task_reopened` on the task-level
DONE -> open transition (logging a `task_reopened` event), and the
tracker re-records the closing turn's OWN stored components at
`_TASK_REOPENED_GRADE` with `source="task_reopened"` — idempotent per
closing turn, skipped when that turn was already negative.
"""
import json

import pytest

from ghost_agent.core.calibration import (
    CalibrationTracker,
    _TASK_REOPENED_GRADE,
)
from ghost_agent.memory.projects import ProjectStore
from ghost_agent.utils.logging import request_id_context


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path)


@pytest.fixture
def tracker(tmp_path):
    return CalibrationTracker(tmp_path / "calibration")


def _project_with_task(store, *, extra_open_task=False):
    pid = store.create_project("Test project", goal="test")
    tid = store.add_task(pid, "the work item")
    other = store.add_task(pid, "still open") if extra_open_task else None
    return pid, tid, other


def _close_as(store, tid, req_id):
    token = request_id_context.set(req_id)
    try:
        assert store.update_task(tid, status="DONE")
    finally:
        request_id_context.reset(token)


def _record_turn(tracker, req_id, *, outcome=1.0):
    tracker.record(
        composite=0.7, entropy_component=0.6, competence_component=0.8,
        uncertainty_pressure=0.1, outcome=outcome, domain="coding",
        entropy_observed=True, effort_component=0.4, effort_observed=True,
        source="turn", req_id=req_id,
    )


def _samples(tracker):
    return tracker._load_samples()


class TestClosingStamp:
    def test_close_stamps_the_closing_turns_req_id(self, store):
        _, tid, _ = _project_with_task(store)
        _close_as(store, tid, "abcd1234")
        assert store.get_task(tid)["closed_req_id"] == "abcd1234"

    def test_system_context_is_not_stamped(self, store):
        # Boot reaper / maintenance closes carry no turn to blame.
        _, tid, _ = _project_with_task(store)
        assert store.update_task(tid, status="DONE")
        assert store.get_task(tid)["closed_req_id"] == ""

    def test_reopen_consumes_the_stamp(self, store):
        # The hook receives the closing req_id, but the column is cleared
        # in the same transition — the retro-negative fires exactly once
        # and a later re-close re-stamps fresh (a stale stamp would
        # attribute a SYSTEM re-close to a turn that didn't perform it).
        _, tid, _ = _project_with_task(store)
        _close_as(store, tid, "abcd1234")
        fired = []
        store.on_task_reopened = lambda *a: fired.append(a)
        store.update_task(tid, status="PENDING")
        assert fired[0][3] == "abcd1234"
        assert store.get_task(tid)["closed_req_id"] == ""

    def test_system_reclose_blanks_rather_than_inherits(self, store):
        # close(A) -> reopen -> re-close under SYSTEM: the stamp must be
        # blank, and the second reopen must carry NO closing turn.
        _, tid, _ = _project_with_task(store)
        _close_as(store, tid, "abcd1234")
        store.update_task(tid, status="PENDING")
        assert store.update_task(tid, status="DONE")  # SYSTEM context
        assert store.get_task(tid)["closed_req_id"] == ""
        fired = []
        store.on_task_reopened = lambda *a: fired.append(a)
        store.update_task(tid, status="PENDING")
        assert fired[0][3] == ""

    def test_stamp_survives_unrelated_updates(self, store):
        _, tid, _ = _project_with_task(store)
        _close_as(store, tid, "abcd1234")
        token = request_id_context.set("ffff9999")
        try:
            store.update_task(tid, description="edited later")
            # Re-asserting DONE from another turn must not steal the close.
            store.update_task(tid, status="DONE")
        finally:
            request_id_context.reset(token)
        assert store.get_task(tid)["closed_req_id"] == "abcd1234"


class TestReopenHook:
    def test_reopen_fires_hook_with_closing_req(self, store):
        pid, tid, _ = _project_with_task(store)
        _close_as(store, tid, "abcd1234")
        fired = []
        store.on_task_reopened = lambda *a: fired.append(a)
        assert store.update_task(tid, status="PENDING")
        assert fired == [(pid, tid, "DONE", "abcd1234")]

    def test_reopen_on_active_project_still_fires(self, store):
        # A task revived on a project that never left ACTIVE (another task
        # still open) is just as much a delayed negative — the TASK-level
        # transition is the trigger, not project_reopened.
        pid, tid, _ = _project_with_task(store, extra_open_task=True)
        _close_as(store, tid, "abcd1234")
        assert store.get_project(pid)["status"] == "ACTIVE"
        fired = []
        store.on_task_reopened = lambda *a: fired.append(a)
        assert store.update_task(tid, status="IN_PROGRESS")
        assert len(fired) == 1

    def test_reopen_logs_task_reopened_event(self, store):
        pid, tid, _ = _project_with_task(store)
        _close_as(store, tid, "abcd1234")
        store.update_task(tid, status="PENDING")
        events = store.list_events(pid, event_type="task_reopened")
        assert len(events) == 1
        payload = events[0]["payload"]
        payload = json.loads(payload) if isinstance(payload, str) else payload
        assert payload["from_status"] == "DONE"
        assert payload["closed_req_id"] == "abcd1234"

    def test_non_done_transitions_do_not_fire(self, store):
        _, tid, _ = _project_with_task(store)
        fired = []
        store.on_task_reopened = lambda *a: fired.append(a)
        store.update_task(tid, status="IN_PROGRESS")
        store.update_task(tid, status="PENDING")
        assert fired == []

    def test_hook_failure_never_breaks_the_transition(self, store):
        _, tid, _ = _project_with_task(store)
        _close_as(store, tid, "abcd1234")

        def _boom(*_a):
            raise RuntimeError("hook exploded")

        store.on_task_reopened = _boom
        assert store.update_task(tid, status="PENDING")
        assert store.get_task(tid)["status"] == "PENDING"


class TestRetroNegative:
    def test_retro_negative_reuses_the_turns_own_components(self, tracker):
        _record_turn(tracker, "abcd1234", outcome=1.0)
        assert tracker.record_task_reopened_negative("abcd1234") is True
        retro = [s for s in _samples(tracker) if s.source == "task_reopened"]
        assert len(retro) == 1
        r = retro[0]
        assert r.outcome == pytest.approx(_TASK_REOPENED_GRADE)
        assert r.req_id == "abcd1234"
        # No-leakage: the FEATURE side is byte-for-byte the closing turn's.
        assert r.composite == pytest.approx(0.7)
        assert r.entropy_component == pytest.approx(0.6)
        assert r.competence_component == pytest.approx(0.8)
        assert r.effort_component == pytest.approx(0.4)
        assert r.entropy_observed and r.effort_observed
        assert r.domain == "coding"

    def test_idempotent_per_closing_turn(self, tracker):
        _record_turn(tracker, "abcd1234")
        assert tracker.record_task_reopened_negative("abcd1234") is True
        assert tracker.record_task_reopened_negative("abcd1234") is False
        retro = [s for s in _samples(tracker) if s.source == "task_reopened"]
        assert len(retro) == 1

    def test_unjoined_req_id_is_a_noop(self, tracker):
        _record_turn(tracker, "abcd1234")
        assert tracker.record_task_reopened_negative("deadbeef") is False
        assert tracker.record_task_reopened_negative("") is False

    def test_already_negative_turn_not_double_counted(self, tracker):
        _record_turn(tracker, "abcd1234", outcome=0.2)
        assert tracker.record_task_reopened_negative("abcd1234") is False
        assert all(s.source != "task_reopened" for s in _samples(tracker))

    def test_req_id_round_trips_through_history(self, tracker):
        _record_turn(tracker, "abcd1234")
        assert _samples(tracker)[0].req_id == "abcd1234"

    def test_legacy_rows_default_to_empty_req_id(self, tracker):
        tracker.dir.mkdir(parents=True, exist_ok=True)
        with tracker.history_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "composite": 0.5, "entropy_component": 0.5,
                "competence_component": 0.5, "uncertainty_pressure": 0.0,
                "outcome": 1.0, "domain": "", "ts": "t"}) + "\n")
        assert _samples(tracker)[0].req_id == ""


class TestEndToEnd:
    def test_close_reopen_writes_retro_row(self, store, tracker):
        # Wire store -> tracker exactly as main.py does.
        def _hook(_pid, _tid, _from, closed_req_id):
            if closed_req_id:
                tracker.record_task_reopened_negative(closed_req_id)

        store.on_task_reopened = _hook
        _record_turn(tracker, "abcd1234", outcome=1.0)
        _, tid, _ = _project_with_task(store)
        _close_as(store, tid, "abcd1234")
        store.update_task(tid, status="PENDING")
        retro = [s for s in _samples(tracker) if s.source == "task_reopened"]
        assert len(retro) == 1
        assert retro[0].req_id == "abcd1234"

    def test_close_reopen_close_reopen_labels_each_closing_turn_once(
            self, store, tracker):
        def _hook(_pid, _tid, _from, closed_req_id):
            if closed_req_id:
                tracker.record_task_reopened_negative(closed_req_id)

        store.on_task_reopened = _hook
        _record_turn(tracker, "turn0001")
        _record_turn(tracker, "turn0002")
        _, tid, _ = _project_with_task(store)
        _close_as(store, tid, "turn0001")
        store.update_task(tid, status="PENDING")
        _close_as(store, tid, "turn0002")
        store.update_task(tid, status="PENDING")
        retro = sorted(s.req_id for s in _samples(tracker)
                       if s.source == "task_reopened")
        assert retro == ["turn0001", "turn0002"]
