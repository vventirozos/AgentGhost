"""§4EE pins for `core/feedback.py` — the human-label channel's survivors,
each with the world where the line decides (R4)."""
from __future__ import annotations

import datetime
import types

import pytest

from ghost_agent.core import feedback as FB


class _Traj:
    def __init__(self, tid, rid, extra=None):
        self.id, self.session_id = tid, ""
        self.extra = {"req_id": rid} if extra is None else extra
        self.outcome, self.failure_reason = "unknown", ""


class _DayCol:
    """A collector whose day partitions the test chooses; a day listed in
    `boom` raises on read."""
    redaction = None

    def __init__(self, by_day, boom=()):
        self.by_day, self.boom, self.updates, self.asked = by_day, set(boom), [], []

    def iter_trajectories(self, day=None):
        self.asked.append(day)
        if day in self.boom:
            raise OSError("bad day file")
        return iter(self.by_day.get(day, []))

    def update_outcome(self, tid, outcome, reason="", source="", **kw):
        self.updates.append((tid, outcome, reason, source))
        return True


def _day(back):
    return (datetime.datetime.utcnow().date() - datetime.timedelta(days=back)).strftime("%Y-%m-%d")


# ── find_trajectory_for_request ────────────────────────────────────────── #

def test_lookup_refuses_an_empty_id_or_a_missing_collector_without_scanning():
    col = _DayCol({_day(0): [_Traj("t", "r")]})
    assert FB.find_trajectory_for_request(col, "") is None and col.asked == []
    assert FB.find_trajectory_for_request(None, "r") is None


def test_lookup_survives_a_bad_day_and_keeps_scanning_older_days():
    col = _DayCol({_day(2): [_Traj("t2", "r")]}, boom={_day(0), _day(1)})
    found = FB.find_trajectory_for_request(col, "chatcmpl-r")
    assert found is not None and found.id == "t2", col.asked


def test_lookup_takes_the_last_match_of_the_newest_day():
    col = _DayCol({_day(0): [_Traj("a", "r"), _Traj("b", "r")], _day(1): [_Traj("c", "r")]})
    assert FB.find_trajectory_for_request(col, "r").id == "b"


# ── _stamp_cache ───────────────────────────────────────────────────────── #

def _ctx_with_cache(rows):
    from collections import OrderedDict
    return types.SimpleNamespace(
        _recent_trajectories_for_correction=OrderedDict((f"fp{i}", r) for i, r in enumerate(rows)))


def test_stamp_reaches_only_the_row_with_that_id_and_keeps_its_extra():
    other, mine = _Traj("other", "x"), _Traj("mine", "y", extra={"req_id": "y", "k": 1})
    ctx = _ctx_with_cache([other, mine])
    FB._stamp_cache(ctx, "mine", "failed", "nope")
    assert mine.outcome == "failed" and mine.failure_reason == "nope"
    assert mine.extra == {"req_id": "y", "k": 1, "human_labeled": True}
    assert other.outcome == "unknown" and "human_labeled" not in other.extra


def test_stamp_creates_extra_when_the_row_has_none():
    row = _Traj("mine", "y", extra=None); row.extra = None
    FB._stamp_cache(_ctx_with_cache([row]), "mine", "passed", "")
    assert row.extra == {"human_labeled": True}


# ── apply_human_label: codes, log line, side effects ───────────────────── #

def _agent(col, tracker=None, **args):
    ctx = types.SimpleNamespace(trajectory_collector=col, calibration_tracker=tracker,
                                _recent_trajectories_for_correction=None, self_model=None,
                                args=types.SimpleNamespace(**args) if args else None)
    return types.SimpleNamespace(context=ctx)


def _capture(monkeypatch):
    seen = []
    monkeypatch.setattr(FB, "pretty_log", lambda t, m, **kw: seen.append((t, m, kw)))
    return seen


def test_empty_request_id_is_a_bad_request_not_a_scan():
    col = _DayCol({_day(0): [_Traj("t", "r")]})
    out = FB.apply_human_label(_agent(col), "chatcmpl-", "positive")
    assert out["ok"] is False and out["code"] == "bad_request" and col.asked == []


def test_line_mentions_calibration_only_when_a_tracker_exists(monkeypatch):
    seen = _capture(monkeypatch)
    col = _DayCol({_day(0): [_Traj("t", "r")]})
    FB.apply_human_label(_agent(col), "r", "positive")
    line = [m for t, m, kw in seen if t == "Human Feedback"][-1]
    assert "calibration" not in line, line

    class _Boom:
        def record_human_label(self, rid, passed):
            raise RuntimeError("disk")
    seen.clear()
    FB.apply_human_label(_agent(_DayCol({_day(0): [_Traj("t", "r")]}), _Boom()), "r", "positive")
    line = [m for t, m, kw in seen if t == "Human Feedback"][-1]
    assert "calibration write failed" in line, line


def test_identical_repeat_is_reported_unchanged_and_not_logged_twice(monkeypatch):
    seen = _capture(monkeypatch)

    class _Col(_DayCol):
        def update_outcome(self, tid, outcome, reason="", source="", **kw):
            self.updates.append((tid, outcome, reason, source))
            return "unchanged" if len(self.updates) > 1 else True
    col = _Col({_day(0): [_Traj("t", "r")]})
    first = FB.apply_human_label(_agent(col), "r", "negative", "wrong")
    second = FB.apply_human_label(_agent(col), "r", "negative", "wrong")
    assert first.get("unchanged") is None and second.get("unchanged") is True
    assert len([1 for t, m, kw in seen if t == "Human Feedback"]) == 1


def test_line_carries_the_redacted_reason_and_the_right_icon(monkeypatch):
    from ghost_agent.utils.logging import Icons
    seen = _capture(monkeypatch)
    col = _DayCol({_day(0): [_Traj("t", "r")]})
    FB.apply_human_label(_agent(col), "r", "negative", "wrong total")
    t, m, kw = [x for x in seen if x[0] == "Human Feedback"][-1]
    assert "wrong total" in m and kw["icon"] == Icons.FEEDBACK_NEG
    seen.clear()
    FB.apply_human_label(_agent(col), "r", "positive")
    t, m, kw = [x for x in seen if x[0] == "Human Feedback"][-1]
    assert " · " not in m.split("→ passed")[-1] and kw["icon"] == Icons.FEEDBACK_POS, m


def test_a_redaction_failure_withholds_the_reason_from_the_line(monkeypatch):
    seen = _capture(monkeypatch)
    import ghost_agent.distill.redact as R
    monkeypatch.setattr(R, "redact_text", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    col = _DayCol({_day(0): [_Traj("t", "r")]})
    out = FB.apply_human_label(_agent(col), "r", "negative", "secret token abc")
    assert out["ok"] is True
    line = [m for t, m, kw in seen if t == "Human Feedback"][-1]
    assert "reason withheld" in line and "secret token" not in line


def test_prm_skip_warning_survives_a_raising_liveness_probe(monkeypatch):
    seen = _capture(monkeypatch)
    import ghost_agent.core.agent as A
    monkeypatch.setattr(A, "prm_consumer_is_live", lambda ctx: (_ for _ in ()).throw(RuntimeError("x")))
    col = _DayCol({_day(0): [_Traj("t", "r")]})
    agent = _agent(col, prm_online_update=True)
    agent.context.prm_scorer = types.SimpleNamespace(has_model=False)
    FB.apply_human_label(agent, "r", "negative", "bad")
    titles = [t for t, m, kw in seen]
    assert "PRM Online Skipped (feedback channel)" in titles, titles
    msg = [m for t, m, kw in seen if t.startswith("PRM Online")][-1]
    assert "SKIPPING too" in msg


def test_an_exception_before_the_write_is_a_dict_not_None(monkeypatch):
    monkeypatch.setattr(FB, "find_trajectory_for_request",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    out = FB.apply_human_label(_agent(_DayCol({})), "r", "positive")
    assert out == {"ok": False, "error": "internal error applying label", "code": "unavailable"}
