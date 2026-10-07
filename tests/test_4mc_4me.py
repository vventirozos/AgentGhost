"""§4MC (growth) and §4ME (answer quality): behaviour pins."""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ── §4MC MAJOR 1: the log rotates, and the liveness read is bounded ──

def test_the_agent_log_rotates(tmp_path, monkeypatch):
    from logging.handlers import RotatingFileHandler
    from ghost_agent.utils import logging as GL
    monkeypatch.setenv("GHOST_LOG_MAX_MB", "0.001")
    monkeypatch.setenv("GHOST_LOG_BACKUPS", "2")
    setup = getattr(GL, "setup_logging", None) or getattr(GL, "configure_logging")
    setup(str(tmp_path / "agent.log"))
    lg = logging.getLogger("GhostAgent")
    assert any(isinstance(h, RotatingFileHandler) for h in lg.handlers)
    for i in range(200):
        lg.info("line %d %s", i, "x" * 50)
    assert (tmp_path / "agent.log.1").exists()
    for h in list(lg.handlers):
        if isinstance(h, RotatingFileHandler):
            h.close()
            lg.removeHandler(h)


def _stamp(ts):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def test_the_liveness_read_covers_the_window_across_rotation_and_no_more(tmp_path):
    from ghost_agent.core import liveness as L
    L._LOG_CACHE.clear()
    now = time.time()
    old = [f"{_stamp(now - 40 * 86400 + i)} - GhostAgent - INFO - ancient {i}\n" for i in range(3000)]
    rot = [f"{_stamp(now - 3 * 86400 + i)} - GhostAgent - INFO - rotated {i}\n" for i in range(50)]
    cur = [f"{_stamp(now - 3600 + i)} - GhostAgent - INFO - current {i}\n" for i in range(50)]
    log = tmp_path / "ghost-agent.log"
    (tmp_path / "ghost-agent.log.1").write_text("".join(old + rot))
    log.write_text("".join(cur))
    e = L._log_entries(log)
    texts = "".join(line for _, line in e)
    assert "ancient" not in texts
    assert "rotated 0" in texts and "current 49" in texts
    assert [t for t, _ in e] == sorted(t for t, _ in e)


def test_a_log_probe_still_counts_inside_its_window(tmp_path):
    from ghost_agent.core import liveness as L
    L._LOG_CACHE.clear()
    now = time.time()
    lines = [f"{_stamp(now - 2 * 86400)} - GhostAgent - INFO - complexity router tick\n",
             f"{_stamp(now - 600)} - GhostAgent - INFO - complexity router tick\n"]
    (tmp_path / "system").mkdir()
    (tmp_path / "system" / "ghost-agent.log").write_text("".join(lines))
    res = L._log_probe(r"complexity router", window_h=24.0)(tmp_path)
    assert res.count == 1


# ── §4MC MAJOR 2: readers that want recent history read recent days ──

def test_since_days_reads_only_recent_partitions(tmp_path):
    import datetime as dt
    from ghost_agent.distill.collector import TrajectoryCollector
    col = TrajectoryCollector(tmp_path)
    root = col.root
    for days, tid in ((60, "old"), (2, "new")):
        d = root / (dt.date.today() - dt.timedelta(days=days)).isoformat()
        d.mkdir(parents=True, exist_ok=True)
        (d / "session-s.jsonl").write_text(json.dumps({"id": tid, "task_kind": "chat"}) + "\n")
    all_ids = {t.id for t in col.iter_trajectories()}
    recent = {t.id for t in col.iter_trajectories(since_days=30)}
    assert all_ids == {"old", "new"} and recent == {"new"}


# ── §4MC MINOR 7: the reflected-id cap drops the OLDEST ──

def test_the_reflected_id_cap_drops_the_oldest(tmp_path):
    from ghost_agent.core.agent import _OrderedIdSet
    s = _OrderedIdSet(f"t{i:03d}" for i in range(10))
    s.add("t999")
    s.discard("t005")
    assert list(s)[-3:] == ["t008", "t009", "t999"]
    assert list(s)[-5:][0] == "t006"


# ── §4ME F1: what a notification reached ──

async def test_a_notification_says_it_reached_only_the_owner(monkeypatch, tmp_path):
    import ghost_agent.tools.notify_tool as N
    from ghost_agent.core.autonomous_activity import ActivityLog
    monkeypatch.setattr(N, "_sent_timestamps", [])
    monkeypatch.setattr(N, "get_activity_log", lambda ctx: ActivityLog(tmp_path / "a.jsonl"))
    out = await N.tool_notify_operator(message="posted to #general", context=MagicMock())
    assert "ONLY to the operator" in out and "NOT done" in out


def test_a_link_to_a_missing_file_is_dropped(tmp_path):
    from ghost_agent.core.agent import _drop_missing_download_links
    (tmp_path / "real.png").write_bytes(b"x")
    text, dropped = _drop_missing_download_links(
        "![a](/api/download/real.png) ![b](/api/download/gen_1024x1024.png)", tmp_path)
    assert dropped == ["gen_1024x1024.png"] and "real.png" in text


# ── §4ME F3: stale world facts are not stored ──

@pytest.mark.parametrize("pred,stale", [("HAS_VERSION", True), ("LATEST_RELEASE", True), ("PRICE", True),
                                        ("WORKS_AT", False), ("LIVES_IN", False), ("LOCATED_IN", False),
                                        ("RELEASED_IN", False), ("ACCURATE", False), ("SEPARATED_FROM", False)])
def test_moving_target_world_facts_are_recognised(pred, stale):
    from ghost_agent.core.agent import _TRANSIENT_WORLD_PREDICATE
    assert bool(_TRANSIENT_WORLD_PREDICATE.search(pred)) is stale


# ── §4ME F5: the background banner ──

def test_a_notification_from_an_owner_conversation_is_not_repeated():
    from ghost_agent.core.autonomous_activity import ActivityRecord, render_activity_digest, SEVERITY_NOTIFY
    mk = lambda summ, rid: ActivityRecord(ts=time.time(), phase="agent_message", summary=summ,
                                          severity=SEVERITY_NOTIFY, meta={"req_id": rid})
    late = ActivityRecord(ts=time.time(), phase="agent_message", summary="Done — the build you asked about",
                          severity=SEVERITY_NOTIFY, meta={"req_id": "39c394ca", "auto": "job finished"})
    out = render_activity_digest([mk("hello from chat", "39c394ca"), mk("job done", "sched-nightly"), late],
                                 severities=(SEVERITY_NOTIFY,))
    assert "job done" in out and "hello from chat" not in out
    assert "the build you asked about" in out          # written LATER under the turn's id


@pytest.mark.parametrize("msg,bound", [("hi, one line please", True), ("answer briefly", True),
                                       ("μία γραμμή μόνο", True), ("reply with only the number", True),
                                       ("what is the weather in Athens", False),
                                       ("tell me about the project", False), ("what exactly went wrong", False),
                                       ("is it just the cache?", False), ("only the first build failed", False)])
def test_a_length_bound_request_holds_the_banner(msg, bound):
    from ghost_agent.core.agent import _FORMAT_BOUND_RE
    assert bool(_FORMAT_BOUND_RE.search(msg)) is bound


# ── §4ME F7: an all-list reply is judged on its own summaries ──

def test_an_all_greek_digest_answering_english_is_caught():
    from ghost_agent.core.reply_language import reply_language_mismatch
    g = "\n".join(f"- «Τίτλος {i}» — Η κυβέρνηση ανακοίνωσε νέα μέτρα για την οικονομία σήμερα" for i in range(4))
    e = "\n".join(f"- «Τίτλος {i}» — The government announced new economic measures today" for i in range(4))
    q = "What are the news headlines today in Greece?"
    assert reply_language_mismatch(q, g) == ("English", "Greek")
    assert reply_language_mismatch(q, e) is None


# ── §4ME F8: a URL is not a file ──

def test_a_url_passed_as_a_file_names_the_right_tool(tmp_path):
    from ghost_agent.tools.file_system import _missing_file_message
    msg = _missing_file_message("https://example.org/page", tmp_path)
    assert "browser" in msg and "download" in msg


def test_the_liveness_read_does_not_parse_the_old_part(tmp_path, monkeypatch):
    from ghost_agent.core import liveness as L
    L._LOG_CACHE.clear()
    now = time.time()
    old = "".join(f"{_stamp(now - 40 * 86400 + i)} - GhostAgent - INFO - ancient {i}\n" for i in range(30000))
    new = "".join(f"{_stamp(now - 60 + i)} - GhostAgent - INFO - fresh {i}\n" for i in range(20))
    log = tmp_path / "ghost-agent.log"
    log.write_text(old + new)
    calls = {"n": 0}
    real = L._line_ts

    def counting(line):
        calls["n"] += 1
        return real(line)
    monkeypatch.setattr(L, "_line_ts", counting)
    e = L._log_entries(log)
    assert len(e) == 20 and calls["n"] < 2000          # a binary search, not a 30k-line parse


def test_a_greek_digest_of_links_is_not_called_english():
    from ghost_agent.core.reply_language import reply_language_mismatch
    q = "Βρες μου τα σημερινά νέα για την οικονομία στην Ελλάδα"
    r = "\n".join(f"- Πληθωρισμός στο 2,9% — https://www.kathimerini.gr/economy/{i}/inflation-news-today"
                  for i in range(5))
    assert reply_language_mismatch(q, r) is None


def test_user_turns_are_counted_across_a_rotation(tmp_path):
    from ghost_agent.core import liveness as L
    L._LOG_CACHE.clear()
    now = time.time()
    (tmp_path / "system").mkdir()
    line = lambda ts, i: f"{_stamp(ts)} - GhostAgent - INFO - request started r{i} origin=user\n"
    (tmp_path / "system" / "ghost-agent.log.1").write_text("".join(line(now - 7200 + i, i) for i in range(3)))
    (tmp_path / "system" / "ghost-agent.log").write_text("".join(line(now - 60 + i, 10 + i) for i in range(2)))
    n, total, _ = L._count_user_turns(tmp_path, 24.0)
    assert n == 5 and total == 5


@pytest.mark.parametrize("deadline,floor", [(1800, 300.0), (600, 150.0), (300, 75.0), (60, 30.0), (0, 300.0), (None, 300.0)])
def test_the_report_reserve_scales_with_the_deadline(deadline, floor):
    from ghost_agent.core.agent import effective_report_floor
    assert effective_report_floor(deadline) == floor
