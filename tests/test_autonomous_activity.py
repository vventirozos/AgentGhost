"""Tests for the autonomous-activity ledger + outbound notifier
(core.autonomous_activity / utils.notify — the agent's "mouth",
2026-07-11) and their wiring into main.py / agent.py / api.routes.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_agent.core.autonomous_activity import (
    SEVERITY_INFO, SEVERITY_NOTIFY,
    ActivityLog, ActivityRecord,
    is_internal_request,
    load_offset, save_offset,
    load_consumer_offset, save_consumer_offset,
    render_activity_digest,
    summarize_turn_content, record_scheduled_result,
)
from ghost_agent.utils.notify import (
    OutboundNotifier, notifier_from_config, url_needs_tor,
)


# ──────────────────────────────────────────────────────────────────────
# ActivityLog
# ──────────────────────────────────────────────────────────────────────

class TestActivityLog:
    def test_record_read_roundtrip(self, tmp_path):
        log = ActivityLog(tmp_path / "act.jsonl")
        assert log.record("dream", "consolidated 2 heuristics", foo="bar")
        recs, off = log.read_since(0)
        assert len(recs) == 1
        r = recs[0]
        assert r.phase == "dream"
        assert r.summary == "consolidated 2 heuristics"
        assert r.severity == SEVERITY_INFO
        assert r.meta == {"foo": "bar"}
        assert off == log.current_offset() > 0

    def test_offset_advances_and_second_read_empty(self, tmp_path):
        log = ActivityLog(tmp_path / "act.jsonl")
        log.record("a", "one")
        recs, off = log.read_since(0)
        assert len(recs) == 1
        recs2, off2 = log.read_since(off)
        assert recs2 == [] and off2 == off
        log.record("b", "two")
        recs3, off3 = log.read_since(off)
        assert [r.summary for r in recs3] == ["two"] and off3 > off

    def test_severity_filter_returns_only_notify_but_advances_offset(self, tmp_path):
        log = ActivityLog(tmp_path / "act.jsonl")
        log.record("a", "info item", severity=SEVERITY_INFO)
        log.record("b", "notify item", severity=SEVERITY_NOTIFY)
        recs, off = log.read_since(0, severity=SEVERITY_NOTIFY)
        assert [r.summary for r in recs] == ["notify item"]
        assert off == log.current_offset()  # info line consumed too

    def test_stale_offset_rebaselines_silently(self, tmp_path):
        log = ActivityLog(tmp_path / "act.jsonl")
        log.record("a", "one")
        recs, off = log.read_since(10_000_000)
        assert recs == [] and off == log.current_offset()

    def test_malformed_line_skipped_but_consumed(self, tmp_path):
        p = tmp_path / "act.jsonl"
        log = ActivityLog(p)
        log.record("a", "good one")
        with open(p, "a") as f:
            f.write("{not json}\n")
        log.record("b", "good two")
        recs, off = log.read_since(0)
        assert [r.summary for r in recs] == ["good one", "good two"]
        assert off == log.current_offset()

    def test_partial_tail_line_not_consumed(self, tmp_path):
        p = tmp_path / "act.jsonl"
        log = ActivityLog(p)
        log.record("a", "complete")
        with open(p, "ab") as f:
            f.write(b'{"ts": 1, "phase": "b", "summary": "parti')  # no \n
        recs, off = log.read_since(0)
        assert [r.summary for r in recs] == ["complete"]
        assert off < log.current_offset()  # tail left for next read

    def test_on_notify_fired_only_for_notify(self, tmp_path):
        seen = []
        log = ActivityLog(tmp_path / "act.jsonl", on_notify=seen.append)
        log.record("a", "quiet", severity=SEVERITY_INFO)
        log.record("b", "loud", severity=SEVERITY_NOTIFY)
        assert [r.summary for r in seen] == ["loud"]

    def test_on_notify_exception_swallowed(self, tmp_path):
        def boom(rec):
            raise RuntimeError("transport down")
        log = ActivityLog(tmp_path / "act.jsonl", on_notify=boom)
        assert log.record("a", "x", severity=SEVERITY_NOTIFY) is True
        recs, _ = log.read_since(0)
        assert len(recs) == 1  # record persisted despite callback failure

    def test_huge_summary_clamped_and_never_raises(self, tmp_path):
        log = ActivityLog(tmp_path / "act.jsonl")
        assert log.record("a", "x" * 100_000)
        recs, _ = log.read_since(0)
        assert len(recs[0].summary) <= 600

    def test_unwritable_path_returns_false(self, tmp_path):
        # Path IS a directory → open(...,'a') fails → False, no raise.
        log = ActivityLog(tmp_path)
        assert log.record("a", "x") is False

    def test_bad_severity_coerced_to_info(self, tmp_path):
        log = ActivityLog(tmp_path / "act.jsonl")
        log.record("a", "x", severity="bogus")
        recs, _ = log.read_since(0)
        assert recs[0].severity == SEVERITY_INFO


# ──────────────────────────────────────────────────────────────────────
# Watermarks
# ──────────────────────────────────────────────────────────────────────

class TestWatermarks:
    def test_digest_offset_none_on_first_run(self, tmp_path):
        assert load_offset(tmp_path / "wm.json") is None

    def test_digest_offset_roundtrip(self, tmp_path):
        p = tmp_path / "wm.json"
        save_offset(p, 1234)
        assert load_offset(p) == 1234

    def test_consumer_offset_none_when_unknown(self, tmp_path):
        p = tmp_path / "consumers.json"
        assert load_consumer_offset(p, "slack") is None
        save_consumer_offset(p, "slack", 10)
        assert load_consumer_offset(p, "slack") == 10
        assert load_consumer_offset(p, "other") is None

    def test_consumer_offsets_independent(self, tmp_path):
        p = tmp_path / "consumers.json"
        save_consumer_offset(p, "slack", 10)
        save_consumer_offset(p, "ntfy", 99)
        assert load_consumer_offset(p, "slack") == 10
        assert load_consumer_offset(p, "ntfy") == 99


# ──────────────────────────────────────────────────────────────────────
# Digest rendering
# ──────────────────────────────────────────────────────────────────────

def _rec(phase, summary, severity=SEVERITY_INFO):
    return ActivityRecord(ts=time.time(), phase=phase, summary=summary,
                          severity=severity)


class TestRenderDigest:
    def test_empty_is_empty_string(self):
        assert render_activity_digest([]) == ""

    def test_project_phase_excluded(self):
        out = render_activity_digest([_rec("project", "needs input")])
        assert out == ""

    def test_notify_items_lead(self):
        out = render_activity_digest([
            _rec("dream", "info item"),
            _rec("scheduled_task", "notify item", SEVERITY_NOTIFY),
        ])
        lines = out.split("\n")
        assert "notify item" in lines[1]
        assert "info item" in lines[2]

    def test_caps_items_with_more_tail(self):
        recs = [_rec("dream", f"item {i}") for i in range(9)]
        out = render_activity_digest(recs, max_items=6)
        assert "…and 3 more" in out

    def test_block_stays_under_banner_strip_bound(self):
        # 6 maximal-summary items must render < 1500 chars, else
        # _strip_leading_banners won't peel the block and the correction
        # fingerprint breaks (the 2026-07-07 class of bug).
        recs = [_rec("scheduled_task", "y" * 600, SEVERITY_NOTIFY)
                for _ in range(9)]
        out = render_activity_digest(recs, max_items=6)
        assert len(out) < 1500

    def test_strip_leading_banners_peels_activity_digest(self):
        from ghost_agent.core.agent import GhostAgent
        digest = render_activity_digest(
            [_rec("dream", "one"), _rec("reflection", "two")])
        body = "The actual answer body."
        combined = f"{digest}\n\n---\n\n{body}"
        assert GhostAgent._strip_leading_banners(combined) == body

    def test_strip_leading_banners_peels_four_stacked_blocks(self):
        from ghost_agent.core.agent import GhostAgent
        body = "Real answer."
        stacked = body
        for banner in ("**Correction**", "*Clarifying:* what?",
                       "**While you were away** — 1 task",
                       "**Background activity while you were away:**\n  - x"):
            stacked = f"{banner}\n\n---\n\n{stacked}"
        assert GhostAgent._strip_leading_banners(stacked) == body


# ──────────────────────────────────────────────────────────────────────
# Scheduled-turn capture helpers
# ──────────────────────────────────────────────────────────────────────

class TestScheduledCapture:
    def test_summarize_collapses_whitespace_and_limits(self):
        out = summarize_turn_content("a\n\n  b   c" + "x" * 500, limit=50)
        assert out.startswith("a b c")
        assert len(out) <= 50

    def test_summarize_strips_stacked_banners(self):
        body = ("**While you were away** — I advanced 1 task(s)\n\n---\n\n"
                "**Background activity while you were away:**\n  - [dream] z"
                "\n\n---\n\nActual conclusion here.")
        assert summarize_turn_content(body) == "Actual conclusion here."

    def test_record_scheduled_result_writes_notify(self, tmp_path):
        log = ActivityLog(tmp_path / "act.jsonl")
        record_scheduled_result(log, job_id="job1", task_name="morning brief",
                                content="Nothing on fire.", ok=True,
                                duration_s=12.3)
        recs, _ = log.read_since(0)
        assert len(recs) == 1
        assert recs[0].phase == "scheduled_task"
        assert recs[0].severity == SEVERITY_NOTIFY
        assert "morning brief" in recs[0].summary
        assert "Nothing on fire." in recs[0].summary
        assert recs[0].meta["ok"] == "True"

    def test_record_scheduled_result_failure_and_none_log(self, tmp_path):
        record_scheduled_result(None, job_id="j", content="x")  # no-op
        log = ActivityLog(tmp_path / "act.jsonl")
        record_scheduled_result(log, job_id="job2", content="Boom",
                                ok=False)
        recs, _ = log.read_since(0)
        assert "FAILED" in recs[0].summary


# ──────────────────────────────────────────────────────────────────────
# Internal-request gate
# ──────────────────────────────────────────────────────────────────────

class TestInternalRequestGate:
    @pytest.mark.parametrize("rid,expected", [
        ("sched-job1", True), ("job-abc", True), ("sub-xyz", True),
        ("slack-123", False), ("a1b2c3d4", False), (None, False), ("", False),
    ])
    def test_prefixes(self, rid, expected):
        assert is_internal_request(rid) is expected


# ──────────────────────────────────────────────────────────────────────
# Outbound notifier
# ──────────────────────────────────────────────────────────────────────

class TestUrlNeedsTor:
    @pytest.mark.parametrize("url,needs", [
        ("http://127.0.0.1:8090/topic", False),
        ("http://localhost:1234/x", False),
        ("http://192.168.0.24:8000/hook", False),
        ("http://10.1.2.3/hook", False),
        ("http://100.100.1.5/hook", False),       # Tailscale CGNAT
        ("http://ghost.lan:8090/t", False),
        ("http://ghost.local/t", False),
        ("http://ghost/t", False),                # bare intranet hostname
        ("https://ntfy.sh/mytopic", True),
        ("http://8.8.8.8/hook", True),
        ("https://hooks.example.com/x", True),
        ("not a url at all://", True),            # fail toward Tor
    ])
    def test_classification(self, url, needs):
        assert url_needs_tor(url) is needs


class TestOutboundNotifier:
    def _mock_transport(self, log, status=200):
        import httpx

        def handler(request):
            log.append(request)
            return httpx.Response(status)
        return httpx.MockTransport(handler)

    def test_unconfigured_send_returns_false(self):
        n = OutboundNotifier()
        assert n.configured is False
        assert asyncio.run(n.send(title="t", body="b")) is False

    def test_webhook_posts_json(self):
        calls = []
        n = OutboundNotifier(webhook_url="http://192.168.0.10/hook",
                             transport=self._mock_transport(calls))
        ok = asyncio.run(n.send(title="Ghost", body="hello",
                                severity="notify", phase="project", ts=1.0))
        assert ok is True and n.sent_count == 1
        payload = json.loads(calls[0].content)
        assert payload == {"title": "Ghost", "body": "hello",
                           "severity": "notify", "phase": "project",
                           "ts": 1.0}

    def test_ntfy_posts_text_with_title_header(self):
        calls = []
        n = OutboundNotifier(ntfy_url="http://ghost.lan:8090/agent",
                             transport=self._mock_transport(calls))
        ok = asyncio.run(n.send(title="Ghost — project", body="needs you",
                                severity="notify"))
        assert ok is True
        req = calls[0]
        assert req.content == b"needs you"
        assert req.headers["Title"] == "Ghost — project".encode(
            "ascii", "ignore").decode("ascii")
        assert req.headers["Priority"] == "high"

    def test_non_2xx_retries_once_then_fails(self):
        calls = []
        n = OutboundNotifier(webhook_url="http://192.168.0.10/hook",
                             transport=self._mock_transport(calls, status=500))
        ok = asyncio.run(n.send(title="t", body="b"))
        assert ok is False and n.failed_count == 1
        assert len(calls) == 2  # one retry

    def test_public_target_without_tor_is_skipped(self):
        calls = []
        n = OutboundNotifier(webhook_url="https://hooks.example.com/x",
                             tor_proxy=None,
                             transport=self._mock_transport(calls))
        ok = asyncio.run(n.send(title="t", body="b"))
        assert ok is False
        assert calls == []  # never attempted a direct public connect

    def test_proxy_for_public_upgrades_to_socks5h(self):
        n = OutboundNotifier(webhook_url="https://hooks.example.com/x",
                             tor_proxy="socks5://127.0.0.1:9050")
        assert n._proxy_for("https://hooks.example.com/x") == \
            "socks5h://127.0.0.1:9050"
        assert n._proxy_for("http://192.168.0.10/hook") is None

    def test_notifier_from_config_env_fallback(self, monkeypatch):
        monkeypatch.setenv("GHOST_NOTIFY_WEBHOOK", "http://ghost.lan/hook")
        monkeypatch.delenv("GHOST_NOTIFY_NTFY", raising=False)
        n = notifier_from_config(None, tor_proxy="socks5://127.0.0.1:9050")
        assert n.webhook_url == "http://ghost.lan/hook"
        assert n.configured

    def test_notifier_from_config_args_win(self, monkeypatch):
        monkeypatch.setenv("GHOST_NOTIFY_WEBHOOK", "http://env-host.lan/hook")
        args = SimpleNamespace(notify_webhook="http://flag-host.lan/hook",
                               notify_ntfy="")
        n = notifier_from_config(args)
        assert n.webhook_url == "http://flag-host.lan/hook"

    def test_send_soon_without_loop_delivers_via_thread(self):
        calls = []
        n = OutboundNotifier(webhook_url="http://192.168.0.10/hook",
                             transport=self._mock_transport(calls))
        rec = SimpleNamespace(phase="project", summary="needs you",
                              severity="notify", ts=1.0)
        n.send_soon(rec)
        deadline = time.time() + 5
        while time.time() < deadline and n.sent_count == 0:
            time.sleep(0.02)
        assert n.sent_count == 1 and len(calls) == 1

    def test_activity_log_on_notify_integration(self, tmp_path):
        calls = []
        n = OutboundNotifier(webhook_url="http://192.168.0.10/hook",
                             transport=self._mock_transport(calls))
        log = ActivityLog(tmp_path / "act.jsonl", on_notify=n.send_soon)
        log.record("project", "task needs your input: deploy?",
                   severity=SEVERITY_NOTIFY)
        deadline = time.time() + 5
        while time.time() < deadline and n.sent_count == 0:
            time.sleep(0.02)
        assert n.sent_count == 1
        payload = json.loads(calls[0].content)
        assert "deploy?" in payload["body"]


# ──────────────────────────────────────────────────────────────────────
# /api/notifications endpoints
# ──────────────────────────────────────────────────────────────────────

fastapi = pytest.importorskip("fastapi")
from fastapi import FastAPI              # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


def _make_app(tmp_path, with_log=True):
    from ghost_agent.api.routes import router
    app = FastAPI()
    app.include_router(router)
    memory_dir = tmp_path / "system" / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    context = SimpleNamespace(
        args=SimpleNamespace(api_key="", model="test-model"),
        memory_dir=memory_dir,
    )
    if with_log:
        context.activity_log = ActivityLog(
            memory_dir.parent / "autonomous_activity.jsonl")
    agent = SimpleNamespace(context=context)
    app.state.agent = agent
    app.state.context = context
    app.state.args = context.args
    return app, context


class TestNotificationsAPI:
    def test_disabled_without_activity_log(self, tmp_path):
        app, _ = _make_app(tmp_path, with_log=False)
        with TestClient(app) as c:
            r = c.get("/api/notifications/pending")
            assert r.status_code == 200
            assert r.json() == {"enabled": False, "records": [],
                                "watermark": 0}

    def test_first_contact_baselines(self, tmp_path):
        app, ctx = _make_app(tmp_path)
        ctx.activity_log.record("project", "old item",
                                severity=SEVERITY_NOTIFY)
        with TestClient(app) as c:
            r = c.get("/api/notifications/pending?consumer=slack").json()
            assert r["baseline"] is True and r["records"] == []
            assert r["watermark"] == ctx.activity_log.current_offset()

    def test_ack_then_new_records_flow(self, tmp_path):
        app, ctx = _make_app(tmp_path)
        with TestClient(app) as c:
            base = c.get("/api/notifications/pending?consumer=slack").json()
            ack = c.post("/api/notifications/ack",
                         json={"consumer": "slack",
                               "watermark": base["watermark"]})
            assert ack.status_code == 200
            ctx.activity_log.record("scheduled_task", "cron says hi",
                                    severity=SEVERITY_NOTIFY)
            ctx.activity_log.record("dream", "info only",
                                    severity=SEVERITY_INFO)
            r = c.get("/api/notifications/pending?consumer=slack").json()
            assert [x["summary"] for x in r["records"]] == ["cron says hi"]
            # Not acked yet → re-served on the next poll.
            r2 = c.get("/api/notifications/pending?consumer=slack").json()
            assert len(r2["records"]) == 1
            c.post("/api/notifications/ack",
                   json={"consumer": "slack", "watermark": r2["watermark"]})
            r3 = c.get("/api/notifications/pending?consumer=slack").json()
            assert r3["records"] == []

    def test_pending_scans_past_info_noise(self, tmp_path):
        # THE 2026-07-13 wedge, server half: read_since's limit bounds
        # SCANNED LINES, so a 20-line poll window full of info-severity
        # records (dream/self-play spam) returned [] even though notify
        # records sat just beyond it. The endpoint must scan forward
        # (bounded) until it finds notify records or EOF.
        app, ctx = _make_app(tmp_path)
        with TestClient(app) as c:
            base = c.get("/api/notifications/pending?consumer=slack").json()
            c.post("/api/notifications/ack",
                   json={"consumer": "slack", "watermark": base["watermark"]})
            for i in range(250):
                ctx.activity_log.record("dream", f"info noise {i}",
                                        severity=SEVERITY_INFO)
            ctx.activity_log.record("agent_message", "the real notification",
                                    severity=SEVERITY_NOTIFY)
            r = c.get("/api/notifications/pending?consumer=slack&limit=20").json()
            assert [x["summary"] for x in r["records"]] == [
                "the real notification"]
            assert r["watermark"] == ctx.activity_log.current_offset()

    def test_pending_empty_window_advances_watermark_to_eof(self, tmp_path):
        # An all-info ledger tail must still return an EOF watermark so an
        # always-acking client makes progress instead of re-scanning the
        # same window forever.
        app, ctx = _make_app(tmp_path)
        with TestClient(app) as c:
            base = c.get("/api/notifications/pending?consumer=slack").json()
            c.post("/api/notifications/ack",
                   json={"consumer": "slack", "watermark": base["watermark"]})
            for i in range(30):
                ctx.activity_log.record("self_play", f"info {i}",
                                        severity=SEVERITY_INFO)
            r = c.get("/api/notifications/pending?consumer=slack&limit=20").json()
            assert r["records"] == []
            assert r["watermark"] == ctx.activity_log.current_offset()

    def test_ack_invalid_body_400(self, tmp_path):
        app, _ = _make_app(tmp_path)
        with TestClient(app) as c:
            r = c.post("/api/notifications/ack", json={"consumer": "x"})
            assert r.status_code == 400
            r2 = c.post("/api/notifications/ack",
                        content=b"not json",
                        headers={"Content-Type": "application/json"})
            assert r2.status_code == 400


# ──────────────────────────────────────────────────────────────────────
# Wiring (source-inspection, matching the suite's convention for main.py
# closures that can't be imported directly)
# ──────────────────────────────────────────────────────────────────────

_SRC = Path(__file__).resolve().parents[1] / "src" / "ghost_agent"


class TestWiring:
    def test_main_wires_ledger_and_scheduled_capture(self):
        src = (_SRC / "main.py").read_text()
        assert "ActivityLog(" in src
        assert "notifier_from_config(" in src
        assert src.count("record_scheduled_result(") >= 2  # ok + failure
        assert "--notify-webhook" in src and "--notify-ntfy" in src

    def test_finalize_renders_activity_digest_with_internal_gate(self):
        src = (_SRC / "core" / "agent.py").read_text()
        assert "render_activity_digest" in src
        assert src.count("is_internal_request") >= 2  # project + activity
        assert "activity_digest.json" in src

    def test_biological_phases_record_activity(self):
        import re as _re
        src = (_SRC / "core" / "agent.py").read_text()
        recorded = set(_re.findall(
            r'_record_autonomous_activity\(\s*"(\w+)"', src))
        expected = {"dream", "reflection", "postmortem", "skills_auto",
                    "prm_train", "router_train", "calibration",
                    "open_questions", "self_play"}
        missing = expected - recorded
        assert not missing, f"phases without an activity record: {missing}"

    def test_project_advancer_pushes_needs_user(self):
        src = (_SRC / "core" / "project_advancer.py").read_text()
        assert src.count("_record_needs_user_activity(") >= 4  # def + 3 sites

    def test_slack_bot_has_notification_poller(self):
        bot = (Path(__file__).resolve().parents[1] / "interface" /
               "externals" / "slack_bot" / "main.py").read_text()
        assert "notification_poller" in bot
        assert "/api/notifications/pending" in bot
        assert "/api/notifications/ack" in bot
