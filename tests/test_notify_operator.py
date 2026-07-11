"""Tests for the notify_operator tool (2026-07-11) — the deliberate
agent→operator push that makes "…and report back in Slack" actually work."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_agent.core.autonomous_activity import (
    ActivityLog, SEVERITY_NOTIFY, save_consumer_offset,
)
from ghost_agent.tools import notify_tool
from ghost_agent.tools.notify_tool import tool_notify_operator, PHASE


@pytest.fixture(autouse=True)
def _reset_rate_limit():
    notify_tool._sent_timestamps.clear()
    yield
    notify_tool._sent_timestamps.clear()


def _ctx(tmp_path, notifier=None):
    memory_dir = tmp_path / "system" / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        memory_dir=memory_dir,
        activity_log=ActivityLog(memory_dir.parent / "act.jsonl"),
        outbound_notifier=notifier,
    )


class TestNotifyOperator:
    def test_writes_notify_record(self, tmp_path):
        ctx = _ctx(tmp_path)
        out = asyncio.run(tool_notify_operator(
            message="GAIA run finished: 121/165", context=ctx))
        assert "Notification queued" in out
        recs, _ = ctx.activity_log.read_since(0)
        assert len(recs) == 1
        assert recs[0].phase == PHASE
        assert recs[0].severity == SEVERITY_NOTIFY
        assert "GAIA run finished" in recs[0].summary

    def test_requires_message(self, tmp_path):
        out = asyncio.run(tool_notify_operator(context=_ctx(tmp_path)))
        assert "Error" in out and "required" in out

    def test_arg_healing(self, tmp_path):
        ctx = _ctx(tmp_path)
        out = asyncio.run(tool_notify_operator(text="healed arg", context=ctx))
        assert "Notification queued" in out
        recs, _ = ctx.activity_log.read_since(0)
        assert recs[0].summary == "healed arg"

    def test_message_clamped(self, tmp_path):
        ctx = _ctx(tmp_path)
        asyncio.run(tool_notify_operator(message="x" * 2000, context=ctx))
        recs, _ = ctx.activity_log.read_since(0)
        assert len(recs[0].summary) <= 500

    def test_no_ledger_graceful_error(self):
        ctx = SimpleNamespace(memory_dir=None, activity_log=None,
                              outbound_notifier=None)
        out = asyncio.run(tool_notify_operator(message="hi", context=ctx))
        assert "Error" in out and "unavailable" in out

    def test_fires_push_callback(self, tmp_path):
        seen = []
        ctx = _ctx(tmp_path)
        ctx.activity_log.on_notify = seen.append
        asyncio.run(tool_notify_operator(message="ping", context=ctx))
        assert len(seen) == 1 and seen[0].summary == "ping"

    def test_rate_limit(self, tmp_path):
        ctx = _ctx(tmp_path)
        for i in range(notify_tool._MAX_PER_HOUR):
            out = asyncio.run(tool_notify_operator(
                message=f"m{i}", context=ctx))
            assert "Notification queued" in out
        out = asyncio.run(tool_notify_operator(message="one too many",
                                               context=ctx))
        assert "rate limit" in out and "NOT sent" in out
        recs, _ = ctx.activity_log.read_since(0)
        assert len(recs) == notify_tool._MAX_PER_HOUR  # 13th not recorded


class TestDeliveryChannelHonesty:
    def test_reports_digest_only_when_nothing_configured(self, tmp_path):
        out = asyncio.run(tool_notify_operator(
            message="hi", context=_ctx(tmp_path)))
        assert "next-turn digest" in out
        assert "Slack" not in out and "webhook" not in out

    def test_reports_push_when_notifier_configured(self, tmp_path):
        notifier = SimpleNamespace(configured=True, send_soon=lambda r: None)
        out = asyncio.run(tool_notify_operator(
            message="hi", context=_ctx(tmp_path, notifier=notifier)))
        assert "push (webhook/ntfy)" in out

    def test_reports_slack_only_after_consumer_has_polled(self, tmp_path):
        ctx = _ctx(tmp_path)
        save_consumer_offset(
            Path(str(ctx.memory_dir)).parent / "notify_consumers.json",
            "slack", 0)
        out = asyncio.run(tool_notify_operator(message="hi", context=ctx))
        assert "Slack DM" in out


class TestWiring:
    def test_advertised_and_dispatchable(self):
        from ghost_agent.tools.registry import TOOL_DEFINITIONS
        names = [t["function"]["name"] for t in TOOL_DEFINITIONS]
        assert "notify_operator" in names
        src = (Path(__file__).resolve().parents[1] / "src" / "ghost_agent"
               / "tools" / "registry.py").read_text()
        assert '"notify_operator": lambda' in src

    def test_subagents_cannot_page_the_operator(self):
        # Delegated sub-agents must not reach this tool — the MAIN agent
        # reports; a delegate paging the operator directly is spam surface.
        from ghost_agent.core.subagent import (
            resolve_allowed_tools, DEFAULT_ALLOWED_TOOLS,
        )
        assert "notify_operator" not in DEFAULT_ALLOWED_TOOLS
        assert "notify_operator" not in resolve_allowed_tools(
            ["notify_operator", "web_search"])

    def test_slack_bot_has_emoji_for_agent_message(self):
        bot = (Path(__file__).resolve().parents[1] / "interface" /
               "externals" / "slack_bot" / "main.py").read_text()
        assert '"agent_message"' in bot

    def test_digest_renders_agent_message_phase(self):
        from ghost_agent.core.autonomous_activity import (
            ActivityRecord, render_activity_digest,
        )
        import time as _t
        out = render_activity_digest([ActivityRecord(
            ts=_t.time(), phase=PHASE, summary="report text",
            severity=SEVERITY_NOTIFY)])
        assert "message from the agent" in out and "report text" in out


class TestNoSameTurnEcho:
    """Regression from the first live test: the digest banner echoed the
    notification the SAME reply had just sent ("Background activity while
    you were away: [message from the agent] …" on top of the turn that
    sent it)."""

    def test_record_stamped_with_request_id(self, tmp_path):
        from ghost_agent.utils.logging import request_id_context
        ctx = _ctx(tmp_path)
        token = request_id_context.set("reqABC12")
        try:
            asyncio.run(tool_notify_operator(message="hi", context=ctx))
        finally:
            request_id_context.reset(token)
        recs, _ = ctx.activity_log.read_since(0)
        assert recs[0].meta.get("req_id") == "reqABC12"

    def test_system_context_not_stamped(self, tmp_path):
        # Default contextvar value is "SYSTEM" — must not stamp that.
        ctx = _ctx(tmp_path)
        asyncio.run(tool_notify_operator(message="hi", context=ctx))
        recs, _ = ctx.activity_log.read_since(0)
        assert "req_id" not in recs[0].meta

    def test_digest_skips_current_turns_records(self, tmp_path):
        from ghost_agent.core.autonomous_activity import (
            ActivityRecord, render_activity_digest,
        )
        import time as _t
        mine = ActivityRecord(ts=_t.time(), phase=PHASE, summary="my own",
                              severity=SEVERITY_NOTIFY,
                              meta={"req_id": "reqABC12"})
        older = ActivityRecord(ts=_t.time(), phase=PHASE, summary="earlier",
                               severity=SEVERITY_NOTIFY,
                               meta={"req_id": "reqOLD99"})
        out = render_activity_digest([mine, older],
                                     current_req_id="reqABC12")
        assert "my own" not in out       # no same-turn echo
        assert "earlier" in out          # other turns still surface

    def test_finalize_passes_current_req_id(self):
        src = (Path(__file__).resolve().parents[1] / "src" / "ghost_agent"
               / "core" / "agent.py").read_text()
        assert "current_req_id=str(fs.req_id" in src
