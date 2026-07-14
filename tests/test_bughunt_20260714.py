"""Assorted bug-hunt fixes, 2026-07-14 review pass.

Covers the smaller confirmed fixes across cohorts:
- jobs swarm result_resolver (C2-3): collect returns content, not "True".
- notify egress consistency (C1-4): notify's LAN classification == the guard's.
- notify_tool rate limit (C1-6): a failed write doesn't burn a slot.
- games extract_move_text (C4-1): whitespace-only reply doesn't crash.
"""

import pytest
from unittest.mock import MagicMock

from ghost_agent.api.games.base import extract_move_text


# ---------------------------------------------------------- C4-1 move parse

class TestExtractMoveText:
    @pytest.mark.parametrize("reply", ["   ", "\n\n", "\t", " \n ", ""])
    def test_whitespace_or_empty_returns_none_no_crash(self, reply):
        assert extract_move_text(reply) is None  # was IndexError → HTTP 500

    def test_move_line_wins(self):
        assert extract_move_text("thinking...\nMOVE: e4") == "e4"

    def test_bare_short_token(self):
        assert extract_move_text("e4") == "e4"

    def test_long_prose_is_not_a_move(self):
        assert extract_move_text("I think the best move here is e4") is None


# ------------------------------------------------------- C1-4 egress parity

class TestNotifyEgressParity:
    def test_lan_suffixes_shared_with_guard(self):
        import ghost_agent.utils.notify as notify
        import ghost_agent.utils.egress_guard as guard
        # The whole point of the fix: one source of truth, no divergence.
        assert notify._LAN_SUFFIXES is guard._LOCAL_HOST_SUFFIXES
        assert notify._LAN_NAMES is guard._LOCAL_HOST_NAMES

    def test_classification_matches_guard_suffixes(self):
        from ghost_agent.utils.notify import url_needs_tor
        # A suffix the guard allows direct → notify must NOT route via Tor.
        assert url_needs_tor("http://relay.local/x") is False
        assert url_needs_tor("http://host.internal/x") is False
        # Public still needs Tor.
        assert url_needs_tor("http://example.com/x") is True
        # Loopback / Tailscale stay direct.
        assert url_needs_tor("http://127.0.0.1:8090/t") is False
        assert url_needs_tor("http://100.90.1.2:8090/t") is False


# --------------------------------------------------- C1-6 rate-limit commit

class TestNotifyRateLimit:
    def test_failed_write_does_not_consume_slot(self, monkeypatch):
        import ghost_agent.tools.notify_tool as nt
        nt._sent_timestamps.clear()
        # _rate_limited only CHECKS now.
        assert nt._rate_limited() is False
        assert nt._sent_timestamps == []   # nothing consumed by the check
        # Slots are committed explicitly, only on success.
        nt._note_sent()
        assert len(nt._sent_timestamps) == 1

    def test_budget_enforced_after_commits(self):
        import ghost_agent.tools.notify_tool as nt
        nt._sent_timestamps.clear()
        for _ in range(nt._MAX_PER_HOUR):
            assert nt._rate_limited() is False
            nt._note_sent()
        assert nt._rate_limited() is True   # budget now exhausted


# ---------------------------------------------------- C2-3 swarm job result

class TestJobResultResolver:
    async def test_resolver_overrides_bool_result(self):
        import asyncio
        from ghost_agent.core.jobs import JobRegistry

        reg = JobRegistry()

        async def worker():
            return True   # swarm workers return a bool; content is elsewhere

        job = reg.register("swarm", "fetch prices", output_key="prices")
        # Resolver maps the raw bool → the real payload (as swarm.py wires it).
        job.result_resolver = lambda _res: "AAPL 190.2\nMSFT 410.1"
        task = asyncio.create_task(worker())
        reg.attach(job.id, task)
        await task
        await asyncio.sleep(0)   # let the done-callback run

        collected = reg.get(job.id)
        assert collected.result == "AAPL 190.2\nMSFT 410.1"   # not "True"

    async def test_no_resolver_keeps_raw_result(self):
        import asyncio
        from ghost_agent.core.jobs import JobRegistry

        reg = JobRegistry()

        async def worker():
            return "plain answer"

        job = reg.register("subagent", "research")
        task = asyncio.create_task(worker())
        reg.attach(job.id, task)
        await task
        await asyncio.sleep(0)
        assert reg.get(job.id).result == "plain answer"

    async def test_resolver_none_falls_back_to_raw(self):
        import asyncio
        from ghost_agent.core.jobs import JobRegistry

        reg = JobRegistry()

        async def worker():
            return True

        job = reg.register("swarm", "x", output_key="k")
        job.result_resolver = lambda _res: None   # scratchpad empty
        task = asyncio.create_task(worker())
        reg.attach(job.id, task)
        await task
        await asyncio.sleep(0)
        assert reg.get(job.id).result == "True"   # graceful fallback
