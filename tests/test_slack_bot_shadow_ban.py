"""Shadow ban for the Slack bot (operator-requested, 2026-08-16).

A shadow-banned user's events are dropped as if the bot never saw them:
no reply, no error, nothing that distinguishes being banned from the bot
being idle. That silence is NOT new behaviour — `is_authorized_message`
already answers nothing on rejection, deliberately, because a reply
"confirms the bot exists and invites probing". A ban routes one user down
the path that already exists.

STATIC by choice: the denylist is read once at startup, so a ban takes a
restart. SILENT by choice: attempts log at DEBUG, keeping them out of the
operator's live stream (WARNING+) while staying recoverable.

⚠ THE PART THAT IS NOT OBVIOUS. There are TWO paths a banned user can
reach, not one. `handle_reaction` turns 👍/👎 into human outcome labels
via `/api/feedback`, and those labels FEED LEARNING. A message-only ban
would leave a banned user able to thumb their own earlier replies and
poison the training signal while appearing ignored — cosmetic, not a ban.
Both gates are pinned here, and the reaction test is the one that matters.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../src')))
pytest.importorskip("slack_bolt")

_BOT_PATH = (Path(__file__).resolve().parents[1] / "interface" / "externals"
             / "slack_bot" / "main.py")

OWNER = "UOWNER123"
BANNED = "UBANNED01"
BANNED2 = "UBANNED02"
NORMAL = "UNORMAL99"


def _load(monkeypatch, banned="", name="slackbot_sb"):
    """Load the bot module with a given denylist in the environment.

    Loaded fresh per case because the denylist is STATIC — read once at
    import, which is the behaviour under test. Reusing a cached module
    would test a constant nobody set.
    """
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test-not-real")
    monkeypatch.setenv("GHOST_API_KEY", "test-key")
    monkeypatch.setenv("GHOST_SLACK_LOG_FILE", "")
    monkeypatch.setenv("GHOST_SLACK_SHADOW_BAN", banned)
    spec = importlib.util.spec_from_file_location(name, _BOT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _msg(user, channel="C1234567", text="hello"):
    return {"user": user, "channel": channel, "text": text, "ts": "1.0"}


class TestTheDenylistParses:
    def test_comma_and_space_separated_both_work(self, monkeypatch):
        m = _load(monkeypatch, f"{BANNED}, {BANNED2}", "sb_parse")
        assert m.SHADOW_BANNED == frozenset({BANNED, BANNED2})

    def test_unset_means_nobody_is_banned(self, monkeypatch):
        m = _load(monkeypatch, "", "sb_empty")
        assert m.SHADOW_BANNED == frozenset()
        assert m.is_shadow_banned(NORMAL, OWNER) is False


class TestTheOwnerCannotBeLockedOut:
    def test_the_owner_is_exempt_even_when_listed(self, monkeypatch):
        """A typo in the env var must not cost the operator their own bot."""
        m = _load(monkeypatch, f"{OWNER} {BANNED}", "sb_owner")
        assert m.is_shadow_banned(OWNER, OWNER) is False
        assert m.is_shadow_banned(BANNED, OWNER) is True

    def test_the_owner_still_passes_the_real_gate_when_listed(self, monkeypatch):
        m = _load(monkeypatch, OWNER, "sb_owner2")
        assert m.is_authorized_message(_msg(OWNER), OWNER, False) is True


class TestMessagesAndMentions:
    def test_a_banned_user_is_rejected_in_OPEN_channel_mode(self, monkeypatch):
        """Open-channel mode is where a stranger can reach the bot at all,
        so it is where a ban has to bite."""
        m = _load(monkeypatch, BANNED, "sb_open")
        assert m.is_authorized_message(_msg(NORMAL), OWNER, True) is True
        assert m.is_authorized_message(_msg(BANNED), OWNER, True) is False

    def test_a_banned_user_is_rejected_in_OWNER_LOCK_mode_too(self, monkeypatch):
        m = _load(monkeypatch, BANNED, "sb_locked")
        assert m.is_authorized_message(_msg(BANNED), OWNER, False) is False

    def test_an_unbanned_stranger_is_unaffected(self, monkeypatch):
        """The ban must be surgical — it is not a second owner lock."""
        m = _load(monkeypatch, BANNED, "sb_surgical")
        assert m.is_authorized_message(_msg(NORMAL), OWNER, True) is True


class TestTheReactionPathIsCoveredToo:
    """THE test that matters. `handle_reaction` writes human outcome
    labels via /api/feedback, so a message-only ban would leave a banned
    user able to poison the training signal while appearing ignored."""

    def test_a_banned_user_writes_no_feedback_label(self, monkeypatch):
        import asyncio
        m = _load(monkeypatch, BANNED, "sb_react")
        m.OWNER_ID = OWNER

        called = {"n": 0}

        async def _boom(*a, **k):               # the label writer
            called["n"] += 1
            raise AssertionError("a shadow-banned user wrote a label")

        # Index a real bot reply so the reaction WOULD otherwise resolve:
        # the banned user is the requester, i.e. normally entitled to
        # label this very turn. Without that, the test would pass on the
        # pre-existing non-party check and prove nothing about the ban.
        m.register_reply("C1234567", "9.9", req_id="r1", requester=BANNED)
        monkeypatch.setattr(m, "post_feedback", _boom, raising=False)

        asyncio.run(m.handle_reaction({
            "user": BANNED, "reaction": "+1",
            "item": {"type": "message", "channel": "C1234567", "ts": "9.9"},
        }))
        assert called["n"] == 0

    def test_an_unbanned_requester_still_labels(self, monkeypatch):
        """Guards the mirror: the ban must not break normal feedback."""
        import asyncio
        m = _load(monkeypatch, BANNED, "sb_react2")
        m.OWNER_ID = OWNER
        seen = {"n": 0}
        m.register_reply("C1234567", "8.8", req_id="r2", requester=NORMAL)

        async def _ok(*a, **k):
            seen["n"] += 1
            return 200

        monkeypatch.setattr(m, "post_feedback", _ok, raising=False)
        asyncio.run(m.handle_reaction({
            "user": NORMAL, "reaction": "+1",
            "item": {"type": "message", "channel": "C1234567", "ts": "8.8"},
        }))
        assert seen["n"] == 1


class TestItIsSilent:
    def test_nothing_above_DEBUG_is_emitted_for_a_banned_attempt(
            self, monkeypatch, caplog):
        """Silent to the BANNED user and quiet for the operator: the live
        pretty stream is WARNING+, so a ban must not appear there."""
        import logging
        m = _load(monkeypatch, BANNED, "sb_quiet")
        with caplog.at_level(logging.INFO):
            m.is_authorized_message(_msg(BANNED), OWNER, True)
        assert not [r for r in caplog.records if r.levelno >= logging.INFO]

    def test_the_attempt_is_still_recoverable_at_DEBUG(
            self, monkeypatch, caplog):
        import logging
        m = _load(monkeypatch, BANNED, "sb_debug")
        with caplog.at_level(logging.DEBUG):
            m.is_authorized_message(_msg(BANNED), OWNER, True)
        assert any("shadow-banned" in r.getMessage() for r in caplog.records)
