"""One Slack message = one agent turn (2026-10-20).

Live 2026-10-19 23:10:15: a member's "@Ghost dress this lady … as a witch"
with a photo reached ``handle_mention`` twice in the same instant — two
downloads, two uploads 4 ms apart, two full turns, two pictures. The bot now
claims each message (channel + ts) before its first await and logs one INFO
line per event it acts on.
"""

import asyncio
import importlib.util
import logging
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("slack_bolt")

_BOT_PATH = (Path(__file__).resolve().parents[1] / "interface" / "externals"
             / "slack_bot" / "main.py")
OWNER = "UOWNER123"
MEMBER = "UMEMBER77"
BOT = "UBOTBOT12"


@pytest.fixture(scope="module")
def bot():
    # same env discipline as test_slack_bot_owner_lock.py: assigned, then restored
    mp = pytest.MonkeyPatch()
    mp.setenv("SLACK_BOT_TOKEN", "xoxb-test-not-real")
    mp.setenv("GHOST_API_KEY", "test-key")
    mp.setenv("GHOST_SLACKBOT_LOG", "")
    mp.setenv("GHOST_SLACK_REPLY_INDEX", "")
    spec = importlib.util.spec_from_file_location("ghost_slack_bot_dedup_under_test", _BOT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    yield mod
    mp.undo()


@pytest.fixture(autouse=True)
def _setup(bot, monkeypatch):
    monkeypatch.setattr(bot, "OWNER_ID", OWNER)
    monkeypatch.setattr(bot, "BOT_USER_ID", BOT)
    monkeypatch.setattr(bot, "MAINTENANCE_FILE", "")
    monkeypatch.setattr(bot, "OPEN_CHANNEL", True)
    monkeypatch.setattr(bot, "_is_open_surface", lambda ch: True)
    bot._SEEN_MESSAGES.clear()
    yield
    bot._SEEN_MESSAGES.clear()


def _mention(ts="1792440607.383519", user=MEMBER, channel="C56FKAHTP", files=None):
    ev = {"type": "app_mention", "user": user, "channel": channel, "ts": ts,
          "thread_ts": "1792440277.327819",
          "text": f"<@{BOT}> dress this lady in the image i am attaching as a witch"}
    if files is not None:
        ev["files"] = files
    return ev


def _slow_proc():
    """A _process_message that takes a while — as the real one does
    (download, upload, a 2-minute turn), so concurrent copies overlap."""
    async def _p(*a, **k):
        await asyncio.sleep(0.05)
    return AsyncMock(side_effect=_p)


def test_the_live_double_delivery_runs_one_turn(bot, monkeypatch):
    proc = _slow_proc()
    monkeypatch.setattr(bot, "_process_message", proc)
    # the thread fetch awaits, so the two copies interleave
    monkeypatch.setattr(bot, "build_thread_context", AsyncMock(return_value=[]))
    ev = _mention(files=[{"id": "F1", "name": "17924402650935945165133712975707.jpg"}])

    async def both():
        await asyncio.gather(bot.handle_mention(dict(ev), AsyncMock(), body={"event_id": "Ev1"}),
                             bot.handle_mention(dict(ev), AsyncMock(), body={"event_id": "Ev2"}))
    asyncio.run(both())
    assert proc.await_count == 1


def test_a_redelivery_seconds_later_is_dropped(bot, monkeypatch):
    proc = AsyncMock()
    monkeypatch.setattr(bot, "_process_message", proc)
    monkeypatch.setattr(bot, "build_thread_context", AsyncMock(return_value=[]))
    asyncio.run(bot.handle_mention(_mention(), AsyncMock(), body={"event_id": "Ev1"}))
    asyncio.run(bot.handle_mention(_mention(), AsyncMock(), body={"event_id": "Ev1"}))
    assert proc.await_count == 1


def test_different_messages_both_run(bot, monkeypatch):
    proc = AsyncMock()
    monkeypatch.setattr(bot, "_process_message", proc)
    monkeypatch.setattr(bot, "build_thread_context", AsyncMock(return_value=[]))
    asyncio.run(bot.handle_mention(_mention(ts="100.1"), AsyncMock()))
    asyncio.run(bot.handle_mention(_mention(ts="100.2"), AsyncMock()))
    # the same ts in ANOTHER channel is another message
    asyncio.run(bot.handle_mention(_mention(ts="100.1", channel="C999"), AsyncMock()))
    assert proc.await_count == 3


def test_a_duplicate_dm_runs_one_turn(bot, monkeypatch):
    proc = AsyncMock()
    monkeypatch.setattr(bot, "_process_message", proc)
    monkeypatch.setattr(bot, "build_thread_context", AsyncMock(return_value=[]))
    dm = {"user": OWNER, "text": "do the thing", "channel": "D1", "channel_type": "im", "ts": "5.5"}
    asyncio.run(bot.handle_direct_message(dict(dm), AsyncMock(), body={"event_id": "Ev9"}))
    asyncio.run(bot.handle_direct_message(dict(dm), AsyncMock(), body={"event_id": "Ev9"}))
    assert proc.await_count == 1


def test_a_dropped_stranger_does_not_claim_the_message(bot, monkeypatch):
    """The gate runs first: a rejected event never claims the ts."""
    monkeypatch.setattr(bot, "_process_message", AsyncMock())
    monkeypatch.setattr(bot, "_is_open_surface", lambda ch: False)
    asyncio.run(bot.handle_mention(_mention(user="UEVIL"), AsyncMock()))
    assert bot._SEEN_MESSAGES == {}


def test_claim_ttl_and_missing_fields(bot):
    assert bot.claim_message("C1", "1.0", "E1", now=1000.0)
    assert not bot.claim_message("C1", "1.0", "E2", now=1000.0 + bot._SEEN_TTL_S - 1)
    assert bot.claim_message("C1", "1.0", "E3", now=1000.0 + bot._SEEN_TTL_S)
    # no channel or no ts: never deduplicated
    assert bot.claim_message("C1", None) and bot.claim_message("C1", None)
    assert bot.claim_message(None, "2.0") and bot.claim_message(None, "2.0")
    assert bot.claim_message("C1", "") and bot.claim_message("C1", "")


def test_the_memory_is_bounded(bot, monkeypatch):
    monkeypatch.setattr(bot, "_SEEN_MESSAGES_MAX", 3)
    for i in range(10):
        assert bot.claim_message("C1", f"{i}.0", now=50.0)
    assert len(bot._SEEN_MESSAGES) == 3
    # expired entries are pruned on the next claim
    bot.claim_message("C1", "x", now=50.0 + bot._SEEN_TTL_S)
    assert list(bot._SEEN_MESSAGES) == ["C1:x"]


def test_each_event_is_logged_and_a_duplicate_says_which_kind(bot, monkeypatch, caplog):
    monkeypatch.setattr(bot, "_process_message", AsyncMock())
    monkeypatch.setattr(bot, "build_thread_context", AsyncMock(return_value=[]))
    caplog.set_level(logging.INFO, logger=bot.logger.name)
    body = {"event_id": "Ev1", "event_time": 1792440607}
    asyncio.run(bot.handle_mention(_mention(files=[{"id": "F1"}]), AsyncMock(), body=body))
    asyncio.run(bot.handle_mention(_mention(), AsyncMock(), body=body))
    asyncio.run(bot.handle_mention(_mention(), AsyncMock(), body={"event_id": "Ev2"}))
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "mention event channel=C56FKAHTP ts=1792440607.383519" in text
    assert "event_id=Ev1" in text and "files=1" in text
    assert "same event id — a redelivery" in text
    assert "first event id Ev1" in text


def test_bolt_hands_the_handlers_the_event_body(bot):
    """The event id comes from Bolt's ``body`` kwarg: Bolt injects by the
    listener's argument names, defaults included."""
    from slack_bolt.util.utils import get_arg_names_of_callable
    assert "body" in get_arg_names_of_callable(bot.handle_mention)
    assert "body" in get_arg_names_of_callable(bot.handle_direct_message)


def test_a_mention_inside_a_dm_runs_one_turn(bot, monkeypatch):
    """Slack fires app_mention AND message for "@Ghost …" in a DM; the
    mention handler leaves D-channels to the DM handler."""
    proc = AsyncMock()
    monkeypatch.setattr(bot, "_process_message", proc)
    monkeypatch.setattr(bot, "build_thread_context", AsyncMock(return_value=[]))
    ev = {"user": OWNER, "text": f"<@{BOT}> hi", "channel": "D1", "channel_type": "im", "ts": "7.7"}
    asyncio.run(bot.handle_mention(dict(ev), AsyncMock(), body={"event_id": "EvM"}))
    asyncio.run(bot.handle_direct_message(dict(ev), AsyncMock(), body={"event_id": "EvD"}))
    assert proc.await_count == 1
