"""Slack bot maintenance mode (2026-11-01, operator): while the maintenance
file exists, nothing is sent to the agent — mentions, DMs (the owner's too),
uploads, reaction feedback — and each message gets the maintenance reply. The
owner toggles it from a DM (`maintenance on[: text] | off | status`), answered
by the bot. The notification poller is untouched.

Loaded via importlib like test_slack_bot_owner_lock.py, under its own module
name so the files never share globals."""
import asyncio
import importlib.util
import os
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
    mp = pytest.MonkeyPatch()
    mp.setenv("SLACK_BOT_TOKEN", "xoxb-test-not-real")
    mp.setenv("GHOST_API_KEY", "test-key")
    mp.setenv("GHOST_SLACKBOT_LOG", "")
    mp.setenv("GHOST_SLACK_REPLY_INDEX", "")
    spec = importlib.util.spec_from_file_location("ghost_slack_bot_maintenance_test", _BOT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    yield mod
    mp.undo()


@pytest.fixture(autouse=True)
def _setup(bot, monkeypatch, tmp_path):
    bot._SEEN_MESSAGES.clear()
    monkeypatch.setattr(bot, "OWNER_ID", OWNER)
    monkeypatch.setattr(bot, "BOT_USER_ID", BOT)
    monkeypatch.setattr(bot, "OPEN_CHANNEL", True)
    monkeypatch.setattr(bot, "BOT_TEAM_IDS", {"T1"})
    monkeypatch.setattr(bot, "MAINTENANCE_FILE", str(tmp_path / "sys" / "maint"))
    # every path that reaches the agent (or Slack's API on the agent's behalf)
    agent = {"process": AsyncMock(), "context": AsyncMock(return_value=[{"role": "user", "content": "hi"}]),
             "feedback": AsyncMock(return_value=200)}
    monkeypatch.setattr(bot, "_process_message", agent["process"])
    monkeypatch.setattr(bot, "build_thread_context", agent["context"])
    monkeypatch.setattr(bot, "post_feedback", agent["feedback"])
    bot._agent_mocks = agent
    yield
    bot._SEEN_MESSAGES.clear()


def _run(coro):
    return asyncio.run(coro)


def _dm(bot, text, ts="1.0", files=None, user=OWNER):
    say = AsyncMock()
    ev = {"user": user, "text": text, "channel": "D1", "channel_type": "im", "ts": ts}
    if files:
        ev["files"] = files
    _run(bot.handle_direct_message(ev, say))
    return say


def _mention(bot, text="hi", user=MEMBER, ts="2.0"):
    say = AsyncMock()
    _run(bot.handle_mention({"user": user, "text": f"<@{BOT}> {text}", "channel": "C1",
                             "ts": ts, "team": "T1"}, say))
    return say


def _said(say):
    return [c.kwargs.get("text") for c in say.await_args_list]


def _agent_untouched(bot):
    m = bot._agent_mocks
    return not (m["process"].await_count or m["context"].await_count or m["feedback"].await_count)


# ══ the switch ═══════════════════════════════════════════════════════════

def test_off_when_the_file_is_absent(bot):
    assert bot.maintenance_text() is None


def test_an_empty_file_means_on_with_the_default_text(bot):
    bot.set_maintenance(True)
    assert bot.maintenance_text() == bot.MAINTENANCE_DEFAULT_TEXT


def test_the_files_text_is_the_reply_and_off_removes_it(bot):
    bot.set_maintenance(True, "  Upgrading the model, back at 20:00  ")
    assert bot.maintenance_text() == "Upgrading the model, back at 20:00"
    bot.set_maintenance(False)
    assert bot.maintenance_text() is None and not os.path.exists(bot.MAINTENANCE_FILE)
    bot.set_maintenance(False)      # already off: no error


def test_a_file_written_by_hand_turns_it_on_without_a_restart(bot):
    os.makedirs(os.path.dirname(bot.MAINTENANCE_FILE))
    Path(bot.MAINTENANCE_FILE).write_text("back soon")
    assert bot.maintenance_text() == "back soon"


def test_an_unreadable_file_still_means_on(bot):
    os.makedirs(bot.MAINTENANCE_FILE)          # a directory: exists, cannot be read
    assert bot.maintenance_text() == bot.MAINTENANCE_DEFAULT_TEXT


def test_an_empty_path_disables_the_mode(bot, monkeypatch):
    monkeypatch.setattr(bot, "MAINTENANCE_FILE", "")
    assert bot.maintenance_text() is None
    with pytest.raises(OSError, match="disabled"):
        bot.set_maintenance(True)


def test_the_write_is_atomic_and_leaves_no_temp_file(bot):
    bot.set_maintenance(True, "x")
    assert os.listdir(os.path.dirname(bot.MAINTENANCE_FILE)) == ["maint"]


def test_long_text_is_capped(bot):
    bot.set_maintenance(True, "a" * 5000)
    assert len(bot.maintenance_text()) == bot._MAINTENANCE_TEXT_MAX
    # the file itself is capped too, not only the read
    assert len(Path(bot.MAINTENANCE_FILE).read_text()) == bot._MAINTENANCE_TEXT_MAX


# ══ nothing reaches the agent ════════════════════════════════════════════

def test_a_members_mention_gets_the_reply_in_its_thread(bot):
    bot.set_maintenance(True, "back at 20:00")
    say = _mention(bot)
    assert _said(say) == ["back at 20:00"]
    assert say.await_args.kwargs.get("thread_ts") == "2.0"
    assert _agent_untouched(bot)


def test_the_owners_dm_gets_the_reply_too(bot):
    """Operator: maintenance applies to the owner as well."""
    bot.set_maintenance(True)
    say = _dm(bot, "what's the weather?")
    assert _said(say) == [bot.MAINTENANCE_DEFAULT_TEXT]
    assert _agent_untouched(bot)


def test_a_dm_with_a_file_is_not_uploaded(bot):
    bot.set_maintenance(True)
    say = _dm(bot, "read this", files=[{"id": "F1", "name": "a.pdf"}])
    assert _said(say) == [bot.MAINTENANCE_DEFAULT_TEXT]
    assert _agent_untouched(bot)


def test_a_reaction_label_is_not_sent(bot):
    bot.register_reply("C1", "9.9", "req42", MEMBER)
    bot.set_maintenance(True)
    _run(bot.handle_reaction({"user": OWNER, "reaction": "+1",
                              "item": {"type": "message", "channel": "C1", "ts": "9.9"}}))
    assert _agent_untouched(bot)


def test_with_maintenance_off_everything_flows_as_before(bot):
    """Control: the same calls reach the agent when the file is absent."""
    bot.register_reply("C1", "9.9", "req42", MEMBER)
    _mention(bot)
    _dm(bot, "what's the weather?", ts="3.0")
    _run(bot.handle_reaction({"user": OWNER, "reaction": "+1",
                              "item": {"type": "message", "channel": "C1", "ts": "9.9"}}))
    m = bot._agent_mocks
    assert m["process"].await_count == 2 and m["feedback"].await_count == 1


def test_a_stranger_still_gets_silence(bot, monkeypatch):
    """The gates run first: maintenance never answers someone the bot ignores."""
    monkeypatch.setattr(bot, "OPEN_CHANNEL", False)
    bot.set_maintenance(True)
    assert _said(_mention(bot, user="UEVIL9999")) == []
    assert _said(_dm(bot, "hi", user="UEVIL9999")) == []


def test_the_reply_cannot_ping_anyone(bot):
    bot.set_maintenance(True, "<!channel> down <@U123>")
    (text,) = _said(_mention(bot))
    assert "<!channel>" not in text and "<@U123>" not in text and "&lt;!channel&gt;" in text


# ══ the owner's DM commands ══════════════════════════════════════════════

@pytest.mark.parametrize("text,want", [
    ("maintenance on", ("on", "")),
    ("Maintenance ON", ("on", "")),
    ("maintenance on: upgrading, back at 20:00", ("on", "upgrading, back at 20:00")),
    ("maintenance on: back at 5!", ("on", "back at 5!")),       # r2: the text keeps its punctuation
    ("maintenance on: is it up?", ("on", "is it up?")),
    ("maintenance on - back soon", ("on", "back soon")),
    ("maintenance on — back soon", ("on", "back soon")),
    ("maintenance off", ("off", "")),
    ("  maintenance off.  ", ("off", "")),
    ("maintenance status", ("status", "")),
    ("maintenance status?", ("status", "")),
    ("maintenance off!!", ("off", "")),
    ("*maintenance off*", ("off", "")),
    ("`maintenance off`", ("off", "")),
    ("maintenance on:", ("on", "")),
    ("maintenance on my car is due next week", None),    # an ordinary request
    # r1 MAJOR: hyphenated words after "on" are ordinary requests
    ("maintenance on-call rota for next week", None),
    ("maintenance on-prem vs cloud?", None),
    ("Maintenance on—the boiler, any tips?", None),
    ("*maintenance on* the boiler", None),
    # r2: one line only — a note headed "maintenance on" with a list under it
    ("maintenance on\n- buy milk\n- call bob", None),
    ("maintenance on\n-\nhelp me fix the car", None),
    ("maintenance on\n: x", None),
    ("maintenance on\ncheck the weather", None),
    ("maintenance status\nalso x", None),
    # r2: only status takes a "?" — "maintenance on?" is a question
    ("maintenance on?", None),
    ("maintenance off?", None),
    ("maintenance on -", None),
    # battery: each guard of the step parser, in the world where it matters
    ("maintenance on- the boiler", None),
    ("maintenance on -call", None),
    ("maintenance on: back soon\nand also fix the printer", None),
    ("maintenanceoff", None),
    ("maintenanceon", None),
    ("maintenance ok", None),
    ("maintenance of: x", None),
    ("*maintenance on: back soon", None),
    # r3
    ("maintenance on :)", None),
    ("maintenance on : x", None),
    ("maintenance on: x\u2028y", None),
    ("maintenance on: x\x0by", None),
    ("`maintenance on: x` and also `y`", None),
    ("_maintenance on: see foo_bar_", None),
    ("maintenance on:back soon", ("on", "back soon")),
    ("maintenance\ton: tab ok", ("on", "tab ok")),
    ("is maintenance on?", None),
    ("maintenance", None),
    ("maintenance off and then check the weather", None),
    ("", None),
])
def test_the_command_is_the_whole_message(bot, text, want):
    assert bot.parse_maintenance_command(text) == want


def test_on_then_off_from_a_dm(bot):
    say = _dm(bot, "maintenance on: back at 20:00")
    assert bot.maintenance_text() == "back at 20:00" and "ON" in _said(say)[0]
    say = _dm(bot, "maintenance off", ts="1.1")
    assert bot.maintenance_text() is None and "OFF" in _said(say)[0]
    assert _agent_untouched(bot)


def test_off_works_while_maintenance_is_on(bot):
    bot.set_maintenance(True)
    _dm(bot, "maintenance off")
    assert bot.maintenance_text() is None


def test_status_reports_both_states(bot):
    assert "OFF" in _said(_dm(bot, "maintenance status"))[0]
    bot.set_maintenance(True, "back soon")
    (text,) = _said(_dm(bot, "maintenance status", ts="1.1"))
    assert "ON" in text and "back soon" in text
    assert _agent_untouched(bot)


def test_a_member_cannot_run_the_command(bot):
    """DMs are owner-only, and a channel mention is never parsed as a command."""
    assert _said(_dm(bot, "maintenance on", user=MEMBER)) == []
    _mention(bot, text="maintenance on")
    assert bot.maintenance_text() is None


def test_a_dm_with_a_file_is_never_a_command(bot):
    _dm(bot, "maintenance on", files=[{"id": "F1", "name": "a.pdf"}])
    assert bot.maintenance_text() is None


def test_a_write_failure_is_told_to_the_owner(bot, monkeypatch):
    monkeypatch.setattr(bot, "MAINTENANCE_FILE", "")
    (text,) = _said(_dm(bot, "maintenance on"))
    assert "Could not turn maintenance on" in text


def test_a_failed_write_leaves_no_temp_file(bot, monkeypatch):
    """r1: a write that raises must not leave `….tmp-<pid>` behind."""
    os.makedirs(os.path.dirname(bot.MAINTENANCE_FILE))
    real_replace = os.replace

    def boom(a, b):
        raise OSError("disk full")
    monkeypatch.setattr(bot.os, "replace", boom)
    with pytest.raises(OSError):
        bot.set_maintenance(True, "x")
    monkeypatch.setattr(bot.os, "replace", real_replace)
    assert os.listdir(os.path.dirname(bot.MAINTENANCE_FILE)) == []


def test_the_start_flag_writes_removes_or_leaves_the_file(bot):
    bot.apply_maintenance_flag("on")
    assert bot.maintenance_text() == bot.MAINTENANCE_DEFAULT_TEXT
    bot.apply_maintenance_flag(None)          # omitted: a restart never ends it
    assert bot.maintenance_text() is not None
    bot.apply_maintenance_flag("off")
    assert bot.maintenance_text() is None
    bot.apply_maintenance_flag(None)          # …nor starts it
    assert bot.maintenance_text() is None


def test_a_long_hand_written_file_is_capped_on_read(bot):
    """The file can be written by hand (echo …), past set_maintenance's cap."""
    os.makedirs(os.path.dirname(bot.MAINTENANCE_FILE))
    Path(bot.MAINTENANCE_FILE).write_text("b" * 5000)
    assert len(bot.maintenance_text()) == bot._MAINTENANCE_TEXT_MAX


def test_a_huge_message_parses_in_linear_time(bot):
    """r2 MAJOR: the old regex took 51 s on 4k chars of padding, inside the
    event loop. 100k chars must take milliseconds."""
    import time
    for text in ("maintenance on: a" + " " * 100_000 + "b",
                 "maintenance on - a" + " " * 100_000 + "b",
                 "maintenance " + " " * 100_000 + "on",
                 "maintenance on" + "." * 100_000 + "x"):
        t0 = time.perf_counter()
        bot.parse_maintenance_command(text)
        assert time.perf_counter() - t0 < 0.5


def test_a_lone_surrogate_is_written_not_raised(bot):
    """r2: the write must not raise past the OSError handling (no reply, temp left)."""
    bot.set_maintenance(True, "x\ud800y")
    assert bot.maintenance_text().startswith("x") and os.listdir(os.path.dirname(bot.MAINTENANCE_FILE)) == ["maint"]
