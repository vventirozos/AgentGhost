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
import json
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
    # ⚠ LIVE-PATH ISOLATION. Two env vars, both load-bearing, and this file
    # originally set NEITHER correctly — it set "GHOST_SLACK_LOG_FILE",
    # a name main.py never reads.
    #
    # `GHOST_SLACK_REPLY_INDEX` unset means REPLY_INDEX_PATH defaults to
    # ~/Data/AI/Logs/ghost-slack-reply-index.json — the OPERATOR'S LIVE
    # INDEX. `register_reply` then calls `_save_reply_index()`, which
    # writes the whole in-memory dict; the module only ever LOADS the file
    # inside the bot's main(), so under pytest the in-memory index starts
    # EMPTY and the save TRUNCATES the live file to this test's fixtures.
    # It happened: the operator's index was reduced to one row
    # ("C1234567:8.8"), destroying the ability to label every bot reply
    # older than that run. Recorded labels in corrections.jsonl survived
    # (50 rows intact); the reply→request mapping did not, and the bot's
    # logs do not record channel:ts, so it is unreconstructible.
    #
    # `GHOST_SLACKBOT_LOG` unset attaches a RotatingFileHandler to the
    # live operator audit log (5MB rotation trigger).
    monkeypatch.setenv("GHOST_SLACK_REPLY_INDEX", "")
    monkeypatch.setenv("GHOST_SLACKBOT_LOG", "")
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


# ── 2026-08-17: the test suite destroyed the operator's live reply index ────

def test_the_LIVE_reply_index_is_never_touched_by_a_test(monkeypatch, tmp_path):
    """⚠ THE REGRESSION THIS FILE CAUSED.

    This suite imported the bot without setting GHOST_SLACK_REPLY_INDEX, so
    REPLY_INDEX_PATH defaulted to the operator's live
    ~/Data/AI/Logs/ghost-slack-reply-index.json. The module loads that file
    only inside main() — never at import — so under pytest the in-memory
    index started EMPTY and the first register_reply wrote the whole dict
    back, TRUNCATING the live file to this suite's two fixtures. Effect: the
    reply→request mapping for every older bot reply was destroyed, so 👍/👎
    on those turns silently produced no label. Recorded labels survived (50
    rows in corrections.jsonl); the mapping is unreconstructible because the
    bot logs no channel:ts.

    Three layers now: conftest autouse env isolation, this suite's own
    isolation, and the writer's refuse-to-replace-what-I-never-read guard.
    This test pins the LAST one, by simulating the exact leak.
    """
    live = tmp_path / "pretend-live-index.json"
    live.write_text(json.dumps({
        "C_REAL:1786899694.380169": {"req_id": "8f6145e8",
                                     "requester": "U55NMJT1N",
                                     "t": 1786899694.0},
        "C_REAL:1786911129.690559": {"req_id": "7c701369",
                                     "requester": "U56CVBHHQ",
                                     "t": 1786911129.0},
    }))
    before = live.read_text()

    m = _load(monkeypatch, "", "sb_live_index_guard")
    # Simulate the leak: the live path reaches the module WITHOUT a load.
    monkeypatch.setattr(m, "REPLY_INDEX_PATH", str(live))
    monkeypatch.setattr(m, "_REPLY_INDEX_LOADED_PATH", None)
    m._REPLY_INDEX.clear()

    m.register_reply("C1234567", "9.9", req_id="fixture", requester="UTEST")

    assert live.read_text() == before, (
        "a test truncated a pre-existing reply index — the live-data "
        "regression of 2026-08-17")
    # ...and the refusal is LOUD, not silent.
    assert m.lookup_reply("C1234567", "9.9") is not None, (
        "in-memory registration must still work; only the WRITE is refused")


def test_a_fresh_index_path_is_still_writable(monkeypatch, tmp_path):
    """The mirror: the guard must not break a first-run bot or a legitimate
    test that points at a new tmp file — writing a file that does not exist
    yet truncates nothing."""
    fresh = tmp_path / "new-index.json"
    m = _load(monkeypatch, "", "sb_fresh_index")
    monkeypatch.setattr(m, "REPLY_INDEX_PATH", str(fresh))
    monkeypatch.setattr(m, "_REPLY_INDEX_LOADED_PATH", None)
    m._REPLY_INDEX.clear()
    m.register_reply("C1", "1.0", req_id="reqA", requester="U1")
    assert fresh.exists() and "reqA" in fresh.read_text()


def test_a_fresh_path_can_be_written_TWICE(monkeypatch, tmp_path):
    """R3 m6: `os.replace` never recorded ownership, so the guard locked
    itself out after its own first write — "loaded a missing path, then
    refused every later save". Production never hit it (main() loads
    first), but the write guard's docstring promises fresh-tmp-path writes
    keep working, and no test registered twice."""
    idx = tmp_path / "fresh.json"
    m = _load(monkeypatch, "", "sb_write_twice")
    monkeypatch.setattr(m, "REPLY_INDEX_PATH", str(idx))
    monkeypatch.setattr(m, "_REPLY_INDEX_LOADED_PATH", None)
    m._REPLY_INDEX.clear()
    m.register_reply("C1", "1.1", req_id="a", requester="U1")
    m.register_reply("C1", "2.2", req_id="b", requester="U2")
    on_disk = json.loads(idx.read_text())
    assert "C1:1.1" in on_disk and "C1:2.2" in on_disk, (
        f"the second save was refused: {sorted(on_disk)}")


def test_the_conftest_isolates_both_live_side_files():
    """Structural: the autouse fixture must cover BOTH env names. This file
    originally set GHOST_SLACK_LOG_FILE — a name main.py never reads — so
    it also attached a RotatingFileHandler to the live operator audit log.

    ⚠ Asserted against the CONFTEST SOURCE, not against os.environ: a
    sibling suite (test_slack_bot_feedback) sets these with a bare
    `os.environ[...] =`, which leaks for the whole session, so an
    env-value assertion passes even with the fixture deleted. Verified —
    that mutation survived the first version of this test.
    """
    import ast
    from pathlib import Path
    src = (Path(__file__).resolve().parent / "conftest.py").read_text()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef)
              and n.name == "_isolate_live_side_files")
    dumped = ast.dump(fn)
    assert "GHOST_SLACK_REPLY_INDEX" in dumped
    assert "GHOST_SLACKBOT_LOG" in dumped
    assert "setenv" in dumped, "the fixture no longer sets anything"
    # ...and it must be autouse, or it protects only tests that ask.
    assert any("autouse" in ast.dump(d) for d in fn.decorator_list), \
        "the live-path isolation must be autouse"


def test_a_LOADED_existing_index_stays_writable(monkeypatch, tmp_path):
    """The mirror of the write guard, and the case the first version of
    these tests missed: the real bot LOADS an existing index at startup and
    must then be able to persist new replies. If `_load_reply_index` stops
    recording which path it read, every save in production is refused —
    silently disabling label attribution for the whole channel."""
    idx = tmp_path / "existing.json"
    idx.write_text(json.dumps({"C_OLD:1.0": {"req_id": "old",
                                             "requester": "U1", "t": 1.0}}))
    m = _load(monkeypatch, "", "sb_loaded_writable")
    monkeypatch.setattr(m, "REPLY_INDEX_PATH", str(idx))
    m._REPLY_INDEX.clear()
    m._load_reply_index()                       # the startup path
    m.register_reply("C_NEW", "2.0", req_id="new", requester="U2")
    on_disk = json.loads(idx.read_text())
    assert "C_NEW:2.0" in on_disk, (
        "a loaded index refused a legitimate write — label attribution "
        "would be dead in production")
    # ...and the PRE-EXISTING entry must survive the save. Without this the
    # test passed on a save that wrote only the new key — i.e. on exactly
    # the truncation it exists to prevent. (The earlier
    # `assert ... is not None or True` line was a tautology and is gone.)
    assert "C_OLD:1.0" in on_disk, (
        "the save dropped an entry loaded from disk — this IS the "
        "truncation regression")
