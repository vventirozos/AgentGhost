"""The autouse isolation guards must survive a test-body `monkeypatch.undo()`.

⚠ THEY DID NOT. `monkeypatch` is ONE instance per test — every fixture that
requests it and the test body itself share a single undo stack — so a
body-level `monkeypatch.undo()` popped `conftest._isolate_ghost_home` along
with whatever the test meant to revert. Measured 2026-08-24:

    GHOST_HOME during test : .../isolated_ghost_homes0/0
    GHOST_HOME after undo(): /Users/vasilis/Data/AI/Data   # the LIVE store

Four tests call `undo()` in their body today, and the surfacing symptom was
unrelated and much smaller: `test_notify_promise.py` also lost its autouse
notify-budget reset that way, inherited the process-global 12/hour
rate-limiter from whichever files shared its xdist worker, and failed
intermittently under `-n 8 --dist loadfile` while passing 30/30 alone.

The isolation fixtures now hold a PRIVATE `pytest.MonkeyPatch`, unreachable
from the test's stack. These pins execute the bypass rather than asserting
on the fixture's source: `guard-a-proxy-not-the-thing`,
`token-pins-vs-executed-pins`.
"""

import os
from pathlib import Path

LIVE = Path("/Users/vasilis/Data/AI/Data")


def _home() -> Path:
    return Path(os.environ.get("GHOST_HOME", ""))


def test_ghost_home_is_isolated_to_begin_with():
    assert _home() != LIVE
    assert "isolated_ghost_homes" in str(_home())


def test_a_body_level_undo_cannot_reach_the_live_store(monkeypatch):
    """THE BYPASS, DRIVEN. Reverting the fixture to the shared
    `monkeypatch` makes this fail with GHOST_HOME == the live store."""
    monkeypatch.setenv("_IRRELEVANT_PATCH", "1")
    monkeypatch.undo()
    assert _home() != LIVE, (
        "a test-body monkeypatch.undo() un-isolated GHOST_HOME and "
        "repointed it at the operator's LIVE data directory")
    assert "isolated_ghost_homes" in str(_home())


def test_the_slack_side_files_also_survive_undo(monkeypatch):
    """⚠ THIS ONE PASSES FOR A DIFFERENT REASON THAN THE PIN ABOVE, and
    saying so is the point. `conftest` assigns both vars process-wide at
    IMPORT time, outside monkeypatch, so they were never reachable by a
    body-level `undo()`. Mutation-checked 2026-08-24: reverting
    `_isolate_live_side_files` to the shared stack leaves this GREEN — it
    does not pin that fixture, and reading it as though it does would be
    exactly the false confidence the GHOST_HOME leak was hiding behind.
    Kept because the property it states is the one that matters: these
    must never resolve to the operator's live files."""
    monkeypatch.setenv("_IRRELEVANT_PATCH", "1")
    monkeypatch.undo()
    for var in ("GHOST_SLACK_REPLY_INDEX", "GHOST_SLACKBOT_LOG"):
        assert os.environ.get(var) == "", (
            f"{var} escaped isolation and may now resolve to the "
            f"operator's live file")


def test_a_test_that_sets_its_own_home_still_WINS(monkeypatch, tmp_path):
    """The private patcher must not make the guard unoverridable — the
    fixtures' own docstrings promise `monkeypatch.setenv` runs after."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path / "mine"))
    assert _home() == tmp_path / "mine"


def test_each_test_gets_a_DISTINCT_home(tmp_path):
    """Recorded so the private patcher's teardown is visible: the counter
    still advances, i.e. the fixture is still running per test."""
    (_home() / "marker").write_text("x")
    assert (_home() / "marker").exists()
    assert not (LIVE / "marker").exists()
