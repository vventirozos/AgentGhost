"""Fixes from the 2026-06-11 core-loop audit.

1. The strike-decay freeze (`persistent_failure_seen`) is no longer
   permanent for the request: 3 consecutive clean tool successes unfreeze
   it (a genuine pivot), while the fail→auto-list→fail oscillation that
   motivated the freeze can never produce 3 in a row. Signature counts
   are kept so the SAME failure re-freezes on its next occurrence.
2. Tool-argument parse failures are never silently swallowed
   (`except: pass` / `except: args_val = {}`) — each site logs a WARNING
   naming the tool so "tool got empty args" loops are diagnosable.
3. Journal appends from the turn loop are bounded (`_journal_append_safe`)
   so a hung journal file op cannot park the event loop until the
   watchdog's opaque activity-timeout.

The in-loop changes use source-level contracts (the suite's established
pattern for fixes whose behaviour needs a live upstream); the new journal
helper is tested behaviourally.
"""

import inspect
import time
import types
from pathlib import Path

from ghost_agent.core import agent as agent_mod
from ghost_agent.core.agent import GhostAgent

SRC = Path(inspect.getfile(agent_mod)).read_text()


# --- 1. reversible decay-freeze (source contract) -------------------------
# The state itself now lives on the StrikeLedger (core.strikes); the loop
# drives it through these call sites. Behaviour is covered exhaustively by
# tests/test_strike_ledger.py — here we just pin the wiring.


def test_turn_loop_drives_clean_success_and_reset():
    assert "strikes.note_clean_success()" in SRC
    assert "strikes.reset_clean_streak()" in SRC


def test_turn_loop_uses_ledger_for_failures():
    assert "strikes.note_failure(" in SRC
    assert "strikes.note_action(" in SRC


def test_decay_gated_on_ledger_freeze_flag():
    # the execution_failure_count decay is gated on the (reversible) flag
    assert "if execution_failure_count > 0 and not strikes.decay_frozen:" in SRC


# --- 2. no silent tool-arg parse failures (source contract) ---------------


def test_no_silent_empty_args_fallback():
    assert "except: args_val = {}" not in SRC
    assert "Unparseable JSON arguments for tool" in SRC


def test_raw_json_fallback_logs():
    assert "has non-JSON" in SRC  # raw-JSON tool-call arguments warning


def test_unescape_step_logs():
    assert "JSON at the unescape step" in SRC


# --- 3. bounded journal appends (behavioural) ------------------------------


def _fake_agent(journal):
    return types.SimpleNamespace(context=types.SimpleNamespace(journal=journal))


class _RecordingJournal:
    def __init__(self, delay=0.0, raises=None):
        self.delay = delay
        self.raises = raises
        self.entries = []

    def append(self, kind, payload):
        if self.delay:
            time.sleep(self.delay)
        if self.raises:
            raise self.raises
        self.entries.append((kind, payload))


async def test_journal_append_safe_writes_through():
    j = _RecordingJournal()
    await GhostAgent._journal_append_safe(_fake_agent(j), "post_mortem", {"x": 1})
    assert j.entries == [("post_mortem", {"x": 1})]


async def test_journal_append_safe_drops_on_timeout_without_raising():
    j = _RecordingJournal(delay=0.5)
    await GhostAgent._journal_append_safe(
        _fake_agent(j), "smart_memory", {"x": 1}, timeout=0.05
    )
    # returned promptly and did not raise; the entry was dropped
    assert j.entries == []


async def test_journal_append_safe_swallows_journal_errors():
    j = _RecordingJournal(raises=OSError("disk full"))
    await GhostAgent._journal_append_safe(_fake_agent(j), "smart_memory", {"x": 1})


async def test_journal_append_safe_no_journal_is_noop():
    await GhostAgent._journal_append_safe(_fake_agent(None), "smart_memory", {})


def test_turn_loop_uses_safe_journal_helper():
    # every turn-loop journal write goes through the bounded helper;
    # the raw to_thread(journal.append) pattern must not reappear there
    assert "await self._journal_append_safe(" in SRC
    assert "asyncio.to_thread(self.context.journal.append" not in SRC
