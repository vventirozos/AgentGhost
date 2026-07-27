"""uncertainty_pressure de-zeroing (2026-07-27, later).

The calibration feature was a constant 0.0 across the entire corpus
(1 distinct value — flagged by learning-health feature-liveness).
Root cause, established against the live durable log (2 records ever,
both hedge-scan; the flag_uncertainty tool has NEVER been called):

  1. Non-streamed finalize: ``_record_calibration_safe`` read
     ``tracker.pressure()`` BEFORE the hedge auto-scan populated the
     tracker — the scan lived in the surfacing block ~20 lines below
     the record, so every sample saw the pre-scan (empty) state.
  2. Streamed path: no hedge scan ran at all before the pressure read
     in the end-of-stream confidence reading.
  3. Feeder volume: the hedge regex missed common phrasings
     ("unable to verify", "can't confirm", "I'm uncertain").

Fixes pinned here: scan moved BEFORE the record (finalize), scan added
before the streamed read, streamed drain resets the tracker so hedge
state can't leak into the next turn's reading, regex broadened
conservatively.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../src')))

from ghost_agent.core.uncertainty import UncertaintyTracker

SRC = (Path(__file__).resolve().parents[1]
       / "src" / "ghost_agent" / "core" / "agent.py").read_text()


# ──────────────────────────────────────────────────────────────────────
# Behavioral: hedge → pressure
# ──────────────────────────────────────────────────────────────────────

class TestHedgePressure:
    def test_hedge_assumption_raises_pressure(self):
        t = UncertaintyTracker()
        assert t.pressure() == 0.0
        t.flag_assumption("I'm assuming the schema is unchanged.",
                          confidence=0.4, basis="auto-detected hedge")
        assert t.pressure() > 0.0

    def test_reset_returns_pressure_to_zero(self):
        t = UncertaintyTracker()
        t.flag_assumption("hedge", confidence=0.4)
        assert t.pressure() > 0.0
        t.reset()
        assert t.pressure() == 0.0

    def test_scan_to_pressure_end_to_end(self):
        """The exact pipeline the turn loop runs: scan the reply text,
        flag each hedge at 0.4, read pressure."""
        t = UncertaintyTracker()
        reply = ("Here is the migration. I couldn't verify the index "
                 "sizes on the replica, so treat that part as a guess.")
        hedges = t.scan_text_for_uncertainty(reply)
        assert hedges
        for h in hedges:
            t.flag_assumption(h, confidence=0.4, basis="auto")
        assert t.pressure() > 0.0

    def test_broadened_hedge_forms_match(self):
        t = UncertaintyTracker()
        for text in (
            "I was unable to verify the checksum.",
            "I can't confirm the deploy landed.",
            "I'm uncertain about the timezone handling.",
            "I have no way to check the remote state.",
        ):
            assert t.scan_text_for_uncertainty(text), text

    def test_plain_confident_text_matches_nothing(self):
        t = UncertaintyTracker()
        assert t.scan_text_for_uncertainty(
            "Done. All 32 checks passed and the deploy is live.") == []


# ──────────────────────────────────────────────────────────────────────
# Wiring: order of scan vs read, on both paths (source-pinned — the
# paths live deep in handle_chat / the finalize chain and are not
# unit-instantiable; same pattern as the entropy wiring tests).
# ──────────────────────────────────────────────────────────────────────

class TestPressureWiring:
    def test_finalize_scans_before_calibration_record(self):
        """The hedge scan must populate the tracker BEFORE
        _record_calibration_safe reads pressure()."""
        first_scan = SRC.index("scan_text_for_uncertainty")
        first_record_call = SRC.index("await self._record_calibration_safe(")
        assert first_scan < first_record_call

    def test_streamed_path_scans_before_pressure_read(self):
        """The end-of-stream confidence reading must scan the streamed
        answer before reading pressure."""
        read_idx = SRC.index("_upress = _utk.pressure()")
        window = SRC[max(0, read_idx - 1500):read_idx]
        assert "scan_text_for_uncertainty" in window

    def test_streamed_drain_resets_tracker(self):
        """Without a streamed-side reset, hedge assumptions leak into
        the NEXT turn's pressure reading (finalize's reset never runs
        on the stream path)."""
        assert "Streamed-turn tracker reset" in SRC
        reset_idx = SRC.index("Streamed-turn tracker reset")
        # The reset must come AFTER the drain's calibration record.
        drain_record = SRC.rindex(
            "await self._record_calibration_safe(", 0, reset_idx)
        assert drain_record < reset_idx

    def test_surfacing_block_no_longer_scans(self):
        """The scan moved out of the surfacing block — exactly one scan
        site per path (finalize + streamed), no double-flagging."""
        assert SRC.count("scan_text_for_uncertainty") == 2
