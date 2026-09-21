"""Pins for scripts/night_bench_round.py's two launch-deciding parsers, on
the REAL strings the instruments printed on 2026-09-20 (§4JD night prep).

World where they fail: the rehearsal that day — a smoke replayed from cache
in 3 s, the rate read 20 cases/min, and the MEASURED gate would have cleared
a 2.4 h run with a 4-minute ETA. The parser now reads live calls, and zero
live calls refuses the launch.
"""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "night_bench_round", Path(__file__).resolve().parents[1] / "scripts" / "night_bench_round.py")
nbr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nbr)

ORACLE_STALE = """==============================================================================
VERIFIER BENCH STATUS   baseline recorded 2026-08-09T12:41:53Z
==============================================================================
  balanced 0.825  95% CI [0.787, 0.864] (±0.038, n=103 non-refute / 330 refute)
  pool now: 58 cases (tier=private)   baseline: 58

  STALE — replayable   (6 component(s) drifted)

  • code.bench  — bench/scoring changed (the ruler, not the system)
      was '43da323adbdeb64e'
  • code.verifier  — verifier logic changed (system under test)
  • faults_sha256  — fault library changed
  • templates.verifier.adjudicate  — rendered prompt changed
  • templates.verifier.claim  — rendered prompt changed
  • templates.verifier.enumerate  — rendered prompt changed

  UNCOMPARABLE (13) — evidence missing, NOT evidence of change:
      verify_flags.GHOST_CLAIM_BINDING_CONFIRM_FIRST absent from baseline
"""


def test_oracle_parse_names_every_drifted_component():
    o = nbr.parse_oracle(ORACLE_STALE, 1)
    assert o["verdict"] == "STALE"
    assert o["pool_cases"] == 58
    assert o["drifted"] == ["code.bench", "code.verifier", "faults_sha256",
                            "templates.verifier.adjudicate", "templates.verifier.claim",
                            "templates.verifier.enumerate"]
    # the UNCOMPARABLE block must NOT count as drift (it is not a bullet)
    assert o["drifted_components"] == 6


@pytest.mark.parametrize("rc, verdict", [(0, "VALID"), (1, "STALE"), (2, "NO_BASELINE"), (3, "rc=3")])
def test_oracle_verdict_follows_the_exit_code(rc, verdict):
    assert nbr.parse_oracle("", rc)["verdict"] == verdict


def test_a_live_write_mode_smoke_counts_its_writes_as_live_calls():
    tail = "cheap leg: 16 calls, 0 failures.\n...\ncache: 0 hits / 0 misses / 20 writes — live judge\n"
    assert nbr.parse_smoke_live_calls(tail) == (0, 20)


def test_a_replayed_smoke_reads_zero_live_calls():
    """The rehearsal case: hits only → 0, and the stage refuses to launch."""
    tail = "cheap leg: 0 calls, 0 failures.\ncache: 20 hits / 0 misses / 0 writes — replayed\n"
    assert nbr.parse_smoke_live_calls(tail) == (20, 0)


def test_read_mode_counts_misses():
    tail = "cache: 159 hits / 1468 misses / 1468 writes — MIXED\n"
    assert nbr.parse_smoke_live_calls(tail) == (159, 2936)


def test_no_cache_line_but_a_route_line_is_the_witness():
    assert nbr.parse_smoke_live_calls("cheap leg: 12 calls, 0 failures.\ncache: 0 hits / 0 misses\n") == (0, 12)


def test_nothing_parseable_is_none_not_zero():
    """Absent must not read as zero (judge-instrument-failures): None means
    the smoke output was unreadable, which is its own refusal."""
    assert nbr.parse_smoke_live_calls("") == (None, None)
