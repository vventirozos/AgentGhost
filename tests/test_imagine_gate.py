"""§4CL I0 — the Imagine calibration gate (`core/imagination.py`).

The gate is the instrument that decides, per `(tool, tclass)` bucket and
from ledger data alone, whether foresight precedent is good enough to
steer with. It ships before any steering site exists, and every consumer
is gated on it — so the pins that matter are the CLOSED ones: a gate that
opens by accident is worse than no gate.

Pinned here:
  * fail-closed on every degenerate input (no file, empty file, truncated
    JSON, wrong shape, unknown bucket, `enabled` missing / not exactly
    True);
  * each enablement condition is INDEPENDENTLY necessary — one synthetic
    ledger per condition, each failing exactly one;
  * the `why` string names the BINDING constraint (a gate that reports
    only `false` cannot tell "no data yet" from "measured dead");
  * the mtime+size cache serves a rebuilt gate, not a stale one;
  * the builder and the reader key buckets identically (a silent key
    mismatch is a permanently-closed gate nobody notices).
"""
import json
import os
from pathlib import Path

import pytest

from ghost_agent.core import imagination as IM


@pytest.fixture(autouse=True)
def _fresh_cache():
    IM.reset_gate_cache_for_tests()
    yield
    IM.reset_gate_cache_for_tests()


def _row(tool, tclass, *, steerable, ok):
    """One ledger row, shaped the way `foresight._write_ledger` writes
    them. `steerable=True` means the row belongs to the population a
    pre-flight steer would ACT on — exact/class basis, support ≥ 3, a
    STRICT majority of real failures, and an error head to report. That
    is the population the gate certifies, so a fixture that omits those
    fields is testing a gate nobody ships."""
    if steerable:
        support, fails = 6, 5           # 2*5 > 6 → a strict claim
    else:
        support, fails = 6, 1
    rec = {"tool": tool, "tclass": tclass, "basis": "exact",
           "support": support, "fails": fails,
           "p_fail": round((fails + 1) / (support + 2), 4),
           "ok": ok}
    rec["match"] = (rec["p_fail"] >= 0.5) == (not ok)
    if fails:
        rec["pred_err"] = "filenotfounderror"
    return rec


def _rows(*, tool="file_system", tclass="ext:py", n_ok_low=0, n_fail_low=0,
          n_ok_high=0, n_fail_high=0):
    """`low` = rows the steer would NOT act on, `high` = rows it would;
    `ok`/`fail` is what ACTUALLY happened."""
    return (
        [_row(tool, tclass, steerable=False, ok=True)] * n_ok_low
        + [_row(tool, tclass, steerable=False, ok=False)] * n_fail_low
        + [_row(tool, tclass, steerable=True, ok=True)] * n_ok_high
        + [_row(tool, tclass, steerable=True, ok=False)] * n_fail_high
    )


def _qualifying():
    """A bucket that passes every condition: 60 rows, 20 predicted-fail
    of which 18 really failed (precision 0.90), 40 predicted-ok of which
    2 failed (0.05) — spread +0.85, comfortably disjoint."""
    return _rows(n_ok_low=38, n_fail_low=2, n_ok_high=2, n_fail_high=18)


# ------------------------------------------------------------------ #
# Fail-closed                                                        #
# ------------------------------------------------------------------ #

def test_no_gate_file_means_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    assert IM.gate_allows("file_system", "ext:py") is False
    assert IM.gate_stats()["present"] is False


def test_no_ghost_home_means_closed(monkeypatch):
    monkeypatch.setenv("GHOST_HOME", "")
    assert IM.gate_allows("file_system", "ext:py") is False


@pytest.mark.parametrize("body", [
    "",                                   # empty file
    "{",                                  # truncated JSON
    "null",                               # parses, wrong type
    "[]",                                 # parses, wrong shape
    '{"buckets": "not-a-dict"}',          # right key, wrong type
    '{"enabled_count": 3}',               # no buckets key at all
])
def test_degenerate_gate_files_are_closed(tmp_path, monkeypatch, body):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    p = tmp_path / "system" / "foresight" / IM.GATE_FILENAME
    p.parent.mkdir(parents=True)
    p.write_text(body)
    assert IM.gate_allows("file_system", "ext:py") is False


@pytest.mark.parametrize("value", [True, "true", 1, "yes", None, 0])
def test_enabled_must_be_exactly_true(tmp_path, monkeypatch, value):
    """An allow-list that accepts a truthy string is a deny-list with
    extra steps — a JSON round-trip through some other tool that writes
    `"true"` must not open a bucket."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    p = tmp_path / "system" / "foresight" / IM.GATE_FILENAME
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(
        {"buckets": {"file_system|ext:py": {"enabled": value}}}))
    assert IM.gate_allows("file_system", "ext:py") is (value is True)


def test_unknown_bucket_is_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    IM.build_gate(_qualifying(), write=True)
    assert IM.gate_allows("file_system", "ext:py") is True
    assert IM.gate_allows("execute", "cmd:rm") is False
    assert IM.gate_allows("file_system", "") is False
    assert IM.gate_allows("", "") is False


# ------------------------------------------------------------------ #
# Each condition is independently necessary                          #
# ------------------------------------------------------------------ #

def test_a_qualifying_bucket_opens(tmp_path, monkeypatch):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    doc = IM.build_gate(_qualifying(), write=True)
    entry = doc["buckets"]["file_system|ext:py"]
    assert entry["enabled"] is True, entry["why"]
    assert doc["enabled_count"] == 1
    assert "DISCRIMINATES" in entry["why"]
    assert IM.gate_allows("file_system", "ext:py") is True


def test_thin_bucket_is_closed_and_says_so(tmp_path, monkeypatch):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    # Same shape, scaled down under the 30-row floor.
    doc = IM.build_gate(
        _rows(n_ok_low=8, n_fail_low=0, n_ok_high=1, n_fail_high=10),
        write=True)
    entry = doc["buckets"]["file_system|ext:py"]
    assert entry["enabled"] is False
    assert entry["why"].startswith("thin bucket")
    assert IM.gate_allows("file_system", "ext:py") is False


def test_precision_without_a_denominator_never_opens(tmp_path, monkeypatch):
    """§4CE: precision 1.00 over 3 rows is not a measurement. This is the
    condition the live ledger actually fails on, so it is the one most
    likely to be 'relaxed' later."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    doc = IM.build_gate(
        _rows(n_ok_low=48, n_fail_low=0, n_ok_high=0, n_fail_high=3),
        write=True)
    entry = doc["buckets"]["file_system|ext:py"]
    assert entry["precision"] == 1.0          # perfect…
    assert entry["fail_n"] == 3               # …over three rows
    assert entry["enabled"] is False
    assert entry["why"].startswith("no denominator")


def test_low_precision_is_closed_even_when_it_discriminates(tmp_path,
                                                            monkeypatch):
    """The measured live case: predicted-fail rows DO fail more than
    predicted-ok rows (real discrimination), but only ~half of them fail
    — so a pre-flight steer would interrupt about as many good calls as
    bad ones. Discrimination is necessary, not sufficient."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    doc = IM.build_gate(
        _rows(n_ok_low=40, n_fail_low=0, n_ok_high=10, n_fail_high=10),
        write=True)
    entry = doc["buckets"]["file_system|ext:py"]
    assert entry["precision"] == 0.5
    assert entry["spread"] == 0.5             # it genuinely discriminates
    assert entry["enabled"] is False
    assert entry["why"].startswith("precision")


def test_flat_bucket_is_closed(tmp_path, monkeypatch):
    """High precision but the predicted-OK rows fail just as often — the
    index is not separating anything, it is in a bucket that always
    fails. Steering on it changes nothing and costs a round-trip."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    doc = IM.build_gate(
        _rows(n_ok_low=4, n_fail_low=36, n_ok_high=2, n_fail_high=18),
        write=True)
    entry = doc["buckets"]["file_system|ext:py"]
    assert entry["precision"] >= 0.6
    assert entry["spread"] is not None and entry["spread"] < 0.10
    assert entry["enabled"] is False
    assert entry["why"].startswith("flat")


def test_no_comparison_group_is_closed(tmp_path, monkeypatch):
    """Every claimed row on one side of p=0.5 — discrimination is
    undefined, and undefined must read as closed, not as 'fine'."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    doc = IM.build_gate(_rows(n_ok_high=5, n_fail_high=45), write=True)
    entry = doc["buckets"]["file_system|ext:py"]
    assert entry["precision"] == 0.9 and entry["fail_n"] == 50
    assert entry["spread"] is None
    assert entry["enabled"] is False
    assert entry["why"].startswith("no comparison group")


def test_overlapping_intervals_are_closed(tmp_path, monkeypatch):
    """Spread and precision both clear, but at this n the anytime-valid
    intervals still overlap — the same 'SPREAD BUT NOT SIGNIFICANT'
    verdict the whole-ledger backtest reports."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    # 20 predicted-fail (13 fail → 0.65), 12 predicted-ok (4 fail → 0.33):
    # spread +0.32 but both subsets are small and noisy.
    doc = IM.build_gate(
        _rows(n_ok_low=8, n_fail_low=4, n_ok_high=7, n_fail_high=13),
        write=True)
    entry = doc["buckets"]["file_system|ext:py"]
    assert entry["precision"] >= 0.6 and entry["spread"] >= 0.10
    assert entry["disjoint"] is False
    assert entry["enabled"] is False
    assert "not significant" in entry["why"]


# ------------------------------------------------------------------ #
# Bookkeeping the gate has to get right                              #
# ------------------------------------------------------------------ #

def test_rows_without_a_claim_count_for_coverage_not_for_precision(
        tmp_path, monkeypatch):
    """basis="none" rows carry no probability. They must still count
    toward the bucket's size (they are real executions) but must not
    move precision or spread."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    rows = _qualifying() + [
        {"tool": "file_system", "tclass": "ext:py", "ok": False}] * 20
    doc = IM.build_gate(rows, write=True)
    entry = doc["buckets"]["file_system|ext:py"]
    assert entry["n"] == 80 and entry["claimed"] == 60
    assert entry["fail_n"] == 20 and entry["precision"] == 0.9


def test_a_corrupt_row_is_skipped_not_fatal(tmp_path, monkeypatch):
    """One unparseable row must degrade to a skipped row, never a blanked
    gate — the §4K lesson, applied to the gate builder."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    rows = _qualifying() + [{"tool": "file_system", "tclass": "ext:py",
                             "p_fail": "not-a-number", "ok": True}]
    doc = IM.build_gate(rows, write=True)
    assert doc["buckets"]["file_system|ext:py"]["enabled"] is True


def test_params_are_recorded_into_the_file(tmp_path, monkeypatch):
    """A gate on disk must name the thresholds that produced it —
    otherwise a threshold change silently re-reads a file that means
    something nobody chose."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    monkeypatch.setenv("GHOST_IMAGINE_GATE_MIN_PRECISION", "0.95")
    doc = IM.build_gate(_qualifying(), write=True)
    on_disk = json.loads(
        (tmp_path / "system" / "foresight" / IM.GATE_FILENAME).read_text())
    assert on_disk["params"]["min_fail_precision"] == 0.95
    # …and the tightened bar BITES: the same ledger that opens the bucket
    # at the default 0.60 (precision 0.90) is closed at 0.95.
    assert doc["buckets"]["file_system|ext:py"]["enabled"] is False
    assert on_disk["buckets"]["file_system|ext:py"]["enabled"] is False
    monkeypatch.delenv("GHOST_IMAGINE_GATE_MIN_PRECISION")
    assert IM.build_gate(_qualifying(), write=False)[
        "buckets"]["file_system|ext:py"]["enabled"] is True


def test_tightening_a_threshold_closes_a_previously_open_bucket(
        tmp_path, monkeypatch):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    IM.build_gate(_qualifying(), write=True)
    assert IM.gate_allows("file_system", "ext:py") is True
    monkeypatch.setenv("GHOST_IMAGINE_GATE_MIN_FAIL_N", "500")
    IM.build_gate(_qualifying(), write=True)
    IM.reset_gate_cache_for_tests()
    assert IM.gate_allows("file_system", "ext:py") is False


def test_cache_serves_a_rebuilt_gate(tmp_path, monkeypatch):
    """The read side caches on (mtime, size). A rebuild that flips a
    bucket must be visible without a process restart — a gate that can
    only tighten at boot is a gate that lies for a day."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    IM.build_gate(_qualifying(), write=True)
    assert IM.gate_allows("file_system", "ext:py") is True
    # Rebuild with a thin ledger; same path, different content/size.
    IM.build_gate(_rows(n_ok_low=5, n_fail_high=5), write=True)
    assert IM.gate_allows("file_system", "ext:py") is False


def test_builder_and_reader_agree_on_the_bucket_key(tmp_path, monkeypatch):
    """A silent key mismatch between the two sides is a permanently
    closed gate that reports itself as healthy."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    doc = IM.build_gate(_rows(tool="browser", tclass="scheme:http",
                              n_ok_low=38, n_fail_low=2,
                              n_ok_high=2, n_fail_high=18), write=True)
    key, = [k for k, v in doc["buckets"].items() if v["enabled"]]
    assert key == IM.bucket_key("browser", "scheme:http")
    assert IM.gate_allows("browser", "scheme:http") is True


def test_empty_tclass_round_trips(tmp_path, monkeypatch):
    """Half the live buckets have an empty tclass (tools with no
    classifiable target). They must be addressable, not collapse into
    each other or into a missing key."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    IM.build_gate(_rows(tool="manage_services", tclass="",
                        n_ok_low=38, n_fail_low=2,
                        n_ok_high=2, n_fail_high=18), write=True)
    assert IM.gate_allows("manage_services", "") is True
    assert IM.gate_allows("manage_services", "ext:py") is False


def test_build_from_a_ledger_file_on_disk(tmp_path, monkeypatch):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    led = tmp_path / "system" / "foresight" / "predictions.jsonl"
    led.parent.mkdir(parents=True)
    led.write_text("\n".join(json.dumps(r) for r in _qualifying()) + "\n")
    doc = IM.build_gate(write=True)
    assert doc["ledger_rows"] == 60
    assert IM.gate_allows("file_system", "ext:py") is True


def test_build_reads_the_rotation_generation_too(tmp_path, monkeypatch):
    """The ledger rotates to `.1` at 8 MB. A gate built from only the
    live file would silently halve its evidence the day it rotates."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    led = tmp_path / "system" / "foresight" / "predictions.jsonl"
    led.parent.mkdir(parents=True)
    rows = _qualifying()
    Path(str(led) + ".1").write_text(
        "\n".join(json.dumps(r) for r in rows[:30]) + "\n")
    led.write_text("\n".join(json.dumps(r) for r in rows[30:]) + "\n")
    doc = IM.build_gate(write=True)
    assert doc["ledger_rows"] == 60
    assert doc["buckets"]["file_system|ext:py"]["enabled"] is True


def test_build_without_ghost_home_is_a_closed_gate_not_a_crash(monkeypatch):
    monkeypatch.setenv("GHOST_HOME", "")
    doc = IM.build_gate()
    assert doc["enabled_count"] == 0 and doc["buckets"] == {}
    assert "no GHOST_HOME" in doc["reason"]


def test_gate_stats_reports_the_closed_case_with_reasons(tmp_path,
                                                         monkeypatch):
    """"waiting for data" and "measured flat" lead opposite places; the
    health surface has to distinguish them."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    rows = (_rows(tool="a", tclass="", n_ok_low=5, n_fail_high=5)
            + _rows(tool="b", tclass="", n_ok_low=48, n_fail_high=3))
    IM.build_gate(rows, write=True)
    st = IM.gate_stats()
    assert st["present"] is True and st["enabled_count"] == 0
    assert st["buckets"] == 2 and st["enabled"] == []
    assert set(st["closed_reasons"]) == {"thin bucket", "no denominator"}


def test_gate_write_is_atomic(tmp_path, monkeypatch):
    """A reader must never see a half-written allow-list. The temp file
    must not survive either — a stray `.json.tmp` is how a later `glob`
    ends up loading a partial gate."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    IM.build_gate(_qualifying(), write=True)
    d = tmp_path / "system" / "foresight"
    assert (d / IM.GATE_FILENAME).exists()
    assert not list(d.glob("*.tmp"))


# ------------------------------------------------------------------ #
# I0 wiring — the idle rebuild, the health surface, the instrument    #
# ------------------------------------------------------------------ #

def test_the_idle_phase_is_registered_for_the_liveness_alarm():
    """A loop that stops writing must be distinguishable from one that
    never existed — that is the whole point of the registry. PERIODIC
    means a zero over the window is an ALARM, which is correct here: a
    stale allow-list is exactly the failure worth waking up for."""
    from ghost_agent.core.autonomous_activity import (
        EXPECT_PERIODIC, PHASE_EXPECTATION, _PHASE_LABELS,
    )
    assert PHASE_EXPECTATION.get("imagine_gate") == EXPECT_PERIODIC
    assert _PHASE_LABELS.get("imagine_gate")   # renders, not a raw slug


def _bio_ctx(idle_secs: float):
    """Minimal context for `_biological_tick`, mirroring the harness in
    tests/test_selfhood_derived_mood.py."""
    import datetime
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    ctx = MagicMock()
    ctx.memory_system = MagicMock()
    ctx.memory_system.collection.get = MagicMock(return_value={"ids": []})
    ctx.llm_client = SimpleNamespace(foreground_tasks=0)
    ctx.journal = None
    ctx.last_activity_time = (datetime.datetime.now()
                              - datetime.timedelta(seconds=idle_secs))
    ctx.args = MagicMock()
    ctx.args.model = "test-model"
    ctx.args.prm_train_cooldown = None
    ctx.args.self_narrative_cooldown = None
    ctx.args.calib_refit_cooldown = None
    ctx.frontier_tracker = None
    ctx.reflector = None
    ctx.trajectory_collector = None
    ctx.prm_scorer = None
    ctx.postmortem_engine = None
    ctx.calibration_tracker = None
    return ctx


async def _tick(ctx, agent=None):
    from ghost_agent.core.agent import GhostAgent
    if agent is None:
        agent = GhostAgent.__new__(GhostAgent)
        agent.context = ctx
    await agent._biological_tick()
    return agent


def _seed_thin_ledger(tmp_path):
    led = tmp_path / "system" / "foresight" / "predictions.jsonl"
    led.parent.mkdir(parents=True, exist_ok=True)
    led.write_text("\n".join(json.dumps(r) for r in _rows(
        tool="execute", tclass="cmd:python3",
        n_ok_low=5, n_fail_high=5)) + "\n")


async def test_the_idle_phase_actually_builds_the_gate(tmp_path, monkeypatch):
    """Executed end-to-end through the REAL `_biological_tick`: the gate
    file appears, and the recorded summary names the binding reason —
    "0 enabled" alone cannot tell "no data yet" from "measured not
    precise enough", and those lead opposite places."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    _seed_thin_ledger(tmp_path)
    ctx = _bio_ctx(idle_secs=1200)
    recorded = []
    agent = await _tick(ctx)
    # `_record_autonomous_activity` writes through ctx.activity_log,
    # which is a MagicMock here — read the call instead.
    for call in ctx.activity_log.record.call_args_list:
        recorded.append(call.args[:2])
    assert (tmp_path / "system" / "foresight" / IM.GATE_FILENAME).exists()
    gate = json.loads((tmp_path / "system" / "foresight"
                       / IM.GATE_FILENAME).read_text())
    assert gate["enabled_count"] == 0
    assert gate["buckets"]["execute|cmd:python3"]["why"].startswith(
        "thin bucket")
    summaries = [s for p, s in recorded if p == "imagine_gate"]
    assert summaries, f"no imagine_gate activity row; got {recorded}"
    assert "thin bucket" in summaries[0]
    assert agent._last_imagine_gate_at > __import__("datetime").datetime.min


async def test_the_idle_phase_respects_its_cooldown(tmp_path, monkeypatch):
    import datetime
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    _seed_thin_ledger(tmp_path)
    ctx = _bio_ctx(idle_secs=1200)
    agent = await _tick(ctx)
    ctx.last_activity_time = (datetime.datetime.now()
                              - datetime.timedelta(seconds=1300))
    await _tick(ctx, agent)
    fired = [c for c in ctx.activity_log.record.call_args_list
             if c.args[0] == "imagine_gate"]
    assert len(fired) == 1, "the 24h cooldown did not hold"


async def test_the_idle_phase_stays_out_of_the_deep_idle_band(
        tmp_path, monkeypatch):
    """The 15-60 min band is reserved; >60 min belongs to the deep phases
    (self-play, bench). Losing the upper bound leaks a nightly job into
    the window those own."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    _seed_thin_ledger(tmp_path)
    for idle in (600, 4000):
        ctx = _bio_ctx(idle_secs=idle)
        await _tick(ctx)
        assert not (tmp_path / "system" / "foresight"
                    / IM.GATE_FILENAME).exists(), f"fired at idle={idle}s"


async def test_the_gate_is_rebuilt_even_when_imagine_is_off(
        tmp_path, monkeypatch):
    """The master flag gates the CONSUMERS, not the instrument. A closed
    gate is a measurement, and the question "has the precedent index
    become good enough yet?" must keep being answered while the feature
    is disabled — otherwise the day it becomes usable never arrives."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    monkeypatch.setenv("GHOST_IMAGINE", "0")
    _seed_thin_ledger(tmp_path)
    ctx = _bio_ctx(idle_secs=1200)
    await _tick(ctx)
    assert (tmp_path / "system" / "foresight" / IM.GATE_FILENAME).exists()


def test_the_health_surface_distinguishes_no_data_from_measured_flat(
        tmp_path, monkeypatch):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    # thin bucket → "no data yet"
    IM.build_gate(_rows(n_ok_low=5, n_fail_high=5), write=True)
    st = IM.gate_stats()
    assert st["closed_reasons"] == {"thin bucket": 1}
    # …and a bucket that HAS the data but not the precision reads
    # differently, which is the distinction the surface exists for.
    IM.reset_gate_cache_for_tests()
    IM.build_gate(_rows(n_ok_low=40, n_ok_high=10, n_fail_high=10),
                  write=True)
    st = IM.gate_stats()
    assert list(st["closed_reasons"])[0].startswith("precision")


def test_the_offline_instrument_buckets_by_the_same_key_as_the_live_one():
    """The gate keys on (tool, tclass). The offline replay used to emit
    rows WITHOUT tclass, so it could only be bucketed by tool — which
    merges `file_system` on a .py file with `file_system` on a URL, and a
    coarser key hides exactly the sub-buckets a gate exists to find."""
    import inspect
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "scripts"))
    import foresight_backtest as FB

    src = inspect.getsource(FB.iter_replay_rows)
    assert '"tclass": pred.tclass' in src
    # …and the key the gate derives from such a row round-trips.
    assert IM.bucket_key("file_system", "ext:py") == "file_system|ext:py"


def test_the_consistency_instrument_reports_unarmed_instead_of_a_number(
        tmp_path, monkeypatch):
    """§4CE: a ratio computed over zero rows is the shape of a
    measurement nobody can act on. Until I4 writes the regret ledger this
    must say so, with its own exit code."""
    import subprocess
    import sys
    from pathlib import Path as _P
    repo = _P(__file__).resolve().parents[1]
    env = dict(os.environ, GHOST_HOME=str(tmp_path),
               PYTHONPATH=str(repo / "src"))
    out = subprocess.run(
        [sys.executable, str(repo / "scripts" / "foresight_backtest.py"),
         "--consistency", "--json"],
        capture_output=True, env=env, timeout=120)
    assert out.returncode == 2, out.stderr.decode()[:400]
    payload = json.loads(out.stdout.decode())
    assert payload["armed"] is False and payload["ratio"] is None


def test_the_consistency_instrument_only_counts_discriminating_pairs(
        tmp_path, monkeypatch):
    """A pair where BOTH branches passed says nothing about the ranking.
    Counting it dilutes the ratio toward the base pass rate, which is the
    §4BR wrong-statistic shape."""
    import subprocess
    import sys
    from pathlib import Path as _P
    repo = _P(__file__).resolve().parents[1]
    reg = tmp_path / "system" / "imagination" / "regret.jsonl"
    reg.parent.mkdir(parents=True)
    rows = (
        # 3 discriminating pairs, the ranking right on 2 of them
        [{"tool": "execute", "tclass": "", "chosen_outcome": "pass",
          "rejected_outcome": "fail"}] * 2
        + [{"tool": "execute", "tclass": "", "chosen_outcome": "fail",
            "rejected_outcome": "pass"}]
        # …plus 20 pairs where both branches agreed — pure dilution
        + [{"tool": "execute", "tclass": "", "chosen_outcome": "pass",
            "rejected_outcome": "pass"}] * 20
    )
    reg.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    env = dict(os.environ, GHOST_HOME=str(tmp_path),
               PYTHONPATH=str(repo / "src"))
    out = subprocess.run(
        [sys.executable, str(repo / "scripts" / "foresight_backtest.py"),
         "--consistency", "--json"],
        capture_output=True, env=env, timeout=120)
    payload = json.loads(out.stdout.decode())
    assert payload["n_graded"] == 23
    assert payload["n_discriminating"] == 3
    assert abs(payload["ratio"] - 2 / 3) < 1e-9
    # …and the point estimate is REFUSED as a verdict: 3 discordant pairs
    # are nowhere near the power the 0.70 bar needs against a 0.50 null.
    assert payload["powered"] is False
    assert payload["verdict"] == "UNDERPOWERED"
    assert out.returncode == 3


def test_the_consistency_ratio_refuses_a_verdict_without_power(
        tmp_path, monkeypatch):
    """§4CE, inside the very feature whose gate module exists because "a
    subset of 1 with precision 1.00 is not evidence". One discordant pair
    at ratio 1.000 used to exit 0 = PASS."""
    import subprocess
    import sys
    from pathlib import Path as _P
    repo = _P(__file__).resolve().parents[1]
    reg = tmp_path / "system" / "imagination" / "regret.jsonl"
    reg.parent.mkdir(parents=True)
    reg.write_text(json.dumps({"tool": "file_system", "tclass": "ext:py",
                               "chosen_outcome": "pass",
                               "rejected_outcome": "fail"}) + "\n")
    env = dict(os.environ, GHOST_HOME=str(tmp_path),
               PYTHONPATH=str(repo / "src"))
    out = subprocess.run(
        [sys.executable, str(repo / "scripts" / "foresight_backtest.py"),
         "--consistency", "--json"],
        capture_output=True, env=env, timeout=120)
    payload = json.loads(out.stdout.decode())
    assert payload["ratio"] == 1.0 and payload["powered"] is False
    assert out.returncode == 3, "a ratio of 1.000 over ONE pair read as PASS"
    assert payload["min_discordant"] >= 49


def test_the_consistency_ratio_passes_once_it_is_powered(tmp_path,
                                                         monkeypatch):
    """The other half — the bar must still be reachable."""
    import subprocess
    import sys
    from pathlib import Path as _P
    repo = _P(__file__).resolve().parents[1]
    reg = tmp_path / "system" / "imagination" / "regret.jsonl"
    reg.parent.mkdir(parents=True)
    rows = ([{"tool": "file_system", "tclass": "ext:py", "req_id": f"r{i}",
              "chosen_outcome": "pass", "rejected_outcome": "fail"}
             for i in range(45)]
            + [{"tool": "file_system", "tclass": "ext:py", "req_id": f"q{i}",
                "chosen_outcome": "fail", "rejected_outcome": "pass"}
               for i in range(10)])
    reg.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    env = dict(os.environ, GHOST_HOME=str(tmp_path),
               PYTHONPATH=str(repo / "src"))
    out = subprocess.run(
        [sys.executable, str(repo / "scripts" / "foresight_backtest.py"),
         "--consistency", "--json"],
        capture_output=True, env=env, timeout=120)
    payload = json.loads(out.stdout.decode())
    assert payload["n_discriminating"] == 55 and payload["powered"] is True
    assert payload["verdict"] == "PASS" and out.returncode == 0
    assert payload["n_clusters"] == 55        # the dependence is reported
    assert payload["ci_radius"] is not None


def test_an_armed_but_null_instrument_is_not_confused_with_an_unwired_one(
        tmp_path, monkeypatch):
    """Opposite facts. A loop or a CI job polls the exit code, not the
    prose — so they must not share one."""
    import subprocess
    import sys
    from pathlib import Path as _P
    repo = _P(__file__).resolve().parents[1]
    reg = tmp_path / "system" / "imagination" / "regret.jsonl"
    reg.parent.mkdir(parents=True)
    # 40 graded pairs, ZERO discordant: a measurement, not an absence.
    reg.write_text("\n".join(json.dumps(
        {"tool": "execute", "tclass": "", "req_id": f"r{i}",
         "chosen_outcome": "pass", "rejected_outcome": "pass"})
        for i in range(40)) + "\n")
    env = dict(os.environ, GHOST_HOME=str(tmp_path),
               PYTHONPATH=str(repo / "src"))
    armed = subprocess.run(
        [sys.executable, str(repo / "scripts" / "foresight_backtest.py"),
         "--consistency", "--json"],
        capture_output=True, env=env, timeout=120)
    assert json.loads(armed.stdout.decode())["armed"] is True
    assert armed.returncode == 3

    reg.unlink()
    unarmed = subprocess.run(
        [sys.executable, str(repo / "scripts" / "foresight_backtest.py"),
         "--consistency", "--json"],
        capture_output=True, env=env, timeout=120)
    assert json.loads(unarmed.stdout.decode())["armed"] is False
    assert unarmed.returncode == 2
    assert armed.returncode != unarmed.returncode


def test_the_consistency_instrument_refuses_a_guessed_relative_path(
        tmp_path):
    """`_regret_path` with GHOST_HOME unset is RELATIVE to cwd, and a
    go/no-go verdict computed from a stray file the operator never
    pointed at is exactly what the script's own ledger guard refuses."""
    import subprocess
    import sys
    from pathlib import Path as _P
    repo = _P(__file__).resolve().parents[1]
    stray = tmp_path / "system" / "imagination"
    stray.mkdir(parents=True)
    (stray / "regret.jsonl").write_text(json.dumps(
        {"tool": "STRAY", "tclass": "", "chosen_outcome": "fail",
         "rejected_outcome": "pass"}) + "\n")
    env = {k: v for k, v in os.environ.items() if k != "GHOST_HOME"}
    env["PYTHONPATH"] = str(repo / "src")
    out = subprocess.run(
        [sys.executable, str(repo / "scripts" / "foresight_backtest.py"),
         "--consistency"],
        capture_output=True, cwd=str(tmp_path), env=env, timeout=120)
    assert b"refusing to guess a relative path" in out.stderr
    assert b"STRAY" not in out.stdout


# ------------------------------------------------------------------ #
# R2 review — the wiring seams                                        #
# ------------------------------------------------------------------ #

def test_the_health_report_reads_the_SAME_home_as_its_other_sections(
        tmp_path, monkeypatch):
    """Every other section of the learning-health report is derived from
    `memory_dir`. Reading GHOST_HOME here made the IMAGINE GATE block
    describe a DIFFERENT home than the rest of the page — which is
    exactly the headless/archive comparison the script exists for."""
    from ghost_agent.core import learning_health as LH

    live = tmp_path / "live"
    archive = tmp_path / "archive"
    for home, rows in ((live, _qualifying()),
                       (archive, _rows(n_ok_low=5, n_fail_high=5))):
        (home / "system" / "memory").mkdir(parents=True)
        IM.reset_gate_cache_for_tests()
        IM.build_gate(rows, write=True, home=str(home))

    monkeypatch.setenv("GHOST_HOME", str(live))
    IM.reset_gate_cache_for_tests()
    report = LH.collect_learning_health(str(archive / "system" / "memory"))
    ig = report.get("imagine_gate") or {}
    assert ig.get("present") is True
    assert ig.get("enabled_count") == 0, (
        "the gate section rendered the LIVE home while the rest of the "
        "report describes the archive")


def test_a_broken_instrument_reports_itself(tmp_path, monkeypatch):
    """"Absence of the instrument is a louder fact than absence of
    activity" — a bare `except: pass` made an import failure look
    identical to a section nobody implemented."""
    from ghost_agent.core import learning_health as LH

    (tmp_path / "system" / "memory").mkdir(parents=True)
    monkeypatch.setattr(
        "ghost_agent.core.imagination.gate_stats",
        lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    report = LH.collect_learning_health(str(tmp_path / "system" / "memory"))
    ig = report.get("imagine_gate") or {}
    assert ig.get("present") is False
    assert "unavailable" in str(ig.get("reason", ""))


def test_the_cooldown_sits_well_inside_the_liveness_window():
    """`imagine_gate` is registered PERIODIC and the alarm fires on zero
    firings in a 24-HOUR window. A cooldown equal to that window
    guarantees an intermittent false alarm — and a monitor that cries on
    a benign zero gets ignored on a real one."""
    from ghost_agent.core.agent import GhostAgent

    assert GhostAgent._IMAGINE_GATE_COOLDOWN <= 21600
    # …with the same order-of-magnitude headroom its PERIODIC siblings have.
    assert GhostAgent._IMAGINE_GATE_COOLDOWN * 4 <= 86400


def test_the_cooldown_anchor_survives_a_restart(tmp_path, monkeypatch):
    """The anchor is in-memory and this process restarts often. Without
    seeding it from the gate file's own `built` stamp, every restart
    rebuilt on the first in-window tick — so the cooldown was not a rate
    limit at all and the real cadence was "once per deploy"."""
    import datetime
    from ghost_agent.core.agent import _imagine_gate_built_at

    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    assert _imagine_gate_built_at() == datetime.datetime.min   # no gate yet
    IM.build_gate(_qualifying(), write=True)
    IM.reset_gate_cache_for_tests()
    seeded = _imagine_gate_built_at()
    assert seeded > datetime.datetime.min
    assert (datetime.datetime.utcnow() - seeded).total_seconds() < 120


async def test_a_restart_does_not_immediately_rebuild(tmp_path, monkeypatch):
    """The observable half of the pin above, driven through the real tick."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    _seed_thin_ledger(tmp_path)
    ctx = _bio_ctx(idle_secs=1200)
    await _tick(ctx)                       # first boot: builds
    gate_path = tmp_path / "system" / "foresight" / IM.GATE_FILENAME
    first = gate_path.read_text()
    IM.reset_gate_cache_for_tests()
    ctx2 = _bio_ctx(idle_secs=1200)        # a FRESH agent = a restart
    await _tick(ctx2)
    assert gate_path.read_text() == first, "the restart rebuilt immediately"


def test_a_bucket_that_cannot_be_evaluated_does_not_blank_the_gate(
        tmp_path, monkeypatch):
    """One unevaluable bucket used to blank the whole document — and the
    writer then wrote the empty allow-list OVER a previously-good one, so
    a transient bug destroyed state instead of merely failing closed."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    rows = _qualifying() + _rows(tool="b", tclass="", n_ok_low=40,
                                 n_fail_high=12)
    real_eval = IM._evaluate_bucket

    def _explode(key, b, params):
        if key.startswith("b|"):
            raise TypeError("'<' not supported between NoneType and float")
        return real_eval(key, b, params)

    monkeypatch.setattr(IM, "_evaluate_bucket", _explode)
    doc = IM.build_gate(rows, write=True)
    assert doc["buckets"]["file_system|ext:py"]["enabled"] is True
    assert doc["buckets"]["b|"]["enabled"] is False
    assert "not evaluable" in doc["buckets"]["b|"]["why"]


def test_a_zero_denominator_knob_does_not_raise(tmp_path, monkeypatch):
    """`GHOST_IMAGINE_GATE_MIN_FAIL_N=0` is a documented knob, and it
    reaches a bucket whose precision is None."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    monkeypatch.setenv("GHOST_IMAGINE_GATE_MIN_FAIL_N", "0")
    doc = IM.build_gate(_rows(n_ok_low=40), write=True)
    entry = doc["buckets"]["file_system|ext:py"]
    assert entry["enabled"] is False
    assert entry["why"].startswith("no denominator")


def test_closed_reasons_do_not_fragment_on_the_precision_branch(
        tmp_path, monkeypatch):
    """The tag before the colon is what the histogram keys on.
    Interpolating the VALUE made every distinct precision its own bucket,
    so the top-N truncation hid the distribution it exists to show."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    rows = []
    for i, (ok_hi, fail_hi) in enumerate([(10, 10), (11, 9), (12, 8)]):
        rows += _rows(tool=f"t{i}", tclass="", n_ok_low=40,
                      n_ok_high=ok_hi, n_fail_high=fail_hi)
    IM.build_gate(rows, write=True)
    st = IM.gate_stats()
    assert st["enabled_count"] == 0
    assert list(st["closed_reasons"]) == ["precision too low"], \
        st["closed_reasons"]


def test_a_bucket_key_cannot_forge_an_operator_line():
    """`target_class`'s `cmd:` branch takes the command head verbatim, so
    a filename carrying an ANSI escape reached the gate file, the nightly
    operator line and the health report unfiltered."""
    key = IM.bucket_key("execute", "cmd:\x1b[2J\x07rm")
    assert "\x1b" not in key and "\x07" not in key
    assert key.startswith("execute|cmd:")


def test_the_gate_measures_the_STEERABLE_population(tmp_path, monkeypatch):
    """A row can carry a failure claim and still be one the steer will
    never act on (coarse basis, thin support, no error head). Counting
    those into the precision certifies a statistic about rows nothing
    touches — and on live data they are the MORE precise subset, so the
    gate reads better than the mechanism it authorises."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    steerable_but_wrong = [
        {"tool": "x", "tclass": "", "basis": "exact", "support": 6,
         "fails": 5, "p_fail": 0.75, "ok": True, "pred_err": "e"}] * 12
    # …and 12 rows that CLAIM failure and really failed, but on the
    # coarsest basis — the steer skips these, so the gate must too.
    claims_but_not_steerable = [
        {"tool": "x", "tclass": "", "basis": "tool", "support": 6,
         "fails": 5, "p_fail": 0.75, "ok": False, "pred_err": "e"}] * 12
    clean = [
        {"tool": "x", "tclass": "", "basis": "exact", "support": 6,
         "fails": 1, "p_fail": 0.25, "ok": True}] * 40
    doc = IM.build_gate(steerable_but_wrong + claims_but_not_steerable
                        + clean, write=True)
    entry = doc["buckets"]["x|"]
    assert entry["fail_n"] == 12, "non-steerable claims entered the gate"
    assert entry["precision"] == 0.0
    assert entry["enabled"] is False
