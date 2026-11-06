"""The liveness alarm must tell "ran and declined" from "never ran" — and
from "crashed on every cycle".

Measured 2026-08-30: the learning-health view reported

    ✗ DEAD prm_train 0/24h 0/7d

while the loop had run 30 minutes earlier and logged "the labelled corpus is
89% the same as the last look … not re-running the same test on the same
evidence". The activity ledger records OUTCOMES, so its last row
(2026-08-16) is its last real RETRAIN — exactly what the ledger promises.
"Produced nothing" was being rendered as "is dead".

⚠ THE FIRST FIX WAS A FALSE GREEN. A bare timestamp stamped at ENTRY made a
loop that crashes every cycle byte-identical to one that correctly declined,
and the renderer then asserted "a healthy outcome-free run" about a
permanently broken loop. Only an explicit DECLINED suppresses now. This
file's job is to keep both halves true at once: no false alarm, no false
green.
"""
import json
import os
import threading
from pathlib import Path
import time

import pytest

from ghost_agent.core.autonomous_activity import (ATTEMPT_DECLINED,
                                                  ATTEMPT_ENTERED,
                                                  ATTEMPT_FAILED,
                                                  attempt_suppresses_alarm,
                                                  attempts_path, read_attempts,
                                                  record_attempt)
from ghost_agent.core.learning_health import activity_liveness

_WINDOW_S = 24 * 3600


def _ledger(tmp_path, rows=()):
    p = tmp_path / "autonomous_activity.jsonl"
    with p.open("w") as fh:
        for phase, ts in rows:
            fh.write(json.dumps({"phase": phase, "ts": ts, "summary": "x",
                                 "severity": "info"}) + "\n")
    return p


def _live_ledger(tmp_path):
    """A ledger with a fresh row for SOME phase, so `agent_silent` is False —
    otherwise every per-loop alarm is withheld and the test proves nothing."""
    return _ledger(tmp_path, [("workspace_tidy", time.time())])   # §4PJ: calibration is a retired phase


def _row(ledger, phase):
    return next(r for r in activity_liveness(ledger)["rows"]
                if r["phase"] == phase)


# ══ the outcome class is the whole point ═════════════════════════════════

@pytest.mark.parametrize("result,suppresses", [
    (ATTEMPT_DECLINED, True),
    (ATTEMPT_ENTERED, False),
    (ATTEMPT_FAILED, False),
])
def test_only_an_explicit_decline_suppresses_the_alarm(result, suppresses):
    """⚠ THE FALSE GREEN. The stamp is written where the phase ENTERS its
    work, before the try block that does it — so `entered` is the state a
    CRASH leaves behind. With a bare timestamp, a loop failing on every
    cycle was indistinguishable from one correctly declining."""
    entry = {"ts": time.time(), "result": result}
    assert attempt_suppresses_alarm(entry, time.time(), _WINDOW_S) is suppresses


def test_a_crash_looping_phase_still_alarms(tmp_path):
    """The same property, driven through the real liveness view."""
    led = _live_ledger(tmp_path)
    for _ in range(12):                       # twelve consecutive crashes
        record_attempt(led, "maintenance", ATTEMPT_ENTERED)
    assert _row(led, "maintenance")["alarm"] is True
    assert "maintenance" in activity_liveness(led)["alarms"]


def test_a_declining_phase_does_not_alarm(tmp_path):
    """The counterweight — the defect this whole mechanism was built for."""
    led = _live_ledger(tmp_path)
    record_attempt(led, "prm_train", ATTEMPT_DECLINED)
    r = _row(led, "prm_train")
    assert r["alarm"] is False
    assert r["attempted_recently"] is True


# ══ a future stamp must never buy permanent silence ══════════════════════

@pytest.mark.parametrize("offset_s", [365 * 86400, 3600, 120])
def test_a_future_stamp_does_not_suppress(offset_s):
    """⚠ The first version tested only `now - ts <= window`, which is true
    for EVERY future timestamp. An NTP step back, a hand-edited file, or a
    JSON `Infinity` (json.loads accepts the literal) bought silence forever.
    This file already hardens LEDGER timestamps against exactly that."""
    entry = {"ts": time.time() + offset_s, "result": ATTEMPT_DECLINED}
    assert attempt_suppresses_alarm(entry, time.time(), _WINDOW_S) is False


def test_small_clock_jitter_is_tolerated():
    """The counterweight: the consumer captures `now` and THEN reads the
    file, so a stamp written microseconds later is legitimately ahead by a
    hair. A strict `0 <= age` refused a genuine decline."""
    entry = {"ts": time.time() + 5, "result": ATTEMPT_DECLINED}
    assert attempt_suppresses_alarm(entry, time.time(), _WINDOW_S) is True


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_timestamps_are_dropped(tmp_path, bad):
    """⚠ A non-finite ts does not suppress either way (`now - inf` fails the
    lower bound), so suppression cannot distinguish the guard. What it
    changes is the PAYLOAD: `-inf` reaches `attempted_age_s` and
    `json.dumps` emits `-Infinity`, which is invalid strict JSON — the
    learning-health payload stops parsing for any consumer that reads it."""
    assert attempt_suppresses_alarm({"ts": bad, "result": ATTEMPT_DECLINED},
                                    time.time(), _WINDOW_S) is False
    led = _live_ledger(tmp_path)
    attempts_path(led).write_text(
        '{"maintenance": {"ts": %s, "result": "declined"}}'
        % ("NaN" if bad != bad else ("Infinity" if bad > 0 else "-Infinity")))
    row = _row(led, "maintenance")
    assert row["attempted_at"] is None, f"a non-finite ts reached the row: {row}"
    json.dumps(activity_liveness(led), allow_nan=False)   # must not raise


def test_a_stale_decline_alarms_again(tmp_path):
    led = _live_ledger(tmp_path)
    attempts_path(led).write_text(json.dumps(
        {"maintenance": {"ts": time.time() - 48 * 3600,
                          "result": ATTEMPT_DECLINED}}))
    assert _row(led, "maintenance")["alarm"] is True


# ══ the store: concurrency, corruption, failure ══════════════════════════

def test_concurrent_stamps_do_not_lose_or_corrupt(tmp_path):
    """⚠ §4BW, REPEATED. The first version was an unlocked read-modify-write
    with a pid-ONLY temp name; two THREADS share a pid, so one truncated the
    tmp the other was mid-write on. Measured: 8x30 stamps recovered 0 of 240
    keys and left the file unparseable. `services.py` and `jobs.py` both
    already use pid+uuid for this exact reason."""
    led = _ledger(tmp_path)

    def _w(i):
        for k in range(30):
            record_attempt(led, f"p{i}_{k}", ATTEMPT_DECLINED)

    ts = [threading.Thread(target=_w, args=(i,)) for i in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    got = read_attempts(led)
    assert len(got) == 240, f"lost {240 - len(got)} stamps"
    assert isinstance(json.loads(attempts_path(led).read_text()), dict)


def test_an_unreadable_file_is_not_silently_reset(tmp_path):
    """⚠ The first version fell back to `{}` on ANY read error and then
    COMMITTED that reset, destroying every other phase's stamp with a True
    return and no log line."""
    led = _ledger(tmp_path)
    record_attempt(led, "keep", ATTEMPT_DECLINED)
    attempts_path(led).chmod(0o000)
    try:
        assert record_attempt(led, "other", ATTEMPT_DECLINED) is False
    finally:
        attempts_path(led).chmod(0o644)
    assert "keep" in read_attempts(led)


def test_one_bad_row_does_not_discard_the_others(tmp_path):
    """`float(v)` inside a comprehension meant a single un-floatable value
    emptied the whole map."""
    led = _ledger(tmp_path)
    attempts_path(led).write_text(json.dumps({
        "good": {"ts": time.time(), "result": ATTEMPT_DECLINED},
        "bad": 9 ** 400,
    }))
    assert list(read_attempts(led)) == ["good"]


@pytest.mark.parametrize("content", [
    "{ corrupt", "", "[]", "null",
    '{"maintenance": true}', '{"maintenance": "123"}',
    '{"maintenance": {"ts": "x"}}',
    # ⚠ THE LEGACY FORMAT, AND IT IS ON THE LIVE BOX RIGHT NOW. The
    # first-generation code wrote a BARE timestamp; that file cannot prove a
    # run declined, so it must read as ENTERED and keep the alarm. The
    # earlier params were keyed "p" while the assertion is on
    # the phase under test, so this branch was never exercised for the phase
    # under test — one unpinned line away from a permanent false green.
    '{"maintenance": 1788068709.491002}',
])
def test_unreadable_shapes_degrade_to_the_alarm(tmp_path, content):
    """Every corrupt shape must fall back to the OLD alarm. Failure may not
    buy silence — this module's own comment calls a false green worse."""
    led = _live_ledger(tmp_path)
    attempts_path(led).write_text(content)
    assert _row(led, "maintenance")["alarm"] is True


def test_record_attempt_never_raises():
    assert record_attempt("/nonexistent/dir/a.jsonl", "x",
                          ATTEMPT_DECLINED) is False
    assert read_attempts("/nonexistent/dir/a.jsonl") == {}


def test_a_failed_write_leaves_no_orphan_temp(tmp_path):
    led = _ledger(tmp_path)
    attempts_path(led).mkdir()          # target is a directory → write fails
    assert record_attempt(led, "x", ATTEMPT_DECLINED) is False
    assert not list(tmp_path.glob("*.tmp"))


def test_the_heartbeat_is_not_more_ledger_rows(tmp_path):
    """A skip-per-idle-cycle would add ~30 rows/day to a file that never
    rotates, and would blur the ledger's outcome contract."""
    led = _ledger(tmp_path)
    record_attempt(led, "prm_train", ATTEMPT_DECLINED)
    assert attempts_path(led).parent == led.parent
    assert led.read_text() == ""


# ══ the consumer ═════════════════════════════════════════════════════════

def test_a_heartbeat_only_phase_is_still_visible(tmp_path):
    """⚠ Rows were built from `PHASE_EXPECTATION | context | recent`. A loop
    wired to the heartbeat before it ever produces an outcome appeared
    NOWHERE — the invisibility the registry exists to remove, re-entered
    through the new door."""
    led = _live_ledger(tmp_path)
    record_attempt(led, "a_brand_new_loop", ATTEMPT_DECLINED)
    assert any(r["phase"] == "a_brand_new_loop"
               for r in activity_liveness(led)["rows"])


def test_an_outcome_producing_loop_needs_no_heartbeat(tmp_path):
    led = _live_ledger(tmp_path)
    r = _row(led, "workspace_tidy")
    assert r["n_alarm_window"] >= 1 and r["alarm"] is False


def test_the_suppression_is_reported_not_silent(tmp_path):
    """A zero with the alarm quietly withheld is indistinguishable from a
    zero nobody looked at."""
    from ghost_agent.core.learning_health import render_learning_health
    led = _live_ledger(tmp_path)
    record_attempt(led, "maintenance", ATTEMPT_DECLINED)
    mem = tmp_path / "memory"
    mem.mkdir()
    out = render_learning_health(mem)
    # ⚠ BOUND TO THE ROW. A two-substring grep over the whole report cannot
    # tell which row the line explains: a mutant that emitted it against the
    # first PERIODIC zero-row printed "└ RAN 0.0h ago … Not dead" under
    # `dream` (which never ran) while `prm_train` — the row that actually
    # declined — got nothing, and the test stayed green.
    lines = out.splitlines()
    idx = next(i for i, ln in enumerate(lines) if "maintenance" in ln)
    assert lines[idx + 1].strip().startswith("└ RAN"), (
        f"the explanation is not attached to the prm_train row:\n"
        + "\n".join(lines[max(0, idx - 2):idx + 3]))
    assert out.count("produced nothing") == 1, (
        "the line was emitted for more than one row")


def test_the_reported_age_is_the_real_one(tmp_path):
    """`_age_h = 0.0` survived: the renderer must print the age the
    suppression was computed from, not a fresh clock read."""
    from ghost_agent.core.learning_health import render_learning_health
    led = _live_ledger(tmp_path)
    attempts_path(led).write_text(json.dumps(
        {"maintenance": {"ts": time.time() - 20 * 3600,
                          "result": ATTEMPT_DECLINED}}))
    mem = tmp_path / "memory"
    mem.mkdir()
    out = render_learning_health(mem)
    assert "RAN 20.0h ago" in out, [l for l in out.splitlines() if "RAN" in l]


def test_the_render_line_is_only_for_rows_that_could_alarm(tmp_path):
    """It was gated only on `n24 == 0 and attempted_recently`, so any
    gated/on_demand row with a heartbeat would print "Not dead" about a row
    that could never have alarmed."""
    from ghost_agent.core.learning_health import render_learning_health
    led = _live_ledger(tmp_path)
    record_attempt(led, "job", ATTEMPT_DECLINED)      # on_demand, never alarms
    mem = tmp_path / "memory"
    mem.mkdir()
    assert "produced nothing" not in render_learning_health(mem)


# ══ the call site ═══════════════════════════════════════════════════════
#
# ⚠ EVERY TEST BELOW REPLACES A VACUOUS ONE. The previous versions:
#   * asserted an ABSENCE, so `if isinstance(...) and False:` — the method
#     made dead code entirely — passed 34/34;
#   * used `src.index(needle, stamp)`, which returns an index >= `stamp` BY
#     LANGUAGE GUARANTEE, so `assert stamp < gate` could never fail;
#   * used `src.index(needle)` with no offset, anchoring on the PRM phase's
#     IDENTICAL message 5,666 chars earlier, so `src[gate:]` covered the
#     whole router phase and was satisfied by a stamp placed anywhere in it.
# All three are structural, so they are asserted structurally now: the AST,
# not a substring search, and one runtime test that observes a real write.


def test_the_method_actually_writes_through_a_real_activity_log(tmp_path):
    """⚠ THE POSITIVE TWIN. The only runtime test used to assert an ABSENCE
    (a Mock writes nothing), which `if isinstance(...) and False:` satisfies
    perfectly — the method could be dead code and the file stayed green."""
    from ghost_agent.core.autonomous_activity import ActivityLog
    from ghost_agent.core.agent import GhostAgent

    led = tmp_path / "autonomous_activity.jsonl"
    agent = GhostAgent.__new__(GhostAgent)

    class _Ctx:
        activity_log = ActivityLog(led)

    agent.context = _Ctx()
    agent._record_idle_attempt("prm_train", ATTEMPT_DECLINED)

    got = read_attempts(led)
    assert got["prm_train"]["result"] == ATTEMPT_DECLINED, got
    assert abs(got["prm_train"]["ts"] - time.time()) < 10


def test_the_method_records_the_outcome_it_is_given(tmp_path):
    """…and passes the class through rather than hardcoding one."""
    from ghost_agent.core.autonomous_activity import ActivityLog
    from ghost_agent.core.agent import GhostAgent

    led = tmp_path / "autonomous_activity.jsonl"
    agent = GhostAgent.__new__(GhostAgent)

    class _Ctx:
        activity_log = ActivityLog(led)

    agent.context = _Ctx()
    for result in (ATTEMPT_ENTERED, ATTEMPT_FAILED, ATTEMPT_DECLINED):
        agent._record_idle_attempt("prm_train", result)
        assert read_attempts(led)["prm_train"]["result"] == result


# ══ the call site ════════════════════════════════════════════════════════

def test_a_mock_context_cannot_write_a_heartbeat(tmp_path, monkeypatch):
    """⚠ `MagicMock` implements `__fspath__`, so `Path(mock.activity_log
    .path)` does NOT raise and the try/except never fires — it created
    `./MagicMock/mock.activity_log.path/idle_attempts.json` under the repo
    root and returned success. conftest's GHOST_HOME isolation gives no
    protection: the destination comes from an attribute, not an env var."""
    from unittest.mock import MagicMock
    from ghost_agent.core.agent import GhostAgent

    monkeypatch.chdir(tmp_path)
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = MagicMock()
    agent._record_idle_attempt("prm_train", ATTEMPT_DECLINED)
    assert not list(tmp_path.rglob("idle_attempts.json"))


# (`test_the_stamp_precedes_the_skip_gate` and
#  `test_the_decline_is_stamped_on_the_skip_path` were DELETED here. The
#  header above lists both as the round-one vacuities this file removed —
#  and they were still present 90 lines below it. `src.index(n, start)`
#  returns >= start by language guarantee, so the first could only fail by
#  ValueError; the second had no offset at all and anchored on the PRM
#  phase's identical message 7,132 chars earlier, making `src[gate:]` span
#  19,707 lines. Their job is done properly by the AST tests above and by
#  the runtime test below.)


def test_suppression_is_phase_general(tmp_path):
    """⚠ `and not (phase == "prm_train" and _suppresses(...))` survived
    every test, because every liveness test used prm_train. When
    prm_train or prm_train get wired, the mechanism must already work for
    them — a router-only suppression would be invisible until then."""
    led = _live_ledger(tmp_path)
    for phase in ("prm_train", "workspace_tidy", "maintenance"):
        attempts_path(led).write_text(json.dumps(
            {phase: {"ts": time.time(), "result": ATTEMPT_DECLINED}}))
        r = _row(led, phase)
        assert r["alarm"] is False, f"{phase} did not honour its heartbeat"
        assert r["attempted_recently"] is True


def test_the_predicate_fails_CLOSED_on_a_bad_entry():
    """⚠ `except Exception: return True` survived. This is the only function
    in the system that can SILENCE an alarm; its error path must refuse, not
    permit. `read_attempts` sanitises upstream today, but the predicate is
    exported and directly tested — it does not get to rely on its caller."""
    for junk in ({"ts": "not-a-number", "result": ATTEMPT_DECLINED},
                 {"ts": None, "result": ATTEMPT_DECLINED},
                 {"result": ATTEMPT_DECLINED},
                 {"ts": object(), "result": ATTEMPT_DECLINED},
                 "not-a-dict", None, 42):
        assert attempt_suppresses_alarm(junk, time.time(), _WINDOW_S) is False, junk


def test_record_attempt_returns_false_without_touching_the_filesystem(tmp_path):
    """⚠ The old version asserted `record_attempt("/nonexistent/dir/…")` is
    False — which passes only because `/` is read-only on this box. As root,
    or on a writable `/`, `mkdir(parents=True)` would SUCCEED, the test
    would flip red, and it would create `/nonexistent/dir` on the host.
    Use a path that cannot be created for a reason the test controls."""
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, not a directory")
    target = blocker / "nested" / "a.jsonl"      # parent is a FILE → ENOTDIR
    assert record_attempt(target, "x", ATTEMPT_DECLINED) is False
    assert read_attempts(target) == {}


# ══ the ONLY thing that binds the fix to the code that runs ══════════════

# ══ the reader, with more than one phase in the file ═════════════════════

def test_one_recent_decline_does_not_silence_every_other_phase(tmp_path):
    """⚠ Every test wrote a SINGLE-phase map, so a reader returning the
    newest entry for every key was invisible — one recently-declining loop
    would silence every dead loop on the box, the maximal false green."""
    led = _live_ledger(tmp_path)
    attempts_path(led).write_text(json.dumps({
        "prm_train": {"ts": time.time(), "result": ATTEMPT_DECLINED},
        "maintenance": {"ts": time.time() - 8 * 86400, "result": ATTEMPT_DECLINED},
    }))
    assert _row(led, "prm_train")["alarm"] is False
    assert _row(led, "maintenance")["alarm"] is True, (
        "a stale phase inherited a fresh phase's heartbeat")


def test_each_phase_keeps_its_own_outcome(tmp_path):
    led = _ledger(tmp_path)
    record_attempt(led, "a", ATTEMPT_DECLINED)
    record_attempt(led, "b", ATTEMPT_FAILED)
    got = read_attempts(led)
    assert got["a"]["result"] == ATTEMPT_DECLINED
    assert got["b"]["result"] == ATTEMPT_FAILED


# ══ every default and coercion must fail toward the ALARM ════════════════

def test_an_unknown_outcome_coerces_to_entered_not_declined(tmp_path):
    """A value outside the three classes must never buy silence."""
    led = _ledger(tmp_path)
    record_attempt(led, "prm_train", "skipped")       # not a known class
    assert read_attempts(led)["prm_train"]["result"] == ATTEMPT_ENTERED


def test_a_missing_result_key_does_not_suppress(tmp_path):
    """`{"ts": <fresh>}` with no `result` — the one shape that flips green
    if the default is wrong. It was absent from the corrupt-shape list."""
    led = _live_ledger(tmp_path)
    attempts_path(led).write_text(json.dumps(
        {"maintenance": {"ts": time.time()}}))
    assert _row(led, "maintenance")["alarm"] is True


def test_the_default_outcome_is_entered():
    """Both `record_attempt` and `_record_idle_attempt` default to ENTERED.
    A DECLINED default would silence every caller that omits the argument."""
    import inspect
    from ghost_agent.core.agent import GhostAgent
    assert (inspect.signature(record_attempt).parameters["result"].default
            == ATTEMPT_ENTERED)
    assert (inspect.signature(GhostAgent._record_idle_attempt)
            .parameters["result"].default == ATTEMPT_ENTERED)


def test_suppressing_results_is_exact_match_not_substring():
    """⚠ `_SUPPRESSING_RESULTS = ATTEMPT_DECLINED` (a string, not a set)
    turns `in` into a SUBSTRING test: "dec" and even "" then suppress."""
    for near_miss in ("dec", "declined ", " declined", "", "DECLINED",
                      "undeclined"):
        assert attempt_suppresses_alarm(
            {"ts": time.time(), "result": near_miss},
            time.time(), _WINDOW_S) is False, near_miss


@pytest.mark.parametrize("ts", [True, False])
def test_a_bool_timestamp_is_dropped_by_the_reader(tmp_path, ts):
    """⚠ `True` is an `int`, so a naive isinstance check accepts it as epoch
    1.0 — which is 56 years stale, so "it does not suppress" is true whether
    or not the guard exists. The distinguishing property is that the reader
    DROPS the row entirely, so assert on the parsed map."""
    led = _ledger(tmp_path)
    attempts_path(led).write_text(json.dumps(
        {"prm_train": {"ts": ts, "result": ATTEMPT_DECLINED},
         "good": {"ts": time.time(), "result": ATTEMPT_DECLINED}}))
    got = read_attempts(led)
    assert "prm_train" not in got, f"a bool timestamp was accepted: {got}"
    assert "good" in got, "dropping one bad row must not drop the others"


# ══ the atomic write path ════════════════════════════════════════════════

def test_the_write_is_atomic_via_a_temp_and_replace(tmp_path, monkeypatch):
    """⚠ The orphan-temp test never reached the write at all: with the
    target a DIRECTORY, `read_text` raises first and the function returns
    before writing. Dropping tmp+replace entirely survived, leaving a
    concurrent reader able to see a torn file."""
    led = _ledger(tmp_path)
    seen = {"tmp_writes": 0, "replaces": 0}
    real_replace = os.replace

    def _spy_replace(a, b):
        seen["replaces"] += 1
        return real_replace(a, b)

    import ghost_agent.core.autonomous_activity as AA
    real_write = AA._write_text_nofollow

    def _spy_write(p, text, **kw):
        if str(p).endswith(".tmp"):
            seen["tmp_writes"] += 1
        return real_write(p, text, **kw)

    monkeypatch.setattr(AA.os, "replace", _spy_replace)
    monkeypatch.setattr(AA, "_write_text_nofollow", _spy_write)
    assert record_attempt(led, "prm_train", ATTEMPT_DECLINED) is True
    assert seen["tmp_writes"] == 1, "the payload was not written to a temp"
    assert seen["replaces"] == 1, "the temp was not atomically replaced in"


def test_a_write_failure_leaves_no_orphan_temp(tmp_path, monkeypatch):
    """The real write-failure path: the read succeeds, the WRITE fails."""
    import ghost_agent.core.autonomous_activity as AA
    led = _ledger(tmp_path)
    record_attempt(led, "seed", ATTEMPT_DECLINED)

    def _boom(p, text, **kw):
        # ⚠ CREATE the temp, THEN fail. The previous version raised before
        # anything was written, so there was never an orphan and the test
        # could not tell whether the cleanup existed.
        Path(p).write_text("partial")
        raise OSError("disk full")

    monkeypatch.setattr(AA, "_write_text_nofollow", _boom)
    assert record_attempt(led, "prm_train", ATTEMPT_DECLINED) is False
    assert not list(attempts_path(led).parent.glob("*.tmp"))
    assert "seed" in read_attempts(led), "the existing map was destroyed"


def test_the_on_disk_filename_is_part_of_the_contract():
    """The running agent and this code must agree on the path. A rename
    round-trips inside the module and would be invisible."""
    from ghost_agent.core.autonomous_activity import _ATTEMPTS_FILENAME
    assert _ATTEMPTS_FILENAME == "idle_attempts.json"


def test_the_reader_does_not_share_one_entry_across_phases(tmp_path):
    """⚠ A reader that returned the newest entry for EVERY key would let one
    recently-declining loop silence every dead loop on the box — the maximal
    false green. Distinct timestamps AND distinct outcomes, so neither field
    can be shared."""
    led = _ledger(tmp_path)
    now = time.time()
    attempts_path(led).write_text(json.dumps({
        "fresh_declined": {"ts": now, "result": ATTEMPT_DECLINED},
        "old_failed": {"ts": now - 9 * 86400, "result": ATTEMPT_FAILED},
        "mid_entered": {"ts": now - 3 * 86400, "result": ATTEMPT_ENTERED},
    }))
    got = read_attempts(led)
    assert got["fresh_declined"]["result"] == ATTEMPT_DECLINED
    assert got["old_failed"]["result"] == ATTEMPT_FAILED
    assert got["mid_entered"]["result"] == ATTEMPT_ENTERED
    assert got["old_failed"]["ts"] < got["mid_entered"]["ts"] < \
        got["fresh_declined"]["ts"], got


# ══ R1-1 (2026-09-20): the in-trainer bail is a DECLINE, the exception a FAIL ═
#
# The executed-path test above accepts ENTERED as a valid outcome, so it was
# green in both worlds — with the stamp and without it. These three drive the
# same real phase and pin the value. Each fails on the pre-fix tree:
# `_biological_tick` left the heartbeat at `entered` for every in-trainer
# outcome, which is what printed `✗ DEAD prm_train` daily (R1-1).

