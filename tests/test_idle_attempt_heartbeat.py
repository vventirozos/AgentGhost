"""The liveness alarm must tell "ran and declined" from "never ran" — and
from "crashed on every cycle".

Measured 2026-08-30: the learning-health view reported

    ✗ DEAD router_train 0/24h 0/7d

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
    return _ledger(tmp_path, [("calibration", time.time())])


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
        record_attempt(led, "router_train", ATTEMPT_ENTERED)
    assert _row(led, "router_train")["alarm"] is True
    assert "router_train" in activity_liveness(led)["alarms"]


def test_a_declining_phase_does_not_alarm(tmp_path):
    """The counterweight — the defect this whole mechanism was built for."""
    led = _live_ledger(tmp_path)
    record_attempt(led, "router_train", ATTEMPT_DECLINED)
    r = _row(led, "router_train")
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
        '{"router_train": {"ts": %s, "result": "declined"}}'
        % ("NaN" if bad != bad else ("Infinity" if bad > 0 else "-Infinity")))
    row = _row(led, "router_train")
    assert row["attempted_at"] is None, f"a non-finite ts reached the row: {row}"
    json.dumps(activity_liveness(led), allow_nan=False)   # must not raise


def test_a_stale_decline_alarms_again(tmp_path):
    led = _live_ledger(tmp_path)
    attempts_path(led).write_text(json.dumps(
        {"router_train": {"ts": time.time() - 48 * 3600,
                          "result": ATTEMPT_DECLINED}}))
    assert _row(led, "router_train")["alarm"] is True


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
    '{"router_train": true}', '{"router_train": "123"}',
    '{"router_train": {"ts": "x"}}',
    # ⚠ THE LEGACY FORMAT, AND IT IS ON THE LIVE BOX RIGHT NOW. The
    # first-generation code wrote a BARE timestamp; that file cannot prove a
    # run declined, so it must read as ENTERED and keep the alarm. The
    # earlier params were keyed "p" while the assertion is on
    # "router_train", so this branch was never exercised for the phase
    # under test — one unpinned line away from a permanent false green.
    '{"router_train": 1788068709.491002}',
])
def test_unreadable_shapes_degrade_to_the_alarm(tmp_path, content):
    """Every corrupt shape must fall back to the OLD alarm. Failure may not
    buy silence — this module's own comment calls a false green worse."""
    led = _live_ledger(tmp_path)
    attempts_path(led).write_text(content)
    assert _row(led, "router_train")["alarm"] is True


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
    record_attempt(led, "router_train", ATTEMPT_DECLINED)
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
    r = _row(led, "calibration")
    assert r["n_alarm_window"] >= 1 and r["alarm"] is False


def test_negative_controls_is_not_judged_on_the_wrong_cadence(tmp_path):
    """⚠ THE SECOND FALSE ALARM, and an earlier version of this file pinned
    it as CORRECT. `negative_controls` runs on a SEVEN DAY interval
    (evolve/negative_controls.py), so EXPECT_PERIODIC against a 24h window
    brands a healthy weekly loop DEAD six days in seven. (The phase does
    have row-writing paths; what is measured is that 0 rows exist across
    4,405 ledger rows / ~45 days while its state file shows a run 6.4 days
    ago — so the ledger cannot serve as its liveness signal.) Its real signal is its own state file, which
    `liveness.py::_negative_controls_probe` already reads."""
    from ghost_agent.core.autonomous_activity import (EXPECT_PERIODIC,
                                                      phase_expectation)
    assert phase_expectation("negative_controls") != EXPECT_PERIODIC
    led = _live_ledger(tmp_path)
    assert "negative_controls" not in activity_liveness(led)["alarms"]


def test_the_suppression_is_reported_not_silent(tmp_path):
    """A zero with the alarm quietly withheld is indistinguishable from a
    zero nobody looked at."""
    from ghost_agent.core.learning_health import render_learning_health
    led = _live_ledger(tmp_path)
    record_attempt(led, "router_train", ATTEMPT_DECLINED)
    mem = tmp_path / "memory"
    mem.mkdir()
    out = render_learning_health(mem)
    # ⚠ BOUND TO THE ROW. A two-substring grep over the whole report cannot
    # tell which row the line explains: a mutant that emitted it against the
    # first PERIODIC zero-row printed "└ RAN 0.0h ago … Not dead" under
    # `dream` (which never ran) while `router_train` — the row that actually
    # declined — got nothing, and the test stayed green.
    lines = out.splitlines()
    idx = next(i for i, ln in enumerate(lines) if "router_train" in ln)
    assert lines[idx + 1].strip().startswith("└ RAN"), (
        f"the explanation is not attached to the router_train row:\n"
        + "\n".join(lines[max(0, idx - 2):idx + 3]))
    assert out.count("produced nothing") == 1, (
        "the line was emitted for more than one row")


def test_the_reported_age_is_the_real_one(tmp_path):
    """`_age_h = 0.0` survived: the renderer must print the age the
    suppression was computed from, not a fresh clock read."""
    from ghost_agent.core.learning_health import render_learning_health
    led = _live_ledger(tmp_path)
    attempts_path(led).write_text(json.dumps(
        {"router_train": {"ts": time.time() - 20 * 3600,
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


def _router_stamp_calls():
    """(lineno, literal) for every `_record_idle_attempt` call in agent.py,
    plus the line of the router skip gate — from the AST, so a call inside a
    COMMENT cannot supply the literal."""
    import ast
    import inspect
    from ghost_agent.core import agent as _agent

    src = inspect.getsource(_agent)
    tree = ast.parse(src)
    calls = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and getattr(node.func, "attr", None) == "_record_idle_attempt"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "router_train"):
            literal = (node.args[1].value
                       if len(node.args) > 1
                       and isinstance(node.args[1], ast.Constant)
                       else None)
            calls.append((node.lineno, literal))
    lines = src.splitlines()
    # The ROUTER gate, not the PRM one: the last occurrence, which follows
    # the first stamp.
    gate_lines = [n for n, ln in enumerate(lines, 1)
                  if "trajectory corpus unchanged since last refit" in ln]
    assert gate_lines, "the router skip gate moved or was renamed"
    first_stamp = min(n for n, _ in calls)
    gate = next(n for n in gate_lines if n > first_stamp)
    return sorted(calls), gate


def test_entered_is_stamped_before_the_gate_and_decline_after_it():
    """The ordering that makes the heartbeat mean anything: a run is marked
    ENTERED when it starts, and only the skip path may mark it DECLINED. A
    `declined` stamped before the gate would suppress the alarm for a phase
    that then crashed."""
    calls, gate = _router_stamp_calls()
    entered = [n for n, lit in calls if lit == "entered"]
    declined = [n for n, lit in calls if lit == "declined"]
    failed = [n for n, lit in calls if lit == "failed"]

    assert len(entered) == 1, f"expected exactly one entered stamp, got {entered}"
    assert len(declined) == 1, (
        f"expected exactly one declined stamp, got {declined} — a second one "
        "before the gate restores the false green")
    assert len(failed) == 1, f"expected exactly one failed stamp, got {failed}"

    assert entered[0] < gate, "the entered stamp must precede the skip gate"
    assert declined[0] > gate, (
        "the declined stamp must be ON the skip path — before the gate it "
        "marks a crash as a healthy decline")
    assert failed[0] > gate


def test_no_call_site_passes_an_unexpected_outcome():
    """A literal outside the three known classes would silently coerce to
    ENTERED and never suppress — a false ALARM this time, and invisible."""
    calls, _ = _router_stamp_calls()
    assert calls, "no router heartbeat call sites found at all"
    for lineno, lit in calls:
        assert lit in (ATTEMPT_ENTERED, ATTEMPT_DECLINED, ATTEMPT_FAILED), (
            f"agent.py:{lineno} passes {lit!r}")


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
    agent._record_idle_attempt("router_train", ATTEMPT_DECLINED)

    got = read_attempts(led)
    assert got["router_train"]["result"] == ATTEMPT_DECLINED, got
    assert abs(got["router_train"]["ts"] - time.time()) < 10


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
        agent._record_idle_attempt("router_train", result)
        assert read_attempts(led)["router_train"]["result"] == result


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
    agent._record_idle_attempt("router_train", ATTEMPT_DECLINED)
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
    """⚠ `and not (phase == "router_train" and _suppresses(...))` survived
    every test, because every liveness test used router_train. When
    prm_train or self_play get wired, the mechanism must already work for
    them — a router-only suppression would be invisible until then."""
    led = _live_ledger(tmp_path)
    for phase in ("prm_train", "self_play", "dream", "skills_auto"):
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

@pytest.mark.asyncio
async def test_the_router_idle_phase_writes_a_declined_heartbeat(tmp_path,
                                                                monkeypatch):
    """⚠ EVERY OTHER CALL-SITE TEST IS SATISFIABLE BY DEAD CODE.

    The AST tests above collect `_record_idle_attempt("router_train", …)`
    calls from anywhere in the module and order them by LINE NUMBER. A
    reviewer demonstrated the consequence: delete all three real stamps from
    the router phase, add one never-called decoy method and two `if False:`
    blocks, and every assertion still passes while the phase writes NO
    heartbeat at all — the original `✗ DEAD router_train` bug, restored,
    with the suite green.

    Nothing in the test tree drove the router idle phase. This does: a real
    `ActivityLog`, a real `TrajectoryCollector` and a real
    `ComplexityDispatcher`, an idle time inside the router band
    (900 < idle <= 3600), and an unchanged corpus so the phase takes its
    SKIP path — the one that must leave `declined`.
    """
    import datetime
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from ghost_agent.core.agent import GhostAgent
    from ghost_agent.core.autonomous_activity import ActivityLog
    from ghost_agent.distill.collector import TrajectoryCollector
    from ghost_agent.router import ComplexityDispatcher

    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    led = tmp_path / "system" / "autonomous_activity.jsonl"
    led.parent.mkdir(parents=True, exist_ok=True)

    ctx = MagicMock()
    ctx.activity_log = ActivityLog(led)
    ctx.trajectory_collector = TrajectoryCollector(
        root=tmp_path / "system" / "trajectories")
    ctx.complexity_dispatcher = ComplexityDispatcher(
        classifier=None, confidence_threshold=0.3, disabled=True)
    ctx.memory_system = MagicMock()
    ctx.memory_system.collection.get = MagicMock(return_value={"ids": []})
    ctx.llm_client = SimpleNamespace(foreground_tasks=0)
    ctx.journal = None
    ctx.frontier_tracker = None
    ctx.reflector = None
    ctx.prm_scorer = None
    ctx.postmortem_engine = None
    ctx.calibration_tracker = None
    ctx.last_activity_time = (datetime.datetime.now()
                              - datetime.timedelta(seconds=1200))
    ctx.args = MagicMock()
    ctx.args.model = "test-model"
    for k in ("prm_train_cooldown", "router_train_cooldown",
              "self_narrative_cooldown", "calib_refit_cooldown"):
        setattr(ctx.args, k, None)

    agent = GhostAgent.__new__(GhostAgent)
    agent.context = ctx
    await agent._biological_tick()

    got = read_attempts(led)
    assert "router_train" in got, (
        "the router idle phase wrote NO heartbeat — the stamps are not on "
        f"the executed path. attempts={got}")
    assert got["router_train"]["result"] in (ATTEMPT_DECLINED,
                                             ATTEMPT_ENTERED,
                                             ATTEMPT_FAILED)


# ══ the reader, with more than one phase in the file ═════════════════════

def test_one_recent_decline_does_not_silence_every_other_phase(tmp_path):
    """⚠ Every test wrote a SINGLE-phase map, so a reader returning the
    newest entry for every key was invisible — one recently-declining loop
    would silence every dead loop on the box, the maximal false green."""
    led = _live_ledger(tmp_path)
    attempts_path(led).write_text(json.dumps({
        "router_train": {"ts": time.time(), "result": ATTEMPT_DECLINED},
        "dream": {"ts": time.time() - 8 * 86400, "result": ATTEMPT_DECLINED},
    }))
    assert _row(led, "router_train")["alarm"] is False
    assert _row(led, "dream")["alarm"] is True, (
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
    record_attempt(led, "router_train", "skipped")       # not a known class
    assert read_attempts(led)["router_train"]["result"] == ATTEMPT_ENTERED


def test_a_missing_result_key_does_not_suppress(tmp_path):
    """`{"ts": <fresh>}` with no `result` — the one shape that flips green
    if the default is wrong. It was absent from the corrupt-shape list."""
    led = _live_ledger(tmp_path)
    attempts_path(led).write_text(json.dumps(
        {"router_train": {"ts": time.time()}}))
    assert _row(led, "router_train")["alarm"] is True


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
        {"router_train": {"ts": ts, "result": ATTEMPT_DECLINED},
         "good": {"ts": time.time(), "result": ATTEMPT_DECLINED}}))
    got = read_attempts(led)
    assert "router_train" not in got, f"a bool timestamp was accepted: {got}"
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
    assert record_attempt(led, "router_train", ATTEMPT_DECLINED) is True
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
    assert record_attempt(led, "router_train", ATTEMPT_DECLINED) is False
    assert not list(attempts_path(led).parent.glob("*.tmp"))
    assert "seed" in read_attempts(led), "the existing map was destroyed"


def test_the_on_disk_filename_is_part_of_the_contract():
    """The running agent and this code must agree on the path. A rename
    round-trips inside the module and would be invisible."""
    from ghost_agent.core.autonomous_activity import _ATTEMPTS_FILENAME
    assert _ATTEMPTS_FILENAME == "idle_attempts.json"


def test_the_decline_is_inside_the_skip_BRANCH_not_merely_after_it():
    """⚠ LINE ORDER IS NOT CONTROL FLOW.

    `declined[0] > gate` is satisfied by moving the decline into the RETRAIN
    branch — which inverts the meaning twice: the skip path then leaves only
    `entered` (the false alarm returns) and a SUCCESSFUL refit stamps
    `declined`, so the view prints "RAN and produced nothing" about a loop
    that just fitted a classifier.

    So the decline is bound to the branch that logs the skip, by AST: same
    `if`/`else` body, not merely a larger line number. The runtime test
    above cannot reach this path — the skip needs an unchanged corpus
    fingerprint across two real refits.
    """
    import ast
    import inspect
    from ghost_agent.core import agent as _agent

    src = inspect.getsource(_agent)
    tree = ast.parse(src)

    def _has_skip_log(node):
        return any(isinstance(n, ast.Constant)
                   and isinstance(n.value, str)
                   and "unchanged since last refit" in n.value
                   for n in ast.walk(node))

    def _declined_calls(node):
        return [n.lineno for n in ast.walk(node)
                if isinstance(n, ast.Call)
                and getattr(n.func, "attr", None) == "_record_idle_attempt"
                and len(n.args) > 1
                and isinstance(n.args[1], ast.Constant)
                and n.args[1].value == ATTEMPT_DECLINED
                and isinstance(n.args[0], ast.Constant)
                and n.args[0].value == "router_train"]

    def _mentions_router(node):
        return any(isinstance(n, ast.Constant) and n.value == "router_train"
                   for n in ast.walk(node))

    # ⚠ DISAMBIGUATE BY THE ROUTER. The PRM phase carries the IDENTICAL skip
    # message and its block is SMALLER (66 lines vs 79), so "the smallest
    # block containing the skip log" selects the wrong phase — the same
    # identical-string trap that made two earlier versions of these tests
    # vacuous. Require the block to mention `router_train` too.
    blocks = [n for n in ast.walk(tree)
              if isinstance(n, (ast.If, ast.Try))
              and _has_skip_log(n) and _mentions_router(n)]
    assert blocks, "the router skip branch moved or was renamed"
    smallest = min(blocks, key=lambda n: (n.end_lineno or 0) - n.lineno)

    inside = _declined_calls(smallest)
    everywhere = _declined_calls(tree)
    assert inside, (
        "no `declined` stamp inside the branch that logs the skip — the "
        "skip path leaves only `entered` and the false alarm returns")
    assert len(everywhere) == len(inside), (
        f"a `declined` stamp exists OUTSIDE the skip branch at lines "
        f"{sorted(set(everywhere) - set(inside))} — a successful retrain "
        "would be reported as a healthy no-op")


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
