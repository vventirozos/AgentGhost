"""E3 — the controls that prove the cascade still refuses.

⚠ A GUARD THAT NEVER DEMONSTRABLY FIRES IS PRESUMED DEAD. E2's entire
value is rejection, and none of today's refusals are evidence next month.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ghost_agent.evolve import negative_controls as NC   # noqa: E402
from ghost_agent.evolve import evaluator as EV           # noqa: E402

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def real_run(tmp_path_factory):
    """ONE real control run, shared by every test that only inspects it.

    ⚠ THIS IS NOT PREMATURE OPTIMISATION. The guard control runs stage 1
    for real, and stage 1 now maps `tools/execute.py` to 45 pin files
    (753 tests, ~125 s) because coverage is the UNION of the filename
    convention and import evidence. Seventeen independent calls in this
    file cost over half an hour, which is how a suite stops being run.
    Tests that monkeypatch `stage1_pins` do their own call — those are
    fast precisely because the real stage never executes.
    """
    home = tmp_path_factory.mktemp("nc_shallow")
    return NC.run_negative_controls(REPO, home), home


@pytest.fixture(scope="session")
def real_deep_run(tmp_path_factory):
    """⚠ `only=` BECAUSE DEEP MODE ONLY DIFFERS FOR THE NO-OP CONTROL.
    Running all three here paid 125 s to re-derive two results the
    shallow fixture already has. The suite is run several times per
    change; a check nobody can afford to run is a check that stops
    running."""
    home = tmp_path_factory.mktemp("nc_deep")
    return NC.run_negative_controls(
        REPO, home, deep=True,
        only=("no_op_claiming_improvement",)), home


def test_the_controls_are_GENERATED_against_the_real_tree():
    """A stored diff rots the moment the file it patches moves, and a
    control that no longer applies reports a cascade failure that is
    really a stale fixture."""
    d = NC.build_controls(REPO)
    assert set(d) == {"edits_a_test", "deletes_a_guard",
                      "no_op_claiming_improvement"}
    for name, diff in d.items():
        assert diff.startswith("--- a/"), name
        assert "@@" in diff, name


def test_the_guard_control_targets_a_guard_a_PIN_ACTUALLY_COVERS():
    """⚠ THE CONTROL IS ONLY AS GOOD AS ITS PIN. The obvious candidate —
    the daemon-block guard — is USELESS here: its only pin asserts
    `"SYSTEM BLOCK" not in result`, the negative case, so deleting the
    guard passes. A control built on an unpinned guard would report the
    cascade broken every month while nothing was wrong."""
    src = (REPO / NC.GUARD_TARGET).read_text()
    assert NC.GUARD_MARK in src, "the guard must still exist to delete"
    # ⚠ The guard's text and the pin's assertion are DIFFERENT strings
    # (`"SYSTEM TIP: …"` vs `"It looks like …"`); the control is valid
    # only while they overlap, so both halves are checked.
    assert NC.GUARD_PIN_MARK in NC.GUARD_MARK, "the marks must overlap"
    covered = [p for p in (REPO / "tests").glob("test_execute*.py")
               if NC.GUARD_PIN_MARK in p.read_text()]
    assert covered, "no pin asserts this guard's message — pick another"


def test_the_guard_control_diff_REMOVES_the_guard():
    d = NC.build_controls(REPO)["deletes_a_guard"]
    removed = [l for l in d.splitlines()
               if l.startswith("-") and NC.GUARD_MARK in l]
    assert removed, d[:400]


def test_the_test_editing_control_targets_an_IMMUTABLE_path():
    from ghost_agent.evolve import fence
    ok, why = fence.is_mutable(NC.TEST_TARGET)
    assert not ok, "the control must aim at something the fence refuses"


@pytest.mark.slow
def test_all_three_controls_are_REFUSED_at_the_right_stage(real_run):
    """The load-bearing test. Each control asserts the rejection AND the
    stage — a candidate refused for the wrong reason is a cascade that
    will refuse the right thing for the wrong reason later."""
    run = real_run[0]; tmp_path = real_run[1]
    by = {r.name: r for r in run.results}
    assert by["edits_a_test"].rejected_at == EV.STAGE_STATIC
    assert by["deletes_a_guard"].rejected_at == EV.STAGE_PINS
    assert by["no_op_claiming_improvement"].rejected_at == EV.STAGE_PAIRED
    assert run.all_ok, [vars(r) for r in run.results]


@pytest.mark.slow
def test_a_control_the_pins_LET_THROUGH_is_flagged_NOT_green(
        tmp_path, monkeypatch):
    """⚠ THE PREVIOUS VERSION COULD NOT FAIL. It branched on `gd.ok`, and
    in a healthy repo `ok` is always True, so the branch asserting the
    loud message was dead — reverting the fix produced a byte-identical
    detail. Force the failure mode instead."""
    monkeypatch.setattr(EV, "stage1_pins",
                        lambda *a, **k: EV.StageResult(EV.STAGE_PINS, True,
                                                       "", {}, 0.0))
    run = NC.run_negative_controls(REPO, tmp_path)
    gd = [r for r in run.results if r.name == "deletes_a_guard"][0]
    assert gd.ok is False and gd.rejected is False and gd.rejected_at == ""
    assert "THE PINS PASSED" in gd.detail, gd.detail
    assert run.all_ok is False


@pytest.mark.slow
def test_a_control_refused_for_the_WRONG_REASON_is_not_a_pass(
        tmp_path, monkeypatch):
    """A stage-1 rejection that is not the PINS — "no pin covers …", say
    — means the pins never ran. Scoring that green certifies nothing."""
    monkeypatch.setattr(EV, "stage1_pins",
                        lambda *a, **k: EV.StageResult(
                            EV.STAGE_PINS, False, "no pin covers x.py",
                            {}, 0.0))
    run = NC.run_negative_controls(REPO, tmp_path)
    gd = [r for r in run.results if r.name == "deletes_a_guard"][0]
    assert gd.rejected is True, "it WAS rejected…"
    assert gd.ok is False, "…but not by the pins, so it is not a pass"


@pytest.mark.slow
def test_DEEP_mode_does_not_manufacture_a_false_red(real_deep_run):
    """⚠ `deep=True` left `ok=False` with a detail that read like a plan,
    so every deep run reported `all_ok=False` — a guaranteed false red
    trains an operator to ignore the whole suite."""
    run = real_deep_run[0]; tmp_path = real_deep_run[1]
    noop = [r for r in run.results
            if r.name == "no_op_claiming_improvement"][0]
    assert "NOT RUN" in noop.detail and "UNVERIFIED" in noop.detail
    assert noop.ok is True, "an unimplemented check must not read as failed"


@pytest.mark.slow
def test_the_shallow_no_op_report_IS_LABELLED_a_proxy(real_run):
    """The docstring contains the word PROXY, so a source-text check
    passes even if the label is stripped from the REPORT. The persisted
    record is what a health view renders."""
    tmp_path = real_run[1]
    got = NC.last_run(tmp_path)
    assert got["results"][2]["detail"].startswith("PROXY"), got["results"][2]


@pytest.mark.slow
def test_the_control_does_not_LEAK_a_snapshot(real_run):
    """⚠ `sweep_work_dirs` keeps only MAX_KEPT_SNAPSHOTS, so a control
    leaking ~10 MB per run EVICTS REAL CANDIDATES — the suite that
    checks the loop would quietly destroy the loop's own work."""
    tmp_path = real_run[1]
    work = tmp_path / "system" / "evolve" / "work"
    left = [d for d in work.iterdir()] if work.is_dir() else []
    assert not left, f"the control left {len(left)} snapshot(s): {left}"


@pytest.mark.slow
def test_UNUSED_the_guard_control_fails_LOUDLY(real_run):
    """⚠ The first version reported "pins failed as required" for a
    control the pins had just LET THROUGH — `reason` is empty on success
    and the `or` fallback fired. The report contradicted its own verdict
    field, which is worse than no report."""
    run = real_run[0]; tmp_path = real_run[1]
    gd = [r for r in run.results if r.name == "deletes_a_guard"][0]
    if gd.ok:
        assert "pins failed" in gd.detail.lower(), gd.detail
    else:
        assert "THE PINS PASSED" in gd.detail, gd.detail


@pytest.mark.slow
def test_the_result_is_PERSISTED_for_the_health_view(real_run):
    run, tmp_path = real_run
    got = NC.last_run(tmp_path)
    assert got and "all_ok" in got and got["results"]
    assert (tmp_path / "system" / "evolve" / NC.RESULT_FILE).is_file()
    # ⚠ CHECKING THAT KEYS EXIST IS NOT CHECKING THE RECORD. `_write`
    # hard-coding `"all_ok": True` — the exact way a health view starts
    # lying — satisfied the three assertions above. Compare the values.
    got = NC.last_run(tmp_path)
    assert got["all_ok"] == run.all_ok
    assert [r["name"] for r in got["results"]] == [r.name for r in run.results]
    assert [r["ok"] for r in got["results"]] == [r.ok for r in run.results]
    assert [r["rejected_at"] for r in got["results"]] == \
        [r.rejected_at for r in run.results]


def test_the_persisted_record_carries_a_RED_run_too(tmp_path, monkeypatch):
    """⚠ COMPARING `got["all_ok"] == run.all_ok` ON A HEALTHY REPO IS
    IDENTICAL-UNDER-BOTH: both are True, so `_write` hard-coding
    `"all_ok": True` survived it. The comparison only means something on
    a run that is actually red."""
    monkeypatch.setattr(EV, "stage1_pins",
                        lambda *a, **k: EV.StageResult(EV.STAGE_PINS, True,
                                                       "", {}, 0.0))
    run = NC.run_negative_controls(REPO, tmp_path)
    assert run.all_ok is False, "the fixture failed to produce a red run"
    got = NC.last_run(tmp_path)
    assert got["all_ok"] is False, \
        "the record says the suite is green while the run says it is not"
    assert [r["ok"] for r in got["results"]] == [r.ok for r in run.results]


def test_NEVER_RUN_is_not_the_same_as_PASSING(tmp_path):
    """`None` must not render as a tick. A control suite that has never
    run is exactly the silent-inoperative-subsystem case this exists to
    prevent. (Deliberately NOT on the shared fixture: this one needs a
    home nothing has written to, which is the whole point.)"""
    assert NC.last_run(tmp_path) is None
    run = NC.ControlRun()
    assert run.all_ok is False, "an empty run must not report all_ok"


def test_the_shallow_no_op_check_SAYS_it_is_a_proxy(real_run):
    """Shallow mode checks the DECISION function, not the full stage.
    That is a proxy and must be labelled as one, or a green month
    implies stages 2-3 ran when they did not."""
    # ⚠ `inspect.getsource` INCLUDES THE DOCSTRING, and the docstring
    # says "that is a PROXY and is recorded as one" — so this test was
    # satisfied by its own subject's prose. Deleting the only thing that
    # actually labels the report left it green. Read the report.
    run = real_run[0]; tmp_path = real_run[1]
    noop = [r for r in run.results if r.name == "no_op_claiming_improvement"][0]
    assert noop.detail.startswith("PROXY"), noop.detail
    # …and the PERSISTED record carries the label too, since that is what
    # any health view renders.
    got = NC.last_run(tmp_path)
    assert got["results"][2]["detail"].startswith("PROXY"), got["results"][2]


# ── the controls must be able to go RED ─────────────────────────────── #

def test_a_control_the_pins_LET_THROUGH_is_reported_as_a_FAILURE(
        tmp_path, monkeypatch):
    """⚠ In a healthy repo the guard control always passes, so the branch
    that reports "the pins let it through" is DEAD in every test run —
    including the one named after it. Force the failure: a stage 1 that
    approves a deleted guard must produce ok=False, an empty
    `rejected_at`, and a report that says so rather than the cheerful
    default the `or` used to supply."""
    monkeypatch.setattr(EV, "stage1_pins",
                        lambda *a, **k: EV.StageResult(EV.STAGE_PINS, True,
                                                       "", {}, 0.0))
    run = NC.run_negative_controls(REPO, tmp_path)
    gd = [r for r in run.results if r.name == "deletes_a_guard"][0]
    assert gd.ok is False
    assert gd.rejected is False and gd.rejected_at == ""
    assert "THE PINS PASSED" in gd.detail, gd.detail
    assert run.all_ok is False


def test_a_no_op_control_that_is_NOT_a_no_op_fails_LOUDLY(tmp_path,
                                                          monkeypatch):
    """⚠ The shallow check used to call `paired_verdict([(True, True)]*20)`
    — a hard-coded input that ignored the control's own diff. Making
    `_noop` delete half the file left the control GREEN: it tested the
    decision function while wearing the name of a control over a no-op
    candidate."""
    real = NC.build_controls

    def _not_a_noop(root):
        d = dict(real(root))
        d["no_op_claiming_improvement"] = NC._diff(
            NC.NOOP_TARGET, Path(root), lambda lines: lines[:len(lines) // 2])
        return d
    monkeypatch.setattr(NC, "build_controls", _not_a_noop)
    run = NC.run_negative_controls(REPO, tmp_path,
                                   only=("no_op_claiming_improvement",))
    noop = [r for r in run.results if r.name == "no_op_claiming_improvement"][0]
    assert noop.ok is False
    assert "NOT A NO-OP" in noop.detail.upper(), noop.detail
    # ⚠ `all_ok` IS FALSE FOR ANY SUBSET RUN BY DESIGN, so
    # asserting it here would pass whatever the control did.
    # `partial_ok` is the question: did what this run
    # SELECTED hold?
    assert run.partial_ok is False
    assert run.all_ok is False


def test_a_control_whose_TARGET_MOVED_is_a_failure_not_a_traceback(
        tmp_path, monkeypatch):
    """A missing target must not escape as an exception — the operator
    would get a traceback instead of a report, the one output that says
    nothing about which controls held — and must not look like a pass."""
    monkeypatch.setattr(NC, "NOOP_TARGET", "src/ghost_agent/tools/gone_xyz.py")
    run = NC.run_negative_controls(REPO, tmp_path,
                                   only=("no_op_claiming_improvement",))
    # ⚠ `all_ok` IS FALSE FOR ANY SUBSET RUN BY DESIGN, so
    # asserting it here would pass whatever the control did.
    # `partial_ok` is the question: did what this run
    # SELECTED hold?
    assert run.partial_ok is False
    assert run.all_ok is False
    bad = [r for r in run.results if r.name == "no_op_claiming_improvement"]
    assert bad and bad[0].ok is False
    assert "TARGET IS GONE" in bad[0].detail, bad[0].detail


def test_deep_mode_is_NOT_GREENER_than_shallow(real_deep_run):
    """⚠ `ok=True, verified=False` is a THIRD STATE, and `all_ok` must
    not count it. Deep mode reported `all_ok=True` for a control it
    never executed — the thorough mode came out greener than the shallow
    one, which is this module's own rule inverted by its own reporting
    field."""
    deep = real_deep_run[0]; tmp_path = real_deep_run[1]
    noop = [r for r in deep.results if r.name == "no_op_claiming_improvement"][0]
    assert noop.verified is False, "an unrun control claimed to be verified"
    assert deep.partial_ok is False, "deep mode was greener than shallow"
    assert deep.all_ok is False
    assert "NOT RUN" in noop.detail
    # …and the persisted record names what was not verified, so a health
    # view can render the gap rather than a green tick.
    got = NC.last_run(tmp_path)
    assert got["unverified"] == ["no_op_claiming_improvement"], got


def test_a_control_whose_PINS_NEVER_RAN_is_not_a_pass(tmp_path, monkeypatch):
    """⚠ THE HOLE A FIX RE-OPENED. Stage 1 phrased *every* decline as
    "pins failed: …", including `no tests ran` and an ImportError banner
    from a plugin that would not load. This control keyed on that
    phrase, so a run in which the pins NEVER EXECUTED scored the guard
    control GREEN — verbatim the regression the control was written to
    catch, re-introduced an hour later by a fix to a different file.
    The kind of failure is now an identity, not a substring."""
    for kind, reason in ((EV.PINS_NOT_RUN,
                          "pins failed: no tests ran in 0.00s"),
                         (EV.PINS_UNREADABLE,
                          'pins failed: ImportError: plugin "x"')):
        monkeypatch.setattr(
            EV, "stage1_pins",
            lambda *a, _k=kind, _r=reason, **kw: EV.StageResult(
                EV.STAGE_PINS, False, _r, {"failure_kind": _k}, 0.0))
        run = NC.run_negative_controls(REPO, tmp_path)
        gd = [r for r in run.results if r.name == "deletes_a_guard"][0]
        assert gd.ok is False, f"{kind}: a run with no pins scored green"
        assert "NEVER RAN" in gd.detail, gd.detail
        assert run.all_ok is False


def test_the_guard_control_is_GREEN_only_for_a_REAL_pin_failure(tmp_path,
                                                                monkeypatch):
    """The positive half: the identity must still accept the real thing,
    or the fix above would have been "refuse everything".

    ⚠ The stub must now name the pin that failed. That is the point of
    the fix — a control claiming "a pin catches a deleted guard" has to
    say WHICH pin — and a stub that cannot supply one is a stub that
    could not tell the two cases apart either."""
    guard_pin = "tests/test_execute_html_guard.py"
    assert NC.GUARD_PIN_MARK in (REPO / guard_pin).read_text(), \
        "the guard's pin no longer asserts the mark, so this stub is stale"
    monkeypatch.setattr(
        EV, "stage1_pins",
        lambda *a, **k: EV.StageResult(
            EV.STAGE_PINS, False, "pins failed: 2 failed, 116 passed in 3s",
            {"failure_kind": EV.PINS_FAILED,
             "failed_pin_files": [guard_pin]}, 0.0))
    run = NC.run_negative_controls(REPO, tmp_path)
    gd = [r for r in run.results if r.name == "deletes_a_guard"][0]
    assert gd.ok is True and gd.rejected_at == EV.STAGE_PINS, gd.detail
    assert gd.detail_pins == [guard_pin], gd.detail_pins


def test_the_guard_control_needs_ITS_OWN_pin_to_have_failed(tmp_path,
                                                            monkeypatch):
    """⚠ `-x` DE-SPECIFIED THIS CONTROL. `tools/execute.py` maps to 45
    pin files, stage 1 stops at the FIRST failure, and
    `test_execute_html_guard.py` sorts late — so one unrelated failing
    pin ended the run and the control scored GREEN on a run where the
    guard's own pin never executed. Measured before the fix:
    `ok=True, all_ok=True, '1 failed, 6 passed'`.

    Asserting the KIND is not asserting the POINT. The control's claim
    is "a pin catches a deleted guard", so it has to name that pin."""
    monkeypatch.setattr(EV, "stage1_pins", lambda *a, **k: EV.StageResult(
        EV.STAGE_PINS, False, "pins failed: 1 failed, 6 passed in 4.59s",
        {"failure_kind": EV.PINS_FAILED,
         "failed_pin_files": ["tests/test_agent_scrubbing.py"]}, 0.0))
    run = NC.run_negative_controls(REPO, tmp_path)
    gd = [r for r in run.results if r.name == "deletes_a_guard"][0]
    assert gd.ok is False, "an unrelated failure satisfied the guard control"
    assert "SOMETHING ELSE FAILED FIRST" in gd.detail, gd.detail
    assert run.all_ok is False


def test_a_candidate_that_BREAKS_the_pins_is_not_a_candidate_they_CAUGHT(
        tmp_path, monkeypatch):
    """⚠ `1 error in 0.02s` IS NOT `1 failed`. An error is what pytest
    prints when a pin file could not be COLLECTED — the candidate broke
    the pins rather than failing them, and zero test bodies ran. Folding
    it into `PINS_FAILED` gave this control two producers of the kind it
    keys on, one of which executed nothing: the exact regression the
    control was rebuilt to detect."""
    ok, why, kind = EV._summary_is_clean("1 error in 0.02s")
    assert not ok and kind == EV.PINS_ERRORED, (why, kind)
    ok2, _w, kind2 = EV._summary_is_clean("1 failed, 1 error in 2s")
    assert not ok2 and kind2 == EV.PINS_FAILED, kind2

    monkeypatch.setattr(EV, "stage1_pins", lambda *a, **k: EV.StageResult(
        EV.STAGE_PINS, False,
        "the pins could not RUN against this candidate: 1 error in 0.02s",
        {"failure_kind": EV.PINS_ERRORED, "failed_pin_files": []}, 0.0))
    run = NC.run_negative_controls(REPO, tmp_path)
    gd = [r for r in run.results if r.name == "deletes_a_guard"][0]
    assert gd.ok is False, "a collection error scored as 'the pins objected'"
    assert "NEVER RAN" in gd.detail, gd.detail


def test_only_ACTUALLY_SKIPS_and_the_record_says_which(tmp_path):
    """⚠ THE SELECTOR WAS UNPINNED IN BOTH DIRECTIONS. Making `_skip`
    always return False — i.e. ignoring `only=` entirely — survived the
    whole file, and so did `run.selected = []`, the field that exists so
    a partial run says it was partial. A subset run that silently
    performs everything is merely slow; one that silently performs LESS
    while reporting `all_ok` is the failure this module is about."""
    run = NC.run_negative_controls(REPO, tmp_path,
                                   only=("edits_a_test",))
    assert [r.name for r in run.results] == ["edits_a_test"], \
        [r.name for r in run.results]
    assert run.selected == ["edits_a_test"], run.selected
    assert run.partial_ok is True, [vars(r) for r in run.results]
    # …and a SUBSET is never `all_ok`: a control that was not selected
    # would otherwise contribute to a green suite by being absent.
    assert run.all_ok is False

    got = NC.last_run(tmp_path)
    assert got["selected"] == ["edits_a_test"], got
    assert got["all_ok"] is False and got["partial_ok"] is True, got


def test_a_MOVED_target_respects_the_selection(tmp_path, monkeypatch):
    """⚠ THE EARLY RETURN IGNORED `only`. With `only=("no_op_…",)` and
    control 1's target moved, the run returned a result for
    `edits_a_test` — a control it was not asked to perform — and never
    ran the one it was. `selected` and `results` then disagreed, which
    is worse than either failure alone: the record names a control the
    run did not do."""
    monkeypatch.setattr(NC, "TEST_TARGET", "tests/test_gone_xyz.py")
    run = NC.run_negative_controls(REPO, tmp_path,
                                   only=("no_op_claiming_improvement",))
    assert [r.name for r in run.results] == ["no_op_claiming_improvement"], \
        [r.name for r in run.results]
    assert run.partial_ok is True, [vars(r) for r in run.results]

    # …and when the MOVED target IS the selected one, it is reported.
    run2 = NC.run_negative_controls(REPO, tmp_path,
                                    only=("edits_a_test",))
    assert [r.name for r in run2.results] == ["edits_a_test"]
    assert run2.partial_ok is False
    assert "TARGET IS GONE" in run2.results[0].detail, run2.results[0].detail
