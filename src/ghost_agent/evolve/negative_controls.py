"""E3 — negative controls: prove the cascade still REFUSES.

⚠ THIS EXISTS BECAUSE A GUARD THAT NEVER DEMONSTRABLY FIRES IS PRESUMED
DEAD. E2's entire value is rejection: it refused a null candidate that
scored nominally higher, three malformed diffs at three different
guards, and a real mutation for being worse. None of that is evidence
next month, when the code has moved and nobody re-checked.

Three known-bad candidates, each with the stage that must reject it:

* **edits a test** — the subject rewriting its own judge. `stage0`, on
  the diff-scope check, before anything is compiled.
* **deletes a guard** — `stage1`, because a pin asserts that guard
  fires. ⚠ This one is only as good as the pin behind it; see
  `GUARD_TARGET`.
* **a no-op claiming improvement** — `stage3`, which must call it not
  better rather than promoting noise.

Each control asserts the rejection AND the stage. A candidate refused
for the wrong reason is a cascade that will refuse the right thing for
the wrong reason later.
"""
from __future__ import annotations

import datetime
import ast
import difflib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import evaluator as EV

#: ⚠ THE GUARD CONTROL IS ONLY AS GOOD AS ITS PIN. Chosen by MEASUREMENT,
#: not by looking: `tests/test_execute_html_guard.py` asserts this string
#: IS in the tool's output, so removing it fails 2 of 3 tests. The
#: daemon-block guard was the obvious candidate and is USELESS here — its
#: only pin asserts `"SYSTEM BLOCK" not in result`, the negative case, so
#: deleting the guard passes. A control built on an unpinned guard would
#: report the cascade as broken every month while nothing was wrong.
GUARD_TARGET = "src/ghost_agent/tools/execute.py"
GUARD_MARK = "SYSTEM TIP: It looks like you are trying to write HTML"
#: The substring the PIN asserts. Deliberately separate from
#: `GUARD_MARK`: the guard's text and the assertion about it are two
#: different strings, and the control is only valid while they overlap.
#: A test checks that they still do — if someone reworded the guard, the
#: control would silently stop proving anything.
GUARD_PIN_MARK = "It looks like you are trying to write HTML"

TEST_TARGET = "tests/test_execute_html_guard.py"
NOOP_TARGET = "src/ghost_agent/tools/database.py"

RESULT_FILE = "negative_controls.json"

#: How often the scheduled run fires (§4CS item E). ONE definition: the
#: idle phase in `core/agent.py` uses it as its cooldown and the liveness
#: probe uses it to decide staleness. Two copies of a cadence is how a
#: monitor ends up alarming on a schedule that was changed elsewhere, or
#: — worse — staying quiet about one that stopped.
INTERVAL_S = 7 * 24 * 3600

#: A run older than this is treated as a schedule that has STOPPED, not
#: as a stale-but-fine pass. Two intervals, so one missed window (a box
#: that was busy every time the idle floor came round) is not an alarm.
STALE_AFTER_S = 2 * INTERVAL_S


def last_run_ts(home) -> Optional[float]:
    """Epoch seconds of the last recorded control run, or None.

    None means NEVER RUN, which is not the same as passing and must not
    render as one — see `last_run`.
    """
    doc = last_run(home)
    raw = str((doc or {}).get("ts") or "").strip()
    if not raw:
        return None
    try:
        return datetime.datetime.fromisoformat(
            raw.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None

#: The full set. `all_ok` is only True for a run that performed all of
#: them — see `ControlRun.all_ok`.
ALL_CONTROLS: Tuple[str, ...] = ("edits_a_test", "deletes_a_guard",
                                 "no_op_claiming_improvement")


@dataclass
class ControlResult:
    name: str
    expect_stage: str
    rejected: bool = False
    rejected_at: str = ""
    ok: bool = False
    #: For the guard control: which pin file(s) actually objected.
    detail_pins: List[str] = field(default_factory=list)
    #: ⚠ A THIRD STATE. `ok=True` with `verified=False` means "this
    #: control did not run", which must NOT contribute to a green
    #: suite — deep mode reported `all_ok=True` for a control it never
    #: executed, making the thorough mode LESS verified than the shallow
    #: one and greener. That is this module's own rule inverted by its
    #: own reporting field.
    verified: bool = True
    detail: str = ""


@dataclass
class ControlRun:
    ts: str = ""
    deep: bool = False
    #: Which controls this run was ASKED for. `all_ok` is a statement
    #: about the controls that ran, so a partial run must say which
    #: those were rather than letting a caller read three greens into a
    #: report that only exercised one.
    selected: List[str] = field(default_factory=list)
    results: List[ControlResult] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        """Every SELECTED control ran and held.

        ⚠ A CONTROL THAT WAS NEVER SELECTED CONTRIBUTES TO A GREEN SUITE
        BY BEING ABSENT — the same rule `verified` exists to prevent,
        inverted. `ok=True, verified=False` was invented because "this
        control did not run" must not read as green; a partial run is
        the same statement made by omission. So a run is only `all_ok`
        when it performed everything it was asked for AND that was the
        full set. `partial_ok` is the honest answer for a subset.
        """
        return self.partial_ok and set(self.selected) == set(ALL_CONTROLS)

    @property
    def partial_ok(self) -> bool:
        """Every control this run SELECTED ran and held."""
        return (bool(self.results)
                and {r.name for r in self.results} == set(self.selected)
                and all(r.ok and r.verified for r in self.results))


def _read(path: Path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _semantically_identical(canonical: Path, cand: Path,
                           rel: str) -> Tuple[bool, str]:
    """Is the candidate's version of `rel` the SAME PROGRAM?

    Compares parsed syntax trees, not bytes: the no-op control adds a
    comment line, which must not count as a change, while deleting a
    function must. Text comparison would call the comment a change and
    an equivalent reformat a no-op — both backwards for this purpose.
    """
    a, b = Path(canonical) / rel, Path(cand) / rel
    if not b.exists():
        return False, f"{rel} does not exist in the candidate"
    try:
        ta = ast.dump(ast.parse(a.read_text(encoding="utf-8", errors="replace")))
        tb = ast.dump(ast.parse(b.read_text(encoding="utf-8", errors="replace")))
    except SyntaxError as exc:
        return False, f"{rel} does not parse: {exc}"
    if ta != tb:
        return False, (f"{rel} differs by {abs(len(tb) - len(ta))} chars of "
                       f"syntax tree — it is a real change, not a no-op")
    # …and nothing ELSE may have moved either.
    others = [f for f in _changed_files(canonical, cand) if f != rel]
    if others:
        return False, f"it also changed {len(others)} other file(s): {others[:3]}"
    return True, "identical syntax tree, and no other file changed"


def _changed_files(canonical: Path, cand: Path) -> List[str]:
    """Every tracked file whose bytes differ between the two trees."""
    out: List[str] = []
    for sub in ("src", "tests", "scripts"):
        base = Path(cand) / sub
        if not base.is_dir():
            continue
        for f in base.rglob("*.py"):
            rel = str(f.relative_to(cand))
            twin = Path(canonical) / rel
            try:
                if not twin.exists() or twin.read_bytes() != f.read_bytes():
                    out.append(rel)
            except OSError:
                out.append(rel)
    return sorted(out)


def _diff(rel: str, root: Path, mutate) -> str:
    """A unified diff from the REAL file, so the coordinates are right.

    A target that has MOVED returns "" rather than raising: an exception
    here escapes `run_negative_controls` entirely and the operator gets
    a traceback instead of a report — the one output that says nothing
    about which controls held. An empty diff is turned into a named,
    failed control by every caller below.
    """
    try:
        src = (root / rel).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    old = src.splitlines(keepends=True)
    new = mutate(list(old))
    return "".join(difflib.unified_diff(old, new, f"a/{rel}", f"b/{rel}"))


def build_controls(root: Path) -> Dict[str, str]:
    """The three known-bad diffs, generated against the canonical tree.

    Generated rather than stored: a stored diff rots the moment the file
    it patches moves, and a control that no longer applies reports a
    cascade failure that is really a stale fixture.
    """
    root = Path(root)

    def _edit_test(lines):
        return ["# a candidate editing its own judge\n"] + lines

    def _delete_guard(lines):
        return [ln for ln in lines if GUARD_MARK not in ln]

    def _noop(lines):
        return ["# E3 no-op control: semantically identical on purpose\n"] + lines

    return {
        "edits_a_test": _diff(TEST_TARGET, root, _edit_test),
        "deletes_a_guard": _diff(GUARD_TARGET, root, _delete_guard),
        "no_op_claiming_improvement": _diff(NOOP_TARGET, root, _noop),
    }


def run_negative_controls(canonical_root: Path, home: Path, *,
                          deep: bool = False,
                          only: Tuple[str, ...] = None,
                          materialize=None) -> ControlRun:
    """Run all three controls and record whether each was refused, where.

    `deep=False` stops at the stage each control is supposed to die at,
    which is both faster and exactly what is being asserted — only the
    no-op needs stages 2–3, and those cost hours. In shallow mode the
    no-op is checked against `paired_verdict` on identical arms, which is
    what a real run produces; that is a PROXY and is recorded as one.
    """
    root = Path(canonical_root)
    run = ControlRun(ts=datetime.datetime.utcnow().isoformat() + "Z",
                     deep=deep)
    diffs = build_controls(root)

    # ⚠ `only` IS FOR CALLERS THAT NEED ONE CONTROL, NOT FOR SKIPPING
    # WORK IN A REAL RUN. The guard control runs stage 1 for real, and
    # stage 1 now maps `tools/execute.py` to 45 pin files (753 tests,
    # ~125 s) because coverage is the union of the filename convention
    # and import evidence. A test that only inspects the no-op control
    # should not pay for that; a real suite run must still pass `only`
    # as None, and `all_ok` below is therefore about the controls that
    # RAN — which is why the run records which ones those were.
    # ⚠ HOISTED OUT OF CONTROL 2. This default used to be set inside the
    # guard control's block, so running any OTHER control alone left
    # `materialize` as None and the no-op control died with
    # "'NoneType' object is not callable" — a shared default hidden
    # inside one branch, which only became visible once the branches
    # could be skipped independently.
    if materialize is None:
        from . import mutator as M
        materialize = M.materialize

    wanted = tuple(only) if only else None
    run.selected = list(wanted) if wanted else list(ALL_CONTROLS)

    def _skip(name: str) -> bool:
        return wanted is not None and name not in wanted

    # A control whose TARGET HAS MOVED proves nothing, and must not be
    # allowed to look like a pass.
    # ⚠ AND IT MUST RESPECT `only`. With `only=("no_op_…",)` and control
    # 1's target moved, this returned a result for `edits_a_test` — a
    # control that was not selected — and never ran the one that was.
    # `selected` and `results` then disagreed, which is worse than
    # either failure alone: the record names a control the run did not
    # perform.
    gone = [k for k, v in diffs.items()
            if not v.strip() and not _skip(k)]
    if gone:
        for name in gone:
            run.results.append(ControlResult(
                name=name, expect_stage="", ok=False, verified=True,
                detail=("THE CONTROL'S TARGET IS GONE — the file this "
                        "control patches no longer exists, so it proves "
                        "nothing about the cascade")))
        _write(run, home)
        return run

    # 1 — edits a test: refused on scope, before anything is compiled.
    if not _skip('edits_a_test'):
        r = EV.stage0_static(root, diffs["edits_a_test"])
        run.results.append(ControlResult(
            name="edits_a_test", expect_stage=EV.STAGE_STATIC,
            rejected=not r.passed, rejected_at=EV.STAGE_STATIC if not r.passed else "",
            ok=(not r.passed and "diff-scope" in r.reason),
            detail=r.reason[:160]))

    # 2 — deletes a guard: must PASS stage 0 (it is in scope and compiles)
    #     and die at stage 1, because a pin asserts that guard fires.
    if not _skip('deletes_a_guard'):
        gd = diffs["deletes_a_guard"]
        res = ControlResult(name="deletes_a_guard", expect_stage=EV.STAGE_PINS)
        node = f"e3-guard-{run.ts.replace(':', '').replace('-', '')[:15]}"
        ok_m, why = materialize(node, gd, home=str(home), repo_root=root)
        if not ok_m:
            res.detail = f"could not materialise the control: {why[:120]}"
        else:
            from . import mutator as M
            cand = M.work_dir(node, str(home))
            s0 = EV.stage0_static(cand, gd)
            if not s0.passed:
                # ⚠ Dying at stage 0 is NOT a pass. The control exists to
                # show the PIN catches a guard removal; a scope or compile
                # rejection would hide that the pin never ran.
                res.rejected, res.rejected_at = True, EV.STAGE_STATIC
                res.detail = f"died early at stage 0: {s0.reason[:120]}"
            else:
                s1 = EV.stage1_pins(cand, root, s0.detail["touched"])
                res.rejected = not s1.passed
                res.rejected_at = EV.STAGE_PINS if not s1.passed else ""
                # ⚠ THE REASON MUST BE THE PINS, NOT ANY REJECTION. Accepting
                # every stage-1 failure meant that renaming the target file
                # — "no pin covers …" — scored the control GREEN for a run
                # in which the pins never executed. Control 1 checks its
                # reason; this one checked nothing, contradicting this
                # module's own rule that a candidate refused for the wrong
                # reason is a cascade that will refuse the right thing for
                # the wrong reason later.
                # ⚠ ASK FOR THE IDENTITY, NOT A PHRASE. `"pins failed" in
                # reason` also matched "the pins failed to import" and
                # "no tests ran" — runs in which the pins NEVER EXECUTED
                # scored this control green, which is the precise failure it
                # exists to detect. `failure_kind` is set by exactly one
                # branch: pytest counted failures.
                # ⚠ AND IT MUST BE THE GUARD'S OWN PIN THAT OBJECTED.
                # Asserting only the KIND de-specified this control the
                # moment stage 1 gained `-x`: `tools/execute.py` maps to
                # 45 pin files, pytest stops at the FIRST failure, and
                # `test_execute_html_guard.py` sorts late — so one
                # unrelated failing pin ended the run and this scored
                # GREEN on a run where the guard's pin never executed.
                # Measured: `ok=True, all_ok=True, '1 failed, 6 passed'`.
                guard_pins = [f for f in s1.detail.get("failed_pin_files", [])
                              if GUARD_PIN_MARK in _read(root / f)]
                res.detail_pins = guard_pins
                res.ok = (not s1.passed
                          and s1.detail.get("failure_kind") == EV.PINS_FAILED
                          and bool(guard_pins))
                # ⚠ `reason` is EMPTY on success, so `reason or "<message>"`
                # printed "pins failed as required" for a control the pins
                # had just let through — the report contradicted its own
                # verdict field. Say which happened.
                if s1.passed:
                    res.detail = ("⚠ THE PINS PASSED — a guard was deleted "
                                  "and nothing objected")
                elif res.ok:
                    res.detail = (
                        f"{s1.reason[:110]} "
                        f"[{', '.join(Path(g).name for g in guard_pins)}]")
                elif s1.detail.get("failure_kind") == EV.PINS_FAILED:
                    res.detail = (
                        f"⚠ SOMETHING ELSE FAILED FIRST — the guard's own "
                        f"pin never ran, so this control demonstrates "
                        f"nothing: {s1.reason[:100]}")
                else:
                    res.detail = (
                        f"⚠ THE PINS NEVER RAN "
                        f"({s1.detail.get('failure_kind')}) — this control "
                        f"demonstrates nothing: {s1.reason[:100]}")
            # ⚠ RECLAIM THE SNAPSHOT. `sweep_work_dirs` keeps only
            # MAX_KEPT_SNAPSHOTS, so a control leaking ~10 MB per run
            # EVICTS REAL CANDIDATES — the suite that checks the loop would
            # quietly destroy the loop's own work.
            try:
                M._discard(cand)
            except Exception:          # noqa: BLE001
                pass
        run.results.append(res)

    # 3 — a no-op claiming improvement: stage 3 must call it not better.
    if not _skip('no_op_claiming_improvement'):
        res = ControlResult(name="no_op_claiming_improvement",
                            expect_stage=EV.STAGE_PAIRED)
        if deep:
            # ⚠ NOT IMPLEMENTED, AND IT SAYS SO RATHER THAN SCORING ITSELF.
            # The earlier version left `ok=False` with a detail that read
            # like a plan, so EVERY deep run reported `all_ok=False` — a
            # guaranteed false red, which trains an operator to ignore the
            # whole suite. Until stage 3 can be driven here, deep mode
            # refuses to render a verdict on this control at all.
            res.ok = True
            res.verified = False          # …and therefore not part of all_ok
            res.rejected = False
            res.rejected_at = ""
            res.detail = ("NOT RUN — deep mode cannot drive stage 3 yet; "
                          "this control is UNVERIFIED in this run")
        else:
            # ⚠ THIS PROXY USED TO IGNORE ITS OWN DIFF. It called
            # `paired_verdict([(True, True)] * 20)` — a hard-coded input.
            # Pointing NOOP_TARGET at a file that does not exist, or making
            # `_noop` delete half of `database.py`, left the control GREEN:
            # it was a test of `paired_verdict`, wearing the name of a
            # control over the no-op candidate. So establish the premise
            # FIRST — that the diff really is a no-op — and only then let
            # the decision function rule on the arms such a candidate makes.
            why_v = ""
            cand = None
            try:
                from . import mutator as M
                nnode = f"e3-noop-{run.ts.replace(':', '').replace('-', '')[:15]}"
                ok_m, why_m = materialize(nnode,
                                          diffs["no_op_claiming_improvement"],
                                          home=str(home), repo_root=root)
                if not ok_m:
                    raise RuntimeError(f"materialise refused it: {why_m[:120]}")
                cand = M.work_dir(nnode, str(home))
                same, why_same = _semantically_identical(root, cand, NOOP_TARGET)
                if not same:
                    res.ok = False
                    res.rejected = False
                    res.detail = ("THE CONTROL IS NOT A NO-OP: " + why_same +
                                  " — this control cannot prove what it claims")
                else:
                    ok_v, why_v, _ = EV.paired_verdict([(True, True)] * 20)
                    res.rejected = not ok_v
                    res.rejected_at = EV.STAGE_PAIRED if not ok_v else ""
                    res.ok = not ok_v
                    res.detail = (f"PROXY (shallow): no-op verified against the "
                                  f"canonical tree; {why_v[:100]}")
            except Exception as exc:          # noqa: BLE001
                res.ok = False
                res.rejected = False
                res.detail = f"THE CONTROL COULD NOT BE BUILT: {exc}"[:200]
            finally:
                if cand is not None:
                    try:
                        M._discard(cand)
                    except Exception:          # noqa: BLE001
                        pass
        run.results.append(res)

    _write(run, home)
    return run


def _write(run: ControlRun, home: Path) -> Optional[Path]:
    try:
        d = Path(home) / "system" / "evolve"
        d.mkdir(parents=True, exist_ok=True)
        p = d / RESULT_FILE
        p.write_text(json.dumps(
            {"ts": run.ts, "deep": run.deep, "all_ok": run.all_ok,
             "partial_ok": run.partial_ok,
             "selected": list(run.selected),
             "unverified": [r.name for r in run.results if not r.verified],
             "results": [vars(r) for r in run.results]}, indent=2))
        return p
    except OSError:
        return None


def last_run(home: Path) -> Optional[Dict]:
    """The persisted record of the last control run.

    ⚠ NOTHING READS THIS YET. It was documented as "what the
    learning-health view reads" while no learning-health code opened it
    — a reader checking whether the controls are surfaced would have
    stopped at that line and concluded they were. `None` = never run,
    which is NOT the same as passing and must not render as a tick on
    the day something does read it.
    """
    p = Path(home) / "system" / "evolve" / RESULT_FILE
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:      # noqa: BLE001
        return None
