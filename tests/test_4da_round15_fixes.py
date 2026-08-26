"""§4DA round 15 — the RETIRE code was pinned by its own source text.

Round 13 gave `recheck_gepa_incumbent.py` a fourth exit code so a script could
tell "the incumbent still earns its place" from "this holdout cannot settle the
question". Its only guard was a **source-shape assertion** — `body.index("if
_unmeasurable:")` followed by `assert "return 2" in body[i:i+80]`. Inserting one
line above the branch:

    _unmeasurable = _unmeasurable and False

restores the exact pre-fix defect (exit 0, "still wins", from the branch that
just printed *"it is evidence that this holdout cannot settle the question"*)
and **survives the full 855-test battery**. The ten tests that DO reach the
branch all assert `rc in (0, 1, 2)` — which admits the pre-fix 0 — with a
comment three lines above naming the very fix they cannot see.

Round 8's own ⚠ recorded *"SIX of round 8's own first pins were SOURCE GREPS"*.
Round 13 did it again, in the branch a script acts on to retire a live artifact.

Everything here is DRIVEN: the module runs, and the assertion is on the integer
it returns.
"""

import json
import sys
from pathlib import Path

import pytest

from ghost_agent.optim import loader as L
from ghost_agent.tools import registry as R

from tests.test_4da_tool_desc_ship_gate import (
    TestTheDecisionIsActuallyUSED as _H,
)
from tests.test_gepa_optim_reaudit import (
    TestTheRecheckInstrumentIsDriven as _RD,
    _corpus,
    _load,
)


# ─────────────────────────────────────────────────────────────────────
# 1. The RETIRE instrument's exit codes, driven end to end.
# ─────────────────────────────────────────────────────────────────────
class TestTheCouldNotMeasureCodeIsDRIVEN:
    """⚠ The branch is what a caller acts on to retire a live artifact,
    and its only pin asserted on the file's own characters."""

    def test_the_unmeasurable_win_exits_2_not_0(self, tmp_path, capsys):
        """`delta` above the bar, evidence too thin to support it. The
        pre-fix code returned `0 if delta > margin` — "still wins" —
        which is what `_unmeasurable = _unmeasurable and False` restores
        and what the source pin cannot see."""
        rc = _RD()._run(tmp_path, delta=0.1290, ships=False, bw=1, cw=5,
                        min_delta=0.02)
        out = capsys.readouterr().out
        assert "NO LONGER MEASURABLE" in out, out
        assert rc == 2, (
            "the branch that says the holdout cannot settle the question "
            f"returned {rc} — 0 means 'still wins', 1 means 'retire it'")

    def test_the_transport_caveated_loss_exits_2_not_1(self, tmp_path,
                                                       capsys):
        """A one-arm outage makes every surviving pair a loss. Without
        the code, `delta < 0` reads as a measured regression and the
        caller retires an artifact over a dead upstream."""
        rc = _RD()._run(tmp_path, delta=-0.30, ships=False, bw=8, cw=0,
                        min_delta=0.02, keep_pairs=5)
        out = capsys.readouterr().out
        assert "TRANSPORT failure wearing" in out, out
        assert rc == 2, (
            f"a transport outage was reported as a measured loss (rc={rc})")

    def test_a_HEALTHY_loss_still_exits_1(self, tmp_path, capsys):
        """⚠ The pair. A code that is 2 for every loss is as useless as
        one that is 0 — the pin must fail in a world where the branch
        over-fires, not only where it under-fires."""
        rc = _RD()._run(tmp_path, delta=-0.30, ships=False, bw=8, cw=0,
                        min_delta=0.02)
        out = capsys.readouterr().out
        assert "NOW WORSE THAN THE BASELINE" in out, out
        assert "TRANSPORT failure wearing" not in out, out
        assert rc == 1, (rc, out)

    def _drive_early(self, tmp_path, *, argv_extra, corpus=True,
                     artifact=None):
        home = tmp_path / "home"
        if corpus:
            _corpus(home / "system" / "trajectories")
        else:
            (home / "system" / "trajectories").mkdir(parents=True)
        art = tmp_path / "planning.decompose.json"
        if artifact is not None:
            art.write_text(json.dumps(artifact))
        mod = _load("recheck_early", "scripts/recheck_gepa_incumbent.py")
        old = sys.argv
        try:
            sys.argv = (["recheck", "--artifact", str(art), "--home",
                         str(home)] + list(argv_extra))
            import asyncio
            return asyncio.run(mod.main())
        finally:
            sys.argv = old

    def test_a_MISSING_artifact_exits_2(self, tmp_path):
        assert self._drive_early(tmp_path, argv_extra=()) == 2

    def test_an_artifact_with_NO_instruction_exits_2(self, tmp_path):
        rc = self._drive_early(tmp_path, argv_extra=(), artifact={
            "signature_name": "planning.decompose",
            "baseline_instruction": "SEED", "optimized_instruction": "",
            "gate_arm": "g"})
        assert rc == 2, rc

    def test_an_EMPTY_private_tier_exits_2(self, tmp_path, capsys):
        rc = self._drive_early(tmp_path, argv_extra=(), corpus=False,
                               artifact={
            "signature_name": "planning.decompose",
            "baseline_instruction": "SEED",
            "optimized_instruction": "LIVE", "gate_arm": "g"})
        assert "empty private tier" in capsys.readouterr().err
        assert rc == 2, rc


# ─────────────────────────────────────────────────────────────────────
# 2. The gate's own verdict/exit-code collision.
# ─────────────────────────────────────────────────────────────────────
class TestNoCandidateIsNotAGateRejection:
    """⚠ `ships=True` with nothing changed printed BOTH "ships=True …
    A/B gate REJECTED — live descriptions stand." and "A/B gate PASSED
    but no component actually changed", then returned 1 — the same code
    as a genuine loss. A caller cannot tell "the candidate lost" from
    "the reflection LM produced nothing"."""

    def test_a_seed_verbatim_run_exits_3(self, tmp_path, monkeypatch,
                                         capsys):
        rc, live, rejected, _n = _H()._run(tmp_path, monkeypatch,
                                           cand_wins=6, mutate=False)
        err = capsys.readouterr().err
        assert not live, [p.name for p in live]
        assert rc == 3, (
            f"a run that produced no candidate exited {rc} — 1 is a "
            f"measured rejection of a real candidate. stderr: {err}")

    def test_a_GENUINE_rejection_still_exits_1(self, tmp_path, monkeypatch,
                                               capsys):
        """The pair: 3 must not swallow the ordinary loss."""
        rc, live, rejected, _n = _H()._run(tmp_path, monkeypatch,
                                           cand_wins=0, inc_wins=6,
                                           n_fixtures=70)
        out = capsys.readouterr().out
        assert rc == 1, (rc, out)
        assert not live, [p.name for p in live]


class TestTheREJECTPathAlsoWritesOnlyChangedComponents:
    """⚠ Round 13 fixed `best -> _changed` on the PROMOTE side and left
    `(_changed if ships else best)` on the reject side. Driven with one
    of three mutated and the gate rejecting, two of three
    `.candidate.rejected` records were byte-identical to the incumbent,
    each stamped with the set's `p_value`/`candidate_wins` and a
    `co_promoted` list that did not contain the file itself."""

    def test_an_untouched_component_gets_no_rejection_record(
            self, tmp_path, monkeypatch):
        rc, live, rejected, _n = _H()._run(tmp_path, monkeypatch,
                                           cand_wins=0, inc_wins=6,
                                           n_fixtures=70, n_tools=3,
                                           mutate="one")
        assert rc == 1 and not live
        assert len(rejected) == 1, (
            "components the optimizer never touched were given a "
            "rejection record carrying the set's statistic: "
            + str([p.name for p in rejected]))
        rec = json.loads(rejected[0].read_text())
        assert rec["optimized_instruction"] != rec["baseline_instruction"]
        # ⚠ A REJECTED record must not claim a promotion. `co_promoted`
        # is what `recheck_gepa_incumbent.py` prints back as "the set
        # this win belongs to"; nothing here was promoted.
        assert "co_promoted" not in rec["gate"], rec["gate"]
        assert rec["gate"]["co_candidates"] == [rec["signature_name"]], (
            rec["gate"]["co_candidates"])


class TestTheGateScopeMatchesTheSetSize:
    """⚠ With ONE changed component the A/B compared seed-set against
    seed-set-with-that-one-change — it measured exactly that
    component's contribution, and the record said "no per-component
    contribution was measured" anyway. One changed component is what a
    real proposal produces, so the false wording was the common case."""

    def test_a_SOLO_change_claims_its_own_contribution(self, tmp_path,
                                                       monkeypatch):
        rc, live, _r, _n = _H()._run(tmp_path, monkeypatch, cand_wins=6,
                                     n_fixtures=70, n_tools=3,
                                     mutate="one")
        assert rc == 0 and len(live) == 1
        g = json.loads(live[0].read_text())["gate"]
        assert g["gate_scope"].startswith("solo"), g["gate_scope"]
        assert "no per-component contribution" not in g["gate_scope"]

    def test_a_MULTI_change_still_disclaims_it(self, tmp_path, monkeypatch):
        """The pair. A `gate_scope` that always says "solo" is a worse
        record than the one round 15 replaced."""
        rc, live, _r, _n = _H()._run(tmp_path, monkeypatch, cand_wins=6,
                                     n_fixtures=70, n_tools=3,
                                     mutate=True)
        assert rc == 0 and len(live) == 3, [p.name for p in live]
        for a in live:
            g = json.loads(a.read_text())["gate"]
            assert "no per-component contribution" in g["gate_scope"], g
            assert len(g["co_promoted"]) == 3, g


# ─────────────────────────────────────────────────────────────────────
# 3. Battery survivors on the round-14 surface.
# ─────────────────────────────────────────────────────────────────────
class TestAStripIdenticalCandidateIsNotAChange:
    """⚠ M02: dropping BOTH `.strip()`s from the `_changed` comparison
    survived. `loader.py` stores `opt.strip()`, so a candidate differing
    from the incumbent only in leading/trailing whitespace renders
    byte-identically in production — promoting it writes a fresh
    `promoted_utc` and a fresh sha, resetting the live check's era for a
    prompt nobody changed."""

    def test_whitespace_only_is_unchanged(self, tmp_path, monkeypatch,
                                          capsys):
        rc, live, _r, _n = _H()._run(tmp_path, monkeypatch, cand_wins=6,
                                     n_fixtures=70, mutate="whitespace")
        err = capsys.readouterr().err
        assert not live, (
            "a candidate that differs from the incumbent only in "
            "surrounding whitespace was promoted: "
            + str([p.name for p in live]))
        assert rc == 3, (rc, err)

    def test_an_INTERNAL_whitespace_change_IS_a_change(self, tmp_path,
                                                       monkeypatch):
        """The pair: the read site preserves internal whitespace, so a
        candidate that only reflows the text really is a different
        served prompt. A `.replace(" ", "")` comparison would call this
        unchanged and refuse a legitimate promotion."""
        rc, live, _r, _n = _H()._run(tmp_path, monkeypatch, cand_wins=6,
                                     n_fixtures=70, mutate="internal_ws")
        assert rc == 0 and len(live) == 1, [p.name for p in live]


class TestTheReDrawGuardBoundaryAndItsReportedAge:
    """⚠ M15 (`>` for `>=`) and M16 (`_young[stem] = 0.0`) both survived
    round 14's battery. The boundary is the one round 10 pinned for both
    power guards; the reported age is the only number an operator has
    to decide whether to wait."""

    def _promoted(self, tmp_path, monkeypatch):
        rc, live, _r, _n = _H()._run(tmp_path, monkeypatch, cand_wins=6,
                                     n_fixtures=70)
        assert rc == 0 and live, rc
        return live[0]

    def test_an_age_EXACTLY_at_the_cap_is_admitted(self, tmp_path,
                                                   monkeypatch, capsys):
        """`run_gepa`'s sibling refuses on `_age < cap`, so `== cap`
        runs. A `>=` here diverges from the gate it was ported from.

        ⚠ THE CLOCK IS FROZEN. With a wall clock the age is always a few
        microseconds OVER the cap, so `>=` and `>` take the same branch
        and the mutant survives — the boundary this test is named for was
        never reached. §4DA round 16."""
        import calendar
        import time as _time
        art = self._promoted(tmp_path, monkeypatch)
        capsys.readouterr()
        rec = json.loads(art.read_text())
        _stamp = _iso_days_ago(3.0)
        rec["gate"]["promoted_utc"] = _stamp
        art.write_text(json.dumps(rec))
        _exact = calendar.timegm(
            _time.strptime(_stamp, "%Y-%m-%dT%H:%M:%SZ")) + 3.0 * 86400.0

        def _freeze(mod):
            monkeypatch.setattr(
                mod, "time",
                _FrozenClock(_exact))
        rc, live, _r, _n = _H()._run(
            tmp_path, monkeypatch, cand_wins=6, n_fixtures=70,
            age_days="DEFAULT", on_module=_freeze,
            extra_argv=("--min-promotion-age-days", "3.0"))
        assert rc == 0, (
            "an artifact exactly at the cap was refused; the sibling "
            "gate admits it: " + capsys.readouterr().err)

    def test_one_TICK_under_the_cap_is_refused(self, tmp_path, monkeypatch,
                                               capsys):
        """The pair, on the same frozen clock: the boundary must be a
        boundary, not an always-admit."""
        import calendar
        import time as _time
        art = self._promoted(tmp_path, monkeypatch)
        capsys.readouterr()
        rec = json.loads(art.read_text())
        _stamp = _iso_days_ago(3.0)
        rec["gate"]["promoted_utc"] = _stamp
        art.write_text(json.dumps(rec))
        _just_under = calendar.timegm(
            _time.strptime(_stamp, "%Y-%m-%dT%H:%M:%SZ")) + 3.0 * 86400.0 - 1

        def _freeze(mod):
            monkeypatch.setattr(
                mod, "time",
                _FrozenClock(_just_under))
        rc, live, _r, _n = _H()._run(
            tmp_path, monkeypatch, cand_wins=6, n_fixtures=70,
            age_days="DEFAULT", on_module=_freeze,
            extra_argv=("--min-promotion-age-days", "3.0"))
        assert rc == 2, (rc, capsys.readouterr().err)

    def test_the_refusal_reports_the_REAL_age(self, tmp_path, monkeypatch,
                                              capsys):
        """⚠ `_young[stem] = 0.0` survived because the only pin asserted
        the literal "was promoted 0.0 days ago" — the fresh case's own
        value. A 6.4-day-old artifact reported as 0.0 passed."""
        art = self._promoted(tmp_path, monkeypatch)
        capsys.readouterr()
        rec = json.loads(art.read_text())
        rec["gate"]["promoted_utc"] = _iso_days_ago(6.4)
        art.write_text(json.dumps(rec))
        rc, live, _r, _n = _H()._run(
            tmp_path, monkeypatch, cand_wins=6, n_fixtures=70,
            age_days="DEFAULT")
        err = capsys.readouterr().err
        assert rc == 2, (rc, err)
        assert "6.4 days ago" in err, (
            "the refusal reported an age that is not the artifact's: "
            + err)


class _FrozenClock:
    """A stand-in for the script's `time` module with a fixed `time()`.

    Everything else delegates, so `strftime`/`gmtime`/`strptime` keep
    working — a shim that answers only the call the test cares about
    breaks the promotion path two hundred lines later.
    """

    def __init__(self, now):
        import time as _t
        self._t, self._now = _t, now

    def time(self):
        return self._now

    def __getattr__(self, name):
        return getattr(self._t, name)


def _iso_days_ago(days: float) -> str:
    import datetime as _dt
    return (_dt.datetime.now(_dt.timezone.utc)
            - _dt.timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")


class TestTheNoneDescriptionGuardCoversTheWITHHELDArm:
    """⚠ M38: dropping `or ""` from `baseline = fn.get("description") or
    ""` survived, because round 13's "both arms" pin makes the artifact
    APPLY — so `tuned != baseline` and the withheld `else:` branch is
    never entered. Round 11 widened the guard from one arm to both and
    the pin covered one arm."""

    def test_a_None_description_on_a_WITHHELD_turn_does_not_raise(
            self, tmp_path, monkeypatch):
        from ghost_agent.core import experiments as EXP
        home = tmp_path / "home"
        (home / "system" / "optim").mkdir(parents=True)
        _base = next(t for t in R.TOOL_DEFINITIONS
                     if t["function"]["name"] == "web_search"
                     )["function"]["description"]
        (home / "system" / "optim"
         / "tool_description.web_search.json").write_text(json.dumps({
             "signature_name": "tool_description.web_search",
             "optimized_instruction": _base + " Prefer it for news.",
             "gate_arm": "g"}))
        monkeypatch.setenv("GHOST_HOME", str(home))
        monkeypatch.setattr(R, "_TUNED_DESC_NAMES", None, raising=False)
        # ⚠ FORCE THE WITHHELD ARM. Without this the artifact applies and
        # the `else:` branch — the one round 11 widened — never runs.
        monkeypatch.setattr(L, "_resolve_arm", lambda *a, **k: "control")
        EXP.reset_registry_cache()
        L.clear_cache()
        tools = [{"type": "function",
                  "function": {"name": "web_search", "description": None,
                               "parameters": {}}}]
        out = R._apply_tuned_descriptions(tools, context=object())
        assert isinstance(out, list) and len(out) == 1
        L.clear_cache()


class TestTheDocumentedFlagListIsTheREALOne:
    """⚠ `docs/cli_reference.html` is the ONLY HTML surface for this
    runner (`docs/self-improvement.html` has zero mentions of
    `tool_description`, §4DA or McNemar), and its Flags list omitted
    `--min-promotion-age-days` — the sole new reason the script refuses
    to start, defaulting to ON — while the `run_gepa.py` row one line
    above listed it. `--allow-insignificant-ship`, `--fixtures`,
    `--recordings` and `--upstream-url` were missing too."""

    @staticmethod
    def _argparse_flags(path):
        import ast
        src = Path(path).read_text()
        out = set()
        for node in ast.walk(ast.parse(src)):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "attr", "") == "add_argument"):
                for a in node.args:
                    v = getattr(a, "value", None)
                    if isinstance(v, str) and v.startswith("--"):
                        out.add(v)
        return out

    def test_every_flag_the_script_accepts_is_documented(self):
        import re
        real = self._argparse_flags("scripts/optimize_tool_descriptions.py")
        assert "--min-promotion-age-days" in real, real
        html = Path("docs/cli_reference.html").read_text()
        # The runner's OWN row — another row mentions the script in prose.
        rows = [r for r in html.split("<tr>")
                if r.lstrip().startswith(
                    "<td><code>scripts/optimize_tool_descriptions.py")]
        assert len(rows) == 1, len(rows)
        doc = set(re.findall(r"<code>(--[a-z-]+)</code>", rows[0]))
        assert not (real - doc), (
            "flags the script accepts and the only HTML surface for it "
            "does not name: " + str(sorted(real - doc)))
        assert not (doc - real), (
            "flags documented that the script does not accept: "
            + str(sorted(doc - real)))


class TestTheTwoGATESShareTheirOwnExitContract:
    """⚠ The first version of this class asserted that all four
    instruments share ONE contract, and checked it by `ast.unparse`-ing
    `main()` and looking for the characters "0","1","2","3" — a
    source-shape pin, in the round whose headline is "the RETIRE code was
    pinned by its own source text". It was false as well as unfalsifiable:
    the GATES and the JUDGES answer different questions, and their 0/1
    are inverted (a gate's 0 means the incumbent was REPLACED; a judge's
    0 means it STANDS). What the four do share is 2 = could not measure.
    The judges' four codes are driven in
    `tests/test_4da_round16_fixes.py`; this drives the gates'.
    """

    def test_gate2_refuses_with_2_and_rejects_with_1(self, tmp_path,
                                                     monkeypatch, capsys):
        rc_reject, live, _r, _n = _H()._run(tmp_path / "a", monkeypatch,
                                            cand_wins=0, inc_wins=6,
                                            n_fixtures=70)
        capsys.readouterr()
        rc_refuse, _l, _r2, _n2 = _H()._run(
            tmp_path / "b", monkeypatch, cand_wins=6, n_fixtures=70,
            min_delta="0")
        capsys.readouterr()
        assert rc_reject == 1 and not live, rc_reject
        assert rc_refuse == 2, rc_refuse

    def test_gate1_uses_2_for_the_SAME_class_of_refusal(self, tmp_path,
                                                        capsys):
        """⚠ `run_gepa.py` returned 1 — "the gate rejected the candidate"
        — from five states in which nothing was ever scored: an unusable
        margin, an empty holdout, a tier below the combined need, and the
        re-draw guard. That is the collision rounds 11/13/15 carved codes
        out for in the judges, left whole in the gate the §4CY rule was
        ported FROM. Driven for the margin, which needs no corpus."""
        from tests.test_gepa_optim_reaudit import _corpus, _drive, _result
        _corpus(tmp_path / "traj")
        rc, _s = _drive(
            ["--signature", "planning.decompose",
             "--trajectories", str(tmp_path / "traj"),
             "--output", str(tmp_path / "optim" / "planning.decompose.json"),
             "--ab-min-delta", "0"],
            gepa_result=_result())
        err = capsys.readouterr().err
        assert rc == 2, (rc, err)
        assert "not a usable margin" in err, err


class TestTheRecheckValidatesItsOwnMargin:
    """⚠ Its docstring said so: "`--min-delta`, which this script does
    not validate." A NEGATIVE margin makes `cmp.delta > _margin`
    trivially true, so the instrument an operator uses to decide whether
    to RETIRE prints "IT STILL EARNS ITS PLACE" about an artifact losing
    by -0.40. Both ship gates already refuse the same values."""

    @pytest.mark.parametrize("bad", ["-0.4", "0", "1.0", "1e-320"])
    def test_an_unusable_margin_refuses(self, tmp_path, bad, capsys):
        rc = _RD()._run(tmp_path, delta=-0.40, ships=False, bw=8, cw=0,
                        min_delta=float(bad))
        assert rc == 2, (bad, rc)
        assert "not a usable margin" in capsys.readouterr().err

    def test_a_USABLE_margin_still_runs(self, tmp_path, capsys):
        """The pair — a validator that refuses everything is not one."""
        rc = _RD()._run(tmp_path, delta=-0.40, ships=False, bw=8, cw=0,
                        min_delta=0.02)
        assert rc == 1, (rc, capsys.readouterr().out)


class TestTheReplayDeadlineIsAFlag:
    """⚠ The adapter's 120s was hard-coded while `run_gepa.py` and
    `recheck_gepa_incumbent.py` both replay at 360 — so a replay THIS
    gate calls a transport failure (excluded from the statistic, and in
    bulk an abort) is one the instrument that re-checks the very same
    artifact scores normally."""

    def test_the_default_matches_the_siblings(self):
        import ast
        src = Path("scripts/optimize_tool_descriptions.py").read_text()
        got = None
        for node in ast.walk(ast.parse(src)):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "attr", "") == "add_argument"
                    and node.args
                    and getattr(node.args[0], "value", "") == "--timeout"):
                got = {k.arg: ast.unparse(k.value) for k in node.keywords}
        assert got is not None, "--timeout is not a flag"
        rc_src = Path("scripts/recheck_gepa_incumbent.py").read_text()
        sib = None
        for node in ast.walk(ast.parse(rc_src)):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "attr", "") == "add_argument"
                    and node.args
                    and getattr(node.args[0], "value", "") == "--timeout"):
                sib = {k.arg: ast.unparse(k.value) for k in node.keywords}
        assert float(got["default"]) == float(sib["default"]), (got, sib)

    def test_the_flag_REACHES_the_adapter(self, tmp_path, monkeypatch):
        """A flag parsed and dropped en route is the shape §4DA round 6
        found in `--components`. Driven: the adapter the RUN built must
        carry the value."""
        import tests.test_4da_tool_desc_ship_gate as _S
        _S.LAST_ADAPTER.clear()
        _H()._run(tmp_path, monkeypatch, cand_wins=6, n_fixtures=70,
                  extra_argv=("--timeout", "17.5"))
        assert _S.LAST_ADAPTER.get("timeout") == 17.5, _S.LAST_ADAPTER

    def test_the_DEFAULT_reaches_it_too(self, tmp_path, monkeypatch):
        """The pair: hard-coding 17.5 at the call site would pass the
        test above."""
        import tests.test_4da_tool_desc_ship_gate as _S
        _S.LAST_ADAPTER.clear()
        _H()._run(tmp_path, monkeypatch, cand_wins=6, n_fixtures=70)
        assert _S.LAST_ADAPTER.get("timeout") == 360.0, _S.LAST_ADAPTER


class TestTheGateScopeBoundaryIsONE:
    """⚠ BATTERY SURVIVOR. `len(_changed) == 1` -> `<= 2` survived the
    whole §4DA suite: with TWO components changed, each artifact claimed
    *"the A/B differed from the seed set in this component alone, so the
    numbers above ARE this component's contribution"* — false, and false
    in the OVER-claiming direction, on a record an operator weighs
    before retiring. Round 15's own pins drove 1 and 3 and skipped the
    only boundary that separates them."""

    def test_TWO_changed_components_is_a_SET(self, tmp_path, monkeypatch):
        rc, live, _r, _n = _H()._run(tmp_path, monkeypatch, cand_wins=6,
                                     n_fixtures=70, n_tools=3, mutate=2)
        assert rc == 0 and len(live) == 2, [p.name for p in live]
        for a in live:
            g = json.loads(a.read_text())["gate"]
            assert len(g["co_promoted"]) == 2, g
            assert "no per-component contribution" in g["gate_scope"], (
                "two co-promoted components claimed a solo measurement: "
                + g["gate_scope"])
