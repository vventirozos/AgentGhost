"""§4CZ — retiring a live artifact on production evidence.

The offline gate decides promotion on a few dozen held-out examples. This is
the other half: did the artifact help REAL turns, and if it measurably did
not, retire it.

Two things this must never do, both of which the session's earlier rounds
produced elsewhere: conclude from un-randomized data (an artifact deployed to
everything cannot be compared with the period before it), and render an
absence of evidence as evidence of equality.
"""

import json
import time
from pathlib import Path
import hashlib
from types import SimpleNamespace

import pytest

from ghost_agent.optim import live_check
from ghost_agent.optim.ab_eval import SHIP_ALPHA


#: The sha `optim/loader.py` stamps for an artifact whose
#: `optimized_instruction` is "T" — which is what every fixture in this
#: file writes. §4DA round 8 scoped the treatment arm to the LIVE
#: artifact's sha (pooling turns across promotions retired a healthy
#: artifact on its predecessor's evidence), so a fixture stamping an
#: arbitrary sha now describes a corpus production cannot produce: every
#: treatment turn would be dropped as stale.
SHA_OF_T = hashlib.sha256(b"T").hexdigest()[:8]


def _t(arm, outcome, sig="planning.decompose", sha=SHA_OF_T):
    return SimpleNamespace(
        outcome=outcome,
        extra={"optim_artifacts": {sig: {"sha": sha, "arm": arm}}})


def _mk(t_pass, t_fail, c_pass, c_fail, un=0):
    rows = ([_t("treatment", "passed")] * t_pass
            + [_t("treatment", "failed")] * t_fail
            + [_t("control", "passed", sha="")] * c_pass
            + [_t("control", "failed", sha="")] * c_fail
            + [_t("unenrolled", "passed")] * un)
    return live_check.collect(rows, "planning.decompose")


class TestTheStatistic:
    def test_it_matches_an_independent_computation(self):
        """Verified against `scipy.stats.fisher_exact(..., "less")` over
        396 random tables (and, by a second reviewer, 27,936 tables with
        margins 0-12 plus exact `Fraction` arithmetic) with zero
        mismatches. These are the values that check in by hand.

        ⚠ THE UNBALANCED TABLES ARE THE POINT. Every table here was once
        arm-balanced (n_t == n_c), which makes the hypergeometric's two
        margins indistinguishable — transposing them survived the whole
        suite while flipping REVERT/KEEP on 6,669 tables. Live arms are
        randomized per request and will not be balanced.
        """
        f = live_check.fisher_one_sided_worse
        assert f(2, 18, 15, 5) == pytest.approx(3.430314887e-05, abs=1e-9)
        assert f(15, 5, 2, 18) == pytest.approx(0.9999988951, abs=1e-9)
        assert f(5, 5, 5, 5) == pytest.approx(0.6718591007, abs=1e-9)
        assert f(0, 12, 4, 9) == pytest.approx(0.05652173913, abs=1e-9)
        assert f(4, 9, 0, 12) == pytest.approx(1, abs=1e-9)
        assert f(1, 15, 6, 6) == pytest.approx(0.01315496098, abs=1e-9)
        assert f(6, 6, 1, 15) == pytest.approx(0.9993311037, abs=1e-9)
        # The high tail must not be folded to exactly 1.0 — the module
        # promises that and the degenerate case alone did not pin it.
        assert f(15, 5, 2, 18) < 1.0

    def test_a_degenerate_margin_is_None_not_1(self):
        """`verdict-without-power`: nothing to compare is an absence of
        evidence, never evidence of equality."""
        f = live_check.fisher_one_sided_worse
        assert f(0, 0, 5, 5) is None
        assert f(5, 5, 0, 0) is None
        assert f(5, 0, 5, 0) is None      # everyone passed
        assert f(0, 5, 0, 5) is None      # everyone failed

    def test_counts_are_REJECTED_not_coerced(self):
        f = live_check.fisher_one_sided_worse
        with pytest.raises(ValueError):
            f(-1, 5, 5, 5)
        with pytest.raises(TypeError):
            f(1.5, 5, 5, 5)
        with pytest.raises(TypeError):
            f(True, 5, 5, 5)

    def test_it_is_UNPAIRED_by_construction(self):
        """⚠ The offline gate runs both prompts on the SAME examples, so a
        sign test is right there. Here the arms are different requests, so
        McNemar does not apply — reaching for the gate's statistic because
        it was to hand would produce a number for the wrong question. A
        paired test on these counts would give a different answer."""
        from ghost_agent.optim.ab_eval import mcnemar_p
        fish = live_check.fisher_one_sided_worse(2, 18, 15, 5)
        paired = mcnemar_p(15, 2, alternative="baseline")
        assert fish != pytest.approx(paired, rel=0.01)


class TestTheBarIsCoupledNotJustParameterised:
    def test_the_DEFAULT_alpha_IS_ship_alpha(self):
        """⚠ `test_the_bar_FOLLOWS_SHIP_ALPHA` passes `alpha` explicitly,
        so it tests the PARAMETER. Hardcoding the default to 0.05 survived
        it — and 0.05 is what SHIP_ALPHA happens to be, so the two are
        twins. This pins the coupling the module docstring claims."""
        import ghost_agent.optim.ab_eval as ab
        assert live_check.verdict.__kwdefaults__["alpha"] is ab.SHIP_ALPHA

    def test_a_p_EXACTLY_at_the_bar_reverts(self):
        """The admit side of the alpha bound. Only its reject side was
        pinned, so `<=` -> `<` survived — the same gap the MIN_PER_ARM
        test's own docstring warns about, one bound over."""
        p = live_check.fisher_one_sided_worse(3, 12, 12, 3)
        v = live_check.verdict(_mk(3, 12, 12, 3), alpha=p)
        assert v.verdict == "REVERT", (
            f"p == alpha ({p}) was treated as failing the bar")

    def test_MIN_PER_ARM_is_the_documented_value(self):
        assert live_check.MIN_PER_ARM == 12

    def test_the_docstrings_arithmetic_claim_is_TRUE(self):
        """⚠ The first version claimed Fisher "cannot reach SHIP_ALPHA"
        below 12/arm. False by 4x. The comment now says n=3, so check it."""
        f = live_check.fisher_one_sided_worse
        assert f(0, 3, 3, 0) == pytest.approx(0.05)
        assert f(0, 2, 2, 0) > live_check.SHIP_ALPHA


class TestBucketing:
    def test_arms_are_kept_apart(self):
        c = _mk(3, 1, 2, 2, un=5)
        assert (c.treatment.passed, c.treatment.failed) == (3, 1)
        assert (c.control.passed, c.control.failed) == (2, 2)
        assert c.unenrolled.n == 5

    def test_UNKNOWN_outcomes_are_dropped_not_counted_as_failures(self):
        """Scoring an unlabelled turn against the artifact would let a
        change in LABELLING rate masquerade as a change in quality."""
        rows = [_t("treatment", "passed"), _t("treatment", "unknown"),
                _t("treatment", ""), _t("treatment", "failed")]
        c = live_check.collect(rows, "planning.decompose")
        assert c.treatment.n == 2 and c.treatment.failed == 1

    def test_turns_for_another_signature_are_ignored(self):
        rows = [_t("treatment", "passed", sig="tool_selection.pick")]
        assert live_check.collect(rows, "planning.decompose").treatment.n == 0

    def test_unattributed_turns_are_ignored(self):
        rows = [SimpleNamespace(outcome="passed", extra={}),
                SimpleNamespace(outcome="passed", extra=None)]
        assert live_check.collect(rows, "planning.decompose").treatment.n == 0


class TestTheVerdict:
    def test_unenrolled_only_is_CONFOUNDED_never_a_number(self):
        """⚠ THE LOAD-BEARING REFUSAL. An artifact is deployed to every
        turn at once, so before/after is confounded by everything else that
        changed. Reporting a p here would be the same category of error as
        §4CY's margin-without-significance: a number that looks like a
        result."""
        v = live_check.verdict(_mk(0, 0, 0, 0, un=500))
        assert v.verdict == "CONFOUNDED"
        assert v.p_worse is None
        # ⚠ THE DETAIL NO LONGER PRESCRIBES A FIX. It used to say
        # "Register the experiment <name>" unconditionally, which is wrong
        # in five of the states the registry can be in — including the
        # correct one. Naming the cause is `registry_diagnosis`'s job,
        # which reads the registry; this string only states what is true
        # from the corpus alone.
        assert "none randomized" in v.detail
        assert "Register the experiment" not in v.detail

    def test_ONE_empty_arm_is_not_called_confounded(self):
        """20 randomized treatment turns and no control is INSUFFICIENT —
        there IS randomized data, just not both halves. Reporting
        "none randomized" would be a false statement to the operator."""
        v = live_check.verdict(_mk(20, 0, 0, 0, un=5))
        assert v.verdict == "INSUFFICIENT"
        assert "none randomized" not in v.detail

    def test_the_artifact_shas_reach_the_report(self):
        """`cmp.shas` is the only surface where artifact identity reaches
        the operator, and nothing asserted it."""
        c = _mk(3, 1, 2, 2)
        assert c.shas == {SHA_OF_T: 4}

    def test_a_SUPERSEDED_artifacts_turns_are_not_pooled_in(self):
        """⚠ MEASURED: `collect` walks the whole trajectory history, and
        re-promoting an already-live signature is the normal case — so
        the treatment arm pooled turns served by artifacts that had since
        been replaced, and `--revert` acted on the pooled verdict. A
        superseded artifact's 20 turns pooled with the current one's 20
        gave REVERT at p=0.0065 where the current artifact alone is KEEP
        at p=0.6122: the healthy artifact retired on the evidence of the
        one it replaced."""
        old_sha, new_sha = "0ldbad00", SHA_OF_T
        rows = ([_t("treatment", "failed", sha=old_sha)] * 18
                + [_t("treatment", "passed", sha=old_sha)] * 2
                + [_t("treatment", "passed", sha=new_sha)] * 14
                + [_t("treatment", "failed", sha=new_sha)] * 6
                + [_t("control", "passed")] * 28
                + [_t("control", "failed")] * 12)
        pooled = live_check.verdict(
            live_check.collect(rows, "planning.decompose"))
        scoped = live_check.verdict(
            live_check.collect(rows, "planning.decompose", sha=new_sha))
        assert pooled.treatment.n == 40 and pooled.verdict == "REVERT"
        assert scoped.treatment.n == 20, scoped.treatment
        assert scoped.stale_treatment == 20
        assert scoped.verdict == "KEEP", (scoped.verdict, scoped.p_worse)

    def test_CONTROL_turns_ARE_filtered_by_era(self):
        """⚠ THIS PIN ASSERTED THE OPPOSITE FOR TWO ROUNDS, in the words
        §4DA round 10 had to refute: "control turns carry `sha=""` **by
        construction**, so they cannot be partitioned — and must not be,
        since the control population is the same whichever artifact is
        live." Both halves are false. `loader._note_served` stamps the
        WITHHELD artifact's sha on control turns, and scoping only
        treatment made it a time window against a control arm of all
        history — measured, a contemporaneous KEEP (p=0.6238) read as a
        REVERT (p=0.0148) that `--revert` acted on.

        It passed only because its fixture built `sha=""` control rows,
        a corpus the loader can no longer produce. Round 10 corrected the
        code comment, the journal and the §4DA pin; this §4CZ one was
        missed — and it is the artifact a future reviewer would read as
        evidence AGAINST the change."""
        rows = ([_t("control", "passed", sha="0ther000")] * 20
                + [_t("control", "passed", sha=SHA_OF_T)] * 12
                + [_t("treatment", "passed")] * 20)
        c = live_check.collect(rows, "planning.decompose", sha=SHA_OF_T)
        assert c.control.n == 12, c.control
        assert c.stale_control == 20
        assert c.treatment.n == 20 and c.stale_treatment == 0

    def test_a_LEGACY_turn_without_a_sha_is_DROPPED_from_both_arms(self):
        """⚠ THIS PIN ASSERTED THE OPPOSITE, AND ITS RATIONALE WAS
        INVERTED. I wrote "those turns must not vanish — that would be
        the same defect one migration later." But no path through
        `tuned_instruction` emits an empty sha any more, and legacy
        TREATMENT turns always carried a real one — so they were already
        dropped as stale, and exempting the empty-sha turns exempted the
        CONTROL arm alone. Keeping them WAS the de-randomization.

        Driven: era-B-only is `10/20 vs 10/20, KEEP p=0.6238`; era B
        plus a pre-stamp corpus is `10/20 vs 40/50, REVERT p=0.0148`,
        with `--revert` retiring on it.

        The symmetric options are drop-both or scope-neither.
        Control-only is neither."""
        rows = ([_t("control", "passed", sha="")] * 20
                + [_t("treatment", "passed")] * 20)
        c = live_check.collect(rows, "planning.decompose", sha=SHA_OF_T)
        assert c.control.n == 0, c.control
        assert c.stale_control == 20 and c.stale_unstamped == 20
        assert c.treatment.n == 20

    def test_an_UNSCOPED_call_keeps_everything(self):
        """The other symmetric option, and the default for any caller
        that does not pass a sha."""
        rows = ([_t("control", "passed", sha="")] * 20
                + [_t("treatment", "passed")] * 20)
        c = live_check.collect(rows, "planning.decompose")
        assert c.control.n == 20 and c.treatment.n == 20
        assert c.stale_control == 0 and c.stale_unstamped == 0

    def test_NO_sha_argument_keeps_the_old_pooling(self):
        """The default must not silently change for any other caller."""
        rows = [_t("treatment", "passed", sha="0ther000")] * 5
        c = live_check.collect(rows, "planning.decompose")
        assert c.treatment.n == 5 and c.stale_treatment == 0

    def test_no_data_at_all_is_INSUFFICIENT(self):
        v = live_check.verdict(_mk(0, 0, 0, 0))
        assert v.verdict == "INSUFFICIENT" and v.p_worse is None

    def test_ONE_arm_below_the_floor_is_enough_to_refuse(self):
        """⚠ `t.n < floor OR c.n < floor` -> `AND` survived: no test had
        one arm above and one below. Under that mutant a 3-turn control
        against 20 treatment turns computes Fisher and REVERTS a live
        artifact — and an asymmetric split is the FIRST shape live
        randomization produces."""
        v = live_check.verdict(_mk(0, 20, 3, 0), min_per_arm=12)
        assert v.verdict == "INSUFFICIENT", (
            "a verdict was computed against a 3-turn control arm")
        v2 = live_check.verdict(_mk(0, 3, 20, 0), min_per_arm=12)
        assert v2.verdict == "INSUFFICIENT"

    def test_a_floor_of_zero_is_clamped(self):
        """`--min-per-arm 0` returned KEEP off "treatment 0/0 vs control
        15/20" — a verdict about an arm with no turns in it."""
        v = live_check.verdict(_mk(0, 0, 15, 5), min_per_arm=0)
        assert v.verdict != "KEEP"

    def test_below_the_floor_is_INSUFFICIENT(self):
        v = live_check.verdict(_mk(1, 9, 9, 1), min_per_arm=12)
        assert v.verdict == "INSUFFICIENT", (
            "a verdict was computed on 10 turns per arm")

    def test_EXACTLY_at_the_floor_is_judged(self):
        """The admit side. Only the reject side of a bound is usually
        pinned, which is how `>=` -> `>` survives."""
        v = live_check.verdict(_mk(0, 12, 12, 0), min_per_arm=12)
        assert v.verdict == "REVERT"

    def test_a_significant_loss_REVERTS(self):
        v = live_check.verdict(_mk(2, 18, 15, 5))
        assert v.verdict == "REVERT"
        assert v.p_worse is not None and v.p_worse <= SHIP_ALPHA
        assert "Fisher one-sided" in v.detail

    def test_a_significant_WIN_does_not_revert(self):
        v = live_check.verdict(_mk(15, 5, 2, 18))
        assert v.verdict == "KEEP"

    def test_a_wash_does_not_revert(self):
        v = live_check.verdict(_mk(10, 10, 10, 10))
        assert v.verdict == "KEEP"

    def test_KEEP_with_no_testable_split_says_so(self):
        """⚠ KEEP must not read as "the artifact is fine" when nothing
        could have distinguished the arms."""
        v = live_check.verdict(_mk(20, 0, 20, 0))
        assert v.verdict == "KEEP" and v.p_worse is None
        assert "absence of evidence" in v.detail

    def test_the_bar_FOLLOWS_SHIP_ALPHA(self):
        """Move the constant: a table at p just under 0.2 must revert at a
        0.2 bar and not at the 0.05 one. Passing `alpha` explicitly is the
        same read the caller's default makes."""
        c = _mk(5, 10, 10, 5)
        p = live_check.fisher_one_sided_worse(5, 10, 10, 5)
        assert 0.05 < p <= 0.2, f"table chosen badly: p={p}"
        assert live_check.verdict(_mk(5, 10, 10, 5),
                                  alpha=0.2).verdict == "REVERT"
        assert live_check.verdict(_mk(5, 10, 10, 5),
                                  alpha=0.05).verdict == "KEEP"


class TestTheScriptOnlyActsOnREVERT:
    def _run(self, tmp_path, rows, argv_extra=()):
        import importlib.util
        import sys as _sys
        home = tmp_path / "home"
        (home / "system" / "trajectories").mkdir(parents=True)
        art_dir = home / "system" / "optim"
        art_dir.mkdir(parents=True)
        art = art_dir / "planning.decompose.json"
        art.write_text(json.dumps({"optimized_instruction": "T"}))
        spec = importlib.util.spec_from_file_location(
            "glc", "scripts/gepa_live_check.py")
        mod = importlib.util.module_from_spec(spec)
        _sys.modules["glc"] = mod
        spec.loader.exec_module(mod)

        class _Coll:
            def __init__(self, **kw): pass
            def iter_trajectories(self): return iter(rows)
        mod.TrajectoryCollector = _Coll
        old = _sys.argv
        try:
            _sys.argv = ["glc", "--home", str(home), *argv_extra]
            rc = mod.main()
        finally:
            _sys.argv = old
        return rc, art

    def test_the_retirement_says_it_needs_a_RESTART(self, tmp_path,
                                                     capsys):
        """⚠ THE ONE ACTION §4CZ CAN TAKE WAS REPORTED AS DONE WHEN IT HAD
        NOT HAPPENED. `loader` caches the artifact text per process and its
        `clear_cache()` must not be called on a live agent, so a rename
        takes effect only on restart — meanwhile every planner turn keeps
        using the retired artifact and `activation_stats` keeps counting it
        as applied. `claim-vs-fact-deliverables`."""
        rows = ([_t("treatment", "failed")] * 18
                + [_t("treatment", "passed")] * 2
                + [_t("control", "passed")] * 15
                + [_t("control", "failed")] * 5)
        self._run(tmp_path, rows, ("--revert",))
        out = capsys.readouterr().out
        assert "RETIRED ON DISK" in out
        assert "still serving it" in out.lower()
        assert "launchctl kickstart" in out
        assert "REMOVES A PREFIX" in out and "prepends" in out, (
            "the retirement's effect on the prompt was misdescribed")

    def test_the_report_body_labels_each_arm_correctly(self, tmp_path,
                                                       capsys):
        """⚠ Nothing asserted the printed arm counts, so swapping the
        treatment and control lines survived — a losing artifact would
        read as WINNING, right next to a REVERT verdict."""
        rows = ([_t("treatment", "failed")] * 18
                + [_t("treatment", "passed")] * 2
                + [_t("control", "passed")] * 15
                + [_t("control", "failed")] * 5)
        rc, _art = self._run(tmp_path, rows)
        # §4DA round 15: the three GEPA instruments share one exit
        # contract — 0 still earns its place, 1 it does not,
        # 2 could not measure, 3 reported but not acted on.
        assert rc == 1, rc
        out = capsys.readouterr().out
        assert "treatment : 2/20" in out, (
            "the treatment line does not show the treatment arm")
        assert "control   : 15/20" in out
        assert "REVERT" in out
        # Two load-bearing theses in the report body, both invertible with
        # the suite green: unenrolled turns are NOT a control group, and
        # retiring REMOVES a prefix rather than adding one.
        assert "NOT a control group" in out

    def test_a_losing_artifact_is_RETIRED_with_revert(self, tmp_path):
        rows = ([_t("treatment", "failed")] * 18 + [_t("treatment", "passed")] * 2
                + [_t("control", "passed")] * 15 + [_t("control", "failed")] * 5)
        rc, art = self._run(tmp_path, rows, ("--revert",))
        # §4DA round 15: the three GEPA instruments share one exit
        # contract — 0 still earns its place, 1 it does not,
        # 2 could not measure, 3 reported but not acted on.
        assert rc == 1, rc
        assert not art.exists(), "the artifact was not retired"
        assert list(art.parent.glob("*.retired-live-*")), "no retired copy"

    def test_without_revert_NOTHING_is_written(self, tmp_path):
        rows = ([_t("treatment", "failed")] * 18 + [_t("treatment", "passed")] * 2
                + [_t("control", "passed")] * 15 + [_t("control", "failed")] * 5)
        rc, art = self._run(tmp_path, rows)
        # §4DA round 15: the three GEPA instruments share one exit
        # contract — 0 still earns its place, 1 it does not,
        # 2 could not measure, 3 reported but not acted on.
        assert rc == 1 and art.exists(), rc
        assert not list(art.parent.glob("*.retired-live-*"))

    def test_the_SIGNATURE_flag_selects_both_the_data_and_the_artifact(
            self, tmp_path, capsys):
        """⚠ `--signature` was unpinned: every script test used the
        default. Hardcoding the signature in the `collect()` call survived
        all 192 tests — and since the artifact path is built from
        `args.signature`, `--signature tool_selection.pick --revert` would
        compute planning.decompose's verdict and RETIRE THE WRONG
        ARTIFACT. `tool_selection.pick` is a real attributed signature."""
        import importlib.util
        import sys as _sys
        home = tmp_path / "home"
        (home / "system" / "trajectories").mkdir(parents=True)
        opt = home / "system" / "optim"
        opt.mkdir(parents=True)
        for sig in ("planning.decompose", "tool_selection.pick"):
            (opt / f"{sig}.json").write_text(json.dumps(
                {"optimized_instruction": "T"}))
        # Only tool_selection.pick is losing; planning.decompose is clean.
        rows = ([_t("treatment", "failed", sig="tool_selection.pick")] * 18
                + [_t("treatment", "passed", sig="tool_selection.pick")] * 2
                + [_t("control", "passed", sig="tool_selection.pick")] * 15
                + [_t("control", "failed", sig="tool_selection.pick")] * 5
                + [_t("treatment", "passed")] * 20
                + [_t("control", "passed")] * 20)
        spec = importlib.util.spec_from_file_location(
            "glc4", "scripts/gepa_live_check.py")
        mod = importlib.util.module_from_spec(spec)
        _sys.modules["glc4"] = mod
        spec.loader.exec_module(mod)

        class _Coll:
            def __init__(self, **kw): pass
            def iter_trajectories(self): return iter(rows)
        mod.TrajectoryCollector = _Coll
        old_argv = _sys.argv
        try:
            _sys.argv = ["glc4", "--home", str(home), "--signature",
                         "tool_selection.pick", "--revert"]
            rc = mod.main()
        finally:
            _sys.argv = old_argv
        # §4DA round 15: the three GEPA instruments share one exit
        # contract — 0 still earns its place, 1 it does not,
        # 2 could not measure, 3 reported but not acted on.
        assert rc == 1, rc
        assert not (opt / "tool_selection.pick.json").exists(), (
            "the named signature's artifact was not retired")
        assert (opt / "planning.decompose.json").exists(), (
            "THE WRONG ARTIFACT WAS RETIRED — the verdict was computed "
            "for a different signature than the one named")
        assert "tool_selection.pick" in capsys.readouterr().out

    def _confounded_with_registry(self, tmp_path, spec, capsys,
                                  rows=None):
        """Drive the script with `spec` (or None) in the registry.

        ⚠ `rows` DEFAULTS TO AN ALL-UNENROLLED CORPUS, WHICH SOME
        REGISTRIES CAN NEVER PRODUCE. A one-known-arm registry stamps
        ~half the turns with the known arm, so the real verdict there is
        INSUFFICIENT, not CONFOUNDED — and for a while the diagnosis was
        gated on CONFOUNDED alone, so it never printed for the state it
        was written for while its test passed on a corpus that registry
        cannot generate. Pass `rows` to match the registry under test.
        """
        import importlib.util
        import sys as _sys
        home = tmp_path / "home"
        (home / "system" / "trajectories").mkdir(parents=True)
        (home / "system" / "optim").mkdir(parents=True)
        (home / "system" / "optim" / "planning.decompose.json").write_text(
            json.dumps({"optimized_instruction": "T"}))
        if spec is not None:
            (home / "system" / "experiments.json").write_text(json.dumps(
                {"salt": "t", "experiments": [spec]}))
        rows = rows if rows is not None else [_t("unenrolled",
                                                  "failed")] * 40
        mod_name = f"glc_{abs(hash(str(spec) + str(len(rows)))) % 99999}"
        m_spec = importlib.util.spec_from_file_location(
            mod_name, "scripts/gepa_live_check.py")
        mod = importlib.util.module_from_spec(m_spec)
        _sys.modules[mod_name] = mod
        m_spec.loader.exec_module(mod)

        class _Coll:
            def __init__(self, **kw): pass
            def iter_trajectories(self): return iter(rows)
        mod.TrajectoryCollector = _Coll
        old_argv = _sys.argv
        try:
            _sys.argv = [mod_name, "--home", str(home)]
            mod.main()
        finally:
            _sys.argv = old_argv
        return capsys.readouterr().out

    def _name(self):
        from ghost_agent.optim.loader import experiment_name
        return experiment_name("planning.decompose")

    def test_a_DISABLED_experiment_is_named_as_the_cause(self, tmp_path,
                                                         capsys):
        """⚠ `load_registry().specs` KEEPS disabled specs and `assign`
        returns "" for them, so a disabled experiment is indistinguishable
        from an unregistered one at the call site — and the report told
        the operator to register what they had already registered.
        Nothing else in the codebase reports a disabled spec, so this is
        their only feedback channel."""
        out = self._confounded_with_registry(tmp_path, {
            "name": self._name(), "arms": ["control", "treatment"],
            "traffic": 1.0, "enabled": False}, capsys)
        assert "REGISTERED BUT DISABLED" in out
        # ⚠ THE FIRST VERSION OF THIS WAS STRUCTURALLY UNFALSIFIABLE: it
        # searched a 29-character fragment before the banner. Assert the
        # whole output, where the contradiction actually lived — the
        # "register it" advice WAS present, on the last content line.
        assert "IS NOT REGISTERED" not in out

    def test_TRAFFIC_ZERO_is_named_as_the_cause(self, tmp_path, capsys):
        out = self._confounded_with_registry(tmp_path, {
            "name": self._name(), "arms": ["control", "treatment"],
            "traffic": 0.0, "enabled": True}, capsys)
        # ⚠ `"traffic=0" in out` also matched the correct-config branch's
        # partial-traffic clause, so removing the traffic-0 branch
        # survived. Assert the banner that only this branch prints.
        assert "REGISTERED WITH traffic=0" in out
        assert "Nothing is misconfigured" not in out

    def test_a_THREE_ARM_design_is_not_condemned_wholesale(self, tmp_path,
                                                            capsys):
        """⚠ The first version printed ALL arms and claimed "every arm was
        served the artifact and every turn recorded unenrolled" — false:
        the control turns were served the BASELINE and stamped `control`.
        Advising a re-register throws away a working two-arm subset."""
        out = self._confounded_with_registry(tmp_path, {
            "name": self._name(),
            "arms": ["control", "treatment", "aggressive"],
            "traffic": 1.0, "enabled": True}, capsys,
            # a real 3-arm split is ~1/3 each, not all-unenrolled
            rows=([_t("treatment", "failed")] * 5
                  + [_t("control", "failed", sha="")] * 5
                  + [_t("unenrolled", "failed")] * 5))
        assert "ARE usable" in out, (
            "a legitimate 3-arm design was condemned wholesale")
        # The verdict is printed BELOW the diagnosis, not above.
        assert "the verdict below" in out
        # Only the EXTRA arm may be named as served-the-artifact; naming
        # all three is the same falsehood one branch over.
        assert "turns assigned ['aggressive'] were served" in out
        assert "['control', 'treatment', 'aggressive'] were served" not in out
        assert "every arm was served the artifact" not in out

    def test_an_UNREGISTERED_experiment_still_says_register_it(
            self, tmp_path, capsys):
        """The admit side — the original message must survive for the case
        it was actually written for."""
        out = self._confounded_with_registry(tmp_path, None, capsys)
        # ⚠ `assert "Register the experiment" in out` was true in ALL
        # TWELVE registry states, so it could not distinguish the one it
        # names. The banner is what differs.
        assert "IS NOT REGISTERED" in out
        assert "DISABLED" not in out
        assert "IS REGISTERED AND ENABLED" not in out

    def test_a_CORRECT_registry_is_not_told_to_register(self, tmp_path,
                                                        capsys):
        """⚠ THE STATE THE CROSS-CHECK EXISTED FOR, AND THE ONE IT MISSED.
        With the exact spec the tool asks for — registered, enabled,
        traffic 1.0, arms ["control","treatment"] — the four-way check had
        no `else`, so it printed nothing and the fallback told the
        operator to register what was already registered. Reachable in the
        normal workflow: pre-registration turns are durable and graded
        while new randomized turns lag."""
        # ⚠ TRAFFIC 1.0 ENROLS 100% OF TURNS, so an all-unenrolled corpus
        # has probability zero under this registry — and running the pin
        # on one is how the on-ramp bug below hid. Two corpora, because
        # the branch must say different things about them.
        out = self._confounded_with_registry(tmp_path, {
            "name": self._name(), "arms": ["control", "treatment"],
            "traffic": 1.0, "enabled": True}, capsys)
        assert "IS REGISTERED AND ENABLED" in out
        assert "Nothing is misconfigured" in out
        assert "IS NOT REGISTERED" not in out, (
            "a correctly-registered experiment was told to register")
        # With NO randomized turns the durability guidance is the
        # actionable half: without it the operator re-runs, which cannot
        # help.
        assert "resolves as NEW turns arrive, not by re-running" in out
        # ⚠ THE PREFIX IS THE SIGNAL. Every other state opens with ⚠; this
        # is the one that says nothing is wrong, and swapping the prefix
        # erases that distinction while every word after it stays true.
        assert out.count("⚠ " + self._name()) == 0, (
            "the healthy state was rendered as a warning")
        assert "i " + self._name() in out

    def test_a_CORRECT_registry_ON_RAMP_does_not_claim_nothing_accrued(
            self, tmp_path, capsys):
        """⚠ THE STATE THE TOOL OCCUPIES FOR THE WHOLE ON-RAMP. Widening
        the gate to INSUFFICIENT made the healthy branch print "no
        randomized turn has accumulated a graded outcome yet" three lines
        above a verdict reading "treatment n=11, control n=11". Two
        adjacent lines contradicting each other is worse than the silence
        the widening replaced — and the pin could not see it because it
        ran on an all-unenrolled corpus this registry cannot produce."""
        rows = ([_t("treatment", "passed")] * 5
                + [_t("treatment", "failed")] * 6
                + [_t("control", "passed", sha="")] * 6
                + [_t("control", "failed", sha="")] * 5)
        out = self._confounded_with_registry(tmp_path, {
            "name": self._name(), "arms": ["control", "treatment"],
            "traffic": 1.0, "enabled": True}, capsys, rows=rows)
        assert "INSUFFICIENT" in out
        assert "22 turn(s) are already in an arm" in out
        # ⚠ THE COUNT MUST BE THE CORPUS'S, NOT A LITERAL. A hardcoded 22
        # printed above "treatment n=5, control n=4" is the same adjacent
        # contradiction this test exists to stop.
        assert "i " + self._name() in out, (
            "the healthy on-ramp state was rendered as a warning")
        assert "see the verdict below" in out, (
            "pointed at a verdict printed below as though it were above")
        assert "no randomized turn has accumulated" not in out, (
            "claimed nothing accrued while 22 randomized graded turns "
            "sat three lines above")
        assert "not by re-running" not in out, (
            "told the operator to wait for NEW turns when the arms are "
            "already filling")

    def test_the_on_ramp_count_is_the_CORPUS_count(self, tmp_path,
                                                   capsys):
        """A second `randomized` value, because one literal cannot tell a
        computed count from a constant — and `randomized > 0` -> `> 1`
        left a single randomized turn reported as none."""
        rows = ([_t("treatment", "passed")] * 3
                + [_t("control", "failed", sha="")] * 2)
        out = self._confounded_with_registry(tmp_path, {
            "name": self._name(), "arms": ["control", "treatment"],
            "traffic": 1.0, "enabled": True}, capsys, rows=rows)
        assert "5 turn(s) are already in an arm" in out
        assert "no randomized turn has accumulated" not in out

    def test_a_SINGLE_randomized_turn_is_not_reported_as_none(
            self, tmp_path, capsys):
        out = self._confounded_with_registry(tmp_path, {
            "name": self._name(), "arms": ["control", "treatment"],
            "traffic": 1.0, "enabled": True}, capsys,
            rows=[_t("treatment", "passed")])
        assert "1 turn(s) are already in an arm" in out
        assert "no randomized turn has accumulated" not in out

    def test_PARTIAL_traffic_is_explained(self, tmp_path, capsys):
        out = self._confounded_with_registry(tmp_path, {
            "name": self._name(), "arms": ["control", "treatment"],
            "traffic": 0.25, "enabled": True}, capsys)
        assert "traffic=0.25" in out
        assert "IS NOT REGISTERED" not in out
        # ⚠ 0.9 IS THE DISCRIMINATING CASE. With only 0.25 and 1.0 pinned,
        # `traffic >= 1.0` could become `> 0.5` and report 0.9 as full
        # traffic — the share of turns silently outside the experiment.
        out9 = self._confounded_with_registry(tmp_path / "b", {
            "name": self._name(), "arms": ["control", "treatment"],
            "traffic": 0.9, "enabled": True}, capsys,
            # at 0.9 an all-unenrolled corpus has probability ~1e-40
            rows=([_t("treatment", "failed")] * 9
                  + [_t("control", "failed", sha="")] * 9
                  + [_t("unenrolled", "failed")] * 2))
        assert "traffic=0.9" in out9, (
            "a partial traffic of 0.9 was reported as full enrolment")

    def test_ONE_known_arm_is_described_precisely(self, tmp_path, capsys):
        """⚠ The old text claimed "every arm was served the artifact and
        every turn recorded unenrolled" here too — false: with
        ["control","baseline"] the control turns were served the
        hand-written BASELINE and stamped `control`.

        ⚠⚠ AND THE CORPUS NOW MATCHES THE REGISTRY. Driven on an
        all-unenrolled corpus this reached CONFOUNDED, but that registry
        actually produces ~half `control` turns — so the real verdict is
        INSUFFICIENT and the diagnosis was gated away from the very state
        it describes."""
        # ⚠ THE CONTROL TURNS CARRY THE LIVE SHA. `sha=""` is a corpus
        # the loader can no longer produce, and §4DA round 12 drops
        # unstamped turns from BOTH arms — so a fixture built that way
        # would empty the control arm and change the verdict this test
        # is about.
        rows = ([_t("control", "failed")] * 20
                + [_t("unenrolled", "failed")] * 20)
        out = self._confounded_with_registry(tmp_path, {
            "name": self._name(), "arms": ["control", "baseline"],
            "traffic": 1.0, "enabled": True}, capsys, rows=rows)
        assert "INSUFFICIENT" in out, (
            "this registry produces control turns, so the verdict is not "
            "CONFOUNDED — the diagnosis must still print")
        assert "honours only 'control'" in out
        # ⚠ THE ARM NAMES, IN THEIR ROLES. Asserting only the framing
        # words let `other = list(arms)` and `usable[0] -> arms[-1]`
        # survive — messages that say the honoured arm was the one served
        # the artifact, i.e. exactly backwards.
        assert "turns assigned ['baseline'] were served the artifact" in out
        assert "the 'control' turns were handled correctly" in out
        assert "every arm was served the artifact" not in out
        # The actionable half — deleting it leaves a description with no
        # instruction.
        assert 'needs BOTH "control" and "treatment"' in out

    def test_a_REJECTED_spec_is_not_called_unregistered(self, tmp_path,
                                                        capsys):
        """⚠ `load_registry` SILENTLY DROPS a spec for at least six
        reasons — duplicate arms, arms not a list, >8 arms, a malformed
        arm name, an unknown scope, a bad/sensitive name — and all of them
        printed "IS NOT REGISTERED" to an operator whose file already
        contains the entry. Verbatim the defect round 4 fixed for the
        CORRECT branch, one branch over."""
        for arms in (["control", "control"], ["control", "tre atment"]):
            out = self._confounded_with_registry(tmp_path / str(arms), {
                "name": self._name(), "arms": arms,
                "traffic": 1.0, "enabled": True}, capsys)
            assert "REJECTED when it was loaded" in out, (
                f"arms={arms} was reported as never registered")
            assert "IS NOT REGISTERED" not in out
            # The reason list must include the cap this branch also fires
            # for, or the operator checks six causes and finds none.
            assert "more specs in the file than the registry's cap" in out

    def test_NO_ARTIFACT_dominates_every_registry_answer(self, tmp_path,
                                                         capsys):
        """⚠ THE STATE PRODUCTION IS IN, AND THE ONE SIX ROUNDS NEVER
        DROVE. The loader stamps only when there is something to serve, so
        with no artifact BOTH arms randomize correctly and ZERO turns are
        attributed — measured 20 turns, 0 stamped. Every registry sentence
        is then beside the point and two are actively false: "no
        randomized turn has accumulated" (they did; they were never
        stamped) and "this resolves as NEW turns arrive" (it never will).

        Every script-driving helper in this file writes an artifact first,
        which is exactly why the chain missed it."""
        import importlib.util
        import sys as _sys
        home = tmp_path / "home"
        (home / "system" / "trajectories").mkdir(parents=True)
        (home / "system" / "optim").mkdir(parents=True)   # NO artifact
        (home / "system" / "experiments.json").write_text(json.dumps(
            {"salt": "t", "experiments": [
                {"name": self._name(), "arms": ["control", "treatment"],
                 "traffic": 1.0, "enabled": True}]}))
        spec = importlib.util.spec_from_file_location(
            "glc_noart", "scripts/gepa_live_check.py")
        mod = importlib.util.module_from_spec(spec)
        _sys.modules["glc_noart"] = mod
        spec.loader.exec_module(mod)

        class _Coll:
            def __init__(self, **kw): pass
            def iter_trajectories(self): return iter([])
        mod.TrajectoryCollector = _Coll
        old_argv = _sys.argv
        try:
            _sys.argv = ["glc_noart", "--home", str(home)]
            mod.main()
        finally:
            _sys.argv = old_argv
        out = capsys.readouterr().out
        assert "THERE IS NO LIVE ARTIFACT" in out
        assert "run_gepa.py" in out, (
            "no actionable next step for the state production is in")
        assert "no randomized turn has accumulated" not in out
        assert "resolves as NEW turns arrive" not in out
        assert "IS REGISTERED AND ENABLED" not in out, (
            "diagnosed the registry when the artifact is what is missing")

    def test_WITH_an_artifact_the_registry_states_still_print(
            self, tmp_path, capsys):
        """The admit side: the artifact check must not swallow the
        registry diagnosis it precedes."""
        out = self._confounded_with_registry(tmp_path, {
            "name": self._name(), "arms": ["control", "treatment"],
            "traffic": 0.0, "enabled": True}, capsys)
        assert "THERE IS NO LIVE ARTIFACT" not in out
        assert "traffic=0" in out

    def test_a_registry_file_WITHOUT_our_spec_is_unregistered(
            self, tmp_path, capsys):
        """⚠ The unregistered pin used a run with NO registry file, so it
        returned before the name lookup — a mutant making that lookup
        always true survived. The discriminating case is a file that
        EXISTS and carries someone else's experiment."""
        out = self._confounded_with_registry(tmp_path, {
            "name": "some_other_experiment",
            "arms": ["control", "treatment"],
            "traffic": 1.0, "enabled": True}, capsys)
        assert "IS NOT REGISTERED" in out
        assert "REJECTED when it was loaded" not in out, (
            "a spec that is genuinely absent was reported as rejected")

    def test_the_name_lookup_normalises_case_and_whitespace(
            self, tmp_path, capsys):
        """`_spec_from_dict` does `.strip().lower()` before matching, so
        the file-scan must too — otherwise a spec written with padding or
        capitals is REJECTED by the registry and then reported as never
        registered. Both normalisations were unpinned."""
        for i, written in enumerate(("  " + self._name() + " ",
                                     self._name().upper())):
            out = self._confounded_with_registry(
                tmp_path / f"case{i}", {
                    "name": written, "arms": ["control", "control"],
                    "traffic": 1.0, "enabled": True}, capsys)
            assert "REJECTED when it was loaded" in out, (
                f"a spec named {written!r} was reported as never "
                f"registered")
            assert "IS NOT REGISTERED" not in out

    def test_a_JUNK_entry_before_ours_does_not_hide_it(self, tmp_path,
                                                       capsys):
        """⚠ `(e or {})` rescued a None entry but not a truthy non-dict,
        which `load_registry` deliberately tolerates — so a junk entry
        BEFORE ours made the scan raise into the blanket except and print
        "IS NOT REGISTERED" for a spec that is in the file. The same file
        with its entries reordered gave two different diagnoses."""
        import importlib.util
        import sys as _sys
        home = tmp_path / "home"
        (home / "system" / "trajectories").mkdir(parents=True)
        (home / "system" / "optim").mkdir(parents=True)
        (home / "system" / "optim" / "planning.decompose.json").write_text(
            json.dumps({"optimized_instruction": "T"}))
        (home / "system" / "experiments.json").write_text(json.dumps({
            "salt": "t", "experiments": [
                "junk", 7,
                {"name": self._name(), "arms": ["control", "control"],
                 "traffic": 1.0, "enabled": True}]}))
        rows = [_t("unenrolled", "failed")] * 40
        spec = importlib.util.spec_from_file_location(
            "glc_junk", "scripts/gepa_live_check.py")
        mod = importlib.util.module_from_spec(spec)
        _sys.modules["glc_junk"] = mod
        spec.loader.exec_module(mod)

        class _Coll:
            def __init__(self, **kw): pass
            def iter_trajectories(self): return iter(rows)
        mod.TrajectoryCollector = _Coll
        old_argv = _sys.argv
        try:
            _sys.argv = ["glc_junk", "--home", str(home)]
            mod.main()
        finally:
            _sys.argv = old_argv
        out = capsys.readouterr().out
        assert "REJECTED when it was loaded" in out, (
            "a junk entry before ours hid a spec that is in the file")
        assert "IS NOT REGISTERED" not in out

    def test_a_NON_LIVE_scope_is_named_as_the_cause(self, tmp_path,
                                                    capsys):
        """⚠ `assign_all` filters on scope and `enroll_request` asks for
        SCOPE_LIVE, so a bench-scoped spec never enrolls a user turn — and
        it reached the CORRECT branch, which says "this resolves as NEW
        turns arrive". Permanently false. The live registry already
        carries a bench-scoped spec as the copy-paste template."""
        out = self._confounded_with_registry(tmp_path, {
            "name": self._name(), "arms": ["control", "treatment"],
            "traffic": 1.0, "enabled": True, "scope": "bench"}, capsys)
        assert "scope='bench'" in out
        assert "never enrolls a user turn" in out
        assert "Nothing is misconfigured" not in out

    def test_a_REAL_verdict_prints_NO_registry_diagnosis(self, tmp_path,
                                                         capsys):
        """The admit side of the gate round 5 moved. Only its reject side
        was pinned, so widening it to `if True` — a registry lecture next
        to a REVERT verdict — survived."""
        rows = ([_t("treatment", "failed")] * 18
                + [_t("treatment", "passed")] * 2
                + [_t("control", "passed")] * 15
                + [_t("control", "failed")] * 5)
        rc, _art = self._run(tmp_path, rows)
        # §4DA round 15: the three GEPA instruments share one exit
        # contract — 0 still earns its place, 1 it does not,
        # 2 could not measure, 3 reported but not acted on.
        assert rc == 1, rc
        out = capsys.readouterr().out
        assert "REVERT" in out
        assert "IS NOT REGISTERED" not in out, (
            "a completed comparison was given a registry diagnosis")
        assert "IS REGISTERED AND ENABLED" not in out

    def test_a_KEEP_verdict_prints_NO_registry_diagnosis(self, tmp_path,
                                                         capsys):
        """KEEP is the commoner completed verdict, and the admit side was
        pinned only for REVERT."""
        rows = ([_t("treatment", "passed")] * 15
                + [_t("treatment", "failed")] * 5
                + [_t("control", "passed")] * 15
                + [_t("control", "failed")] * 5)
        rc, _art = self._run(tmp_path, rows)
        # §4DA round 15: the three GEPA instruments share one exit
        # contract — 0 still earns its place, 1 it does not,
        # 2 could not measure, 3 reported but not acted on.
        assert rc == 0, rc
        out = capsys.readouterr().out
        assert "KEEP" in out
        assert "IS NOT REGISTERED" not in out
        assert "THERE IS NO LIVE ARTIFACT" not in out

    def test_the_KILL_SWITCH_is_named_as_the_cause(self, tmp_path,
                                                   capsys, monkeypatch):
        """`GHOST_EXPERIMENTS=0` disables assignment entirely, so every
        turn is `unenrolled` with a perfectly good registry."""
        monkeypatch.setenv("GHOST_EXPERIMENTS", "0")
        out = self._confounded_with_registry(tmp_path, {
            "name": self._name(), "arms": ["control", "treatment"],
            "traffic": 1.0, "enabled": True}, capsys)
        assert "FRAMEWORK IS DISABLED BY ENV" in out
        assert "Nothing is misconfigured" not in out

    def test_a_MALFORMED_registry_is_named_as_the_cause(self, tmp_path,
                                                        capsys):
        """`load_registry` sets `degraded=True` when the file exists but
        cannot be parsed, precisely so a report can say so — and the
        built-in defaults then silently stand in for the operator's."""
        import importlib.util
        import sys as _sys
        home = tmp_path / "home"
        (home / "system" / "trajectories").mkdir(parents=True)
        (home / "system" / "optim").mkdir(parents=True)
        (home / "system" / "optim" / "planning.decompose.json").write_text(
            json.dumps({"optimized_instruction": "T"}))
        (home / "system" / "experiments.json").write_text("{not json")
        rows = [_t("unenrolled", "failed")] * 40
        spec = importlib.util.spec_from_file_location(
            "glc_bad", "scripts/gepa_live_check.py")
        mod = importlib.util.module_from_spec(spec)
        _sys.modules["glc_bad"] = mod
        spec.loader.exec_module(mod)

        class _Coll:
            def __init__(self, **kw): pass
            def iter_trajectories(self): return iter(rows)
        mod.TrajectoryCollector = _Coll
        old_argv = _sys.argv
        try:
            _sys.argv = ["glc_bad", "--home", str(home)]
            mod.main()
        finally:
            _sys.argv = old_argv
        assert "COULD NOT BE PARSED" in capsys.readouterr().out

    def test_the_diagnosis_follows_the_SIGNATURE_flag(self, tmp_path,
                                                      capsys):
        """⚠ The third use of `args.signature`, added in round 3 and not
        covered by round 3's own pin: hardcoding it here prints a
        diagnosis about the wrong experiment."""
        import importlib.util
        import sys as _sys
        home = tmp_path / "home"
        (home / "system" / "trajectories").mkdir(parents=True)
        (home / "system" / "optim").mkdir(parents=True)
        (home / "system" / "optim" /
         "tool_selection.pick.json").write_text(
            json.dumps({"optimized_instruction": "T"}))
        rows = [_t("unenrolled", "failed", sig="tool_selection.pick")] * 40
        spec = importlib.util.spec_from_file_location(
            "glc_sig", "scripts/gepa_live_check.py")
        mod = importlib.util.module_from_spec(spec)
        _sys.modules["glc_sig"] = mod
        spec.loader.exec_module(mod)

        class _Coll:
            def __init__(self, **kw): pass
            def iter_trajectories(self): return iter(rows)
        mod.TrajectoryCollector = _Coll
        old_argv = _sys.argv
        try:
            _sys.argv = ["glc_sig", "--home", str(home), "--signature",
                         "tool_selection.pick"]
            mod.main()
        finally:
            _sys.argv = old_argv
        out = capsys.readouterr().out
        assert "gepa_tool_selection_pick" in out
        assert "gepa_planning_decompose" not in out, (
            "the diagnosis named a different experiment than the one "
            "asked about")

    def test_a_MIS_ARMED_registry_is_named_as_the_cause(self, tmp_path,
                                                        capsys):
        """⚠ "none randomized" is the right words for the wrong reason
        when the operator HAS registered the experiment but with arm names
        the loader cannot act on. The registry legally accepts
        `["baseline","tuned"]`; neither is "control", so both arms get the
        artifact and every turn is filed `unenrolled` — and the report
        told them to register what they had just registered."""
        import importlib.util
        import sys as _sys
        from ghost_agent.optim import loader as _loader
        home = tmp_path / "home"
        (home / "system" / "trajectories").mkdir(parents=True)
        (home / "system" / "optim").mkdir(parents=True)
        (home / "system" / "optim" / "planning.decompose.json").write_text(
            json.dumps({"optimized_instruction": "T"}))
        (home / "system" / "experiments.json").write_text(json.dumps({
            "salt": "t", "experiments": [
                {"name": _loader.experiment_name("planning.decompose"),
                 "arms": ["baseline", "tuned"],
                 "traffic": 1.0, "enabled": True}]}))
        rows = [_t("unenrolled", "failed")] * 40
        spec = importlib.util.spec_from_file_location(
            "glc3", "scripts/gepa_live_check.py")
        mod = importlib.util.module_from_spec(spec)
        _sys.modules["glc3"] = mod
        spec.loader.exec_module(mod)

        class _Coll:
            def __init__(self, **kw): pass
            def iter_trajectories(self): return iter(rows)
        mod.TrajectoryCollector = _Coll
        old_argv = _sys.argv
        try:
            _sys.argv = ["glc3", "--home", str(home)]
            mod.main()
        finally:
            _sys.argv = old_argv
        out = capsys.readouterr().out
        assert "ARMS THIS LOADER CANNOT ACT ON" in out, (
            "a mis-armed registry was reported as 'not registered'")
        assert "baseline" in out and "tuned" in out

    def test_CONFOUNDED_never_retires_even_with_revert(self, tmp_path):
        """⚠ The flag must not override the refusal to conclude. 500
        un-randomized turns, all failures, is still not evidence."""
        rows = [_t("unenrolled", "failed")] * 500
        rc, art = self._run(tmp_path, rows, ("--revert",))
        # §4DA round 15: the three GEPA instruments share one exit
        # contract — 0 still earns its place, 1 it does not,
        # 2 could not measure, 3 reported but not acted on.
        assert rc == 2 and art.exists(), (
            "an artifact was retired on confounded data")

    def test_a_WINNING_artifact_is_never_retired(self, tmp_path):
        rows = ([_t("treatment", "passed")] * 18 + [_t("treatment", "failed")] * 2
                + [_t("control", "failed")] * 15 + [_t("control", "passed")] * 5)
        rc, art = self._run(tmp_path, rows, ("--revert",))
        # §4DA round 15: the three GEPA instruments share one exit
        # contract — 0 still earns its place, 1 it does not,
        # 2 could not measure, 3 reported but not acted on.
        assert rc == 0 and art.exists(), rc


class TestTheCLI_PlumbingIsReal:
    def test_min_per_arm_REACHES_the_verdict(self, tmp_path, monkeypatch):
        """⚠ The flag could be dropped en route and no test noticed: every
        script test used 20 turns per arm, the region where the default
        and any override agree."""
        import importlib.util
        import sys as _sys
        home = tmp_path / "home"
        (home / "system" / "trajectories").mkdir(parents=True)
        (home / "system" / "optim").mkdir(parents=True)
        (home / "system" / "optim" / "planning.decompose.json").write_text(
            json.dumps({"optimized_instruction": "T"}))
        rows = ([_t("treatment", "failed")] * 5
                + [_t("control", "passed")] * 5)
        spec = importlib.util.spec_from_file_location(
            "glc2", "scripts/gepa_live_check.py")
        mod = importlib.util.module_from_spec(spec)
        _sys.modules["glc2"] = mod
        spec.loader.exec_module(mod)

        class _Coll:
            def __init__(self, **kw): pass
            def iter_trajectories(self): return iter(rows)
        mod.TrajectoryCollector = _Coll

        seen = {}
        real = mod.live_check.verdict
        mod.live_check.verdict = lambda c, **kw: (
            seen.update(kw) or real(c, **kw))
        old = _sys.argv
        try:
            _sys.argv = ["glc2", "--home", str(home), "--min-per-arm", "3"]
            mod.main()
        finally:
            _sys.argv = old
            mod.live_check.verdict = real
        assert seen.get("min_per_arm") == 3, (
            "--min-per-arm never reached verdict(); the flag is inert")


class TestTheSeamFromLoaderToVerdict:
    """⚠ NO TEST JOINED THE TWO HALVES. The loader can attribute perfectly
    and `live_check` can judge perfectly while the stamp one writes is not
    the shape the other reads — `built-but-unwired-loops` between two
    correct components.

    Scope, stated rather than implied: this drives loader -> agent ->
    collector -> `collect()` on real objects and proves the STAMP round
    trips and buckets by arm. It does NOT drive a REVERT end to end,
    because `_record_turn_trajectory` can only write `failed`/`unknown` —
    a turn's PASSED verdict is backfilled later by the verifier path. The
    verdict half is covered by the unit tests above; claiming more here
    would be the kind of end-to-end theatre this session keeps finding.
    """

    def test_the_stamp_round_trips_and_buckets_by_arm(self, tmp_path,
                                                       monkeypatch):
        from unittest.mock import MagicMock
        from ghost_agent.core.agent import GhostAgent
        from ghost_agent.distill.collector import TrajectoryCollector
        from ghost_agent.optim import loader

        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        d = tmp_path / "system" / "optim"
        d.mkdir(parents=True)
        (d / "planning.decompose.json").write_text(json.dumps(
            {"optimized_instruction": "TUNED", "gate_arm": "g"}))
        loader.clear_cache()
        loader._SERVED_RING.clear()
        exp = loader.experiment_name("planning.decompose")

        coll = TrajectoryCollector(root=tmp_path / "traj", session_id="e2e")
        ctx = MagicMock()
        ctx.trajectory_collector = coll
        agent = GhostAgent.__new__(GhostAgent)
        agent.context = ctx

        class _C:
            def __init__(self, rid, arm):
                self._experiment_arms = (rid, {exp: arm})

        for i in range(7):
            rid = f"t{i}"
            loader.tuned_instruction("planning.decompose", "BASE",
                                     context=_C(rid, "treatment"),
                                     req_id=rid)
            agent._record_turn_trajectory(
                messages=[{"role": "user", "content": "q"},
                          {"role": "assistant", "content": "a"}],
                final_content="a", req_id=rid, model="m",
                execution_failed=True)
        for i in range(4):
            rid = f"c{i}"
            loader.tuned_instruction("planning.decompose", "BASE",
                                     context=_C(rid, "control"), req_id=rid)
            agent._record_turn_trajectory(
                messages=[{"role": "user", "content": "q"},
                          {"role": "assistant", "content": "a"}],
                final_content="a", req_id=rid, model="m",
                execution_failed=True)

        trajs = list(coll.iter_trajectories())
        c = live_check.collect(trajs, "planning.decompose")
        assert (c.treatment.n, c.control.n) == (7, 4), (
            f"the stamp did not survive the round trip: treatment="
            f"{c.treatment}, control={c.control}")
        assert c.treatment.failed == 7 and c.control.failed == 4
        assert list(c.shas) and "" not in c.shas, (
            "treatment turns reached collect() without an artifact sha")


class TestEveryDiagnosisCarriesItsActionableHalf:
    """⚠ A DESCRIPTION WITHOUT AN INSTRUCTION IS HALF A DIAGNOSIS, and a
    text sweep found twelve of these deletable with the suite green: the
    fix in each message could be removed while every framing word survived.
    The text is correct today — this is drift exposure, pinned in one
    batch rather than one assertion at a time.
    """

    def _diag(self, tmp_path, spec, *, artifact=True, randomized=0):
        from ghost_agent.optim import live_check as lc
        from ghost_agent.optim.loader import experiment_name
        home = tmp_path
        (home / "system" / "optim").mkdir(parents=True, exist_ok=True)
        if artifact:
            (home / "system" / "optim" /
             "planning.decompose.json").write_text(
                json.dumps({"optimized_instruction": "T"}))
        if spec is not None:
            (home / "system" / "experiments.json").write_text(json.dumps(
                {"salt": "t", "experiments": [spec]}))
        return lc.registry_diagnosis("planning.decompose", home,
                                     randomized=randomized), \
            experiment_name("planning.decompose")

    def test_each_state_names_what_to_DO(self, tmp_path):
        cases = [
            ("no artifact", None, {"artifact": False},
             "scripts/run_gepa.py"),
            ("unregistered", None, {},
             '["control", "treatment"]'),
            ("disabled", {"arms": ["control", "treatment"],
                          "traffic": 1.0, "enabled": False}, {},
             'Set "enabled": true'),
            ("traffic 0", {"arms": ["control", "treatment"],
                           "traffic": 0.0, "enabled": True}, {},
             "Raise traffic above 0"),
            ("rejected", {"arms": ["control", "control"],
                          "traffic": 1.0, "enabled": True}, {},
             "Check the agent log"),
            ("bench scope", {"arms": ["control", "treatment"],
                             "traffic": 1.0, "enabled": True,
                             "scope": "bench"}, {},
             'Set "scope"'),
            ("no known arms", {"arms": ["baseline", "tuned"],
                               "traffic": 1.0, "enabled": True}, {},
             'Re-register'),
            ("one known arm", {"arms": ["control", "baseline"],
                               "traffic": 1.0, "enabled": True}, {},
             'needs BOTH "control" and "treatment"'),
        ]
        for i, (label, spec, kw, must) in enumerate(cases):
            if spec is not None:
                spec = dict(spec)
                spec["name"] = self._name_of()
            out, _ = self._diag(tmp_path / f"c{i}", spec, **kw)
            assert must in out, (
                f"[{label}] lost its actionable half; got: {out[:160]}")

    def _name_of(self):
        from ghost_agent.optim.loader import experiment_name
        return experiment_name("planning.decompose")
