"""Queue #8 — the measurement instruments must not lie by omission.

Two defects, both of the queue's stated shape ("a wrong verdict looks exactly
like a verdict"):

1. **"no difference detected yet" was indistinguishable from "this design
   cannot detect a difference."** Measured on the live board 2026-08-21:
   every live arm, on BOTH quality metrics, had a CS half-width 2-6x larger
   than the biggest improvement the metric can physically show — e.g.
   `failure_rate` control 0.203 with a half-width of 0.367, so the rate would
   have to fall below zero to be called. Five arms x two metrics of "no
   difference detected yet" reads as evidence the features don't help; it was
   the instrument having no power. Absence of evidence is evidence of absence
   only when the design could have found something.

2. **The stamp-coverage alarm could not fire.** Its own docstring says it
   exists so "a broken stamp is [not] indistinguishable from a young
   experiment", but the denominator is every user turn ever recorded — ~1,320
   of which predate the framework — so a TOTAL outage moves the headline from
   16.8% to ~16.3%, under a "under half" warning that has been permanently on.
   Per-day coverage has been 100% since 2026-08-04 the whole time.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from types import SimpleNamespace

import pytest

from ghost_agent.core import experiments as X


# ──────────────────────────────────────────────────────────────────────
# The metric partition — the drift guard for everything below
# ──────────────────────────────────────────────────────────────────────

class TestMetricPartition:
    def test_rate_and_unbounded_PARTITION_the_metric_registry(self):
        """A new metric must be classified. Without this, adding one silently
        opts it out of the power check (`max_possible_improvement` returns
        None) and it goes back to reading "no difference detected yet"
        forever — the defect this file exists for, re-introduced by an
        unrelated addition."""
        names = {m for m, _ in X._METRICS}

        assert X._RATE_METRICS | X._UNBOUNDED_METRICS == names
        assert not (X._RATE_METRICS & X._UNBOUNDED_METRICS)

    def test_every_rate_metric_really_is_bounded_in_unit_interval(self):
        """The classification is a claim about the DATA, so check it against
        the producer rather than trusting the name."""
        traj = SimpleNamespace(
            outcome="failed", failure_reason="", n_steps=3, duration_s=1.0,
            extra={"outcome_source": "human_feedback:web"}, tool_calls=[],
        )
        vals = X._metric_values(traj)
        for metric in X._RATE_METRICS:
            if metric in vals:
                assert 0.0 <= vals[metric] <= 1.0


# ──────────────────────────────────────────────────────────────────────
# n_for_detectable
# ──────────────────────────────────────────────────────────────────────

class TestNForDetectable:
    ALPHA = 0.05 / 4 / 2

    def test_returns_an_n_whose_interval_actually_clears_the_effect(self):
        """Identity, not a literal: the n it returns must be one at which the
        REAL radius function gives a half-width below the effect — and n-1
        must not. Pins the answer to the estimator the operator is reading,
        so the two cannot drift apart."""
        p = 0.203
        n = X.n_for_detectable(p, p, alpha=self.ALPHA)
        assert n is not None

        def half_width(k):
            ones = int(round(p * k))
            series = [1.0] * ones + [0.0] * (k - ones)
            return 2.0 * X.asymp_cs_radius(series, alpha=self.ALPHA)

        assert half_width(n) < p
        assert half_width(n - 1) >= p

    def test_is_deterministic(self):
        """An audit number that moves on its own is not one."""
        a = X.n_for_detectable(0.2, 0.2, alpha=self.ALPHA)
        b = X.n_for_detectable(0.2, 0.2, alpha=self.ALPHA)

        assert a == b and a is not None

    def test_a_smaller_effect_needs_more_data(self):
        big = X.n_for_detectable(0.20, 0.20, alpha=self.ALPHA)
        small = X.n_for_detectable(0.05, 0.20, alpha=self.ALPHA)

        assert big is not None and small is not None
        assert small > big

    def test_the_search_probes_the_CAP_it_reports(self):
        """The caller's message names `_POWER_SEARCH_MAX_N`. Plain doubling
        stopped at 131,072 and then claimed "unreachable within 200,000" — a
        bound it never evaluated. Any effect reachable at exactly the cap
        must therefore be found."""
        probed = []
        real = X.asymp_cs_radius

        def spy(vals, **kw):
            probed.append(len(vals))
            return real(vals, **kw)

        X.asymp_cs_radius = spy
        try:
            X.n_for_detectable(1e-9, 0.2, alpha=self.ALPHA)
        finally:
            X.asymp_cs_radius = real

        assert max(probed) == X._POWER_SEARCH_MAX_N

    def test_an_unreachable_effect_returns_None_not_a_huge_number(self):
        assert X.n_for_detectable(1e-9, 0.2, alpha=self.ALPHA) is None

    @pytest.mark.parametrize("effect,rate,alpha", [
        (0.0, 0.2, 0.00625), (-0.1, 0.2, 0.00625),
        (0.2, 0.2, 0.0), (0.2, 0.2, 1.0),
    ])
    def test_bad_input_returns_None(self, effect, rate, alpha):
        assert X.n_for_detectable(effect, rate, alpha=alpha) is None


# ──────────────────────────────────────────────────────────────────────
# The verdict: "no effect" vs "no power"
# ──────────────────────────────────────────────────────────────────────

def _cmp(metric="failure_rate", *, lower_is_better=True, control=0.203,
         treatment=0.203, lo=-0.368, hi=0.366, n=71, arm_alpha=0.05 / 4 / 2):
    return X.MetricComparison(
        metric=metric, lower_is_better=lower_is_better,
        control_mean=control, treatment_mean=treatment,
        control_n=n, treatment_n=n,
        diff=treatment - control, diff_lo=lo, diff_hi=hi,
        arm_alpha=arm_alpha,
    )


class TestNoPowerVerdict:
    def test_the_live_shape_says_NO_POWER_not_no_difference(self):
        """The exact numbers off the live board on 2026-08-21."""
        v = _cmp().verdict

        assert "NO POWER" in v
        assert "no difference detected yet" not in v
        assert "0.203" in v            # names the ceiling it cannot clear
        assert "/arm" in v             # and how much data would

    def test_a_powered_comparison_still_reads_no_difference(self):
        """The guard must not swallow the ordinary case: when the interval
        IS narrower than the achievable improvement, "no difference detected
        yet" is the honest verdict and must survive."""
        v = _cmp(lo=-0.05, hi=0.05, n=400).verdict

        assert v == "no difference detected yet"

    def test_unbounded_metrics_never_claim_no_power(self):
        """`n_steps`/`duration_s` have no ceiling to compare against, so the
        power statement would be unfounded."""
        for metric in sorted(X._UNBOUNDED_METRICS):
            v = _cmp(metric=metric, control=3.5, treatment=3.3,
                     lo=-4.4, hi=4.0).verdict

            assert v == "no difference detected yet"
            assert "NO POWER" not in v

    def test_a_real_verdict_is_not_suppressed(self):
        """An interval that excludes zero must still call it, no matter how
        wide — the power note applies only to the straddling case."""
        v = _cmp(control=0.20, treatment=0.60, lo=+0.10, hi=+0.70,
                 n=100).verdict

        assert "TREATMENT WORSE" in v
        assert "NO POWER" not in v

    def test_insufficient_n_still_wins_over_the_power_note(self):
        """Below `_MIN_VERDICT_N` the CS is not trustworthy at all, so that
        caveat must come first — two caveats in one line would bury it."""
        v = _cmp(n=X._MIN_VERDICT_N - 1).verdict

        assert v == f"insufficient data (n<{X._MIN_VERDICT_N}/arm)"

    def test_a_control_already_at_the_floor_is_not_reported_as_a_failure(
            self):
        """`max_possible_improvement == 0` is the GOOD case — a 0% failure
        rate — and the generic wording rendered it as an instrument problem
        ("the largest improvement this metric can show (0.000)... unreachable
        within 200,000/arm"), which is false at every n."""
        v = _cmp(control=0.0, treatment=0.0, lo=-0.10, hi=0.10, n=80).verdict

        assert "no improvement is POSSIBLE" in v
        assert "already at 0" in v
        assert "unreachable" not in v
        assert "NO POWER" not in v

    def test_missing_alpha_does_not_fabricate_an_impossibility_claim(self):
        """THREE cases, not two: "no alpha supplied" is not "no n can reach
        it". Collapsing them made a missing field print a false
        impossibility — the shape this whole pass is about."""
        v = _cmp(arm_alpha=0.0).verdict

        assert "NO POWER" in v
        assert "unreachable" not in v
        assert "/arm (have" not in v

    def test_higher_is_better_uses_the_distance_to_one(self):
        """The ceiling is 1 - control for a metric that wants to go UP; using
        `control` there would understate the room by a lot at low rates."""
        c = _cmp(metric="human_label_rate", lower_is_better=False,
                 control=0.2)

        assert c.max_possible_improvement == pytest.approx(0.8)

    def test_lower_is_better_uses_the_distance_to_zero(self):
        assert _cmp(control=0.2).max_possible_improvement == pytest.approx(0.2)

    def test_half_width_is_half_the_rendered_interval(self):
        assert _cmp(lo=-0.368, hi=0.366).half_width == pytest.approx(0.367)


# ──────────────────────────────────────────────────────────────────────
# Coverage: the window that can actually move
# ──────────────────────────────────────────────────────────────────────

def _traj(stamped=True, kind="user_request", outcome="passed"):
    extra = {"experiments": {"exp_a": "control"}} if stamped else {}
    return SimpleNamespace(
        task_kind=kind, outcome=outcome, failure_reason="",
        n_steps=1, duration_s=1.0, extra=extra, tool_calls=[],
    )


def _walk(trajs):
    _all, _trig, cov = X.summarize_streaming(trajs)
    return cov


class TestRecentCoverageWindow:
    def test_a_dead_stamp_shows_in_the_window_while_lifetime_barely_moves(
            self):
        """The whole point. 1,000 historical unstamped turns, 300 stamped,
        then the stamp dies for the last 60: lifetime cannot fall far, the
        window goes to zero."""
        trajs = ([_traj(stamped=False)] * 1000 + [_traj(stamped=True)] * 300
                 + [_traj(stamped=False)] * 60)
        cov = _walk(trajs)

        lifetime = cov["stamped"] / cov["user_turns"]
        recent = cov["recent_stamped"] / cov["recent_admitted"]

        assert lifetime > 0.20               # still reads "fine-ish"
        assert recent == 0.0                 # the window is unambiguous

    def test_a_healthy_stamp_reads_100_percent_in_the_window(self):
        cov = _walk([_traj(stamped=False)] * 500 + [_traj(stamped=True)] * 80)

        assert cov["recent_admitted"] == X._RECENT_COVERAGE_WINDOW
        assert cov["recent_stamped"] == X._RECENT_COVERAGE_WINDOW

    def test_the_window_is_bounded(self):
        cov = _walk([_traj(stamped=True)] * 5000)

        assert cov["recent_admitted"] == X._RECENT_COVERAGE_WINDOW

    def test_short_corpora_report_what_they_have(self):
        cov = _walk([_traj(stamped=True)] * 3)

        assert cov["recent_admitted"] == 3
        assert cov["recent_stamped"] == 3

    def test_non_admitted_records_are_excluded_from_the_window(self):
        """A reflection/bench record is not part of this population, so it
        must not dilute (or pad) the live stamp rate."""
        cov = _walk([_traj(stamped=True)] * 10
                    + [_traj(stamped=True, kind="reflection")] * 40)

        assert cov["recent_admitted"] == 10
        assert cov["recent_stamped"] == 10

    def test_a_report_scoped_out_of_every_name_is_NOT_a_stamp_regression(
            self):
        """The window answers "did enrollment stamp this record", not "does
        this report's scope include the stamp". Feeding it the scoped answer
        made a fully-stamped corpus render a false "the stamp is regressing
        NOW" alarm — reproduced with 20 records carrying only a denied name.
        The lifetime counter stays scoped (it feeds the arm stats); the two
        deliberately differ here, which is why both are pinned."""
        rows = [SimpleNamespace(
            task_kind="user_request", outcome="passed", failure_reason="",
            n_steps=1, duration_s=1.0, tool_calls=[],
            extra={"experiments": {"tts_bon": "control"}})] * 20
        _a, _t, cov = X.summarize_streaming(rows, deny_names={"tts_bon"})

        assert cov["recent_admitted"] == 20
        assert cov["recent_stamped"] == 20     # the stamp did its job
        assert cov["stamped"] == 0             # ...and this report shows none

    def test_window_and_lifetime_agree_on_what_covered_MEANS(self):
        """A malformed stamp ({"name": null}) enters no arm, so the lifetime
        counter deliberately does not count it as covered. The window must
        use the SAME definition or the two lines contradict each other."""
        bad = SimpleNamespace(
            task_kind="user_request", outcome="passed", failure_reason="",
            n_steps=1, duration_s=1.0, tool_calls=[],
            extra={"experiments": {"exp_a": None}},
        )
        cov = _walk([bad] * 10)

        assert cov["stamped"] == 0
        assert cov["recent_admitted"] == 10
        assert cov["recent_stamped"] == 0


class TestCoverageRendering:
    def _render(self, cov):
        stats = X.ArmStats(arm="control")
        stats.n = 5
        summary = {"exp_a": {"control": stats}}
        return X.render_report(summary, coverage=cov)

    def test_a_diluted_lifetime_line_says_it_cannot_fall(self):
        out = self._render({"user_turns": 1587, "admitted_turns": 1587,
                            "stamped": 267, "recent_admitted": 50,
                            "recent_stamped": 50})

        assert "LIFETIME" in out
        assert "cannot fall far" in out

    def test_an_UNDILUTED_lifetime_line_does_not_claim_it_cannot_fall(self):
        """The bench population reads 116/116. Asserting "cannot fall far"
        there is false — and it is the same measured-sounding-but-untrue
        shape this whole pass exists to remove."""
        out = self._render({"user_turns": 116, "admitted_turns": 116,
                            "stamped": 116, "recent_admitted": 50,
                            "recent_stamped": 50})

        assert "LIFETIME" in out
        assert "cannot fall far" not in out

    def test_a_regressing_window_raises_the_alarm(self):
        out = self._render({"user_turns": 1587, "admitted_turns": 1587,
                            "stamped": 267, "recent_admitted": 50,
                            "recent_stamped": 10})

        assert "regressing NOW" in out

    def test_a_healthy_window_raises_nothing(self):
        out = self._render({"user_turns": 1587, "admitted_turns": 1587,
                            "stamped": 267, "recent_admitted": 50,
                            "recent_stamped": 50})

        assert "regressing NOW" not in out
