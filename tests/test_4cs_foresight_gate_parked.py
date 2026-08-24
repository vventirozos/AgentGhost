"""§4CS item G — the Imagine gate qualifies nothing, and now SAYS WHY.

Live state 2026-08-23: 754 ledger rows, 63 buckets, ZERO enabled, 57 thin
and 6 without a denominator. Every bucket's `why` reads "needs N more",
which is indistinguishable from a gate that is merely waiting for data —
the failure mode this project has a name for.

The measurement says otherwise, and it is sharper than "years to qualify":

  * the DENOMINATOR is not far away — the best-placed bucket is ~5 weeks
    from `min_fail_n` at its own observed arrival rate;
  * the PRECISION never arrives. Pooled over every steerable row in the
    ledger the index scores 1/14 = 0.071, Wilson 95% CI [0.013, 0.315] —
    entirely below the 0.60 bar. Predicted-fail rows fail 7.1% while
    predicted-ok rows fail 10.6%, so the spread is NEGATIVE;
  * `_evaluate_bucket` checks precision BEFORE the interval test, so a
    bucket whose true precision is under the bar cannot enable at ANY n.
    Time-to-qualify is not long, it is UNDEFINED.

`_pooled_discrimination` is DERIVED, so the verdict retracts itself the
moment predicted-fail rows start failing more than predicted-ok ones.
"""

import pytest

from ghost_agent.core import imagination as IM


def _agg(fail_n, fail_hits, ok_n, ok_hits):
    return {"b": {"n": fail_n + ok_n, "claimed": fail_n + ok_n, "failed": 0,
                  "matched": 0, "fail_n": fail_n, "fail_hits": fail_hits,
                  "fail_outcomes": [1.0] * fail_hits
                                   + [0.0] * (fail_n - fail_hits),
                  "ok_outcomes": [1.0] * ok_hits + [0.0] * (ok_n - ok_hits),
                  "brier_sum": 0.0}}


PARAMS = {"min_bucket_n": 30, "min_fail_n": 10,
          "min_fail_precision": 0.6, "min_spread": 0.1}


class TestPooledDiscrimination:
    def test_a_NEGATIVE_spread_is_called_anti_predictive_not_thin(self):
        """The live shape. "Waiting for data" and "the sign is wrong" lead
        opposite places — one waits, the other stops."""
        d = IM._pooled_discrimination(_agg(14, 1, 564, 60), PARAMS)
        assert d["precision"] == pytest.approx(1 / 14)
        assert d["ok_fail_rate"] == pytest.approx(60 / 564)
        assert d["spread"] < 0
        assert "ANTI-PREDICTIVE" in d["verdict"]
        assert "THE SIGN IS WRONG" in d["verdict"]
        assert "only a better index does" in d["verdict"]

    def test_the_verdict_RETRACTS_when_the_index_starts_working(self):
        """Derived, not asserted: nothing about the park is hardcoded."""
        d = IM._pooled_discrimination(_agg(20, 16, 100, 5), PARAMS)
        assert d["spread"] > 0
        assert "ANTI-PREDICTIVE" not in d["verdict"]
        assert "clear both bars" in d["verdict"]

    def test_right_direction_but_under_the_bar_is_its_own_verdict(self):
        d = IM._pooled_discrimination(_agg(20, 8, 100, 5), PARAMS)
        assert 0 < d["precision"] < PARAMS["min_fail_precision"]
        assert d["spread"] > 0
        assert "right direction" in d["verdict"]
        assert "ANTI-PREDICTIVE" not in d["verdict"]

    def test_no_comparison_group_claims_NOTHING(self):
        """With no steerable rows at all there is no verdict to give, and
        inventing one would be the "verdict without power" shape."""
        d = IM._pooled_discrimination(_agg(0, 0, 100, 5), PARAMS)
        assert d["spread"] is None
        assert "has not been asked enough" in d["verdict"]
        assert "ANTI-PREDICTIVE" not in d["verdict"]

    def test_the_gate_document_carries_it(self, tmp_path, monkeypatch):
        rows = ([{"tool": "web_search", "tclass": "other", "ok": True,
                  "p_fail": 0.1, "basis": "exact", "support": 5, "fails": 0}] * 20)
        doc = IM.build_gate(rows=rows, write=False)
        assert "discrimination" in doc
        assert doc["discrimination"]["verdict"]


class TestTheIncumbentPassesItsOwnChecker:
    """Instruments in this project lie plausibly, and this one decides
    whether any steering site may exist."""

    def test_the_enabled_path_is_REACHABLE(self):
        """A checker that cannot say yes is not a gate — "0 enabled" would
        then be a property of the rule, not of the data."""
        hits = 9
        b = {"n": 30, "claimed": 30, "failed": 0, "matched": 0,
             "fail_n": 10, "fail_hits": hits,
             "fail_outcomes": [1.0] * hits + [0.0] * (10 - hits),
             "ok_outcomes": [0.0] * 19 + [1.0], "brier_sum": 0.0}
        e = IM._evaluate_bucket("synthetic|x", b, PARAMS)
        assert e["enabled"] is True, e["why"]
        assert "DISCRIMINATES" in e["why"]

    def test_the_precision_bar_is_checked_BEFORE_the_interval_test(self):
        """This is what makes a low-precision bucket unqualifiable at ANY
        n, and it is the whole park argument. Asserted on the ORDER of the
        verdicts, not restated."""
        hits = 3
        b = {"n": 200, "claimed": 200, "failed": 0, "matched": 0,
             "fail_n": 100, "fail_hits": hits * 10,
             "fail_outcomes": [1.0] * 30 + [0.0] * 70,
             "ok_outcomes": [0.0] * 100, "brier_sum": 0.0}
        e = IM._evaluate_bucket("synthetic|x", b, PARAMS)
        # precision 0.30: a huge denominator and a huge spread, and it
        # still fails — on precision, and precision is what it names.
        assert e["enabled"] is False
        assert e["why"].startswith("precision too low")
        assert e["spread"] > PARAMS["min_spread"], \
            "the spread test would have PASSED — precision is what binds"

    def test_a_thin_bucket_names_the_binding_constraint_first(self):
        b = {"n": 5, "claimed": 5, "failed": 0, "matched": 0, "fail_n": 0,
             "fail_hits": 0, "fail_outcomes": [], "ok_outcomes": [0.0] * 5,
             "brier_sum": 0.0}
        e = IM._evaluate_bucket("x|y", b, PARAMS)
        assert e["why"].startswith("thin bucket")
        assert e["needs"] == PARAMS["min_bucket_n"] - 5


class TestLiveGateIsSurfaced:
    def test_the_yield_row_carries_the_pooled_verdict(self, tmp_path):
        import json
        from ghost_agent.core.liveness import _yield_foresight_gate
        d = tmp_path / "system" / "foresight"
        d.mkdir(parents=True)
        (d / "gate.json").write_text(json.dumps({
            "buckets": {f"b{i}": {} for i in range(63)},
            "enabled_count": 0, "ledger_rows": 754,
            "discrimination": {"verdict": "⚠ ANTI-PREDICTIVE, pooled ..."}}))
        res = _yield_foresight_gate(tmp_path)
        assert "POOLED:" in res.note
        assert "ANTI-PREDICTIVE" in res.note

    def test_a_gate_without_a_verdict_still_renders(self, tmp_path):
        import json
        from ghost_agent.core.liveness import _yield_foresight_gate
        d = tmp_path / "system" / "foresight"
        d.mkdir(parents=True)
        (d / "gate.json").write_text(json.dumps(
            {"buckets": {"a": {}}, "enabled_count": 0, "ledger_rows": 3}))
        res = _yield_foresight_gate(tmp_path)
        assert "POOLED:" not in res.note
        assert res.invoked == 0
