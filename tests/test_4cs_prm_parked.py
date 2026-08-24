"""§4CS item C — the PRM is PARKED, and the telemetry says so honestly.

The state on 2026-08-23: no fitted checkpoint (only
`checkpoint.json.pre-1c-schema`), `_MCTS_TURNSTART_ENABLED = False` with
its re-enable criterion unmet, and `--frontier-selfplay` absent from the
launcher exec line. `prm_consumer_is_live` correctly refuses to retrain a
model nothing reads — that gate exists because of a real 41-wasted-retrains
incident, so it is NOT the bug.

The bug was that nothing recorded this as a DECISION. The wiring rows read
as a configuration that might change at any moment, which is
indistinguishable from a subsystem quietly broken — the exact ambiguity
§4CN removed for P7 and E4 by parking them with their arithmetic beside
them.

The load-bearing property is that the park is DERIVED and therefore
AUTO-RETRACTS. A hardcoded banner would go on claiming the decision after
an operator enables a consumer, which is the lying-instrument class this
project keeps paying for.
"""

import pytest

from ghost_agent.core.learning_health import PRM_PARKED_ON, _prm_parked_state


def _prm(score=None, uncertainty=None):
    return {"score_consumer_enabled": score,
            "uncertainty_consumer_enabled": uncertainty}


class TestParkIsDerived:
    def test_both_consumers_off_is_PARKED(self):
        out = _prm_parked_state(_prm(score=False, uncertainty=False))
        assert out.startswith("PARKED by operator decision")
        assert PRM_PARKED_ON in out

    @pytest.mark.parametrize("legs", [
        {"score": True, "uncertainty": False},
        {"score": False, "uncertainty": True},
        {"score": True, "uncertainty": True},
    ])
    def test_ANY_live_consumer_retracts_the_park(self, legs):
        """The pin that stops the banner outliving its decision."""
        out = _prm_parked_state(_prm(**legs))
        assert "LIFTED" in out
        assert "PARKED by operator decision" not in out

    @pytest.mark.parametrize("legs", [
        {"score": False, "uncertainty": None},
        {"score": None, "uncertainty": False},
        {"score": None, "uncertainty": None},
    ])
    def test_an_UNREADABLE_leg_claims_nothing(self, legs):
        """An unreadable leg is not an off one. Claiming a park from
        `None` would report a decision on a box whose state we cannot
        see — the favourable-outcome-on-failure shape."""
        out = _prm_parked_state(_prm(**legs))
        assert "undetermined" in out
        assert "PARKED by operator decision" not in out
        assert "LIFTED" not in out

    def test_a_live_leg_wins_over_an_unreadable_one(self):
        out = _prm_parked_state(_prm(score=True, uncertainty=None))
        assert "LIFTED" in out


class TestParkCarriesItsArithmetic:
    def test_the_park_names_BOTH_reopen_routes_and_their_costs(self):
        out = _prm_parked_state(_prm(score=False, uncertainty=False))
        assert "--frontier-selfplay" in out
        assert "_MCTS_TURNSTART_ENABLED" in out
        # The trap an operator would otherwise walk into.
        assert "1.0 uniformly" in out
        assert "uniform-over-rarity" in out
        # And the MCTS criterion is stated as UNMET, not merely named.
        assert "NOT met" in out


class TestTheParkedCodeIsMarked:
    def test_frontier_selection_says_it_is_parked(self):
        import ghost_agent.core.frontier_selection as fs
        doc = fs.__doc__ or ""
        assert "PARKED IN PRODUCTION" in doc
        assert "--frontier-selfplay" in doc

    def test_the_untrained_contract_the_park_rests_on_still_holds(self):
        """The park's arithmetic claims `uncertainty()` returns 1.0 for
        every cluster with no model. Verified against the function, not
        restated — if the contract changes, the park's reasoning is stale
        and this must fail."""
        from ghost_agent.core.frontier_selection import (
            combine_weights, compute_cluster_rarity, compute_cluster_uncertainty,
        )
        from ghost_agent.prm import PRMScorer
        keys = ["a", "b", "c"]
        # (i) no scorer at all
        assert set(compute_cluster_uncertainty(None, keys).values()) == {1.0}
        # (ii) THE PRODUCTION SHAPE: a scorer exists but nothing is fitted.
        # This is what `--frontier-selfplay` would actually meet on the
        # live box, and it is the leg the park's arithmetic rests on.
        untrained = PRMScorer()
        assert getattr(untrained, "has_model", False) is False
        unc = compute_cluster_uncertainty(untrained, keys)
        assert set(unc.values()) == {1.0}, unc
        # ...so the product IS rarity, exactly. Compared against a
        # recomputed rarity map, not against a restatement of the rule.
        counts = {"a": 0, "b": 3, "c": 40}
        rarity = compute_cluster_rarity(counts, keys)
        combined = combine_weights(unc, rarity)
        assert combined == pytest.approx(rarity), (combined, rarity)

    def test_mcts_turnstart_is_still_OFF(self):
        """The park explicitly does NOT flip this; its criterion is unmet."""
        from ghost_agent.core import agent as _ag
        assert _ag._MCTS_TURNSTART_ENABLED is False
