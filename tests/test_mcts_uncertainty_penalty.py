"""Roadmap phase 1.5 — PRM uncertainty downweighting in MCTS.

When ``uncertainty_penalty > 0`` and the scorer exposes an
``uncertainty(state, action)`` callable, MCTS should prefer branches
the PRM is confident about. Default behaviour (penalty=0) must be
unchanged from the pre-uplift baseline.
"""

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from ghost_agent.core.mcts import ActionCandidate, MCTSReasoner


@dataclass
class _FakeAction:
    pass


class _FakeState:
    pass


class _StubScorer:
    """A scorer where every action gets the same raw score (0.7), but
    uncertainty varies per ``tool_name``. Without the penalty the two
    actions tie; with the penalty the low-uncertainty one wins.
    """

    has_model = True

    def __init__(self):
        self._unc = {"safe": 0.0, "risky": 1.0}

    def score(self, state, action):
        return 0.7

    def uncertainty(self, state, action):
        return self._unc.get(action.tool_name, 0.5)


@pytest.mark.asyncio
async def test_penalty_zero_keeps_legacy_behaviour():
    reasoner = MCTSReasoner(
        llm_client=MagicMock(),
        prm_scorer=_StubScorer(),
        uncertainty_penalty=0.0,
    )
    candidates = [
        ActionCandidate(description="A", tool_name="safe"),
        ActionCandidate(description="B", tool_name="risky"),
    ]
    scored = reasoner._score_with_prm(candidates, _FakeState())
    assert scored[0].score == pytest.approx(scored[1].score)


@pytest.mark.asyncio
async def test_penalty_downweights_uncertain_branch():
    reasoner = MCTSReasoner(
        llm_client=MagicMock(),
        prm_scorer=_StubScorer(),
        uncertainty_penalty=0.5,
    )
    candidates = [
        ActionCandidate(description="A", tool_name="safe"),
        ActionCandidate(description="B", tool_name="risky"),
    ]
    scored = reasoner._score_with_prm(candidates, _FakeState())
    safe = next(c for c in scored if c.tool_name == "safe")
    risky = next(c for c in scored if c.tool_name == "risky")
    # safe: 0.7 - 0.5*0 = 0.7
    # risky: 0.7 - 0.5*1.0 = 0.2
    assert safe.score == pytest.approx(0.7)
    assert risky.score == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_penalty_clamps_at_zero():
    """A penalty larger than the raw score must clamp to 0, not go
    negative — downstream sort assumes scores live in [0, 1]."""

    class _Scorer:
        has_model = True

        def score(self, state, action):
            return 0.1

        def uncertainty(self, state, action):
            return 1.0

    reasoner = MCTSReasoner(
        llm_client=MagicMock(),
        prm_scorer=_Scorer(),
        uncertainty_penalty=0.9,
    )
    candidates = [ActionCandidate(description="X", tool_name="foo")]
    scored = reasoner._score_with_prm(candidates, _FakeState())
    assert 0.0 <= scored[0].score <= 1.0


@pytest.mark.asyncio
async def test_penalty_survives_scorer_without_uncertainty_method():
    """A custom scorer that lacks ``uncertainty`` must not break the
    penalty path — the penalty is silently treated as zero."""

    class _BareScorer:
        has_model = True

        def score(self, state, action):
            return 0.6

    reasoner = MCTSReasoner(
        llm_client=MagicMock(),
        prm_scorer=_BareScorer(),
        uncertainty_penalty=0.5,
    )
    candidates = [ActionCandidate(description="X", tool_name="foo")]
    scored = reasoner._score_with_prm(candidates, _FakeState())
    assert scored[0].score == pytest.approx(0.6)
