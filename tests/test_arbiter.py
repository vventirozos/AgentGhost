"""Unit tests for ghost_agent.core.arbiter — dual-solver arbitration.

ABLATED-LAYER TESTS: the dual-solver arbiter is DISABLED on the request path
(`_METACOG_ARBITER_ENABLED = False`, core/agent.py) since the 2026-06 cognitive
-layer redesign — see COGNITIVE_LAYER_REDESIGN.md. These tests pin the parked
subsystem's behavior so it stays correct if ever re-enabled under its criterion;
they do NOT assert live request-path behavior.
"""

import asyncio
import math

import pytest

from ghost_agent.core.arbiter import (
    ArbitrationDecision,
    Candidate,
    DualSolverArbiter,
    SemanticDivergence,
    _cosine,
    _jaccard,
)


# ──────────────────────────────────────────────────────────────────────
# Vector helpers
# ──────────────────────────────────────────────────────────────────────

class TestCosine:
    def test_identical(self):
        assert _cosine([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)

    def test_orthogonal(self):
        assert _cosine([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_opposite(self):
        assert _cosine([1, 1], [-1, -1]) == pytest.approx(-1.0)

    def test_empty(self):
        assert _cosine([], [1, 2]) == 0.0

    def test_zero_vec(self):
        assert _cosine([0, 0, 0], [1, 2, 3]) == 0.0

    def test_pads_unequal_lengths(self):
        # Should not raise; pads with zeros
        v = _cosine([1, 0], [1, 0, 0])
        assert math.isfinite(v)


class TestJaccard:
    def test_identical(self):
        assert _jaccard("hello world", "hello world") == 1.0

    def test_disjoint(self):
        assert _jaccard("a b c", "x y z") == 0.0

    def test_partial(self):
        # {"a","b"} vs {"a","c"} → 1/3
        assert _jaccard("a b", "a c") == pytest.approx(1 / 3)

    def test_empty_both(self):
        assert _jaccard("", "") == 1.0

    def test_empty_one(self):
        assert _jaccard("a", "") == 0.0


# ──────────────────────────────────────────────────────────────────────
# SemanticDivergence
# ──────────────────────────────────────────────────────────────────────

class TestSemanticDivergence:
    @pytest.mark.asyncio
    async def test_no_embedder_falls_back_to_jaccard(self):
        sd = SemanticDivergence(embedder=None)
        sim = await sd.similarity("hello world", "hello world")
        assert sim == 1.0

    @pytest.mark.asyncio
    async def test_with_sync_embedder(self):
        def embed(texts):
            return [[1.0, 0.0], [1.0, 0.0]]  # identical
        sd = SemanticDivergence(embedder=embed)
        sim = await sd.similarity("a", "b")
        assert sim == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_with_async_embedder(self):
        async def embed(texts):
            return [[1.0, 0.0], [0.0, 1.0]]  # orthogonal
        sd = SemanticDivergence(embedder=embed)
        sim = await sd.similarity("a", "b")
        assert sim == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.asyncio
    async def test_failing_embedder_falls_back(self):
        async def embed(texts):
            raise RuntimeError("upstream down")
        sd = SemanticDivergence(embedder=embed)
        sim = await sd.similarity("hello world", "hello world")
        # Falls back to Jaccard, which returns 1.0 for identical
        assert sim == 1.0

    def test_threshold_decision(self):
        sd = SemanticDivergence(threshold=0.85)
        assert sd.diverged(0.5)
        assert not sd.diverged(0.9)


# ──────────────────────────────────────────────────────────────────────
# DualSolverArbiter — basic flow
# ──────────────────────────────────────────────────────────────────────

def make_runner(outputs_by_temp):
    """Build a runner stub that returns scripted outputs per temperature."""
    async def runner(payload):
        t = payload["temperature"]
        out = outputs_by_temp.get(t, "")
        return out
    return runner


def identity_embedder(text_to_vec):
    """Build a deterministic embedder from a {text: vec} map."""
    def embed(texts):
        return [text_to_vec.get(t, [0.0, 0.0]) for t in texts]
    return embed


@pytest.mark.asyncio
async def test_converged_executes_low_temp():
    runner = make_runner({0.2: "the answer is 42", 0.7: "the answer is 42"})
    embedder = identity_embedder({"the answer is 42": [1.0, 0.0]})
    arbiter = DualSolverArbiter(runner=runner, embedder=embedder)
    decision = await arbiter.arbitrate("what is the answer?")
    assert decision.action == "execute"
    assert decision.chosen.temperature == 0.2
    assert decision.similarity == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_diverged_no_validator_routes_to_validate():
    runner = make_runner({
        0.2: "delete from users where id = 1",
        0.7: "drop table users",
    })

    def embed(texts):
        # Make them look very different
        return [[1.0, 0.0] if "delete" in t else [0.0, 1.0] for t in texts]

    arbiter = DualSolverArbiter(runner=runner, embedder=embed,
                                divergence_threshold=0.85)
    decision = await arbiter.arbitrate("clean up the users table")
    assert decision.action == "validate"
    assert decision.similarity < 0.85


@pytest.mark.asyncio
async def test_diverged_with_validator_picks_passing():
    runner = make_runner({
        0.2: "DELETE FROM users WHERE id = 1",
        0.7: "DROP TABLE users",
    })

    def embed(texts):
        return [[1.0, 0.0] if "DELETE" in t else [0.0, 1.0] for t in texts]

    from ghost_agent.tools.validators import validate_sql

    arbiter = DualSolverArbiter(runner=runner, embedder=embed,
                                divergence_threshold=0.85)
    decision = await arbiter.arbitrate(
        "clean up the users table", validator=validate_sql,
    )
    assert decision.action == "execute"
    # validate_sql rejects DROP → DELETE wins
    assert decision.chosen.output.startswith("DELETE")


@pytest.mark.asyncio
async def test_both_validator_rejected_routes_to_user():
    runner = make_runner({
        0.2: "DROP TABLE a",
        0.7: "DROP DATABASE b",
    })

    def embed(texts):
        return [[1.0, 0.0], [0.0, 1.0]]

    from ghost_agent.tools.validators import validate_sql

    arbiter = DualSolverArbiter(runner=runner, embedder=embed,
                                divergence_threshold=0.85)
    decision = await arbiter.arbitrate("zap", validator=validate_sql)
    assert decision.action == "ask_user"


@pytest.mark.asyncio
async def test_one_candidate_fails_other_executes():
    async def runner(payload):
        if payload["temperature"] == 0.2:
            raise RuntimeError("simulated")
        return "valid answer"
    arbiter = DualSolverArbiter(runner=runner)
    decision = await arbiter.arbitrate("question")
    assert decision.action == "execute"
    assert decision.chosen.temperature == 0.7


@pytest.mark.asyncio
async def test_both_fail_routes_to_user():
    async def runner(payload):
        raise RuntimeError("down")
    arbiter = DualSolverArbiter(runner=runner)
    decision = await arbiter.arbitrate("question")
    assert decision.action == "ask_user"


@pytest.mark.asyncio
async def test_timeout_handled_as_failure():
    async def runner(payload):
        await asyncio.sleep(0.5)
        return "late"
    arbiter = DualSolverArbiter(runner=runner, per_sample_timeout_s=0.05)
    decision = await arbiter.arbitrate("question")
    # Both should time out
    assert decision.action == "ask_user"
    assert all("timeout" in c.error for c in decision.candidates)


def test_default_per_sample_timeout_clears_real_model_latency():
    """Regression: the default per-sample timeout was 10.0s, shorter than a
    real LLM completion over Tor (20-40s), so BOTH candidates timed out
    every turn and the arbiter degenerated into a constant ask_user. The
    default must comfortably clear real model latency."""
    arbiter = DualSolverArbiter(runner=lambda p: "x")
    assert arbiter.per_sample_timeout_s >= 30.0


@pytest.mark.asyncio
async def test_empty_prompt_skipped():
    runner = make_runner({})
    arbiter = DualSolverArbiter(runner=runner)
    decision = await arbiter.arbitrate("")
    assert decision.action == "skipped"


@pytest.mark.asyncio
async def test_dict_runner_output_extracted():
    async def runner(payload):
        return {"output": "from dict", "extra": "ignored"}
    arbiter = DualSolverArbiter(runner=runner)
    decision = await arbiter.arbitrate("q")
    assert decision.action == "execute"
    assert decision.chosen.output == "from dict"


# ──────────────────────────────────────────────────────────────────────
# Sync runners — the Runner alias permits plain-sync callables, whose
# results used to bypass asyncio.wait_for entirely (the call blocked the
# event loop with no deadline)
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_sync_runner_supported():
    def runner(payload):
        return f"answer at T={payload['temperature']}"

    arbiter = DualSolverArbiter(runner=runner)
    decision = await arbiter.arbitrate("question")
    assert all(c.ok for c in decision.candidates)
    assert decision.chosen.output.startswith("answer at T=")


@pytest.mark.asyncio
async def test_sync_runner_timeout_applies():
    import time as _time

    def runner(payload):
        _time.sleep(0.5)  # would previously block the loop, deadline-free
        return "late"

    arbiter = DualSolverArbiter(runner=runner, per_sample_timeout_s=0.05)
    decision = await arbiter.arbitrate("question")
    assert decision.action == "ask_user"
    assert all("timeout" in c.error for c in decision.candidates)
    # The deadline actually applied — the samples returned at ~0.05s,
    # not after the runner's 0.5s block.
    assert all(c.duration_s < 0.4 for c in decision.candidates)


@pytest.mark.asyncio
async def test_sync_runner_returning_awaitable_still_works():
    # Third Runner shape: a sync callable that RETURNS an awaitable.
    def runner(payload):
        async def _inner():
            return "from awaitable"
        return _inner()

    arbiter = DualSolverArbiter(runner=runner)
    decision = await arbiter.arbitrate("question")
    assert decision.action == "execute"
    assert decision.chosen.output == "from awaitable"


# ──────────────────────────────────────────────────────────────────────
# Convergence with text-only fallback
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_jaccard_fallback_when_no_embedder():
    runner = make_runner({0.2: "the cat sat", 0.7: "the cat sat"})
    arbiter = DualSolverArbiter(runner=runner, embedder=None)
    decision = await arbiter.arbitrate("question")
    assert decision.action == "execute"
    assert decision.similarity == 1.0
