"""Unit tests for MetacogBundle.arbitrate_tool_calls — the mid-turn
auto-route gate that decides whether the dual-solver arbiter fires
before a tool is dispatched.

Five-gate contract:
  1. Bundle disabled                       → None
  2. Tool not in GATED_DOMAINS             → None (cheap pass-through)
  3. Arbitration cap exhausted             → None
  4. No prior confidence reading           → None (cold start)
  5. Last confidence above threshold       → None
  Otherwise → arbiter is run and decision returned.
"""

from __future__ import annotations

import argparse
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.core.arbiter import ArbitrationDecision, Candidate
from ghost_agent.core.confidence import ConfidenceReading
from ghost_agent.core.metacog import MetacogBundle


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

def _args(**overrides):
    a = argparse.Namespace(
        enable_metacog=True,
        metacog_confidence_threshold=0.55,
        metacog_disable_logprobs=False,
        metacog_disable_arbiter=False,
    )
    for k, v in overrides.items():
        setattr(a, k, v)
    return a


@pytest.fixture
def bundle(tmp_path):
    ctx = MagicMock()
    ctx.memory_dir = tmp_path
    ctx.args = MagicMock()
    ctx.args.model = "test"
    ctx.llm_client = MagicMock()
    if hasattr(ctx.llm_client, "get_embeddings"):
        del ctx.llm_client.get_embeddings
    ctx.project_store = None
    ctx.current_project_id = None
    b = MetacogBundle.from_args(ctx, _args())
    return b


def _msgs(*texts):
    """Build a fake messages list with alternating user/assistant turns
    where the last user message is `texts[-1]` (most recent)."""
    out = []
    for i, t in enumerate(texts):
        role = "user" if i % 2 == 0 else "assistant"
        out.append({"role": role, "content": t})
    return out


def _below_threshold_reading():
    return ConfidenceReading(
        composite=0.3, entropy_component=0.4, competence_component=0.2,
        threshold=0.55, below_threshold=True,
    )


def _above_threshold_reading():
    return ConfidenceReading(
        composite=0.9, entropy_component=0.95, competence_component=0.85,
        threshold=0.55, below_threshold=False,
    )


def _stub_decision(action="execute", reason="stubbed"):
    return ArbitrationDecision(
        action=action,
        chosen=Candidate(output="A", temperature=0.2),
        other=Candidate(output="B", temperature=0.7),
        similarity=0.9 if action == "execute" else 0.4,
        reason=reason,
        candidates=[],
    )


# ──────────────────────────────────────────────────────────────────────
# Gate 1: bundle disabled
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_disabled_bundle_returns_none(tmp_path):
    """Static contract: a disabled bundle never invokes the arbiter."""
    bundle = MetacogBundle()  # default enabled=False
    out = await bundle.arbitrate_tool_calls(
        messages=_msgs("delete things"), tool_name="execute",
    )
    assert out is None


@pytest.mark.asyncio
async def test_arbiter_off_returns_none(bundle):
    bundle.arbiter_enabled = False
    bundle.arbiter = None
    bundle.record_confidence(_below_threshold_reading())
    out = await bundle.arbitrate_tool_calls(
        messages=_msgs("rm -rf /tmp/foo"), tool_name="execute",
    )
    assert out is None


# ──────────────────────────────────────────────────────────────────────
# Gate 2: tool domain not in GATED_DOMAINS
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_non_mutating_domain_skips_arbitration(bundle):
    bundle.record_confidence(_below_threshold_reading())
    bundle.arbiter.arbitrate = AsyncMock(return_value=_stub_decision())
    # web_search is in "fetch" domain — not gated
    out = await bundle.arbitrate_tool_calls(
        messages=_msgs("look something up"), tool_name="web_search",
    )
    assert out is None
    bundle.arbiter.arbitrate.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("tool,gated", [
    ("execute", True),
    ("postgres_admin", True),
    ("file_system", False),
    ("web_search", False),
    ("update_profile", False),
    ("vision", False),
])
async def test_gating_per_domain(bundle, tool, gated):
    bundle.record_confidence(_below_threshold_reading())
    bundle.arbiter.arbitrate = AsyncMock(return_value=_stub_decision())
    out = await bundle.arbitrate_tool_calls(
        messages=_msgs("do the thing"), tool_name=tool,
    )
    if gated:
        assert out is not None
        bundle.arbiter.arbitrate.assert_called_once()
    else:
        assert out is None
        bundle.arbiter.arbitrate.assert_not_called()


# ──────────────────────────────────────────────────────────────────────
# Gate 3: per-request arbitration cap
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_arbitration_cap_per_request(bundle):
    bundle.record_confidence(_below_threshold_reading())
    bundle.arbiter.arbitrate = AsyncMock(return_value=_stub_decision())
    # First call exhausts the cap (MAX = 1)
    first = await bundle.arbitrate_tool_calls(
        messages=_msgs("clean up users"), tool_name="postgres_admin",
    )
    assert first is not None
    second = await bundle.arbitrate_tool_calls(
        messages=_msgs("now do something else"), tool_name="execute",
    )
    assert second is None
    assert bundle.arbiter.arbitrate.call_count == 1


@pytest.mark.asyncio
async def test_reset_counter_re_enables_arbitration(bundle):
    bundle.record_confidence(_below_threshold_reading())
    bundle.arbiter.arbitrate = AsyncMock(return_value=_stub_decision())
    await bundle.arbitrate_tool_calls(
        messages=_msgs("first turn"), tool_name="execute",
    )
    bundle.reset_arbitration_counter()
    out = await bundle.arbitrate_tool_calls(
        messages=_msgs("second user turn"), tool_name="execute",
    )
    assert out is not None
    assert bundle.arbiter.arbitrate.call_count == 2


# ──────────────────────────────────────────────────────────────────────
# Gate 4: cold start (no prior reading)
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cold_start_no_arbitration(bundle):
    bundle.arbiter.arbitrate = AsyncMock(return_value=_stub_decision())
    # No record_confidence call
    out = await bundle.arbitrate_tool_calls(
        messages=_msgs("delete it"), tool_name="execute",
    )
    assert out is None
    bundle.arbiter.arbitrate.assert_not_called()


@pytest.mark.asyncio
async def test_force_bypasses_cold_start(bundle):
    bundle.arbiter.arbitrate = AsyncMock(return_value=_stub_decision())
    # No record_confidence — but force=True
    out = await bundle.arbitrate_tool_calls(
        messages=_msgs("delete it"), tool_name="execute", force=True,
    )
    assert out is not None


# ──────────────────────────────────────────────────────────────────────
# Gate 5: above-threshold confidence
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_above_threshold_no_arbitration(bundle):
    bundle.record_confidence(_above_threshold_reading())
    bundle.arbiter.arbitrate = AsyncMock(return_value=_stub_decision())
    out = await bundle.arbitrate_tool_calls(
        messages=_msgs("do the thing"), tool_name="execute",
    )
    assert out is None
    bundle.arbiter.arbitrate.assert_not_called()


# ──────────────────────────────────────────────────────────────────────
# Gate happy path: fires on the right conditions
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fires_on_low_confidence_mutating_tool(bundle):
    bundle.record_confidence(_below_threshold_reading())
    expected = _stub_decision(action="execute")
    bundle.arbiter.arbitrate = AsyncMock(return_value=expected)
    out = await bundle.arbitrate_tool_calls(
        messages=_msgs("clean up users"), tool_name="postgres_admin",
    )
    assert out is expected
    # Arbiter must have been called with the user's most recent message
    bundle.arbiter.arbitrate.assert_called_once_with("clean up users")


@pytest.mark.asyncio
async def test_returns_ask_user_decision_unchanged(bundle):
    bundle.record_confidence(_below_threshold_reading())
    expected = _stub_decision(action="ask_user", reason="diverged")
    bundle.arbiter.arbitrate = AsyncMock(return_value=expected)
    out = await bundle.arbitrate_tool_calls(
        messages=_msgs("zap the table"), tool_name="postgres_admin",
    )
    assert out.action == "ask_user"
    assert out.reason == "diverged"


# ──────────────────────────────────────────────────────────────────────
# Safety: no user message → no arbitration
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_no_user_message_returns_none(bundle):
    bundle.record_confidence(_below_threshold_reading())
    bundle.arbiter.arbitrate = AsyncMock(return_value=_stub_decision())
    out = await bundle.arbitrate_tool_calls(
        messages=[{"role": "assistant", "content": "thinking..."}],
        tool_name="execute",
    )
    assert out is None
    bundle.arbiter.arbitrate.assert_not_called()


@pytest.mark.asyncio
async def test_arbiter_exception_returns_none(bundle):
    bundle.record_confidence(_below_threshold_reading())
    bundle.arbiter.arbitrate = AsyncMock(side_effect=RuntimeError("upstream down"))
    out = await bundle.arbitrate_tool_calls(
        messages=_msgs("clean up"), tool_name="execute",
    )
    assert out is None


# ──────────────────────────────────────────────────────────────────────
# Confidence stash round-trip
# ──────────────────────────────────────────────────────────────────────

def test_record_confidence_round_trip(bundle):
    r = _below_threshold_reading()
    bundle.record_confidence(r)
    assert bundle._last_confidence is r
    bundle.record_confidence(_above_threshold_reading())
    assert bundle._last_confidence.below_threshold is False
