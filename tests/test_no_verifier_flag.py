"""The --no-verifier ablation off-switch: when set, the post-response verifier
is skipped entirely (no verdict computed) at BOTH compute call sites.

This exists so the ablation harness can measure whether the late (async) verifier
— which costs a full extra LLM call per substantive turn — earns its keep.
"""
from __future__ import annotations

import types

import pytest

from ghost_agent.core.agent import GhostAgent


def _fake_self(no_verifier: bool, verifier=None):
    args = types.SimpleNamespace(no_verifier=no_verifier)
    ctx = types.SimpleNamespace(args=args, verifier=verifier)
    return types.SimpleNamespace(context=ctx)


class _ExplodingVerifier:
    """Any attribute access means the verifier was (wrongly) engaged."""
    def __getattr__(self, name):
        raise AssertionError(f"verifier was touched despite --no-verifier ({name})")


async def test_no_verifier_skips_compute():
    fake = _fake_self(no_verifier=True, verifier=_ExplodingVerifier())
    v, last_tool = await GhostAgent._compute_verifier_verdict(
        fake,
        tools_run_this_turn=[],
        messages=[],
        final_ai_content="the answer",
        last_user_content="the question",
        lc=None,
    )
    assert v is None  # no verdict computed → ships unverified


async def test_no_verifier_skips_gated_front_door():
    # The gated path must also short-circuit (and not spawn a background task).
    fake = _fake_self(no_verifier=True, verifier=_ExplodingVerifier())
    v, last_tool = await GhostAgent._compute_verifier_verdict_gated(
        fake,
        tools_run_this_turn=[],
        messages=[],
        final_ai_content="the answer",
        last_user_content="the question",
        lc=None,
        trajectory_id="t1",
    )
    assert v is None


async def test_flag_absent_does_not_crash_guard():
    # A context without the attribute (older callers) must not raise in the guard.
    ctx = types.SimpleNamespace(args=types.SimpleNamespace(), verifier=None)
    fake = types.SimpleNamespace(context=ctx)
    v, last_tool = await GhostAgent._compute_verifier_verdict(
        fake, tools_run_this_turn=[], messages=[], final_ai_content="x",
        last_user_content="y", lc=None)
    # verifier is None → natural no-op path also returns None; the point is it
    # doesn't raise on a missing no_verifier attribute.
    assert v is None
