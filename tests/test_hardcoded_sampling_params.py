"""Tests for the hardcoded LLM sampling profiles.

The legacy code dynamically raised the temperature on tool-call
failures, emitting "Adjusting variance to X to solve error" log lines.
That adaptive loop was removed in favor of two fixed profiles:

  General (non-coding): temp=1.0, top_p=0.95, top_k=20, min_p=0, presence_penalty=1.5
  Coding  (specialist): temp=0.6, top_p=0.95, top_k=20, min_p=0, presence_penalty=0

These tests pin the profiles, prove the selector picks the right one
based on coding intent, and confirm the agent threads those exact
numbers into the upstream chat payload — while the dynamic-variance
code path (and its `--temperature` CLI arg) are gone for good.
"""

import argparse
import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.core import agent as agent_module
from ghost_agent.core.agent import (
    CODING_SAMPLING_PARAMS,
    GENERAL_SAMPLING_PARAMS,
    GhostAgent,
    get_sampling_params,
)


# ---------------------------------------------------------------------------
# 1. The two profiles are frozen to the spec
# ---------------------------------------------------------------------------

def test_general_profile_matches_spec():
    assert GENERAL_SAMPLING_PARAMS == {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0,
        "presence_penalty": 1.5,
    }


def test_coding_profile_matches_spec():
    assert CODING_SAMPLING_PARAMS == {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0,
        "presence_penalty": 0,
    }


def test_profiles_contain_all_required_keys():
    required = {"temperature", "top_p", "top_k", "min_p", "presence_penalty"}
    assert required <= GENERAL_SAMPLING_PARAMS.keys()
    assert required <= CODING_SAMPLING_PARAMS.keys()


# ---------------------------------------------------------------------------
# 2. get_sampling_params() picks by coding intent and returns a *copy*
# ---------------------------------------------------------------------------

def test_get_sampling_params_coding_branch():
    assert get_sampling_params(True) == CODING_SAMPLING_PARAMS


def test_get_sampling_params_general_branch():
    assert get_sampling_params(False) == GENERAL_SAMPLING_PARAMS


def test_get_sampling_params_returns_independent_copy():
    """Callers mutate the returned dict (merging into a payload).
    The module-level constants must stay pristine."""
    a = get_sampling_params(True)
    a["temperature"] = 42.0
    assert CODING_SAMPLING_PARAMS["temperature"] == 0.6
    b = get_sampling_params(False)
    b["presence_penalty"] = 42.0
    assert GENERAL_SAMPLING_PARAMS["presence_penalty"] == 1.5


def test_get_sampling_params_truthy_and_falsy_inputs():
    # The function takes the bool literally; any truthy value = coding.
    assert get_sampling_params(1) == CODING_SAMPLING_PARAMS
    assert get_sampling_params(0) == GENERAL_SAMPLING_PARAMS
    assert get_sampling_params("coding") == CODING_SAMPLING_PARAMS
    assert get_sampling_params("") == GENERAL_SAMPLING_PARAMS
    assert get_sampling_params(None) == GENERAL_SAMPLING_PARAMS


# ---------------------------------------------------------------------------
# 3. The dynamic-variance logic is really gone
# ---------------------------------------------------------------------------

def test_dynamic_temperature_adjustment_source_removed():
    src = inspect.getsource(agent_module)
    assert "Adjusting variance" not in src, \
        "Adaptive temperature log line should have been deleted"
    assert "active_temp" not in src, \
        "active_temp was the dynamic-temp variable; it should be gone"
    assert "current_temp" not in src, \
        "current_temp fed active_temp; it should be gone"
    assert "args.temperature" not in src, \
        "The --temperature CLI arg is no longer read by the agent"


def test_cli_no_longer_exposes_temperature_flag():
    from ghost_agent import main as main_module
    parser = argparse.ArgumentParser()
    # Rebuild the same parser main.py builds, by calling the parser
    # construction block indirectly: we inspect main.py's source.
    src = inspect.getsource(main_module)
    assert '"--temperature"' not in src
    assert '"-t"' not in src


# ---------------------------------------------------------------------------
# 4. End-to-end: the agent threads the fixed params into the LLM payload
# ---------------------------------------------------------------------------

def _make_agent_with_captured_payload():
    """Build a GhostAgent whose LLM client records every payload it sees."""
    ctx = MagicMock()
    ctx.args = MagicMock()
    ctx.args.verbose = False
    ctx.args.max_context = 32768
    ctx.args.smart_memory = 0.0
    ctx.args.use_planning = False
    ctx.args.no_memory = True

    captured = []

    async def fake_chat_completion(payload, *a, **kw):
        captured.append(dict(payload))
        return {
            "choices": [{
                "message": {"content": "ok", "tool_calls": []},
                "finish_reason": "stop",
            }]
        }

    ctx.llm_client = MagicMock()
    ctx.llm_client.chat_completion = AsyncMock(side_effect=fake_chat_completion)
    ctx.llm_client.stream_chat_completion = AsyncMock()

    # Minimal memory stubs that return strings where the agent expects them.
    for attr in ("profile_memory", "scratchpad", "vector_memory",
                 "skill_memory", "journal", "graph_memory"):
        m = MagicMock()
        setattr(ctx, attr, m)
    ctx.profile_memory.get_context_string.return_value = ""
    ctx.scratchpad.list_all.return_value = ""

    return ctx, captured


def test_payload_merges_sampling_params_without_extras():
    """Unit-level check: the merging idiom used in agent.py produces a
    payload that contains exactly the five sampling keys plus
    frequency_penalty, and nothing resembling an `active_temp` override.

    Post-audit signature: `get_sampling_params(is_tool_turn, query,
    is_coding)`. A tool-using turn (coding OR non-coding) routes through
    the coding base profile; only conversational turns use GENERAL.
    """
    sampling = get_sampling_params(is_tool_turn=True, is_coding=True)
    payload = {
        "model": "test-model",
        "messages": [],
        "stream": False,
        **sampling,
        "frequency_penalty": 0.0,
    }
    assert payload["temperature"] == 0.6
    assert payload["top_p"] == 0.95
    assert payload["top_k"] == 20
    assert payload["min_p"] == 0
    assert payload["presence_penalty"] == 0
    assert payload["frequency_penalty"] == 0.0
    # No leftover keys from the adaptive era.
    assert "active_temp" not in payload
    assert "variance" not in payload


def test_payload_general_profile_merge():
    # Conversational turn (is_tool_turn=False) → GENERAL.
    sampling = get_sampling_params(is_tool_turn=False)
    payload = {"model": "m", "messages": [], "stream": False, **sampling}
    assert payload["temperature"] == 1.0
    assert payload["presence_penalty"] == 1.5
    assert payload["top_p"] == 0.95
    assert payload["top_k"] == 20
    assert payload["min_p"] == 0


# ---------------------------------------------------------------------------
# 5. Selector contract: any non-conversational turn routes precise.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("is_tool_turn,is_coding,expected", [
    (True,  True,  CODING_SAMPLING_PARAMS),   # coding tool turn → precise base
    (True,  False, CODING_SAMPLING_PARAMS),   # non-coding tool turn → precise base
    (False, False, GENERAL_SAMPLING_PARAMS),  # conversational turn → general
    (False, True,  GENERAL_SAMPLING_PARAMS),  # conversational even if coding flag set
])
def test_selector_contract(is_tool_turn, is_coding, expected):
    """Every non-conversational turn must drop temperature to 0.6.

    The earlier contract gated the drop on coding-intent only, which
    left `update_profile` / `web_search` / `manage_tasks` on temp=1.0
    and produced the duplicate-setter-call regression.
    """
    assert get_sampling_params(is_tool_turn, is_coding=is_coding) == expected


def test_call_site_routes_tool_turns_to_precise_profile():
    """Source-level pin: the agent's call site must pass a flag derived
    from `turn_is_conversational`, not just a coding-intent boolean.
    A future revert to the old shape would silently bring the
    duplicate-call regression back.
    """
    src = inspect.getsource(agent_module)
    assert "is_tool_turn = not turn_is_conversational" in src
    assert "get_sampling_params(" in src
    assert "is_coding=has_coding_intent and not is_meta_task" in src


# ---------------------------------------------------------------------------
# 6. Code-generating callers outside the main loop use CODING_SAMPLING_PARAMS
# ---------------------------------------------------------------------------

def test_synthetic_self_play_routes_to_coding_node_when_available():
    """The challenge generator writes `.setup.py` and `.validator.py`,
    so it must be routed to the Ghost Specialist (coding node) when
    one is configured — not the generic worker node, which runs on a
    smaller/general model with the wrong temperature profile."""
    from ghost_agent.core import dream as dream_module
    src = inspect.getsource(dream_module.Dreamer.synthetic_self_play)
    assert "use_coding=has_coding_node" in src, \
        "Challenge generation must prefer coding nodes when available"
    assert "use_worker=not has_coding_node" in src, \
        "Worker node must only be a fallback when no coding node is configured"


def test_synthetic_self_play_uses_coding_sampling_params():
    """Self-play generates Python `setup_script` and `validation_script`,
    so its challenge-generation call must ride the coding profile, not
    the legacy hand-tuned `temperature=0.6` literal.

    As of the self-play reliability redesign, challenge generation
    starts from a COPY of CODING_SAMPLING_PARAMS and then overrides
    `temperature` down to 0.3 — structured-XML emission converges
    faster with a tighter temperature. This test verifies both that
    the base profile is inherited AND that the intentional temperature
    override is in place."""
    from ghost_agent.core import dream as dream_module
    src = inspect.getsource(dream_module.Dreamer.synthetic_self_play)
    # Base profile must be inherited — covers every param other than
    # temperature (top_p, top_k, min_p, presence_penalty).
    assert "_challenge_sampling = dict(CODING_SAMPLING_PARAMS)" in src, (
        "synthetic_self_play must start from CODING_SAMPLING_PARAMS "
        "so the unchanged params (top_p, top_k, etc.) stay in sync"
    )
    # Spread into the payload.
    assert "**_challenge_sampling" in src, \
        "synthetic_self_play must spread the challenge sampling dict into its payload"
    # The deliberate temperature override.
    assert '_challenge_sampling["temperature"] = 0.3' in src, (
        "synthetic_self_play must lower temperature to 0.3 for challenge gen "
        "— 0.6 wanders too much and extends generation latency"
    )
    # The legacy literal is still forbidden.
    assert '"temperature": 0.6' not in src, (
        "Remove the hardcoded `temperature: 0.6` literal — "
        "use the _challenge_sampling override"
    )


def test_synthetic_self_play_logs_specialist_mode_switch():
    """Visual confirmation: the self-play challenge generator emits a
    'Ghost Specialist Activated' log line so logs make the mode switch
    obvious to operators watching the stream."""
    from ghost_agent.core import dream as dream_module
    src = inspect.getsource(dream_module.Dreamer.synthetic_self_play)
    assert "Ghost Specialist Activated" in src, \
        "synthetic_self_play must log the specialist mode switch before generation"


@pytest.mark.asyncio
async def test_synthetic_self_play_actually_sends_coding_params(monkeypatch, tmp_path, disable_self_play_templates):
    """End-to-end behavior check: when synthetic_self_play generates a
    challenge, the payload it hands to chat_completion must carry the
    exact CODING_SAMPLING_PARAMS values — not the general profile, not
    the legacy 0.6 literal, not anything else. We stub everything after
    the LLM call so the test doesn't actually spin up a Docker sandbox."""
    from ghost_agent.core import dream as dream_module
    from ghost_agent.core.agent import CODING_SAMPLING_PARAMS

    captured = []

    async def fake_chat_completion(payload, *a, **kw):
        captured.append({"payload": dict(payload), "kwargs": dict(kw)})
        # Return a deliberately malformed response so the generation
        # loop bails out early. That's fine — we only need to inspect
        # the payload that was sent to the LLM.
        return {"choices": [{"message": {"content": "<challenge_prompt>x</challenge_prompt>"}}]}

    ctx = MagicMock()
    ctx.llm_client = MagicMock()
    ctx.llm_client.chat_completion = AsyncMock(side_effect=fake_chat_completion)
    ctx.llm_client.coding_clients = [{"model": "coder", "client": MagicMock()}]
    ctx.frontier_tracker = None
    ctx.sandbox_dir = tmp_path
    ctx.skill_memory = MagicMock()
    ctx.memory_system = MagicMock()

    dreamer = dream_module.Dreamer.__new__(dream_module.Dreamer)
    dreamer.context = ctx
    dreamer.memory = MagicMock()
    dreamer.last_compression_delta = 0.0

    # Don't let the method proceed past challenge generation (we only
    # care about the LLM payload). We stub the isolated sandbox bring-up
    # to raise so the rest of the method short-circuits.
    monkeypatch.setattr(
        "ghost_agent.sandbox.docker.DockerSandbox",
        MagicMock(side_effect=RuntimeError("stop after generation")),
    )

    try:
        await dreamer.synthetic_self_play(model_name="test-model", is_background=True)
    except Exception:
        pass  # Expected — we intentionally stopped the run after generation.

    assert captured, "synthetic_self_play must call chat_completion at least once"
    first_payload = captured[0]["payload"]

    # Challenge generation is an EXCEPTION to the "identical to
    # CODING_SAMPLING_PARAMS" rule: `temperature` is intentionally
    # overridden to 0.3 (see _challenge_sampling in dream.py). Every
    # OTHER coding param must still match the shared profile so a
    # future top_p / top_k / presence_penalty change lands uniformly.
    CHALLENGE_GEN_TEMPERATURE_OVERRIDE = 0.3
    assert first_payload.get("temperature") == CHALLENGE_GEN_TEMPERATURE_OVERRIDE, (
        f"Self-play challenge payload[temperature]={first_payload.get('temperature')!r}, "
        f"expected {CHALLENGE_GEN_TEMPERATURE_OVERRIDE!r} "
        "(deliberate override for faster structured-XML convergence)"
    )
    for key, expected in CODING_SAMPLING_PARAMS.items():
        if key == "temperature":
            continue  # intentionally overridden above
        assert first_payload.get(key) == expected, (
            f"Self-play challenge payload[{key}]={first_payload.get(key)!r}, "
            f"expected {expected!r} (CODING_SAMPLING_PARAMS)"
        )
    # And it must have gone to the coding pool, not the worker pool.
    kwargs = captured[0]["kwargs"]
    assert kwargs.get("use_coding") is True
    assert kwargs.get("use_worker") is False


def test_coding_callers_do_not_mutate_shared_profile():
    """Callers use the `**CODING_SAMPLING_PARAMS` splat idiom, which
    creates a new dict per-payload. This test reasserts the invariant
    that the shared constant is never accidentally mutated."""
    from ghost_agent.core.agent import CODING_SAMPLING_PARAMS
    snapshot = dict(CODING_SAMPLING_PARAMS)
    # Simulate the exact merge used in agent.py and dream.py.
    payload = {"model": "m", "messages": [], **CODING_SAMPLING_PARAMS, "max_tokens": 8192}
    payload["temperature"] = 99.0
    assert CODING_SAMPLING_PARAMS == snapshot
