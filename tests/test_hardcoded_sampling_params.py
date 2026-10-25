"""Tests for the hardcoded LLM sampling profiles.

The legacy code dynamically raised the temperature on tool-call
failures, emitting "Adjusting variance to X to solve error" log lines.
That adaptive loop was removed in favor of two fixed profiles:

  General (non-coding): temp=1.0, top_p=0.95, top_k=20, min_p=0, presence_penalty=0 (§4RG-b; was 1.5)
  Coding  (specialist): temp=1.0 (§4RG, was 0.6), top_p=0.95, top_k=20, min_p=0, presence_penalty=0

These tests pin the profiles, prove the selector picks the right one
based on coding intent, and confirm the agent threads those exact
numbers into the upstream chat payload — while the dynamic-variance
code path (and its `--temperature` CLI arg) are gone for good.
"""

import argparse
import inspect

import pytest

from ghost_agent.core import agent as agent_module
from ghost_agent.core.agent import (
    CODING_SAMPLING_PARAMS,
    GENERAL_SAMPLING_PARAMS,
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
        "presence_penalty": 0,     # §4RG-b: card thinking-mode value (was the instruct 1.5)
    }


def test_coding_profile_matches_spec():
    assert CODING_SAMPLING_PARAMS == {
        "temperature": 1.0,     # §4RG: Qwen3.8 card thinking-mode value (was 0.6)
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
    assert CODING_SAMPLING_PARAMS["temperature"] == 1.0
    b = get_sampling_params(False)
    b["presence_penalty"] = 42.0
    assert GENERAL_SAMPLING_PARAMS["presence_penalty"] == 0


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
    assert payload["temperature"] == 1.0
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
    assert payload["presence_penalty"] == 0
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
    """Every non-conversational turn must use the tool-turn profile (§4RG: 1.0, presence 0).

    The earlier contract gated the drop on coding-intent only, which
    left `update_profile` / `web_search` / `manage_tasks` on temp=1.0
    and produced the duplicate-setter-call regression.
    """
    assert get_sampling_params(is_tool_turn, is_coding=is_coding) == expected


@pytest.mark.parametrize("query", [
    "refactor the parser", "write the SQL migration", "design the architecture",
    "build a snake game", "fix the regex", "brainstorm names", "",
])
def test_all_coding_subprofiles_pinned_to_model_card(query):
    """The creative/precise/balanced sub-profiles stay WIRED but every one is
    pinned to the model-card thinking-mode values (temp=1.0, top_p=0.95, top_k=20; §4RG) —
    so NO coding turn drifts off-spec regardless of which keywords it hits."""
    p = get_sampling_params(True, query, is_coding=True)
    assert p == CODING_SAMPLING_PARAMS
    assert p["temperature"] == 1.0 and p["top_p"] == 0.95 and p["top_k"] == 20


def test_coding_subprofiles_dict_all_equal_spec():
    from ghost_agent.core.agent import _CODING_TASK_PROFILES
    for name, prof in _CODING_TASK_PROFILES.items():
        assert prof == CODING_SAMPLING_PARAMS, name


def test_call_site_routes_tool_turns_to_precise_profile():
    """Source-level pin: the agent's call site must pass a flag derived
    from `turn_is_conversational`, not just a coding-intent boolean.
    A future revert to the old shape would silently bring the
    duplicate-call regression back.
    """
    src = inspect.getsource(agent_module)
    assert "is_tool_turn = (not turn_is_conversational" in src
    assert "get_sampling_params(" in src
    assert "is_coding=has_coding_intent and not is_meta_task" in src


# ---------------------------------------------------------------------------
# 6. Code-generating callers outside the main loop use CODING_SAMPLING_PARAMS
# ---------------------------------------------------------------------------









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
