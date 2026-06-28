"""Tests for the factual-vs-conversational sampling policy.

Redesign #1: a non-tool turn is NOT automatically chit-chat. A graded/factual
query must sample near-greedy (temp low, presence_penalty=0) instead of the
conversational temp=1.0 / presence_penalty=1.5 profile, which samples AWAY from
the correct answer tokens.
"""

from ghost_agent.core.agent import (
    _is_factual_query,
    get_sampling_params,
    FACTUAL_SAMPLING_PARAMS,
    GENERAL_SAMPLING_PARAMS,
    CODING_SAMPLING_PARAMS,
)


import pytest


@pytest.mark.parametrize("q", [
    "What is the capital of France?",
    "How many vowels are in the word banana?",
    "what is 7 times 6?",
    "Compute 2 to the power of 20.",
    "How much is 15% of 240?",
    "Which is larger, 3/7 or 5/12?",
    "what year did the Berlin wall fall?",
    "count the prime numbers below 100",
])
def test_factual_queries_detected(q):
    assert _is_factual_query(q) is True


@pytest.mark.parametrize("q", [
    "how are you?",
    "hi",
    "tell me a story about a dragon",
    "write a poem",
    "what do you think about modern art",
    "good morning!",
    "thanks for your help",
    "",
])
def test_conversational_queries_not_factual(q):
    assert _is_factual_query(q) is False


def test_factual_turn_samples_near_greedy():
    sp = get_sampling_params(is_tool_turn=False, query="what is 7 times 6?")
    assert sp["temperature"] <= 0.3
    assert sp["presence_penalty"] == 0
    assert sp == FACTUAL_SAMPLING_PARAMS


def test_chitchat_turn_keeps_warm_profile():
    sp = get_sampling_params(is_tool_turn=False, query="tell me a story")
    assert sp == GENERAL_SAMPLING_PARAMS
    assert sp["temperature"] == 1.0


def test_tool_turn_unaffected():
    sp = get_sampling_params(is_tool_turn=True, query="anything")
    # tool turns still use the precise coding/base profile, not factual/general
    assert sp["presence_penalty"] == 0
    assert sp["temperature"] == CODING_SAMPLING_PARAMS["temperature"]


def test_factual_profile_is_a_copy():
    # callers must not be able to mutate the module-level dict
    sp = get_sampling_params(is_tool_turn=False, query="how many primes below 10?")
    sp["temperature"] = 99
    assert FACTUAL_SAMPLING_PARAMS["temperature"] != 99
