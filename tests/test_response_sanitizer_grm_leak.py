"""The non-streaming response sanitizer must strip Qwen's leaked
Generative-Reward-Model JSON.

The keys ``c_relevance_to_query``, ``c_correctness_of_content``,
``c_completeness_of_task``, ``c_communication_quality``, and
``c_potential_to_prefer_agent_response`` are NOT produced anywhere in
Ghost — they are a training-distribution artifact emitted by the
upstream Qwen model when a meta-prompt asks it to rate or score itself.
The double ``{{...}}`` braces are an f-string template escape carried
verbatim from evaluator prompts in the model's training corpus.

Observed during the consciousness-probe run on 2026-04-30,
attention-schema turn 4: the model emitted the GRM blob in place of a
real answer and the existing scrub block (XML tool tags, tool_response,
execution-result delimiters, plan markers) had no rule for it, so it
passed straight through to the API client.

The new regex (added in core/agent.py just before
``final_ai_content.strip()``) is conservative: it requires BOTH
``c_relevance_to_query`` AND ``c_correctness_of_content`` to appear
together with the right shape, so a legitimate response that happens
to mention one key in isolation is not clobbered.
"""

from __future__ import annotations

import re

# The exact regex used in core/agent.py — duplicated here so the
# behavioural guarantee can be unit-tested without spinning up the full
# agent. If the regex in agent.py changes, this test file must change
# too — that's intentional.
_GRM_REGEX = re.compile(
    r'\{\{?\s*"c_relevance_to_query"\s*:\s*\d+\s*,\s*"c_correctness_of_content"\s*:.*?\}\}?',
    flags=re.DOTALL,
)


def _scrub(s: str) -> str:
    return _GRM_REGEX.sub("", s).strip()


def test_double_brace_grm_blob_is_stripped():
    leaked = (
        '{{"c_relevance_to_query": 10, "c_correctness_of_content": 10, '
        '"c_completeness_of_task": 10, "c_communication_quality": 10, '
        '"c_potential_to_prefer_agent_response": 10}}'
    )
    assert _scrub(leaked) == ""


def test_single_brace_grm_blob_is_stripped():
    leaked = (
        '{"c_relevance_to_query": 8, "c_correctness_of_content": 9, '
        '"c_completeness_of_task": 10}'
    )
    assert _scrub(leaked) == ""


def test_grm_blob_followed_by_real_answer_keeps_only_answer():
    payload = (
        '{{"c_relevance_to_query": 10, "c_correctness_of_content": 10, '
        '"c_completeness_of_task": 10}}\n\n---\n\n'
        "The term 'artificial intelligence' was coined in 1956 by John McCarthy."
    )
    out = _scrub(payload)
    assert "c_relevance_to_query" not in out
    assert "John McCarthy" in out


def test_only_one_grm_key_present_is_left_alone():
    """A legitimate response that mentions just one of the GRM keys (e.g.
    in a docstring or explanation) must NOT be touched. The regex
    requires BOTH ``c_relevance_to_query`` AND ``c_correctness_of_content``
    to appear together."""
    legitimate = (
        "The evaluator schema includes a key called "
        '"c_relevance_to_query" which scores 0-10.'
    )
    assert _scrub(legitimate) == legitimate


def test_keys_in_wrong_order_do_not_match():
    """Conservative: regex requires the canonical key order
    (relevance then correctness). Any other order is left alone — the
    GRM template emits keys in this fixed order, so a different order
    is more likely a legitimate JSON response than a leak."""
    reversed_keys = '{"c_correctness_of_content": 10, "c_relevance_to_query": 10}'
    # No match → string preserved.
    assert _scrub(reversed_keys) == reversed_keys


def test_unrelated_json_payload_untouched():
    payload = '{"name": "ghost", "version": "0.1.24"}'
    assert _scrub(payload) == payload


def test_regex_is_present_in_agent_source():
    """Belt-and-braces: confirm the regex literal still lives in the
    source file. If someone deletes it during a refactor without also
    deleting this test file, the suite goes red and tells them why."""
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "src" / "ghost_agent" / "core" / "agent.py"
    text = src.read_text()
    assert '"c_relevance_to_query"' in text
    assert '"c_correctness_of_content"' in text
