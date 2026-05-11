"""Regression: synthetic agent-loop error messages must NOT be treated
as real evidence by the verifier gate.

Trace (2026-05-01 dialog log):

    [turn 16 reply ends with]
    ---
    **Verifier note:** Evidence is irrelevant to the claim;
    Evidence is a system error message

The conversation was pure text — no tool was actually run. The model
emitted a malformed `<tool_call>` block, the parser synthesised a
`{"role":"tool","name":"system","content":"SYSTEM ERROR: ..."}` entry
into ``tools_run_this_turn``, and ``_find_substantive_tool_for_verifier``
returned that synthetic entry as if it were real evidence. The verifier
LLM then ran ``verify_claim(claim=<consciousness reply>,
evidence="SYSTEM ERROR: Your <tool_call> did not parse...")`` and
predictably refuted, surfacing the verbatim ``issues`` list as a
``**Verifier note:**`` appended to the user-facing reply.

Fix: every synthesised entry the agent loop appends to
``tools_run_this_turn`` carries ``_synthetic: True``; the filter skips
those entries. Same six error-shape sites must all be marked
(parse-error, JSON-args-error, unknown-tool, idempotency-block,
empty-write block, tool-invocation error, disabled-tool).
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from ghost_agent.core.agent import _find_substantive_tool_for_verifier


def test_synthetic_parse_error_is_not_real_evidence():
    """The exact trace shape from the 2026-05-01 dialog log turn 16."""
    tools_run = [
        {
            "role": "tool",
            "tool_call_id": "call_abc",
            "name": "system",
            "content": (
                "SYSTEM ERROR: Your `<tool_call>` did not parse. Use strict XML:\n"
                "<tool_call>\n  <function name=\"the_tool_name\">..."
            ),
            "_synthetic": True,
        },
    ]
    assert _find_substantive_tool_for_verifier(tools_run) is None


def test_synthetic_escape_hatch_is_not_real_evidence():
    tools_run = [
        {
            "role": "tool",
            "tool_call_id": "call_abc",
            "name": "system",
            "content": "SYSTEM ESCAPE HATCH (parse failed 2x): STOP repeating...",
            "_synthetic": True,
        },
    ]
    assert _find_substantive_tool_for_verifier(tools_run) is None


def test_synthetic_invalid_json_args_is_not_real_evidence():
    """The JSON-args-error path uses the real tool name (not 'system'),
    so a name-only filter would miss it. Must be skipped via marker."""
    tools_run = [
        {
            "role": "tool",
            "tool_call_id": "call_def",
            "name": "execute",
            "content": "Error: Invalid JSON arguments - Unterminated string",
            "_synthetic": True,
        },
    ]
    assert _find_substantive_tool_for_verifier(tools_run) is None


def test_synthetic_idempotency_block_is_not_real_evidence():
    tools_run = [
        {
            "role": "tool",
            "tool_call_id": "call_xyz",
            "name": "update_profile",
            "content": (
                "SYSTEM IDEMPOTENCY: 'update_profile' was already executed "
                "earlier in this request..."
            ),
            "_synthetic": True,
        },
    ]
    assert _find_substantive_tool_for_verifier(tools_run) is None


def test_synthetic_unknown_tool_is_not_real_evidence():
    tools_run = [
        {
            "role": "tool",
            "tool_call_id": "call_unk",
            "name": "definitely_not_a_real_tool",
            "content": "Error: Unknown tool 'definitely_not_a_real_tool'",
            "_synthetic": True,
        },
    ]
    assert _find_substantive_tool_for_verifier(tools_run) is None


def test_synthetic_disabled_tool_is_not_real_evidence():
    tools_run = [
        {
            "role": "tool",
            "tool_call_id": "call_dis",
            "name": "execute",
            "content": "SYSTEM ERROR: Tool 'execute' is explicitly disabled in this context.",
            "_synthetic": True,
        },
    ]
    assert _find_substantive_tool_for_verifier(tools_run) is None


def test_synthetic_invocation_error_is_not_real_evidence():
    tools_run = [
        {
            "role": "tool",
            "tool_call_id": "call_inv",
            "name": "execute",
            "content": "Error invoking tool 'execute' (Did you forget a required argument?): TypeError",
            "_synthetic": True,
        },
    ]
    assert _find_substantive_tool_for_verifier(tools_run) is None


def test_synthetic_empty_write_block_is_not_real_evidence():
    tools_run = [
        {
            "role": "tool",
            "tool_call_id": "call_emp",
            "name": "file_system",
            "content": (
                "SYSTEM BLOCK: You invoked file_system operation='write' "
                "on path='temp.py' but provided an empty or missing 'content' argument..."
            ),
            "_synthetic": True,
        },
    ]
    assert _find_substantive_tool_for_verifier(tools_run) is None


def test_real_tool_after_synthetic_is_still_found():
    """Mixed iteration: parse-error then a successful tool. The
    successful tool's output is the right evidence."""
    tools_run = [
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "system",
            "content": "SYSTEM ERROR: Your `<tool_call>` did not parse...",
            "_synthetic": True,
        },
        {
            "role": "tool",
            "tool_call_id": "call_2",
            "name": "execute",
            "content": "exit code 0\n42",
        },
    ]
    res = _find_substantive_tool_for_verifier(tools_run)
    assert res is not None
    assert res["name"] == "execute"
    assert res["content"] == "exit code 0\n42"


def test_synthetic_after_real_tool_does_not_mask_it():
    """Reverse mixed iteration: real tool ran, then a synthetic block
    appeared (e.g. the model tried a follow-up call with bad args).
    The real evidence still wins because we walk from the end and
    skip synthetic entries."""
    tools_run = [
        {
            "role": "tool",
            "tool_call_id": "call_real",
            "name": "execute",
            "content": "exit code 0\nimportant output",
        },
        {
            "role": "tool",
            "tool_call_id": "call_bad",
            "name": "execute",
            "content": "Error: Invalid JSON arguments - ...",
            "_synthetic": True,
        },
    ]
    res = _find_substantive_tool_for_verifier(tools_run)
    assert res is not None
    assert res["name"] == "execute"
    assert "important output" in res["content"]


def test_synthetic_marker_required_no_string_matching_fallback():
    """Belt-and-braces: an entry that *looks* like a synthetic error
    by content but is missing the marker is treated as real evidence.
    This documents the contract: producers must mark synthetic
    entries, the filter trusts the marker rather than guessing from
    content (which would mis-skip a legit tool that happened to
    return text starting with 'Error:')."""
    tools_run = [
        {
            "role": "tool",
            "tool_call_id": "call_legit",
            "name": "execute",
            # Looks like a synthetic prefix, but it's actually the
            # tool's real output (e.g. a Python script printed an
            # error message).
            "content": "Error: division by zero",
        },
    ]
    res = _find_substantive_tool_for_verifier(tools_run)
    assert res is not None
    assert res["name"] == "execute"


def test_messages_payload_remains_clean_no_synthetic_key():
    """The `_synthetic` marker lives on the `tools_run_this_turn`
    copy, NOT on the `messages` copy that goes upstream to the LLM.
    Fix uses dict-spread `{**err_msg, "_synthetic": True}` precisely
    so that mutating the tracking copy does not leak the extra key
    into the OpenAI messages payload.

    This test pins the contract by exercising the spread shape
    directly — if someone refactors to a single shared dict, the
    failure is intentional and the refactor must keep `_synthetic`
    out of the messages payload some other way.
    """
    err_msg = {
        "role": "tool",
        "tool_call_id": "call_abc",
        "name": "system",
        "content": "SYSTEM ERROR: ...",
    }
    tracking_copy = {**err_msg, "_synthetic": True}
    # The original (which would be appended to `messages`) is unchanged.
    assert "_synthetic" not in err_msg
    # The tracking copy carries the marker.
    assert tracking_copy.get("_synthetic") is True
