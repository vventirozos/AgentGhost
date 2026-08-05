"""Upstream token accounting.

`Trajectory.tokens_in/out` and `eval.TaskResult.tokens_used` were declared
fields that read 0 on every record for months, because nothing read the
upstream `usage` block. These tests pin the capture so it cannot silently
go back to zero — the field existing is not evidence that it is populated.
"""
import json

import pytest

from ghost_agent.core.llm import LLMClient
from ghost_agent.utils.logging import request_id_context


def _client():
    return LLMClient("http://upstream.invalid:8080")


def _resp(prompt=0, completion=0, cached=None):
    usage = {"prompt_tokens": prompt, "completion_tokens": completion,
             "total_tokens": prompt + completion}
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    return {"choices": [{"message": {"content": "x"}}], "usage": usage}


def _with_req(req_id, fn):
    token = request_id_context.set(req_id)
    try:
        return fn()
    finally:
        request_id_context.reset(token)


# ---------------------------------------------------------------- summing

def test_usage_sums_across_calls_in_one_request():
    """A turn makes many calls; the trajectory number is the TURN total."""
    c = _client()
    _with_req("req-a", lambda: (c._note_usage(_resp(100, 20)),
                                c._note_usage(_resp(300, 45))))
    got = c.usage_for("req-a")
    assert got["tokens_in"] == 400
    assert got["tokens_out"] == 65
    assert got["calls"] == 2


def test_requests_do_not_bleed_into_each_other():
    c = _client()
    _with_req("req-a", lambda: c._note_usage(_resp(10, 1)))
    _with_req("req-b", lambda: c._note_usage(_resp(500, 70)))
    assert c.usage_for("req-a")["tokens_in"] == 10
    assert c.usage_for("req-b")["tokens_in"] == 500


def test_cached_prompt_tokens_are_tracked_separately():
    """The prefill cache is measured in CHARACTERS in the log today; this
    is the first real hit count."""
    c = _client()
    _with_req("req-a", lambda: c._note_usage(_resp(1000, 10, cached=980)))
    assert c.usage_for("req-a")["cached_tokens"] == 980


# ------------------------------------------------------------- robustness

def test_unknown_request_is_distinguishable_from_a_zero_token_turn():
    c = _client()
    assert c.usage_for("never-seen") == {}
    _with_req("req-z", lambda: c._note_usage(_resp(0, 0)))
    assert c.usage_for("req-z") == {"tokens_in": 0, "tokens_out": 0,
                                    "cached_tokens": 0, "calls": 1}


@pytest.mark.parametrize("result", [
    None, "a bare string from the route path", 42, [], {},
    {"choices": []},                       # no usage key at all
    {"usage": "not-a-dict"},               # malformed
    {"usage": {"prompt_tokens": "abc"}},   # non-numeric
])
def test_malformed_results_never_raise(result):
    """Accounting must never break a turn — the route path hands this a
    plain string."""
    c = _client()
    _with_req("req-a", lambda: c._note_usage(result))
    assert c.usage_for("req-a").get("tokens_in", 0) == 0


def test_ring_is_bounded():
    c = _client()
    for i in range(LLMClient._USAGE_RING_MAX + 10):
        _with_req(f"req-{i}", lambda: c._note_usage(_resp(1, 1)))
    assert len(c._usage_ring()) <= LLMClient._USAGE_RING_MAX
    # the most recent survive; the oldest are evicted
    assert c.usage_for(f"req-{LLMClient._USAGE_RING_MAX + 9}")["calls"] == 1
    assert c.usage_for("req-0") == {}


# ---------------------------------------------------------------- streaming

def test_usage_parsed_from_the_final_sse_chunk():
    """An OpenAI-compatible server sends usage in a final chunk whose
    `choices` is empty."""
    c = _client()
    final = "data: " + json.dumps({
        "choices": [],
        "usage": {"prompt_tokens": 11, "completion_tokens": 1,
                  "prompt_tokens_details": {"cached_tokens": 7}},
    })
    _with_req("req-s", lambda: c._note_usage_from_sse(final))
    got = c.usage_for("req-s")
    assert (got["tokens_in"], got["tokens_out"], got["cached_tokens"]) == (11, 1, 7)


@pytest.mark.parametrize("line", [
    "", "data: [DONE]", "data: {not json", ": keepalive", "garbage",
    b"", b"data: [DONE]", b"garbage",
])
def test_non_usage_sse_lines_are_ignored(line):
    c = _client()
    _with_req("req-s", lambda: c._note_usage_from_sse(line))
    assert c.usage_for("req-s") == {}


def test_sse_usage_is_parsed_from_bytes_too():
    """Chunks arrive as `str` from `aiter_lines()` but as `bytes` on other
    paths. Assuming either one raised TypeError in the stream loop and broke
    four unrelated test modules — the cost of guessing a chunk's type."""
    c = _client()
    final = ("data: " + json.dumps(
        {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 2}}
    )).encode("utf-8")
    _with_req("req-b", lambda: c._note_usage_from_sse(final))
    got = c.usage_for("req-b")
    assert (got["tokens_in"], got["tokens_out"]) == (9, 2)


def test_every_completion_path_counts_usage():
    """A missed funnel under-reports the turn's cost silently, and in the
    direction that flatters it. `route()` was missed on the first pass — it
    serves verify / decompose / classification on the worker pool.
    """
    import ast
    import inspect
    import textwrap

    for name in ("route", "chat_completion", "_do_stream_chat_completion"):
        fn = getattr(LLMClient, name)
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        calls = {c.func.attr for c in ast.walk(tree)
                 if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)}
        assert calls & {"_note_usage", "_note_usage_from_sse"}, (
            f"LLMClient.{name} completes a call without counting its tokens")


def test_api_usage_block_survives_a_mock_llm_client():
    """The route reads `usage_for` off the live client. A MagicMock returns a
    truthy mock whose .get() is also a mock — which serialised to a 500
    instead of a reply, breaking 10 API tests. Only real dicts get through."""
    from unittest.mock import MagicMock
    llm = MagicMock()
    usage = llm.usage_for("req-1")
    assert not isinstance(usage, dict), "precondition: mock is not a dict"
    assert usage, "precondition: mock is truthy — truthiness alone is not a gate"


def test_streaming_payload_requests_usage():
    """Without `stream_options.include_usage` the server sends no usage on a
    stream at all, so this flag is the whole feature on the streamed path —
    which is the path the main tool loop uses."""
    import inspect
    src = inspect.getsource(LLMClient._do_stream_chat_completion)
    assert "stream_options" in src
    assert "include_usage" in src


def test_stream_usage_capture_is_not_gated_on_the_opt_in_recorder():
    """`_stream_rec_accumulate` only runs under GHOST_LLM_RECORD=1 (off by
    default). Token accounting must run on EVERY stream, so its call site
    must not sit inside that gate.

    Asserted structurally rather than by indentation: the two calls are
    legitimately at the same depth under different guards, so an indent
    comparison reports a false failure.
    """
    import ast
    import inspect
    import textwrap

    src = textwrap.dedent(inspect.getsource(LLMClient._do_stream_chat_completion))
    tree = ast.parse(src)

    def calls_in(node):
        return {n.func.attr for n in ast.walk(node)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}

    assert "_note_usage_from_sse" in calls_in(tree), \
        "streaming path never calls _note_usage_from_sse"

    gates_checked = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test_names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
        if "_rec_on" not in test_names:
            continue
        gates_checked += 1
        body_calls = set()
        for stmt in node.body:
            body_calls |= calls_in(stmt)
        assert "_note_usage_from_sse" not in body_calls, (
            "usage capture sits inside an `if _rec_on:` block — it would be "
            "dark whenever GHOST_LLM_RECORD is off, which is the default")

    # Without this the test passes vacuously the moment the recorder gate is
    # renamed — which is precisely how a guard stops guarding.
    assert gates_checked, (
        "no `if _rec_on:` gate found in the streaming path; this test no "
        "longer checks anything — update it to the new gate name")
