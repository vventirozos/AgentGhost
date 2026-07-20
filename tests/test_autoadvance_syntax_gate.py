"""Autoadvance consumes the post-write syntax-check signal (2026-07-14).

Observed live (first build after the marker-leak deploy): during a 6-task
`autoadvance` batch, every rewrite of index.html carried the same
`⚠ SYNTAX CHECK FAILED … Identifier 'WebOS' has already been declared`
warning in its write result — the check fired five times — yet every task
closed DONE and the broken build was only caught when the final turn browsed
the page. The executor discarded the diagnostic: `_looks_like_write_error`
only inspects the head of the result, and nothing else read the warning.

`_syntax_fail_reason` now turns that warning into an apply-failure so the
build's retry-with-feedback loop gets the exact line to fix; on exhausted
retries the task fails honestly (stopping the batch) instead of piling more
features onto a file that doesn't parse.
"""
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.core.coding_executor import (
    MAX_ATTEMPTS,
    _apply_edits,
    _apply_file,
    _syntax_fail_reason,
    build_coding_task,
)


def _ctx(llm):
    return SimpleNamespace(llm_client=llm, args=SimpleNamespace(model="m"))


class FakeLLM:
    def __init__(self, content):
        self.content = content
        self.calls = 0
        self.payloads = []

    async def chat_completion(self, payload, is_background=False, **_kw):
        self.calls += 1
        self.payloads.append(payload)
        return {"choices": [{"message": {"content": self.content}}]}

    def user_prompt(self, i):
        msgs = self.payloads[i]["messages"]
        return next(m["content"] for m in msgs if m["role"] == "user")


_TAINT = (
    "SUCCESS: File 'index.html' written.\n"
    "⚠ SYNTAX CHECK FAILED: 'index.html' was written but does NOT parse. "
    "Fix this BEFORE any other step — a browser loads a broken script "
    "silently and every downstream symptom (dead buttons, blank page) "
    "traces back here:\n"
    "line 495: SyntaxError: Identifier 'WebOS' has already been declared"
)

_SPEC_HTML = json.dumps({
    "files": [{"path": "index.html",
               "content": "<html><script>const a=1;</script></html>"}],
    "summary": "wrote index.html",
    "ledger": "index.html shell",
})


class SeqRunner:
    """file_system returns the next canned output per call; execute returns OK.
    Reads are answered out-of-band (``read_out``) so the canned write/replace
    sequence stays aligned now that the executor live-reads before appends and
    between retry attempts (2026-07-20 snapshot-clobber fixes)."""
    def __init__(self, outs, read_out=None):
        self.outs = list(outs)
        self.calls = []
        self.read_out = read_out

    async def __call__(self, name, args):
        self.calls.append((name, args))
        if name == "file_system":
            if args.get("operation") == "read":
                if self.read_out is not None:
                    return self.read_out
                return (f"Error: '{args.get('path')}' does not exist in the "
                        f"current project's sandbox.")
            return self.outs.pop(0) if self.outs else "SUCCESS: File written."
        return "OK"

    def writes(self):
        return [a for (n, a) in self.calls
                if n == "file_system" and a.get("operation") != "read"]


# ---------------------------------------------------------------------------
# _syntax_fail_reason
# ---------------------------------------------------------------------------

def test_reason_extracted_from_tainted_result():
    reason = _syntax_fail_reason("index.html", _TAINT)
    assert reason is not None
    assert "does NOT parse" in reason
    assert "Identifier 'WebOS' has already been declared" in reason
    assert "edits" in reason  # steers to a surgical fix, not a rewrite


def test_clean_success_gives_no_reason():
    assert _syntax_fail_reason("a.py", "SUCCESS: File 'a.py' written.") is None
    assert _syntax_fail_reason("a.py", "") is None
    assert _syntax_fail_reason("a.py", None) is None


# ---------------------------------------------------------------------------
# _apply_file / _apply_edits surface the failure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_write_with_syntax_fail_is_apply_failure():
    runner = SeqRunner([_TAINT])
    path, reason = await _apply_file(
        runner, {"path": "index.html", "content": "<html></html>"}, {})
    assert path is None
    assert "does NOT parse" in reason


@pytest.mark.asyncio
async def test_append_with_syntax_fail_is_apply_failure():
    # the append live-reads the current file, then the write comes back tainted
    runner = SeqRunner(
        [_TAINT],
        read_out="--- index.html CONTENTS ---\n<html><body></body></html>")
    existing = {"index.html": "<html><body></body></html>"}
    path, reason = await _apply_file(
        runner, {"path": "index.html", "append": "<script>let x=1;</script>"},
        existing)
    assert path is None
    assert "does NOT parse" in reason


@pytest.mark.asyncio
async def test_edits_leaving_broken_file_is_apply_failure():
    runner = SeqRunner(["SUCCESS: Exact match found and replaced. "
                        "⚠ SYNTAX CHECK FAILED: 'index.html' … "
                        "line 495: SyntaxError: Identifier 'WebOS' has "
                        "already been declared"])
    reason = await _apply_edits(
        runner, "index.html", [{"find": "let a=1;", "replace": "let a=2;"}])
    assert reason is not None
    assert "does NOT parse" in reason


@pytest.mark.asyncio
async def test_clean_write_still_succeeds():
    runner = SeqRunner(["SUCCESS: File 'index.html' written."])
    path, reason = await _apply_file(
        runner, {"path": "index.html", "content": "<html></html>"}, {})
    assert path == "index.html"
    assert reason is None


# ---------------------------------------------------------------------------
# build_coding_task: retry-with-feedback, honest failure on exhaust
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_persistent_syntax_fail_exhausts_retries_and_fails():
    # Every write comes back tainted → all attempts consumed → ok=False with
    # the diagnostic in the summary (so _finalize_coding marks FAILED, not DONE).
    llm = FakeLLM(_SPEC_HTML)
    runner = SeqRunner([_TAINT] * MAX_ATTEMPTS)
    res = await build_coding_task(_ctx(llm), "build the shell",
                                  tool_runner=runner)
    assert not res.ok
    assert "does NOT parse" in res.summary
    assert llm.calls == MAX_ATTEMPTS
    # The retry prompts carried the diagnostic as feedback.
    assert "Identifier 'WebOS' has already been declared" in llm.user_prompt(1)


@pytest.mark.asyncio
async def test_syntax_fail_then_fix_succeeds_on_retry():
    llm = FakeLLM(_SPEC_HTML)
    runner = SeqRunner([_TAINT, "SUCCESS: File 'index.html' written."])
    res = await build_coding_task(_ctx(llm), "build the shell",
                                  tool_runner=runner)
    assert res.ok
    assert res.files == ["index.html"]
    assert llm.calls == 2
