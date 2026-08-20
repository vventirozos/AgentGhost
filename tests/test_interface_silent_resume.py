"""Silent Safari suspend/resume in the web UI (2026-08-16, operator request).

On iOS, Safari suspends the tab whenever the phone locks or the user
switches apps, killing the in-flight fetch. The client's recovery path
(retry via ``sendMessage(true)`` on visibility-return, resume bubble via
``ensureAgentBubbleForResume``) used to announce itself with two system
bubbles — "Safari suspended the UI. Detached Ghost continues
calculation..." and "Reconnected directly to Ghost Server." — which
spammed the chat on every lock/unlock cycle. The operator asked for them
to be gone entirely: the reply stream continuing IS the signal.

These tests pin that the messages stayed removed while the recovery
machinery they decorated is still present (removing the retry logic
itself would also make the strings vanish — pin both sides).
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_APP_JS = _ROOT / "interface" / "static" / "app.js"


def _src() -> str:
    """RAW source — for asserting that an explanation still exists."""
    return _APP_JS.read_text(encoding="utf-8")


def _code() -> str:
    """CODE ONLY. ⚠ Every assertion in this file used to run against the raw
    text, and `sendMessage(true)` appears in COMMENTS at two places in
    app.js — so twelve mutations survived, including deleting the whole
    visibility-retry implementation (review R1 M4). Line comments are
    stripped BEFORE block comments: a `//`-comment containing `/api/*` would
    otherwise open a phantom block comment that swallows real code (the same
    trap the web-UI suites document).
    """
    # ⚠ Shared implementation. This file carried a PRIVATE 4th copy of the
    # regex pair, so `TestTheStripperItself` (which pins the known-wrong
    # versions) did not guard the code this suite actually ran (R2 lens B).
    from tests.helpers import strip_js_comments
    return strip_js_comments(_APP_JS.read_text(encoding="utf-8"))


class TestNoSuspendResumeBubbles:
    def test_no_safari_suspended_message(self):
        # Case-insensitive: the bubble must not come back under a reworded
        # addMessage either.
        assert "safari suspended" not in _src().lower()

    def test_no_reconnected_message(self):
        assert "reconnected directly" not in _src().lower()


class TestRecoveryMachineryStillWired:
    """The silence must come from dropping the bubbles, not the recovery."""

    def test_visibility_retry_still_present(self):
        code = _code()
        # Suspend recovery: retry the send when the tab becomes visible.
        assert "visibilitychange" in code
        # The RESUME CALL, in code — not the token, which also appears in two
        # comments. Deleting the implementation used to leave this green.
        assert "sendMessage(true)" in code

    def test_the_resume_is_LATCHED_and_BOUNDED(self):
        """The network-error resume path is a FOURTH trigger that used to
        bypass `resumeLatch`: after the latched path nulled currentTaskId it
        fell into the POST branch and sent a whole phantom turn, and with no
        attempt cap a down proxy retried forever behind a permanent thinking
        bubble (review R1 M6)."""
        code = _code()
        assert "_netResumeTries" in code, "the attempt cap is gone"
        assert "resumeLatch" in code, "the synchronous re-entry guard is gone"
        # The guarded helper must check BOTH conditions before re-sending.
        i = code.index("_resumeAttempt")
        window = code[i:i + 900]
        assert "currentTaskId" in window, "resumes without a task to resume"
        assert "isProcessingRequest" in window, "resumes on top of a live turn"

    def test_resume_bubble_still_present(self):
        assert "ensureAgentBubbleForResume()" in _code()

    def test_real_errors_still_surface(self):
        code = _code()
        # Only the suspend/reconnect chatter is silenced — genuine failures
        # and user cancels must still produce system messages.
        assert "Network Error" in code
        assert "Stopped by user" in code

    def test_stop_cancels_the_AGENT_turn_not_just_the_proxy_relay(self):
        """`/api/chat/cancel/<task>` tears down the interface's buffered
        relay ONLY: the agent ran the turn to completion, held the global
        turn lock, and its reply landed in the durable session while this
        tab said "Request cancelled by user." (review R1 M8).

        The wording also had to change: a message claiming the request was
        cancelled, emitted on a path that cancelled nothing agent-side, is
        the defect — so pin the honest wording AND the real call."""
        code = _code()
        assert "/api/turn/cancel" in code, (
            "Stop never reaches the agent's own cancel — the proxy-side "
            "teardown leaves the turn running")
        assert "_cancelAgentTurn" in code
        # The abort handler must actually invoke it, next to the proxy call.
        i = code.index("/api/chat/cancel/")
        assert "_cancelAgentTurn(" in code[i:i + 900], (
            "the real cancel is defined but not called from the abort path")
        # A cancel that did NOT take must be reported: silence there is the
        # same lie in a new place.
        j = code.index("function _cancelAgentTurn")
        body = code[j:j + 1600]
        assert "404" in body, "a benign already-finished turn is not special-cased"
        assert "addMessage" in body, "a failed cancel is swallowed"
        assert "Request cancelled by user" not in code, (
            "the dishonest wording is back")

    def test_structured_stream_errors_render_their_MESSAGE(self):
        """`Error: ${data.error}` rendered every structured error frame as
        "Error: [object Object]" — so "the agent is down" (UpstreamError),
        an evicted task, a blown buffer and an agent InternalError all read
        identically, and the error_id that correlates to the log was
        destroyed (review R1 M3)."""
        code = _code()
        assert "[object Object]" not in code
        assert "_e.message" in code, "the structured message is not extracted"
        assert "error_id" in code, "the log-correlation id is not surfaced"

    def test_an_error_frame_does_not_persist_a_phantom_reply(self):
        """After an error frame the accumulator is empty, so the finalize
        branch pushed {role:'assistant', content:'No response'} into
        chatHistory — and from there into the DURABLE session store, where it
        becomes context for later turns."""
        code = _code()
        assert "streamHadError" in code
        i = code.index("No response")
        window = code[max(0, i - 400):i]
        assert "streamHadError" in window, (
            "the phantom-reply branch is not gated on the error flag")

    def test_a_module_load_failure_is_VISIBLE(self):
        """workspace.js owns the sessions rail and `__ghostSessionId`; when
        it fails to load every turn silently reverts to client-carried
        history with no durable session and no on-screen signal — a stale
        `?v=` is enough to cause it (review R1 M7)."""
        code = _code()
        i = code.index("workspace.js?v=")
        window = code[i:i + 1200]
        assert "addMessage(" in window, (
            "a console-module load failure is console.warn-only")
        # Token, not a phrase: the message is split across string
        # concatenation, so no multi-word substring is contiguous.
        assert "durable" in window.lower()
        assert "sessions rail" in window.lower()
