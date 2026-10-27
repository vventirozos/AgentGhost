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


# ── §4RN (2026-10-25): the turn clock keeps counting across an app switch ──
# Operator: "in the web ui, while on mobile, the request timer resets when i
# change focus or open another app". The dropped stream's teardown stopped the
# clock; a reattach that rebuilt the bubble restarted it at 0:00 (`tickerStart
# = now`), and a same-page reattach kept the bubble with no clock at all. The
# in-flight handle now carries `startedAt`, an adopted handle restores it, and
# the resume bubble (new or surviving) counts from it. EXECUTED under node —
# these replace the old `"ensureAgentBubbleForResume()" in code` text pin.

from tests.helpers import eval_js, extract_js_function  # noqa: E402

# A fake page: clock elements, a controllable clock, inert timers.
_PRELUDE = r"""
let NOW = 1_000_000_000;
Date.now = () => NOW;
const el = () => ({ textContent: '', classList: { remove() {}, add() {} },
                    insertAdjacentElement() {}, isConnected: true,
                    appendChild() {}, setAttribute() {}, dataset: {} });
const turnStatusEl = el(), turnStatusDescEl = el(), turnStatusClockEl = el(), turnStatusIconEl = el();
let tickerTimer = null, tickerStart = 0, tickerReqId = null, tickerSeen = [], tickerOwnRid = null;
let tickerLastStart = 0, tickerLastReqId = null;
// a bubble made for the turn sent at `start`
const bubble = (start) => { const b = el(); b.dataset.turnStart = String(start); return b; };
let ticks = [];
globalThis.setInterval = (fn) => { ticks.push(fn); return ticks.length; };
globalThis.clearInterval = () => {};
let currentTurnStartedAt = 0, currentAgentMessageDiv = null, currentTaskId = null;
const store = {};
const safeStorage = { get: (k) => store[k] ?? null, set: (k, v) => { store[k] = v; } };
const TAB_ID = 'tab-1';
function _inflightKey() { return 'ghost_inflight_turn:' + TAB_ID; }
function storedSessionId() { return 'sess-1'; }
globalThis.window = {};
let inflightHeartbeat = null;
const chatLog = { appendChild() {} };
const document = { createElement: () => el() };
function dismissEmptyStateHero() {}
function scrollToBottom() {}
"""

_FNS = "\n".join(extract_js_function(_code(), n) for n in (
    "_turnStartFromHandle", "_tickerClockText", "startTurnTicker", "stopTurnTicker",
    "saveInflightHandle", "ensureAgentBubbleForResume"))


def _run(code: str):
    """`stmt; stmt; [result]` — the statements run, the last part is returned."""
    body, _, result = code.rpartition(";")
    return eval_js(_PRELUDE + _FNS, f"(() => {{ {body}; return {result}; }})()")


def test_the_clock_text_counts_minutes_and_seconds():
    assert _run("[_tickerClockText(0, 0), _tickerClockText(0, 59_999), _tickerClockText(0, 61_000),"
                " _tickerClockText(0, 3_725_000), _tickerClockText(5_000, 0)]") == \
        ["0:00", "0:59", "1:01", "62:05", "0:00"]


def test_a_reattached_clock_continues_from_the_send_not_zero():
    got = _run("startTurnTicker(el(), NOW - 95_000);"
               " const first = turnStatusClockEl.textContent;"
               " NOW += 2_000; ticks[ticks.length - 1]();"
               " [first, turnStatusClockEl.textContent, tickerStart]")
    assert got == ["1:35", "1:37", 1_000_000_000 - 95_000]


def test_a_new_turn_and_a_bad_start_time_count_from_now():
    got = _run("startTurnTicker(el()); const a = turnStatusClockEl.textContent;"
               " startTurnTicker(el(), NOW + 60_000); const b = [turnStatusClockEl.textContent, tickerStart === NOW];"
               " startTurnTicker(el(), NaN); [a, b, turnStatusClockEl.textContent]")
    assert got == ["0:00", ["0:00", True], "0:00"]


def test_the_in_flight_handle_carries_the_send_time():
    got = _run("currentTaskId = 't1'; currentTurnStartedAt = NOW - 40_000; NOW += 1; saveInflightHandle();"
               " const h = JSON.parse(store[_inflightKey()]); [h.startedAt, h.ts]")
    assert got == [1_000_000_000 - 40_000, 1_000_000_001]


def test_the_heartbeat_keeps_the_send_time():
    got = _run("currentTaskId = 't1'; currentTurnStartedAt = NOW - 40_000; saveInflightHandle();"
               " NOW += 5_000; ticks[ticks.length - 1]();"
               " const h = JSON.parse(store[_inflightKey()]); [h.startedAt, h.beat]")
    assert got == [1_000_000_000 - 40_000, 1_000_005_000]


def test_a_handle_restores_its_send_time_and_old_handles_fall_back_to_ts():
    got = _run("[_turnStartFromHandle({startedAt: NOW - 30_000, ts: NOW - 1_000}) === NOW - 30_000,"
               " _turnStartFromHandle({ts: NOW - 7_000}) === NOW - 7_000,"
               " _turnStartFromHandle({startedAt: NOW + 9_000}) === NOW + 9_000,"
               " _turnStartFromHandle({startedAt: 'junk'}) === NOW,"
               " _turnStartFromHandle(null) === NOW]")
    assert got == [True, True, True, True, True]


def test_a_rebuilt_resume_bubble_counts_from_the_send():
    """Reload/discard path: no bubble, the handle's send time was restored."""
    got = _run("currentTurnStartedAt = NOW - 125_000; currentAgentMessageDiv = null;"
               " ensureAgentBubbleForResume(); [turnStatusClockEl.textContent, tickerTimer !== null]")
    assert got == ["2:05", True]


def test_a_same_page_resume_gives_the_surviving_bubble_its_clock_back():
    """The dropped stream stopped the clock; the bubble survived."""
    got = _run("currentTurnStartedAt = NOW - 20_000; const b = bubble(currentTurnStartedAt);"
               " currentAgentMessageDiv = b;"
               " startTurnTicker(currentAgentMessageDiv, currentTurnStartedAt); tickerReqId = 'ab12cd34';"
               " stopTurnTicker(); NOW += 30_000; ensureAgentBubbleForResume();"
               " [tickerTimer !== null, turnStatusClockEl.textContent, currentAgentMessageDiv === b,"
               "  tickerReqId, turnStatusDescEl.textContent]")
    # the same bubble, its clock from the send, and its log corridor kept;
    # nothing captioned yet reads "thinking…" (operator 10-27; was "working…")
    assert got == [True, "0:50", True, "ab12cd34", "thinking…"]


def test_a_finished_reply_of_another_turn_is_not_reused_for_a_resume():
    """Adopting another tab's turn: the pointer still names this tab's last,
    FINISHED reply — the resume gets a new bubble, the old one no clock."""
    got = _run("const old = bubble(NOW - 300_000); currentAgentMessageDiv = old;"
               " currentTurnStartedAt = NOW - 20_000; ensureAgentBubbleForResume();"
               " [currentAgentMessageDiv !== old, currentAgentMessageDiv.dataset.turnStart === String(NOW - 20_000),"
               "  turnStatusClockEl.textContent, tickerReqId]")
    assert got == [True, True, "0:20", None]


def test_a_running_clock_is_not_restarted_by_a_second_resume_call():
    got = _run("currentTurnStartedAt = NOW - 20_000; currentAgentMessageDiv = bubble(currentTurnStartedAt);"
               " startTurnTicker(currentAgentMessageDiv, currentTurnStartedAt); const t = tickerTimer;"
               " ensureAgentBubbleForResume(); [tickerTimer === t, ticks.length]")
    assert got == [True, 1]


def test_a_clock_set_back_still_matches_the_bubble_and_counts_from_now():
    """The device clock moved back between send and reattach: the raw stamp
    still finds the turn's bubble (no duplicate), the clock counts from now."""
    got = _run("const start = NOW + 9_000; const b = bubble(start); currentAgentMessageDiv = b;"
               " currentTurnStartedAt = _turnStartFromHandle({startedAt: start}); ensureAgentBubbleForResume();"
               " [currentAgentMessageDiv === b, turnStatusClockEl.textContent, tickerStart === NOW]")
    assert got == [True, "0:00", True]
