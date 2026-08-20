"""The web UI's feedback client — 25 of 50 live human labels come from here.

⚠ WHY THIS FILE EXISTS. A fresh-eye review of the feedback channel
(2026-08-17) found this path had ZERO automated coverage while carrying the
retry budget, the stale-reqId clear, the double-tap latch and the
re-find-live-node logic — all documented in
docs/interfaces/web_server.html and asserted nowhere. Two mutations proved
it: deleting the retry entirely, and sending `source:'api'` instead of
`'web'` (which would make every web label indistinguishable from a script's
in the corpus), both survived the whole suite.

Text/structure assertions, like the other JS suites here — there is no JS
runtime in this environment. The value is regression detection on the
specific properties the review found unpinned, not simulated execution.
"""

import re
from pathlib import Path

import pytest

from tests.helpers import strip_js_comments

_STATIC = Path(__file__).resolve().parent.parent / "interface" / "static"
_APP = _STATIC / "app.js"
_HTML = _STATIC / "index.html"


# One shared implementation (tests/helpers.strip_js_comments): three
# near-copies of this fragile regex pair existed, and only one of them
# carried the pins in TestTheStripperItself. Alias, so those pins guard
# the code every JS suite actually uses.
_strip_comments = strip_js_comments


@pytest.fixture(scope="module")
def app_js():
    return _strip_comments(_APP.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def app_js_raw():
    """Raw text — only for asserting that an explanation still EXISTS."""
    return _APP.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def send_feedback(app_js):
    """The sendFeedback function body, comments stripped."""
    i = app_js.index("async function sendFeedback")
    j = app_js.index("\nfunction ", i + 10)
    k = app_js.index("\nasync function ", i + 10)
    return app_js[i:min(x for x in (j, k) if x > i)]


class TestTheLabelIsAttributedToTheWebSURFACE:
    def test_the_source_is_web(self, send_feedback):
        """`source` becomes `human_feedback:<source>` in the corpus and is
        now the §4BR provenance key — 'api' would merge web labels with
        script writes and make the per-surface count unreadable."""
        assert "source: 'web'" in send_feedback or \
               'source: "web"' in send_feedback, send_feedback[:400]

    def test_it_posts_the_request_id_and_signal(self, send_feedback):
        assert "request_id" in send_feedback
        assert "signal" in send_feedback


class TestTheRetryBudget:
    """The agent records the trajectory AFTER the stream ends, so a fast tap
    404s; and a 503 is the agent RESTARTING — the window a deploy creates.
    Slack retries both; this client retried only the 404 (review C3)."""

    def test_a_404_is_retried(self, send_feedback):
        assert re.search(r"r\.status\s*===\s*404\b", send_feedback), (
            "the fast-tap 404 retry is gone")
        assert "fbRetried" in send_feedback

    def test_server_errors_and_rate_limits_are_retried_too(self, send_feedback):
        assert re.search(r"r\.status\s*>=\s*500\b", send_feedback), (
            "a 503 during a restart silently loses the label")
        assert re.search(r"r\.status\s*===\s*429\b", send_feedback), (
            "a rate-limited label is dropped")

    def test_the_retry_is_bounded_to_ONE_attempt(self, send_feedback):
        """An unbounded retry on a persistent 500 would hammer the agent."""
        assert "dataset.fbRetried" in send_feedback
        assert re.search(r"!div\.dataset\.fbRetried", send_feedback)

    def test_a_failure_still_tells_the_user(self, send_feedback):
        assert "Feedback not recorded" in send_feedback


class TestTheDoubleTapAndStaleIdGuards:
    def test_a_reused_bubble_cannot_label_the_previous_turn(self, app_js):
        """The stale-reqId clear is the only thing preventing a re-rendered
        bubble from attributing a label to an older turn.

        Asserted as the CLEAR, not the mere presence of `dataset.reqId`
        (which occurs on six unrelated lines, so the first version of this
        assertion could not fail)."""
        assert re.search(r"delete\s+\w+\.dataset\.feedback", app_js), (
            "nothing clears the feedback latch when the bubble is reused")
        assert re.search(r"dataset\.reqId\s*!==?\s*", app_js), (
            "no stale-reqId comparison remains")

    def test_the_verdict_latches_on_the_LIVE_node(self, send_feedback):
        """A drift re-render can detach the node between the tap and the
        response; latching the detached one showed no verdict and invited a
        duplicate tap."""
        assert re.search(r"!\s*\w+\.isConnected", send_feedback), (
            "the detached-node check is gone")
        assert re.search(r"CSS\.escape\s*\(", send_feedback), (
            "the relatch selector no longer escapes the reqId")


class TestCacheBustMovedWithTheChange:
    def test_app_js_version_was_bumped(self, app_js):
        html = _HTML.read_text(encoding="utf-8")
        m = re.search(r"app\.js\?v=([\d.]+)", html)
        assert m and float(m.group(1)) >= 9.6, (
            "a browser holding the old module keeps the un-retried client")


class TestTheStripperItself:
    """R5 F4/R4 m8: this helper had THREE wrong versions — two regexes and
    an ordering — and its failures are silent for negative assertions."""

    def test_line_comments_are_stripped_BEFORE_block_comments(self):
        """A `/api/*` inside a line comment opens a phantom block comment
        that swallows real code until the next `*/`. Measured on app.js:
        5,655 chars and 206 lines deleted, incl. `installAuthFetch` and
        `X-Ghost-Key`."""
        probe = ("// touches /api/* here\n"
                 "const KEEP_ME = 1;\n"
                 "/* a real block */\n"
                 "const ALSO_KEEP = 2;\n")
        out = _strip_comments(probe)
        assert "KEEP_ME" in out and "ALSO_KEEP" in out, out
        assert "a real block" not in out
        assert "touches" not in out

    def test_real_app_js_survives_the_stripper(self):
        """The regression test with teeth: tokens that exist in the shipped
        client must survive. The wrong ordering deleted all three."""
        raw = _APP.read_text(encoding="utf-8")
        out = _strip_comments(raw)
        for tok in ("installAuthFetch", "X-Ghost-Key", "dataset.pdfHandled",
                    "sendFeedback"):
            assert tok in out, f"{tok} was eaten by the comment stripper"
        # A generous floor: the wrong order lost ~5.6k chars of code.
        assert len(out) > len(raw) * 0.6

    def test_protocol_slashes_are_not_comments(self):
        probe = ("const a = `${p}//${host}/ws`;\n"
                 "const b = 'https://example.com/x';\n"
                 "foo();  // trailing gone\n")
        out = _strip_comments(probe)
        assert "//${host}/ws" in out
        assert "https://example.com/x" in out
        assert "trailing gone" not in out
