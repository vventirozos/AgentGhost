"""Promised-notification finish-line guard (2026-07-13, req 11fe11d8).

The user asked "deep research this … notify me in slack when you're done".
The model planned the notify_operator call in its reasoning, then emitted
the final response without ever calling the tool — the one explicit
delivery requirement was silently dropped, and the verifier confirmed the
turn because the deliverable itself was fine. The turn loop now steers
ONCE toward notify_operator when a turn is about to finalize with an
unfulfilled explicit notify-me ask.

The intent matcher is deliberately narrow: a false fire injects a bogus
SYSTEM ALERT into casual conversation, so questions ABOUT Slack must not
arm it.
"""

import pytest

from ghost_agent.core.agent import _user_asked_for_notification


ASKS = [
    # the live failure, verbatim
    "deep research this and come up with a detailed plan notify me in slack when you're done",
    "Notify me when the job finishes.",
    "ping me when it's done",
    "alert me once the deploy completes",
    "let me know in slack when the tests pass",
    "tell me on slack once it's deployed",
    "report back via slack when you finish",
    "update me in Slack after the run",
    "message me on slack when done",
    "dm me in slack once ready",
    "send me a notification when complete",
    "send me a slack message when the download finishes",
]

NOT_ASKS = [
    # negations
    "don't notify me about this",
    "do not ping me when it's done",
    "no need to notify me, just log it",
    "finish the task without notifying me",
    # questions/chat ABOUT slack — must never arm the guard
    "what is slack?",
    "how do I format a slack message?",
    "we discussed this in slack yesterday",
    "tell me about slack integrations",
    "the slack bot is broken, can you look at it?",
    # unrelated
    "summarize the log file",
    "",
    None,
]


@pytest.mark.parametrize("text", ASKS)
def test_intent_detected(text):
    assert _user_asked_for_notification(text) is True


@pytest.mark.parametrize("text", NOT_ASKS)
def test_intent_not_detected(text):
    assert _user_asked_for_notification(text) is False


def test_incidental_slack_deep_inside_pasted_document_ignored():
    doc = "please review this pasted doc\n" + ("x" * 5000) + "\nnotify me in slack"
    # Beyond the 4000-char truncation window → not an ask.
    assert _user_asked_for_notification(doc) is False


def test_guard_wiring_in_turn_loop():
    # Source pin (suite convention for deep turn-loop branches): the guard
    # must be one-shot, respect force-finalisation, and key on the
    # notify_operator tool not having run this request.
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "src" / "ghost_agent"
           / "core" / "agent.py").read_text()
    assert "notify_steer_fired = False" in src
    assert '"notify_operator" not in raw_tools_called' in src
    assert "_user_asked_for_notification(last_user_content)" in src
    guard = src[src.index("Catch a PROMISED NOTIFICATION"):][:1600]
    assert "not force_final_response" in guard
    assert "not is_final_generation" in guard
    assert "not force_stop" in guard
    assert "notify_steer_fired = True" in guard
    assert "SYSTEM ALERT" in guard
