"""Unit tests for scripts/claude_trainer.py correction construction.

The trainer's whole value rests on one invariant: the correction message
it sends must actually trigger the agent's ``classify_user_correction``
learning predicate (BOTH the anchored-phrase signal AND the rephrase
signal). These tests pin that invariant against the REAL classifier so a
future tweak to either side can't silently break the learning loop.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ghost_agent.distill.user_correction import classify_user_correction


def _load_trainer():
    """Import the trainer module by path (scripts/ is not a package)."""
    path = _REPO / "scripts" / "claude_trainer.py"
    spec = importlib.util.spec_from_file_location("claude_trainer", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    # Register before exec so @dataclass can resolve the module's __dict__
    # when it inspects string annotations.
    sys.modules["claude_trainer"] = mod
    spec.loader.exec_module(mod)
    return mod


trainer = _load_trainer()


# --- The core invariant: built correction fires the real classifier ----

@pytest.mark.parametrize(
    "question,teaching",
    [
        (
            "What is the default isolation level in PostgreSQL?",
            "READ COMMITTED is the default isolation level.",
        ),
        (
            "How many bits are in an IPv6 address?",
            "An IPv6 address is 128 bits long.",
        ),
        (
            "Which Python keyword defines an asynchronous function?",
            "The 'async def' syntax defines a coroutine function.",
        ),
    ],
)
def test_built_correction_triggers_real_classifier(question, teaching):
    wrong_answer = "Some incorrect answer the agent produced."
    msg = trainer.build_correction_message(question, teaching)
    verdict = classify_user_correction(
        prev_user_request=question,
        prev_assistant_response=wrong_answer,
        current_user_text=msg,
    )
    assert verdict.is_correction, (
        f"correction did not fire learning. signals={verdict.signals}\nmsg={msg}"
    )
    # Both signals must be present — that's the whole point.
    assert "phrase" in verdict.signals
    assert any(s.startswith("rephrase") for s in verdict.signals)


def test_correction_will_trigger_agrees_with_real_classifier():
    q = "What port does HTTPS use by default?"
    msg = trainer.build_correction_message(q, "HTTPS uses TCP port 443 by default.")
    assert trainer.correction_will_trigger(q, "wrong", msg) is True


def test_plain_rebuttal_without_restatement_does_not_trigger():
    # A naive correction that omits the question restatement must NOT be
    # reported as triggering — this is exactly the failure mode the
    # trainer guards against.
    q = "What is the capital of Australia?"
    naive = "No, that's wrong."  # phrase only, no rephrase overlap
    assert trainer.correction_will_trigger(q, "Sydney", naive) is False


def test_opener_is_anchored_at_start():
    msg = trainer.build_correction_message("What is 2+2?", "2+2 equals 4.")
    assert msg.startswith("No, that's not right.")


def test_empty_inputs_do_not_crash():
    msg = trainer.build_correction_message("", "")
    assert isinstance(msg, str)
    # With no question tokens, the rephrase signal can't fire, so it must
    # not be reported as a learning trigger.
    assert trainer.correction_will_trigger("", "", msg) is False
