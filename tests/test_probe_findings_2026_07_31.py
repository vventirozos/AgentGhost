"""Structural pins for the 2026-07-31 probe-log fixes that live INLINE in
the turn/finalize loops, where a full-harness behavioural test would cost
more than the invariant it protects. Each assert names the incident that
motivated it; a refactor that drops the condition regresses that incident.

(The behavioural halves of this fix cluster are covered elsewhere:
test_browser_target_closed_retry.py, test_dispatch_pipeline_extraction.py
(producer→consumer batch ordering), test_verifier_visual.py (evidence
provenance), test_vision_verify_ui.py.)
"""

import inspect

import ghost_agent.core.agent as agent_mod


def _source():
    return inspect.getsource(agent_mod)


def test_image_markdown_guard_whitelists_browser():
    """Probe req d02db9d6: the guard fired on a legitimate browser
    screenshot DOWNLOAD link because 'browser' was missing from the
    valid-image-tool whitelist, and the agent burned a turn arguing with
    the SYSTEM ALERT. A browser screenshot result IS an image source."""
    src = _source()
    assert '["image_generation", "execute", "file_system", "browser"]' in src


def test_turn_outcome_failed_requires_terminal_failure():
    """Probe req d02db9d6: the operator-facing Turn Outcome line said
    'failed · 0.86' off ONE recovered mid-turn strike while the answer
    was correct and both late verdicts CONFIRMED 100%. The failed state
    must require the strike ledger AND a terminal last_was_failure — the
    same rule the trajectory corpus uses — with recovered strikes
    surfaced via the suffix note instead.

    Extended 2026-07-31 (honest-failure rule): the line's priority must
    also mirror resolve_turn_outcome — a verifier PASS outranks a
    terminal execution failure, and the suffix must not claim a recovery
    that did not happen. The priority ladder itself now lives in the
    SHARED `_turn_outcome_label` helper (also used by the late-verdict
    correction) so the printed line and its correction cannot drift."""
    src = _source()
    assert "recovered {execution_failure_count} strike(s)" in src
    # Terminality is still coupled to the ledger… (whitespace-insensitive
    # since 2026-08-04: the assignment moved earlier in finalize so the
    # shape rule can read it, and pinning its exact continuation indent
    # made a reformat look like a behaviour change).
    import re as _re
    assert _re.search(
        r"_exec_terminal\s*=\s*\(execution_failure_count\s*>\s*0\s*"
        r"and bool\(last_was_failure\)\)", src), (
            "the Turn Outcome line's terminality must stay coupled to the "
            "strike ledger AND the last call's failure")
    # …the finalize line resolves its state through the shared helper…
    assert "_state = self._turn_outcome_label(" in src
    # …and the truthful suffix survives.
    assert "tool failure(s), honestly reported" in src


def test_batch_order_producers_before_vision():
    """Probe req d02db9d6: a repaired merged tool_call batch ran browser
    and vision_analysis concurrently; vision probed the screenshot before
    it existed → spurious FATAL strike. The dispatch batch must hold the
    producer set and sequence vision behind it."""
    src = _source()
    assert '_FILE_PRODUCERS = {"browser", "image_generation"}' in src
