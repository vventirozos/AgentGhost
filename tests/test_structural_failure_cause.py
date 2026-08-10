"""`structural failure` must record WHAT broke — without breaking the
load-bearing exact match it used to rely on.

WHY (measured 2026-08-10). Of 160 recorded FAILED trajectories, **42 (26%)
carried the bare string `structural failure`** — which says only THAT
execution broke, never WHAT. So "was this a hard task, or a flaky tool?"
was unanswerable from the corpus. That matters because these labels train
the complexity router (§4AA), where a failure caused by infrastructure
teaches nothing about a request's difficulty.

⚠ THE TRAP THIS NAVIGATES. `STRUCTURAL_FAILURE_REASON` is not cosmetic:
`resolve_turn_outcome` matched it EXACTLY to decide whether a late verifier
PASS may upgrade that FAILED (the 2026-07-31 honest-failure rule). Naively
appending a cause would have silently stopped that match and disabled the
rule on the async-verdict path — precisely the writer/reader drift the
constant's own comment warns about. So the constant became a PREFIX and
every reader goes through `is_structural_reason()`.

⚠ AND THE ORIGINAL PREMISE WAS WRONG, recorded so nobody re-derives it: this
started as "exclude infrastructure failures from the router corpus". Measured:
**zero** of the 160 failures were infrastructure (no timeout, connection or
node errors). 99 were `verifier refuted` — the agent answered wrongly, which
is a legitimate difficulty signal. The real gap was that `structural failure`
could not be classified at all, which is what this fixes.
"""

import pytest

from ghost_agent.distill.outcome_heuristics import (
    STRUCTURAL_FAILURE_REASON,
    is_structural_reason,
    resolve_turn_outcome,
    structural_cause_for_trajectory,
    structural_reason,
)

F, P, U = "failed", "passed", "unknown"


class _TC:
    def __init__(self, name="", error="", result=""):
        self.name, self.error, self.result = name, error, result


class _Traj:
    def __init__(self, tool_calls=None):
        self.tool_calls = tool_calls or []


# ── the prefix contract ─────────────────────────────────────────────────────

def test_a_bare_reason_is_unchanged():
    assert structural_reason() == STRUCTURAL_FAILURE_REASON
    assert structural_reason("") == STRUCTURAL_FAILURE_REASON


def test_a_cause_is_appended_as_a_suffix():
    r = structural_reason("execute: connection refused")
    assert r.startswith(STRUCTURAL_FAILURE_REASON + ":")
    assert "connection refused" in r


@pytest.mark.parametrize("reason", [
    STRUCTURAL_FAILURE_REASON,
    structural_reason("execute: boom"),
    "  structural failure  ",
])
def test_every_structural_form_is_recognised(reason):
    assert is_structural_reason(reason)


@pytest.mark.parametrize("reason", [
    "", "verifier refuted", "runtime abort marker [ATTEMPT_ABORTED_NO_PROGRESS]",
    "structural failures elsewhere",   # must not match on a loose prefix
])
def test_other_reasons_are_not_mistaken_for_structural(reason):
    assert not is_structural_reason(reason)


def test_the_reason_is_length_capped():
    assert len(structural_reason("x" * 999)) < 200


# ── THE LOAD-BEARING BEHAVIOUR ──────────────────────────────────────────────

def test_a_CAUSED_reason_still_earns_the_late_verifier_upgrade():
    """⚠ THE WHOLE RISK OF THIS CHANGE. `resolve_turn_outcome` used an exact
    match, so a qualified reason would have silently lost the late-PASS
    upgrade — disabling the 2026-07-31 honest-failure rule on the async
    path, with nothing failing loudly."""
    assert resolve_turn_outcome(
        current=F, current_reason=structural_reason("execute: connection refused"),
        verifier="passed") == P


def test_the_bare_reason_still_earns_it_too():
    """Legacy trajectories on disk carry the unqualified string."""
    assert resolve_turn_outcome(
        current=F, current_reason=STRUCTURAL_FAILURE_REASON,
        verifier="passed") == P


def test_a_non_structural_failure_is_still_never_upgraded():
    assert resolve_turn_outcome(
        current=F, current_reason="verifier refuted", verifier="passed") == F


def test_the_unacked_total_failure_carve_out_survives():
    """Rule 2b must still withhold the PASS, caused reason or not."""
    assert resolve_turn_outcome(
        current=F, current_reason=structural_reason("execute: boom"),
        verifier="passed", unacked_total_failure=True) == F


# ── deriving the cause ──────────────────────────────────────────────────────

def test_the_cause_names_the_tool_and_its_error():
    c = structural_cause_for_trajectory(
        _Traj([_TC("execute", error="connection refused")]))
    assert "execute" in c and "connection refused" in c


def test_several_broken_tools_are_counted():
    c = structural_cause_for_trajectory(
        _Traj([_TC("execute", error="boom"), _TC("browser", error="nope")]))
    assert "execute" in c and "+1 more" in c


def test_a_healthy_trajectory_yields_no_cause():
    """No identifiable failure ⇒ keep the bare constant. An unqualified
    reason is better than an invented one."""
    assert structural_cause_for_trajectory(_Traj([_TC("execute", result="ok")])) == ""
    assert structural_cause_for_trajectory(_Traj([])) == ""
    assert structural_reason(structural_cause_for_trajectory(_Traj([]))) == \
        STRUCTURAL_FAILURE_REASON


def test_it_uses_the_SHARED_failure_sniffer_not_a_second_copy():
    """A duplicated "did this tool fail?" rule is how the corpus and its
    consumers drift — the lesson this module already records."""
    import inspect

    from ghost_agent.distill import outcome_heuristics as oh
    src = inspect.getsource(oh.structural_cause_for_trajectory)
    assert "tool_call_failed" in src


def test_it_never_raises_on_a_malformed_trajectory():
    """Instrumentation must not be able to break a turn."""
    class Bad:
        @property
        def tool_calls(self):
            raise RuntimeError("boom")
    assert structural_cause_for_trajectory(Bad()) == ""


def test_the_writer_records_the_cause():
    """Structural: the consolidation site must call through the helpers, or
    the corpus keeps getting bare strings."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "src" / "ghost_agent"
           / "core" / "agent.py").read_text()
    assert "structural_reason(\n" in src or "structural_reason(" in src
    assert "structural_cause_for_trajectory(traj)" in src
