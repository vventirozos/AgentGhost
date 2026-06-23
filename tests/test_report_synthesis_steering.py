"""Report-path prompt steering — synthesise large data files via `execute`.

Companion to the per-batch read budget (tests/test_read_budget_overflow.py).
The budget STOPS the overflow by refusing later whole-file reads, but the
model only avoids the dead-end if it reaches for a digest script BEFORE
hitting the limit. SYSTEM_PROMPT now carries an explicit rule for exactly
that: when writing a report/summary that synthesises result files, load them
in an `execute` script, print a compact digest, and write from the digest —
not from whole-file reads.
"""

from src.ghost_agent.core.prompts import SYSTEM_PROMPT


def test_synthesis_steering_present():
    assert "SYNTHESIS OVER LARGE DATA FILES" in SYSTEM_PROMPT


def test_synthesis_steering_routes_to_execute_not_whole_reads():
    # Find the bullet and assert it steers to a digest script and away from
    # whole-file reads.
    line = next(l for l in SYSTEM_PROMPT.splitlines()
                if "SYNTHESIS OVER LARGE DATA FILES" in l)
    low = line.lower()
    # Steers TO a script + the cheaper scan ops.
    assert "execute" in low
    assert "digest" in low
    assert "read_chunked" in low and "inspect" in low
    # Steers AWAY from pulling whole files into context.
    assert 'operation="read"' in line
    assert "do not" in low or "don't" in low
    # Names the failure it prevents so the model connects the rule to the
    # read-budget refusal it would otherwise hit.
    assert "overflow" in low


def test_synthesis_steering_is_proactive():
    # The key nuance: act BEFORE the limit, not after the refusal.
    line = next(l for l in SYSTEM_PROMPT.splitlines()
                if "SYNTHESIS OVER LARGE DATA FILES" in l)
    assert "before you hit" in line.lower()
