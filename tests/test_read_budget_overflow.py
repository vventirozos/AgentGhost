"""Per-batch read budget — guards against context-window overflow.

Regression for an observed production crash: asked to synthesise three
experiment JSON files into a report, the agent issued parallel whole-file
reads. Each file individually cleared the per-file read cap, but two of them
(178 KB + 168 KB) together injected ~99 K tokens of raw JSON into a single
turn, producing a 136 K-token request against a 131 K-token window. The
upstream returned HTTP 400 (exceed_context_size), emergency pruning fired,
and the prune-retry then crashed on an empty upstream body — a recoverable
overflow turned into a hard failure.

The fix caps cumulative raw-read bytes PER BATCH (one assistant message's
tool calls): the first read proceeds (already bounded by the per-file cap),
but a later read that would breach the remaining allowance is refused with a
message steering the model to read_chunked / search / execute.

Covers:
  - read_byte_budget formula (lowered factor + floor)
  - ReadBudget accounting (charge / remaining / clamping)
  - first read in a batch always proceeds and charges the budget
  - a second read that would overflow is refused (steering message)
  - per-file cap still blocks an oversized single file
  - read_budget=None preserves the original behaviour (backward compat)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.tools.file_system import (
    ReadBudget,
    read_byte_budget,
    tool_read_file,
)


# --------------------------------------------------------- budget primitives

def test_read_byte_budget_lowered_factor_and_floor():
    # 131 K window → 0.40 factor → ~183 KB, well under the old 0.5 (~229 KB)
    # that let a single read eat ~70% of the window.
    big = read_byte_budget(131072)
    assert big == int(131072 * 3.5 * 0.40)
    assert big < int(131072 * 3.5 * 0.5)          # strictly tighter than before
    # Floor protects small-context configs.
    assert read_byte_budget(1000) == 150000


def test_readbudget_accounting():
    b = ReadBudget(100)
    assert b.remaining == 100
    b.charge(30)
    assert b.spent == 30 and b.remaining == 70
    b.charge(1000)                                 # overspend clamps remaining
    assert b.remaining == 0
    # Negative / junk charges never decrease spent.
    b2 = ReadBudget(50)
    b2.charge(-10)
    assert b2.spent == 0 and b2.remaining == 50


def test_readbudget_negative_limit_clamped():
    assert ReadBudget(-5).limit == 0


# ------------------------------------------------------------- read behaviour

@pytest.fixture
def sandbox(tmp_path):
    return tmp_path


async def test_first_read_proceeds_and_charges(sandbox):
    f = sandbox / "a.txt"
    body = "x" * 5000
    f.write_text(body)
    budget = ReadBudget(read_byte_budget(131072))   # generous
    out = await tool_read_file("a.txt", sandbox, max_context=131072, read_budget=budget)
    assert "CONTENTS" in out and body in out
    # The budget was charged by the content length actually injected.
    assert budget.spent == len(body)


async def test_second_read_refused_when_it_would_overflow(sandbox):
    big1 = sandbox / "exp1.json"
    big2 = sandbox / "exp3.json"
    big1.write_text("1" * 170_000)
    big2.write_text("3" * 170_000)
    # Mirror the live failure: budget ~183 KB; one 170 KB read fits, the
    # second does not.
    budget = ReadBudget(read_byte_budget(131072))

    first = await tool_read_file("exp1.json", sandbox, max_context=131072, read_budget=budget)
    assert "CONTENTS" in first                      # first read succeeds
    assert budget.spent == 170_000

    second = await tool_read_file("exp3.json", sandbox, max_context=131072, read_budget=budget)
    assert second.startswith("Error:")              # refused before injecting
    assert "overflow" in second.lower()
    # Steers to the cheaper paths.
    assert "read_chunked" in second and "execute" in second
    # A refused read must NOT charge the budget.
    assert budget.spent == 170_000


async def test_per_file_cap_still_blocks_oversized_single_file(sandbox):
    huge = sandbox / "huge.json"
    huge.write_text("z" * (read_byte_budget(131072) + 1))
    budget = ReadBudget(read_byte_budget(131072))
    out = await tool_read_file("huge.json", sandbox, max_context=131072, read_budget=budget)
    assert out.startswith("Error:") and "too large" in out
    assert budget.spent == 0                        # nothing injected


async def test_no_budget_preserves_legacy_behaviour(sandbox):
    # Two large reads with NO budget object behave exactly as before: both
    # succeed as long as each clears the per-file cap. (Backward compat for
    # callers / tests that don't pass a budget.)
    a = sandbox / "a.json"; a.write_text("a" * 170_000)
    b = sandbox / "b.json"; b.write_text("b" * 170_000)
    out_a = await tool_read_file("a.json", sandbox, max_context=131072)
    out_b = await tool_read_file("b.json", sandbox, max_context=131072)
    assert "CONTENTS" in out_a and "CONTENTS" in out_b
