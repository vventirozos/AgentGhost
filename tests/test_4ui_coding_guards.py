"""§4UI Batch 4 (2026-11-12) — coding guardrails relaxed for the dense model
(operator: "we generally don't want to restrict the model too much while it's
coding"); each case is from the log review of 10-25..11-12."""
from ghost_agent.core import stream_guards as G

_PRE = ("User's request: write the whole conversion as an end-to-end SQL script and save it.\n"
        "Let me think about the steps carefully before writing the file out in full now.\n") * 3
_VERIFY = "SELECT relname, n_live_tup FROM pg_stat_user_tables WHERE relname LIKE 'test_1%';\n"


def _sql_draft(steps=8):
    out = "```sql\n"
    for i in range(1, steps + 1):
        out += ("-- ============================================================\n"
                f"-- Step {i}: one more stage of the online conversion here\n"
                "-- Each partition inherits the primary key index from the parent table automatically\n"
                f"ALTER TABLE test_1_new ADD COLUMN c{i} integer NOT NULL DEFAULT {i};\n" + _VERIFY)
    return out


def test_a_drafted_sql_script_is_not_a_paragraph_loop():
    # req afe8a49d: the requested SQL file, drafted in thinking, was aborted as a loop
    buf = _PRE + _sql_draft(24) + "\n"
    assert len(buf) > G.PARAGRAPH_LOOP_MIN_BUF
    assert not any(G._detect_paragraph_loop(buf[:n]) for n in range(G.PARAGRAPH_LOOP_MIN_BUF, len(buf), 250))


def test_a_comment_line_is_code():
    assert not G._is_text_line("-- Each partition inherits the primary key index from the parent")
    assert not G._is_text_line("# verify the row counts of every single partition now")
    assert G._is_text_line("Each partition inherits the primary key index from the parent")


def test_a_prose_loop_still_fires_after_a_closed_fence():
    loop = "Wait, I need to reconsider whether the swap should run inside one transaction.\n"
    buf = _PRE + _sql_draft(24) + "```\n"
    buf += "".join(f"Thought {i}: the order of statements matters for the lock taken here.\n{loop}"
                   for i in range(8)) + "x"
    assert len(buf) > G.PARAGRAPH_LOOP_MIN_BUF and G._detect_paragraph_loop(buf)


import pytest  # noqa: E402
from ghost_agent.core import agent as A  # noqa: E402


@pytest.mark.parametrize("text", [
    "The file is ready. Say **yes** and I'll delete the old copy.",
    "Three options above. Tell me the number and I'll build it.",
    "Reply go and I'll start the migration.",
    "If you want, I'll add the tests next.",
    "Once you confirm, I will run it against the live database.",
    "Want me to run the full suite now?",
])
def test_an_offer_that_waits_on_the_user_is_not_a_pending_promise(text):
    assert A._ends_with_action_promise(text) == ""


@pytest.mark.parametrize("text", [
    "That's what's causing the error. Let me fix it.",
    "I'll run the tests now.",
    "Found it. I will patch the parser and rerun.",
])
def test_a_real_trailing_promise_still_counts(text):
    assert A._ends_with_action_promise(text)


# ── A1: a failing program run is debugging, not a broken call ───────────────

@pytest.mark.parametrize("name,code,text,rej,want", [
    ("execute", 1, "--- EXECUTION RESULT ---\nEXIT CODE: 1\nSTDOUT/STDERR:\n=== 2 failed ===", False, True),
    ("execute", 2, "--- COMMAND RESULT ---\nEXIT CODE: 2\nSTDOUT/STDERR:\nusage: x", False, True),
    ("execute", 1, "--- EXECUTION RESULT ---\nEXIT CODE: 1\nSTDOUT/STDERR:\nblocked", True, False),   # a refusal
    ("execute", 137, "EXIT CODE: 137 (timed out / killed after 600s)\nSTDOUT/STDERR:\n", False, False),
    ("execute", 124, "EXIT CODE: 124\nSTDOUT/STDERR:\n", False, False),
    ("execute", 0, "EXIT CODE: 0\nSTDOUT/STDERR:\nok", False, False),
    ("file_system", 1, "EXIT CODE: 1\nSTDOUT/STDERR:\nx", False, False),
    ("execute", 1, "Error: sandbox unavailable", False, False),
])
def test_a_programs_own_failure_is_told_apart(name, code, text, rej, want):
    assert A._program_failed_on_its_own(name, code, text, rej) is want


def test_only_a_changed_output_is_free():
    from types import SimpleNamespace
    st = SimpleNamespace()
    assert A._program_fail_repeats(st, "=== 3 failed, 9 passed in 0.12s ===") is False
    assert A._program_fail_repeats(st, "=== 2 failed, 10 passed in 0.31s ===") is False
    assert A._program_fail_repeats(st, "=== 2 failed, 10 passed in 0.44s ===") is True    # timing only


def test_a_flip_flop_between_two_failures_is_a_repeat():
    """r1 MAJOR 3: only the previous output was compared — A, B, A, B was
    always "changed"."""
    from types import SimpleNamespace
    st = SimpleNamespace()
    assert A._program_fail_repeats(st, "NameError: name 'x' is not defined") is False
    assert A._program_fail_repeats(st, "TypeError: bad operand") is False
    assert A._program_fail_repeats(st, "NameError: name 'x' is not defined") is True


def test_an_endlessly_varying_failure_counts_again_after_the_free_budget():
    from types import SimpleNamespace
    st = SimpleNamespace()
    free = [A._program_fail_repeats(st, f"AssertionError: got seed {i * 7919} want 0")
            for i in range(A._PROGRAM_FAIL_FREE_MAX + 3)]
    assert free[:A._PROGRAM_FAIL_FREE_MAX] == [False] * A._PROGRAM_FAIL_FREE_MAX
    assert all(free[A._PROGRAM_FAIL_FREE_MAX:])


@pytest.mark.parametrize("text", [
    "If you confirm — I'll delete the old branch.",
    "If you'd like — I'll add tests too.",
    "Once you've checked it — I'll merge.",
    "When you're ready; I'll deploy it.",
])
def test_a_condition_before_a_dash_is_still_an_offer(text):
    """review r2 M1: a per-clause split cut the condition off its "I'll"."""
    assert A._ends_with_action_promise(text) == ""
