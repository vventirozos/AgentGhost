"""§4HP — a duplicate call inside ONE batch is not a no-progress loop.

Live (req 1234e131, 2026-09-16): turn 9 issued four parallel browser calls
with gncrypto.news listed twice. Both succeeded with identical text; the
no-progress breaker (threshold 2) read them as "repeated 2x with no new
info" and forced a text-only conclusion — barring task_6 (write report.md
+ PDF). The model tried the write anyway, the call was scrubbed, and §4HF's
retry produced the report inline. A loop is re-observing AFTER seeing the
result; two calls in one batch never saw each other. 5 of the 63 historical
2x trips were this shape.
"""
from __future__ import annotations

import ast
import inspect

from ghost_agent.core import agent as A
from ghost_agent.core.strikes import StrikeLedger

FP = "fp-identical"


def test_a_doubled_call_in_one_batch_counts_once():
    st = StrikeLedger()
    st.begin_batch()
    sig1, c1, t1 = st.note_action("browser", "https://x/a", FP, threshold=2)
    sig2, c2, t2 = st.note_action("browser", "https://x/a", FP, threshold=2)
    assert (c1, t1) == (1, False)
    assert (c2, t2) == (1, False)          # reported, not advanced, not tripped
    assert sig1 == sig2


def test_the_same_call_in_the_next_batch_still_trips():
    st = StrikeLedger()
    st.begin_batch()
    st.note_action("browser", "https://x/a", FP, threshold=2)
    st.note_action("browser", "https://x/a", FP, threshold=2)
    st.begin_batch()
    _, c, tripped = st.note_action("browser", "https://x/a", FP, threshold=2)
    assert (c, tripped) == (2, True)       # a real re-observation across turns


def test_a_different_target_in_the_same_batch_is_its_own_observation():
    st = StrikeLedger()
    st.begin_batch()
    st.note_action("browser", "https://x/a", FP, threshold=2)
    _, c, _ = st.note_action("browser", "https://x/b", FP, threshold=2)
    assert c == 1
    st.begin_batch()
    _, c, tripped = st.note_action("browser", "https://x/b", FP, threshold=2)
    assert (c, tripped) == (2, True)


def test_a_different_result_in_the_same_batch_is_its_own_observation():
    st = StrikeLedger()
    st.begin_batch()
    st.note_action("browser", "https://x/a", FP, threshold=2)
    _, c, _ = st.note_action("browser", "https://x/a", "fp-other", threshold=2)
    assert c == 1


def test_legacy_callers_without_begin_batch_keep_the_old_counting():
    st = StrikeLedger()
    st.note_action("browser", "https://x/a", FP, threshold=2)
    _, c, tripped = st.note_action("browser", "https://x/a", FP, threshold=2)
    assert (c, tripped) == (2, True)


def test_world_changed_still_forgets_everything():
    st = StrikeLedger()
    st.begin_batch()
    st.note_action("browser", "https://x/a", FP, threshold=2)
    st.note_world_changed()
    st.begin_batch()
    _, c, tripped = st.note_action("browser", "https://x/a", FP, threshold=2)
    assert (c, tripped) == (1, False)


def test_the_turn_loop_opens_a_batch_before_noting_actions():
    """AST pin: `strikes.begin_batch()` is called in the same function as
    `strikes.note_action(...)`, before it."""
    tree = ast.parse(inspect.getsource(A))
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute)
                 and isinstance(n.func.value, ast.Name)
                 and n.func.value.id == "strikes"
                 and n.func.attr in ("begin_batch", "note_action")]
        notes = [c for c in calls if c.func.attr == "note_action"]
        if not notes:
            continue
        begins = [c for c in calls if c.func.attr == "begin_batch"]
        assert begins, f"{fn.name}: note_action without begin_batch"
        assert min(b.lineno for b in begins) < min(n.lineno for n in notes)
        return
    raise AssertionError("no turn loop calling strikes.note_action found")
