"""Ten different ways of finding nothing is ten times no progress.

Request e0f4a8bd, 2026-09-08. The no-progress loop breaker keys on
``tool | target | result-fingerprint``, so it fires only when the same call
returns the same bytes. The agent asked ONE document ten re-worded
questions; every answer was a different set of irrelevant passages, so:

    knowledge_base | postgresql-19-a4.pdf | 7c1e…     count 1
    knowledge_base | postgresql-19-a4.pdf | 9ab3…     count 1
    knowledge_base | postgresql-19-a4.pdf | 41f0…     count 1      … ×10

Same tool, same target, same futility, ten counts of one. The breaker never
moved, and only the 40-turn budget was left to stop it.

The fix does not try to GUESS futility from the passages — that is the
lexical-proxy trap this project keeps relearning. The tool already computes
it (see `tests/test_kb_query_footer.py`) and says so in its own output;
the breaker reads that one shared constant, and every fruitless probe of a
document collapses onto ONE signature.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from ghost_agent.core import strikes
from ghost_agent.core.strikes import (NO_ANSWER_FP, StrikeLedger,
                                      action_result_fingerprint,
                                      breaker_fingerprint,
                                      result_says_nothing_found)
from ghost_agent.tools.memory import KB_NO_ANSWER_MARKER

TARGET = "postgresql-19-a4.pdf"


def _fruitless(passages):
    """A weak-match result, in the tool's real shape."""
    return (f"PASSAGES FROM '{TARGET}' (8 closest; best match is weak, "
            f"distance 0.38 — LOWER IS CLOSER):\n"
            f"⚠ {KB_NO_ANSWER_MARKER}. Every passage above is in the "
            f"unrelated band…\n\n{passages}")


def _fingerprint(result):
    """THE SHIPPED decision, not a copy of it — a test that re-implements
    the rule it is checking passes with and without the fix."""
    return breaker_fingerprint(result)


# --- the blind spot itself -----------------------------------------------

def test_ten_rewordings_of_a_hopeless_search_count_as_ten_repeats():
    """THE REGRESSION. World where it fails: the fingerprint is taken from
    the bytes, so ten different irrelevant answers look like ten different
    observations and nothing ever trips."""
    st = StrikeLedger()
    trips = []
    for i in range(10):
        # every call returns DIFFERENT passages — as the live ones did
        res = _fruitless(f"--- [1] (weak match, distance 0.3{i}) ---\npassage {i}")
        sig, count, tripped = st.note_action(
            "knowledge_base", TARGET, _fingerprint(res), threshold=2)
        trips.append(tripped)
    assert trips[0] is False, "one search is not a loop"
    assert trips[1] is True, "the SECOND fruitless search must trip"
    assert all(trips[1:]), trips
    assert sig.endswith("|" + NO_ANSWER_FP)


def test_the_old_fingerprint_would_not_have_tripped_on_any_of_them():
    """The control that proves the test above is not vacuous: run the SAME
    ten results through the byte fingerprint and nothing trips, ever."""
    st = StrikeLedger()
    tripped_any = False
    for i in range(10):
        res = _fruitless(f"--- [1] ---\npassage {i}")
        _, _, tripped = st.note_action(
            "knowledge_base", TARGET, action_result_fingerprint(res),
            threshold=2)
        tripped_any = tripped_any or tripped
    assert tripped_any is False


def test_a_productive_search_never_trips_however_often_it_is_repeated():
    """The mirror. Real searches that return real passages must stay free
    to iterate — muting them would trade this bug for its opposite."""
    st = StrikeLedger()
    for i in range(6):
        res = (f"PASSAGES FROM '{TARGET}' (8 closest; best match is strong, "
               f"distance 0.24…)\n--- [1] ---\nwal_level passage {i}")
        _, _, tripped = st.note_action(
            "knowledge_base", TARGET, _fingerprint(res), threshold=2)
        assert tripped is False, f"a productive query tripped at repeat {i}"


def test_two_different_documents_are_two_different_loops():
    """Futility is per-document: a hopeless search of one manual says
    nothing about the next one."""
    st = StrikeLedger()
    for target in ("a.pdf", "b.pdf"):
        for _ in range(1):
            _, _, tripped = st.note_action(
                "knowledge_base", target, _fingerprint(_fruitless("x")),
                threshold=2)
            assert tripped is False


def test_a_fruitless_probe_is_recognised_from_the_tools_own_constant():
    """One home. The marker the breaker looks for is the constant the tool
    prints — reworded in one place, followed in the other. A second copy of
    the sentence here is how the check goes dark."""
    import inspect
    src = inspect.getsource(result_says_nothing_found)
    assert "KB_NO_ANSWER_MARKER" in src
    assert "NOTHING IN THIS DOCUMENT" not in src, \
        "the phrase is duplicated instead of imported"
    assert result_says_nothing_found(_fruitless("p")) is True
    assert result_says_nothing_found("PASSAGES FROM 'x' … strong match") is False
    assert result_says_nothing_found(None) is False
    assert result_says_nothing_found("") is False


def test_the_marker_is_really_in_what_the_tool_prints():
    """…and the constant is not merely declared: the live weak footer must
    contain it, or the breaker watches for something nothing emits."""
    import asyncio

    from ghost_agent.tools.memory import tool_query_document

    class Mem:
        def get_library(self):
            return [TARGET]

        def search_document(self, filename, question, k=8):
            return [{"text": "[x] A\nbody", "id": "a", "dist": 0.38}]

    out = asyncio.run(tool_query_document(TARGET, "how many chapters?", Mem()))
    assert KB_NO_ANSWER_MARKER in out
    assert result_says_nothing_found(out) is True


# --- the steer must not bar the remedy -----------------------------------

def test_the_fingerprint_decision_has_one_home_and_the_site_uses_it():
    """The choice between "same bytes" and "same futility" is made in
    `breaker_fingerprint`, and the turn loop CALLS it. An inline copy at
    the site is a second authority that drifts — and a mutant that reverts
    the site alone survived every earlier pin here (2026-09-09)."""
    import ast
    import inspect

    from ghost_agent.core import agent as agent_mod
    from ghost_agent.core.agent import GhostAgent
    src = inspect.getsource(GhostAgent)
    # …by the MODULE-level name. At that site `strikes` is the StrikeLedger
    # INSTANCE: `strikes.breaker_fingerprint(...)` parses, greps green, and
    # raises AttributeError through the whole dispatch path. This pin did
    # not see that on 2026-09-09; the EXECUTED dispatch pins
    # (`tests/test_4ec_dispatch_pins.py`, 75 failures) did.
    assert "_breaker_fingerprint(str_res)" in src
    assert "strikes.breaker_fingerprint" not in src
    assert "breaker_fingerprint as _breaker_fingerprint" in inspect.getsource(agent_mod)
    # both branches are the function's, and both are reachable
    assert breaker_fingerprint(_fruitless("p")) == NO_ANSWER_FP
    assert breaker_fingerprint("ordinary result") == action_result_fingerprint(
        "ordinary result")


def test_the_fruitless_steer_branch_is_reachable():
    """`if _fruitless:` → `if False:` leaves the whole message in the source
    and unreachable — a text pin cannot tell the difference. Checked by
    AST: the branch that carries the message must be guarded by a NAME, not
    a constant."""
    import ast
    import inspect
    import textwrap

    from ghost_agent.core.agent import GhostAgent
    tree = ast.parse(textwrap.dedent(inspect.getsource(GhostAgent)))
    guarded = [n for n in ast.walk(tree)
               if isinstance(n, ast.If) and "has now told you" in ast.dump(n)]
    assert guarded, "the fruitless steer is gone from the turn loop"
    tests = [n.test for n in guarded]
    assert any(isinstance(t, ast.Name) and t.id == "_fruitless" for t in tests), \
        [ast.dump(t)[:60] for t in tests]
    assert not any(isinstance(t, ast.Constant) for t in tests), "branch pinned open/closed"


def test_knowledge_base_keeps_its_tools_after_the_steer():
    """The remedy for a fruitless search is ANOTHER call to this tool —
    `action='outline'`. Force-finalising the turn (the default steer) drops
    the toolset and bars exactly that, so `knowledge_base` is on the
    read/write exemption list and gets the softer steer."""
    assert "knowledge_base" in strikes.READWRITE_LOOP_TOOLS
    assert strikes.is_readwrite_loop_exempt("knowledge_base") is True
    # …and the entry is listed ONCE: it was already there for the
    # read/write reason, so the e0f4a8bd reason is a comment on that entry
    # rather than a second copy (a duplicate in a frozenset is invisible,
    # and deleting either half then changes nothing — which is how a
    # "removed the exemption" mutant survived on 2026-09-09).
    import inspect
    src = inspect.getsource(strikes)
    body = src[src.index("READWRITE_LOOP_TOOLS = frozenset({"):]
    body = body[:body.index("})")]
    assert body.count('"knowledge_base",') == 1, body


def test_the_steer_text_names_the_outline_route():
    """A steer that says "stop" without saying "do this instead" leaves the
    model where it was. Checked at the real site."""
    import inspect

    from ghost_agent.core.agent import GhostAgent
    src = inspect.getsource(GhostAgent)
    assert "NO_ANSWER_FP" in src, "the site never distinguishes a fruitless loop"
    import re
    i = src.index("has now told you")
    # Adjacent string literals are joined before matching: the message is
    # wrapped across source lines, so a raw grep would pin the WRAPPING
    # rather than the words the model reads.
    steer = re.sub(r'"\s*\n\s*"', "", src[i:i + 3000])
    assert "action='outline'" in steer
    assert "a semantic search cannot count and never will" in steer
    assert "Do NOT issue another search of this document." in steer
    assert "STOP SEARCHING THIS WAY" in steer


def test_the_hard_stop_still_exists_for_a_document_that_is_never_dropped():
    """The steer is not a licence to keep going: the READWRITE backstop
    aborts at 5."""
    st = StrikeLedger()
    counts = []
    for _ in range(6):
        _, count, _ = st.note_action(
            "knowledge_base", TARGET, _fingerprint(_fruitless("p")), threshold=2)
        counts.append(count)
    assert max(counts) >= strikes.READWRITE_HARD_STOP


def test_a_successful_write_forgets_the_fruitless_history():
    """`note_world_changed` clears every observation: after an ingest the
    document is different and a fresh search is not the old loop."""
    st = StrikeLedger()
    st.note_action("knowledge_base", TARGET, _fingerprint(_fruitless("p")), threshold=2)
    st.note_world_changed()
    _, count, tripped = st.note_action(
        "knowledge_base", TARGET, _fingerprint(_fruitless("p")), threshold=2)
    assert count == 1 and tripped is False
