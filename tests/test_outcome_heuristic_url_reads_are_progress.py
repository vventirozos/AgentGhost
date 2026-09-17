"""A url-bearing browser read is a page load, not a repeated click (§4HE).

Request 5fa6aa97 read fourteen different articles with
``browser(operation='extract_text', url=…, selector='body')`` and was
labelled FAILED — "browser selector 'body' used 14× in one turn (≥ 4
threshold)" — over a verifier CONFIRMED (0.95) reply. Signal 2 reset its
window only on ``navigate``/``goto``; the url-bearing read shape (the way
the agent reads pages since the §4GL/§4GN routes) never counted as
progress. The same label on 4d2098c2 (13 distinct urls) seeded a
reflection that told the agent to use non-body selectors "to stay under
the threshold" — the heuristic teaching the agent to game it. A NEW url
that did not fail now clears the window; the same url re-read with the
same selector still accumulates. Each pin names the world it fails in.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ghost_agent.distill.outcome_heuristics import classify_chat_outcome
from ghost_agent.distill.schema import Outcome, ToolCall, Trajectory

_OK = "--- BROWSER RESULT ---\nSTATUS: OK\nOP: extract_text\n"
_ERR = ("[FAILURE BANNER] --- BROWSER RESULT ---\nSTATUS: ERROR\n"
        "Runner failed (exit 1): Error: Page.query_selector: Execution context was destroyed")


def _read(url, result=_OK):
    return ToolCall(name="browser",
                    arguments={"operation": "extract_text", "selector": "body",
                               "max_chars": "7000", "url": url},
                    result=result)


def _traj(calls):
    return Trajectory(outcome=Outcome.UNKNOWN.value, tool_calls=calls,
                      final_response="Here's the forensic synthesis.")


def test_fourteen_reads_of_fourteen_articles_are_not_a_stuck_loop():
    """FAILS IF: the url-bearing read does not count as progress — the
    live world (the recorded 5fa6aa97 shape, 14 distinct urls)."""
    v = classify_chat_outcome(_traj([_read(f"https://site{i}.example/article") for i in range(14)]))
    assert v.outcome == Outcome.UNKNOWN.value, v.reason


def test_the_same_page_re_read_with_the_same_selector_still_counts():
    """FAILS IF: any url-bearing call clears the window — a re-read of the
    SAME url is exactly the "stuck" shape the signal exists for."""
    v = classify_chat_outcome(_traj([_read("https://one.example/") for _ in range(5)]))
    assert v.outcome == Outcome.FAILED.value
    assert "'body' used 5×" in v.reason


def test_a_failed_load_of_a_new_url_is_not_progress():
    """FAILS IF: the error guard is dropped — a load that died is not a
    page the agent got to read."""
    calls = [_read("https://one.example/") for _ in range(3)] + [_read("https://two.example/", _ERR)]
    v = classify_chat_outcome(_traj(calls))
    assert v.outcome == Outcome.FAILED.value


def test_navigate_records_the_page_so_its_re_reads_accumulate():
    """FAILS IF: `navigate` clears the window but forgets the url — a
    url-bearing re-read of that same page would then look NEW and clear
    the window the url-less reads had filled. Here: navigate to A, three
    reads of the current page, then one read that names A — four reads of
    one page with one selector."""
    calls = [ToolCall(name="browser", arguments={"operation": "navigate", "url": "https://one.example/"}, result=_OK)]
    calls += [ToolCall(name="browser", arguments={"operation": "extract_text", "selector": "body"}, result=_OK)
              for _ in range(3)]
    calls += [_read("https://one.example/")]
    v = classify_chat_outcome(_traj(calls))
    assert v.outcome == Outcome.FAILED.value, v.reason


def test_two_pages_alternating_is_two_reads_each():
    """FAILS IF: the window keys on the selector alone — alternating
    between two pages is research, not thrash."""
    calls = [_read("https://a.example/"), _read("https://b.example/")] * 3
    v = classify_chat_outcome(_traj(calls))
    assert v.outcome == Outcome.UNKNOWN.value, v.reason
