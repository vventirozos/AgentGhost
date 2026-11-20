"""§4VD-C — pass 2 never supersedes a summary whose facts the later one does not carry.

Pass 2 dropped an earlier summary group whose words overlapped a later one (Jaccard ≥ 0.55): two
parallel results ("accuracy 0.91 … batch 32" / "accuracy 0.84 … batch 64") were read as one
restating the other and the first was deleted. The earlier group now goes only when every figure,
code span and URL it states is in the later one.

The opener rule (pass 1 drops any short paragraph that OPENS like a beat — "Great, your flight is
confirmed for 09:40." included) was NOT changed: three review rounds each found new deletions or
leaks inside the fix (PROJECT_JOURNAL §4VD); it stays an open item for the operator. The pin below
records the known loss so a later change sees it.
"""
import pytest

from ghost_agent.core import reply_smoothing as RS
from ghost_agent.core.reply_smoothing import smooth_reply


def test_two_parallel_results_are_not_one_superseding_the_other():
    a = "The first run reached accuracy 0.91 on the validation set with batch size 32."
    b = "The second run reached accuracy 0.84 on the validation set with batch size 64."
    out = smooth_reply(f"{a}\n\n{b}")
    assert a in out and b in out


def test_a_restated_summary_with_the_same_facts_is_still_superseded():
    first = ("The service is fixed:\n\n- port 8100 answers\n- `render()` reuses the grid\n"
             "- 5 apps load")
    second = ("The service is fixed and restarted:\n\n- port 8100 answers\n- `render()` reuses the grid\n"
              "- 5 apps load")
    out = smooth_reply(f"{first}\n\n{second}")
    assert out.count("port 8100") == 1 and "restarted" in out


def test_a_figure_only_the_earlier_summary_has_keeps_it():
    """The accepted cost, pinned so it is a decision and not an accident."""
    first = "Tests ran on the branch:\n\n- 7 passed\n- 3 failed in the parser module"
    second = "Tests ran on the branch:\n\n- 10 passed\n- the parser module is fixed"
    out = smooth_reply(f"{first}\n\n{second}")
    assert "7 passed" in out and "10 passed" in out


@pytest.mark.parametrize("text,facts", [
    ("accuracy 0.91 at https://x.io/a, see `run.py`.", {"0.91", "https://x.io/a", "`run.py`"}),
    ("no facts here", set()),
    ("at 09:40 on 1,200 rows", {"09:40", "1,200"}),
])
def test_facts(text, facts):
    assert RS._facts(text) == facts


def test_a_later_summary_with_more_facts_supersedes_the_earlier():
    """The facts check is earlier ⊆ later, not the reverse."""
    first = "The service is fixed:\n\n- the grid is reused on every click\n- the resize handle works"
    second = ("The service is fixed and restarted:\n\n- the grid is reused on every click\n"
              "- the resize handle works\n- port 8100 answers")
    out = smooth_reply(f"{first}\n\n{second}")
    assert out.count("the grid is reused") == 1 and "port 8100" in out


def test_facts_keep_case():
    first = "Build summary:\n\n- the entry point is `Run.py`\n- the tests pass on the branch"
    second = "Build summary:\n\n- the entry point is `run.py`\n- the tests pass on the branch"
    out = smooth_reply(f"{first}\n\n{second}")
    assert "`Run.py`" in out and "`run.py`" in out


def test_the_known_opener_loss_is_still_open():
    """NOT a wish — the recorded state. The opener rule still deletes a status line that opens like a
    beat; when the operator decides the opener question, this pin flips with it."""
    text = ("I checked the configuration and the credentials.\n\nGreat, your flight is confirmed for 09:40."
            "\n\nEverything else is in order: 3 services restarted and the queue is empty.")
    assert "09:40" not in smooth_reply(text)
