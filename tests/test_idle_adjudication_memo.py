"""Ask once, not once per dream cycle (§4FQ, item 5, 2026-09-09).

The idle-loop diet: count background LLM calls per kind from the recording
window, name each kind's consumer, cut what has none. Measured over 25 live
hours (2,590 recorded calls):

    bg keepalive                1869   (2 chars each — a max_tokens=1 ping
                                        that prevents measured 5 s
                                        ReadTimeouts; consumer: the node's
                                        network path. Kept.)
    bg tag failure-dimension     151   ← this file
    bg (unlabelled)               81   (verifier / postmortem / self-play)
    …everything else             ≤52 each

`adjudicate_dimension` re-classifies failure records the heuristics left
`unknown`. Of its 151 calls, **147 came back `unknown`** — the value the
heuristic already had — and those 147 were only **SIX distinct records,
each asked 29 times**: once per dream cycle, for ever. Most are not even
failures ("None observed; the solution was direct and efficient"), so
`unknown` is the right and permanent answer.

The docstring already promised this could not happen — "adjudicated
playbook records are persisted so the work isn't repeated next cycle" — but
the persist ran only when the LLM produced a USABLE label, so the common
answer was written nowhere. Same shape as §4FP's fail-closed stores: a
mechanism whose "done" state is never recorded for the common case.
"""
import asyncio
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from ghost_agent.core import failure_distill as FD


class CountingLLM:
    def __init__(self, verdict="unknown"):
        self.verdict, self.calls = verdict, 0

    async def route(self, **kwargs):
        self.calls += 1
        return self.verdict


def _corpus(n=6, text="Multi-file aggregation — None observed; direct and efficient."):
    return [{"handle": f"wl:{i}", "dimension": "unknown", "text": text,
             "cluster": "c", "ts": ""} for i in range(n)]


def _ctx():
    """A context whose state lives in memory (no GHOST_HOME needed) — the
    fallback branch `_load_state`/`_save_state` already support."""
    return SimpleNamespace(_failure_distill_state={})


@pytest.fixture(autouse=True)
def _no_state_file(monkeypatch):
    monkeypatch.setattr(FD, "_state_path", lambda: None)


# --- the regression ------------------------------------------------------

def test_the_same_record_is_adjudicated_once_not_once_per_cycle():
    """THE REGRESSION. World where it fails: nothing remembers an `unknown`
    verdict, so every dream cycle re-asks the same six questions — 29 times
    each in the live window."""
    llm = CountingLLM("unknown")
    ctx = _ctx()

    async def run():
        for _ in range(5):                       # five dream cycles
            await FD.adjudicate_unknowns(llm, _corpus(), context=ctx)
    asyncio.run(run())
    assert llm.calls == 6, f"{llm.calls} calls for 6 records over 5 cycles"


def test_without_the_memo_it_asks_every_cycle():
    """The control that proves the test above is not vacuous: drop the
    context and the old behaviour is back, six calls per cycle."""
    llm = CountingLLM("unknown")

    async def run():
        for _ in range(5):
            await FD.adjudicate_unknowns(llm, _corpus(), context=None)
    asyncio.run(run())
    assert llm.calls == 30


def test_an_edited_record_is_asked_again():
    """The memo keys on the record AND its text, so a lesson someone
    rewrote is a new question — otherwise the fix trades repeated work for
    a permanently stale answer."""
    llm = CountingLLM("unknown")
    ctx = _ctx()

    async def run():
        await FD.adjudicate_unknowns(llm, _corpus(1), context=ctx)
        changed = _corpus(1, text="a completely different failure record")
        await FD.adjudicate_unknowns(llm, changed, context=ctx)
    asyncio.run(run())
    assert llm.calls == 2


def test_a_remembered_LABEL_is_applied_without_a_call():
    """A useful verdict must survive the memo too: the next cycle adopts it
    from the record rather than asking again — and rather than losing it."""
    llm = CountingLLM("orchestration")
    ctx = _ctx()

    async def run():
        first = _corpus(1)
        n1 = await FD.adjudicate_unknowns(llm, first, context=ctx)
        assert first[0]["dimension"] == "orchestration"
        second = _corpus(1)
        await FD.adjudicate_unknowns(llm, second, context=ctx)
        return n1, second
    n1, second = asyncio.run(run())
    assert n1 == 1
    assert llm.calls == 1, "the second cycle asked again"
    assert second[0]["dimension"] == "orchestration", "the label was forgotten"


def test_the_memo_is_bounded():
    """It grows with the corpus, so it must not grow for ever."""
    llm = CountingLLM("unknown")
    ctx = _ctx()

    async def run():
        await FD.adjudicate_unknowns(
            llm, _corpus(FD._ADJUDICATION_MEMO_CAP + 50),
            cap=FD._ADJUDICATION_MEMO_CAP + 50, context=ctx)
    asyncio.run(run())
    memo = FD._load_state(ctx).get(FD._ADJUDICATED_KEY) or {}
    assert len(memo) == FD._ADJUDICATION_MEMO_CAP, len(memo)


def test_a_record_the_heuristic_already_classified_is_never_asked():
    """Unchanged behaviour, pinned: adjudication is for `unknown` only."""
    llm = CountingLLM("model")
    ctx = _ctx()
    corpus = [{"handle": "wl:1", "dimension": "orchestration", "text": "t"}]
    asyncio.run(FD.adjudicate_unknowns(llm, corpus, context=ctx))
    assert llm.calls == 0


def test_the_memo_key_is_the_question_not_just_the_record():
    same = {"handle": "pb:abc", "text": "one"}
    edited = {"handle": "pb:abc", "text": "two"}
    assert FD._adjudication_key(same) == FD._adjudication_key(dict(same))
    assert FD._adjudication_key(same) != FD._adjudication_key(edited)
    assert FD._adjudication_key({}) == ""


def test_a_memo_write_failure_never_breaks_the_pass(monkeypatch):
    """This runs in the idle loop; an unwritable state file must cost the
    optimisation, not the cycle."""
    llm = CountingLLM("unknown")
    ctx = _ctx()

    def boom(*a, **k):
        raise OSError(5, "Input/output error")
    monkeypatch.setattr(FD, "_save_state", boom)
    asyncio.run(FD.adjudicate_unknowns(llm, _corpus(2), context=ctx))
    assert llm.calls == 2                        # the work still happened


def test_the_pass_hands_the_memo_its_context():
    """A memo the caller never enables is dead code — the defect this file
    exists for, one level up."""
    import inspect
    src = inspect.getsource(FD.distill_failure_clusters)
    call = src[src.index("adjudicate_unknowns("):][:300]
    assert "context=context" in call, \
        "distill_failure_clusters does not give adjudicate_unknowns its memo"
