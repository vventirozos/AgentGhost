"""§4EC — `_head_insert_below_start_with` over its input space (2026-09-02).

§4BZ pinned the fence walk through `_finalize_and_return` with fixtures whose
first paragraph break was already outside every fence, so the walk itself
(25 mutants: the loop, its `_fm is None` exit, the single-newline fallback and
the `---` tail fallback) was deletable-green. Extracted to a pure function;
every branch has a row in which it is the one that decides.
"""
import pytest

from ghost_agent.core.agent import _head_insert_below_start_with as hi

B = "NOTE\n\n---\n\n"          # a finalize block, as the callers build them
SW = "BLUF:"


@pytest.mark.parametrize("reply,expected", [
    # no constraint → plain prepend
    pytest.param("BLUF: x\n\nBody", None, id="no_constraint"),
    # constraint, first paragraph break outside any fence → after it
    ("BLUF: x\n\nBody", "BLUF: x\n\n" + B + "Body"),
    # first break INSIDE a fence → walk to the break after the closing fence
    ("BLUF: x\n```\ncode\n\nmore\n```\n\nBody",
     "BLUF: x\n```\ncode\n\nmore\n```\n\n" + B + "Body"),
    # tilde fences walk the same way (dual dialect, §4BZ B-D3)
    ("BLUF: x\n~~~\ncode\n\nmore\n~~~\n\nBody",
     "BLUF: x\n~~~\ncode\n\nmore\n~~~\n\n" + B + "Body"),
    # break inside a fence and NO break after the closing fence → tail rule
    ("BLUF: x\n```\ncode\n\nmore\n```\nBody",
     "BLUF: x\n```\ncode\n\nmore\n```\nBody\n\n---\n\nNOTE"),
    # no paragraph break, a line break, no fences → after the first line
    ("BLUF: x\nBody", "BLUF: x\n\n" + B + "Body"),
    # no paragraph break, a line break, but a fence pair → tail rule
    ("BLUF: x\n```\ncode\n```", "BLUF: x\n```\ncode\n```\n\n---\n\nNOTE"),
    # single line, no breaks at all → tail rule
    ("BLUF: x", "BLUF: x\n\n---\n\nNOTE"),
    # reply ENDS inside an open fence (odd count) → plain prepend
    ("BLUF: x\n```\nstill open", B + "BLUF: x\n```\nstill open"),
    # FIRST break, not last (find vs rfind) — plain, fenced-then-two-breaks, single newlines
    ("BLUF: x\n\nP2\n\nBody", "BLUF: x\n\n" + B + "P2\n\nBody"),
    ("BLUF: x\n```\nc\n\nd\n```\n\nP2\n\nBody", "BLUF: x\n```\nc\n\nd\n```\n\n" + B + "P2\n\nBody"),
    ("BLUF: x\nL2\nL3", "BLUF: x\n\n" + B + "L2\nL3"),
])
def test_placement_table(reply, expected):
    if expected is None:
        assert hi(B, reply, None) == B + reply
    else:
        assert hi(B, reply, SW) == expected


def test_empty_block_is_identity():
    assert hi("", "BLUF: x\n\nBody", SW) == "BLUF: x\n\nBody"


def test_tail_rule_keeps_a_block_without_the_rule_suffix():
    assert hi("PLAIN", "BLUF: x", SW) == "BLUF: x\n\n---\n\nPLAIN"


@pytest.mark.asyncio
async def test_finalize_places_a_correction_after_a_fenced_head_through_the_pure_function():
    """Driven through the real `_finalize_and_return`: an active start-with
    constraint, a reply whose first paragraph break is INSIDE a fence, and a
    deferred correction. The correction must land after the closing fence —
    the walk's decision, reached through the closure's delegate."""
    from unittest.mock import MagicMock
    from tests.test_finalize_stream_pins import _fs, make_fin_agent
    a = make_fin_agent()
    a._take_active_correction = MagicMock(return_value="⚠ CORRECTION: revised.\n\n---\n\n")
    reply = "BLUF: all clear.\n```\ncode\n\nmore\n```\n\nDetails follow."
    out, _, _ = await a._finalize_and_return(_fs(
        final_ai_content=reply, last_user_content='Begin your response with "BLUF:"'))
    assert out.startswith("BLUF: all clear.\n```\ncode\n\nmore\n```\n\n⚠ CORRECTION"), out[:120]
    assert out.endswith("Details follow.")
