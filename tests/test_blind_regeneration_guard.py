"""No second image blind — 2026-09-24 (§4KJ).

Four Slack requests generated a SECOND image (~200 s of the image node each)
without looking at the first. The rule is mechanical: a second
`image_generation` in one request is refused until the earlier output was
inspected by an ANSWERED `vision_analysis` call. The user's words never
decide (a multi-image ask regex was built and removed after three rounds of
misclassification); a verifier repair may regenerate; a second image in the
same batch is always blind and refused.
"""
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.core.agent import (GhostAgent, _blind_regeneration_block,
                                    _images_inspected_this_turn, _turn_generated_images)
from tests.helpers import FakeBgTasks, make_context

SUCCESS = ("SUCCESS: Image generated and saved to sandbox. Rendered at 768x512 in 30 steps.\n\n"
           "(Seed 42 — a record for reproducibility; the result is saved and this request is complete.)\n\n"
           "DO NOT CALL THIS TOOL AGAIN with the same prompt.\n\nRespond DIRECTLY to the user. First, "
           "display the image using EXACTLY this markdown line (keep the short alt text — do NOT paste "
           "the full prompt into it):\n\n![generated image](/api/download/{name})\n\nThen one line.")


def _tool_row(name="gen_a1.png", ok=True, synthetic=False, tool="image_generation"):
    row = {"role": "tool", "name": tool,
           "content": SUCCESS.format(name=name) if ok else "SYSTEM ERROR: node down"}
    if synthetic:
        row["_synthetic"] = True
    return row


_VC = [0]


def _vision_call(target, answered=True):
    """An assistant call row + the tool row the dispatch appends for it —
    a look only when the node ANSWERED (R2 review: an empty/cap result
    must not unlock the guard)."""
    _VC[0] += 1
    cid = f"v{_VC[0]}"
    call = {"role": "assistant", "content": "", "tool_calls": [
        {"id": cid, "type": "function",
         "function": {"name": "vision_analysis",
                      "arguments": json.dumps({"action": "describe_picture", "target": target})}}]}
    result = {"role": "tool", "tool_call_id": cid, "name": "vision_analysis",
              "content": ("VISION ANALYSIS RESULT: one grey cat on a sofa." if answered else
                          "Vision API Error: the vision model spent its whole token budget reasoning and returned NO answer.")}
    return [call, result]


def test_generated_images_are_read_off_success_results_only():
    # by CONTENT (§4KJ R7: a row carries the model's raw name, e.g. an alias):
    # the image tool's success head counts under any name; a vision result
    # that merely mentions a download link does not
    vision_row = {"role": "tool", "name": "vision_analysis",
                  "content": "VISION ANALYSIS RESULT: see ![generated image](/api/download/gen_d4.png)"}
    rows = [_tool_row("gen_a1.png"), _tool_row("gen_b2.png", ok=False),
            _tool_row("gen_c3.png", synthetic=True), vision_row,
            _tool_row("gen_e5.png", tool="imagegen"), _tool_row("gen_a1.png")]
    assert _turn_generated_images(rows) == ["gen_a1.png", "gen_e5.png"]
    assert _turn_generated_images(None) == [] and _turn_generated_images([{"x": 1}, None]) == []


def test_inspected_images_are_answered_vision_calls_only():
    msgs = (_vision_call("/workspace/gen_a1.png") + _vision_call("gen_b2.png")
            + _vision_call("gen_c3.png", answered=False)
            + [{"role": "assistant", "tool_calls": [{"id": "w1", "function": {"name": "web_search", "arguments": "{}"}}]},
               {"role": "tool", "tool_call_id": "w1", "content": "VISION ANALYSIS RESULT: not a vision call"},
               {"role": "user", "content": "gen_z9.png"}])
    assert _images_inspected_this_turn(msgs) == {"gen_a1.png", "gen_b2.png"}


@pytest.mark.parametrize("fname,prior,msgs,blocked", [
    ("image_generation", [], [], False),                                          # first image: free
    ("image_generation", ["gen_a1.png"], [], True),                               # second, blind: blocked
    ("image_generation", ["gen_a1.png"], _vision_call("/workspace/gen_a1.png"), False),  # inspected
    ("image_generation", ["gen_a1.png"], _vision_call("gen_a1.png", answered=False), True),  # asked, not answered
    ("image_generation", ["gen_a1.png"], _vision_call("gen_other.png"), True),    # looked elsewhere
    ("image_generation", ["projects/p1/gen_a1.png"], _vision_call("gen_a1.png"), False),  # project prefix, basename match
    ("image_generation", ["projects/p1/gen_a1.png"], [], True),
    ("vision_analysis", ["gen_a1.png"], [], False),                               # not the image tool
])
def test_the_rule_table(fname, prior, msgs, blocked):
    rows = [_tool_row(p) for p in prior]
    out = _blind_regeneration_block(fname, rows, msgs, "any words at all")
    assert bool(out) is blocked, out
    if blocked:
        assert "gen_a1.png" in out and "projects/" not in out and "vision_analysis" in out and "SYSTEM BLOCK" in out


@pytest.mark.parametrize("ask", [
    "make two images: a cat and a dog", "κάνε άλλη μία", "give me 3 variations", "make it more blue",
])
def test_the_users_words_never_decide(ask):
    """R4 review: the multi-image ask regex misclassified a third of realistic
    asks in two languages (a lexical proxy) and was removed — the escape is an
    answered inspection, whatever the user typed."""
    rows = [_tool_row("gen_a1.png")]
    assert _blind_regeneration_block("image_generation", rows, [], ask)
    assert _blind_regeneration_block("image_generation", rows, _vision_call("gen_a1.png"), ask) is None


def test_a_verifier_repair_may_regenerate():
    """R4 review: a pixel-refuted image could never be regenerated under a
    DONE plan — the guard consumed the repair's one tool batch. The verifier's
    own pixel check is the inspection."""
    rows = [_tool_row("gen_a1.png")]
    assert _blind_regeneration_block("image_generation", rows, [], "x", repair_active=True) is None


def test_a_second_image_in_the_same_batch_is_blind():
    assert _blind_regeneration_block("image_generation", [], [], "x", batch_images=1)
    assert _blind_regeneration_block("image_generation", [], [], "x", batch_images=0) is None


def _resp(content, tool_calls=None):
    return {"choices": [{"message": {"role": "assistant", "content": content,
                                     "tool_calls": tool_calls or []}}]}


def _tc(cid, name, args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


async def _run(monkeypatch, ask, script, *, with_vision=False):
    monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")
    ctx = make_context()
    agent = GhostAgent(ctx)
    gen = AsyncMock(side_effect=[SUCCESS.format(name="gen_a1.png"), SUCCESS.format(name="gen_b2.png")])
    vis = AsyncMock(return_value="VISION ANALYSIS RESULT: one cat, grey, on a sofa.")
    agent.available_tools = {"image_generation": gen, "vision_analysis": vis}
    ctx.llm_client.chat_completion = AsyncMock(side_effect=script)
    out, _, _ = await agent.handle_chat({"messages": [{"role": "user", "content": ask}]}, FakeBgTasks())
    return gen, vis, out


async def test_end_to_end_a_second_blind_generation_is_rejected(monkeypatch):
    """Scripted model: generate, then generate AGAIN without looking, then
    answer. The image tool must run ONCE and the model must have been
    handed the block (it answers with the first image)."""
    script = [
        _resp("", [_tc("c0", "image_generation", {"prompt": "a cat"})]),
        _resp("", [_tc("c1", "image_generation", {"prompt": "a better cat"})]),
        _resp("![generated image](/api/download/gen_a1.png)\n\nA grey cat on a sofa."),
        _resp("(unreachable)"),
    ]
    gen, vis, out = await _run(monkeypatch, "create an image of a cat", script)
    assert gen.await_count == 1
    assert "gen_a1.png" in out


async def test_end_to_end_inspect_first_then_a_second_take_is_allowed(monkeypatch):
    script = [
        _resp("", [_tc("c0", "image_generation", {"prompt": "a cat"})]),
        _resp("", [_tc("c1", "vision_analysis", {"action": "describe_picture", "target": "gen_a1.png"})]),
        _resp("", [_tc("c2", "image_generation", {"prompt": "a cat, two of them"})]),
        _resp("![generated image](/api/download/gen_b2.png)\n\nTwo cats."),
        _resp("(unreachable)"),
    ]
    gen, vis, out = await _run(monkeypatch, "create an image of a cat", script, with_vision=True)
    assert gen.await_count == 2 and vis.await_count == 1


async def test_end_to_end_two_image_calls_in_one_batch_run_once(monkeypatch):
    """R4 review: [image_generation, image_generation] in ONE batch ran two
    blind renders (~200 s each on the one node) — the guard ran before any
    result was appended."""
    script = [
        _resp("", [_tc("c0", "image_generation", {"prompt": "a cat"}),
                   _tc("c1", "image_generation", {"prompt": "a dog"})]),
        _resp("![generated image](/api/download/gen_a1.png)"),
        _resp("(unreachable)"),
    ]
    gen, vis, out = await _run(monkeypatch, "make two images: a cat and a dog", script)
    assert gen.await_count == 1



def test_a_second_image_in_a_batch_is_blocked_even_after_an_inspection_and_in_a_repair():
    rows = [_tool_row("gen_a1.png")]
    looked = _vision_call("gen_a1.png")
    assert _blind_regeneration_block("image_generation", rows, looked, "x", batch_images=1)
    assert _blind_regeneration_block("image_generation", rows, looked, "x", batch_images=1, repair_active=True)


async def test_an_alias_batch_is_counted_by_its_canonical_name(monkeypatch):
    script = [
        _resp("", [_tc("c0", "imagegen", {"prompt": "a cat"}), _tc("c1", "imagegen", {"prompt": "a dog"})]),
        _resp("![generated image](/api/download/gen_a1.png)"), _resp("(unreachable)"),
    ]
    # an alias is not a key of available_tools, so dispatch would rebuild the
    # table and replace the mocks — hold the table still (the rebuild is not
    # the subject here)
    monkeypatch.setattr(GhostAgent, "_rebuild_available_tools", lambda self: None)
    gen, vis, out = await _run(monkeypatch, "draw a cat and a dog", script)
    assert gen.await_count == 1


async def test_a_verifier_repair_regenerates_end_to_end(monkeypatch):
    """R6 pins review: only the function was pinned; dropping the flag at the
    TurnState construction or the dispatch call survived. A REFUTED image
    turn's repair must be allowed to generate again."""
    from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "0")
    monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")
    ctx = make_context()
    verifier = MagicMock(); verifier.llm_client = MagicMock()
    verdicts = AsyncMock(side_effect=[
        VerifyResult(verdict=VerifyVerdict.REFUTED, confidence=0.95, reasoning="r", issues=["the image shows one cat, not two"]),
        VerifyResult(verdict=VerifyVerdict.CONFIRMED, confidence=0.95, reasoning="r", issues=[])])
    verifier.verify_claim = verdicts; verifier.verify_code_output = verdicts
    verifier.verify_visual = AsyncMock(return_value=None)
    ctx.verifier = verifier
    agent = GhostAgent(ctx)
    gen = AsyncMock(side_effect=[SUCCESS.format(name="gen_a1.png"), SUCCESS.format(name="gen_b2.png")])
    agent.available_tools = {"image_generation": gen}
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[
        _resp("", [_tc("c0", "image_generation", {"prompt": "two cats"})]),
        _resp("![generated image](/api/download/gen_a1.png)\n\nTwo cats."),
        _resp("", [_tc("c1", "image_generation", {"prompt": "two cats, clearly two"})]),
        _resp("![generated image](/api/download/gen_b2.png)\n\nTwo cats."),
        _resp("(unreachable)")])
    await agent.handle_chat({"messages": [{"role": "user", "content": "create an image of two cats"}]}, FakeBgTasks())
    assert gen.await_count == 2



def test_an_answered_inspection_through_an_alias_counts():
    msgs = [{"role": "assistant", "content": "", "tool_calls": [
                {"id": "v9", "type": "function",
                 "function": {"name": "vision", "arguments": json.dumps({"target": "gen_a1.png"})}}]},
            {"role": "tool", "tool_call_id": "v9", "name": "vision",
             "content": "VISION ANALYSIS RESULT: a cat"}]
    assert _images_inspected_this_turn(msgs) == {"gen_a1.png"}
