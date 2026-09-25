"""The image-turn visual gate, and the two stand-downs that were REMOVED —
2026-09-24 (§4KJ).

* Four of six image turns were CONFIRMED at 0.9-1.0 by a text judge that
  never saw the pixels: "Create an image depicting…" carried no visual word,
  so the gate never opened. An image generated this turn now opens it; a
  CONFIRMED without a DECISIVE pixel verdict (≥ 0.7, CONFIRMED or REFUTED) is
  capped to UNCERTAIN and stamped `image-unseen`.
* Two lexical stand-downs (an "inability" refute, a "fabricated count"
  refute) were built for two live false refutes and REMOVED after four review
  rounds each found grounded refutes they downgraded. The last tests here pin
  that removal: the live issues stay REFUTED.
"""
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.core.agent import GhostAgent, _turn_generated_images
from ghost_agent.core.verifier import VerifyResult, VerifyVerdict as VV
from tests.helpers import make_context

G = GhostAgent


def _r(issues, verdict=VV.REFUTED, conf=0.9):
    return VerifyResult(verdict=verdict, confidence=conf, reasoning="judge said so", issues=list(issues))


LIVE_INABILITY = ("The vision analysis tool returned empty results, so the agent could not verify "
                  "the content of the generated image.")
LIVE_INABILITY_PARAPHRASE = ("The vision analysis tool returned empty results, so the claim about the "
                             "image could not be verified.")
LIVE_FABRICATION = ("Fabrication of 'Six spec cards': The evidence shows only one file written "
                    "('dvda_schematic.html'). The claim details six spec cards.")
# 6 divs + 1 CSS rule + 2 prose mentions = 9 adjacent-phrase occurrences, the
# live file's shape (6 rendered cards, 9 phrase hits): the guard's count is a
# FLOOR that separates six from twelve, not six from nine.
SIX_CARDS_DOC = ("<style>.spec-card{border:1px}</style><p>The spec cards below; each spec card is one view.</p>" +
                 "".join(f'<div class="spec-card"><h3>Card {i}</h3></div>' for i in range(6)))
SIX_CARDS_HTML = "[file_system] " + SIX_CARDS_DOC
VISION_FAILED_ROW = {"role": "tool", "name": "vision_analysis",
                     "content": "Vision API Error: the vision model spent its whole token budget reasoning and returned NO answer."}


# ── unseen-image cap ──────────────────────────────────────────────────────

def test_cap_turns_confirmed_into_uncertain_and_leaves_the_rest():
    c = _r([], verdict=VV.CONFIRMED, conf=1.0)
    out = G._cap_unseen_image_confirm(c)
    assert out.verdict == VV.UNCERTAIN and out.confidence <= 0.6 and out.unseen_image_capped is True
    for v in (None, _r(["x"]), _r([], verdict=VV.UNCERTAIN, conf=0.8)):
        assert G._cap_unseen_image_confirm(v) is v


SUCCESS = ("SUCCESS: Image generated and saved to sandbox. Rendered at 768x512.\n\n"
           "Respond DIRECTLY to the user. First, display the image using EXACTLY this markdown line:\n\n"
           "![generated image](/api/download/{name})\n\nThen one line.")
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


async def _verdict(tmp_path, *, image: bool, visual, extra_messages=()):
    """Replay through `_compute_verifier_verdict`: the judge CONFIRMS at
    0.95 (the escalation-overturn shape); `visual` is what verify_visual
    returns."""
    ctx = make_context()
    agent = GhostAgent(ctx)
    agent.context.sandbox_dir = str(tmp_path)
    agent.context.current_project_id = None
    agent.context.project_store = None
    agent.context.trajectory_collector = None
    verifier = MagicMock()
    verifier.llm_client = MagicMock()

    async def judge(*a, **k):
        return VerifyResult(verdict=VV.CONFIRMED, confidence=0.95, reasoning="looks fine", issues=[])
    for name in ("verify_response", "verify", "verify_turn", "run", "verify_claim", "verify_code_output"):
        setattr(verifier, name, judge)
    verifier.verify_visual = (AsyncMock(side_effect=visual) if isinstance(visual, BaseException)
                              else AsyncMock(return_value=visual))
    agent.context.verifier = verifier
    agent._is_strict_trivial_chat = lambda lc: False
    agent._verify_depth_for_turn = lambda *a, **k: False
    req = "Create an image depicting a lighthouse at dusk"
    if image:
        (tmp_path / "gen_a1.png").write_bytes(PNG)
        body = SUCCESS.format(name="gen_a1.png")
        tools = [{"name": "image_generation", "arguments": {"prompt": "a lighthouse at dusk"},
                  "content": body, "result": body}]
        reply = "![generated image](/api/download/gen_a1.png)\n\nA lighthouse at dusk, warm light on the water."
    else:
        body = "### 1. Lighthouse\nBuilt 1873.\n"
        tools = [{"name": "web_search", "arguments": {"query": "lighthouse"}, "content": body, "result": body}]
        reply = "The lighthouse was built in 1873."
        req = "when was the lighthouse built"
    messages = [{"role": "user", "content": req},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "c0", "type": "function", "function": {"name": tools[0]["name"],
                                                                   "arguments": json.dumps(tools[0]["arguments"])}}]},
                {"role": "tool", "tool_call_id": "c0", "name": tools[0]["name"], "content": body}] + list(extra_messages)
    res, _ = await agent._compute_verifier_verdict(
        tools_run_this_turn=tools, messages=messages, final_ai_content=reply,
        last_user_content=req, lc=req.lower(), req_id="t1", trajectory_id="t1")
    return res, verifier.verify_visual


async def test_an_image_turn_opens_the_visual_gate_without_a_visual_word(tmp_path):
    seen = VerifyResult(verdict=VV.CONFIRMED, confidence=0.9, reasoning="the pixels show a lighthouse", issues=[])
    res, vv = await _verdict(tmp_path, image=True, visual=seen)
    vv.assert_awaited_once()
    assert res is not None and res.verdict == VV.CONFIRMED and not getattr(res, "unseen_image_capped", False)


async def test_an_image_turn_with_no_pixel_verdict_is_capped(tmp_path):
    res, vv = await _verdict(tmp_path, image=True, visual=None)
    vv.assert_awaited_once()
    assert res is not None and res.verdict == VV.UNCERTAIN and res.confidence <= 0.6
    assert res.unseen_image_capped is True


async def test_a_visual_refute_still_overrides(tmp_path):
    bad = VerifyResult(verdict=VV.REFUTED, confidence=0.9, reasoning="one object, not two", issues=["one object"])
    res, _ = await _verdict(tmp_path, image=True, visual=bad)
    assert res.verdict == VV.REFUTED


async def test_a_text_turn_is_untouched(tmp_path):
    res, vv = await _verdict(tmp_path, image=False, visual=None)
    vv.assert_not_awaited()
    assert res is not None and res.verdict == VV.CONFIRMED and res.confidence == 0.95


def test_generated_images_helper_reads_the_success_line():
    assert _turn_generated_images([{"name": "image_generation", "content": SUCCESS.format(name="gen_z.png")}]) == ["gen_z.png"]




async def _refuting_verdict(tmp_path, issue, tool_body, extra_rows=None):
    """Replay through `_compute_verifier_verdict` with the judge saying
    exactly what the live judge said."""
    ctx = make_context()
    agent = GhostAgent(ctx)
    agent.context.sandbox_dir = str(tmp_path)
    agent.context.current_project_id = None
    agent.context.project_store = None
    agent.context.trajectory_collector = None
    verifier = MagicMock()
    verifier.llm_client = MagicMock()

    async def judge(*a, **k):
        return VerifyResult(verdict=VV.REFUTED, confidence=0.9, reasoning="r", issues=[issue])
    for name in ("verify_response", "verify", "verify_turn", "run", "verify_claim", "verify_code_output"):
        setattr(verifier, name, judge)
    verifier.verify_visual = AsyncMock(return_value=None)
    agent.context.verifier = verifier
    agent._is_strict_trivial_chat = lambda lc: False
    agent._verify_depth_for_turn = lambda *a, **k: False
    req = "create schematics for the thing as an html file"
    (tmp_path / "dvda_schematic.html").write_text(SIX_CARDS_DOC, encoding="utf-8")
    # the production write receipt never echoes content (tools/file_system.py); the
    # guard must read the written file back off disk to reach the live case
    tools = [{"name": "file_system", "arguments": {"operation": "write", "path": "dvda_schematic.html"},
              "content": tool_body, "result": tool_body}]
    tools = tools + list(extra_rows or [])
    reply = "Here's your schematic — six spec cards and three views, in dvda_schematic.html."
    res, _ = await agent._compute_verifier_verdict(
        tools_run_this_turn=tools, messages=[{"role": "user", "content": req}],
        final_ai_content=reply, last_user_content=req, lc=req.lower(), req_id="s1", trajectory_id="s1")
    return res


RECEIPT = ("SUCCESS: Wrote 1200 chars to 'dvda_schematic.html'. Script-side path (from sandbox cwd): "
           "'dvda_schematic.html'.")                     # the production receipt text (tools/file_system.py)


async def test_the_removed_stand_downs_leave_the_live_refutes_refuted(tmp_path):
    """Both live issues reach the verdict unchanged: no lexical rule decides
    whether a judge's issue is a contradiction any more."""
    res = await _refuting_verdict(tmp_path, LIVE_INABILITY, RECEIPT, extra_rows=[VISION_FAILED_ROW])
    assert res is not None and res.verdict == VV.REFUTED
    res2 = await _refuting_verdict(tmp_path, LIVE_FABRICATION, RECEIPT)
    assert res2 is not None and res2.verdict == VV.REFUTED
    from ghost_agent.core.agent import GhostAgent
    assert not hasattr(GhostAgent, "_stand_down_instrument_failure_refute")
    assert not hasattr(GhostAgent, "_stand_down_refute_on_present_term")


async def test_an_indecisive_pixel_verdict_does_not_count_as_seen(tmp_path):
    """R4 review: an UNCERTAIN (or a REFUTED below 0.7) from the vision node
    counted as "seen", and the text judge's CONFIRMED stood uncapped."""
    for vv in (VerifyResult(verdict=VV.UNCERTAIN, confidence=0.9, reasoning="hmm", issues=[]),
               VerifyResult(verdict=VV.REFUTED, confidence=0.5, reasoning="maybe", issues=["x"])):
        res, _ = await _verdict(tmp_path, image=True, visual=vv)
        assert res is not None and res.verdict == VV.UNCERTAIN and res.unseen_image_capped is True


async def test_the_cap_is_stamped_for_the_override_report(tmp_path):
    res, _ = await _verdict(tmp_path, image=True, visual=None)
    assert "image-unseen" in str(getattr(res, "override", "") or getattr(res, "override_chain", "") or res.__dict__)



async def test_a_vision_failure_still_caps_the_image_turn(tmp_path):
    """R6: the cap sat inside the try; an exception before it skipped it."""
    res, _ = await _verdict(tmp_path, image=True, visual=RuntimeError("vision node down"))
    assert res is not None and res.verdict == VV.UNCERTAIN and res.unseen_image_capped is True


async def test_the_generated_image_is_the_one_judged(tmp_path):
    """R6: the newest referenced image on disk (a later screenshot) was
    judged instead of the image generated this turn."""
    import os as _os, time as _t
    seen = VerifyResult(verdict=VV.CONFIRMED, confidence=0.9, reasoning="ok", issues=[])
    shot = tmp_path / "zz_newer_screenshot.png"
    shot.write_bytes(PNG)
    _os.utime(shot, (_t.time() + 3600, _t.time() + 3600))       # newer than the generated image
    later = [{"role": "tool", "name": "browser", "tool_call_id": "b1",
              "content": "Screenshot saved to zz_newer_screenshot.png"}]
    res, vv = await _verdict(tmp_path, image=True, visual=seen, extra_messages=later)
    after = vv.await_args.kwargs.get("after_image") or ""
    assert str(after).endswith("gen_a1.png"), after
