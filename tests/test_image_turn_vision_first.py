"""A generated image is judged by its pixels FIRST (2026-09-25, req slack-5e).

Live: the cheap text judge refuted "Dendias popping his head around a door"
because the image tool's output — a status line — "does not confirm the
action"; the strong model overturned it (a WARNING and a main-model call on
every image turn), and the vision check then CONFIRMED at 100%. A wrong refute
the escalation upheld was never lifted by that visual CONFIRMED. Now the vision
verdict runs before the text judge and joins its evidence as an independent
check; the visual arm reuses it instead of looking twice.
"""
import json
from unittest.mock import AsyncMock, MagicMock

from ghost_agent.core import agent as A
from ghost_agent.core.agent import GhostAgent
from ghost_agent.core.verifier import (VerifyResult, VerifyVerdict as VV, _VERIFY_CLAIM_PROMPT,
                                       _VERIFY_ENUMERATE_PROMPT, _VERIFY_ADJUDICATE_PROMPT, _REQUIRED_RULE_MARKERS)
from tests.helpers import make_context
from tests.test_verifier_stand_downs_2026_09_24 import PNG, SUCCESS


async def _run(tmp_path, *, image=True, visual=None, text=None, extra_image=None, extra_tools=()):
    order, seen = [], {}
    agent = GhostAgent(make_context())
    agent.context.sandbox_dir = str(tmp_path)
    agent.context.current_project_id = None
    agent.context.project_store = None
    agent.context.trajectory_collector = None
    verifier = MagicMock()
    verifier.llm_client = MagicMock()
    text = text or VerifyResult(verdict=VV.CONFIRMED, confidence=0.95, reasoning="fine", issues=[])

    async def judge(claim, evidence, context="", **k):
        order.append("text")
        seen["evidence"] = evidence
        return text

    async def look(**k):
        order.append("vision")
        if isinstance(visual, BaseException):
            raise visual
        return visual
    verifier.verify_claim = judge
    verifier.verify_visual = AsyncMock(side_effect=look)
    agent.context.verifier = verifier
    agent._is_strict_trivial_chat = lambda lc: False
    agent._verify_depth_for_turn = lambda *a, **k: False
    if image:
        (tmp_path / "gen_a1.png").write_bytes(PNG)
        body = SUCCESS.format(name="gen_a1.png")
        name, args, req = "image_generation", {"prompt": "a man peeking round a door"}, "make an image of a man peeking round a door"
        reply = "![generated image](/api/download/gen_a1.png)\n\nHere he is, peeking round the door."
    else:
        body, name, args = "### 1. Lighthouse\nBuilt 1873.\n", "web_search", {"query": "lighthouse"}
        req, reply = "when was the lighthouse built", "It was built in 1873."
    tools = [{"name": name, "content": body, "tool_call_id": "c0"}]
    if extra_image:
        tools.insert(0, {"name": "image_generation", "content": SUCCESS.format(name=extra_image), "tool_call_id": "c9"})
    tools = list(extra_tools) + tools
    messages = [{"role": "user", "content": req},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "c0", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]},
                {"role": "tool", "tool_call_id": "c0", "name": name, "content": body}]
    res, _ = await agent._compute_verifier_verdict(
        tools_run_this_turn=tools, messages=messages, final_ai_content=reply,
        last_user_content=req, lc=req.lower(), req_id="t1", trajectory_id="t1")
    return res, order, seen, verifier.verify_visual


SEEN_OK = VerifyResult(verdict=VV.CONFIRMED, confidence=1.0, reasoning="a man peeks round a half-open door", issues=[])


async def test_vision_runs_first_and_its_verdict_reaches_the_text_judge(tmp_path):
    res, order, seen, vv = await _run(tmp_path, visual=SEEN_OK)
    assert order == ["vision", "text"]
    vv.assert_awaited_once()                                      # reused by the visual arm, not repeated
    assert "[vision_check]" in seen["evidence"] and "the pixels MATCH" in seen["evidence"]
    assert "peeks round a half-open door" not in seen["evidence"]   # the verdict only — vision's reasoning restates the reply
    assert res.verdict == VV.CONFIRMED


async def test_a_visual_refute_still_overrides_a_text_confirm(tmp_path):
    bad = VerifyResult(verdict=VV.REFUTED, confidence=0.9, reasoning="no door in the image", issues=["no door"])
    res, order, seen, vv = await _run(tmp_path, visual=bad)
    assert res.verdict == VV.REFUTED and "the pixels DO NOT MATCH" in seen["evidence"]
    vv.assert_awaited_once()


async def test_an_undecided_vision_verdict_adds_nothing_and_the_cap_still_applies(tmp_path):
    weak = VerifyResult(verdict=VV.UNCERTAIN, confidence=0.5, reasoning="blurry", issues=[])
    res, order, seen, vv = await _run(tmp_path, visual=weak)
    assert "[vision_check]" not in seen["evidence"]
    assert res.verdict == VV.UNCERTAIN and res.unseen_image_capped is True
    vv.assert_awaited_once()


async def test_a_failed_look_is_not_paid_for_twice(tmp_path):
    res, order, seen, vv = await _run(tmp_path, visual=None)
    vv.assert_awaited_once()
    assert "[vision_check]" not in seen["evidence"] and res.unseen_image_capped is True


async def test_a_text_turn_never_looks(tmp_path):
    res, order, seen, vv = await _run(tmp_path, image=False)
    vv.assert_not_awaited()
    assert order == ["text"] and "[vision_check]" not in seen["evidence"]


def test_a_confident_uncertain_is_not_a_vision_verdict():
    assert A._vision_evidence_block(VerifyResult(verdict=VV.UNCERTAIN, confidence=0.9, reasoning="r", issues=[]), "x") == ""



def test_the_block_carries_no_reply_text_for_either_verdict():
    """Review R20 CRIT: vision's reasoning AND a refute's issues restate the
    reply; in the evidence they made its figures and names "supported"."""
    ok = VerifyResult(verdict=VV.CONFIRMED, confidence=0.95, reasoning="the sign reads OPEN 24/7 and 3 people stand there", issues=[])
    bad = VerifyResult(verdict=VV.REFUTED, confidence=0.9, reasoning="claims 330 m", issues=["the response claims 330 m, built 1889"])
    for vv in (ok, bad):
        blk = A._vision_evidence_block(vv, "g.png")
        assert blk and not any(t in blk for t in ("24/7", "3 people", "330", "1889"))


async def test_the_block_leaves_the_binders_audit_unmoved(tmp_path):
    from ghost_agent.core import claim_binding as CB
    reply = "The Eiffel Tower, built by Gustave Eiffel in 1889, is 330 metres tall."
    blk = A._vision_evidence_block(VerifyResult(verdict=VV.CONFIRMED, confidence=0.95,
                                                reasoning="Eiffel Tower 1889 330 metres Gustave Eiffel", issues=[]), "g.png")
    base = {a.text: a.status for a in CB.audit_numbers(reply, "[image_generation] SUCCESS", "")}
    withb = {a.text: a.status for a in CB.audit_numbers(reply, "[image_generation] SUCCESS\n" + blk, "")}
    assert base == withb


async def test_a_multi_image_turn_keeps_the_old_order(tmp_path):
    """Review R20: vision sees only the last image; the early look (and the
    rule it enables) applies to single-image turns only."""
    from tests.test_verifier_stand_downs_2026_09_24 import PNG as _P
    (tmp_path / "gen_b2.png").write_bytes(_P)
    res, order, seen, vv = await _run(tmp_path, visual=SEEN_OK, extra_image="gen_b2.png")
    assert order[0] == "text" and "[vision_check]" not in seen["evidence"]


async def test_the_tool_evidence_stays_beside_the_block(tmp_path):
    res, order, seen, vv = await _run(tmp_path, visual=SEEN_OK)
    assert "SUCCESS: Image generated" in seen["evidence"] and "[vision_check]" in seen["evidence"]


async def test_a_raising_early_look_is_retried_once_by_the_visual_arm(tmp_path):
    res, order, seen, vv = await _run(tmp_path, visual=RuntimeError("vision timeout"))
    assert vv.await_count == 2 and "[vision_check]" not in seen["evidence"]



async def test_the_evidence_stays_within_its_budget(tmp_path, monkeypatch):
    """The block is fitted INSIDE the budget: the digest is sliced to make room."""
    monkeypatch.setattr(A, "_evidence_budget_for", lambda tools: (400, 3))
    res, order, seen, vv = await _run(tmp_path, visual=SEEN_OK)
    assert "[vision_check]" in seen["evidence"] and len(seen["evidence"]) <= 400


async def test_an_unresolvable_generated_image_gets_no_block(tmp_path, monkeypatch):
    """Review R21: the fallback file's name (and pixels) stood in for the
    generated image."""
    monkeypatch.setattr(A, "_resolve_image_path", lambda name, sbx: None)
    monkeypatch.setattr(A, "_select_visual_evidence", lambda *a, **k: (None, str(tmp_path / "Gustave Eiffel 1889.png")))
    res, order, seen, vv = await _run(tmp_path, visual=SEEN_OK)
    assert "[vision_check]" not in seen["evidence"] and order[0] == "text"
    vv.assert_awaited_once()          # the visual arm still looks, at its own fallback image, and caps as before


# ── review R22 (two fresh reviewers) ──

def test_the_block_names_the_image_and_carries_no_digits():
    """R22 MAJOR: "(90%)" supported a reply's "90%" in the number audit."""
    blk = A._vision_evidence_block(SEEN_OK, "/sbx/gen_a1.png")
    assert blk == ("[vision_check] (an independent look at the pixels of gen_a1.png; it covers ONLY what that image "
                   "shows, never the reply's other facts) clearly: the pixels MATCH the reply's description of this image")
    at = A._vision_evidence_block(VerifyResult(verdict=VV.CONFIRMED, confidence=0.70, reasoning="r", issues=[]), "g.png")
    assert at.endswith("probably: the pixels MATCH the reply's description of this image")
    assert not any(ch.isdigit() for ch in at.replace("g.png", ""))
    assert A._vision_evidence_block(VerifyResult(verdict=VV.CONFIRMED, confidence=0.69, reasoning="r", issues=[]), "x") == ""


def test_the_confidence_does_not_support_a_percentage_in_the_reply():
    from ghost_agent.core import claim_binding as CB
    reply = "About 90% of orange tabby cats are male."
    blk = A._vision_evidence_block(VerifyResult(verdict=VV.CONFIRMED, confidence=0.90, reasoning="r", issues=[]), "g.png")
    st = {a.text: a.status for a in CB.audit_numbers(reply, "[image_generation] SUCCESS\n" + blk, "")}
    assert st.get("90%") != "supported", st


def test_every_judge_prompt_carries_the_image_rule_and_scopes_it():
    """R22 MAJOR: the default two-stage cheap judge never saw the rule."""
    for p in (_VERIFY_CLAIM_PROMPT, _VERIFY_ADJUDICATE_PROMPT):
        assert "IMAGE DESCRIPTIONS: an image-generation tool's output is only a status line" in p
        assert "what is depicted, style, colours, composition, mood" in p
        assert "a number, name, date or event the CLAIM presents as image content" in p
    assert "do not name the reply's description of how that image LOOKS" in _VERIFY_ENUMERATE_PROMPT
    assert "a number, name, date or event presented as image content is still a candidate" in _VERIFY_ENUMERATE_PROMPT
    from ghost_agent.core.verifier import _VERIFY_CODE_PROMPT
    assert "A `[vision_check]` line in the TOOL OUTPUT is an independent look at the pixels" in _VERIFY_CODE_PROMPT
    assert any("IMAGE DESCRIPTIONS" in m for m in _REQUIRED_RULE_MARKERS["verifier.adjudicate"])


import pytest as _pt


@_pt.mark.parametrize("long_output", [False, True])
async def test_the_code_lens_sees_the_vision_verdict(tmp_path, long_output):
    """R22 MAJOR: image then a command → the code lens judged with no word
    of the pixels and its refute stood over a 100% vision MATCH."""
    agent = GhostAgent(make_context())
    agent.context.sandbox_dir = str(tmp_path)
    agent.context.current_project_id = None
    agent.context.project_store = None
    agent.context.trajectory_collector = None
    seen = {}
    v = MagicMock(); v.llm_client = MagicMock()

    async def code(code, output, intent, **k):
        seen["output"] = output
        return VerifyResult(verdict=VV.CONFIRMED, confidence=0.9, reasoning="r", issues=[])
    v.verify_code_output = code
    v.verify_visual = AsyncMock(return_value=SEEN_OK)
    agent.context.verifier = v
    agent._is_strict_trivial_chat = lambda lc: False
    agent._verify_depth_for_turn = lambda *a, **k: False
    (tmp_path / "gen_a1.png").write_bytes(PNG)
    body = SUCCESS.format(name="gen_a1.png")
    tools = [{"name": "image_generation", "content": body, "tool_call_id": "c0"},
             {"name": "execute", "content": "EXIT CODE: 0\n768x512\n" + "z" * (6000 if long_output else 0), "tool_call_id": "c1"}]
    msgs = [{"role": "user", "content": "draw a door and print its size"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c0", "type": "function", "function": {"name": "image_generation", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c0", "name": "image_generation", "content": body},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "execute", "arguments": json.dumps({"command": "identify gen_a1.png"})}}]},
            {"role": "tool", "tool_call_id": "c1", "name": "execute", "content": "EXIT CODE: 0\n768x512"}]
    await agent._compute_verifier_verdict(tools_run_this_turn=tools, messages=msgs,
                                          final_ai_content="A door, 768x512.", last_user_content="draw a door and print its size",
                                          lc="draw a door and print its size", req_id="t", trajectory_id="t")
    assert "[vision_check]" in seen["output"] and len(seen["output"]) <= 4000


async def test_the_project_ledger_survives_beside_the_vision_block(tmp_path, monkeypatch):
    """R22 MAJOR: re-slicing after the ledger was appended pushed it out."""
    ledger = "[project_ledger] task t1: DONE — deliver the lighthouse image"
    monkeypatch.setattr(A, "_project_ledger_evidence", lambda *a, **k: ledger)
    monkeypatch.setattr(A, "_evidence_budget_for", lambda tools: (500, 3))
    res, order, seen, vv = await _run(tmp_path, visual=SEEN_OK)
    ev = seen["evidence"]
    assert ledger in ev and "[vision_check]" in ev and len(ev) <= 500



async def test_the_digest_is_cut_once_not_twice(tmp_path, monkeypatch):
    """R22 MINOR: packing at the full budget and re-slicing for the tail cut
    the digest twice — the second mark measured the already-cut digest and
    the truncation severity undercounted the loss."""
    monkeypatch.setattr(A, "_project_ledger_evidence", lambda *a, **k: "[project_ledger] task t1: DONE")
    monkeypatch.setattr(A, "_evidence_budget_for", lambda tools: (900, 3))
    big = {"name": "web_search", "content": "### 1. lighthouse history " + "fact " * 800, "tool_call_id": "w0"}
    res, order, seen, vv = await _run(tmp_path, visual=SEEN_OK, extra_tools=[big])
    import re
    ev = seen["evidence"]
    assert ev.count("…[PACKER CUT") == 1 and len(ev) <= 900
    shown, total = map(int, re.search(r"PACKER CUT#\w+: (\d+) of (\d+) chars shown", ev).groups())
    assert total >= 4000, total          # measured against the SOURCE, not an already-cut digest


async def test_a_multi_step_code_turn_keeps_room_for_the_vision_verdict(tmp_path):
    """The transcript is sized around the ledger AND the vision block; sized
    around the ledger alone, the consumer's [:4000] cut the vision block."""
    agent = GhostAgent(make_context())
    agent.context.sandbox_dir = str(tmp_path)
    agent.context.current_project_id = None
    agent.context.project_store = None
    agent.context.trajectory_collector = None
    seen = {}
    v = MagicMock(); v.llm_client = MagicMock()

    async def code(code, output, intent, **k):
        seen["output"] = output
        return VerifyResult(verdict=VV.CONFIRMED, confidence=0.9, reasoning="r", issues=[])
    v.verify_code_output = code
    v.verify_visual = AsyncMock(return_value=SEEN_OK)
    agent.context.verifier = v
    agent._is_strict_trivial_chat = lambda lc: False
    agent._verify_depth_for_turn = lambda *a, **k: False
    (tmp_path / "gen_a1.png").write_bytes(PNG)
    body = SUCCESS.format(name="gen_a1.png")
    steps = [("image_generation", {}, body), ("execute", {"command": "identify gen_a1.png"}, "a" * 5000),
             ("execute", {"command": "exiftool gen_a1.png"}, "b" * 5000)]
    tools, msgs = [], [{"role": "user", "content": "draw a door, inspect it"}]
    for i, (n, a, out) in enumerate(steps):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}]})
        tools.append({"name": n, "content": out, "tool_call_id": f"c{i}"})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "name": n, "content": out})
    await agent._compute_verifier_verdict(tools_run_this_turn=tools, messages=msgs, final_ai_content="A door.",
                                          last_user_content="draw a door, inspect it", lc="draw a door, inspect it",
                                          req_id="t", trajectory_id="t")
    assert "[step 1/2: execute]" in seen["output"]
    assert "[vision_check]" in seen["output"] and len(seen["output"]) <= 4000   # nothing for verify_code_output's [:4000] to cut
