"""Tests for visual verification — the vision-enabled verifier path that
re-reads rendered pixels so a reported VISUAL symptom ("blocks look
transparent") can't sail through as a text-only CONFIRMED.

Covers:
  * Verifier.verify_visual verdict mapping (REFUTED / CONFIRMED)
  * before+after vs after-only image payloads (use_vision=True, image count)
  * graceful skip (None) when after_image is missing or unreadable
  * the agent-side gate + evidence selection helpers
"""

from __future__ import annotations

import base64

import pytest

from ghost_agent.core.verifier import Verifier, VerifyVerdict
from ghost_agent.core.agent import (
    _is_visual_intent,
    _extract_image_tokens,
    _resolve_image_path,
    _select_visual_evidence,
)

# Smallest valid 1×1 PNG.
_PNG_1x1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYPhfDwAChwGA"
    "60e6kgAAAABJRU5ErkJggg=="
)


class _StubVisionClient:
    """Captures the payload and returns a canned vision verdict."""

    def __init__(self, content: str):
        self.content = content
        self.last_payload = None
        self.last_use_vision = None

    async def chat_completion(self, payload, use_vision=False):
        self.last_payload = payload
        self.last_use_vision = use_vision
        return {"choices": [{"message": {"content": self.content}}]}


def _imgs_in(payload) -> int:
    content = payload["messages"][1]["content"]
    return sum(1 for c in content if c.get("type") == "image_url")


# ───────────────────────────── verify_visual ─────────────────────────────


async def test_verify_visual_refuted_overrides_nothing_here(tmp_path):
    after = tmp_path / "after.png"
    after.write_bytes(_PNG_1x1)
    client = _StubVisionClient(
        '{"verdict":"REFUTED","confidence":0.9,'
        '"reasoning":"blocks still render see-through",'
        '"issues":["transparent terrain"]}'
    )
    v = Verifier(llm_client=client)
    res = await v.verify_visual(
        symptom="the blocks look transparent making the game unusable",
        claim="I split water/leaves into separate meshes",
        after_image=str(after),
    )
    assert res is not None
    assert res.verdict == VerifyVerdict.REFUTED
    assert res.confidence == pytest.approx(0.9)
    assert res.issues == ["transparent terrain"]
    # Routed to the vision path, not the text VERIFY worker.
    assert client.last_use_vision is True
    assert _imgs_in(client.last_payload) == 1  # after only


async def test_verify_visual_confirmed(tmp_path):
    after = tmp_path / "after.png"
    after.write_bytes(_PNG_1x1)
    client = _StubVisionClient(
        '{"verdict":"CONFIRMED","confidence":0.85,"reasoning":"terrain is solid"}'
    )
    v = Verifier(llm_client=client)
    res = await v.verify_visual(symptom="x", claim="y", after_image=str(after))
    assert res.verdict == VerifyVerdict.CONFIRMED


async def test_verify_visual_includes_before_and_after(tmp_path):
    before = tmp_path / "before.png"
    after = tmp_path / "after.png"
    before.write_bytes(_PNG_1x1)
    after.write_bytes(_PNG_1x1)
    client = _StubVisionClient('{"verdict":"CONFIRMED","confidence":0.8}')
    v = Verifier(llm_client=client)
    await v.verify_visual(
        symptom="x", claim="y",
        after_image=str(after), before_image=str(before),
    )
    assert _imgs_in(client.last_payload) == 2  # before + after


async def test_verify_visual_none_without_after():
    client = _StubVisionClient('{"verdict":"REFUTED","confidence":0.9}')
    v = Verifier(llm_client=client)
    res = await v.verify_visual(symptom="x", claim="y", after_image="")
    assert res is None
    assert client.last_payload is None  # model never called


async def test_verify_visual_skips_when_image_unreadable(tmp_path):
    """A missing/unreadable after-image means no pixels to judge → skip
    (None), never a fabricated verdict, and the model is not called."""
    client = _StubVisionClient('{"verdict":"REFUTED","confidence":0.9}')
    v = Verifier(llm_client=client)
    res = await v.verify_visual(
        symptom="x", claim="y", after_image=str(tmp_path / "missing.png"),
    )
    assert res is None
    assert client.last_payload is None


# ───────────────────────────── gate helpers ──────────────────────────────


def test_visual_prompt_judges_claim_not_just_symptom():
    """The verdict must be about whether the agent's RESPONSE honestly
    matches the pixels — NOT merely whether the symptom is still present.
    Otherwise an honest "it's still broken" report gets REFUTED and the
    agent is penalized for telling the truth. Guard the prompt intent."""
    from ghost_agent.core.verifier import _VERIFY_VISUAL_PROMPT
    p = _VERIFY_VISUAL_PROMPT.lower()
    assert "honest" in p
    assert "must not be refuted" in p
    assert "misrepresents" in p


def test_is_visual_intent():
    assert _is_visual_intent("the blocks look transparent making the game unusable")
    assert _is_visual_intent("look at the screen_mine2.png that i just put")
    assert _is_visual_intent("the layout is broken and the UI overlaps")
    assert not _is_visual_intent("refactor the database connection pool")
    assert not _is_visual_intent("")
    assert not _is_visual_intent(None)


def test_extract_image_tokens():
    assert _extract_image_tokens("see /sandbox/screen_mine2.png now") == [
        "/sandbox/screen_mine2.png"
    ]
    assert _extract_image_tokens("a.PNG and b.jpeg") == ["a.PNG", "b.jpeg"]
    assert _extract_image_tokens("no images here") == []


def test_resolve_image_path_strips_container_prefixes(tmp_path):
    (tmp_path / "shot.png").write_bytes(_PNG_1x1)
    assert _resolve_image_path("/sandbox/shot.png", tmp_path) == str(tmp_path / "shot.png")
    assert _resolve_image_path("/workspace/shot.png", tmp_path) == str(tmp_path / "shot.png")
    assert _resolve_image_path("shot.png", tmp_path) == str(tmp_path / "shot.png")
    assert _resolve_image_path("nope.png", tmp_path) is None


def test_select_visual_evidence_picks_before_and_after(tmp_path):
    (tmp_path / "screen_mine2.png").write_bytes(_PNG_1x1)
    (tmp_path / "screenshot_fix.png").write_bytes(_PNG_1x1)
    user = "look at screen_mine2.png, the blocks look transparent"
    messages = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "browser",
                          "arguments": '{"op":"screenshot","out_path":"/workspace/screenshot_fix.png"}'}}
        ]},
        {"role": "tool", "name": "browser", "content": "saved to screenshot_fix.png"},
    ]
    before, after = _select_visual_evidence(messages, user, tmp_path)
    assert before == str(tmp_path / "screen_mine2.png")
    assert after == str(tmp_path / "screenshot_fix.png")


def test_select_visual_evidence_survives_injected_user_message(tmp_path):
    """Regression (found in live testing): the agent loop injects synthetic
    user-role messages mid-turn (planning nudges, '### ACTIVE STRATEGY'), so
    the LAST user message is often NOT the human's request and the agent's
    screenshot can sit BEFORE it. Evidence selection must still find the
    after-image by scanning all assistant/tool messages."""
    (tmp_path / "vtest_box.png").write_bytes(_PNG_1x1)
    user = "screenshot box.html — the circle should be blue not red"
    messages = [
        {"role": "user", "content": user},
        {"role": "assistant", "content":
            '<tool_call>\n<function name="browser">\n'
            '<parameter name="out_path">/workspace/vtest_box.png</parameter>\n'
            '</function>\n</tool_call>'},
        {"role": "tool", "name": "browser",
         "content": '{"path": "/workspace/vtest_box.png"}'},
        # synthetic user-role nudge injected AFTER the screenshot
        {"role": "user", "content": "### ACTIVE STRATEGY: Proceed directly to using a tool."},
    ]
    before, after = _select_visual_evidence(messages, user, tmp_path)
    assert before is None
    assert after == str(tmp_path / "vtest_box.png")


def test_select_visual_evidence_no_after_when_only_user_image(tmp_path):
    """If the agent only re-looked at the user's own screenshot and never
    rendered a new one, there is NO post-fix evidence — after must be None
    so we don't judge a stale frame as 'fixed'."""
    (tmp_path / "screen_mine2.png").write_bytes(_PNG_1x1)
    user = "fix screen_mine2.png, blocks are transparent"
    messages = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "vision_analysis",
                          "arguments": '{"action":"describe_picture","target":"/sandbox/screen_mine2.png"}'}}
        ]},
    ]
    before, after = _select_visual_evidence(messages, user, tmp_path)
    assert before == str(tmp_path / "screen_mine2.png")
    assert after is None
