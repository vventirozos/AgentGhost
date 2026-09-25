"""Every vision action runs without a thinking prelude, and an empty answer
that stopped on the token cap is named as such — 2026-09-24.

Live: five of six `describe_picture` calls on 2026-09-24 returned EMPTY.
llama-server's timing lines show each was a ~410-token prompt that generated
exactly 4096 tokens (the payload's `max_tokens`) in 44 s: the vision model
spent its whole budget inside <think>. The no-think switch existed but only
for `verify_ui`; the verifier's own no-think visual call on the same node
answered in 3.4 s. And the tool's error said "retry the same call once", so
every failure bought a second identical 44 s failure.
"""
from unittest.mock import AsyncMock

import pytest

import ghost_agent.tools.vision as vision_mod
from ghost_agent.tools.vision import tool_vision_analysis, _hit_token_cap

PNG_BYTES = (b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)


def _llm(content="a cat", finish_reason=None):
    llm = AsyncMock()
    choice = {"message": {"content": content}}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    llm.chat_completion = AsyncMock(return_value={"choices": [choice]})
    return llm


@pytest.mark.parametrize("action", ["describe_picture", "graph_analysis", "extract_text_picture", "weird_action"])
async def test_every_action_suppresses_thinking(tmp_path, action, monkeypatch):
    monkeypatch.setattr(vision_mod, "_VISION_NO_THINK", True)      # the flag is read at import; pin the world
    (tmp_path / "img.png").write_bytes(PNG_BYTES)
    llm = _llm()
    await tool_vision_analysis(action=action, target="img.png", llm_client=llm,
                               sandbox_dir=tmp_path, prompt="What is here?")
    payload = llm.chat_completion.await_args[0][0]
    text = payload["messages"][1]["content"][0]["text"]
    assert text.count("/no_think") == 1, text
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


async def test_flag_off_restores_thinking_for_captions(tmp_path, monkeypatch):
    monkeypatch.setattr(vision_mod, "_VISION_NO_THINK", False)
    (tmp_path / "img.png").write_bytes(PNG_BYTES)
    llm = _llm()
    await tool_vision_analysis(action="describe_picture", target="img.png",
                               llm_client=llm, sandbox_dir=tmp_path)
    payload = llm.chat_completion.await_args[0][0]
    assert "/no_think" not in payload["messages"][1]["content"][0]["text"]
    assert "chat_template_kwargs" not in payload


async def test_empty_answer_on_the_token_cap_says_do_not_retry(tmp_path):
    """The live shape: content "" with finish_reason=length. The result
    must carry the cap reason code and must NOT recommend a retry."""
    (tmp_path / "img.png").write_bytes(PNG_BYTES)
    llm = _llm(content="", finish_reason="length")
    out = await tool_vision_analysis(action="describe_picture", target="img.png",
                                     llm_client=llm, sandbox_dir=tmp_path)
    assert out.reason_code == "vision_thinking_cap"
    assert "Do NOT retry" in str(out)
    assert "Retry the same call once" not in str(out)


async def test_empty_answer_without_the_cap_keeps_the_one_retry(tmp_path):
    """Control: an empty answer that did NOT hit the cap is contention —
    the pre-existing advice (one retry) stands, with its own reason code."""
    (tmp_path / "img.png").write_bytes(PNG_BYTES)
    llm = _llm(content="", finish_reason="stop")
    out = await tool_vision_analysis(action="describe_picture", target="img.png",
                                     llm_client=llm, sandbox_dir=tmp_path)
    assert out.reason_code == "vision_empty_result" and "Retry the same call once" in str(out)


async def test_a_null_content_on_the_cap_is_the_cap_too(tmp_path):
    (tmp_path / "img.png").write_bytes(PNG_BYTES)
    llm = _llm(content=None, finish_reason="length")
    out = await tool_vision_analysis(action="describe_picture", target="img.png",
                                     llm_client=llm, sandbox_dir=tmp_path)
    assert out.reason_code == "vision_thinking_cap"


@pytest.mark.parametrize("resp,expect", [
    ({"choices": [{"finish_reason": "length"}]}, True),
    ({"choices": [{"finish_reason": "LENGTH "}]}, True),
    ({"choices": [{"finish_reason": "stop"}]}, False),
    ({"choices": [{}]}, False),
    ({"choices": []}, False),
    ({}, False),
    (None, False),
    ("garbage", False),
])
def test_hit_token_cap_reads_defensively(resp, expect):
    assert _hit_token_cap(resp) is expect


async def test_a_real_caption_still_ships_as_success(tmp_path):
    (tmp_path / "img.png").write_bytes(PNG_BYTES)
    llm = _llm(content="Two cats on a sofa.", finish_reason="stop")
    out = await tool_vision_analysis(action="describe_picture", target="img.png",
                                     llm_client=llm, sandbox_dir=tmp_path)
    assert out.startswith("VISION ANALYSIS RESULT") and "Two cats" in out



async def test_a_truncated_answer_with_content_says_so(tmp_path):
    """R4 review: finish_reason=length WITH content shipped as a clean
    success (OCR / PDF extracts at the cap)."""
    (tmp_path / "img.png").write_bytes(PNG_BYTES)
    llm = _llm(content="Page 1 text … Page 2 te", finish_reason="length")
    out = await tool_vision_analysis(action="extract_text_picture", target="img.png",
                                     llm_client=llm, sandbox_dir=tmp_path)
    assert "cut at the model's token cap" in str(out)
    llm2 = _llm(content="Complete text.", finish_reason="stop")
    out2 = await tool_vision_analysis(action="extract_text_picture", target="img.png",
                                      llm_client=llm2, sandbox_dir=tmp_path)
    assert "token cap" not in str(out2)
