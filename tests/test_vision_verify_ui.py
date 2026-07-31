"""vision_analysis action='verify_ui' — the verdict-shaped question channel.

Incident context (2026-07-31, pinball session, req 66d64313)
------------------------------------------------------------
While debugging its own pinball game the agent verified every wall-geometry
fix with a bare `describe_picture` screenshot round-trip. Five ~30s vision
calls returned generic captions; the agent had to infer "is the ball out of
the launcher channel?" from prose, misjudged it once, and the late verifier
REFUTED the turn because the coordinates the claim leaned on were "not found
in the vision" output — a caption structurally cannot carry them.

`verify_ui` is the fix on the vision side: the agent asks ONE specific
question and gets a JSON verdict {answer, confidence, evidence, details}
judged only from the pixels. The prompt is mandatory (there is nothing to
verify without a question), and thinking is suppressed the same way as the
verifier's visual gate (GHOST_VISUAL_NO_THINK) so a thinking vision model
can't burn the whole token budget on a <think> prelude and return an empty
content field.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from unittest.mock import AsyncMock

from ghost_agent.tools.vision import tool_vision_analysis

# Tiny valid PNG header + padding — enough for the magic-byte sniffer.
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64

_VERDICT = (
    '{"answer": "NO", "confidence": 0.9, '
    '"evidence": "the ball is visible inside the right-side channel", '
    '"details": "ball at approx x=370, y=150"}'
)


def _llm(content=_VERDICT):
    client = AsyncMock()
    client.chat_completion = AsyncMock(
        return_value={"choices": [{"message": {"content": content}}]})
    return client


async def test_verify_ui_requires_prompt(tmp_path):
    """No question → nothing to verify. Must fail fast with an instructive
    error (before any file I/O), not fall back to a generic caption."""
    (tmp_path / "img.png").write_bytes(PNG_BYTES)
    llm = _llm()
    out = await tool_vision_analysis(
        action="verify_ui", target="img.png",
        llm_client=llm, sandbox_dir=tmp_path)
    assert "SYSTEM ERROR" in out
    assert "verify_ui" in out and "prompt" in out
    llm.chat_completion.assert_not_awaited()


async def test_verify_ui_accepts_question_alias(tmp_path):
    """The prompt-alias healing (question/query/text/instruction) must
    satisfy verify_ui's mandatory-prompt check too."""
    (tmp_path / "img.png").write_bytes(PNG_BYTES)
    llm = _llm()
    out = await tool_vision_analysis(
        action="verify_ui", target="img.png",
        llm_client=llm, sandbox_dir=tmp_path,
        question="Is the ball in the play area?")
    assert "SYSTEM ERROR" not in out
    llm.chat_completion.assert_awaited()


async def test_verify_ui_builds_verdict_shaped_prompt(tmp_path):
    (tmp_path / "img.png").write_bytes(PNG_BYTES)
    llm = _llm()
    await tool_vision_analysis(
        action="verify_ui", target="img.png",
        llm_client=llm, sandbox_dir=tmp_path,
        prompt="Is the ball inside the main play area?")
    payload = llm.chat_completion.await_args[0][0]
    text = payload["messages"][1]["content"][0]["text"]
    # The agent's question is embedded verbatim...
    assert "Is the ball inside the main play area?" in text
    # ...inside a fixed JSON verdict schema.
    for key in ('"answer"', '"confidence"', '"evidence"', '"details"'):
        assert key in text
    # Judged from pixels, with an explicit no-guess escape hatch.
    assert "UNCERTAIN" in text
    # Auditor system prompt, not the generic vision one.
    assert "auditor" in payload["messages"][0]["content"].lower()


async def test_verify_ui_suppresses_thinking(tmp_path, monkeypatch):
    """A thinking vision model spends the entire max_tokens budget on the
    <think> prelude and returns EMPTY content (reproduced live 2026-07-31:
    finish_reason=length, 1024/1024 reasoning tokens, 0 content chars).
    verify_ui must ship both the /no_think soft-switch and the
    enable_thinking=False hard-switch. Flag pinned on the module attr —
    the env is read at import, so an operator shell exercising the
    documented GHOST_VISUAL_NO_THINK=0 kill must not turn this red."""
    import ghost_agent.tools.vision as vision_mod
    monkeypatch.setattr(vision_mod, "_VERIFY_UI_NO_THINK", True)
    (tmp_path / "img.png").write_bytes(PNG_BYTES)
    llm = _llm()
    await tool_vision_analysis(
        action="verify_ui", target="img.png",
        llm_client=llm, sandbox_dir=tmp_path, prompt="Is it red?")
    payload = llm.chat_completion.await_args[0][0]
    text = payload["messages"][1]["content"][0]["text"]
    assert "/no_think" in text
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


async def test_verify_ui_flag_off_restores_thinking(tmp_path, monkeypatch):
    import ghost_agent.tools.vision as vision_mod
    monkeypatch.setattr(vision_mod, "_VERIFY_UI_NO_THINK", False)
    (tmp_path / "img.png").write_bytes(PNG_BYTES)
    llm = _llm()
    await tool_vision_analysis(
        action="verify_ui", target="img.png",
        llm_client=llm, sandbox_dir=tmp_path, prompt="Is it red?")
    payload = llm.chat_completion.await_args[0][0]
    text = payload["messages"][1]["content"][0]["text"]
    assert "/no_think" not in text
    assert "chat_template_kwargs" not in payload


async def test_verify_ui_action_spelling_is_normalized(tmp_path):
    """'Verify-UI' / 'VERIFY_UI' must reach the verdict branch, not fall
    into the generic else and return a caption where a verdict was asked
    for — that silent degradation is the exact failure the action exists
    to eliminate."""
    (tmp_path / "img.png").write_bytes(PNG_BYTES)
    for spelling in ("Verify-UI", "VERIFY_UI", " verify_ui "):
        llm = _llm()
        out = await tool_vision_analysis(
            action=spelling, target="img.png",
            llm_client=llm, sandbox_dir=tmp_path, prompt="Is it red?")
        assert out.startswith("UI VERIFICATION RESULT"), spelling
        # And the mandatory-prompt gate fires for the near-miss too.
        llm2 = _llm()
        out2 = await tool_vision_analysis(
            action=spelling, target="img.png",
            llm_client=llm2, sandbox_dir=tmp_path)
        assert "SYSTEM ERROR" in out2 and "prompt" in out2, spelling


async def test_verify_ui_result_header_is_distinct(tmp_path):
    """The result banner tells the downstream verifier this is a pixel
    verdict, not a caption."""
    (tmp_path / "img.png").write_bytes(PNG_BYTES)
    llm = _llm()
    out = await tool_vision_analysis(
        action="verify_ui", target="img.png",
        llm_client=llm, sandbox_dir=tmp_path, prompt="Is it red?")
    assert out.startswith("UI VERIFICATION RESULT")
    assert _VERDICT in out


async def test_other_actions_unaffected_by_no_think(tmp_path):
    """describe_picture keeps its open-ended shape: no forced JSON schema,
    no thinking suppression — deep looking is allowed there."""
    (tmp_path / "img.png").write_bytes(PNG_BYTES)
    llm = _llm("a pinball table")
    out = await tool_vision_analysis(
        action="describe_picture", target="img.png",
        llm_client=llm, sandbox_dir=tmp_path)
    payload = llm.chat_completion.await_args[0][0]
    text = payload["messages"][1]["content"][0]["text"]
    assert "/no_think" not in text
    assert "chat_template_kwargs" not in payload
    assert out.startswith("VISION ANALYSIS RESULT")


async def test_bare_describe_picture_result_carries_verify_ui_tip(tmp_path):
    """Live retry finding (req 2c5ec4b5): the prompt-level steer loses to
    habit + auto-learned lessons embedding the old caption workflow. The
    moment-of-use steer — a TIP on the RESULT — lands mid-decision, same
    pattern as browser's PRE_INTERACTION line. Only for BARE
    describe_picture; a caller-supplied prompt already targeted the ask."""
    (tmp_path / "img.png").write_bytes(PNG_BYTES)
    out = await tool_vision_analysis(
        action="describe_picture", target="img.png",
        llm_client=_llm("a pinball table"), sandbox_dir=tmp_path)
    assert "verify_ui" in out and "TIP" in out
    # Targeted call → no tip.
    out2 = await tool_vision_analysis(
        action="describe_picture", target="img.png",
        llm_client=_llm("yes it renders"), sandbox_dir=tmp_path,
        prompt="Does the table render?")
    assert "TIP" not in out2
    # Verdict action → no tip either.
    out3 = await tool_vision_analysis(
        action="verify_ui", target="img.png",
        llm_client=_llm(_VERDICT), sandbox_dir=tmp_path, prompt="Renders?")
    assert "TIP" not in out3


async def test_empty_vision_result_is_an_error_not_silent_success(tmp_path):
    """Live retry finding (req 2c5ec4b5): a contended node returned ""
    under the success banner and the agent had to notice the emptiness
    itself (57s lost). Empty content must ship as a named error."""
    (tmp_path / "img.png").write_bytes(PNG_BYTES)
    out = await tool_vision_analysis(
        action="describe_picture", target="img.png",
        llm_client=_llm(""), sandbox_dir=tmp_path)
    assert "EMPTY" in out and out.startswith("Vision API Error")
    assert not out.startswith("VISION ANALYSIS RESULT")


def test_verify_ui_steer_present_in_both_prompt_personas():
    """Live-log finding (2026-07-31 functional probes): screenshot-verify
    turns run under the SPECIALIST persona ('Ghost Specialist Activated'),
    which had no verify_ui steer — so the agent kept reaching for bare
    describe_picture there. Both personas must carry the steer, and the
    specialist (which owns the browser guidance) must steer evaluate too."""
    from ghost_agent.core.prompts import SYSTEM_PROMPT, SPECIALIST_SYSTEM_PROMPT
    assert "verify_ui" in SYSTEM_PROMPT
    assert "verify_ui" in SPECIALIST_SYSTEM_PROMPT
    assert '"action":"evaluate"' in SPECIALIST_SYSTEM_PROMPT


async def test_verify_ui_exposed_in_registry():
    """The enum + descriptions are what the LLM sees. verify_ui must be in
    the action enum, and the prompt param must say it is required there —
    otherwise the agent keeps reaching for bare describe_picture."""
    from ghost_agent.tools.registry import get_active_tool_definitions

    tools = get_active_tool_definitions(None)
    vision_defs = [t for t in tools
                   if t.get("function", {}).get("name") == "vision_analysis"]
    assert vision_defs, "vision_analysis must be advertised"
    fn = vision_defs[0]["function"]
    assert "verify_ui" in fn["parameters"]["properties"]["action"]["enum"]
    assert "verify_ui" in fn["parameters"]["properties"]["prompt"]["description"]
