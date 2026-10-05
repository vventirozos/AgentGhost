"""§4LM — image understanding: behaviour pins for the review's fixes.

Each test drives the real function with inputs shaped like the failure the
review found (live probe V5, traffic rows #18/#49, reviewer probes)."""
from __future__ import annotations

import base64
import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image, ImageDraw

from ghost_agent.tools import vision as vision_mod
from ghost_agent.tools.vision import tool_vision_analysis, _normalize_for_node


def _png(img) -> bytes:
    b = io.BytesIO()
    img.save(b, "PNG")
    return b.getvalue()


def _transparent_text_png(fill=(0, 0, 0, 255)) -> bytes:
    im = Image.new("RGBA", (200, 80), (0, 0, 0, 0))
    ImageDraw.Draw(im).rectangle((20, 20, 120, 60), fill=fill)
    return _png(im)


def _llm(content="a caption"):
    llm = SimpleNamespace()
    llm.chat_completion = AsyncMock(return_value={"choices": [{"message": {"content": content}}]})
    return llm


def _sent_image(llm) -> Image.Image:
    payload = llm.chat_completion.await_args[0][0]
    url = [c for c in payload["messages"][1]["content"] if c.get("type") == "image_url"][0]["image_url"]["url"]
    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")


# ── B: transparency is composited, not dropped (live probe V5: "solid black") ──

async def test_transparent_png_reaches_the_node_opaque_with_its_text_visible(tmp_path):
    (tmp_path / "t.png").write_bytes(_transparent_text_png())
    llm = _llm()
    await tool_vision_analysis(action="extract_text_picture", target="t.png", llm_client=llm, sandbox_dir=tmp_path)
    img = _sent_image(llm)
    assert img.getpixel((5, 5)) == (255, 255, 255)        # background no longer black
    assert img.getpixel((50, 40)) == (0, 0, 0)            # the dark content stays dark


def test_light_content_on_transparency_gets_a_dark_background():
    mime, out = _normalize_for_node("image/png", _transparent_text_png(fill=(255, 255, 255, 255)))
    img = Image.open(io.BytesIO(out)).convert("RGB")
    assert mime == "image/png"
    assert img.getpixel((5, 5)) != (255, 255, 255) and img.getpixel((50, 40)) == (255, 255, 255)


def test_opaque_png_and_jpeg_ship_byte_identical():
    png = _png(Image.new("RGB", (10, 10), "red"))
    j = io.BytesIO()
    Image.new("RGB", (10, 10), "red").save(j, "JPEG")
    assert _normalize_for_node("image/png", png) == ("image/png", png)
    assert _normalize_for_node("image/jpeg", j.getvalue()) == ("image/jpeg", j.getvalue())


async def test_verifier_vision_call_flattens_alpha_and_types_from_bytes(tmp_path):
    from ghost_agent.core.verifier import Verifier

    class _Stub:
        last = None

        async def chat_completion(self, payload, use_vision=False):
            _Stub.last = payload
            return {"choices": [{"message": {"content": '{"verdict":"CONFIRMED","confidence":0.8}'}}]}
    p = tmp_path / "vision_0123456789ab.jpg"            # PNG bytes under a .jpg name
    p.write_bytes(_transparent_text_png())
    await Verifier(llm_client=_Stub()).verify_visual(symptom="x", claim="y", after_image=str(p))
    url = [c for c in _Stub.last["messages"][1]["content"] if c.get("type") == "image_url"][0]["image_url"]["url"]
    assert url.startswith("data:image/png;")
    img = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")
    assert img.getpixel((5, 5)) == (255, 255, 255)


# ── C: a caption that QUOTES an error is not a failed call ──

@pytest.mark.parametrize("caption", [
    "Traceback (most recent call last):\n  File \"app.py\", line 3",
    "A terminal card reading 'EXIT CODE: 2' in red.",
    "A retro CRT poster with the words SYSTEM ERROR in green.",
    "Error: 404 — the page could not be found.",
])
async def test_a_caption_of_an_error_is_a_declared_success(tmp_path, caption):
    from ghost_agent.distill.outcome_heuristics import _looks_like_tool_error
    from ghost_agent.core.agent import _action_failed
    (tmp_path / "s.png").write_bytes(_png(Image.new("RGB", (8, 8), "white")))
    res = await tool_vision_analysis(action="describe_picture", target="s.png", llm_client=_llm(caption),
                                     sandbox_dir=tmp_path, prompt="what does it say?")
    assert res.declared and not res.is_failure and not res.exit_code_failed
    assert not _looks_like_tool_error(res, "vision_analysis")
    assert not _looks_like_tool_error(str(res), "vision_analysis")      # a rehydrated JSONL row
    assert not _action_failed(str(res), "vision_analysis")


def test_a_real_vision_failure_is_still_a_failure():
    from ghost_agent.distill.outcome_heuristics import _looks_like_tool_error
    assert _looks_like_tool_error("Error: File 'x.png' not found. Use the `file_system` tool", "vision_analysis")
    assert _looks_like_tool_error("SYSTEM ERROR: The 'action' and 'target' parameters are MANDATORY.",
                                  "vision_analysis")


# ── D: the visual verifier's evidence ──

def _data_url(raw: bytes, mime="image/png") -> str:
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


def test_a_pasted_image_is_the_users_before_never_the_after(tmp_path):
    from ghost_agent.core.agent import _select_visual_evidence, _attachment_filename
    raw = _png(Image.new("RGB", (8, 8), "blue"))
    name = _attachment_filename(raw, "data:image/png;base64")
    (tmp_path / name).write_bytes(raw)
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "the button overlaps the footer, fix it"},
                                     {"type": "image_url", "image_url": {"url": _data_url(raw)}}]},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {
            "name": "vision_analysis", "arguments": json.dumps({"action": "describe_picture", "target": name})}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "VISION ANALYSIS RESULT:\nA login page."},
    ]
    before, after = _select_visual_evidence(msgs, "the button overlaps the footer, fix it", tmp_path)
    assert before == str(tmp_path / name)
    assert after is None                                  # no render → nothing to judge


def test_an_image_from_another_project_is_never_resolved(tmp_path):
    from ghost_agent.core.agent import _resolve_image_path
    (tmp_path / "projects" / "p1").mkdir(parents=True)
    (tmp_path / "projects" / "other").mkdir(parents=True)
    (tmp_path / "projects" / "other" / "chart.png").write_bytes(b"x")
    assert _resolve_image_path("chart.png", tmp_path / "projects" / "p1") is None
    (tmp_path / "shots").mkdir()
    (tmp_path / "shots" / "chart.png").write_bytes(b"x")          # the root (shared) still resolves
    assert _resolve_image_path("chart.png", tmp_path / "projects" / "p1") == str(tmp_path / "shots" / "chart.png")


@pytest.mark.parametrize("raw,header,ext", [
    (_png(Image.new("RGB", (4, 4))), "data:image/png;base64", "png"),
    (b"\xff\xd8\xff\xe0rest", "data:image/jpeg;base64", "jpg"),
    (b"<svg xmlns='http://www.w3.org/2000/svg'/>", "data:image/svg+xml;base64", "svg"),
    (b"???", "", "jpg"),
    (_png(Image.new("RGB", (4, 4))), "data:image/jpeg;base64", "png"),     # the bytes outrank the header
])
def test_a_pasted_image_is_named_by_its_type(raw, header, ext):
    from ghost_agent.core.agent import _attachment_filename, _ATTACHMENT_NAME_RE
    name = _attachment_filename(raw, header)
    assert name.endswith("." + ext) and _ATTACHMENT_NAME_RE.match(name)


# ── A: no cleartext DNS for a URL fetched over Tor ──

async def test_vision_url_guard_does_not_resolve_over_tor(monkeypatch):
    import ghost_agent.utils.helpers as helpers
    seen = []

    def _rec(url, *, resolve=True):
        seen.append(resolve)
        return "blocked for the test"
    monkeypatch.setattr(helpers, "url_ssrf_reason", _rec)
    res = await tool_vision_analysis(action="describe_picture", target="https://example.org/a.png",
                                     llm_client=_llm(), sandbox_dir=None, tor_proxy="socks5://127.0.0.1:9050")
    assert seen == [False] and "blocked for the test" in res


async def test_download_url_guard_does_not_resolve_over_tor(monkeypatch, tmp_path):
    import ghost_agent.utils.helpers as helpers
    from ghost_agent.tools.file_system import tool_download_file
    seen = []

    def _rec(url, *, resolve=True):
        seen.append(resolve)
        return "blocked for the test"
    monkeypatch.setattr(helpers, "url_ssrf_reason", _rec)
    res = await tool_download_file("https://example.org/a.png", tmp_path, "socks5://127.0.0.1:9050")
    assert seen == [False] and "blocked for the test" in res


async def test_vision_redirect_hop_is_checked_without_resolving(monkeypatch):
    import ghost_agent.utils.helpers as helpers
    calls = []

    def _rec(url, *, resolve=True):
        calls.append((url, resolve))
        return "hop refused" if "evil" in url else None
    monkeypatch.setattr(helpers, "url_ssrf_reason", _rec)

    class _Resp:
        status_code = 302
        headers = {"location": "https://evil.example/x.png"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def stream(self, method, url):
            return _Resp()
    monkeypatch.setattr(vision_mod.httpx, "AsyncClient", _Client)
    res = await tool_vision_analysis(action="describe_picture", target="https://example.org/a.png",
                                     llm_client=_llm(), sandbox_dir=None, tor_proxy="socks5://127.0.0.1:9050")
    assert ("https://evil.example/x.png", False) in calls and "refused" in str(res)


# ── L / undecodable / timeout ──

async def test_an_unknown_action_is_refused_before_any_call(tmp_path):
    llm = _llm()
    res = await tool_vision_analysis(action="identify_faces", target="x.png", llm_client=llm, sandbox_dir=tmp_path)
    assert "unknown vision_analysis action" in res and "extract_text_picture" in res
    llm.chat_completion.assert_not_awaited()


async def test_an_undecodable_format_is_refused_with_do_not_retry(tmp_path):
    (tmp_path / "logo.svg").write_bytes(b"<svg xmlns='http://www.w3.org/2000/svg'><rect/></svg>")
    llm = _llm()
    res = await tool_vision_analysis(action="describe_picture", target="logo.svg", llm_client=llm, sandbox_dir=tmp_path)
    assert "Do NOT retry" in res and "svg" in res
    llm.chat_completion.assert_not_awaited()


async def test_the_vision_call_is_bounded(tmp_path):
    (tmp_path / "a.png").write_bytes(_png(Image.new("RGB", (4, 4))))
    llm = _llm()
    await tool_vision_analysis(action="describe_picture", target="a.png", llm_client=llm, sandbox_dir=tmp_path)
    assert 0 < llm.chat_completion.await_args.kwargs["timeout"] <= 600


async def test_a_thinking_cap_on_an_attached_image_says_it_was_not_seen(tmp_path):
    (tmp_path / "a.png").write_bytes(_png(Image.new("RGB", (4, 4))))
    llm = SimpleNamespace(chat_completion=AsyncMock(return_value={
        "choices": [{"message": {"content": ""}, "finish_reason": "length"}]}))
    res = await tool_vision_analysis(action="describe_picture", target="a.png", llm_client=llm, sandbox_dir=tmp_path)
    assert res.is_failure and "Do NOT retry" in res and "NOT seen this image" in res
    assert "as-is" not in res


# ── F: invented download links; the blind-regeneration text ──

def test_a_link_to_a_file_that_does_not_exist_is_removed(tmp_path):
    from ghost_agent.core.agent import _drop_missing_download_links
    (tmp_path / "projects" / "p1").mkdir(parents=True)
    (tmp_path / "projects" / "p1" / "gen_aaaa1111.png").write_bytes(b"x")
    (tmp_path / "plot.png").write_bytes(b"x")
    text = ("Here:\n![generated image](/api/download/gen_d7f2a1b3.png)\n"
            "![a](/api/download/gen_aaaa1111.png) [p](/api/download/plot.png) "
            "/api/download/projects/p1/gen_aaaa1111.png")
    out, dropped = _drop_missing_download_links(text, tmp_path)
    assert dropped == ["gen_d7f2a1b3.png"]
    assert "gen_d7f2a1b3" in out and "/api/download/gen_d7f2a1b3" not in out
    assert out.count("/api/download/") == 3                 # every real file keeps its link


def test_the_blind_regeneration_block_does_not_invite_a_description():
    from ghost_agent.core.agent import _blind_regeneration_block
    rows = [{"name": "image_generation",
             "content": "SUCCESS: Image generated and saved to sandbox.\n![generated image](/api/download/gen_ab12cd34.png)"}]
    msg = _blind_regeneration_block("image_generation", rows, [], "draw a cat")
    assert "you have not seen it" in msg and "one-line description" not in msg


# ── G: the member wall accepts the forms the schema teaches ──

@pytest.mark.parametrize("value,ok", [
    ("gen_ab12cd34.png", True), ("/gen_ab12cd34.png", True), ("/api/download/gen_ab12cd34.png", True),
    ("/owner.png", False), ("//gen_ab12cd34.png", False), ("/api/download/../gen_ab12cd34.png", False),
    ("/api/download/projects/p/gen_ab12cd34.png", False), (" /gen_ab12cd34.png", False),
    ("https://example.org/a.png", True),
])
def test_member_vision_target_forms(value, ok):
    from ghost_agent.core.agent import GhostAgent
    stub = SimpleNamespace(_MEMBER_URL_RE=GhostAgent._MEMBER_URL_RE)
    assert GhostAgent._member_file_value_ok(stub, value, {"gen_ab12cd34.png"}) is ok


# ── H / K: advice after a failure ──

def test_no_file_system_fallback_after_a_vision_failure():
    from ghost_agent.tools.fallback_chains import get_fallback_chain_hint
    assert get_fallback_chain_hint("vision_analysis", "Vision API Error: EMPTY") is None


def test_a_url_404_gets_url_advice_and_a_missing_file_gets_sandbox_advice():
    from ghost_agent.tools.tool_failure import get_fallback_hint
    url = get_fallback_hint("vision_analysis",
                            "Vision API Error: Client error '404 Not Found' for url 'https://x/a.png'")
    local = get_fallback_hint("vision_analysis", "Error: File 'a.png' not found. Use the `file_system` tool")
    assert "URL returned 404" in url and "sandbox" not in url
    assert "sandbox" in local
    assert get_fallback_hint("vision_analysis", "Vision API Error: model 'eva' not found") is None


# ── J: the salvage parser keeps each field ──

def test_salvaged_vision_call_keeps_target_action_and_prompt():
    from ghost_agent.core.agent import _salvage_vision_args
    got = _salvage_vision_args("<prompt>Is the ball left of the wall?</prompt>"
                               "<target>shot.png</target><action>verify_ui</action>")
    assert got == {"target": "shot.png", "action": "verify_ui", "prompt": "Is the ball left of the wall?"}
    assert _salvage_vision_args("<prompt>only a question</prompt>") == {}
    assert _salvage_vision_args('{"prompt": "what is it?", "target": "a.png"}') == {"target": "a.png",
                                                                                   "prompt": "what is it?"}


async def test_a_verify_ui_verdict_quoting_an_error_is_a_declared_success(tmp_path):
    (tmp_path / "s.png").write_bytes(_png(Image.new("RGB", (8, 8), "white")))
    res = await tool_vision_analysis(
        action="verify_ui", target="s.png", sandbox_dir=tmp_path, prompt="Is the build green?",
        llm_client=_llm('{"answer": "NO", "evidence": "the console shows EXIT CODE: 2 and SYSTEM ERROR"}'))
    assert res.declared and not res.is_failure and not res.exit_code_failed


def test_the_failure_banner_skips_a_declared_success_only():
    from ghost_agent.core.agent import _result_failure_shaped
    from ghost_agent.tools.outcome import ToolOutcome
    cap = "VISION ANALYSIS RESULT:\nA poster reading SYSTEM ERROR."
    assert not _result_failure_shaped(cap, False, ToolOutcome.ok(cap))
    assert _result_failure_shaped(cap, False, ToolOutcome.coerce(cap))          # undeclared text keeps the rule
    assert _result_failure_shaped("Error: x", False, ToolOutcome.ok("Error: x"))


def test_a_pasted_image_is_never_the_after_even_when_the_user_names_another_file(tmp_path):
    from ghost_agent.core.agent import _select_visual_evidence, _attachment_filename
    raw = _png(Image.new("RGB", (8, 8), "blue"))
    name = _attachment_filename(raw, "data:image/png;base64")
    (tmp_path / name).write_bytes(raw)
    (tmp_path / "layout.png").write_bytes(_png(Image.new("RGB", (8, 8), "red")))
    text = "compare with layout.png — the footer overlaps"
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": text},
                                     {"type": "image_url", "image_url": {"url": _data_url(raw)}}]},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {
            "name": "vision_analysis", "arguments": json.dumps({"action": "describe_picture", "target": name})}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "VISION ANALYSIS RESULT:\nA page."},
    ]
    before, after = _select_visual_evidence(msgs, text, tmp_path)
    assert before == str(tmp_path / "layout.png") and after is None


def test_a_malformed_vision_call_is_salvaged_with_every_field():
    from tests.test_4ec_parser_dialects import _agent
    calls = _agent()._parse_assistant_tool_calls(
        "<tool_call>\n<name>vision_analysis</name><prompt>Is the ball left of the wall?</prompt>"
        "<target>shot.png</target><action>verify_ui</action>\n</tool_call>", {})[0]
    args = json.loads(calls[0]["function"]["arguments"])
    assert args == {"target": "shot.png", "action": "verify_ui", "prompt": "Is the ball left of the wall?"}


async def test_finalize_removes_an_invented_image_link_for_the_owner(tmp_path):
    from unittest.mock import MagicMock
    from tests.test_finalize_extraction import _make_agent, _fs
    agent = _make_agent()
    agent.context.sandbox_dir = tmp_path
    (tmp_path / "real.png").write_bytes(b"x")
    out, _, _ = await agent._finalize_and_return(_fs(final_ai_content=(
        "Here it is:\n![generated image](/api/download/gen_d7f2a1b3.png)\n![p](/api/download/real.png)")))
    assert "/api/download/gen_d7f2a1b3.png" not in out and "/api/download/real.png" in out


async def test_the_loop_gives_no_failure_banner_to_a_caption_quoting_system_error(monkeypatch, tmp_path):
    from unittest.mock import AsyncMock
    from tests.test_requester_role import _agent as _loop_agent, _tc
    from tests.test_4kl_member_capability import _resp, FakeBgTasks
    from ghost_agent.tools.outcome import ToolOutcome
    agent, ctx, _ = _loop_agent(monkeypatch, tmp_path)
    cap = ToolOutcome.ok("VISION ANALYSIS RESULT:\nA retro poster reading SYSTEM ERROR in green.", world_changed=False)
    agent.available_tools = {"vision_analysis": AsyncMock(return_value=cap)}
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[
        _resp("", [_tc("c0", "vision_analysis", {"action": "describe_picture", "target": "poster.png"})]),
        _resp("It reads SYSTEM ERROR."), _resp("It reads SYSTEM ERROR.")])
    await agent.handle_chat({"messages": [{"role": "user", "content": "what does poster.png say?"}]},
                            FakeBgTasks(), request_id="web-4lm-b")
    second = ctx.llm_client.chat_completion.call_args_list[1]
    payload = second.args[0] if second.args and isinstance(second.args[0], dict) else second.kwargs
    text = "\n".join(str(m.get("content")) for m in payload["messages"])
    assert "A retro poster reading SYSTEM ERROR" in text
    assert "[FAILURE BANNER]" not in text


async def test_an_action_alias_runs_the_action_it_names(tmp_path):
    (tmp_path / "a.png").write_bytes(_png(Image.new("RGB", (4, 4))))
    llm = _llm("HELLO")
    await tool_vision_analysis(action="ocr", target="a.png", llm_client=llm, sandbox_dir=tmp_path)
    sent = llm.chat_completion.await_args[0][0]["messages"][1]["content"][0]["text"]
    assert sent.startswith("Extract all text from this image exactly as written")


# ── fresh-eye review of the §4LM diff ──

@pytest.mark.parametrize("link", [
    "see /api/download/plot.png.", "(chart at /api/download/plot.png)", "| /api/download/plot.png|",
    "`/api/download/plot.png`", "![x](/api/download/my%20plot.png)", "![x](/api/download/shot[1].png)",
])
def test_a_real_file_keeps_its_link_whatever_punctuation_follows(tmp_path, link):
    from ghost_agent.core.agent import _drop_missing_download_links
    (tmp_path / "sub").mkdir()
    for n in ("plot.png", "my plot.png", "shot[1].png"):
        (tmp_path / "sub" / n).write_bytes(b"x")          # found by NAME, not at the link's path
    out, dropped = _drop_missing_download_links(link, tmp_path)
    assert dropped == [] and out == link


def test_a_removed_bare_link_keeps_the_sentences_parenthesis(tmp_path):
    from ghost_agent.core.agent import _drop_missing_download_links
    out, dropped = _drop_missing_download_links("(saved at /api/download/ghost.png)", tmp_path)
    assert dropped == ["ghost.png"] and out.endswith(")") and out.startswith("(")


def test_an_image_pasted_in_an_earlier_request_is_not_this_requests_before(tmp_path):
    from ghost_agent.core.agent import _select_visual_evidence, _attachment_filename
    raw = _png(Image.new("RGB", (8, 8), "blue"))
    (tmp_path / _attachment_filename(raw, "")).write_bytes(raw)
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "what is this?"},
                                     {"type": "image_url", "image_url": {"url": _data_url(raw)}}]},
        {"role": "assistant", "content": "A blue square."},
        {"role": "user", "content": "now draw me a sunset, the layout should look clean"},
    ]
    before, _ = _select_visual_evidence(msgs, "now draw me a sunset, the layout should look clean", tmp_path)
    assert before is None


async def test_the_verifier_makes_no_call_when_one_of_its_images_is_unusable(tmp_path):
    from ghost_agent.core.verifier import Verifier

    class _Stub:
        calls = 0

        async def chat_completion(self, payload, use_vision=False):
            _Stub.calls += 1
            return {"choices": [{"message": {"content": '{"verdict":"REFUTED","confidence":0.9}'}}]}
    (tmp_path / "before.svg").write_bytes(b"<svg xmlns='http://www.w3.org/2000/svg'/>")
    (tmp_path / "after.png").write_bytes(_png(Image.new("RGB", (4, 4))))
    res = await Verifier(llm_client=_Stub()).verify_visual(
        symptom="x", claim="y", after_image=str(tmp_path / "after.png"), before_image=str(tmp_path / "before.svg"))
    assert res is None and _Stub.calls == 0


def test_stored_file_reads_answer_as_the_live_loop_did():
    from ghost_agent.core.agent import _action_failed
    assert _action_failed("--- job ---\nEXIT CODE: 1", "file_system") is True        # the loop's banner rule
    assert _action_failed("VISION ANALYSIS RESULT:\nA card: EXIT CODE: 1", "vision_analysis") is False


def test_no_active_project_still_never_reads_a_projects_file(tmp_path):
    from ghost_agent.core.agent import _resolve_image_path
    (tmp_path / "projects" / "other").mkdir(parents=True)
    (tmp_path / "projects" / "other" / "chart.png").write_bytes(b"x")
    assert _resolve_image_path("chart.png", tmp_path) is None


def test_a_salvaged_prompt_keeps_its_apostrophe():
    from ghost_agent.core.agent import _salvage_vision_args
    got = _salvage_vision_args('{"target": "a.png", "prompt": "Is the user\'s avatar visible?"}')
    assert got["prompt"] == "Is the user's avatar visible?"


def test_a_fully_opaque_rgba_png_ships_unchanged():
    raw = _png(Image.new("RGBA", (3000, 2000), (10, 20, 30, 255)))
    assert _normalize_for_node("image/png", raw) == ("image/png", raw)


def test_a_paste_is_found_behind_a_synthetic_mid_turn_user_row(tmp_path):
    from ghost_agent.core.agent import _select_visual_evidence, _attachment_filename
    raw = _png(Image.new("RGB", (8, 8), "blue"))
    name = _attachment_filename(raw, "")
    (tmp_path / name).write_bytes(raw)
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "the footer overlaps, fix it"},
                                     {"type": "image_url", "image_url": {"url": _data_url(raw)}}]},
        {"role": "assistant", "content": "Looking."},
        {"role": "user", "content": "### ACTIVE STRATEGY\nReproduce first."},      # injected by the loop
    ]
    before, _ = _select_visual_evidence(msgs, "the footer overlaps, fix it", tmp_path)
    assert before == str(tmp_path / name)
