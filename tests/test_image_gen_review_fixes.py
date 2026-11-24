"""§4QG (2026-10-21) — image_generation review fixes: pins for each.

1. transparency is decided from the PIXELS (the node writes RGBA always);
2. the deadline estimate covers the collage ladder of a ≥2-panel call, and
   a singular `subject` beside a prompt is an edit-kind render;
3. no sandbox → refused BEFORE the node renders;
4. size is validated like width/height and a dropped size is SAID;
5. collage attempt 3 reuses the FIRST render's seed at cfg 4.

§4QG-b (fresh-eye review): the estimate is the TYPICAL ladder (2 renders +
one verdict) on Turbo defaults; the cfg-4 leg books under its own key; the
ladder skips a re-render that would cross the report reserve.
"""
import base64
import io

import pytest
from PIL import Image

import ghost_agent.tools.image_gen as IG
from ghost_agent.tools.image_gen import tool_generate_image
from tests.test_4oz_subjects_collage import _llm, _resp, _sent, web, YES, NO, TWO, PNG  # noqa: F401 — fixture


def _png_bytes(mode="RGBA", bg=(0, 0, 0, 0), subject_box=(16, 16, 48, 48), faint=0) -> bytes:
    im = Image.new(mode, (64, 64), bg)
    if subject_box:
        im.paste((200, 30, 30, 255), subject_box)
    for i in range(faint):                  # a few see-through pixels in an opaque picture
        im.putpixel((i % 64, i // 64), (0, 0, 0, 0))
    buf = io.BytesIO(); im.save(buf, format="PNG"); return buf.getvalue()


def _plain_llm(png: bytes):
    llm = _llm()
    llm.generate_image.return_value = {"data": [{"b64_json": base64.b64encode(png).decode()}],
                                       "width": 64, "height": 64, "seed": 3}
    return llm


# ── 1. transparency from the pixels ──

async def test_an_rgba_but_opaque_render_gets_the_opaque_warning(tmp_path):
    png = _png_bytes(bg=(240, 240, 240, 250), subject_box=None)
    assert png[25] == 6                                            # RGBA, as the node writes every picture
    out = await tool_generate_image(prompt="a sticker", transparent=True, llm_client=_plain_llm(png), sandbox_dir=tmp_path)
    assert "OPAQUE image although a transparent background was asked" in out
    assert "really transparent" not in out


async def test_an_rgba_cutout_is_transparent(tmp_path):
    out = await tool_generate_image(prompt="a sticker", transparent=True,
                                    llm_client=_plain_llm(_png_bytes()), sandbox_dir=tmp_path)
    assert "the background is really transparent" in out and "OPAQUE" not in out


def test_a_few_clear_pixels_are_not_a_transparent_background():
    # 100 of 4096 pixels (2.4%) clear — under the 5% share
    assert not IG._png_has_alpha(_png_bytes(bg=(240, 240, 240, 255), subject_box=None, faint=100))
    assert IG._png_has_alpha(_png_bytes(bg=(240, 240, 240, 255), subject_box=None, faint=400))


# ── 2. the deadline estimate covers the ladder ──

@pytest.fixture
def timings(monkeypatch):
    for d in (*IG.RECENT_RENDER_SECONDS.values(), IG.RECENT_VERDICT_SECONDS):
        d.clear()
    IG.RECENT_RENDER_SECONDS["edit"].extend([150.0, 200.0])
    IG.RECENT_VERDICT_SECONDS.extend([10.0, 20.0])
    monkeypatch.setattr(IG, "COLLAGE_RETRIES", 2)
    yield
    for d in (*IG.RECENT_RENDER_SECONDS.values(), IG.RECENT_VERDICT_SECONDS):
        d.clear()


@pytest.fixture
def empty_history(monkeypatch):
    for d in (*IG.RECENT_RENDER_SECONDS.values(), IG.RECENT_VERDICT_SECONDS):
        d.clear()
    monkeypatch.setattr(IG, "COLLAGE_RETRIES", 2)
    yield
    for d in (*IG.RECENT_RENDER_SECONDS.values(), IG.RECENT_VERDICT_SECONDS):
        d.clear()


def test_two_subjects_are_estimated_as_the_typical_ladder(timings):
    # §4QG-b: two renders + one verdict (the worst ladder is guarded by the
    # ladder's own pre-leg deadline check, not refused up front)
    want = 200.0 * 2 + 20.0
    assert IG.expected_render_seconds({"prompt": "k", "subjects": TWO}) == pytest.approx(want)
    # subjects + an uploaded/existing picture is two panels too
    assert IG.expected_render_seconds({"prompt": "k", "subjects": ["A"], "reference_images": ["me.jpg"]}) \
        == pytest.approx(want)
    # one panel: one render
    assert IG.expected_render_seconds({"prompt": "k", "subjects": ["A"]}) == 200.0
    assert IG.expected_render_seconds({"prompt": "k", "reference_images": ["me.jpg"]}) == 200.0


def test_the_estimate_follows_the_retry_switch(timings, monkeypatch):
    monkeypatch.setattr(IG, "COLLAGE_RETRIES", 1)
    assert IG.expected_render_seconds({"prompt": "k", "subjects": TWO}) == pytest.approx(200.0 * 2 + 20.0)
    monkeypatch.setattr(IG, "COLLAGE_RETRIES", 0)
    assert IG.expected_render_seconds({"prompt": "k", "subjects": TWO}) == 200.0


def test_the_agents_wait_reads_the_ladder_estimate(timings):
    from ghost_agent.core.agent import declared_wait_s
    assert declared_wait_s("image_generation", {"prompt": "k", "subjects": TWO}) == pytest.approx(2 * 200.0 + 20.0)


def test_a_singular_subject_beside_a_prompt_is_an_edit_kind_render(timings):
    assert IG.render_kind({"prompt": "k", "subject": "Alexis Tsipras"}) == "edit"
    assert IG.render_kind({"subject": "a red apple"}) == "generate"            # no prompt: it IS the prompt
    assert IG.render_kind({"prompt": "a red apple", "subject": "a red apple"}) == "generate"
    assert IG.expected_render_seconds({"prompt": "k", "subject": "Alexis Tsipras and Antonis Samaras"}) \
        == pytest.approx(2 * 200.0 + 20.0)


async def test_collage_verdicts_are_timed(tmp_path, web):
    IG.RECENT_VERDICT_SECONDS.clear()
    await tool_generate_image(prompt="k", subjects=TWO, llm_client=_llm(verdicts=[NO]), sandbox_dir=tmp_path)
    assert len(IG.RECENT_VERDICT_SECONDS) == 1
    IG.RECENT_VERDICT_SECONDS.clear()


# ── 3. no sandbox: refused before the render ──

async def test_no_sandbox_is_refused_before_the_node_renders():
    llm = _llm()
    out = await tool_generate_image(prompt="a red apple", llm_client=llm, sandbox_dir=None)
    assert out.startswith("ERROR: no sandbox") and "Nothing was rendered" in out
    llm.generate_image.assert_not_called()


# ── 4. size validation ──

@pytest.mark.parametrize("args", [{"size": "-5x-5"}, {"width": 1024}, {"size": "1024px"},
                                  {"width": -5, "height": 512}, {"size": "largexhuge"}])
async def test_an_unusable_size_is_not_sent_and_is_said(tmp_path, args):
    llm = _llm()
    out = await tool_generate_image(prompt="k", llm_client=llm, sandbox_dir=tmp_path, **args)
    assert "width" not in _sent(llm) and "height" not in _sent(llm)
    assert "NOTE: the requested size (" in out and "was not usable" in out


@pytest.mark.parametrize("args", [{"width": "1024px", "height": "768px"}, {"size": "1024 x 768"},
                                  {"size": "1024×768"}, {"dimensions": [1024, 768]}])
async def test_a_usable_size_is_sent_without_a_note(tmp_path, args):
    llm = _llm()
    out = await tool_generate_image(prompt="k", llm_client=llm, sandbox_dir=tmp_path, **args)
    assert (_sent(llm)["width"], _sent(llm)["height"]) == (1024, 768)
    assert "was not usable" not in out


async def test_no_size_at_all_has_no_note(tmp_path):
    llm = _llm()
    out = await tool_generate_image(prompt="k", llm_client=llm, sandbox_dir=tmp_path)
    assert "width" not in _sent(llm) and "was not usable" not in out


# ── 5. the cfg-4 leg reuses the first render's seed ──

async def test_attempt_three_reuses_the_seed_the_node_reported(tmp_path, web, monkeypatch):
    monkeypatch.setattr(IG, "COLLAGE_CFG_LEG", True)        # §4VJ: the cfg-4 leg is opt-in
    llm = _llm(verdicts=[YES, YES, NO],
               renders=[_resp(seed=11, png=PNG + b"1"), _resp(seed=22, png=PNG + b"2"), _resp(seed=33, png=PNG + b"3")])
    await tool_generate_image(prompt="k", subjects=TWO, llm_client=llm, sandbox_dir=tmp_path)   # no seed sent
    assert "seed" not in _sent(llm, 0) and "seed" not in _sent(llm, 1)
    assert _sent(llm, 2)["seed"] == 11 and _sent(llm, 2)["guidance_scale"] == IG.COLLAGE_RETRY_GUIDANCE


# ── §4QG-b: C1 — the deadline estimate after a restart, the cfg leg, the ladder ──

def test_an_empty_history_two_subject_call_fits_an_1800s_budget(empty_history):
    """[CRITICAL §4QG-b] empty history: 600 × 3.8 + 90 = 2370 s > 1800 s, so
    every multi-subject call after a restart was refused by agent.py's
    deadline preflight. Called through agent.py's own functions."""
    from ghost_agent.core.agent import declared_wait_s, wait_crosses_deadline, effective_report_floor
    for args in ({"prompt": "two at a cafe", "subjects": TWO},
                 {"prompt": "two at a cafe", "subject": "Alexis Tsipras and Antonis Samaras"},
                 {"prompt": "x", "subjects": ["Alexis Tsipras"], "reference_images": ["me.jpg"]}):
        w = declared_wait_s("image_generation", args)
        assert w == pytest.approx(2 * IG.DEFAULT_RENDER_SECONDS["edit"] + IG.COLLAGE_VERDICT_DEFAULT_S)
        assert not wait_crosses_deadline(w, 1800.0, effective_report_floor(1800.0))
    # §4VJ: Turbo at 12 steps, 1152x768 (T2I 245 s, edits 375 s) — still inside 1800 s
    assert IG.DEFAULT_RENDER_SECONDS == {"edit": 375.0, "generate": 245.0}


async def test_the_cfg_leg_does_not_inflate_the_next_estimate(tmp_path, web, empty_history, monkeypatch):
    """[CRITICAL §4QG-b] the cfg-4 leg was booked into "edit" and then
    multiplied by COLLAGE_CFG_COST again in the next estimate."""
    monkeypatch.setattr(IG, "COLLAGE_CFG_LEG", True)        # §4VJ: the cfg-4 leg is opt-in
    clock = {"t": 0.0}
    monkeypatch.setattr(IG, "time", __import__("types").SimpleNamespace(monotonic=lambda: clock["t"]))
    durations = iter([100.0, 110.0, 230.0])

    async def render(payload):
        clock["t"] += next(durations)
        return _resp()
    llm = _llm(verdicts=[YES, YES, NO])
    llm.generate_image.side_effect = render
    monkeypatch.setattr(IG, "_request_time_left", lambda: None)
    await tool_generate_image(prompt="k", subjects=TWO, llm_client=llm, sandbox_dir=tmp_path)
    assert list(IG.RECENT_RENDER_SECONDS["edit"]) == [100.0, 110.0]
    assert list(IG.RECENT_RENDER_SECONDS["edit_cfg"]) == [230.0]
    v = IG._verdict_seconds()
    assert IG.expected_render_seconds({"prompt": "k", "subjects": TWO}) == pytest.approx(110.0 * 2 + v)
    assert IG.expected_render_seconds({"prompt": "k", "subjects": ["A"]}) == 110.0


@pytest.mark.parametrize("left,renders,note", [
    (50.0, 1, True),          # not even the reseed leg fits
    (200.0, 2, True),         # the reseed fits (150 + 0 + 30), the cfg leg (150 × 1.8 + 0 + 30) does not
    (1000.0, 3, False),       # the whole ladder fits
    (None, 3, False),         # no deadline known: as before
])
async def test_the_ladder_skips_a_retry_that_would_cross_the_report_reserve(
        tmp_path, web, empty_history, monkeypatch, left, renders, note):
    """[CRITICAL §4QG-b (d)] the ladder is deadline-aware: before a re-render,
    the request's time left (before the report reserve) must cover the leg
    + its verdict + COLLAGE_LEG_MARGIN_S; otherwise the picture is kept."""
    monkeypatch.setattr(IG, "COLLAGE_CFG_LEG", True)        # §4VJ: the cfg-leg cost row needs the leg on
    monkeypatch.setattr(IG, "_request_time_left", lambda: left)
    clock = {"t": 0.0}                     # each render takes 150 s, a verdict 0 s
    monkeypatch.setattr(IG, "time", __import__("types").SimpleNamespace(monotonic=lambda: clock["t"]))

    async def render(payload):
        clock["t"] += 150.0
        return _resp()
    llm = _llm(verdicts=[YES, YES, YES])
    llm.generate_image.side_effect = render
    out = await tool_generate_image(prompt="k", subjects=TWO, llm_client=llm, sandbox_dir=tmp_path)
    assert out.startswith("SUCCESS"), out
    assert llm.generate_image.await_count == renders
    assert ("no time left in this request for another attempt" in out) is note
    assert "two-panel picture" in out


async def test_the_default_ladder_prices_attempt_three_as_a_plain_render(
        tmp_path, web, empty_history, monkeypatch):
    # §4VJ: with the cfg leg off, 200 s left covers a 150 s reseed (+30) for
    # BOTH re-renders — the cfg row above (2 renders) does not apply
    assert IG.COLLAGE_CFG_LEG is False
    monkeypatch.setattr(IG, "_request_time_left", lambda: 200.0)
    clock = {"t": 0.0}
    monkeypatch.setattr(IG, "time", __import__("types").SimpleNamespace(monotonic=lambda: clock["t"]))

    async def render(payload):
        clock["t"] += 150.0
        return _resp()
    llm = _llm(verdicts=[YES, YES, YES])
    llm.generate_image.side_effect = render
    out = await tool_generate_image(prompt="k", subjects=TWO, llm_client=llm, sandbox_dir=tmp_path)
    assert llm.generate_image.await_count == 3
    assert "no time left in this request" not in out


def test_time_left_reads_the_request_deadline_and_the_report_reserve(monkeypatch):
    from ghost_agent.utils import logging as L
    monkeypatch.setattr(L, "request_remaining_s", lambda rid: 1000.0)
    monkeypatch.setattr(L, "request_deadline_s", lambda rid="": 1800.0)
    from ghost_agent.core.agent import effective_report_floor
    tok = L.request_id_context.set("")
    try:
        assert IG._request_time_left() == pytest.approx(1000.0 - effective_report_floor(1800.0, ""))
        monkeypatch.setattr(L, "request_remaining_s", lambda rid: None)
        assert IG._request_time_left() is None
    finally:
        L.request_id_context.reset(tok)
