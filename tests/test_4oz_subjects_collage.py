"""§4OZ (2026-10-14) — two-panel outputs from the subjects path.

Three two-subject renders in one morning came out as side-by-side panels:
the ONE reference the node takes was a white-framed strip of two photos and
the edit path copied its layout (the negative prompt that countered it is
inert on Turbo at cfg 1). Fix: after a multi-subject render the tool asks the vision node whether the
picture is a collage and, on a confident YES, renders again — a fresh seed,
then a fresh seed under true CFG — never more than two retries. (A seamless
composite was built and measured WORSE — see below.)
"""
import base64
import io
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from PIL import Image

import ghost_agent.tools.subject_photos as SP
import ghost_agent.tools.image_gen as IG
from ghost_agent.tools.image_gen import tool_generate_image


def _jpeg(w=300, h=400, colour=(200, 50, 50)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), colour).save(buf, format="JPEG")
    return buf.getvalue()


PNG = b"\x89PNG\r\n\x1a\nxx"


def _resp(seed=7, png=PNG):
    return {"data": [{"b64_json": base64.b64encode(png).decode()}], "width": 768, "height": 512, "seed": seed}


def _vision(answer):
    """A vision-node reply; `answer` is the JSON the model wrote, or raw text."""
    txt = answer if isinstance(answer, str) else json.dumps(answer)
    return {"choices": [{"message": {"content": txt}}]}


def _llm(verdicts=(), renders=None):
    llm = AsyncMock()
    llm.image_gen_clients = [{"model": "Ghost"}]
    llm.generate_image.side_effect = list(renders) if renders else None
    if not renders:
        llm.generate_image.return_value = _resp()
    llm.chat_completion.side_effect = [_vision(v) for v in verdicts] or None
    if not verdicts:
        llm.chat_completion.return_value = _vision({"collage": "NO", "confidence": 0.9, "evidence": "one scene"})
    return llm


@pytest.fixture
def web(monkeypatch):
    photos = {"Alexis Tsipras": _jpeg(300, 400, (10, 10, 10)),
              "Antonis Samaras": _jpeg(400, 300, (240, 240, 240)),
              "Parthenon": _jpeg(500, 300, (100, 100, 0))}

    def lookup(subject, proxy):
        if subject not in photos:
            return None
        return (f"https://img.example/{subject}.jpg", subject, f"https://en.wikipedia.org/wiki/{subject}")

    async def download(url, sandbox_dir, tor_proxy, filename=None):
        name = url.rsplit("/", 1)[1][:-4]
        (Path(sandbox_dir) / filename).write_bytes(photos[name])
        return f"SUCCESS: Downloaded '{url}' to '{filename}'."

    monkeypatch.setattr(SP, "lookup_photo_url", lookup)
    import ghost_agent.tools.file_system as FS
    monkeypatch.setattr(FS, "tool_download_file", download)


def _sent(llm, n=0):
    return llm.generate_image.call_args_list[n].args[0]


def _sent_ref(llm, n=0):
    return Image.open(io.BytesIO(base64.b64decode(_sent(llm, n)["reference_images"][0])))


YES = {"collage": "YES", "confidence": 0.9, "evidence": "two panels with a seam"}
NO = {"collage": "NO", "confidence": 0.9, "evidence": "one scene"}
TWO = ["Alexis Tsipras", "Antonis Samaras"]


# ── the join stays the framed strip ─────────────────────────────────────────
# Measured 2026-10-14 (10 ordinary two-person scenes, same seed per pair, judged
# by eye and by the vision node, which agreed 20/20): a seamless composite —
# head-and-shoulders crops touching, feathered seam, blurred margins — came out
# as two panels 7/10 against 5/10 for the white-gap strip. An edit PRESERVES what
# looks like a finished picture; the frame is what tells the model these are
# separate photos to recompose. The composite was removed; the strip stays.

async def test_two_subjects_still_get_the_framed_strip(tmp_path, web):
    llm = _llm()
    out = await tool_generate_image(prompt="k", subjects=TWO, llm_client=llm, sandbox_dir=tmp_path)
    im = _sent_ref(llm)
    strip_w = round(300 * 1024 / 400) + round(400 * 1024 / 300) + SP.COMBINED_GAP
    assert im.width == strip_w and abs(im.width / im.height - 1.5) < 0.01
    assert not hasattr(SP, "combine_seamless")                         # removed, not switched off
    assert "left to right: Alexis Tsipras, Antonis Samaras" in _sent(llm)["prompt"]
    assert "REFERENCE PHOTO: subject_photos/ref_combined_" in out and "THIS WAS A NEW SCENE" in out


# ── the collage check and the one retry ─────────────────────────────────────

async def test_a_confident_collage_verdict_rerenders_with_a_fresh_seed(tmp_path, web):
    llm = _llm(verdicts=[YES, NO], renders=[_resp(seed=1, png=PNG + b"1"), _resp(seed=2, png=PNG + b"2")])
    out = await tool_generate_image(prompt="k", subjects=TWO, seed=1, llm_client=llm, sandbox_dir=tmp_path)
    assert llm.generate_image.await_count == 2
    assert _sent(llm, 0)["seed"] == 1 and "seed" not in _sent(llm, 1)        # fresh seed, same everything else
    assert "guidance_scale" not in _sent(llm, 1)                             # attempt 2 is cfg-free
    assert {k: v for k, v in _sent(llm, 0).items() if k != "seed"} == _sent(llm, 1)
    assert llm.chat_completion.await_count == 2                              # both pictures were looked at
    # the SECOND picture is the one saved and shown
    saved = next(tmp_path.glob("gen_*.png")).read_bytes()
    assert saved == PNG + b"2" and "PANELS" not in out
    assert out.startswith("SUCCESS")


async def test_the_third_attempt_runs_under_true_cfg(tmp_path, web):
    llm = _llm(verdicts=[YES, YES, NO],
               renders=[_resp(seed=1, png=PNG + b"1"), _resp(seed=2, png=PNG + b"2"), _resp(seed=3, png=PNG + b"3")])
    out = await tool_generate_image(prompt="k", subjects=TWO, llm_client=llm, sandbox_dir=tmp_path)
    assert llm.generate_image.await_count == 3 and llm.chat_completion.await_count == 3
    assert "guidance_scale" not in _sent(llm, 1)
    # §4QG: attempt 3 reuses the FIRST render's seed (read back from the node)
    # at cfg 4 — the §4OZ measurement; it used to draw a fresh seed
    assert _sent(llm, 2)["guidance_scale"] == IG.COLLAGE_RETRY_GUIDANCE == 4.0 and _sent(llm, 2)["seed"] == 1
    assert _sent(llm, 2)["negative_prompt"].startswith("collage")           # the anti-collage negative rides along
    assert next(tmp_path.glob("gen_*.png")).read_bytes() == PNG + b"3" and "PANELS" not in out


async def test_three_collages_keep_the_last_and_say_so_without_a_fourth_render(tmp_path, web):
    llm = _llm(verdicts=[YES, YES, YES, YES],
               renders=[_resp(seed=1, png=PNG + b"1"), _resp(seed=2, png=PNG + b"2"), _resp(seed=3, png=PNG + b"3"),
                        _resp(seed=4, png=PNG + b"4")])
    out = await tool_generate_image(prompt="k", subjects=TWO, llm_client=llm, sandbox_dir=tmp_path)
    assert llm.generate_image.await_count == 3                               # never a fourth
    assert "side-by-side PANELS 3 times" in out and "Do NOT render again on your own" in out
    assert next(tmp_path.glob("gen_*.png")).read_bytes() == PNG + b"3"
    # the record the agent parses is still first and intact
    assert "REFERENCE PHOTO: subject_photos/ref_combined_" in out
    assert out.index("REFERENCE PHOTO") < out.index("PANELS")


async def test_retries_can_be_limited_to_the_reseed(tmp_path, web, monkeypatch):
    monkeypatch.setattr(IG, "COLLAGE_RETRIES", 1)
    llm = _llm(verdicts=[YES, YES, NO], renders=[_resp(seed=1, png=PNG + b"1"), _resp(seed=2, png=PNG + b"2"),
                                                 _resp(seed=3, png=PNG + b"3")])
    out = await tool_generate_image(prompt="k", subjects=TWO, llm_client=llm, sandbox_dir=tmp_path)
    assert llm.generate_image.await_count == 2 and "PANELS 2 times" in out
    assert next(tmp_path.glob("gen_*.png")).read_bytes() == PNG + b"2"


async def test_a_clean_verdict_never_rerenders(tmp_path, web):
    llm = _llm(verdicts=[NO])
    out = await tool_generate_image(prompt="k", subjects=TWO, llm_client=llm, sandbox_dir=tmp_path)
    assert llm.generate_image.await_count == 1 and llm.chat_completion.await_count == 1
    assert "PANELS" not in out


@pytest.mark.parametrize("verdict", [
    {"collage": "YES", "confidence": 0.5, "evidence": "maybe"},     # below the floor
    {"collage": "MAYBE", "confidence": 0.9},                       # not a verdict
    "I cannot tell from this image.",                              # no JSON
    "",                                                            # empty answer
])
async def test_an_unsure_or_unusable_verdict_keeps_the_picture(tmp_path, web, verdict):
    llm = _llm(verdicts=[verdict])
    out = await tool_generate_image(prompt="k", subjects=TWO, llm_client=llm, sandbox_dir=tmp_path)
    assert llm.generate_image.await_count == 1 and out.startswith("SUCCESS") and "PANELS" not in out


async def test_a_vision_node_failure_keeps_the_picture(tmp_path, web):
    llm = _llm()
    llm.chat_completion.side_effect = RuntimeError("vision down")
    out = await tool_generate_image(prompt="k", subjects=TWO, llm_client=llm, sandbox_dir=tmp_path)
    assert llm.generate_image.await_count == 1 and out.startswith("SUCCESS")


async def test_a_failed_rerender_keeps_the_first_picture_and_says_panels(tmp_path, web):
    llm = _llm(verdicts=[YES], renders=[_resp(seed=1, png=PNG + b"1"), RuntimeError("node busy")])
    out = await tool_generate_image(prompt="k", subjects=TWO, llm_client=llm, sandbox_dir=tmp_path)
    assert next(tmp_path.glob("gen_*.png")).read_bytes() == PNG + b"1"
    assert "another attempt could not be rendered" in out and out.startswith("SUCCESS")
    assert llm.generate_image.await_count == 2                               # a failed attempt ends the ladder


async def test_an_empty_rerender_keeps_the_first_picture(tmp_path, web):
    llm = _llm(verdicts=[YES], renders=[_resp(seed=1, png=PNG + b"1"), {"data": [{"b64_json": ""}]}])
    out = await tool_generate_image(prompt="k", subjects=TWO, llm_client=llm, sandbox_dir=tmp_path)
    assert next(tmp_path.glob("gen_*.png")).read_bytes() == PNG + b"1" and out.startswith("SUCCESS")


async def test_one_subject_and_plain_renders_are_never_checked(tmp_path, web):
    llm = _llm(verdicts=[YES, YES])
    await tool_generate_image(prompt="k", subjects=["Parthenon"], llm_client=llm, sandbox_dir=tmp_path)
    await tool_generate_image(prompt="k", llm_client=llm, sandbox_dir=tmp_path)
    assert llm.chat_completion.await_count == 0 and llm.generate_image.await_count == 2


async def test_the_kill_switch_skips_the_check(tmp_path, web, monkeypatch):
    monkeypatch.setattr(IG, "COLLAGE_RETRIES", 0)
    llm = _llm(verdicts=[YES, YES])
    await tool_generate_image(prompt="k", subjects=TWO, llm_client=llm, sandbox_dir=tmp_path)
    assert llm.chat_completion.await_count == 0 and llm.generate_image.await_count == 1


async def test_the_check_sends_the_rendered_pixels_without_thinking(tmp_path, web):
    llm = _llm(verdicts=[NO])
    await tool_generate_image(prompt="k", subjects=TWO, llm_client=llm, sandbox_dir=tmp_path)
    call = llm.chat_completion.call_args
    payload = call.args[0]
    assert call.kwargs["use_vision"] is True and call.kwargs["timeout"] == IG.COLLAGE_CHECK_TIMEOUT_S
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    parts = payload["messages"][0]["content"]
    img = next(p for p in parts if p["type"] == "image_url")["image_url"]["url"]
    assert base64.b64decode(img.split(",", 1)[1]) == PNG
    assert "COLLAGE" in next(p for p in parts if p["type"] == "text")["text"]


def test_image_bytes_of_strips_a_data_uri_and_rejects_garbage():
    assert IG._image_bytes_of({"data": [{"b64_json": "data:image/png;base64," + base64.b64encode(PNG).decode()}]}) == PNG
    assert IG._image_bytes_of({"data": [{"b64_json": ""}]}) == b""
    assert IG._image_bytes_of({"data": [{"b64_json": None}]}) == b""
    assert IG._image_bytes_of("not a dict") == b""
    assert IG._image_bytes_of({"data": [{"b64_json": "%%%notb64"}]}) in (b"", IG._image_bytes_of({"data": [{"b64_json": "%%%notb64"}]}))


@pytest.mark.parametrize("answer,expect", [
    ({"collage": "YES", "confidence": 0.8, "evidence": "seam"}, {"collage": True, "confidence": 0.8}),
    ({"collage": "no", "confidence": "0.95"}, {"collage": False, "confidence": 0.95}),
    ({"collage": "YES", "confidence": 7}, {"collage": True, "confidence": 1.0}),      # clamped
    ({"collage": "MAYBE", "confidence": 0.9}, None),                                  # not a verdict → None, never False
    ({"confidence": 0.9}, None),
    ("no json here", None),
    ("", None),
])
async def test_collage_verdict_only_accepts_a_yes_or_no(answer, expect):
    llm = AsyncMock()
    llm.chat_completion.return_value = _vision(answer)
    v = await IG.collage_verdict(llm, PNG)
    if expect is None:
        assert v is None
    else:
        assert v["collage"] == expect["collage"] and abs(v["confidence"] - expect["confidence"]) < 1e-9


async def test_collage_verdict_without_a_client_or_pixels_is_none():
    assert await IG.collage_verdict(None, PNG) is None
    assert await IG.collage_verdict(AsyncMock(), b"") is None


async def test_the_render_log_line_carries_the_whole_prompt_under_its_new_title(tmp_path, web, capsys):
    # operator 2026-10-14: "always show the full prompt … rename it to image generation"
    long_prompt = "A lighthouse at dusk. " * 40                      # 880 chars, well past the old 30 and the stream cut
    llm = _llm(verdicts=[NO])
    await tool_generate_image(prompt=long_prompt, subjects=TWO, llm_client=llm, sandbox_dir=tmp_path)
    out = capsys.readouterr().out
    line = next(l for l in out.splitlines() if "image generation" in l.lower() and "prompt (" in l)
    assert "image gen " not in out.lower().replace("image generation", "")
    assert "size=768x512 | refs=1 | prompt (" in line
    import re
    flat = re.sub(r"\s+", " ", out)                                         # the stream wraps long content
    sent = _sent(llm)["prompt"]
    assert sent.endswith(long_prompt.strip()) and re.sub(r"\s+", " ", sent) in flat   # the WHOLE prompt is in the log
    assert f"prompt ({len(sent)} chars)" in line
    assert "..." not in line.split("prompt (")[0]                          # no 30-char cut


# ── §4PA: `subjects` + one uploaded photo ───────────────────────────────────
# Live slack-c7 (2026-10-14): an uploaded photo of one person plus a named other
# was refused ("cannot be combined"). Operator: combine them for ordinary
# requests. The upload is joined as the LAST panel of the one reference.

async def test_the_upload_counts_toward_the_three_panel_limit(tmp_path, web):
    (tmp_path / "me.jpg").write_bytes(_jpeg())
    llm = _llm()
    out = await tool_generate_image(prompt="k", subjects=["Alexis Tsipras", "Antonis Samaras", "Parthenon"],
                                    reference_images=["me.jpg"], llm_client=llm, sandbox_dir=tmp_path)
    assert out.startswith("ERROR: at most 3 subjects") and "got 4" in out and "uploaded photo me.jpg" in out
    llm.generate_image.assert_not_called()


async def test_too_many_uploads_with_subjects_are_refused(tmp_path, web):
    for n in ("a.jpg", "b.jpg", "c.jpg"):
        (tmp_path / n).write_bytes(_jpeg())
    llm = _llm()
    out = await tool_generate_image(prompt="k", subjects=["Alexis Tsipras"], reference_images=["a.jpg", "b.jpg", "c.jpg"],
                                    llm_client=llm, sandbox_dir=tmp_path)
    assert out.startswith("ERROR: at most 3 subjects") and "got 4" in out and "a.jpg, b.jpg, c.jpg" in out
    llm.generate_image.assert_not_called()


async def test_a_bad_upload_is_refused_before_any_lookup(tmp_path, web):
    (tmp_path / "notes.txt").write_bytes(b"not an image at all")
    llm = _llm()
    out = await tool_generate_image(prompt="k", subjects=["Alexis Tsipras"], reference_images=["notes.txt"],
                                    llm_client=llm, sandbox_dir=tmp_path)
    assert out.startswith("ERROR: reference image 'notes.txt' is empty or not a PNG/JPEG/WEBP")
    llm.generate_image.assert_not_called()


async def test_subject_plus_upload_gets_the_collage_check_and_retry(tmp_path, web):
    (tmp_path / "me.jpg").write_bytes(_jpeg(400, 300, (240, 240, 240)))
    llm = _llm(verdicts=[YES, NO], renders=[_resp(seed=1, png=PNG + b"1"), _resp(seed=2, png=PNG + b"2")])
    out = await tool_generate_image(prompt="k", subjects=["Alexis Tsipras"], reference_images=["me.jpg"],
                                    llm_client=llm, sandbox_dir=tmp_path)
    assert llm.generate_image.await_count == 2 and llm.chat_completion.await_count == 2   # two panels → checked
    assert next(tmp_path.glob("gen_*.png")).read_bytes() == PNG + b"2" and out.startswith("SUCCESS")


async def test_an_upload_alone_is_still_a_plain_edit(tmp_path, web):
    (tmp_path / "me.jpg").write_bytes(_jpeg())
    llm = _llm(verdicts=[YES])
    out = await tool_generate_image(prompt="k", reference_images=["me.jpg"], llm_client=llm, sandbox_dir=tmp_path)
    assert llm.chat_completion.await_count == 0 and "THIS WAS AN EDIT" in out


def test_the_model_is_told_the_two_combine_and_how_to_add_a_person():
    from unittest.mock import MagicMock
    from ghost_agent.tools.registry import get_active_tool_definitions
    ctx = MagicMock(); ctx.llm_client.image_gen_clients = [{"model": "Ghost"}]; ctx.tor_proxy = None
    fn = next(t["function"] for t in get_active_tool_definitions(ctx, serve_tuned=False)
              if t["function"]["name"] == "image_generation")
    desc, ref = fn["description"], fn["parameters"]["properties"]["reference_images"]["description"]
    assert "Cannot be combined" not in ref and "The two COMBINE" in ref and "ADDING someone" in ref
    assert "To ADD a real person to an existing picture" in desc and "together" in desc


async def test_a_generated_picture_plus_a_subject_keeps_the_scene_first(tmp_path, web):
    # turn one made gen_aa.png (X and Y hugging); turn two: "add Z"
    (tmp_path / "gen_aa.png").write_bytes(_jpeg(400, 300, (240, 240, 240)))   # the light "scene"
    llm = _llm(verdicts=[NO])
    out = await tool_generate_image(prompt="add Alexis Tsipras standing between them",
                                    reference_images=["gen_aa.png"], subjects=["Alexis Tsipras"],
                                    llm_client=llm, sandbox_dir=tmp_path)
    assert out.startswith("SUCCESS")
    sent = _sent(llm)
    assert len(sent["reference_images"]) == 1
    im = _sent_ref(llm)
    mid = im.height // 2
    # the existing picture is the LEFT panel, the fetched (dark) subject the right one
    assert im.getpixel((50, mid))[0] > 200 and im.getpixel((im.width - 50, mid))[0] < 60
    assert "left to right: the existing picture, Alexis Tsipras" in sent["prompt"]
    assert "The existing picture is the scene to keep" in sent["prompt"]
    assert "the existing picture with them added into it" in sent["prompt"]
    assert sent["prompt"].endswith("The change: add Alexis Tsipras standing between them")
    assert "collage" in sent["negative_prompt"]
    assert "left to right: existing picture (gen_aa.png), Alexis Tsipras (ref_alexis_tsipras.jpg). THIS WAS A NEW SCENE" in out
    assert "the existing picture plus the subjects' photos" in out
    assert llm.chat_completion.await_count == 1                              # two panels → collage-checked


async def test_a_user_photo_plus_a_subject_puts_the_photo_last(tmp_path, web):
    (tmp_path / "me.jpg").write_bytes(_jpeg(400, 300, (240, 240, 240)))
    llm = _llm(verdicts=[NO])
    await tool_generate_image(prompt="k", reference_images=["me.jpg"], subjects=["Alexis Tsipras"],
                              llm_client=llm, sandbox_dir=tmp_path)
    im = _sent_ref(llm); mid = im.height // 2
    assert im.getpixel((50, mid))[0] < 60 and im.getpixel((im.width - 50, mid))[0] > 200
    assert "left to right: Alexis Tsipras, the person in the uploaded photo" in _sent(llm)["prompt"]
    assert "The scene: k" in _sent(llm)["prompt"]
