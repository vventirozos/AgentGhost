"""Req 8473053b (2026-11-07): two uploads, a face to put on a game cover. The
node takes one reference and only `subjects` could join pictures, so the model
wrote its own PIL montage (4 of 5 scripts failed), rendered it as a plain edit
(no panel check) and cropped the result by hand.

Now `reference_images` takes up to 3 files and the tool joins them: a new scene
by default, or with `keep_first` an edit of the first file, cut back out of the
render. Measured on the node: the new-scene wording never made this change (0/6,
two panels), the edit-the-left wording made it 5/5.
"""
import base64
import io
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from PIL import Image

import ghost_agent.tools.image_gen as IG
import ghost_agent.tools.subject_photos as SP
from ghost_agent.core.agent import (GhostAgent, _SUBJECT_REFERENCE_RE, _turn_edit_reference, _turn_edit_was_joined,
                                     _turn_subject_reference)
from ghost_agent.tools.image_gen import expected_render_seconds, tool_generate_image

RED, BLUE, GREEN = (200, 20, 20), (20, 20, 200), (20, 200, 20)


def _jpeg(w=300, h=400, colour=RED) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), colour).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _png(im) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _render_like_strip(payload, w=736, gap_on=True, shift=0):
    """What the node does with a keep-first request: the strip's layout, at its
    own size. Any panel colour is kept, so the crop's result can be read."""
    ref = Image.open(io.BytesIO(base64.b64decode(payload["reference_images"][0]))).convert("RGB")
    h = round(ref.height * w / ref.width)
    out = ref.resize((w, h))
    if shift:
        moved = Image.new("RGB", (w, h), (255, 255, 255))
        moved.paste(out, (shift, 0))
        out = moved
    if not gap_on:
        out = Image.new("RGB", (w, h), GREEN)
    return {"data": [{"b64_json": base64.b64encode(_png(out)).decode()}], "width": w, "height": h, "seed": 7}


def _llm(render=None, verdict="NO"):
    llm = AsyncMock()
    llm.image_gen_clients = [{"model": "Ghost"}]
    llm.generate_image.side_effect = render or (lambda p, **k: _render_like_strip(p))
    llm.chat_completion.return_value = {"choices": [{"message": {"content": json.dumps(
        {"collage": verdict, "confidence": 0.9, "evidence": "e", "scene": "a scene"})}}]}
    return llm


def _files(d, *names):
    colours = [RED, BLUE, GREEN]
    for n, c in zip(names, colours):
        (Path(d) / n).write_bytes(_jpeg(colour=c))


def _sent(llm, n=0):
    return llm.generate_image.call_args_list[n].args[0]


def _sent_ref(llm, n=0):
    return Image.open(io.BytesIO(base64.b64decode(_sent(llm, n)["reference_images"][0]))).convert("RGB")


def _saved(out, d):
    name = out.split("/api/download/")[1].split(")")[0]
    return Image.open(Path(d) / name).convert("RGB")


def _near(px, colour, tol=45):
    return all(abs(a - b) < tol for a, b in zip(px, colour))


# ── the default: several files are a new scene ──────────────────────────────

async def test_two_uploads_join_into_one_reference_in_the_order_given(tmp_path):
    _files(tmp_path, "a.jpg", "b.jpg")
    llm = _llm(lambda p, **k: _render_like_strip(p))
    out = await tool_generate_image(prompt="they hug", reference_images=["a.jpg", "b.jpg"],
                                    llm_client=llm, sandbox_dir=tmp_path)
    assert out.startswith("SUCCESS")
    sent = _sent(llm)
    assert len(sent["reference_images"]) == 1          # the node's real limit
    ref = _sent_ref(llm)
    assert _near(ref.getpixel((ref.width // 2 - 40, ref.height // 2)), RED)
    assert _near(ref.getpixel((ref.width // 2 + 40, ref.height // 2)), BLUE)
    assert "2 separate photos side by side" in sent["prompt"] and sent["prompt"].endswith("they hug")
    assert (sent["width"], sent["height"]) == (768, 512)


async def test_a_new_scene_from_uploads_gets_the_panel_check_and_a_re_render(tmp_path):
    _files(tmp_path, "a.jpg", "b.jpg")
    llm = _llm(verdict="YES")
    await tool_generate_image(prompt="they hug", reference_images=["a.jpg", "b.jpg"],
                              llm_client=llm, sandbox_dir=tmp_path)
    assert llm.generate_image.call_count > 1


async def test_the_new_scene_result_keeps_the_record_the_visual_check_reads(tmp_path):
    _files(tmp_path, "a.jpg", "b.jpg")
    out = await tool_generate_image(prompt="they hug", reference_images=["a.jpg", "b.jpg"],
                                    llm_client=_llm(), sandbox_dir=tmp_path)
    assert _SUBJECT_REFERENCE_RE.search(out)
    ref, who = _turn_subject_reference([{"content": out}])
    assert ref.startswith(SP.PHOTO_DIR) and (tmp_path / ref).exists()
    assert who == "uploaded photo, uploaded photo"
    assert "MORE SPECIFIC name" not in out            # nothing was searched for by name
    assert "built from the joined reference pictures" in out


async def test_a_generated_picture_among_uploads_is_the_scene_and_goes_first(tmp_path):
    (tmp_path / "me.jpg").write_bytes(_jpeg(colour=RED))
    (tmp_path / "gen_aa.jpg").write_bytes(_jpeg(colour=BLUE))
    llm = _llm()
    await tool_generate_image(prompt="add them", reference_images=["me.jpg", "gen_aa.jpg"],
                              llm_client=llm, sandbox_dir=tmp_path)
    ref, prompt = _sent_ref(llm), _sent(llm)["prompt"]
    assert _near(ref.getpixel((ref.width // 2 - 40, ref.height // 2)), BLUE)
    assert "the scene to keep" in prompt and "Keep the person in the uploaded photo looking" in prompt


@pytest.mark.parametrize("n,ok", [(3, True), (4, False)])
async def test_the_file_limit_is_the_panel_limit(tmp_path, n, ok):
    names = [f"{i}.jpg" for i in range(n)]
    for x in names:
        (tmp_path / x).write_bytes(_jpeg())
    llm = _llm()
    out = await tool_generate_image(prompt="p", reference_images=names, llm_client=llm, sandbox_dir=tmp_path)
    assert out.startswith("SUCCESS") is ok
    assert llm.generate_image.called is ok
    if not ok:
        assert "at most 3 reference images" in out and "got 4" in out and "Nothing was rendered" in out


def test_the_join_cap_cannot_drift_from_the_panel_cap():
    assert IG.MAX_JOINED_REFERENCES == SP.MAX_SUBJECTS


async def test_a_corrupt_file_in_a_join_is_a_result_and_nothing_renders(tmp_path):
    (tmp_path / "a.jpg").write_bytes(_jpeg())
    (tmp_path / "b.jpg").write_bytes(b"\xff\xd8\xff" + b"junk" * 20)
    llm = _llm()
    out = await tool_generate_image(prompt="p", reference_images=["a.jpg", "b.jpg"], llm_client=llm, sandbox_dir=tmp_path)
    assert out.startswith("ERROR: could not prepare the reference pictures")
    llm.generate_image.assert_not_called()


async def test_one_file_is_still_a_plain_edit_and_sends_its_own_bytes(tmp_path):
    (tmp_path / "a.jpg").write_bytes(_jpeg())
    llm = _llm(lambda p, **k: {"data": [{"b64_json": base64.b64encode(_png(Image.new("RGB", (8, 8)))).decode()}]})
    out = await tool_generate_image(prompt="p", reference_images=["a.jpg"], keep_first=True,
                                    llm_client=llm, sandbox_dir=tmp_path)
    assert base64.b64decode(_sent(llm)["reference_images"][0]) == (tmp_path / "a.jpg").read_bytes()
    assert _sent(llm)["prompt"] == "p"
    assert "The model saw ONLY a.jpg" in out and "cut back" not in out


# ── keep_first: edit the first file, get that picture back ──────────────────

async def test_keep_first_returns_only_the_first_picture(tmp_path):
    _files(tmp_path, "cover.jpg", "face.jpg")
    llm = _llm()
    out = await tool_generate_image(prompt="put the face on him", reference_images=["cover.jpg", "face.jpg"],
                                    keep_first=True, llm_client=llm, sandbox_dir=tmp_path)
    assert out.startswith("SUCCESS") and "cut back to the first picture" in out
    whole = out.split("the whole render is ")[1].split(".png")[0] + ".png"
    assert Image.open(tmp_path / whole).width == 736            # a wrong cut can be undone
    assert whole.startswith("joined_") and "joined_" not in out.split("/api/download/")[1].split(")")[0]
    pic = _saved(out, tmp_path)
    assert pic.width < 400                                    # not the 736-wide strip
    for x in (2, pic.width // 2, pic.width - 3):              # no gap, no second panel
        assert _near(pic.getpixel((x, pic.height // 2)), RED), x
    assert f"Rendered at {pic.width}x{pic.height}" in out     # the size of what was saved


async def test_keep_first_uses_the_measured_wording_and_no_panel_ladder(tmp_path):
    _files(tmp_path, "cover.jpg", "face.jpg")
    llm = _llm(verdict="YES")           # panels are the method: a YES must not re-render
    await tool_generate_image(prompt="the change", reference_images=["cover.jpg", "face.jpg"],
                              keep_first=True, llm_client=llm, sandbox_dir=tmp_path)
    assert llm.generate_image.call_count == 1
    sent = _sent(llm)
    assert sent["prompt"] == ("This image is 2 pictures side by side with a gap between them. Edit ONLY the LEFT "
                              "picture. Keep the RIGHT picture exactly as it is: it only shows what to use. Keep the "
                              "pictures side by side in the same places. The change to the left picture: the change")
    assert "negative_prompt" not in sent
    assert "width" not in sent and "height" not in sent       # the strip's own shape


async def test_keep_first_is_an_edit_of_the_first_file_for_the_visual_check(tmp_path):
    _files(tmp_path, "cover.jpg", "face.jpg")
    out = await tool_generate_image(prompt="c", reference_images=["cover.jpg", "face.jpg"], keep_first=True,
                                    llm_client=_llm(), sandbox_dir=tmp_path)
    assert _turn_edit_reference([{"content": out}]) == "cover.jpg"
    assert _turn_subject_reference([{"content": out}]) is None
    assert "THIS WAS AN EDIT" in out and "cover.jpg, face.jpg" in out
    assert "saw ONLY" not in out                              # it saw both
    assert _turn_edit_was_joined([{"content": out}])
    assert "edited from the first picture" in out
    # the LAST image decides: a plain edit after it is judged the plain way
    assert not _turn_edit_was_joined([{"content": out}, {"content": "SUCCESS: Image generated. EDITED FROM: x.png"}])


async def test_a_joined_edit_is_judged_against_its_own_first_file_not_the_first_one_named(tmp_path):
    # the live request named the face first: "put the face of giokarinis into ocarina"
    _files(tmp_path, "cover.jpg", "face.jpg")
    out = await tool_generate_image(prompt="c", reference_images=["cover.jpg", "face.jpg"], keep_first=True,
                                    llm_client=_llm(), sandbox_dir=tmp_path)
    before, who = GhostAgent._likeness_evidence([{"content": out}], tmp_path, tmp_path / "face.jpg")
    assert Path(before).name == "cover.jpg" and who is None


async def test_a_plain_edit_still_lets_the_users_named_image_win(tmp_path):
    _files(tmp_path, "cover.jpg", "shot.jpg")
    out = await tool_generate_image(prompt="c", reference_images=["cover.jpg"],
                                    llm_client=_llm(lambda p, **k: {"data": [{"b64_json": base64.b64encode(_png(Image.new("RGB", (8, 8)))).decode()}]}),
                                    sandbox_dir=tmp_path)
    assert not _turn_edit_was_joined([{"content": out}])
    before, _ = GhostAgent._likeness_evidence([{"content": out}], tmp_path, tmp_path / "shot.jpg")
    assert Path(before).name == "shot.jpg"


async def test_a_joined_edit_leaves_no_copy_of_the_uploads_on_disk(tmp_path):
    _files(tmp_path, "cover.jpg", "face.jpg")
    await tool_generate_image(prompt="c", reference_images=["cover.jpg", "face.jpg"], keep_first=True,
                              llm_client=_llm(), sandbox_dir=tmp_path)
    assert not (tmp_path / SP.PHOTO_DIR).exists()


async def test_a_duplicate_does_not_count_toward_the_limit(tmp_path):
    _files(tmp_path, "a.jpg", "b.jpg", "c.jpg")
    out = await tool_generate_image(prompt="p", reference_images=["a.jpg", "a.jpg", "b.jpg", "c.jpg"],
                                    llm_client=_llm(), sandbox_dir=tmp_path)
    assert out.startswith("SUCCESS")


async def test_several_generated_pictures_are_labelled_as_generated(tmp_path):
    for n, c in (("gen_a.jpg", RED), ("gen_b.jpg", BLUE), ("gen_c.jpg", GREEN)):
        (tmp_path / n).write_bytes(_jpeg(colour=c))
    llm = _llm()
    await tool_generate_image(prompt="p", reference_images=["gen_a.jpg", "gen_b.jpg", "gen_c.jpg"],
                              llm_client=llm, sandbox_dir=tmp_path)
    p = _sent(llm)["prompt"]
    assert "generated picture 1" in p and "generated picture 2" in p and "uploaded" not in p


async def test_the_same_file_twice_is_one_picture(tmp_path):
    (tmp_path / "a.jpg").write_bytes(_jpeg())
    llm = _llm(lambda p, **k: {"data": [{"b64_json": base64.b64encode(_png(Image.new("RGB", (8, 8)))).decode()}]})
    await tool_generate_image(prompt="p", reference_images=["a.jpg", "a.jpg"], llm_client=llm, sandbox_dir=tmp_path)
    assert _sent(llm)["prompt"] == "p"
    assert base64.b64decode(_sent(llm)["reference_images"][0]) == (tmp_path / "a.jpg").read_bytes()


async def test_a_second_generated_picture_is_not_called_a_persons_upload(tmp_path):
    (tmp_path / "gen_a.jpg").write_bytes(_jpeg(colour=RED))
    (tmp_path / "gen_b.jpg").write_bytes(_jpeg(colour=BLUE))
    llm = _llm()
    out = await tool_generate_image(prompt="merge", reference_images=["gen_a.jpg", "gen_b.jpg"],
                                    llm_client=llm, sandbox_dir=tmp_path)
    assert "uploaded photo" not in _sent(llm)["prompt"] and "the other generated picture" in _sent(llm)["prompt"]
    assert "generated picture (gen_b.jpg)" in out and "uploaded photo (gen_b.jpg)" not in out


async def test_a_scene_without_subjects_does_not_mention_subjects_photos(tmp_path):
    (tmp_path / "me.jpg").write_bytes(_jpeg(colour=RED))
    (tmp_path / "gen_a.jpg").write_bytes(_jpeg(colour=BLUE))
    out = await tool_generate_image(prompt="add", reference_images=["me.jpg", "gen_a.jpg"],
                                    llm_client=_llm(), sandbox_dir=tmp_path)
    assert "subjects' photos" not in out and "built from the joined reference pictures" in out


async def test_a_size_asked_with_keep_first_is_dropped_and_said(tmp_path):
    _files(tmp_path, "cover.jpg", "face.jpg")
    llm = _llm()
    out = await tool_generate_image(prompt="c", reference_images=["cover.jpg", "face.jpg"], keep_first=True,
                                    width=1024, height=1024, llm_client=llm, sandbox_dir=tmp_path)
    assert "width" not in _sent(llm)
    assert "requested size was ignored" in out and "you asked for" not in out


async def test_a_render_that_lost_the_layout_is_saved_whole_and_says_so(tmp_path):
    _files(tmp_path, "cover.jpg", "face.jpg")
    llm = _llm(lambda p, **k: _render_like_strip(p, gap_on=False))
    out = await tool_generate_image(prompt="c", reference_images=["cover.jpg", "face.jpg"], keep_first=True,
                                    llm_client=llm, sandbox_dir=tmp_path)
    pic = _saved(out, tmp_path)
    assert pic.width == 736                                   # not cut
    assert "could not confirm" in out and "saved whole and not cut" in out and "cut back" not in out
    assert not list(tmp_path.glob("joined_*"))


@pytest.mark.parametrize("flag,joined", [("true", True), ("false", False), (True, True), (False, False)])
async def test_keep_first_arrives_as_text_too(tmp_path, flag, joined):
    _files(tmp_path, "cover.jpg", "face.jpg")
    llm = _llm()
    await tool_generate_image(prompt="c", reference_images=["cover.jpg", "face.jpg"], keep_first=flag,
                              llm_client=llm, sandbox_dir=tmp_path)
    assert ("Edit ONLY the LEFT" in _sent(llm)["prompt"]) is joined


async def test_keep_first_with_a_subject_lists_who_is_where(tmp_path, monkeypatch):
    async def fetch(subject, folder, proxy, is_image=None):
        p = Path(folder) / "subject.jpg"
        p.write_bytes(_jpeg(colour=BLUE))
        return (p,)
    monkeypatch.setattr(SP, "fetch_subject_photo", fetch)
    (tmp_path / "cover.jpg").write_bytes(_jpeg(colour=RED))
    llm = _llm()
    out = await tool_generate_image(prompt="c", reference_images=["cover.jpg"], subjects=["Alexis Tsipras"],
                                    keep_first=True, llm_client=llm, sandbox_dir=tmp_path)
    assert "Left to right: the picture to edit, Alexis Tsipras." in _sent(llm)["prompt"]
    assert _near(_saved(out, tmp_path).getpixel((5, 5)), RED)


async def test_keep_first_with_three_files_names_them(tmp_path):
    _files(tmp_path, "a.jpg", "b.jpg", "c.jpg")
    llm = _llm()
    await tool_generate_image(prompt="c", reference_images=["a.jpg", "b.jpg", "c.jpg"], keep_first=True,
                              llm_client=llm, sandbox_dir=tmp_path)
    p = _sent(llm)["prompt"]
    assert "This image is 3 pictures" in p and "Keep the other pictures exactly as they are: they only show" in p
    assert "Left to right: the picture to edit, uploaded picture 2, uploaded picture 3." in p


# ── the strip and the cut ───────────────────────────────────────────────────

def _textured(w, h):
    """A picture with detail in every column, as a real first panel has."""
    return Image.effect_noise((w, h), 80).convert("RGB")


def _person(w, h, bg=(255, 255, 255)):
    """A panel with structure: an off-centre figure on a backdrop."""
    im = Image.new("RGB", (w, h), bg)
    im.paste((60, 40, 30), (w // 4, h // 3, w // 2, h))
    im.paste((200, 170, 150), (w // 3, h // 8, w // 2, h // 3))
    im.paste((20, 20, 90), (w // 2, h // 2, 3 * w // 4, h))
    return im


def _strip(w=736, h=400, edge=300, gap=8, bg=(255, 255, 255)):
    im = Image.new("RGB", (w, h), (255, 255, 255))
    im.paste(_textured(edge, h), (0, 0))
    im.paste(_person(w - edge - gap, h, bg), (edge + gap, 0))
    return im, (edge / w, 0.0, 1.0, gap / w)


def _photo(d, name, w, h, colour):
    p = Path(d) / name
    p.write_bytes(_jpeg(w, h, colour))
    return p


def test_the_first_picture_keeps_all_of_itself_and_others_are_narrowed_not_cropped(tmp_path):
    cover = _photo(tmp_path, "c.jpg", 500, 700, RED)
    # a group photo whose target stands far left: a centre crop lost him (review)
    group = Image.new("RGB", (1920, 1080), BLUE)
    group.paste(GREEN, (0, 0, 200, 1080))
    gp = tmp_path / "g.png"
    gp.write_bytes(_png(group))
    data, (edge, top, bottom, gap) = SP.combine_keep_first([cover, gp])
    strip = Image.open(io.BytesIO(data)).convert("RGB")
    first_w = round(500 * SP.COMBINED_HEIGHT / 700)
    assert (top, bottom) == (0.0, 1.0)
    assert round(edge * strip.width) == first_w
    assert round(gap * strip.width) == SP.COMBINED_GAP
    assert strip.width - first_w - SP.COMBINED_GAP == first_w      # no wider than the first
    left_of_other = first_w + SP.COMBINED_GAP + 3
    assert _near(strip.getpixel((left_of_other, strip.height // 2)), GREEN)    # all of it is there
    assert _near(strip.getpixel((strip.width - 4, strip.height // 2)), BLUE)
    assert strip.getpixel((strip.width - 4, 4))[0] > 240           # above it is white, not stretched


def test_a_panorama_first_is_letterboxed_never_cropped_and_the_box_says_where(tmp_path):
    pano = _photo(tmp_path, "p.jpg", 1000, 300, RED)
    face = _photo(tmp_path, "f.jpg", 300, 400, BLUE)
    data, box = SP.combine_keep_first([pano, face])
    strip = Image.open(io.BytesIO(data)).convert("RGB")
    assert 0.2 < box[1] < box[2] < 0.8
    assert _near(strip.getpixel((5, round(box[1] * strip.height) + 4)), RED)
    assert strip.getpixel((5, 4))[0] > 240                         # the band above it is white
    cut = SP.crop_first_panel(_png(strip), box, data)
    pic = Image.open(io.BytesIO(cut))
    assert abs(pic.width / pic.height - 1000 / 300) < 0.1          # the panorama's own shape


def test_a_large_first_picture_does_not_give_the_others_more_than_the_cap(tmp_path):
    wide = _photo(tmp_path, "w.jpg", 1200, 800, RED)
    other = _photo(tmp_path, "o.jpg", 1600, 800, BLUE)
    data, (edge, _, _, _) = SP.combine_keep_first([wide, other])
    strip = Image.open(io.BytesIO(data))
    assert strip.width - round(edge * strip.width) - SP.COMBINED_GAP == SP._OTHER_PANEL_MAX


def test_a_tiny_first_picture_does_not_shrink_the_face_to_a_sliver(tmp_path):
    thin = _photo(tmp_path, "t.jpg", 250, 800, RED)
    face = _photo(tmp_path, "f.jpg", 800, 800, BLUE)
    data, (edge, _, _, _) = SP.combine_keep_first([thin, face])
    strip = Image.open(io.BytesIO(data))
    assert strip.width - round(edge * strip.width) - SP.COMBINED_GAP == SP._OTHER_PANEL_MIN


@pytest.mark.parametrize("shift", [-2, -1, 0, 1, 2, 4])
def test_the_cut_finds_the_gap_when_the_render_moved_it(shift):
    strip, box = _strip()
    w, h, edge = 736, 400, 300
    render, _ = _strip(edge=edge + shift)
    cut = Image.open(io.BytesIO(SP.crop_first_panel(_png(render), box, _png(strip))))
    assert cut.width == edge + shift                    # no gap sliver, nothing lost


def test_a_gap_beyond_the_search_is_ignored_and_the_cut_is_at_the_edge():
    strip, box = _strip()
    render = strip.copy()
    render.paste(_textured(40, 400), (290, 0))           # fills the real gap
    render.paste((255, 255, 255), (340, 0, 348, 400))    # a flat run the same width, 40 px on
    assert Image.open(io.BytesIO(SP.crop_first_panel(_png(render), box, _png(strip)))).width == 300


def test_a_keyline_inside_the_picture_is_not_the_gap():
    # review: a dark line 3% in cost everything to the right of it
    strip, box = _strip()
    strip.paste((0, 0, 0), (292, 0, 293, 400))
    cut = Image.open(io.BytesIO(SP.crop_first_panel(_png(strip), box, _png(strip))))
    assert cut.width == 300


def test_a_gap_with_grain_is_still_found_where_it_moved():
    strip, box = _strip()
    grain, _ = _strip(edge=304)
    grain.paste(Image.effect_noise((8, 400), 4).convert("RGB"), (304, 0))
    assert Image.open(io.BytesIO(SP.crop_first_panel(_png(grain), box, _png(strip)))).width == 304


def test_detail_is_not_taken_for_a_gap():
    strip, box = _strip()
    busy = strip.copy()
    busy.paste(_textured(30, 400), (290, 0))
    busy.paste(Image.effect_noise((8, 400), 40).convert("RGB"), (304, 0))    # would be a cut at 304
    assert Image.open(io.BytesIO(SP.crop_first_panel(_png(busy), box, _png(strip)))).width == 300


def test_a_painted_over_gap_with_the_panels_intact_is_cut_at_the_edge():
    # live, twice: purple with soft edges, no flat column anywhere near
    strip, box = _strip()
    render = strip.copy()
    render.paste(_textured(40, 400), (290, 0))
    assert Image.open(io.BytesIO(SP.crop_first_panel(_png(render), box, _png(strip)))).width == 300


@pytest.mark.parametrize("bg,wall", [((120, 120, 120), (120, 120, 120)), ((210, 200, 180), (210, 200, 180)),
                                     ((20, 20, 20), (20, 20, 20)), ((255, 255, 255), (245, 245, 245))])
def test_a_plain_wall_is_not_the_other_panel_even_at_the_same_brightness(bg, wall):
    # review: a brightness check cut these, and the second person was lost
    strip, box = _strip(bg=bg)
    render = Image.new("RGB", strip.size, wall)
    render.paste(strip.crop((0, 0, 300, 400)), (0, 0))
    assert SP.crop_first_panel(_png(render), box, _png(strip)) is None


def test_another_figure_on_the_same_backdrop_is_not_the_other_panel():
    # a different figure elsewhere. One with nearly the same outline in the same
    # place can pass the check, and that is why the cut keeps the whole render.
    strip, box = _strip(bg=(120, 120, 120))
    render = Image.new("RGB", strip.size, (120, 120, 120))
    render.paste(strip.crop((0, 0, 300, 400)), (0, 0))
    render.paste((240, 230, 90), (620, 20, 720, 200))
    assert SP.crop_first_panel(_png(render), box, _png(strip)) is None


def test_a_repainted_background_is_still_the_kept_layout():
    # live, seed 11: the gap and the white bands turned purple; the panel with the
    # person was untouched and the edit was good
    strip, box = _strip()
    render = Image.new("RGB", strip.size, (130, 40, 200))
    render.paste(strip.crop((0, 0, 300, 400)), (0, 0))
    person = _person(428, 400)
    mask = person.convert("L").point(lambda v: 255 if v < 240 else 0)
    render.paste(person, (308, 0), mask)
    assert Image.open(io.BytesIO(SP.crop_first_panel(_png(render), box, _png(strip)))).width == 300


def test_a_few_cells_of_content_are_too_few_to_trust():
    w, h, edge, gap = 736, 416, 300, 8
    strip = Image.new("RGB", (w, h), (255, 255, 255))
    strip.paste(_textured(edge, h), (0, 0))
    strip.paste((0, 0, 0), (450, 200, 466, 216))
    assert SP.crop_first_panel(_png(strip), (edge / w, 0.0, 1.0, gap / w), _png(strip)) is None


def test_light_content_counts_as_content():
    # pale line art or a pale document beside the first picture
    w, h, edge, gap = 736, 416, 300, 8
    strip = Image.new("RGB", (w, h), (255, 255, 255))
    strip.paste(_textured(edge, h), (0, 0))
    for k in range(40):
        shade = 170 + (k * 13) % 55
        x, y = 320 + (k % 10) * 40, 10 + (k // 10) * 100
        strip.paste((shade, shade, shade), (x, y, x + 36, y + 90))
    assert Image.open(io.BytesIO(SP.crop_first_panel(_png(strip), (edge / w, 0.0, 1.0, gap / w), _png(strip)))).width == edge


@pytest.mark.parametrize("patches,cuts", [(0, False), (1, False), (24, True)])
def test_too_little_content_beside_the_first_is_no_evidence(patches, cuts):
    w, h, edge, gap = 736, 416, 300, 8
    strip = Image.new("RGB", (w, h), (255, 255, 255))
    strip.paste(_textured(edge, h), (0, 0))
    for k in range(patches):
        shade = 10 + (k * 37) % 200
        x, y = 320 + (k % 12) * 34, 20 + (k // 12) * 60
        size = 6 if patches == 1 else 30
        strip.paste((shade, shade, shade), (x, y, x + size, y + size))
    out = SP.crop_first_panel(_png(strip), (edge / w, 0.0, 1.0, gap / w), _png(strip))
    assert (out is not None) is cuts


def test_the_correlation_limit_sits_between_what_was_measured():
    assert 0.134 < SP._OTHERS_MIN_CORRELATION < 0.846     # lost layouts' best, kept layouts' worst


@pytest.mark.parametrize("sizes", [[(5000, 10), (300, 400)], [(400, 4000), (400, 4000)]])
async def test_a_shape_the_node_cannot_render_is_refused_before_it_runs(tmp_path, sizes):
    for n, (w, h) in zip(("a.jpg", "b.jpg"), sizes):
        (tmp_path / n).write_bytes(_jpeg(w, h))
    llm = _llm()
    out = await tool_generate_image(prompt="p", reference_images=["a.jpg", "b.jpg"], keep_first=True,
                                    llm_client=llm, sandbox_dir=tmp_path)
    assert out.startswith("ERROR: could not prepare the reference pictures")
    llm.generate_image.assert_not_called()


def test_a_rotated_and_transparent_picture_join_as_they_look(tmp_path):
    im = Image.new("RGB", (200, 400), RED)
    exif = Image.Exif()
    exif[0x0112] = 6                                     # stored sideways
    buf = io.BytesIO()
    im.save(buf, format="JPEG", exif=exif)
    rot = tmp_path / "r.jpg"
    rot.write_bytes(buf.getvalue())
    clear = Image.new("RGBA", (300, 400), (0, 0, 0, 0))
    clear.paste((20, 20, 200, 255), (100, 100, 200, 300))
    cl = tmp_path / "c.png"
    cl.write_bytes(_png(clear))
    for first, second in ((rot, cl), (cl, rot)):
        data, (edge, _, _, _) = SP.combine_keep_first([first, second])
        strip = Image.open(io.BytesIO(data)).convert("RGB")
        fw = round(edge * strip.width)
        if first is rot:
            assert fw == 2 * SP.COMBINED_HEIGHT * 2 // 3 or abs(fw - 1536) <= 2
            assert strip.getpixel((strip.width - 8, 8))[0] > 240     # transparent is white, not black
        else:
            assert strip.getpixel((8, 8))[0] > 240
            assert _near(strip.getpixel((fw + SP.COMBINED_GAP + 4, strip.height // 2)), RED)
            assert strip.getpixel((fw + SP.COMBINED_GAP + 4, 8))[0] > 240   # wider than tall: letterboxed


def test_an_unreadable_render_means_no_cut():
    ok = _png(Image.new("RGB", (64, 64)))
    assert SP.crop_first_panel(b"not an image", (0.5, 0.0, 1.0, 0.01), ok) is None
    assert SP.crop_first_panel(ok, (0.5, 0.0, 1.0, 0.01), b"not an image") is None


# ── what else reads the arguments ───────────────────────────────────────────

@pytest.mark.parametrize("args,expect", [
    ({"prompt": "p", "reference_images": ["a"]}, "single"),
    ({"prompt": "p", "reference_images": "a"}, "single"),
    ({"prompt": "p", "reference_images": ["a", "b"]}, "ladder"),
    ({"prompt": "p", "reference_images": ["a", "a"]}, "single"),
    ({"prompt": "p", "input_image": ["a", "b"]}, "ladder"),
    ({"prompt": "p", "reference_images": ["a", "b"], "keep_first": True}, "single"),
    ({"prompt": "p", "reference_images": ["a"], "subjects": ["X Y"]}, "ladder"),
    ({"prompt": "p", "reference_images": ["a"], "subjects": ["X Y"], "keep_first": True}, "single"),
    ({"prompt": "p", "subjects": ["X Y", "Z W"], "keep_first": True}, "ladder"),   # no file: the flag does nothing
])
def test_the_time_estimate_matches_what_will_run(args, expect, monkeypatch):
    monkeypatch.setattr(IG, "render_look_enabled", lambda: False)
    IG.RECENT_RENDER_SECONDS["edit"].clear()
    one = IG.DEFAULT_RENDER_SECONDS["edit"]
    assert expected_render_seconds(args) == (one if expect == "single" else one * 2 + IG._verdict_seconds())


def test_the_model_is_told_about_joining_where_it_chooses_tools():
    from unittest.mock import MagicMock
    from ghost_agent.core.prompts import SYSTEM_PROMPT
    from ghost_agent.tools.registry import get_active_tool_definitions
    ctx = MagicMock()
    ctx.llm_client.image_gen_clients = ["http://gpu"]
    fn = next(t for t in get_active_tool_definitions(ctx) if t["function"]["name"] == "image_generation")["function"]
    props = fn["parameters"]["properties"]
    assert props["reference_images"]["maxItems"] == IG.MAX_JOINED_REFERENCES
    assert props["keep_first"]["type"] == "boolean"
    assert "never build a montage" in fn["description"] and "keep_first" in fn["description"]
    assert "never make a montage yourself" in SYSTEM_PROMPT
    assert "One reference on this node" not in props["reference_images"]["description"]
