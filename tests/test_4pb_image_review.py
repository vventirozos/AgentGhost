"""§4PB (2026-10-14) — fresh-eye review of image generation, editing and
multi-step edits: pins for every fix.

Tool: a failed photo DOWNLOAD is transient, never "no photo of this person";
the lookup takes the name as a title (redirects followed) before searching,
prefers exact titles, and lets ONE token of a multi-word name be one edit
off when the surname anchors it; the record names the article each photo
came from; a wide panel is letterboxed, never cropped; a picture from
another scope is still found; each render leg books its own time; the
transparent claim is read from the PNG; `subject` beside a prompt is the
subjects list; a non-dict node answer is an honest error; the scene keeps
its shape.  Node: references are letterboxed, not stretched; the key is
checked before the body; preflight errors are scrubbed and 503 while
retrying; image_size never decodes pixels.  Agent: an edit is judged
before/after against the picture it started from.
"""
import base64
import io
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

import ghost_agent.tools.subject_photos as SP
import ghost_agent.tools.image_gen as IG
from ghost_agent.tools.image_gen import tool_generate_image
import ghost_agent.core.agent as A


def _jpeg(w=300, h=400, colour=(200, 50, 50)) -> bytes:
    buf = io.BytesIO(); Image.new("RGB", (w, h), colour).save(buf, format="JPEG"); return buf.getvalue()


def _png(w=768, h=512, colour_type=2) -> bytes:
    return (b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0dIHDR" + w.to_bytes(4, "big") + h.to_bytes(4, "big")
            + bytes([8, colour_type, 0, 0, 0]) + b"xx")


def _resp(png=None, seed=7):
    return {"data": [{"b64_json": base64.b64encode(png or _png()).decode()}], "width": 768, "height": 512, "seed": seed}


def _llm(verdict="NO"):
    llm = AsyncMock(); llm.image_gen_clients = [{"model": "Ghost"}]
    llm.generate_image.return_value = _resp()
    llm.chat_completion.return_value = {"choices": [{"message": {"content": json.dumps(
        {"collage": verdict, "confidence": 0.95, "evidence": "e"})}}]}
    return llm


def _sent(llm, n=0):
    return llm.generate_image.call_args_list[n].args[0]


@pytest.fixture
def web(monkeypatch):
    photos = {"Alexis Tsipras": _jpeg(300, 400, (10, 10, 10)), "Antonis Samaras": _jpeg(400, 300, (240, 240, 240))}
    state = {"download": "ok", "lookups": []}

    def lookup(subject, proxy):
        state["lookups"].append(subject)
        if subject not in photos:
            return None
        return (f"https://img.example/{subject}.jpg", subject, f"https://en.wikipedia.org/wiki/{subject}")

    async def download(url, sandbox_dir, tor_proxy, filename=None):
        if state["download"] == "fail":
            return "Error: Failed after 3 attempts over Tor — HTTP 503"
        name = url.rsplit("/", 1)[1][:-4]
        data = photos[name] if state["download"] == "ok" else b"\xff\xd8\xff\xe0 truncated"
        (Path(sandbox_dir) / filename).write_bytes(data)
        return f"SUCCESS: Downloaded '{url}' to '{filename}'."

    monkeypatch.setattr(SP, "lookup_photo_url", lookup)
    import ghost_agent.tools.file_system as FS
    monkeypatch.setattr(FS, "tool_download_file", download)
    return state


TWO = ["Alexis Tsipras", "Antonis Samaras"]


# ── 1. a failed download is transient, never "upload a photo of this famous person" ──

@pytest.mark.parametrize("mode", ["fail", "truncated"])
async def test_a_failed_or_truncated_download_is_transient_not_missing(tmp_path, web, mode):
    web["download"] = mode
    llm = _llm()
    out = await tool_generate_image(prompt="k", subjects=["Alexis Tsipras"], llm_client=llm, sandbox_dir=tmp_path)
    assert out.startswith("ERROR: the photo could not be downloaded right now for Alexis Tsipras")
    assert "NOT a missing photo" in out and "Call image_generation again once" in out
    assert "upload" not in out.lower() and "[subjects:" not in out       # no designed stop, no name block
    assert getattr(out, "reason_code", None) != "subject_photo_missing"
    llm.generate_image.assert_not_called()


async def test_no_article_is_still_the_designed_stop(tmp_path, web):
    out = await tool_generate_image(prompt="k", subjects=["Nobody Atall"], llm_client=_llm(), sandbox_dir=tmp_path)
    assert "no usable photo found for Nobody Atall" in out and "[subjects:" in out


def test_every_raise_in_the_fetch_is_one_of_the_three_classes():
    """R1 enumeration: the tool maps SearchUnavailable → retry, PhotoUnavailable →
    retry, plain LookupError → ask the user. A fourth kind would fall into the
    'ask the user' arm unseen."""
    import ast, inspect
    tree = ast.parse(inspect.getsource(SP))
    # §4QG-b: the fetch is split — the cache/stale wrapper and _fetch_fresh
    fns = [n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)
           and n.name in ("fetch_subject_photo", "_fetch_fresh")]
    assert len(fns) == 2
    kinds = set()
    for n in (x for fn in fns for x in ast.walk(fn)):
        if isinstance(n, ast.Raise) and n.exc is not None:
            call = n.exc
            kinds.add(call.func.id if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) else "?")
    assert kinds == {"LookupError", "PhotoUnavailable"}, kinds       # SearchUnavailable is re-raised from lookup
    assert issubclass(SP.PhotoUnavailable, LookupError) and issubclass(SP.SearchUnavailable, LookupError)


# ── 2. the lookup: title first, exact before superset, anchored one-edit tokens ──

@pytest.mark.parametrize("subject,title,ok", [
    ("Zoi Konstantopoulou", "Zoe Konstantopoulou", True),      # live slack-8d / slack-15: refused twice
    ("Zoe Konstantopoulou", "Zoi Konstantopoulou", True),
    ("Obama", "Osama bin Laden", False),                        # single token: exact only
    ("Obama", "Michelle Obama", True),                          # superset still accepted (search ranks)
    ("Alexis Tsipras", "First Cabinet of Alexis Tsipras", True),
    ("Alexis Tsipras and Antonis Samaras", "Alexis Tsipras", False),
    ("Zoi Konstantopoulu", "Zoe Konstantopoulou", False),        # two tokens off
    ("Zoia Konstantopoulou", "Zoe Konstantopoulou", False),      # one token, TWO edits off
    ("Zoi Konstantopoulou", "Zey Konstantopoulou", False),       # one token, two SUBSTITUTIONS
    ("Zoe Konstantopoulou", "Zoey Konstantopoulou", False),      # §4QG-b: an added letter is another name
    ("Ana Smith", "Anna Smyth", False),                          # the anchor itself must be exact
])
def test_title_matches_table(subject, title, ok):
    assert SP.title_matches(subject, title) is ok


def test_title_is_exact_ignores_a_qualifier():
    assert SP.title_is_exact("Toyota Corolla", "Toyota Corolla (E210)")
    assert not SP.title_is_exact("Obama", "Barack Obama")


def _creq(monkeypatch, answers):
    """Fake Wikipedia: answers[(lang, kind)] → query dict; records the calls."""
    import curl_cffi.requests as creq
    calls = []

    class R:
        status_code = 200

        def __init__(self, q): self._q = q

        def json(self): return {"query": self._q}

    def get(url, params=None, **kw):
        lang = url.split("/")[2].split(".")[0]
        kind = "titles" if "titles" in params else "search"
        calls.append((lang, kind, params.get("titles") or params.get("gsrsearch")))
        return R(answers.get((lang, kind), {"pages": {}}))
    monkeypatch.setattr(creq, "get", get)
    return calls


def test_a_redirect_from_the_exact_name_is_trusted_before_any_search(monkeypatch):
    calls = _creq(monkeypatch, {("en", "titles"): {
        "redirects": [{"from": "Napoleon Bonaparte", "to": "Napoleon"}],
        "pages": {"1": {"title": "Napoleon", "thumbnail": {"source": "https://x/nap.jpg"}}}}})
    assert SP.lookup_photo_url("Napoleon Bonaparte", None) == ("https://x/nap.jpg", "Napoleon", "https://en.wikipedia.org/wiki/Napoleon")
    assert [c[1] for c in calls] == ["titles"]                              # no search needed


def test_a_title_hit_that_is_neither_exact_nor_a_redirect_falls_through_to_search(monkeypatch):
    calls = _creq(monkeypatch, {
        ("en", "titles"): {"pages": {"1": {"title": "Obama", "thumbnail": {"source": "https://x/surname-page.jpg"}}}},
        ("en", "search"): {"pages": {"1": {"index": 1, "title": "Barack Obama", "thumbnail": {"source": "https://x/barack.jpg"}}}}})
    # "Obama" IS exact for the page titled "Obama" (a surname article) — that is accepted as exact
    assert SP.lookup_photo_url("Obama", None)[1] == "Obama"
    calls.clear()
    monkeypatch.setattr(SP, "title_is_exact", lambda s, t: False)
    assert SP.lookup_photo_url("Obama", None)[1] == "Barack Obama" and [c[1] for c in calls] == ["titles", "search"]


def test_search_prefers_the_exact_title_over_an_earlier_superset_hit(monkeypatch):
    _creq(monkeypatch, {("en", "search"): {"pages": {
        "1": {"index": 1, "title": "Konstantinos Mitsotakis", "thumbnail": {"source": "https://x/father.jpg"}},
        "2": {"index": 2, "title": "Kyriakos Mitsotakis", "thumbnail": {"source": "https://x/pm.jpg"}}}}})
    assert SP.lookup_photo_url("Kyriakos Mitsotakis", None)[2].endswith("/Kyriakos_Mitsotakis")
    assert SP.lookup_photo_url("Mitsotakis", None)[1] == "Konstantinos Mitsotakis"   # no exact: search order wins
    # an EXACT title later in the search order beats an earlier superset hit
    _creq(monkeypatch, {("en", "search"): {"pages": {
        "1": {"index": 1, "title": "Konstantinos Mitsotakis", "thumbnail": {"source": "https://x/father.jpg"}},
        "2": {"index": 2, "title": "Mitsotakis (surname)", "thumbnail": {"source": "https://x/crest.jpg"}}}}})
    assert SP.lookup_photo_url("Mitsotakis", None)[1] == "Mitsotakis (surname)"


def test_a_missing_title_page_is_skipped(monkeypatch):
    # a missing page carries no thumbnail — that is what skips it
    _creq(monkeypatch, {("en", "titles"): {"pages": {"-1": {"title": "Nobody Atall", "missing": ""}}}})
    assert SP.lookup_photo_url("Nobody Atall", None) is None


# ── 3. the article is on record; a surname match is called out ──

async def test_the_record_names_the_article_and_flags_a_surname_match(tmp_path, web):
    llm = _llm()
    out = await tool_generate_image(prompt="k", subjects=TWO, llm_client=llm, sandbox_dir=tmp_path)
    assert "Photos from Wikipedia: Alexis Tsipras → Alexis Tsipras; Antonis Samaras → Antonis Samaras." in out
    assert "matched the article" not in out
    # the agent's parser still reads the record line unchanged
    m = A._SUBJECT_REFERENCE_RE.search(out)
    assert m and m.group(2) == "Alexis Tsipras (ref_alexis_tsipras.jpg), Antonis Samaras (ref_antonis_samaras.jpg)"
    # a surname that matched a fuller title is flagged
    meta = tmp_path / SP.PHOTO_DIR / "ref_alexis_tsipras.jpg.json"
    meta.write_text(json.dumps({**json.loads(meta.read_text()), "title": "First Cabinet of Alexis Tsipras"}))
    out2 = await tool_generate_image(prompt="k", subjects=["Alexis Tsipras"], llm_client=_llm(), sandbox_dir=tmp_path)
    assert "NOTE: 'Alexis Tsipras' matched the article 'First Cabinet of Alexis Tsipras'" in out2


async def test_a_cached_photo_logs_its_article(tmp_path, web, capsys):
    await tool_generate_image(prompt="k", subjects=["Alexis Tsipras"], llm_client=_llm(), sandbox_dir=tmp_path)
    capsys.readouterr()
    await tool_generate_image(prompt="k", subjects=["Alexis Tsipras"], llm_client=_llm(), sandbox_dir=tmp_path)
    assert "(Alexis Tsipras, cached)" in capsys.readouterr().out


# ── 4. panels are never cropped; absurd aspects are refused before a canvas ──

def test_a_wide_panel_is_letterboxed_not_cropped(tmp_path):
    wide = tmp_path / "w.jpg"; wide.write_bytes(_jpeg(2000, 500, (10, 10, 10)))
    tall = tmp_path / "t.jpg"; tall.write_bytes(_jpeg(300, 400, (240, 240, 240)))
    im = Image.open(io.BytesIO(SP.combine_side_by_side([wide, tall])))
    cap = round(SP.COMBINED_HEIGHT * 1.5)
    assert im.width == cap + round(300 * 1024 / 400) + SP.COMBINED_GAP
    # the wide picture is whole: dark across its full capped width at mid-height, white bands above/below
    mid = im.height // 2
    assert im.getpixel((5, mid))[0] < 40 and im.getpixel((cap - 5, mid))[0] < 40
    assert im.getpixel((cap // 2, 10))[0] > 240 and im.getpixel((cap // 2, im.height - 10))[0] > 240


@pytest.mark.parametrize("w,h", [(200000, 1), (1, 20000), (10000, 100)])
async def test_an_absurd_shape_is_refused_before_any_canvas(tmp_path, web, w, h):
    llm = _llm()
    out = await tool_generate_image(prompt="k", subjects=TWO, width=w, height=h, llm_client=llm, sandbox_dir=tmp_path)
    assert out.startswith("ERROR: the requested shape") and "between 1:4 and 4:1" in out
    llm.generate_image.assert_not_called()


def test_pad_to_aspect_refuses_outside_the_band():
    with pytest.raises(ValueError):
        SP._pad_to_aspect(Image.new("RGB", (10, 10)), 50.0)


# ── 5. a picture from the root sandbox is found from a project scope ──

async def test_a_picture_made_at_the_root_is_still_an_edit_reference_inside_a_project(tmp_path):
    root = tmp_path; proj = root / "projects" / "p1"; proj.mkdir(parents=True)
    (root / "gen_old.png").write_bytes(_png())
    assert IG._resolve_reference("gen_old.png", proj) == (root / "gen_old.png").resolve()
    assert IG._resolve_reference("/api/download/gen_old.png", proj) == (root / "gen_old.png").resolve()
    # the project's own file wins when both exist; traversal is still refused from both roots
    (proj / "gen_old.png").write_bytes(_png())
    assert IG._resolve_reference("gen_old.png", proj) == (proj / "gen_old.png").resolve()
    with pytest.raises(ValueError):
        IG._resolve_reference("../../etc/passwd", proj)
    with pytest.raises(ValueError):
        IG._resolve_reference("../p2/gen_x.png", proj)


# ── 6. each render leg books its own time ──

async def test_a_three_render_ladder_books_three_legs_not_one_wall(tmp_path, web, monkeypatch):
    monkeypatch.setattr(IG, "COLLAGE_CFG_LEG", True)        # §4VJ: the cfg-4 leg is opt-in
    IG.RECENT_RENDER_SECONDS["edit"].clear()
    IG.RECENT_RENDER_SECONDS["edit_cfg"].clear()
    llm = _llm()
    llm.chat_completion.side_effect = [{"choices": [{"message": {"content": json.dumps({"collage": v, "confidence": 0.95, "evidence": "e"})}}]}
                                       for v in ("YES", "YES", "NO")]
    await tool_generate_image(prompt="k", subjects=TWO, llm_client=llm, sandbox_dir=tmp_path)
    # §4QG-b: the cfg-4 leg is booked under its own key, not as an "edit"
    assert llm.generate_image.await_count == 3 and len(IG.RECENT_RENDER_SECONDS["edit"]) == 2
    assert len(IG.RECENT_RENDER_SECONDS["edit_cfg"]) == 1
    assert max(IG.RECENT_RENDER_SECONDS["edit"]) < 5.0


async def test_the_default_ladder_books_every_leg_as_an_edit(tmp_path, web):
    # §4VJ: with the cfg-4 leg off, attempt 3 is a plain edit — booked as one
    IG.RECENT_RENDER_SECONDS["edit"].clear()
    IG.RECENT_RENDER_SECONDS["edit_cfg"].clear()
    assert IG.COLLAGE_CFG_LEG is False
    llm = _llm()
    llm.chat_completion.side_effect = [{"choices": [{"message": {"content": json.dumps({"collage": v, "confidence": 0.95, "evidence": "e"})}}]}
                                       for v in ("YES", "YES", "NO")]
    await tool_generate_image(prompt="k", subjects=TWO, llm_client=llm, sandbox_dir=tmp_path)
    assert llm.generate_image.await_count == 3 and len(IG.RECENT_RENDER_SECONDS["edit"]) == 3
    assert len(IG.RECENT_RENDER_SECONDS["edit_cfg"]) == 0


# ── 7. transparent: from the bytes, never with a reference ──

async def test_transparent_is_dropped_with_a_reference_and_said_so(tmp_path):
    (tmp_path / "me.jpg").write_bytes(_jpeg())
    llm = _llm()
    out = await tool_generate_image(prompt="k", reference_images=["me.jpg"], transparent=True, llm_client=llm, sandbox_dir=tmp_path)
    assert "transparent" not in _sent(llm) and "`transparent` was ignored" in out
    assert "alpha channel" not in out


def _real_png(mode, colour) -> bytes:
    im = Image.new(mode, (64, 64), colour)
    if mode in ("RGBA", "LA"):           # an opaque subject in the middle
        im.paste((200, 30, 30, 255) if mode == "RGBA" else (90, 255), (16, 16, 48, 48))
    buf = io.BytesIO(); im.save(buf, format="PNG"); return buf.getvalue()


def test_png_alpha_probe():
    # §4QG: the old pin used header-only colour-type bytes and passed
    # vacuously — the node writes RGBA (type 6) for EVERY picture, so the
    # decision is made from the pixels
    assert IG._png_has_alpha(_real_png("RGBA", (0, 0, 0, 0))) and IG._png_has_alpha(_real_png("LA", (0, 0)))
    assert not IG._png_has_alpha(_real_png("RGBA", (10, 200, 10, 240)))     # RGBA but opaque (min alpha 183–253 live)
    assert not IG._png_has_alpha(_real_png("RGB", (10, 200, 10)))
    assert not IG._png_has_alpha(_png(colour_type=6)) and not IG._png_has_alpha(b"\x89PNG")   # undecodable


# ── 8. argument healing and honest errors ──

async def test_subject_beside_a_prompt_is_the_subjects_list(tmp_path, web):
    llm = _llm()
    await tool_generate_image(prompt="k", subject="Alexis Tsipras", llm_client=llm, sandbox_dir=tmp_path)
    assert web["lookups"] == ["Alexis Tsipras"] and len(_sent(llm)["reference_images"]) == 1


async def test_subject_without_a_prompt_is_still_the_prompt(tmp_path, web):
    llm = _llm()
    await tool_generate_image(subject="a red apple", llm_client=llm, sandbox_dir=tmp_path)
    assert web["lookups"] == [] and _sent(llm)["prompt"] == "a red apple"


@pytest.mark.parametrize("bad", [{"a": 1}, 7, True])
async def test_a_wrong_typed_subjects_is_an_error_not_a_stranger(tmp_path, web, bad):
    llm = _llm()
    out = await tool_generate_image(prompt="k", subjects=bad, llm_client=llm, sandbox_dir=tmp_path)
    assert out.startswith("ERROR: `subjects` must be a list of names")
    llm.generate_image.assert_not_called()


async def test_a_non_dict_node_answer_is_the_no_image_error(tmp_path):
    llm = _llm(); llm.generate_image.return_value = ["not", "a", "dict"]
    out = await tool_generate_image(prompt="k", llm_client=llm, sandbox_dir=tmp_path)
    assert out.startswith("ERROR: image generation returned no image data")


async def test_the_limit_error_caps_the_echo_and_steers_the_add_case(tmp_path, web):
    (tmp_path / "gen_aa.png").write_bytes(_png())
    out = await tool_generate_image(prompt="add", reference_images=["gen_aa.png"], subjects=TWO + ["Parthenon"],
                                    llm_client=_llm(), sandbox_dir=tmp_path)
    assert "got 4" in out and "list only the NEW person in `subjects`" in out
    names = [f"Person {i}" for i in range(40)]
    out2 = await tool_generate_image(prompt="k", subjects=names, llm_client=_llm(), sandbox_dir=tmp_path)
    assert "Person 4, …" in out2 and "Person 39" not in out2 and len(out2) < 600


async def test_a_long_negative_prompt_is_cut_with_a_note(tmp_path):
    llm = _llm()
    out = await tool_generate_image(prompt="k", negative_prompt="n" * 9000, llm_client=llm, sandbox_dir=tmp_path)
    assert len(_sent(llm)["negative_prompt"]) == 8000 and "negative prompt was truncated" in out


# ── 9. the scene keeps its shape and its wording ──

async def test_the_scene_keeps_its_own_shape_and_the_result_says_what_it_was_built_from(tmp_path, web):
    buf = io.BytesIO(); Image.new("RGB", (512, 768), (240, 240, 240)).save(buf, format="PNG")
    (tmp_path / "gen_p.png").write_bytes(buf.getvalue())
    llm = _llm()
    out = await tool_generate_image(prompt="add him", reference_images=["gen_p.png"], subjects=["Alexis Tsipras"],
                                    llm_client=llm, sandbox_dir=tmp_path)
    assert (_sent(llm)["width"], _sent(llm)["height"]) == (512, 768)          # not the strip's 768x512
    assert "built from the existing picture and the subjects' photos" in out
    assert "what you ASKED the model to change" in out


# ── 10. an edit's record and the agent's before/after evidence ──

async def test_a_plain_edit_records_what_it_edited_from_and_the_agent_uses_it(tmp_path):
    (tmp_path / "gen_a.png").write_bytes(_png())
    out = await tool_generate_image(prompt="remove the hat", reference_images=["gen_a.png"], llm_client=_llm(), sandbox_dir=tmp_path)
    assert "EDITED FROM: gen_a.png" in out
    rows = [{"name": "image_generation", "content": out}]
    assert A._turn_edit_reference(rows) == "gen_a.png"
    before, likeness = A.GhostAgent._likeness_evidence(rows, tmp_path, None)
    assert before == str((tmp_path / "gen_a.png")) and likeness is None
    # a user's own screenshot still wins
    assert A.GhostAgent._likeness_evidence(rows, tmp_path, "/u/shot.png") == ("/u/shot.png", None)
    # a subjects render keeps likeness mode (no EDITED FROM record)
    assert A._turn_edit_reference([{"name": "image_generation", "content": "SUCCESS: Image generated… REFERENCE PHOTO: x — A (a.jpg). THIS WAS A NEW SCENE"}]) is None


def test_parentheses_in_a_filename_do_not_break_the_record():
    assert IG._plain("me (1).jpg") == "me _1_.jpg" and "\n" not in IG._plain("a\nb")


# ── 11. the node ──

def _node():
    import importlib.util, sys
    path = Path(__file__).resolve().parent.parent / "interface" / "externals" / "image_generation" / "img_gen_server.py"
    spec = importlib.util.spec_from_file_location("img_gen_server_4pb", path)
    mod = importlib.util.module_from_spec(spec); sys.modules["img_gen_server_4pb"] = mod; spec.loader.exec_module(mod)
    return mod


def test_fit_reference_letterboxes_instead_of_stretching():
    node = _node()
    buf = io.BytesIO(); Image.new("RGB", (960, 1264), (10, 10, 10)).save(buf, format="JPEG")
    out = Image.open(io.BytesIO(node.fit_reference(buf.getvalue(), 768, 512)))
    assert out.size == (768, 512)
    # the dark portrait sits in the middle at its own aspect; white bands left and right
    assert out.getpixel((384, 256))[0] < 40 and out.getpixel((10, 256))[0] > 240 and out.getpixel((757, 256))[0] > 240
    assert abs((out.getpixel((384 - 190, 256))[0] < 40) - (out.getpixel((384 + 190, 256))[0] < 40)) == 0


def test_image_size_reads_the_header_and_exif_only(monkeypatch):
    node = _node()
    from PIL import ImageOps
    called = {"n": 0}
    monkeypatch.setattr(ImageOps, "exif_transpose", lambda *a, **k: called.__setitem__("n", called["n"] + 1))
    buf = io.BytesIO(); im = Image.new("RGB", (640, 480)); ex = im.getexif(); ex[0x0112] = 6
    im.save(buf, format="JPEG", exif=ex.tobytes())
    assert node.image_size(buf.getvalue()) == (480, 640) and called["n"] == 0


def test_preflight_errors_are_scrubbed_and_503_while_retrying(monkeypatch):
    node = _node()
    secret = str(node.SD_CLI)                                 # a configured path, as sd-cli's tail prints it
    monkeypatch.setattr(node, "_load_error", f"RuntimeError: {secret} exited 1")
    monkeypatch.setattr(node, "_preflight_done", False)
    e = node._preflight_error(); assert e.status_code == 503 and secret not in e.detail and Path(secret).name in e.detail
    monkeypatch.setattr(node, "_preflight_done", True)
    e = node._preflight_error(); assert e.status_code == 500 and secret not in e.detail


async def test_the_key_is_checked_before_the_body_is_read(monkeypatch):
    node = _node()
    monkeypatch.setattr(node, "API_KEY", "secret")
    inner_calls = []

    async def inner(scope, receive, send):
        inner_calls.append(scope["path"])
    mw = node._BodyCapMiddleware(inner)
    sent = []
    received = {"n": 0}

    async def receive():
        received["n"] += 1
        return {"type": "http.request", "body": b"x" * 100, "more_body": False}

    async def send(msg): sent.append(msg)
    await mw({"type": "http", "method": "POST", "path": "/v1/images/generations", "headers": [(b"content-length", b"100")]}, receive, send)
    assert inner_calls == [] and received["n"] == 0 and sent[0]["status"] == 401
    sent.clear()
    await mw({"type": "http", "method": "POST", "path": "/v1/images/generations",
              "headers": [(b"content-length", b"100"), (node.API_KEY_NAME.lower().encode(), b"secret")]}, receive, send)
    assert inner_calls == ["/v1/images/generations"]


def test_the_costly_record_recognises_the_image_tool_by_its_result_too():
    # an alias tool name ("imagegen") used to miss the record
    a = A.GhostAgent.__new__(A.GhostAgent); a._costly_turns = {}
    assert a._note_costly_turn("conv-x", [{"name": "imagegen", "content": "SUCCESS: Image generated and saved…"}]) is True
    assert a._note_costly_turn("conv-y", [{"name": "imagegen", "content": "ERROR: nothing"}]) is False
