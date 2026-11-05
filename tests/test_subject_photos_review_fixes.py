"""§4QG (2026-10-21) — review findings in the `subjects` photo lookup, pinned.

1. a FAILED query is never read as "no match" (an answered title query no
   longer makes a failed search mean "no article");
2. the transliteration allowance accepts only transliteration-like spellings
   ("Zoi"/"Zoe"), never another real name ("Mark"/"Mary", "Jon"/"John"), and
   a fuzzy hit ranks below exact / full-name hits;
3. a redirect is trusted only when its target is still the person (no
   section redirects, no "Family of Barack Obama");
4. transparency is composited onto white, 16-bit greyscale is scaled;
5. a singular `subject=` used as the subjects list is scrubbed / guarded;
6. "Martin Luther King, Jr." and "Simon & Garfunkel" stay one name;
7. leading titles, joined initials and ß are matched;
8. an unsupported-format photo is permanent, a disk error is the machine's;
9. the cache expires and long names do not share a file;
10. one lookup budget per subject.

§4QG-b (fresh-eye review of the above, same day):
  * a redirect FROM the typed name is trusted again unless it is a section
    redirect / group-or-event article / surname page (Kate Middleton,
    Meghan Markle);
  * "and"/"&"/"και" always split a string again (Tsipras and Mitsotakis);
  * the transliteration swap is word-final only, no added letter;
  * 60 s budget, attempt 2 only with a full query timeout left;
  * an expired photo is used when its re-fetch fails transiently;
  * float ("F") photos are scaled, not black;
  * the tool and the egress guard compare `subject` to the same prompt.
"""
import base64
import io
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

import ghost_agent.tools.subject_photos as SP
from ghost_agent.tools.image_gen import tool_generate_image


# ── helpers ─────────────────────────────────────────────────────────────────

def _img(mode="RGB", size=(300, 400), colour=(200, 50, 50), fmt="JPEG", **save) -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, colour).save(buf, format=fmt, **save)
    return buf.getvalue()


def _llm():
    llm = AsyncMock()
    llm.image_gen_clients = [{"model": "Ghost"}]
    llm.generate_image.return_value = {"data": [{"b64_json": base64.b64encode(b"\x89PNG\r\n\x1a\nxx").decode()}],
                                       "width": 768, "height": 512, "seed": 7}
    return llm


@pytest.fixture
def web(monkeypatch):
    """Fake encyclopedia + downloader: state["photo"] is what every download serves."""
    state = {"lookup": [], "download": [], "photo": _img()}

    def lookup(subject, proxy):
        state["lookup"].append(subject)
        return (f"https://img.example/{len(state['lookup'])}.jpg", subject,
                f"https://en.wikipedia.org/wiki/{subject.replace(' ', '_')}")

    async def download(url, sandbox_dir, tor_proxy, filename=None):
        state["download"].append(url)
        (Path(sandbox_dir) / filename).write_bytes(state["photo"])
        return f"SUCCESS: Downloaded '{url}' to '{filename}'."

    monkeypatch.setattr(SP, "lookup_photo_url", lookup)
    import ghost_agent.tools.file_system as FS
    monkeypatch.setattr(FS, "tool_download_file", download)
    return state


class _R:
    def __init__(self, code=200, q=None):
        self.status_code, self._q = code, q or {"pages": {}}

    def json(self):
        return {"query": self._q}


def _wiki(monkeypatch, answers):
    """answers[(lang, kind)] → query dict, or an int status code; records calls."""
    import curl_cffi.requests as creq
    calls = []

    def get(url, params=None, **kw):
        lang = url.split("/")[2].split(".")[0]
        kind = "titles" if "titles" in params else "search"
        calls.append((lang, kind, params.get("titles") or params.get("gsrsearch"), kw.get("timeout")))
        a = answers.get((lang, kind), {"pages": {}})
        if isinstance(a, BaseException):
            raise a
        return _R(a) if isinstance(a, int) else _R(200, a)
    monkeypatch.setattr(creq, "get", get)
    return calls


def _page(title, src="https://x/p.jpg", index=1, **extra):
    return {"index": index, "title": title, "thumbnail": {"source": src}, **extra}


# ── 1. a failed query is not "no match" ─────────────────────────────────────

@pytest.mark.parametrize("search,limited", [(429, True), (503, False), (TimeoutError("t"), False)])
def test_a_failed_search_after_an_answered_title_query_is_unavailable(monkeypatch, search, limited):
    """[MAJOR] answered=True after the title query: a 429 on the SEARCH
    returned None → "no article" → "ask the user for a photo of Stalin"."""
    _wiki(monkeypatch, {("en", "titles"): {"pages": {"-1": {"title": "Joseph Stalin", "missing": ""}}},
                        ("en", "search"): search})
    with pytest.raises(SP.SearchUnavailable) as ei:
        SP.lookup_photo_url("Joseph Stalin", None)
    assert ei.value.rate_limited is limited


def test_greek_answers_no_match_and_english_fails_is_unavailable(monkeypatch):
    calls = _wiki(monkeypatch, {("en", "titles"): 503})
    with pytest.raises(SP.SearchUnavailable) as ei:
        SP.lookup_photo_url("Ζωή Κωνσταντοπούλου", None)
    assert "HTTP 503" in str(ei.value) and not ei.value.rate_limited
    assert [c[:2] for c in calls] == [("el", "titles"), ("el", "search"), ("en", "titles")]


def test_a_hit_still_wins_over_a_failure_and_all_answered_is_none(monkeypatch):
    _wiki(monkeypatch, {("el", "titles"): 429,
                        ("en", "titles"): {"pages": {"1": _page("Ζωή Κωνσταντοπούλου")}}})
    assert SP.lookup_photo_url("Ζωή Κωνσταντοπούλου", None)[1] == "Ζωή Κωνσταντοπούλου"
    _wiki(monkeypatch, {})                         # every query answered, nothing matches
    assert SP.lookup_photo_url("Ζωή Κωνσταντοπούλου", None) is None


# ── 2. transliteration, not another name ────────────────────────────────────

@pytest.mark.parametrize("subject,title,ok", [
    ("Mark Johnson", "Mary Johnson", False),          # consonant swap = another person
    ("Paul Allen", "Paula Allen", False),             # inserted letter = another name
    ("Jon Stewart", "John Stewart", False),
    ("John Stewart", "Jon Stewart", False),
    ("Dana White", "Dina White", False),              # a↔i is not a transliteration class
    ("Zoi Konstantopoulou", "Zoe Konstantopoulou", True),     # refused twice live
    ("Zoe Konstantopoulou", "Zoey Konstantopoulou", False),   # §4QG-b: no added letter
    ("Joe Biden", "Joey Biden", False),                       # §4QG-b: another name
    ("Bill Gates", "Bell Gates", False),                      # §4QG-b: swap inside the word
    ("Lynn Jones", "Linn Jones", False),                      # §4QG-b: swap inside the word
    ("Alexei Navalny", "Alexey Navalny", True),               # word-final i/y
    ("Zoe Konstantopoulou", "Zoi Konstantopoulou", True),     # word-final e/i
    ("Alexei Navalny", "Alexey Navalny", True),
    ("Sophia Loren", "Sofia Loren", True),            # ph ↔ f
    ("Nikos Kazantzakis", "Nicos Kazantzakis", True),  # k ↔ c
    ("Ζωή Κωνσταντοπούλου", "Ζωη Κωνσταντοπουλου", True),
])
def test_the_transliteration_allowance_table(subject, title, ok):
    assert SP.title_matches(subject, title) is ok


def test_a_transliterated_hit_ranks_below_a_full_name_hit(monkeypatch):
    _wiki(monkeypatch, {("en", "search"): {"pages": {
        "1": _page("Alexey Navalny", "https://x/fuzzy.jpg", 1),
        "2": _page("Poisoning of Alexei Navalny", "https://x/full.jpg", 2)}}})   # a superset, not exact
    assert SP.lookup_photo_url("Alexei Navalny", None)[0] == "https://x/full.jpg"
    # alone, the transliterated hit is still found
    _wiki(monkeypatch, {("en", "search"): {"pages": {"1": _page("Alexey Navalny", "https://x/fuzzy.jpg", 1)}}})
    assert SP.lookup_photo_url("Alexei Navalny", None)[0] == "https://x/fuzzy.jpg"


# ── 3. redirects must still be the person ───────────────────────────────────

@pytest.mark.parametrize("redirect", [
    {"from": "Sasha Obama", "to": "Family of Barack Obama"},
    {"from": "Sasha Obama", "to": "Family of Barack Obama", "tofragment": "Malia and Sasha Obama"},
    {"from": "Sasha Obama", "to": "Obama"},                       # a surname page
    {"from": "Sasha Obama", "to": "Sasha Obama Smith", "tofragment": "Early life"},
])
def test_a_redirect_to_someone_else_is_not_her_face(monkeypatch, redirect):
    calls = _wiki(monkeypatch, {("en", "titles"): {"redirects": [redirect], "pages": {
        "1": _page(redirect["to"], "https://x/group.jpg")}}})
    assert SP.lookup_photo_url("Sasha Obama", None) is None
    assert [c[1] for c in calls] == ["titles", "search"]


@pytest.mark.parametrize("name,target", [
    ("Kate Middleton", "Catherine, Princess of Wales"),       # §4QG-b: a formal / married name
    ("Meghan Markle", "Meghan, Duchess of Sussex"),
    ("Lady Gaga", "Lady Gaga (singer)"),
])
def test_a_redirect_from_the_typed_name_to_a_formal_name_is_trusted(monkeypatch, name, target):
    """[CRITICAL §4QG-b] §4QG demanded the target match the typed name: Kate
    Middleton → None ("no photo, upload one"), Meghan Markle → the search's
    "Wedding of Prince Harry and Meghan Markle" (a two-person photo)."""
    calls = _wiki(monkeypatch, {("en", "titles"): {"redirects": [{"from": name, "to": target}], "pages": {
        "1": _page(target, "https://x/her.jpg")}},
        ("en", "search"): {"pages": {"1": _page(f"Wedding of Prince Harry and {name}", "https://x/two.jpg")}}})
    assert SP.lookup_photo_url(name, None)[:2] == ("https://x/her.jpg", target)
    assert [c[1] for c in calls] == ["titles"]


@pytest.mark.parametrize("target,fragment,ok", [
    ("Catherine, Princess of Wales", "", True),
    ("Family of Barack Obama", "Malia and Sasha Obama", False),   # the live Sasha Obama redirect
    ("Family of Barack Obama", "", False),
    ("Wedding of Prince Harry and Meghan Markle", "", False),
    ("Obama", "", False),
    ("Napoleon", "", True),
    ("Some Article", "Early life", False),
])
def test_the_redirect_rule_table(target, fragment, ok):
    name = {"Napoleon": "Napoleon Bonaparte", "Obama": "Sasha Obama"}.get(target, "Sasha Obama")
    if target.startswith("Catherine"):
        name = "Kate Middleton"
    assert SP._redirect_is_the_subject(name, target, fragment) is ok


def test_a_redirect_to_the_shortened_name_is_still_trusted(monkeypatch):
    _wiki(monkeypatch, {("en", "titles"): {"redirects": [{"from": "Napoleon Bonaparte", "to": "Napoleon"}],
                                           "pages": {"1": _page("Napoleon", "https://x/nap.jpg")}}})
    assert SP.lookup_photo_url("Napoleon Bonaparte", None)[1] == "Napoleon"
    _wiki(monkeypatch, {("en", "titles"): {"redirects": [{"from": "Zoi Konstantopoulou", "to": "Zoe Konstantopoulou"}],
                                           "pages": {"1": _page("Zoe Konstantopoulou")}}})
    assert SP.lookup_photo_url("Zoi Konstantopoulou", None)[1] == "Zoe Konstantopoulou"


# ── 4. transparency → white; 16-bit greyscale scaled ────────────────────────

def _cutout(tmp_path, mode="RGBA"):
    """A red figure on a fully transparent background (RGBA, or a palette
    image whose background index is the transparent one)."""
    im = Image.new("RGBA", (300, 400), (0, 0, 0, 0))
    im.paste((200, 30, 30, 255), (100, 100, 200, 300))
    p = tmp_path / f"cut_{mode}.png"
    (im if mode == "RGBA" else _palette(im)).save(p)
    return p


def _palette(im):
    p = im.convert("RGB").convert("P", palette=Image.Palette.ADAPTIVE, colors=4)
    # the corner's palette index becomes the transparent one
    p.info["transparency"] = p.getpixel((0, 0))
    return p


@pytest.mark.parametrize("mode", ["RGBA", "P"])
def test_transparent_photos_are_joined_on_white_not_black(tmp_path, mode):
    cut = _cutout(tmp_path, mode)
    other = tmp_path / "o.jpg"
    other.write_bytes(_img())
    joined = Image.open(io.BytesIO(SP.combine_side_by_side([cut, other])))
    assert min(joined.getpixel((5, 5))) > 240                    # was (0, 0, 0)
    padded = Image.open(io.BytesIO(SP.pad_photo(cut, 1.5)))
    assert min(padded.getpixel((padded.width // 2, 20))) > 240


def test_a_16_bit_greyscale_photo_is_scaled_not_clipped(tmp_path):
    p = tmp_path / "g16.png"
    Image.new("I;16", (200, 300), 32768).save(p)                 # mid grey in 16 bits
    v = Image.open(io.BytesIO(SP.pad_photo(p, 1.0))).convert("L").getpixel((100, 150))
    assert 110 < v < 145                                         # was 255 (white)


async def test_one_transparent_photo_with_no_shape_is_flattened_before_sending(tmp_path, web):
    buf = io.BytesIO()
    im = Image.new("RGBA", (300, 400), (0, 0, 0, 0))
    im.paste((200, 30, 30, 255), (100, 100, 200, 300))
    im.save(buf, format="PNG")
    web["photo"] = buf.getvalue()
    llm = _llm()
    out = await tool_generate_image(prompt="him on the moon", subjects=["Some Logo"], llm_client=llm,
                                    sandbox_dir=tmp_path)
    assert "SUCCESS" in out or "REFERENCE PHOTO" in out, out
    sent = Image.open(io.BytesIO(base64.b64decode(llm.generate_image.call_args.args[0]["reference_images"][0])))
    assert sent.mode == "RGB" and min(sent.getpixel((5, 5))) > 240
    assert sent.size == (300, 400)                               # its own shape, only flattened


async def test_an_opaque_photo_with_no_shape_still_goes_as_its_own_bytes(tmp_path, web):
    llm = _llm()
    await tool_generate_image(prompt="x", subjects=["Some One"], llm_client=llm, sandbox_dir=tmp_path)
    photo = next((tmp_path / SP.PHOTO_DIR).glob("ref_some_one.jpg"))
    assert llm.generate_image.call_args.args[0]["reference_images"] == [base64.b64encode(photo.read_bytes()).decode()]


# ── 5. `subject=` is the subjects list for the egress guards too ────────────

def _profile():
    import re
    pm = MagicMock()
    pm.egress_scrubber = lambda: ([re.compile(r"Kaledonias\s*83", re.I)], "Athens")
    pm.address_values = lambda: ["Kaledonias 83, Athens"]
    pm.load = lambda: {"root": {"name": "Vasilis Example", "address": "Kaledonias 83, Athens"}}
    return pm


def test_a_singular_subject_is_scrubbed_like_subjects():
    from ghost_agent.memory.egress import scrub_tool_args
    ctx = SimpleNamespace(profile_memory=_profile(), egress_profile=None)
    new, changed = scrub_tool_args("image_generation", {"prompt": "my house", "subject": "Kaledonias 83"}, ctx)
    assert changed and "Kaledonias" not in new["subject"] and new["prompt"] == "my house"
    # JSON-string args too
    new, changed = scrub_tool_args("image_generation", json.dumps({"prompt": "my house", "subject": "Kaledonias 83"}), ctx)
    assert changed and "Kaledonias" not in new
    # a `subject` that IS the prompt is the prompt: left alone (scrubbing it
    # alone would make it differ from the prompt and turn it into a search)
    args = {"subject": "my house at Kaledonias 83"}
    assert scrub_tool_args("image_generation", args, ctx) == (args, False)
    args = {"prompt": "my house at Kaledonias 83", "subject": "my house at Kaledonias 83"}
    assert scrub_tool_args("image_generation", args, ctx) == (args, False)


def test_a_singular_subject_with_the_owner_name_is_refused_after_outside_content(monkeypatch):
    from ghost_agent.memory import egress as EG
    import ghost_agent.utils.provenance as PV
    monkeypatch.setattr(PV, "untrusted_seen", lambda: ["https://evil.example/page"])
    monkeypatch.setattr(PV, "user_message", lambda: "draw a cat")
    monkeypatch.setattr(EG, "identifier_values", lambda pm: [("root.name", "Vasilis Example")])
    ctx = SimpleNamespace(profile_memory=_profile(), egress_profile=None)
    assert EG.content_egress_refusal("image_generation", {"prompt": "x", "subject": "Vasilis Example"}, ctx)
    assert EG.content_egress_refusal("image_generation", {"subject": "Vasilis Example as a cat"}, ctx) is None


@pytest.mark.parametrize("args", [
    {"prompt": "a kiss", "subject": "Alexis Tsipras"},
    {"prompt": "Alexis Tsipras", "subject": "Alexis Tsipras"},
    {"subject": "Alexis Tsipras"},
    {"description": "a kiss", "subject": "Alexis Tsipras"},
    {"prompt": "a kiss", "subject": "  "},
    {"prompt": "a kiss", "subjects": ["Antonis Samaras"], "subject": "Alexis Tsipras"},
    {"prompt": ["a", "kiss"], "subject": "a kiss"},
])
async def test_the_guard_and_the_tool_agree_on_what_is_searched(tmp_path, web, args):
    await tool_generate_image(llm_client=_llm(), sandbox_dir=tmp_path, **args)
    assert web["lookup"] == SP.normalise_subjects(SP.subjects_arg(args))


# ── 6. names that contain separators ────────────────────────────────────────

@pytest.mark.parametrize("raw,want", [
    ("Martin Luther King, Jr.", ["Martin Luther King, Jr."]),
    ("Martin Luther King, Jr., Rosa Parks", ["Martin Luther King, Jr.", "Rosa Parks"]),
    ("Sammy Davis, Jr.; Frank Sinatra", ["Sammy Davis, Jr.", "Frank Sinatra"]),
    # §4QG-b (operator): a string ALWAYS splits on and/&/και, as before §4QG —
    # single surnames joined by "and" are the original §4MG live failure
    ("Tsipras and Mitsotakis", ["Tsipras", "Mitsotakis"]),
    ("Τσίπρας και Σαμαράς", ["Τσίπρας", "Σαμαράς"]),
    ("Tsipras & Mitsotakis", ["Tsipras", "Mitsotakis"]),
    ("Simon & Garfunkel", ["Simon", "Garfunkel"]),
    ("Alexis Tsipras and Antonis Samaras", ["Alexis Tsipras", "Antonis Samaras"]),
    ("Alexis Tsipras & Antonis Samaras", ["Alexis Tsipras", "Antonis Samaras"]),
    ("Barack Obama, Michelle Obama and Joe Biden", ["Barack Obama", "Michelle Obama", "Joe Biden"]),
    (["Simon and Garfunkel", "Martin Luther King, Jr."], ["Simon and Garfunkel", "Martin Luther King, Jr."]),
])
def test_names_with_separators_inside_stay_whole(raw, want):
    assert SP.normalise_subjects(raw) == want


# ── 7. titles, initials, ß ──────────────────────────────────────────────────

@pytest.mark.parametrize("subject,title,ok", [
    ("Prime Minister Kyriakos Mitsotakis", "Kyriakos Mitsotakis", True),
    ("Dr. Anthony Fauci", "Anthony Fauci", True),
    ("President Emmanuel Macron", "Emmanuel Macron", True),
    ("κ. Κυριάκος Μητσοτάκης", "Κυριάκος Μητσοτάκης", True),
    ("JK Rowling", "J. K. Rowling", True),
    ("J.K. Rowling", "J. K. Rowling", True),
    ("Johann Strauß", "Johann Strauss II", True),
    ("Pope Francis", "Pope Francis", True),
    ("Pope Francis", "Francis of Assisi", False),          # one word left is never enough
    ("Saint Paul", "Paul McCartney", False),
])
def test_titles_initials_and_eszett_match(subject, title, ok):
    assert SP.title_matches(subject, title) is ok


def test_a_title_stripped_name_is_exact_and_asked_in_the_same_call(monkeypatch):
    assert SP.title_is_exact("Prime Minister Kyriakos Mitsotakis", "Kyriakos Mitsotakis")
    assert not SP.title_is_exact("Pope Francis", "Francis")
    calls = _wiki(monkeypatch, {("en", "titles"): {"pages": {
        "-1": {"title": "Prime Minister Kyriakos Mitsotakis", "missing": ""},
        "1": _page("Kyriakos Mitsotakis", "https://x/pm.jpg")}}})
    assert SP.lookup_photo_url("Prime Minister Kyriakos Mitsotakis", None)[0] == "https://x/pm.jpg"
    assert calls[0][2] == "Prime Minister Kyriakos Mitsotakis|Kyriakos Mitsotakis" and len(calls) == 1
    calls = _wiki(monkeypatch, {})
    SP.lookup_photo_url("Kyriakos Mitsotakis", None)
    assert calls[0][2] == "Kyriakos Mitsotakis"              # no title: the query is unchanged


# ── 8. permanent vs transient vs this machine ───────────────────────────────

async def test_a_gif_lead_image_is_no_usable_photo_not_call_again(tmp_path, web):
    web["photo"] = _img("RGB", (50, 50), (1, 2, 3), "GIF")
    llm = _llm()
    out = await tool_generate_image(prompt="x", subjects=["Some One"], llm_client=llm, sandbox_dir=tmp_path)
    assert "no usable photo found for Some One" in out and "GIF image" in out
    assert "Call image_generation again once" not in out
    assert getattr(out, "reason_code", None) == "subject_photo_missing"
    llm.generate_image.assert_not_called()


async def test_a_garbage_download_is_still_transient(tmp_path, web):
    web["photo"] = b"<html>busy</html>"
    out = await tool_generate_image(prompt="x", subjects=["Some One"], llm_client=_llm(), sandbox_dir=tmp_path)
    assert "could not be downloaded right now" in out and "Call image_generation again once" in out


@pytest.mark.parametrize("where", ["replace", "mkdir"])
async def test_a_disk_error_blames_the_machine_not_the_photo(tmp_path, web, monkeypatch, where):
    def boom(*a, **k):
        raise OSError(28, "No space left on device")
    if where == "replace":
        monkeypatch.setattr(SP.os, "replace", boom)
    else:
        monkeypatch.setattr(SP.Path, "mkdir", boom)
    llm = _llm()
    out = await tool_generate_image(prompt="x", subjects=["Some One"], llm_client=llm, sandbox_dir=tmp_path)
    assert "could not save the photo" in out and "disk error" in out and "No space left" in out
    assert "upload" not in out.lower() and "no usable photo" not in out and "[subjects:" not in out
    llm.generate_image.assert_not_called()
    assert not list(tmp_path.rglob("*.part-*"))


# ── 9. the cache: max age, collision-proof names ────────────────────────────

async def test_an_old_cached_photo_is_fetched_again(tmp_path, web):
    await tool_generate_image(prompt="x", subjects=["Some One"], llm_client=_llm(), sandbox_dir=tmp_path)
    await tool_generate_image(prompt="x", subjects=["Some One"], llm_client=_llm(), sandbox_dir=tmp_path)
    assert len(web["lookup"]) == 1                              # fresh: reused
    photo = tmp_path / SP.PHOTO_DIR / "ref_some_one.jpg"
    old = time.time() - SP.CACHE_MAX_AGE_S - 60
    os.utime(photo, (old, old))
    await tool_generate_image(prompt="x", subjects=["Some One"], llm_client=_llm(), sandbox_dir=tmp_path)
    assert len(web["lookup"]) == 2


async def test_two_long_names_sharing_sixty_characters_get_their_own_files(tmp_path, web):
    head = "Maria Konstantina Alexandra Papadopoulou Georgiou Nikolaidou"   # > 60 chars
    a, b = f"{head} Senior", f"{head} Junior"
    assert SP.slug(a) == SP.slug(b) and SP.photo_key(a) != SP.photo_key(b)
    assert SP.normalise_subjects([a, b]) == [a, b]              # was deduplicated to one
    assert len(SP.photo_key(a)) <= 60
    await tool_generate_image(prompt="x", subjects=[a, b], llm_client=_llm(), sandbox_dir=tmp_path)
    assert len(list((tmp_path / SP.PHOTO_DIR).glob("ref_maria*.jpg"))) == 2
    assert SP.photo_key("Alexis Tsipras") == "alexis_tsipras"   # short names keep their file names


# ── 10. one budget per subject ──────────────────────────────────────────────

def _clock(monkeypatch):
    now = {"t": 1000.0}
    monkeypatch.setattr(SP, "time", SimpleNamespace(monotonic=lambda: now["t"], time=time.time))
    return now


def test_the_lookup_stops_at_its_budget(monkeypatch):
    now = _clock(monkeypatch)
    import curl_cffi.requests as creq
    calls = []

    def slow(url, params=None, **kw):
        calls.append(kw["timeout"])
        now["t"] += kw["timeout"]                 # each call answers (no match) at its timeout
        return _R(200)
    monkeypatch.setattr(creq, "get", slow)
    with pytest.raises(SP.SearchUnavailable) as ei:
        SP.lookup_photo_url("Ζωή Κωνσταντοπούλου", None)
    # el. title + search + en. title at 20 s each (§4QG-b: 60 s budget), en.
    # search never sent: out of budget is "unreachable", never "no photo"
    assert calls == [20, 20, 20] and "budget" in str(ei.value)


async def test_no_second_attempt_once_the_budget_is_spent(tmp_path, monkeypatch):
    now = _clock(monkeypatch)
    tries = []

    def down(subject, proxy):
        tries.append(subject)
        now["t"] += 44
        raise SP.SearchUnavailable("the encyclopedia could not be reached (TimeoutError)")
    monkeypatch.setattr(SP, "lookup_photo_url", down)
    import ghost_agent.utils.helpers as H
    monkeypatch.setattr(H, "request_new_tor_identity", lambda *a, **k: None)
    monkeypatch.setattr(SP.asyncio, "sleep", AsyncMock())
    out = await tool_generate_image(prompt="x", subjects=["Some One"], llm_client=_llm(), sandbox_dir=tmp_path)
    assert out.startswith("ERROR: the photo search could not reach the encyclopedia")
    assert tries == ["Some One"]                                # was LOOKUP_ATTEMPTS (2)


async def test_the_budget_reaches_the_lookup_thread(tmp_path, monkeypatch):
    seen = []

    def lookup(subject, proxy):
        seen.append(SP._LOOKUP_DEADLINE.get())
        return None
    monkeypatch.setattr(SP, "lookup_photo_url", lookup)
    t0 = time.monotonic()
    await tool_generate_image(prompt="x", subjects=["A B", "C D"], llm_client=_llm(), sandbox_dir=tmp_path)
    assert len(seen) == 2 and all(d and t0 < d <= time.monotonic() + SP.LOOKUP_DEADLINE_S for d in seen)
    assert SP._LOOKUP_DEADLINE.get() is None                    # reset after the fetch


# ── §4QG-b: the retry needs a full query; stale fallback; float photos ──────

@pytest.mark.parametrize("spent,tries", [(40, 1), (30, 2)])
async def test_attempt_two_runs_only_with_a_full_query_timeout_left(tmp_path, monkeypatch, spent, tries):
    """[MINOR §4QG-b] 45 s budget, attempt 2 started with ~5 s left: one
    clamped query, then "out of budget". Now: 60 s, and attempt 2 only with
    3 s pause + one full LOOKUP_TIMEOUT_S left (40 s spent → 20 left → skip)."""
    now = _clock(monkeypatch)
    seen = []

    def down(subject, proxy):
        seen.append(subject)
        now["t"] += spent
        raise SP.SearchUnavailable("the encyclopedia could not be reached (TimeoutError)")
    monkeypatch.setattr(SP, "lookup_photo_url", down)
    import ghost_agent.utils.helpers as H
    monkeypatch.setattr(H, "request_new_tor_identity", lambda *a, **k: None)
    monkeypatch.setattr(SP.asyncio, "sleep", AsyncMock())
    assert SP.LOOKUP_DEADLINE_S == 60
    with pytest.raises(SP.SearchUnavailable):
        await SP.fetch_subject_photo("Some One", tmp_path, None, is_image=lambda b: True)
    assert len(seen) == tries


@pytest.mark.parametrize("fail", ["search", "download"])
async def test_an_expired_photo_is_used_when_the_refetch_fails(tmp_path, web, monkeypatch, fail):
    """[MINOR §4QG-b] after 30 days the photo was fetched again — and when
    that failed (Tor down, a 503 on the thumbnail) the render was refused
    although a photo of the person sat in the cache."""
    await tool_generate_image(prompt="x", subjects=["Some One"], llm_client=_llm(), sandbox_dir=tmp_path)
    photo = tmp_path / SP.PHOTO_DIR / "ref_some_one.jpg"
    old = time.time() - SP.CACHE_MAX_AGE_S - 60
    os.utime(photo, (old, old))
    if fail == "search":
        def down(subject, proxy):
            raise SP.SearchUnavailable("the encyclopedia could not be reached (TimeoutError)")
        monkeypatch.setattr(SP, "lookup_photo_url", down)
        import ghost_agent.utils.helpers as H
        monkeypatch.setattr(H, "request_new_tor_identity", lambda *a, **k: None)
        monkeypatch.setattr(SP.asyncio, "sleep", AsyncMock())
    else:
        web["photo"] = b"<html>busy</html>"               # a garbage download: transient
    llm = _llm()
    out = await tool_generate_image(prompt="x", subjects=["Some One"], llm_client=llm, sandbox_dir=tmp_path)
    assert out.startswith("SUCCESS"), out
    llm.generate_image.assert_awaited_once()
    # with nothing cached the same failure is still an error
    photo.unlink()
    out = await tool_generate_image(prompt="x", subjects=["Some One"], llm_client=_llm(), sandbox_dir=tmp_path)
    assert out.startswith("ERROR")


async def test_an_expired_photo_is_not_used_for_a_permanent_failure(tmp_path, web):
    await tool_generate_image(prompt="x", subjects=["Some One"], llm_client=_llm(), sandbox_dir=tmp_path)
    photo = tmp_path / SP.PHOTO_DIR / "ref_some_one.jpg"
    old = time.time() - SP.CACHE_MAX_AGE_S - 60
    os.utime(photo, (old, old))
    web["photo"] = _img("RGB", (50, 50), (1, 2, 3), "GIF")     # the article's photo is now a GIF
    out = await tool_generate_image(prompt="x", subjects=["Some One"], llm_client=_llm(), sandbox_dir=tmp_path)
    assert "no usable photo" in out


@pytest.mark.parametrize("value,want", [(0.5, 127), (1.0, 255), (200.0, 255), (0.0, 0)])
def test_a_float_photo_is_scaled_not_black(value, want):
    """[MINOR §4QG-b] mode "F" (0–1 floats) converted straight to 8 bits was black."""
    out = SP._on_white(Image.new("F", (4, 4), value))
    assert out.mode == "RGB" and abs(out.getpixel((0, 0))[0] - want) <= 1


def test_a_float_photo_keeps_its_contrast():
    im = Image.new("F", (2, 1), 0.0)
    im.putpixel((1, 0), 100.0)
    out = SP._on_white(im)
    assert out.getpixel((0, 0))[0] == 0 and out.getpixel((1, 0))[0] == 255


async def test_a_long_subject_equal_to_the_prompt_is_the_prompt_for_both(tmp_path, web):
    """[MINOR §4QG-b] the tool compared `subject` to the prompt TRUNCATED to
    MAX_PROMPT_CHARS, the guard to the whole prompt: a 9000-char
    `subject` == prompt was a subjects search for the tool only."""
    from ghost_agent.tools.image_gen import MAX_PROMPT_CHARS
    long = "a quiet harbour at dawn " * 400
    assert len(long) > MAX_PROMPT_CHARS
    args = {"prompt": long, "subject": long}
    assert SP.subjects_arg(args) is None
    await tool_generate_image(llm_client=_llm(), sandbox_dir=tmp_path, **args)
    assert web["lookup"] == []
