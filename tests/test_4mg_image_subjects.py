"""§4MG (2026-10-07): `image_generation(subjects=[...])` — the tool fetches
photos of named real people / places / products itself, joins several into
the node's one reference, and the checks downstream use them.

Live failure: asked for Tsipras and Samaras, the agent planned to download
photos, then rendered from the names (two strangers); on the follow-up it
passed one portrait (the node takes one) and then "fixed" the scene by
editing the OTHER man's portrait — four podium portraits, no kiss — and the
time-budget report showed no image at all.
"""
import base64
import json
import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

import ghost_agent.tools.subject_photos as SP
from ghost_agent.tools.image_gen import tool_generate_image
import ghost_agent.core.agent as A


def _jpeg(w=300, h=400, colour=(200, 50, 50)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), colour).save(buf, format="JPEG")
    return buf.getvalue()


def _llm():
    llm = AsyncMock()
    llm.image_gen_clients = [{"model": "Ghost"}]
    llm.generate_image.return_value = {"data": [{"b64_json": base64.b64encode(b"\x89PNG\r\n\x1a\nxx").decode()}],
                                       "width": 768, "height": 512, "seed": 7}
    return llm


@pytest.fixture
def web(monkeypatch):
    """Fake encyclopedia: name → photo; records lookups and downloads."""
    photos = {"Alexis Tsipras": _jpeg(300, 400, (10, 10, 10)),
              "Antonis Samaras": _jpeg(400, 300, (240, 240, 240)),
              "Parthenon": _jpeg(500, 300, (100, 100, 0))}
    calls = {"lookup": [], "download": []}

    def lookup(subject, proxy):
        calls["lookup"].append(subject)
        if subject not in photos:
            return None
        return (f"https://img.example/{subject}.jpg", subject, f"https://en.wikipedia.org/wiki/{subject}")

    async def download(url, sandbox_dir, tor_proxy, filename=None):
        calls["download"].append(url)
        name = url.rsplit("/", 1)[1][:-4]
        (Path(sandbox_dir) / filename).write_bytes(photos[name])
        return f"SUCCESS: Downloaded '{url}' to '{filename}'."

    monkeypatch.setattr(SP, "lookup_photo_url", lookup)
    import ghost_agent.tools.file_system as FS
    monkeypatch.setattr(FS, "tool_download_file", download)
    return calls


def _sent(llm):
    return llm.generate_image.call_args.args[0]


# ---------------------------------------------------------------------------
# picking the right article
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("subject,title,ok", [
    ("Antonis Samaras", "Antonis Samaras", True),
    ("Antonis Samaras", "New Democracy (Greece)", False),     # the party flag was hit #2 live
    ("Tsipras", "Alexis Tsipras", True),
    ("Αντώνης Σαμαράς", "Αντώνης Σαμαράς", True),
    ("αντωνης σαμαρας", "Αντώνης Σαμαράς", True),             # accents and case fold
    ("Parthenon", "Parthenon", True),
    # review: the reverse direction (title ⊆ name) rendered ONE man's photo as both
    ("Alexis Tsipras and Antonis Samaras", "Alexis Tsipras", False),
    ("Michael Jordan", "Jordan", False),
    ("the Parthenon in Athens", "Parthenon", False),
    ("Toyota Corolla", "Toyota Supra", False),
])
def test_title_must_be_about_the_subject(subject, title, ok):
    assert SP.title_matches(subject, title) is ok


def test_lookup_skips_off_topic_and_disambiguation_hits(monkeypatch):
    seen = {}

    class R:
        status_code = 200

        def json(self):
            return {"query": {"pages": {
                "1": {"index": 1, "title": "New Democracy (Greece)", "thumbnail": {"source": "https://x/flag.png"}},
                # a disambiguation page matches the name and HAS an image — still not his face
                "2": {"index": 2, "title": "Samaras", "pageprops": {"disambiguation": ""},
                      "thumbnail": {"source": "https://x/samaras_family_crest.png"}},
                "3": {"index": 3, "title": "Antonis Samaras", "thumbnail": {"source": "https://x/samaras.jpg"}},
            }}}

    import curl_cffi.requests as creq

    def get(url, params=None, **kw):
        seen.setdefault("hosts", []).append(url)
        return R()
    monkeypatch.setattr(creq, "get", get)
    url, title, page = SP.lookup_photo_url("Antonis Samaras", None)
    assert url == "https://x/samaras.jpg" and title == "Antonis Samaras"
    assert page == "https://en.wikipedia.org/wiki/Antonis_Samaras"
    n = len(seen["hosts"])
    SP.lookup_photo_url("Αντώνης Σαμαράς", None)
    # Greek name → Greek encyclopedia first, then English
    assert [h.split("/")[2] for h in seen["hosts"][n:]] == ["el.wikipedia.org", "en.wikipedia.org"]


# ---------------------------------------------------------------------------
# the tool
# ---------------------------------------------------------------------------

async def test_one_subject_renders_from_its_photo(tmp_path, web):
    llm = _llm()
    out = await tool_generate_image(prompt="him on the moon", subjects=["Alexis Tsipras"],
                                    llm_client=llm, sandbox_dir=tmp_path)
    sent = _sent(llm)
    photo = tmp_path / SP.PHOTO_DIR / "ref_alexis_tsipras.jpg"
    assert sent["reference_images"] == [base64.b64encode(photo.read_bytes()).decode()]
    assert "width" not in sent                     # one photo, no size asked: the photo's own shape
    assert sent["prompt"].startswith("The reference photo shows Alexis Tsipras.")
    assert sent["prompt"].endswith("him on the moon")
    assert "negative_prompt" not in sent          # one photo: nothing to copy a layout from
    assert "REFERENCE PHOTO: subject_photos/ref_alexis_tsipras.jpg — Alexis Tsipras (ref_alexis_tsipras.jpg)." in out
    # the live four-portraits failure: never "fix" a likeness by editing one subject's portrait
    assert "do NOT edit the picture with one subject's portrait as the reference" in out
    assert "THIS WAS A NEW SCENE" in out and "THIS WAS AN EDIT" not in out
    assert "built from the subjects' photos" in out


async def test_two_subjects_are_joined_into_the_one_reference(tmp_path, web):
    llm = _llm()
    out = await tool_generate_image(prompt="the fraternal kiss", subjects=["Alexis Tsipras", "Antonis Samaras"],
                                    llm_client=llm, sandbox_dir=tmp_path)
    sent = _sent(llm)
    assert len(sent["reference_images"]) == 1
    combined = Image.open(io.BytesIO(base64.b64decode(sent["reference_images"][0])))
    # both photos at a common height, side by side with a gap, PADDED to the
    # render's 3:2 — the node stretches the reference to the output size
    strip_w = round(300 * 1024 / 400) + round(400 * 1024 / 300) + SP.COMBINED_GAP
    assert combined.width == strip_w and abs(combined.width / combined.height - 1.5) < 0.01
    assert (sent["width"], sent["height"]) == (768, 512)
    assert "you asked for" not in out              # the tool chose that size, not the user
    # …even when the node renders another size than the one the tool sent
    llm2 = _llm(); llm2.generate_image.return_value = {**llm2.generate_image.return_value, "width": 736}
    out2 = await tool_generate_image(prompt="k", subjects=["Alexis Tsipras", "Antonis Samaras"],
                                     llm_client=llm2, sandbox_dir=tmp_path)
    assert "Rendered at 736x512" in out2 and "you asked for" not in out2
    mid = combined.height // 2
    # left is the first subject's (dark) photo, right the second's (light)
    assert combined.getpixel((50, mid))[0] < 60 and combined.getpixel((combined.width - 50, mid))[0] > 200
    assert "left to right: Alexis Tsipras, Antonis Samaras" in sent["prompt"]
    # live probe-4mg-kiss: without these the model copied the joined layout — a two-panel collage
    assert "ONE single photograph of ONE continuous scene" in sent["prompt"]
    assert "collage" in sent["negative_prompt"] and "diptych" in sent["negative_prompt"]
    assert list((tmp_path / SP.PHOTO_DIR).glob("ref_combined_*.jpg"))
    assert "— left to right: Alexis Tsipras (ref_alexis_tsipras.jpg), Antonis Samaras (ref_antonis_samaras.jpg)." in out


async def test_a_subject_without_a_photo_renders_nothing(tmp_path, web):
    llm = _llm()
    out = await tool_generate_image(prompt="x", subjects=["Alexis Tsipras", "Nobody Atall"],
                                    llm_client=llm, sandbox_dir=tmp_path)
    assert out.startswith("ERROR: no usable photo found for Nobody Atall")
    # the instruction that made the live agent ASK — pinned by its claim
    assert "STOP and reply to the user now" in out
    assert "Do not render anything else for this request" in out
    assert out.rstrip().endswith('[subjects: {"missing": ["Nobody Atall"], "found": ["Alexis Tsipras"]}]')
    llm.generate_image.assert_not_called()


async def test_subjects_and_reference_images_cannot_be_combined(tmp_path, web):
    (tmp_path / "gen_aa.png").write_bytes(_jpeg())
    llm = _llm()
    out = await tool_generate_image(prompt="x", subjects=["Alexis Tsipras"], reference_images=["gen_aa.png"],
                                    llm_client=llm, sandbox_dir=tmp_path)
    assert out.startswith("ERROR: `subjects` and `reference_images` cannot be combined")
    llm.generate_image.assert_not_called()
    assert not web["lookup"]


async def test_more_than_three_subjects_is_refused_before_any_lookup(tmp_path, web):
    llm = _llm()
    out = await tool_generate_image(prompt="x", subjects=["a b", "c d", "e f", "g h"],
                                    llm_client=llm, sandbox_dir=tmp_path)
    assert "at most 3 subjects" in out and not web["lookup"]
    llm.generate_image.assert_not_called()


async def test_a_photo_fetched_before_is_reused(tmp_path, web):
    await tool_generate_image(prompt="x", subjects="Alexis Tsipras", llm_client=_llm(), sandbox_dir=tmp_path)
    n = len(web["lookup"])
    await tool_generate_image(prompt="y", subjects="Alexis Tsipras", llm_client=_llm(), sandbox_dir=tmp_path)
    assert len(web["lookup"]) == n and len(web["download"]) == 1


async def test_non_image_download_is_not_used(tmp_path, monkeypatch):
    monkeypatch.setattr(SP, "lookup_photo_url", lambda s, p: ("https://x/a.jpg", s, "https://w/a"))

    async def download(url, sandbox_dir, tor_proxy, filename=None):
        (Path(sandbox_dir) / filename).write_bytes(b"<html>blocked</html>")
        return "SUCCESS: Downloaded"
    import ghost_agent.tools.file_system as FS
    monkeypatch.setattr(FS, "tool_download_file", download)
    llm = _llm()
    out = await tool_generate_image(prompt="x", subjects=["Some One"], llm_client=llm, sandbox_dir=tmp_path)
    assert "not a complete photo" in out and not list(tmp_path.rglob("ref_*"))
    llm.generate_image.assert_not_called()


async def test_an_edit_from_a_downloaded_photo_says_the_scene_did_not_carry_over(tmp_path):
    (tmp_path / "samaras.jpg").write_bytes(_jpeg())
    (tmp_path / "gen_aa.png").write_bytes(_jpeg())
    out = await tool_generate_image(prompt="change the man on the right", reference_images=["/samaras.jpg"],
                                    llm_client=_llm(), sandbox_dir=tmp_path)
    assert "The model saw ONLY samaras.jpg" in out
    out = await tool_generate_image(prompt="make the sky red", reference_images=["/gen_aa.png"],
                                    llm_client=_llm(), sandbox_dir=tmp_path)
    assert "saw ONLY" not in out and "THIS WAS AN EDIT" in out


def test_schema_and_dispatch_carry_subjects_and_tor(monkeypatch):
    from ghost_agent.tools.registry import get_active_tool_definitions, get_available_tools
    import ghost_agent.tools.image_gen as IG
    ctx = MagicMock()
    ctx.llm_client.image_gen_clients = [{"model": "Ghost"}]
    ctx.tor_proxy = "socks5://127.0.0.1:9050"
    fn = next(t["function"] for t in get_active_tool_definitions(ctx) if t["function"]["name"] == "image_generation")
    assert fn["parameters"]["properties"]["subjects"]["maxItems"] == SP.MAX_SUBJECTS
    assert "`subjects`" in fn["description"]
    seen = {}

    async def fake(**kw):
        seen.update(kw)
        return "ok"
    monkeypatch.setattr(IG, "tool_generate_image", fake)
    tools = get_available_tools(ctx)
    import asyncio
    asyncio.run(tools["image_generation"](prompt="x", subjects=["A B"], tor_proxy="evil"))
    assert seen["tor_proxy"] == "socks5://127.0.0.1:9050" and seen["subjects"] == ["A B"]


# ---------------------------------------------------------------------------
# downstream: the likeness check and the reply
# ---------------------------------------------------------------------------

async def test_the_tool_record_is_what_the_agent_parses(tmp_path, web):
    """Round trip on the tool's REAL output, not a hand-written fixture."""
    out = await tool_generate_image(prompt="k", subjects=["Alexis Tsipras", "Antonis Samaras"],
                                    llm_client=_llm(), sandbox_dir=tmp_path)
    ref = A._turn_subject_reference([{"content": out}])
    assert ref and ref[0].startswith("subject_photos/ref_combined_") and ref[1] == "Alexis Tsipras, Antonis Samaras"
    one = await tool_generate_image(prompt="k", subjects=["Parthenon"], llm_client=_llm(), sandbox_dir=tmp_path)
    assert A._turn_subject_reference([{"content": out}, {"content": one}]) == ("subject_photos/ref_parthenon.jpg", "Parthenon")
    # a later plain render is not judged against an earlier render's photos
    plain = await tool_generate_image(prompt="a cat", llm_client=_llm(), sandbox_dir=tmp_path)
    assert A._turn_subject_reference([{"content": out}, {"content": plain}]) is None


async def test_likeness_mode_sends_reference_then_result(tmp_path):
    from ghost_agent.core.verifier import Verifier, VerifyVerdict
    v = object.__new__(Verifier)
    v._call_llm_vision = AsyncMock(return_value={"verdict": "REFUTED", "confidence": 0.9,
                                                 "reasoning": "right man is not the reference", "issues": ["x"]})
    r = await v.verify_visual(symptom="kiss", claim="here they are", after_image="/g.png",
                              before_image="/ref.jpg", likeness_subjects="A, B")
    prompt, images = v._call_llm_vision.call_args.args[:2]
    assert images == ["/ref.jpg", "/g.png"]
    assert "left to right: A, B" in prompt and "Do NOT identify anyone by name" in prompt
    assert r.verdict == VerifyVerdict.REFUTED
    # without subjects: the old UI-symptom prompt
    await v.verify_visual(symptom="s", claim="c", after_image="/g.png", before_image="/b.png")
    assert "UI auditor" in v._call_llm_vision.call_args.args[0]


def test_likeness_evidence_uses_the_reference_photo(tmp_path):
    (tmp_path / "ref_combined_ab.jpg").write_bytes(_jpeg())
    body = ("SUCCESS: Image generated and saved to sandbox. Rendered.\n\nREFERENCE PHOTO: ref_combined_ab.jpg — "
            "left to right: A (ref_a.jpg), B (ref_b.jpg). THIS WAS A NEW SCENE built from those photos.\n\n"
            "![generated image](/api/download/gen_x.png)")
    before, subj = A.GhostAgent._likeness_evidence([{"content": body}], tmp_path, "/user.png")
    assert before.endswith("ref_combined_ab.jpg") and subj == "A, B"
    # the photo is gone → the usual evidence, no likeness mode
    (tmp_path / "ref_combined_ab.jpg").unlink()
    assert A.GhostAgent._likeness_evidence([{"content": body}], tmp_path, "/user.png") == ("/user.png", None)


def test_a_reply_without_the_image_gets_the_last_one():
    rows = [{"content": "SUCCESS: Image generated\n![generated image](/api/download/gen_a.png)"},
            {"content": "SUCCESS: Image generated\n![generated image](/api/download/projects/p1/gen_b.png)"}]
    assert A._unshown_image_note("I tried twice.", rows) == "\n\n![generated image](/api/download/projects/p1/gen_b.png)"
    # a bare name DISPLAYS nothing (review: it used to count as shown)
    assert A._unshown_image_note("saved as gen_a.png", rows).endswith("(/api/download/projects/p1/gen_b.png)")
    assert A._unshown_image_note("![x](/api/download/gen_a.png)", rows) == ""          # one is shown
    assert A._unshown_image_note("no images", [{"content": "ERROR: x"}]) == ""


def test_a_member_may_name_subjects(monkeypatch, tmp_path):
    """Review: a member was SHOWN `subjects` and refused for using it — the
    refusal steered to a name-only render. Other unknown keys stay refused."""
    from tests.test_requester_role import _agent as _member_agent
    agent, _, _ = _member_agent(monkeypatch, tmp_path)
    assert agent._member_tool_refusal("image_generation", {"prompt": "x", "subjects": ["A B"]}, []) is None
    assert "takes only the arguments" in agent._member_tool_refusal(
        "image_generation", {"prompt": "x", "style": "y"}, [])


def test_the_system_prompt_routes_real_subjects_and_wrong_people_to_subjects():
    from ghost_agent.core.prompts import SYSTEM_PROMPT
    block = SYSTEM_PROMPT[SYSTEM_PROMPT.index("- IMAGE GENERATION:"):]
    block = block[:block.index("- CHECKING YOUR OWN UI")]
    assert "name each one in the tool's `subjects` argument" in block
    assert "do not download them yourself" in block
    # "that's not X" is a new render from photos, never an edit of a portrait
    assert "an edit cannot fix it" in block and "render again with them in `subjects`" in block


async def test_the_no_panels_negative_keeps_the_models_own(tmp_path, web):
    llm = _llm()
    await tool_generate_image(prompt="k", subjects=["Alexis Tsipras", "Antonis Samaras"],
                              negative_prompt="blurry", llm_client=llm, sandbox_dir=tmp_path)
    neg = _sent(llm)["negative_prompt"]
    assert neg.startswith("blurry, ") and "split screen" in neg


# ---------------------------------------------------------------------------
# a subject with no photo → the agent asks; it does not draw a stand-in
# (operator 2026-10-07, after live probe-4mg-missing re-rendered without asking)
# ---------------------------------------------------------------------------

async def _missing_row(tmp_path, web, subjects):
    """The tool's REAL error text, as the loop records it."""
    out = await tool_generate_image(prompt="x", subjects=subjects, llm_client=_llm(), sandbox_dir=tmp_path)
    assert "no usable photo found" in out
    return {"role": "tool", "content": out}


async def test_dropping_the_missing_subject_is_blocked(tmp_path, web):
    row = await _missing_row(tmp_path, web, ["Alexis Tsipras", "Nobody Atall"])
    blk = A._missing_subject_block("image_generation", {"prompt": "two men"}, [row])
    assert blk and "no photo of Nobody Atall" in blk and "ask them to upload" in blk
    assert A._missing_subject_block("image_generation", json.dumps(
        {"prompt": "x", "subjects": ["Alexis Tsipras"]}), [row])            # one dropped
    # a corrected spelling keeps every subject → may run
    assert A._missing_subject_block("image_generation", {"prompt": "x", "subjects":
                                    ["Alexis Tsipras", "Nobody At All"]}, [row]) is None
    # other tools, other requests, synthetic rows: untouched
    assert A._missing_subject_block("web_search", {"query": "x"}, [row]) is None
    assert A._missing_subject_block("image_generation", {"prompt": "x"}, []) is None
    assert A._missing_subject_block("image_generation", {"prompt": "x"}, [{**row, "_synthetic": True}]) is None


async def test_the_block_reads_a_single_missing_subject(tmp_path, web):
    row = await _missing_row(tmp_path, web, ["Vasilis Ventiroplakos"])
    blk = A._missing_subject_block("image_generation", {"prompt": "a violinist"}, [row])
    assert blk and "no photo of Vasilis Ventiroplakos" in blk


async def test_the_loop_refuses_the_stand_in_and_the_agent_asks(monkeypatch, tmp_path, web):
    """End to end through handle_chat: the second image call never reaches the tool."""
    from tests.test_requester_role import _agent as _owner_agent, _tc, _resp
    from tests.helpers import FakeBgTasks
    agent, ctx, _ = _owner_agent(monkeypatch, tmp_path)
    err = await tool_generate_image(prompt="x", subjects=["Vasilis Ventiroplakos"], llm_client=_llm(),
                                    sandbox_dir=tmp_path)
    img = AsyncMock(side_effect=[err, "SUCCESS: Image generated … ![generated image](/api/download/gen_zz.png)"])
    agent.available_tools = {"image_generation": img, "web_search": AsyncMock(return_value="x")}
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[
        _resp("", [_tc("c0", "image_generation", {"prompt": "him on the Acropolis",
                                                  "subjects": ["Vasilis Ventiroplakos"]})]),
        _resp("", [_tc("c1", "image_generation", {"prompt": "a violinist on the Acropolis"})]),
        _resp("I couldn't find a photo of Vasilis Ventiroplakos — can you upload one, or is a generic violinist fine?"),
        _resp("x"), _resp("x")])
    await agent.handle_chat({"messages": [{"role": "user", "content": "draw Vasilis Ventiroplakos on the Acropolis"}]},
                            FakeBgTasks(), request_id="web-4mg-ms")
    assert img.await_count == 1                      # the stand-in render never ran
    # §4MO: a designed stop now closes the tool phase at once — the stand-in
    # call is never even dispatched; the model is told to ask the user
    seen = ""
    for call in ctx.llm_client.chat_completion.call_args_list[1:]:
        payload = call.args[0] if call.args and isinstance(call.args[0], dict) else call.kwargs
        seen += "\n".join(str(m.get("content") or "") for m in payload.get("messages", []) if isinstance(m, dict))
    assert "ask the user first" in seen or "Ask the user exactly what the tool asked" in seen


# ---------------------------------------------------------------------------
# the "no photo → ask" turn is not left in the corpus as a failure
# ---------------------------------------------------------------------------

async def test_a_late_pass_lifts_the_no_photo_ask_turn_for_a_real_user(tmp_path, monkeypatch, web):
    """Live probe-4mg-missing2 kept `failed` because PROBE turns are never
    cached for correction (by design). A real user turn of the same shape —
    the tool's real error, the agent's real question — must be upgraded."""
    import asyncio
    from collections import OrderedDict
    from types import SimpleNamespace
    from ghost_agent.core.agent import GhostAgent
    from ghost_agent.distill.collector import TrajectoryCollector
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    err = await tool_generate_image(prompt="x", subjects=["Vasilis Ventiroplakos"], llm_client=_llm(),
                                    sandbox_dir=tmp_path)
    reply = ("I couldn't find a matching encyclopedia photo for **Vasilis Ventiroplakos** — the image "
             "model needs a reference photo to render the right likeness.\n\nCould you upload a photo of "
             "him? Or should I generate the scene with a generic violinist instead?")
    ctx = MagicMock()
    ctx.trajectory_collector = TrajectoryCollector(root=tmp_path / "system" / "trajectories", session_id="s")
    ctx.skill_memory = SimpleNamespace(is_read_only=False)
    del ctx.turn_origin_label
    ctx._recent_trajectories_for_correction = OrderedDict()
    ctx.calibration_tracker = SimpleNamespace(record_late_verdict_correction=lambda rid, v: None)
    ctx.self_model = None
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = ctx
    tid = "f" * 32
    ask = "draw Vasilis Ventiroplakos, the famous Greek violinist, playing on the Acropolis"
    agent._record_turn_trajectory(
        messages=[{"role": "user", "content": ask},
                  {"role": "assistant", "tool_calls": [{"id": "c0", "function": {
                      "name": "image_generation",
                      "arguments": json.dumps({"prompt": "x", "subjects": ["Vasilis Ventiroplakos"]})}}]},
                  {"role": "tool", "tool_call_id": "c0", "name": "image_generation", "content": err},
                  {"role": "assistant", "content": reply}],
        final_content=reply, req_id="web-4mg-late", model="m", trajectory_id=tid, user_request=ask,
        execution_failed=True)                              # the loop booked the tool error as a strike
    cached = next(t for t in ctx._recent_trajectories_for_correction.values() if t.id == tid)
    assert cached.outcome == "failed"                       # the structural label the live turn got
    assert cached.failure_reason.startswith("structural failure")   # upgradable, unlike a shape FAILED
    agent._backfill_trajectory_outcome(tid, "passed", "")
    for _ in range(30):
        await asyncio.sleep(0.02)
        if ctx.trajectory_collector.latest_correction(tid):
            break
    latest = ctx.trajectory_collector.latest_correction(tid)
    assert latest and latest.get("outcome") == "passed", latest
