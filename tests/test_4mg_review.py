"""§4MG fresh-eye review (2026-10-07): pins for every confirmed finding in
the `subjects` change — the tool, the agent-side wiring, the privacy guards.
Each names the world it fails in."""
import asyncio
import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

import ghost_agent.core.agent as A
import ghost_agent.tools.subject_photos as SP
from ghost_agent.tools.image_gen import tool_generate_image
from tests.test_4mg_image_subjects import _jpeg, _llm, _sent, web  # noqa: F401 — fixture


# ── the photo cache: provenance, completeness ───────────────────────────────

async def test_a_file_without_a_provenance_record_is_not_used_as_the_photo(tmp_path, web):
    """[MAJOR] a user's own `ref_alexis_tsipras.jpg` was silently his face."""
    (tmp_path / SP.PHOTO_DIR).mkdir()
    (tmp_path / SP.PHOTO_DIR / "ref_alexis_tsipras.jpg").write_bytes(_jpeg(10, 10, (0, 255, 0)))
    await tool_generate_image(prompt="x", subjects=["Alexis Tsipras"], llm_client=_llm(), sandbox_dir=tmp_path)
    assert web["lookup"] == ["Alexis Tsipras"]


async def test_another_subject_with_the_same_slug_is_fetched_again(tmp_path, web):
    await tool_generate_image(prompt="x", subjects=["Alexis Tsipras"], llm_client=_llm(), sandbox_dir=tmp_path)
    meta = tmp_path / SP.PHOTO_DIR / "ref_alexis_tsipras.jpg.json"
    rec = json.loads(meta.read_text())
    rec["subject"] = "Alexis Tsipras the painter"            # a different person, same slug file
    meta.write_text(json.dumps(rec))
    await tool_generate_image(prompt="y", subjects=["Alexis Tsipras"], llm_client=_llm(), sandbox_dir=tmp_path)
    assert web["lookup"] == ["Alexis Tsipras", "Alexis Tsipras"]


async def test_a_truncated_photo_is_fetched_again_not_reused(tmp_path, web):
    """[MAJOR] a download cut after its header passed the 16-byte check
    and broke every later render of that subject."""
    await tool_generate_image(prompt="x", subjects=["Alexis Tsipras"], llm_client=_llm(), sandbox_dir=tmp_path)
    photo = tmp_path / SP.PHOTO_DIR / "ref_alexis_tsipras.jpg"
    # a NOISY photo, cut after 60%: the header still opens, the pixels do not decode
    import os as _os
    noisy = Image.frombytes("RGB", (300, 400), _os.urandom(300 * 400 * 3))
    buf = io.BytesIO(); noisy.save(buf, format="JPEG")
    photo.write_bytes(buf.getvalue()[:int(len(buf.getvalue()) * 0.6)])
    with Image.open(photo) as _probe:          # the cut file still OPENS (so only a full decode catches it)
        assert _probe.size == (300, 400)
    await tool_generate_image(prompt="y", subjects=["Alexis Tsipras"], llm_client=_llm(), sandbox_dir=tmp_path)
    assert len(web["download"]) == 2 and SP.photo_is_complete(photo)


async def test_a_failed_download_leaves_no_file_under_the_final_name(tmp_path, monkeypatch):
    monkeypatch.setattr(SP, "lookup_photo_url", lambda s, p: ("https://x/a.jpg", s, "https://w/a"))

    async def half(url, sandbox_dir, tor_proxy, filename=None):
        (Path(sandbox_dir) / filename).write_bytes(_jpeg()[:300])   # wrote part, then failed
        return "Error: connection reset"
    import ghost_agent.tools.file_system as FS
    monkeypatch.setattr(FS, "tool_download_file", half)
    out = await tool_generate_image(prompt="x", subjects=["Some One"], llm_client=_llm(), sandbox_dir=tmp_path)
    assert "could not be downloaded" in out and not list(tmp_path.rglob("ref_*"))


# ── picking the photo ───────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,want", [
    ("Alexis Tsipras and Antonis Samaras", ["Alexis Tsipras", "Antonis Samaras"]),
    ("Τσίπρας και Σαμαράς", ["Τσίπρας", "Σαμαράς"]),
    ('["Alexis Tsipras", "Antonis Samaras"]', ["Alexis Tsipras", "Antonis Samaras"]),
    (["Alexis\nTsipras", "Alexis-Tsipras"], ["Alexis Tsipras"]),          # newline collapsed; same slug once
])
def test_names_are_split_cleaned_and_deduplicated(raw, want):
    assert SP.normalise_subjects(raw) == want


async def test_an_unreachable_encyclopedia_is_not_reported_as_a_missing_photo(tmp_path, monkeypatch):
    """[MINOR→MAJOR] a Tor timeout told the model "no photo — ask the user to
    upload one" for a famous person. Retried once on a fresh circuit."""
    calls = []

    def down(subject, proxy):
        calls.append(subject)
        raise SP.SearchUnavailable("the encyclopedia could not be reached (HTTP 503)")
    monkeypatch.setattr(SP, "lookup_photo_url", down)
    import ghost_agent.utils.helpers as H
    renew = []
    monkeypatch.setattr(H, "request_new_tor_identity", lambda *a, **k: renew.append(1))
    monkeypatch.setattr(SP.asyncio, "sleep", AsyncMock())
    llm = _llm()
    out = await tool_generate_image(prompt="x", subjects=["Kyriakos Mitsotakis"], llm_client=llm, sandbox_dir=tmp_path)
    assert out.startswith("ERROR: the photo search could not reach the encyclopedia")
    assert "NOT a missing photo" in out and "no usable photo" not in out
    assert len(calls) == SP.LOOKUP_ATTEMPTS and renew == [1]
    llm.generate_image.assert_not_called()


def test_lookup_distinguishes_no_answer_from_no_match(monkeypatch):
    import curl_cffi.requests as creq

    class R:
        status_code = 503
    monkeypatch.setattr(creq, "get", lambda *a, **k: R())
    with pytest.raises(SP.SearchUnavailable):
        SP.lookup_photo_url("Antonis Samaras", None)

    class Empty:
        status_code = 200

        def json(self):
            return {"query": {"pages": {}}}
    monkeypatch.setattr(creq, "get", lambda *a, **k: Empty())
    assert SP.lookup_photo_url("Antonis Samaras", None) is None


async def test_nothing_is_fetched_when_the_node_is_offline(tmp_path, web):
    llm = _llm()
    llm.image_gen_clients = []
    out = await tool_generate_image(prompt="x", subjects=["Alexis Tsipras"], llm_client=llm, sandbox_dir=tmp_path)
    assert "offline" in out and not web["lookup"] and not list(tmp_path.rglob("ref_*"))


# ── the reference's shape ───────────────────────────────────────────────────

async def test_a_requested_shape_pads_the_reference_instead_of_squashing_it(tmp_path, web):
    """[MAJOR] the node stretches the reference to the render size."""
    llm = _llm()
    await tool_generate_image(prompt="x", subjects=["Alexis Tsipras", "Antonis Samaras"], width=512, height=768,
                              llm_client=llm, sandbox_dir=tmp_path)
    im = Image.open(io.BytesIO(base64.b64decode(_sent(llm)["reference_images"][0])))
    assert abs(im.width / im.height - 512 / 768) < 0.01
    llm = _llm()
    await tool_generate_image(prompt="x", subjects=["Antonis Samaras"], width=768, height=512,
                              llm_client=llm, sandbox_dir=tmp_path)
    im = Image.open(io.BytesIO(base64.b64decode(_sent(llm)["reference_images"][0])))
    assert abs(im.width / im.height - 1.5) < 0.01                    # one photo, padded too


def test_a_panorama_cannot_squeeze_the_other_photos(tmp_path):
    wide, tall = tmp_path / "w.jpg", tmp_path / "t.jpg"
    wide.write_bytes(_jpeg(4000, 500))
    tall.write_bytes(_jpeg(300, 400))
    im = Image.open(io.BytesIO(SP.combine_side_by_side([wide, tall])))
    assert im.width == round(SP.COMBINED_HEIGHT * 1.5) + round(300 * 1024 / 400) + SP.COMBINED_GAP


# ── prompt and negative prompt ─────────────────────────────────────────────

async def test_a_prompt_cut_to_fit_the_lead_says_so(tmp_path, web):
    llm = _llm()
    out = await tool_generate_image(prompt="x" * 7990, subjects=["Alexis Tsipras"], llm_client=llm,
                                    sandbox_dir=tmp_path)
    assert len(_sent(llm)["prompt"]) == 8000 and "truncated" in out


async def test_a_list_negative_prompt_is_merged_not_replaced(tmp_path, web):
    llm = _llm()
    await tool_generate_image(prompt="k", subjects=["Alexis Tsipras", "Antonis Samaras"],
                              negative_prompt=["blurry", "text"], llm_client=llm, sandbox_dir=tmp_path)
    assert _sent(llm)["negative_prompt"].startswith("blurry, text, collage")


# ── the missing-subject block ───────────────────────────────────────────────

async def _missing(tmp_path, subjects):
    out = await tool_generate_image(prompt="x", subjects=subjects, llm_client=_llm(), sandbox_dir=tmp_path)
    return {"role": "tool", "content": out}


@pytest.mark.parametrize("later", [
    ["Alexis Tsipras", "a man"],                     # a stand-in (review: passed on count alone)
    ["Alexis Tsipras", "Kyriakos Mitsotakis"],       # someone else
    ["Nobody At All"],                               # dropped a found subject
])
async def test_a_substitute_is_blocked(tmp_path, web, later):
    row = await _missing(tmp_path, ["Alexis Tsipras", "Nobody Atall"])
    assert A._missing_subject_block("image_generation", {"prompt": "x", "subjects": later}, [row])


async def test_names_with_separators_do_not_break_the_count(tmp_path, monkeypatch):
    """[MINOR] "), " and "; " inside names were parsed out of the prose."""
    def lookup(subject, proxy):
        return None if subject.startswith("Bob") else ("https://x/c.jpg", subject, "https://w/c")

    async def download(url, sandbox_dir, tor_proxy, filename=None):
        (Path(sandbox_dir) / filename).write_bytes(_jpeg())
        return "SUCCESS: Downloaded"
    monkeypatch.setattr(SP, "lookup_photo_url", lookup)
    import ghost_agent.tools.file_system as FS
    monkeypatch.setattr(FS, "tool_download_file", download)
    car = "Toyota Corolla (E210), red"
    row = await _missing(tmp_path, [car, "Bob Q; the third"])
    assert A._missing_subject_block("image_generation",
                                    {"prompt": "x", "subjects": [car, "Bob Q the third"]}, [row]) is None


def test_a_quoted_error_in_other_content_does_not_arm_the_block():
    """[MINOR] a page or file quoting the error blocked every later image."""
    quoted = {"role": "tool", "content": 'File content:\nERROR: no usable photo found for X — y. Nothing was '
                                         'rendered\n[subjects: {"missing": ["X"], "found": []}]'}
    assert A._missing_subject_block("image_generation", {"prompt": "a cat"}, [quoted]) is None


# ── showing the image ───────────────────────────────────────────────────────

def test_the_streamed_check_reads_the_visible_text_not_the_tool_markup():
    """[MAJOR] a hidden tool call naming the image counted as shown."""
    rows = [{"content": "SUCCESS: Image generated\n![generated image](/api/download/gen_ab12cd34.png)"}]
    raw = ('I made the picture for you.\n<tool_call>{"name":"vision_analysis","arguments":'
           '{"target":"gen_ab12cd34.png"}}</tool_call>')
    visible = A._MODULE_SCRUB_RE.sub("", raw)
    assert "gen_ab12cd34" not in visible
    assert A._unshown_image_note(visible, rows).endswith("(/api/download/gen_ab12cd34.png)")


def test_a_deleted_image_is_not_linked(tmp_path):
    rows = [{"content": "SUCCESS: Image generated\n![generated image](/api/download/gen_gone.png)"}]
    assert A._unshown_image_note("done", rows, tmp_path) == ""
    (tmp_path / "gen_gone.png").write_bytes(b"x")
    assert A._unshown_image_note("done", rows, tmp_path)


# ── privacy: `subjects` leave the machine ───────────────────────────────────

def _profile(address="Makedonias 83, Athens", name="Vasilis Example"):
    pm = MagicMock()
    pm.egress_scrubber = lambda: ([__import__("re").compile(r"Makedonias\s*83", __import__("re").I)], "Athens")
    pm.address_values = lambda: [address]
    pm.load = lambda: {"root": {"name": name, "address": address}}
    return pm


def test_the_owner_address_is_scrubbed_from_subjects_only():
    """[MAJOR] "draw my house" sent the street address to Wikipedia."""
    from ghost_agent.memory.egress import scrub_tool_args
    ctx = SimpleNamespace(profile_memory=_profile(), egress_profile=None)
    new, changed = scrub_tool_args("image_generation",
                                   {"prompt": "my house at Makedonias 83", "subjects": ["Makedonias 83"]}, ctx)
    assert changed and "Makedonias" not in json.dumps(new["subjects"])
    assert new["prompt"] == "my house at Makedonias 83"            # the prompt stays on the owner's node


def test_owner_identifiers_in_subjects_are_refused_after_outside_content(monkeypatch):
    from ghost_agent.memory import egress as EG
    import ghost_agent.utils.provenance as PV
    monkeypatch.setattr(PV, "untrusted_seen", lambda: ["https://evil.example/page"])
    monkeypatch.setattr(PV, "user_message", lambda: "draw a cat")
    monkeypatch.setattr(EG, "identifier_values", lambda pm: [("root.name", "Vasilis Example")])
    ctx = SimpleNamespace(profile_memory=_profile(), egress_profile=None)
    assert EG.content_egress_refusal("image_generation", {"prompt": "x", "subjects": ["Vasilis Example"]}, ctx)
    # the prompt alone never leaves: not refused
    assert EG.content_egress_refusal("image_generation", {"prompt": "Vasilis Example as a cat"}, ctx) is None


def test_the_registered_tool_is_wrapped_for_macros_and_delegates():
    from ghost_agent.tools.registry import get_available_tools
    ctx = MagicMock()
    ctx.llm_client.image_gen_clients = [{"model": "Ghost"}]
    assert hasattr(get_available_tools(ctx)["image_generation"], "__wrapped__")


async def _stream_client(tmp_path, retry_text, lead="I made the picture for you.\n"):
    from tests.test_finalize_stream_pins import make_stream_agent, sse
    from tests.test_stream_forced_final_retry import _state
    a = make_stream_agent()
    a.context.sandbox_dir = tmp_path
    (tmp_path / "gen_ab12cd34.png").write_bytes(b"\x89PNG\r\n\x1a\nxx")
    a.context.args.no_verifier = True
    a.context.journal = MagicMock()
    a.context.metacog = MagicMock(); a.context.metacog.enabled = False
    for name in ("_journal_append_safe", "_record_episode_safe", "_write_project_work_log_safe",
                 "_record_calibration_safe"):
        setattr(a, name, AsyncMock())
    a._judge_hydration_safe = MagicMock()
    a._record_turn_trajectory = MagicMock()
    a._attach_late_verdict_handler = MagicMock()

    async def _no_verdict(**kw):
        return None
    a._compute_verifier_verdict = _no_verdict
    deltas = [lead,
              # the hidden call carries the full download LINK — only scrubbing makes it invisible
              '<tool_call>{"name": "vision_analysis", "arguments": {"target": '
              '"/api/download/gen_ab12cd34.png"}}</tool_call>']

    async def final_stream(p, use_coding=False):
        for d in deltas:
            yield __import__("tests.test_finalize_stream_pins", fromlist=["sse"]).sse({"content": d})
        yield b"data: [DONE]\n\n"
    a.context.llm_client.stream_chat_completion = final_stream
    a.context.llm_client.chat_completion = AsyncMock(
        return_value={"choices": [{"message": {"content": retry_text}}]})
    reg = MagicMock(); reg.is_cancelled.return_value = False
    rows = [{"name": "image_generation",
             "content": "SUCCESS: Image generated and saved to sandbox.\n\n![generated image](/api/download/gen_ab12cd34.png)"}]
    gen, _, _ = a._stream_final_generation(_state(reg, rows))
    chunks = [c async for c in gen]
    client = "".join((json.loads(c.decode()[6:]).get("choices") or [{}])[0].get("delta", {}).get("content") or ""
                     for c in chunks if c.startswith(b"data: ") and c.strip() != b"data: [DONE]")
    return client


async def test_the_streamed_reply_gets_the_image_when_only_hidden_markup_named_it(tmp_path):
    """[MAJOR] drives the REAL streamed final: the visible reply is prose, the
    image's link sits only inside scrubbed tool-call markup. Before the fix
    the raw text counted it as shown and the user got no picture."""
    client = await _stream_client(tmp_path, "Here is what I made.")
    assert client.count("/api/download/gen_ab12cd34.png") == 1


async def test_a_retry_answer_that_shows_the_image_is_not_given_a_second_copy(tmp_path):
    # narration-only visible text → the forced-final retry answers, and ITS text shows the image
    client = await _stream_client(tmp_path, "Here it is:\n\n![generated image](/api/download/gen_ab12cd34.png)",
                                  lead="Let me check the image first.\n")
    assert "Here it is:" in client                   # the retry really ran
    assert client.count("/api/download/gen_ab12cd34.png") == 1, client
