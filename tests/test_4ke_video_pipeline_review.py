"""§4KE (2026-09-24): the video-reading review and the YouTube-over-Tor route.

Three fresh-eye lenses read the transcript pipeline; every pin here is a
behaviour one of them found missing or wrong, written in the CONSUMER's
words (what the model or the operator sees), never as a source-text grep.

Ground truth the design rests on (measured live 2026-09-24, over Tor):
plain yt-dlp 0/9 circuits ("Sign in to confirm you're not a bot"); with a
locally-minted PO token 5/12 circuits, failures in 2–9 s; the old macro had
3 uses and 0 successes because its shell step could not finish inside the
sandbox's 90 s job promotion.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
for p in (str(_ROOT), str(_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from ghost_agent.memory import audio_ingest as ai  # noqa: E402
from ghost_agent.memory import youtube_ingest as yi  # noqa: E402
from ghost_agent.tools import composed_skills as cs  # noqa: E402
from ghost_agent.tools import memory as mem  # noqa: E402
from ghost_agent.tools import yt_download as ytd  # noqa: E402

VID = "jNQXAC9IVRw"
URL = f"https://www.youtube.com/watch?v={VID}"
TOR = "socks5://127.0.0.1:9050"


# ─────────────────────────── fakes ───────────────────────────

class FakeMemory:
    """The slice of VectorMemory the ingest paths touch, recording every write."""

    def __init__(self, library=None):
        self.library = list(library or [])
        self.ingested: List[tuple] = []
        self.outline: Dict[str, dict] = {}
        self.text: Dict[str, dict] = {}
        self.rows: List[tuple] = []

    def get_library(self):
        return list(self.library)

    def ingest_document(self, filename, chunks, _batch=False):
        self.ingested.append((filename, list(chunks), _batch))
        if filename not in self.library:
            self.library.append(filename)
        return True, f"ok {len(chunks)}"

    def set_document_outline(self, filename, record):
        self.outline[filename] = record

    def get_document_outline(self, filename):
        return self.outline.get(filename, {})

    def set_document_text(self, filename, record):
        self.text[filename] = record

    def get_document_text(self, filename):
        return self.text.get(filename, {})

    def add(self, text, meta):
        self.rows.append((text, meta))


def _info(**over):
    base = {
        "id": VID, "title": "Me at the zoo", "duration": 19, "language": "en",
        "subtitles": {}, "automatic_captions": {},
        "webpage_url": URL,
    }
    base.update(over)
    return base


JSON3 = json.dumps({"events": [
    {"tStartMs": 1200, "dDurationMs": 2160, "segs": [{"utf8": "All right, "}, {"utf8": "so here we are"}]},
    {"tStartMs": 3400, "dDurationMs": 10, "segs": [{"utf8": "\n"}]},
    {"tStartMs": 5318, "dDurationMs": 2656, "segs": [{"utf8": "the cool thing about these guys"}]},
    {"tStartMs": 16881, "dDurationMs": 2000, "segs": [{"utf8": "and that's pretty much all there is to say"}]},
]})

BOT_ERR = ("WARNING: [youtube] No title found in player responses\n"
           "ERROR: [youtube] jNQXAC9IVRw: Sign in to confirm you’re not a bot. Use --cookies-from-browser")


def _proxy_of(cmd: List[str]) -> str:
    return cmd[cmd.index("--proxy") + 1]


class ScriptedRunner:
    """A yt-dlp stand-in: a list of behaviours consumed per invocation.
    Each behaviour is a callable ``(cmd, timeout, cwd) -> (rc, out, err)``."""

    def __init__(self, behaviours):
        self.behaviours = list(behaviours)
        self.calls: List[List[str]] = []

    def __call__(self, cmd, timeout, cwd):
        self.calls.append(list(cmd))
        if not self.behaviours:
            raise AssertionError(f"unexpected yt-dlp call #{len(self.calls)}: {cmd[-1]}")
        return self.behaviours.pop(0)(cmd, timeout, cwd)


def bot_wall(cmd, timeout, cwd):
    return 1, "", BOT_ERR


def info_ok(info):
    def _b(cmd, timeout, cwd):
        assert "--dump-single-json" in cmd
        return 0, json.dumps(info) + "\n", ""
    return _b


def captions_ok(text=JSON3, ext="json3"):
    def _b(cmd, timeout, cwd):
        assert "--write-subs" in cmd and "--write-auto-subs" in cmd
        lang = cmd[cmd.index("--sub-langs") + 1]
        assert "," not in lang, "one language per request — a second one drew a 429"
        out_tpl = cmd[cmd.index("-o") + 1]
        Path(f"{out_tpl}.{lang}.{ext}").write_text(text, encoding="utf-8")
        return 0, "", ""
    return _b


def audio_ok(cmd, timeout, cwd):
    assert "ba[ext=m4a]/ba" in cmd
    tpl = cmd[cmd.index("-o") + 1]
    path = tpl.replace("%(ext)s", "m4a")
    Path(path).write_bytes(b"\x00" * 2048)
    return 0, path + "\n", ""


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    monkeypatch.setattr(yi, "ensure_pot_server", lambda **kw: True)
    monkeypatch.setattr(yi, "_run_subprocess", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("the real yt-dlp must never run in a unit test")))


# ─────────────────────────── URL handling ───────────────────────────

@pytest.mark.parametrize("url", [
    URL, f"https://youtu.be/{VID}", f"https://youtu.be/{VID}?t=42",
    f"https://m.youtube.com/watch?v={VID}&list=PL123&index=2",
    f"https://www.youtube.com/shorts/{VID}", f"https://www.youtube.com/live/{VID}?feature=share",
    f"https://www.youtube.com/embed/{VID}", f"https://music.youtube.com/watch?v={VID}",
    f"youtube.com/watch?v={VID}",
])
def test_every_video_url_shape_yields_the_id(url):
    assert yi.youtube_video_id(url) == VID
    assert yi.is_youtube_url(url)


@pytest.mark.parametrize("url", [
    "https://www.youtube.com/@somechannel", "https://www.youtube.com/playlist?list=PL123",
    f"https://notyoutube.com/watch?v={VID}", f"https://youtube.com.evil.example/watch?v={VID}",
    f"https://example.com/watch?v={VID}", "https://www.youtube.com/watch?v=tooshort",
    "", None, 42,
])
def test_channels_playlists_and_lookalike_hosts_are_not_videos(url):
    """A channel page or a look-alike host must NOT take the YouTube route —
    it would fall to the generic web fetch, which is the right thing there."""
    assert yi.youtube_video_id(url) is None
    assert not yi.is_youtube_url(url)


def test_the_circuit_proxy_keeps_dns_inside_tor_and_differs_per_attempt():
    """``socks5://`` resolves names LOCALLY — the video host lookup would leave
    Tor. Every attempt must carry its own SOCKS identity (that is what makes
    Tor build a fresh circuit)."""
    a = yi.circuit_proxy(TOR, "ytresolveabc1")
    b = yi.circuit_proxy(TOR, "ytresolveabc2")
    assert a.startswith("socks5h://") and b.startswith("socks5h://")
    assert a != b
    assert "ytresolveabc1:" in a and "@127.0.0.1:9050" in a


def test_ytdlp_runs_from_this_interpreter_with_proxy_and_pot_helper():
    cmd = yi.ytdlp_base_command("socks5h://t:isolate@127.0.0.1:9050")
    assert cmd[:3] == [sys.executable, "-m", "yt_dlp"]
    assert cmd[cmd.index("--proxy") + 1] == "socks5h://t:isolate@127.0.0.1:9050"
    assert any(a.startswith("youtubepot-bgutilhttp:base_url=") for a in cmd)
    assert "--no-playlist" in cmd


# ─────────────────────────── failure classes ───────────────────────────

@pytest.mark.parametrize("stderr,kind", [
    (BOT_ERR, "retry"),
    ("ERROR: Unable to download video subtitles for 'en': HTTP Error 429: Too Many Requests", "retry"),
    ("ERROR: [youtube] x: Requested format is not available. Use --list-formats", "retry"),
    ("ERROR: [youtube] x: Unable to download API page: HTTP Error 403: Forbidden", "retry"),
    ("ERROR: timed out after 75s", "retry"),
    ("ERROR: [youtube] x: Private video. Sign in if you've been granted access to this video", "terminal"),
    ("ERROR: [youtube] x: Video unavailable. This video has been removed by the uploader", "terminal"),
    ("ERROR: [youtube] x: The uploader has not made this video available in your country", "retry"),
    ("ERROR: [youtube] x: No video formats found!; please report this issue", "retry"),
    ("ERROR: [generic] Unsupported URL: https://example.com/", "terminal"),
    ("ERROR: something nobody has seen before", "terminal"),
])
def test_only_circuit_faults_earn_another_circuit(stderr, kind):
    """The old script rotated on "video unavailable" too — 8 circuits and
    57 s of backoff for a removed video, and a misdiagnosis ("every exit
    blocked") at the end. Unknown errors STOP: rotation is for the network."""
    got, reason = yi.classify_failure(stderr)
    assert got == kind
    assert reason.startswith("ERROR"), "the reason is yt-dlp's own last ERROR line"


# ─────────────────────────── captions ───────────────────────────

def test_caption_choice_prefers_manual_then_the_original_auto_track_never_a_translation():
    """yt-dlp lists machine TRANSLATIONS under automatic_captions too; only
    ``<lang>-orig`` is the speaker's language. A Greek talk must not come
    back as English because 'en' sorted first."""
    auto = {"en": [{"ext": "json3"}], "de": [{"ext": "json3"}], "el-orig": [{"ext": "json3"}],
            "el": [{"ext": "json3"}]}
    assert yi.choose_caption_track(_info(language="el", automatic_captions=auto)) == ("el-orig", "auto")
    assert yi.choose_caption_track(_info(language="el", automatic_captions=auto,
                                         subtitles={"el": [{"ext": "vtt"}]})) == ("el", "manual")
    # an explicit request wins, and a regional manual track satisfies it
    assert yi.choose_caption_track(_info(language="el", automatic_captions=auto,
                                         subtitles={"en-US": [{"ext": "vtt"}]}), "en") == ("en-US", "manual")
    # no language on the video: any manual track, else any -orig track
    assert yi.choose_caption_track(_info(language=None, automatic_captions=auto)) == ("el-orig", "auto")
    # live chat is not a caption; nothing usable → None (→ audio tier)
    assert yi.choose_caption_track(_info(subtitles={"live_chat": [{"ext": "json"}]})) is None


def test_a_requested_language_is_never_satisfied_by_a_machine_translation():
    """Round 2: `language='en'` against a Greek talk used to return the
    translated automatic 'en' track and file it as English captions, against
    the registry's own promise. Now it falls through to the original and the
    reply says so."""
    auto = {"en": [{"ext": "json3"}], "el-orig": [{"ext": "json3"}], "el": [{"ext": "json3"}]}
    chosen = yi.choose_caption_track(_info(language="el", automatic_captions=auto), "en")
    assert chosen == ("el-orig", "auto")
    note = yi.caption_language_note("en", chosen)
    assert "machine translation" in note and "'el'" in note
    assert yi.caption_language_note("el", chosen) == ""
    # an original English track in a regional key satisfies 'en', and vice versa
    assert yi.choose_caption_track(_info(language="en-US", automatic_captions={"en-orig": [1], "de": [1]})) == ("en-orig", "auto")
    assert yi.choose_caption_track(_info(language="en", automatic_captions={"en-US": [1], "de": [1]})) == ("en-US", "auto")
    assert yi._base_lang("zh-Hans") == "zh-hans" and yi._base_lang("pt-BR") == "pt" and yi._base_lang("el-orig") == "el"


def test_json3_joins_word_segments_and_drops_window_markers():
    segs = yi.parse_json3(JSON3)
    assert [t for _s, _e, t in segs] == [
        "All right, so here we are", "the cool thing about these guys",
        "and that's pretty much all there is to say"]
    assert segs[0][0] == pytest.approx(1.2) and segs[0][1] == pytest.approx(3.36)


def test_vtt_strips_tags_and_collapses_rolling_duplicates():
    # Rolling automatic captions repeat the previous cue's LINE verbatim
    # before adding the next one; the repeat must not be stored twice.
    vtt = ("WEBVTT\nKind: captions\nLanguage: en\n\n"
           "00:00:01.200 --> 00:00:03.360\n<c>All right,</c> so here we are\n\n"
           "00:00:03.360 --> 00:00:05.000\nAll right, so here we are\nin front of the elephants\n\n"
           "01:02:03.000 --> 01:02:04.000\nan hour in\n")
    segs = yi.parse_vtt(vtt)
    assert segs[0] == (pytest.approx(1.2), pytest.approx(3.36), "All right, so here we are")
    assert segs[1][2] == "in front of the elephants", "the repeated rolling line is not stored twice"
    assert segs[2][0] == pytest.approx(3723.0)


def test_passages_group_by_time_then_size_and_carry_the_audio_style_crumb():
    segs = [(float(i * 10), float(i * 10 + 9), f"line {i} " + "x" * 50) for i in range(30)]
    passages = yi.segments_to_passages(segs, passage_seconds=90, max_chars=1200)
    assert len(passages) >= 3
    assert all(e - s <= 90 + 9 for s, e, _t in passages)
    assert passages[0][2].startswith("line 0") and passages[-1][2].endswith("x" * 50)
    chunk = yi.passage_chunk("yt-abc.captions.en", 0.0, 89.0, "hello")
    assert chunk.split("\n", 1)[0] == "[yt-abc.captions.en] [0:00–1:29]", (
        "same breadcrumb shape as audio_ingest, so citation and outline derivation are shared")
    # size cap, not only time
    big = [(0.0, 1.0, "w" * 700), (1.0, 2.0, "v" * 700)]
    assert len(yi.segments_to_passages(big, passage_seconds=90, max_chars=1200)) == 2


# ─────────────────────────── the route ───────────────────────────

def test_a_blocked_exit_is_rotated_past_and_the_captions_tier_indexes_the_video(tmp_path):
    """Two circuits hit the bot wall, the third resolves; captions are fetched
    on THAT circuit (the PO token is cached per session), indexed under a
    per-video name, and the reply carries the transcript itself."""
    m = FakeMemory()
    runner = ScriptedRunner([bot_wall, bot_wall, info_ok(_info(
        automatic_captions={"en-orig": [{"ext": "json3"}]})), captions_ok()])
    res = yi.ingest_youtube(URL, sandbox_dir=tmp_path, memory_system=m, tor_proxy=TOR,
                            run=runner, now="2026-09-24T10:00:00Z")
    assert res.ok, res.message
    proxies = [_proxy_of(c) for c in runner.calls]
    assert len(set(proxies[:3])) == 3, "each resolve attempt rode a different circuit"
    assert proxies[3] == proxies[2], "captions were fetched on the circuit that passed"
    assert res.fetch.attempts == 3 and res.fetch.bot_walls == 2
    assert res.route == "captions" and res.filename == f"yt-{VID}.captions.en"
    # stored: chunks with crumbs, an outline entry per passage, the ordered text, a content summary
    assert m.ingested and m.ingested[0][0] == res.filename and m.ingested[0][2] is True
    assert m.ingested[0][1][0].startswith(f"[yt-{VID}.captions.en] [0:01–")
    assert m.outline[res.filename]["entries"] and m.outline[res.filename]["title"] == "Me at the zoo"
    assert m.text[res.filename]["passages"][0][2].startswith("All right")
    summary = m.rows[0][0]
    assert "Me at the zoo" in summary and "Opening words: [0:01" in summary and m.rows[0][1]["type"] == "document_summary"
    # the reply is the transcript, not a receipt
    assert res.message.startswith("SUCCESS:") and "TRANSCRIPT (" in res.message
    assert "All right, so here we are" in res.message
    assert "action='transcript'" in res.message
    assert tmp_path.exists() and not list(tmp_path.glob("*.json3")), "caption files do not litter the sandbox"


def test_a_terminal_fault_stops_after_one_circuit_and_names_the_cause(tmp_path):
    m = FakeMemory()
    runner = ScriptedRunner([lambda c, t, w: (1, "", "ERROR: [youtube] x: Private video. Sign in if you've been granted access")])
    res = yi.ingest_youtube(URL, sandbox_dir=tmp_path, memory_system=m, tor_proxy=TOR, run=runner)
    assert not res.ok and res.message.startswith("Error:")
    assert len(runner.calls) == 1, "no rotation for a wall no exit can pass"
    assert "Private video" in res.message
    assert not m.ingested and not m.rows and not m.text


def test_when_every_circuit_is_walled_the_report_counts_them_and_stores_nothing(tmp_path):
    m = FakeMemory()
    runner = ScriptedRunner([bot_wall] * 3)
    res = yi.ingest_youtube(URL, sandbox_dir=tmp_path, memory_system=m, tor_proxy=TOR,
                            run=runner, attempts=3)
    assert not res.ok and "after 3 circuit(s)" in res.message
    assert "not a bot" in res.message
    assert not m.ingested and not m.rows


def test_a_down_pot_helper_is_named_as_the_cause(tmp_path, monkeypatch):
    """Without the helper every exit is refused. The old script reported that
    as "every Tor exit was blocked" — the wrong lever. Name the real one."""
    monkeypatch.setattr(yi, "ensure_pot_server", lambda **kw: False)
    m = FakeMemory()
    res = yi.ingest_youtube(URL, sandbox_dir=tmp_path, memory_system=m, tor_proxy=TOR,
                            run=ScriptedRunner([bot_wall] * 2), attempts=2)
    assert not res.ok
    assert "PO-token helper" in res.message and "com.local.ghost-pot" in res.message


def test_no_captions_falls_back_to_audio_with_a_per_video_filename(tmp_path):
    """The audio tier downloads ``yt-<id>.m4a`` into the sandbox and hands it
    to the audio ingest; gaps the transcriber reports reach the reply."""
    m = FakeMemory()
    runner = ScriptedRunner([info_ok(_info(duration=600)), audio_ok])
    seen: Dict[str, Any] = {}

    def fake_audio_ingest(path, filename, memory_system, *, progress=None, **kw):
        seen["path"] = Path(path); seen["filename"] = filename
        st = ai.AudioIngestStats(windows=2, seconds=590.0, total_seconds=600.0, chunks=4)
        st.transcript = [(0.0, 300.0, "first half of the talk"), (300.0, 590.0, "second half")]
        st.gaps = ["9:50–10:00 (node timeout)"]
        if progress:
            progress(st)
        return st

    res = yi.ingest_youtube(URL, sandbox_dir=tmp_path, memory_system=m, tor_proxy=TOR,
                            run=runner, audio_ingest=fake_audio_ingest)
    assert res.ok, res.message
    assert res.route == "audio" and res.filename == f"yt-{VID}.m4a"
    assert seen["path"] == tmp_path / f"yt-{VID}.m4a" and seen["path"].exists()
    assert not m.ingested, "the audio ingest stores its own chunks; the route must not double-store"
    assert m.text[res.filename]["passages"][1][2] == "second half"
    assert "9:50–10:00" in res.message and "Gaps (not transcribed)" in res.message
    assert "via audio transcription" in res.message
    assert _proxy_of(runner.calls[1]) != _proxy_of(runner.calls[0]), "the download gets its own circuit"


def test_a_partial_audio_transcription_is_kept_and_labelled(tmp_path):
    m = FakeMemory()
    runner = ScriptedRunner([info_ok(_info(duration=600)), audio_ok])

    def dying_ingest(path, filename, memory_system, *, progress=None, **kw):
        st = ai.AudioIngestStats(windows=1, seconds=100.0, total_seconds=600.0, chunks=1)
        st.transcript = [(0.0, 100.0, "the part that made it")]
        st.gaps = ["1:30–10:00 (not attempted: 3 consecutive windows failed)"]
        st.aborted = "3 consecutive windows failed (last: HTTP 503) — the audio node is not answering"
        return st
    res = yi.ingest_youtube(URL, sandbox_dir=tmp_path, memory_system=m, tor_proxy=TOR,
                            run=runner, audio_ingest=dying_ingest)
    assert res.ok and res.message.startswith("SUCCESS (partial):")
    assert "STOPPED EARLY" in res.message and f"target='yt-{VID}.m4a'" in res.message
    assert m.text[res.filename]["passages"] == [[0.0, 100.0, "the part that made it"]]


def test_audio_left_behind_by_an_earlier_attempt_is_reused_not_redownloaded(tmp_path):
    (tmp_path / f"yt-{VID}.m4a").write_bytes(b"\x00" * 4096)
    (tmp_path / f"yt-{VID}.m4a.part").write_bytes(b"\x00")
    m = FakeMemory()
    runner = ScriptedRunner([info_ok(_info(duration=60))])
    seen = {}

    def fake_ingest(path, filename, memory_system, *, progress=None, **kw):
        seen["path"] = Path(path)
        st = ai.AudioIngestStats(windows=1, seconds=60.0, total_seconds=60.0, chunks=1)
        st.transcript = [(0.0, 60.0, "words")]
        return st
    res = yi.ingest_youtube(URL, sandbox_dir=tmp_path, memory_system=m, tor_proxy=TOR,
                            run=runner, audio_ingest=fake_ingest)
    assert res.ok and len(runner.calls) == 1, "no download call was made"
    assert seen["path"].name == f"yt-{VID}.m4a"


def test_the_pot_hint_judges_the_stage_that_failed_not_the_whole_run():
    st = yi.FetchStats(attempts=3, bot_walls=2, stage="audio", stage_attempts=2, stage_walls=2)
    hint = yi._pot_hint(st, pot_up=True)
    assert "audio stage" in hint and "bot wall" in hint
    st2 = yi.FetchStats(attempts=3, bot_walls=1, stage="audio", stage_attempts=2, stage_walls=1)
    assert yi._pot_hint(st2, pot_up=True) == ""


def test_the_audio_tier_refuses_a_recording_past_the_duration_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(yi, "MAX_AUDIO_SECONDS", 3600.0)
    m = FakeMemory()
    runner = ScriptedRunner([info_ok(_info(duration=5 * 3600))])
    res = yi.ingest_youtube(URL, sandbox_dir=tmp_path, memory_system=m, tor_proxy=TOR, run=runner)
    assert not res.ok and "capped" in res.message and len(runner.calls) == 1


def test_the_same_video_is_skipped_by_id_before_any_network(tmp_path):
    """Dedup is keyed on the VIDEO, whichever tier ingested it — the old
    constant filename made the SECOND video ever a silent no-op."""
    m = FakeMemory(library=[f"yt-{VID}.captions.en"])
    runner = ScriptedRunner([])
    res = yi.ingest_youtube(f"https://youtu.be/{VID}", sandbox_dir=tmp_path, memory_system=m,
                            tor_proxy=TOR, run=runner)
    assert res.ok and res.message.startswith("Skipped:") and f"yt-{VID}.captions.en" in res.message
    assert "action='transcript'" in res.message
    assert runner.calls == []
    other = yi.ingest_youtube("https://youtu.be/abcdefghijk", sandbox_dir=tmp_path, memory_system=m,
                              tor_proxy=TOR, run=ScriptedRunner([bot_wall]), attempts=1)
    assert not other.message.startswith("Skipped:"), "a different video is not the same document"


def test_the_route_is_tor_only(tmp_path):
    res = yi.ingest_youtube(URL, sandbox_dir=tmp_path, memory_system=FakeMemory(), tor_proxy=None,
                            run=ScriptedRunner([]))
    assert not res.ok and "Tor-only" in res.message


# ─────────────────────────── audio_ingest: the transcriber ───────────────────────────

def _post(content, finish="stop"):
    def _p(url, payload, timeout):
        _p.payloads.append(payload)
        return {"choices": [{"message": {"content": content}, "finish_reason": finish}]}
    _p.payloads = []
    return _p


def test_thinking_is_off_for_transcription_unless_asked(monkeypatch):
    """Measured on nova: thinking cost 1346 of 2425 completion tokens and 36 s
    on a 5-minute window for an identical transcript, and pushed a dense
    12-minute window toward the 8192 cap."""
    p = _post("hello")
    ai.transcribe_window(b"WAV", post_fn=p)
    assert p.payloads[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert p.payloads[0]["temperature"] == 0.0
    monkeypatch.setattr(ai, "THINKING_ENABLED", True)
    p2 = _post("hello")
    ai.transcribe_window(b"WAV", post_fn=p2)
    assert "chat_template_kwargs" not in p2.payloads[0]


def test_a_window_cut_at_the_token_budget_is_flagged_not_stored_as_complete():
    """Only the empty+length shape used to raise; text+length was stored under
    a crumb claiming the whole window."""
    text, cut = ai.transcribe_window_full(b"WAV", post_fn=_post("half a transcript", "length"))
    assert text == "half a transcript" and cut is True
    text, cut = ai.transcribe_window_full(b"WAV", post_fn=_post("whole transcript", "stop"))
    assert cut is False


@pytest.mark.parametrize("reply", [
    "(no speech)", "No speech.", '"(no speech)"', "**(no speech)**", "Transcript: (no speech)",
    "(χωρίς ομιλία)", "(music)", "[silence]", "There is no intelligible speech in this audio.",
    "  (NO SPEECH)  ", "(κανένας λόγος)",
])
def test_every_honest_nothing_here_reply_is_silence_not_a_passage(reply):
    """A prefix match on one English literal stored every other honest
    "nothing here" answer as a citable passage of the talk."""
    assert ai.transcribe_window(b"WAV", post_fn=_post(reply)) == ""


def test_real_speech_behind_a_stray_sentinel_is_kept_and_short_speech_is_not_silence():
    speech = "(no speech) " + "and then the speaker actually began the talk about " * 3
    out = ai.transcribe_window(b"WAV", post_fn=_post(speech))
    assert out.startswith("and then the speaker"), "the old prefix match discarded the speech"
    assert ai.transcribe_window(b"WAV", post_fn=_post("hello there")) == "hello there"
    assert ai.transcribe_window(b"WAV", post_fn=_post("(Γεια σας, καλώς ήρθατε στην ομιλία μας για τα δίκτυα Tor και την ανωνυμία)")) != ""


def _windows(monkeypatch, total, posts, **kw):
    """Drive iter_audio_chunks with ffmpeg/ffprobe stubbed; ``posts`` is a
    list of behaviours per POST (a string = transcript, an Exception = raise)."""
    monkeypatch.setattr(ai, "probe_duration_seconds", lambda p: total)
    monkeypatch.setattr(ai, "extract_window_wav", lambda p, s, d: b"WAV")
    calls = []

    def post(url, payload, timeout):
        calls.append(payload)
        b = posts.pop(0) if posts else "tail text"
        if isinstance(b, Exception):
            raise b
        return {"choices": [{"message": {"content": b}, "finish_reason": "stop"}]}
    st = ai.AudioIngestStats()
    chunks = list(ai.iter_audio_chunks(Path("x.mp3"), "x.mp3", stats=st, post_fn=post,
                                       window_seconds=100, window_overlap=10, sleep=lambda s: None, **kw))
    return st, chunks, calls


def test_a_failed_middle_window_is_a_gap_and_coverage_excludes_it(monkeypatch):
    """1:00:00 was reported after window 3 of 5 died — the coverage number
    came from the max endpoint and the error list reached nobody."""
    st, chunks, _ = _windows(monkeypatch, 280.0,
                             ["first window words", RuntimeError("audio node returned HTTP 400: bad"), "third window words"])
    assert st.windows == 2 and st.skipped_windows == 1
    assert st.total_seconds == 280.0
    assert st.seconds == pytest.approx(100.0 + (280.0 - 180.0))
    assert st.gaps == ["1:30–3:10 (audio node returned HTTP 400: bad)"]
    assert [t for _s, _e, t in st.transcript] == ["first window words", "third window words"]
    assert all(c.startswith("[x.mp3] [") for c in chunks)


def test_a_transient_node_fault_is_retried_once_on_the_same_window(monkeypatch):
    st, _, calls = _windows(monkeypatch, 100.0, [ai.TransientNodeError("reset"), "recovered words"])
    assert st.windows == 1 and st.retries == 1 and st.skipped_windows == 0
    assert len(calls) == 2
    st2, _, calls2 = _windows(monkeypatch, 100.0, [ai.TransientNodeError("a"), ai.TransientNodeError("b")])
    assert st2.skipped_windows == 1 and st2.retries == 1
    assert st2.gaps[0] == "0:00–1:40 (b)", "one retry, then the window is a gap naming the LAST cause"
    assert len(calls2) >= 2 and st2.windows <= 1, "the failed window is not retried a third time"


def test_three_consecutive_failures_stop_the_loop_and_report_the_rest_as_a_gap(monkeypatch):
    """A starved node used to cost windows × 900 s; now the loop stops after
    three NODE failures in a row, says why, and reports the untried tail as
    one gap. Nothing already transcribed is lost (see the partial-kept pin)."""
    posts = ["good first window"] + [RuntimeError("HTTP 400")] * 40
    st, chunks, calls = _windows(monkeypatch, 2000.0, posts)
    assert "3 consecutive windows failed" in st.aborted and "not answering" in st.aborted
    assert len(calls) == 4, "one good window, then exactly three attempts (non-transient → no retry)"
    assert st.windows == 1 and chunks and st.transcript == [(0.0, 100.0, "good first window")]
    assert st.gaps[-1].startswith("6:00–33:20 (not attempted: 3 consecutive")


def test_windows_transcribed_before_the_node_died_are_kept_and_the_reply_says_partial(monkeypatch):
    """Round 2: the abort used to raise out of the generator, so the caller's
    pending batch — every good window — was thrown away, and past 256 flushed
    chunks the library entry made the video a permanent 'Skipped'."""
    monkeypatch.setattr(ai, "probe_duration_seconds", lambda p: 460.0)
    monkeypatch.setattr(ai, "extract_window_wav", lambda p, s, d: b"WAV")
    m = FakeMemory()
    st = ai.ingest_audio_streaming(Path("talk.mp3"), "talk.mp3", m,
                                   post_fn=_post_seq(["kept words"] + [RuntimeError("HTTP 400")] * 3),
                                   window_seconds=100, window_overlap=10, sleep=lambda s: None)
    assert st.aborted and m.ingested and "kept words" in m.ingested[0][1][0]
    msg = mem._audio_success_message("talk.mp3", st, st.transcript, st.gaps)
    assert msg.startswith("SUCCESS (partial):") and "STOPPED EARLY" in msg
    assert "action='forget', target='talk.mp3'" in msg, "the redo path is named"
    assert "[0:00–1:40] kept words" in msg


def test_ffmpeg_failures_are_the_files_fault_and_never_trip_the_node_abort(monkeypatch):
    monkeypatch.setattr(ai, "probe_duration_seconds", lambda p: 460.0)

    def bad_cut(p, s, d):
        raise RuntimeError("'ffmpeg' failed: Invalid data found when processing input")
    monkeypatch.setattr(ai, "extract_window_wav", bad_cut)
    st = ai.AudioIngestStats()
    list(ai.iter_audio_chunks(Path("x.mp3"), "x.mp3", stats=st, post_fn=_post("never"),
                              window_seconds=100, window_overlap=10, sleep=lambda s: None))
    assert st.skipped_windows == 6 and st.aborted == "", "every window skipped (nothing covered, so the tail runs too), no abort"
    assert all("ffmpeg" in g for g in st.gaps)


def test_the_wall_budget_stops_the_loop_and_keeps_the_rest_as_a_gap(monkeypatch):
    st, _, calls = _windows(monkeypatch, 1000.0, ["a", "b"], total_budget_s=0.0)
    assert calls == [] and st.windows == 0
    assert "time budget" in st.aborted and st.gaps == ["0:00–16:40 (not attempted: time budget of 0:00 exhausted)"]


def test_a_second_ingest_waits_a_bounded_time_then_says_so(monkeypatch):
    monkeypatch.setattr(ai, "LOCK_WAIT_S", 0.2)
    monkeypatch.setattr(ai, "probe_duration_seconds", lambda p: 100.0)
    monkeypatch.setattr(ai, "extract_window_wav", lambda p, s, d: b"WAV")
    assert ai._INGEST_LOCK.acquire(timeout=1)
    try:
        with pytest.raises(RuntimeError, match="another transcription"):
            ai.ingest_audio_streaming(Path("x.mp3"), "x.mp3", FakeMemory(), post_fn=_post("w"),
                                      window_seconds=100, window_overlap=10)
    finally:
        ai._INGEST_LOCK.release()
    assert ai._INGEST_LOCK.acquire(timeout=1); ai._INGEST_LOCK.release()


def _post_seq(behaviours):
    seq = list(behaviours)

    def post(url, payload, timeout):
        b = seq.pop(0) if seq else "tail"
        if isinstance(b, Exception):
            raise b
        return {"choices": [{"message": {"content": b}, "finish_reason": "stop"}]}
    return post


def test_a_tail_already_inside_the_previous_window_is_not_transcribed_again(monkeypatch):
    """1425 s at 720/15 used to produce a THIRD window [23:30–23:45] lying
    entirely inside the second one's overlap — a full node round-trip for
    text the store already held."""
    st, _, calls = _windows(monkeypatch, 190.0, ["a", "b", "c"])
    assert st.windows == 2 and len(calls) == 2
    assert st.seconds == 190.0 and st.transcript[-1][1] == 190.0


def test_a_truncated_window_is_counted_and_its_range_reported(monkeypatch):
    monkeypatch.setattr(ai, "probe_duration_seconds", lambda p: 100.0)
    monkeypatch.setattr(ai, "extract_window_wav", lambda p, s, d: b"WAV")
    st = ai.AudioIngestStats()
    list(ai.iter_audio_chunks(Path("x.mp3"), "x.mp3", stats=st, post_fn=_post("cut off mid", "length"),
                              window_seconds=100, window_overlap=10, sleep=lambda s: None))
    assert st.truncated_windows == 1 and st.windows == 1
    assert st.gaps == ["0:00–1:40 (tail cut at the token budget)"]
    assert st.transcript == [(0.0, 100.0, "cut off mid")]


def test_progress_fires_after_every_window_not_at_the_batch_boundary(monkeypatch):
    """The batch boundary (256 chunks) is hours of audio away; a 45-minute
    talk used to report nothing until it was done."""
    monkeypatch.setattr(ai, "probe_duration_seconds", lambda p: 280.0)
    monkeypatch.setattr(ai, "extract_window_wav", lambda p, s, d: b"WAV")
    ticks = []
    m = FakeMemory()
    st = ai.ingest_audio_streaming(Path("x.mp3"), "x.mp3", m, progress=lambda s: ticks.append(s.windows),
                                   post_fn=_post("words"), window_seconds=100, window_overlap=10)
    assert ticks == [1, 2, 3]
    assert st.windows == 3 and m.ingested and m.ingested[0][2] is True


def test_the_cut_takes_the_first_audio_stream_and_a_short_timeout(monkeypatch, tmp_path):
    seen = {}

    def fake_run(cmd, *, timeout):
        seen["cmd"], seen["timeout"] = cmd, timeout
        out = Path(cmd[cmd.index("-f") + 2])
        out.write_bytes(b"RIFF")
        return MagicMock(returncode=0, stdout=b"")
    monkeypatch.setattr(ai, "_run", fake_run)
    assert ai.extract_window_wav(tmp_path / "in.mp4", 10.0, 5.0) == b"RIFF"
    cmd = seen["cmd"]
    assert cmd[cmd.index("-map") + 1] == "0:a:0", "ffmpeg's default picks the MOST channels (system audio over the mic)"
    assert seen["timeout"] == ai.CUT_TIMEOUT_S and seen["timeout"] < ai.WINDOW_TIMEOUT_S


def test_the_probe_requires_an_audio_stream_and_survives_a_header_without_duration(monkeypatch):
    def with_json(payload):
        monkeypatch.setattr(ai, "_run", lambda cmd, *, timeout: MagicMock(stdout=json.dumps(payload).encode()))
    with_json({"format": {"duration": "N/A"}, "streams": [{"codec_type": "audio", "duration": "12.5"}]})
    assert ai.probe_duration_seconds(Path("a.mkv")) == 12.5
    with_json({"format": {"duration": "90.0"}, "streams": [{"codec_type": "video"}]})
    with pytest.raises(RuntimeError, match="no audio stream"):
        ai.probe_duration_seconds(Path("silent.mp4"))
    with_json({"format": {}, "streams": [{"codec_type": "audio"}]})
    with pytest.raises(RuntimeError, match="reports none"):
        ai.probe_duration_seconds(Path("cut.aac"))


def test_post_json_splits_connect_from_read_and_classifies_transient_status(monkeypatch):
    import httpx
    captured = {}

    class FakeClient:
        def __init__(self, timeout):
            captured["timeout"] = timeout
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def post(self, url, json=None):
            return MagicMock(status_code=captured.get("status", 200), text="x", json=lambda: {"ok": 1})
    monkeypatch.setattr(httpx, "Client", FakeClient)
    assert ai._post_json("http://n/v1", {}, 900.0) == {"ok": 1}
    t = captured["timeout"]
    assert t.connect == ai.CONNECT_TIMEOUT_S and t.read == 900.0
    captured["status"] = 503
    with pytest.raises(ai.TransientNodeError):
        ai._post_json("http://n/v1", {}, 900.0)
    captured["status"] = 400
    with pytest.raises(RuntimeError) as exc:
        ai._post_json("http://n/v1", {}, 900.0)
    assert not isinstance(exc.value, ai.TransientNodeError), "a 400 is not worth a retry"


def test_window_knobs_fall_back_instead_of_raising_or_zeroing(monkeypatch):
    monkeypatch.setenv("GHOST_AUDIO_WINDOW_S", "0")
    monkeypatch.setenv("GHOST_AUDIO_WINDOW_OVERLAP_S", "abc")
    monkeypatch.setenv("GHOST_AUDIO_MAX_TOKENS", "-5")
    try:
        importlib.reload(ai)
        assert ai.WINDOW_SECONDS == 720.0 and ai.WINDOW_OVERLAP_SECONDS == 15.0 and ai.WINDOW_MAX_TOKENS == 8192
        monkeypatch.setenv("GHOST_AUDIO_WINDOW_S", "10")
        monkeypatch.setenv("GHOST_AUDIO_WINDOW_OVERLAP_S", "50")
        importlib.reload(ai)
        assert ai.WINDOW_OVERLAP_SECONDS < ai.WINDOW_SECONDS, "an overlap ≥ the window meant 720 node calls per window"
    finally:
        for k in ("GHOST_AUDIO_WINDOW_S", "GHOST_AUDIO_WINDOW_OVERLAP_S", "GHOST_AUDIO_MAX_TOKENS"):
            monkeypatch.delenv(k, raising=False)
        importlib.reload(ai)


def test_audiobooks_voice_memos_and_matroska_audio_are_media():
    for ext in (".m4b", ".3gp", ".amr", ".oga", ".mka", ".aif"):
        assert ext in mem._AUDIO_INGEST_EXTS, ext
    assert ".ts" not in mem._AUDIO_INGEST_EXTS, "TypeScript files are not recordings"


# ─────────────────────────── knowledge_base: the tool ───────────────────────────

@pytest.mark.asyncio
async def test_a_youtube_url_takes_the_youtube_route_never_the_page_fetch(tmp_path, monkeypatch):
    """Before: the watch page's HTML shell was ingested as a document and
    the tool said SUCCESS."""
    async def boom(*a, **k):
        raise AssertionError("the generic web fetch must not see a YouTube URL")
    monkeypatch.setattr(mem, "helper_fetch_url_content", boom)
    seen = {}

    def fake_route(url, *, sandbox_dir, memory_system, tor_proxy, language=None, progress=None, **kw):
        seen.update(url=url, sandbox_dir=sandbox_dir, tor_proxy=tor_proxy, language=language)
        return yi.YoutubeIngestResult(True, "SUCCESS: routed")
    monkeypatch.setattr(yi, "ingest_youtube", fake_route)
    out = await mem.tool_gain_knowledge(f"https://youtu.be/{VID}", tmp_path, FakeMemory(),
                                        tor_proxy=TOR, language="el")
    assert out == "SUCCESS: routed"
    assert seen["tor_proxy"] == TOR and seen["language"] == "el" and seen["sandbox_dir"] == tmp_path


@pytest.mark.asyncio
async def test_a_schemeless_or_very_long_youtube_link_still_takes_the_route(tmp_path, monkeypatch):
    """Round 2: `youtu.be/<id>` without a scheme fell to the sandbox-file path
    ("File not found … pass the URL itself") and a 300-char watch URL with
    tracking params was refused as a too-long filename."""
    seen = []
    monkeypatch.setattr(yi, "ingest_youtube", lambda url, **kw: (seen.append(url), yi.YoutubeIngestResult(True, "SUCCESS: routed"))[1])
    for url in (f"youtu.be/{VID}", URL + "&" + "&".join(f"utm_{i}=x" * 3 for i in range(40))):
        out = await mem.tool_gain_knowledge(url, tmp_path, FakeMemory(), tor_proxy=TOR)
        assert out == "SUCCESS: routed", url[:40]
    assert seen[0] == f"youtu.be/{VID}"


@pytest.mark.asyncio
async def test_transcribe_dispatch_threads_tor_proxy_and_language(monkeypatch):
    seen = {}

    async def fake_gain(filename, sandbox_dir, memory_system, **kw):
        seen.update(kw)
        return "ok"
    monkeypatch.setattr(mem, "tool_gain_knowledge", fake_gain)
    await mem.tool_knowledge_base(action="transcribe", filename=URL, tor_proxy=TOR, language="en")
    assert seen == {"tor_proxy": TOR, "language": "en"}


@pytest.mark.asyncio
async def test_the_resolver_never_ingests_a_download_still_in_flight(tmp_path):
    """``yt_audio.m4a.part`` matched ``yt_audio.m4a`` by substring and partial
    AAC went down the plain-text branch as replacement-character noise."""
    (tmp_path / "talk.m4a.part").write_bytes(b"\xff\xf1" * 100)
    out = await mem.tool_gain_knowledge("talk.m4a", tmp_path, FakeMemory())
    assert out.startswith("Error:") and "not found" in out
    assert "YouTube link needs no download" in out


@pytest.mark.asyncio
async def test_transcript_action_pages_the_ordered_text_and_says_how_to_continue():
    m = FakeMemory(library=["talk.mp3"])
    m.text["talk.mp3"] = {"title": "A talk", "passages": [[0, 90, "one " * 100], [90, 180, "two " * 100],
                                                          [180, 270, "three " * 100]], "gaps": ["3:00–4:00 (x)"]}
    page = await mem.tool_knowledge_base(action="transcript", filename="talk.mp3", max_chars=500, memory_system=m)
    assert page.startswith("TRANSCRIPT of 'talk.mp3' — \"A talk\"")
    assert "[0:00–1:30] one one" in page and "Not transcribed: 3:00–4:00" in page
    assert "offset=500)" in page
    tail = await mem.tool_knowledge_base(action="transcript", filename="talk", offset=1300, memory_system=m)
    assert "[end of transcript]" in tail and "three three" in tail
    whole = await mem.tool_knowledge_base(action="transcript", filename="talk.mp3", max_chars=60000, memory_system=m)
    assert "[3:00–4:30] three" in whole and "[end of transcript]" in whole
    past = await mem.tool_knowledge_base(action="transcript", filename="talk.mp3", offset=10 ** 6, memory_system=m)
    assert "past the end" in past
    # the natural follow-up after "Skipped": the URL itself finds the video's document
    m.library.append(f"yt-{VID}.captions.en"); m.text[f"yt-{VID}.captions.en"] = {"passages": [[0, 5, "zoo"]]}
    by_url = await mem.tool_knowledge_base(action="transcript", filename=f"https://youtu.be/{VID}", memory_system=m)
    assert by_url.startswith(f"TRANSCRIPT of 'yt-{VID}.captions.en'")
    # a bare prefix that names several documents is refused, not guessed
    m.library += ["talk-2.mp3"]
    amb = await mem.tool_knowledge_base(action="transcript", filename="tal", memory_system=m)
    assert amb.startswith("Error:") and "several" in amb
    m2 = FakeMemory(library=["manual.pdf"])
    none = await mem.tool_knowledge_base(action="transcript", filename="manual.pdf", memory_system=m2)
    assert none.startswith("Error:") and "action='query'" in none
    missing = await mem.tool_knowledge_base(action="transcript", memory_system=m2)
    assert missing.startswith("SYSTEM ERROR") and "filename" in missing


def test_a_sandbox_recording_gets_structure_text_and_a_content_summary():
    m = FakeMemory()
    st = ai.AudioIngestStats(windows=2, seconds=190.0, total_seconds=200.0, chunks=3, skipped_windows=1,
                             truncated_windows=1)
    passages = [(0.0, 100.0, "opening words of the talk"), (100.0, 190.0, "closing words")]
    gaps = ["3:10–3:20 (node timeout)", "0:00–1:40 (tail cut at the token budget)"]
    mem._persist_audio_structure(m, "talk.mp3", st, passages, gaps, "2026-09-24T00:00:00Z")
    assert [e[1][:9] for e in m.outline["talk.mp3"]["entries"]] == ["0:00–1:40", "1:40–3:10"]
    assert m.text["talk.mp3"]["passages"][0][2] == "opening words of the talk"
    assert m.text["talk.mp3"]["gaps"] == gaps
    row = m.rows[0][0]
    assert "Opening words: opening words of the talk" in row and "gaps: 3:10–3:20" in row
    msg = mem._audio_success_message("talk.mp3", st, passages, gaps)
    assert msg.startswith("SUCCESS:") and "3:10 of 3:20 transcribed" in msg
    assert "NOT transcribed: 3:10–3:20 (node timeout)" in msg
    assert "1 window cut short at the token budget" in msg and "1 window failed" in msg
    assert "[0:00–1:40] opening words of the talk" in msg and "action='transcript'" in msg


# ─────────────────────────── composed skills ───────────────────────────

def _step(tool, **params):
    return cs.SkillStep(tool_name=tool, description="s", param_template=params)


def test_a_line_break_in_a_value_spliced_into_a_shell_command_is_refused():
    """A quoted heredoc blocks expansion but not its own TERMINATOR: a value
    with a line break followed by the delimiter line ends it, and the rest
    runs as shell. Tool arguments decode ``\\n`` to real newlines, so the
    resolver refuses control characters in any value it splices into
    ``execute.command``. A tab is not a line break; other params and other
    tools are not shell sinks; a WHOLE-value command is the caller's own."""
    reg = cs.ComposedSkillRegistry()
    hostile = "https://x/\n__DELIM__\ntouch /tmp/pwned"
    with pytest.raises(cs.ComposedArgError) as exc:
        reg._resolve_args(_step("execute", command="cat > f <<'__DELIM__'\n$url\n__DELIM__"), {"url": hostile})
    assert "line break" in str(exc.value) and "command" in str(exc.value)
    with pytest.raises(cs.ComposedArgError):
        reg._resolve_args(_step("execute", command="echo ${url}"), {"url": "a\rb"})
    assert reg._resolve_args(_step("execute", command="echo $url"), {"url": "a\tb"}) == {"command": "echo a\tb"}
    assert reg._resolve_args(_step("knowledge_base", filename="$url"), {"url": hostile}) == {"filename": hostile}
    assert reg._resolve_args(_step("execute", command="$cmd"), {"cmd": "a\nb"}) == {"command": "a\nb"}


@pytest.mark.asyncio
async def test_the_sequential_runner_records_the_refusal_and_never_calls_the_tool():
    reg = cs.ComposedSkillRegistry()
    skill = cs.ComposedSkill(name="m", trigger_description="t",
                             steps=[_step("execute", command="echo $url"), _step("knowledge_base", filename="x")])
    calls = []

    async def executor(tool, args):
        calls.append(tool)
        return "SUCCESS"
    out = await reg._execute_sequential(skill, executor, {"url": "a\nrm -rf /"})
    assert out["success"] is False and calls == []
    assert out["results"][0]["success"] is False and "line break" in out["results"][0]["error"]
    par = await reg._run_parallel_step(_step("execute", command="echo $u"), executor, {"u": "x\ny"})
    assert par["success"] is False and calls == []


def test_the_youtube_macro_is_one_knowledge_base_step_with_no_shell():
    d = ytd.build_youtube_transcribe_definition()
    assert d["name"] == "youtube_transcribe" and d["mode"] == "sequential"
    assert [s["tool"] for s in d["steps"]] == ["knowledge_base"]
    assert d["steps"][0]["params"] == {"action": "transcribe", "filename": "$url"}
    assert "Tor" in d["description"] and "transcript" in d["description"]
    assert not (Path(ytd.__file__).with_name("yt_tor_download.sh")).exists(), "the smuggled script is gone"
    assert "youtube_transcribe" in ytd.CODE_OWNED_MACROS


def test_a_stored_copy_of_a_code_owned_macro_is_reconciled_at_load(tmp_path):
    """The live store held the two-step shell version; changing the code used
    to need the agent stopped and a sync script. Now the registry overwrites
    the stored steps at load (usage counters kept) and never CREATES one."""
    store = {"youtube_transcribe": {
        "name": "youtube_transcribe", "trigger_description": "old words",
        "steps": [{"tool_name": "execute", "description": "dl", "param_template": {"command": "echo $url"}},
                  {"tool_name": "knowledge_base", "description": "t", "param_template": {"action": "transcribe", "filename": "yt_audio.m4a"}}],
        "branches": {}, "execution_mode": "sequential", "status": "active",
        "usage_count": 3, "success_count": 0, "last_used": 1.0, "created_at": 1.0,
    }, "other_macro": {
        "name": "other_macro", "trigger_description": "mine",
        "steps": [{"tool_name": "execute", "description": "x", "param_template": {"command": "ls"}}],
        "status": "active",
    }}
    (tmp_path / "composed_skills.json").write_text(json.dumps(store))
    reg = cs.ComposedSkillRegistry(storage_dir=tmp_path)
    sk = reg.skills["youtube_transcribe"]
    assert [(s.tool_name, s.param_template) for s in sk.steps] == [("knowledge_base", {"action": "transcribe", "filename": "$url"})]
    assert sk.usage_count == 3 and sk.success_count == 0, "history survives the reconcile"
    assert sk.trigger_description == ytd.build_youtube_transcribe_definition()["description"]
    on_disk = json.loads((tmp_path / "composed_skills.json").read_text())
    assert len(on_disk["youtube_transcribe"]["steps"]) == 1, "the store was rewritten, not just the in-memory copy"
    assert on_disk["other_macro"]["steps"][0]["param_template"] == {"command": "ls"}, "operator macros untouched"
    # a store WITHOUT the macro stays without it
    (tmp_path / "composed_skills.json").write_text(json.dumps({"other_macro": store["other_macro"]}))
    reg2 = cs.ComposedSkillRegistry(storage_dir=tmp_path)
    assert "youtube_transcribe" not in reg2.skills


# ─────────────────────────── the store sidecar ───────────────────────────

def test_forgetting_a_document_drops_its_ordered_text_too():
    """The transcript sidecar is a claim about a document; `delete_document_by_name`
    already drops the outline in the same critical section — the text must go
    with it. PARSED pin (AST), not a source grep: the call must be inside that
    method's body."""
    import ast, inspect
    from ghost_agent.memory import vector as vec
    tree = ast.parse(inspect.getsource(vec))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "delete_document_by_name")
    called = {n.func.attr for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert {"drop_document_outline", "drop_document_text", "_update_library_index"} <= called


def test_the_sidecar_round_trips_and_is_dropped(tmp_path):
    from ghost_agent.memory import vector as vec
    import threading
    obj = vec.VectorMemory.__new__(vec.VectorMemory)
    obj.outlines_file = tmp_path / "document_outlines.json"
    obj._lock = threading.RLock()
    obj._get_lock = lambda: obj._lock
    obj.set_document_text("talk.mp3", {"passages": [[0, 1, "hi"]]})
    assert obj.get_document_text("talk.mp3")["passages"] == [[0, 1, "hi"]]
    assert obj.get_document_text("talk.mp3")["filename"] == "talk.mp3"
    assert obj.get_document_text("other") == {}
    obj.drop_document_text("talk.mp3")
    assert obj.get_document_text("talk.mp3") == {}


# ─────────────────────────── what the model is told ───────────────────────────

def _kb_schema():
    from ghost_agent.tools import registry as reg
    return next(t["function"] for t in reg.TOOL_DEFINITIONS if t["function"]["name"] == "knowledge_base")


def test_the_description_names_the_youtube_route_and_the_transcript_action():
    fn = _kb_schema()
    desc = fn["description"]
    assert "YOUTUBE LINK" in desc and "over Tor" in desc
    assert "LOCALLY" not in desc, "transcription runs on the private audio node, not on this host"
    assert "transcript (" in desc
    props = fn["parameters"]["properties"]
    assert "transcript" in props["action"]["enum"] and props["action"]["enum"][0] == "transcribe"
    for p in ("language", "offset", "max_chars"):
        assert p in props, p
    assert set(props["action"]["enum"]) == set(mem._KB_ACTIONS), "the enum and the dispatcher disagree"


def test_the_prompt_rule_routes_youtube_to_the_tool_not_to_a_download():
    from ghost_agent.core import prompts
    low = prompts.PLANNING_SYSTEM_PROMPT.lower()
    assert "never plan yt-dlp" in low
    assert "action='transcript'" in low
    assert "transcription and ingestion are one step" in low
