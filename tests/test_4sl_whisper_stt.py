"""§4SL (2026-10-31): speech-to-text moves from Gemma 4 E4B to whisper.cpp's
``whisper-server`` on nova, behind ``GHOST_STT_URL``.

Measured on nova with interface/voice.py's exact Gemma request (FLEURS read
speech, 16-bit WAV, 100 Greek + 50 English clips): Greek WER 60.8 % on Gemma
(30/100 clips transliterated into Latin letters) vs 12.2 % on Whisper
large-v3 q5_0 + tuned VAD; English 6.6 % vs 5.2 %; median 9.1 s vs 3.4 s.

Pinned here — each test names the world where it fails:
* with the URL set, push-to-talk and audio ingest reach ``/inference`` with
  whisper-server's real multipart field names, and NEVER the Gemma chat path
  (no fallback — a fallback would hand the agent 60 %-wrong Greek);
* a 200 that carries ``{"error": …}`` is a failure, never silence;
* whisper's own tags (``[BLANK_AUDIO]``) are silence, real words are kept;
* segment line breaks become spaces (one message, like the Gemma path);
* ingest windows are not fitted to a llama slot (whisper has no /props);
* with the URL unset, the Gemma path is unchanged.
"""

import io
import struct
import sys
import wave
from email.parser import BytesParser
from email.policy import default as _email_policy
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

_ROOT = Path(__file__).resolve().parent.parent
for p in (str(_ROOT), str(_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from interface import voice  # noqa: E402
from ghost_agent.memory import audio_ingest as ai  # noqa: E402

URL = "http://100.83.184.117:8090"


def _wav(seconds=1.0, rate=16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(struct.pack("<h", 0) * int(rate * seconds))
    return buf.getvalue()


def _parse_multipart(request: httpx.Request) -> dict:
    """Decode the REAL encoded multipart body httpx put on the wire."""
    ctype = request.headers["content-type"]
    msg = BytesParser(policy=_email_policy).parsebytes(
        b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + request.content)
    out = {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        out[name] = part.get_payload(decode=True)
    return out


class _Recorder:
    """A transport that records every request and answers like whisper-server."""

    def __init__(self, body=None, status=200):
        self.requests, self.body, self.status = [], body, status

    def __call__(self, request: httpx.Request):
        self.requests.append(request)
        if self.status != 200:
            return httpx.Response(self.status, text="server boom")
        return httpx.Response(200, json=self.body)


# ── push-to-talk (interface/voice.py) ────────────────────────────────────

async def _voice_call(monkeypatch, rec, *, language="auto", seconds=1.0):
    monkeypatch.setattr(voice, "STT_URL", URL)
    monkeypatch.setattr(voice, "STT_LANGUAGE", language)
    async with httpx.AsyncClient(transport=httpx.MockTransport(rec)) as client:
        with patch.object(voice, "transcode_to_wav16k", AsyncMock(return_value=_wav(seconds))):
            return await voice.transcribe(b"raw-upload", client=client)


@pytest.mark.asyncio
async def test_voice_posts_real_multipart_to_inference_and_never_gemma(monkeypatch):
    rec = _Recorder({"text": "Καλημέρα, τι καιρό θα κάνει αύριο;"})
    got = await _voice_call(monkeypatch, rec, language="el")
    assert got == "Καλημέρα, τι καιρό θα κάνει αύριο;"
    assert [str(r.url) for r in rec.requests] == [f"{URL}/inference"], \
        "exactly one call, to whisper — the Gemma chat path must not be consulted"
    parts = _parse_multipart(rec.requests[0])
    assert parts["file"][:4] == b"RIFF", "the transcoded WAV must be the 'file' part"
    assert parts["language"] == b"el"
    assert parts["response_format"] == b"json"
    assert parts["temperature"] == b"0.0"


@pytest.mark.asyncio
async def test_voice_joins_segment_lines_into_one_message(monkeypatch):
    rec = _Recorder({"text": " Η ζεστή σοκολάτα.\n Οι φρουτοχυμοί είναι ακριβοί.\n"})
    got = await _voice_call(monkeypatch, rec)
    assert got == "Η ζεστή σοκολάτα. Οι φρουτοχυμοί είναι ακριβοί."


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["", "   ", "[BLANK_AUDIO]", " [BLANK_AUDIO]\n", "(music)", "[Μουσική]",
                                  " ♪", " .\n", "[BLANK_AUDIO] ♪ (music)"])
async def test_voice_whisper_tags_are_silence(monkeypatch, text):
    assert await _voice_call(monkeypatch, _Recorder({"text": text})) == ""


@pytest.mark.asyncio
async def test_voice_words_around_a_tag_are_kept(monkeypatch):
    got = await _voice_call(monkeypatch, _Recorder({"text": "[BLANK_AUDIO] turn the lights off"}))
    assert got == "[BLANK_AUDIO] turn the lights off"


@pytest.mark.asyncio
async def test_voice_error_body_in_a_200_is_a_failure_not_silence(monkeypatch):
    with pytest.raises(voice.VoiceError) as exc:
        await _voice_call(monkeypatch, _Recorder({"error": "failed to read WAV file"}))
    assert exc.value.status == 502
    assert "failed to read WAV file" in str(exc.value)


@pytest.mark.asyncio
async def test_voice_http_error_is_a_failure(monkeypatch):
    with pytest.raises(voice.VoiceError) as exc:
        await _voice_call(monkeypatch, _Recorder(status=500))
    assert exc.value.status == 502 and "HTTP 500" in str(exc.value)


@pytest.mark.asyncio
async def test_voice_non_object_json_is_a_failure(monkeypatch):
    with pytest.raises(voice.VoiceError):
        await _voice_call(monkeypatch, _Recorder(["not", "an", "object"]))


@pytest.mark.asyncio
async def test_voice_overlong_message_does_not_blame_a_llama_slot(monkeypatch):
    monkeypatch.setattr(voice, "STT_MAX_SECONDS", 1.0)
    rec = _Recorder({"text": "x"})
    with pytest.raises(voice.VoiceError) as exc:
        await _voice_call(monkeypatch, rec, seconds=2.0)
    assert exc.value.status == 413 and "slot" not in str(exc.value)
    assert rec.requests == []


@pytest.mark.asyncio
async def test_voice_without_the_url_still_uses_gemma(monkeypatch):
    monkeypatch.setattr(voice, "STT_URL", "")
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, json={"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with patch.object(voice, "transcode_to_wav16k", AsyncMock(return_value=_wav())):
            got = await voice.transcribe(b"raw", client=client)
    assert got == "hello"
    assert seen == [f"{voice.AUDIO_NODE_URL}/v1/chat/completions"]


# ── long-form ingest (memory/audio_ingest.py) ────────────────────────────

def _install_transport(monkeypatch, rec):
    real = httpx.Client

    def _client(*a, **kw):
        kw["transport"] = httpx.MockTransport(rec)
        return real(*a, **kw)
    monkeypatch.setattr(httpx, "Client", _client)


def test_ingest_window_posts_real_multipart_and_never_gemma(monkeypatch):
    monkeypatch.setattr(ai, "STT_URL", URL)
    monkeypatch.setattr(ai, "STT_LANGUAGE", "auto")
    rec = _Recorder({"text": " first segment\n second segment "})
    _install_transport(monkeypatch, rec)
    text, truncated = ai.transcribe_window_full(_wav())
    assert (text, truncated) == ("first segment second segment", False)
    assert [str(r.url) for r in rec.requests] == [f"{URL}/inference"]
    parts = _parse_multipart(rec.requests[0])
    assert parts["file"][:4] == b"RIFF" and parts["language"] == b"auto"


@pytest.mark.parametrize("status", [429, 500, 503])
def test_ingest_server_overload_is_transient(monkeypatch, status):
    monkeypatch.setattr(ai, "STT_URL", URL)
    _install_transport(monkeypatch, _Recorder(status=status))
    with pytest.raises(ai.TransientNodeError):
        ai.transcribe_window_full(_wav())


def test_ingest_client_error_is_permanent(monkeypatch):
    monkeypatch.setattr(ai, "STT_URL", URL)
    _install_transport(monkeypatch, _Recorder(status=400))
    with pytest.raises(RuntimeError) as exc:
        ai.transcribe_window_full(_wav())
    assert not isinstance(exc.value, ai.TransientNodeError)


def test_ingest_error_body_becomes_a_gap_not_silence(monkeypatch):
    monkeypatch.setattr(ai, "STT_URL", URL)
    calls = []

    def post(url, payload, timeout):
        calls.append(url)
        return {"error": "failed to process audio"}
    stats = ai.AudioIngestStats()
    with patch.object(ai, "probe_duration_seconds", return_value=120), \
         patch.object(ai, "extract_window_wav", return_value=b"WAVE"):
        chunks = list(ai.iter_audio_chunks(Path("t.mp3"), "t.mp3", stats=stats, post_fn=post,
                                           window_seconds=120, sleep=lambda s: None))
    assert chunks == []
    assert stats.gaps and "failed to process audio" in stats.gaps[0]
    assert calls and all(u == f"{URL}/inference" for u in calls)


@pytest.mark.parametrize("text", ["", "[BLANK_AUDIO]", " (music) ", " ♪ ", " .", "(Μουσική) ♪"])
def test_ingest_whisper_tags_are_silence(monkeypatch, text):
    monkeypatch.setattr(ai, "STT_URL", URL)
    assert ai.transcribe_window_full(b"WAVE", post_fn=lambda u, p, t: {"text": text}) == ("", False)


def test_ingest_windows_are_not_fitted_to_a_llama_slot(monkeypatch):
    """whisper-server has no /props; asking for it would size windows from a
    different server (or log a failure per recording)."""
    monkeypatch.setattr(ai, "STT_URL", URL)
    asked = []
    monkeypatch.setattr(ai, "node_slot_ctx", lambda: asked.append(1) or 4096)
    monkeypatch.setattr(ai, "_post_inference", lambda u, p, t: {"text": "words " * 20})
    stats = ai.AudioIngestStats()
    with patch.object(ai, "probe_duration_seconds", return_value=900), \
         patch.object(ai, "extract_window_wav", return_value=b"WAVE"):
        list(ai.iter_audio_chunks(Path("t.mp3"), "t.mp3", stats=stats, window_seconds=480,
                                  window_overlap=15))
    assert asked == []
    assert stats.windows == 2, "900 s at 480 s windows (step 465) = 2 windows, unshrunk"


def test_ingest_without_the_url_still_fits_the_gemma_slot(monkeypatch):
    monkeypatch.setattr(ai, "STT_URL", "")
    asked = []
    monkeypatch.setattr(ai, "node_slot_ctx", lambda: asked.append(1) or None)
    monkeypatch.setattr(ai, "_post_json", lambda u, p, t: {
        "choices": [{"message": {"content": "words"}, "finish_reason": "stop"}]})
    with patch.object(ai, "probe_duration_seconds", return_value=60), \
         patch.object(ai, "extract_window_wav", return_value=b"WAVE"):
        list(ai.iter_audio_chunks(Path("t.mp3"), "t.mp3", stats=ai.AudioIngestStats()))
    assert asked == [1]


def test_both_processes_read_the_same_variables(monkeypatch):
    """The interface (voice.py) and the agent (audio_ingest.py) are separate
    processes started by different launchers; ONE pair of variables must steer
    both, or the mic and ingest silently disagree about the backend."""
    import importlib
    monkeypatch.setenv("GHOST_STT_URL", URL + "/")
    monkeypatch.setenv("GHOST_STT_LANGUAGE", "el")
    try:
        for mod in (voice, ai):
            importlib.reload(mod)
            assert mod.STT_URL == URL, mod.__name__     # trailing slash dropped
            assert mod.STT_LANGUAGE == "el", mod.__name__
        monkeypatch.setenv("GHOST_STT_LANGUAGE", "  ")
        for mod in (voice, ai):
            importlib.reload(mod)
            assert mod.STT_LANGUAGE == "auto", mod.__name__
    finally:
        monkeypatch.delenv("GHOST_STT_URL", raising=False)
        monkeypatch.delenv("GHOST_STT_LANGUAGE", raising=False)
        importlib.reload(voice)
        importlib.reload(ai)


@pytest.mark.parametrize("text,kept", [
    ("(Music) Thanks everyone for coming.", "(Music) Thanks everyone for coming."),
    ("♪ Thanks", "♪ Thanks"),
    ("ok", "ok"),
])
def test_ingest_speech_after_a_tag_is_kept_whole(monkeypatch, text, kept):
    """The Gemma sentinel rule dropped a tag followed by < 80 chars; whisper
    writes real lead tags, so a short closing line must survive — and the mic
    path keeps the same text (one rule for both)."""
    monkeypatch.setattr(ai, "STT_URL", URL)
    assert ai.transcribe_window_full(b"WAVE", post_fn=lambda u, p, t: {"text": text}) == (kept, False)


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["(Music) Thanks everyone for coming.", "♪ Thanks", "ok"])
async def test_voice_and_ingest_agree_on_speech(monkeypatch, text):
    assert await _voice_call(monkeypatch, _Recorder({"text": text})) == text


def test_whisper_windows_are_short_and_gemma_windows_unchanged(monkeypatch):
    """whisper-server serves one request at a time and push-to-talk shares it:
    an 8-minute window held it ~1 min, past the uConsole's 60 s STT wait."""
    import importlib
    try:
        monkeypatch.setenv("GHOST_STT_URL", URL)
        importlib.reload(ai)
        assert (ai.WINDOW_SECONDS, ai.WINDOW_OVERLAP_SECONDS) == (120.0, 5.0)
        assert ai.WINDOW_TIMEOUT_S == 180.0, "a hung whisper server must not cost 900 s a window"
        monkeypatch.setenv("GHOST_STT_WINDOW_S", "90")
        importlib.reload(ai)
        assert ai.WINDOW_SECONDS == 90.0
        monkeypatch.delenv("GHOST_STT_URL")
        monkeypatch.delenv("GHOST_STT_WINDOW_S")
        importlib.reload(ai)
        assert (ai.WINDOW_SECONDS, ai.WINDOW_OVERLAP_SECONDS) == (480.0, 15.0)
        assert ai.WINDOW_TIMEOUT_S == 900.0
    finally:
        monkeypatch.delenv("GHOST_STT_URL", raising=False)
        monkeypatch.delenv("GHOST_STT_WINDOW_S", raising=False)
        importlib.reload(ai)


def test_failure_messages_name_the_whisper_server(monkeypatch):
    monkeypatch.setattr(ai, "STT_URL", URL)

    def post(url, payload, timeout):
        raise ai.TransientNodeError("whisper server unreachable: ConnectError")
    stats = ai.AudioIngestStats()
    with patch.object(ai, "probe_duration_seconds", return_value=600), \
         patch.object(ai, "extract_window_wav", return_value=b"WAVE"):
        list(ai.iter_audio_chunks(Path("t.mp3"), "t.mp3", stats=stats, post_fn=post,
                                  window_seconds=60, window_overlap=5, sleep=lambda s: None))
    assert "whisper server is not answering" in stats.aborted
    assert "audio node" not in stats.aborted


def test_ingest_unreachable_server_is_transient(monkeypatch):
    """A connect failure gets the window's one retry, not an immediate gap."""
    monkeypatch.setattr(ai, "STT_URL", URL)

    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)
    _install_transport(monkeypatch, refuse)
    with pytest.raises(ai.TransientNodeError) as exc:
        ai.transcribe_window_full(_wav())
    assert "whisper server unreachable" in str(exc.value)


def test_ingest_non_json_body_is_transient(monkeypatch):
    monkeypatch.setattr(ai, "STT_URL", URL)
    _install_transport(monkeypatch, lambda request: httpx.Response(200, text="<html>proxy page</html>"))
    with pytest.raises(ai.TransientNodeError):
        ai.transcribe_window_full(_wav())


def test_ingest_non_object_json_is_a_failure_not_silence(monkeypatch):
    monkeypatch.setattr(ai, "STT_URL", URL)
    with pytest.raises(RuntimeError):
        ai.transcribe_window_full(b"WAVE", post_fn=lambda u, p, t: ["text", "x"])


def test_ingest_sends_whisper_server_field_names(monkeypatch):
    monkeypatch.setattr(ai, "STT_URL", URL)
    monkeypatch.setattr(ai, "STT_LANGUAGE", "el")
    rec = _Recorder({"text": "λόγια"})
    _install_transport(monkeypatch, rec)
    ai.transcribe_window_full(_wav())
    parts = _parse_multipart(rec.requests[0])
    assert set(parts) == {"file", "response_format", "temperature", "language"}
    assert (parts["language"], parts["response_format"], parts["temperature"]) == (b"el", b"json", b"0.0")
