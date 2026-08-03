"""Unit tests for interface/voice.py — the local voice loop.

Context: the interface's push-to-talk UI shipped in 2026-07 but both of its
endpoints proxied to a Raspberry-Pi voice server that no longer exists, so the
mic button was a no-op against a dead host. STT now runs on nova's Gemma 4
audio node and TTS on the macOS synthesiser.

These tests pin the two traps that were MEASURED against the live node on
2026-08-02, because both fail silently if they regress:

1. The empty-response trap — Gemma 4's thinking blocks are stripped by its
   chat template, so too small a max_tokens returns EMPTY content with
   finish_reason="length". That must raise, never return "" (an empty
   transcript would be auto-sent to the agent as an empty prompt).
2. The dialect trap — audio must ride an `input_audio` part; the `audio_url`
   data-URI shape that vision.py uses for images is rejected by the node with
   400 "unsupported content[].type".
"""

import io
import re
import struct
import sys
import wave
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from interface import voice  # noqa: E402


def _make_wav(seconds: float, rate: int = 16000) -> bytes:
    """A real, parseable mono PCM WAV of a given duration."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack("<h", 0) * int(rate * seconds))
    return buf.getvalue()


def _fake_response(content, finish_reason="stop", status=200):
    resp = MagicMock()
    resp.status_code = status
    resp.text = "" if status == 200 else "upstream boom"
    resp.json = MagicMock(return_value={
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}]
    })
    return resp


def _client_returning(resp):
    client = MagicMock()
    client.post = AsyncMock(return_value=resp)
    return client


# ── duration parsing ─────────────────────────────────────────────────────

def test_wav_duration_is_parsed_from_header():
    assert voice.wav_duration_seconds(_make_wav(2.0)) == pytest.approx(2.0, abs=0.01)


def test_unreadable_wav_is_a_client_error():
    with pytest.raises(voice.VoiceError) as exc:
        voice.wav_duration_seconds(b"not a wav at all")
    assert exc.value.status == 400


# ── the dialect trap ─────────────────────────────────────────────────────

def test_payload_uses_input_audio_not_audio_url():
    """Regression guard: the node rejects the vision-style audio_url shape."""
    payload = voice._audio_message_payload(_make_wav(0.5))
    parts = payload["messages"][0]["content"]
    kinds = [p["type"] for p in parts]
    assert "input_audio" in kinds, "audio must ride an input_audio part"
    assert "audio_url" not in kinds, "audio_url is rejected by the node (HTTP 400)"
    audio_part = next(p for p in parts if p["type"] == "input_audio")
    assert audio_part["input_audio"]["format"] == "wav"
    assert audio_part["input_audio"]["data"], "base64 payload must be populated"


def test_payload_max_tokens_clears_the_thinking_budget():
    """256 was measured returning empty content; the default must exceed it."""
    assert voice._audio_message_payload(_make_wav(0.5))["max_tokens"] >= 512


# ── the empty-response (thinking-token) trap ─────────────────────────────

@pytest.mark.asyncio
async def test_thinking_token_exhaustion_raises_not_returns_empty():
    """Empty content + finish_reason=length is the silent-failure shape."""
    with patch.object(voice, "transcode_to_wav16k", AsyncMock(return_value=_make_wav(1.0))):
        with pytest.raises(voice.VoiceError) as exc:
            await voice.transcribe(b"audio", client=_client_returning(
                _fake_response("", finish_reason="length")))
    assert "thinking" in str(exc.value).lower()
    assert "GHOST_STT_MAX_TOKENS" in str(exc.value)


@pytest.mark.asyncio
async def test_genuine_silence_returns_empty_string():
    """Empty content that STOPPED normally is real silence, not a failure."""
    with patch.object(voice, "transcode_to_wav16k", AsyncMock(return_value=_make_wav(1.0))):
        got = await voice.transcribe(b"audio", client=_client_returning(
            _fake_response("", finish_reason="stop")))
    assert got == ""


@pytest.mark.asyncio
async def test_no_speech_sentinel_maps_to_empty_string():
    with patch.object(voice, "transcode_to_wav16k", AsyncMock(return_value=_make_wav(1.0))):
        got = await voice.transcribe(b"audio", client=_client_returning(
            _fake_response("(no speech)")))
    assert got == ""


@pytest.mark.asyncio
async def test_transcript_is_returned_stripped():
    with patch.object(voice, "transcode_to_wav16k", AsyncMock(return_value=_make_wav(1.0))):
        got = await voice.transcribe(b"audio", client=_client_returning(
            _fake_response("  hello there  ")))
    assert got == "hello there"


# ── bounds and failure paths ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_overlong_audio_is_rejected_before_the_node_call():
    client = _client_returning(_fake_response("should never be reached"))
    long_wav = _make_wav(voice.STT_MAX_SECONDS + 30)
    with patch.object(voice, "transcode_to_wav16k", AsyncMock(return_value=long_wav)):
        with pytest.raises(voice.VoiceError) as exc:
            await voice.transcribe(b"audio", client=client)
    assert exc.value.status == 413
    client.post.assert_not_awaited()


@pytest.mark.asyncio
async def test_node_http_error_becomes_voice_error():
    with patch.object(voice, "transcode_to_wav16k", AsyncMock(return_value=_make_wav(1.0))):
        with pytest.raises(voice.VoiceError) as exc:
            await voice.transcribe(b"audio", client=_client_returning(
                _fake_response("", status=503)))
    assert exc.value.status == 502
    assert "503" in str(exc.value)


@pytest.mark.asyncio
async def test_empty_upload_is_a_client_error():
    with pytest.raises(voice.VoiceError) as exc:
        await voice.transcode_to_wav16k(b"")
    assert exc.value.status == 400


# ── TTS ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_synthesize_rejects_empty_text():
    with pytest.raises(voice.VoiceError) as exc:
        await voice.synthesize("   ")
    assert exc.value.status == 400


@pytest.mark.asyncio
async def test_synthesize_passes_text_by_file_never_argv():
    """Agent replies contain quotes/backticks/newlines. Text goes through a
    FILE so none of it can ever reach a shell or be mangled in argv."""
    captured = {}

    async def _fake_run(cmd, *, timeout, stdin=None):
        captured["cmd"] = cmd
        out = Path(cmd[cmd.index("-o") + 1])
        out.write_bytes(b"RIFF-fake-wav")
        return b"", b""

    danger = 'say "; rm -rf /" `whoami` $(id)'
    with patch.object(voice, "_run_binary", _fake_run):
        audio = await voice.synthesize(danger)

    assert audio == b"RIFF-fake-wav"
    assert "-f" in captured["cmd"], "text must be supplied via -f <file>"
    assert danger not in captured["cmd"], "raw text must never appear in argv"


@pytest.mark.asyncio
async def test_synthesize_truncates_overlong_text():
    captured = {}

    async def _fake_run(cmd, *, timeout, stdin=None):
        captured["text"] = Path(cmd[cmd.index("-f") + 1]).read_text()
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"RIFF")
        return b"", b""

    with patch.object(voice, "_run_binary", _fake_run):
        await voice.synthesize("word " * 5000)

    assert len(captured["text"]) <= voice.TTS_MAX_CHARS + 2


@pytest.mark.asyncio
async def test_configured_voice_is_passed_to_say_as_one_argv_element():
    """The premium voice name contains spaces and parentheses ("Ava (Premium)").

    It is passed as a SINGLE list element after -v, so no quoting/splitting
    can mangle it. A name split across argv would make `say` see "Ava" and
    "(Premium)" as separate arguments, exit non-zero, and silently drop the
    turn onto the default-voice fallback — the failure this guards.
    """
    captured = {}

    async def _fake_run(cmd, *, timeout, stdin=None):
        captured["cmd"] = cmd
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"RIFF")
        return b"", b""

    with patch.object(voice, "_run_binary", _fake_run):
        await voice.synthesize("hello")

    cmd = captured["cmd"]
    assert cmd[cmd.index("-v") + 1] == voice.TTS_VOICE


@pytest.mark.asyncio
async def test_unavailable_voice_falls_back_without_the_v_flag():
    """A voice absent from this macOS build must degrade, not kill speech.

    The first `say` invocation (with -v) fails; the retry must carry NO -v at
    all, so it lands on the system default rather than re-failing on the same
    missing name.
    """
    calls = []

    async def _fake_run(cmd, *, timeout, stdin=None):
        calls.append(cmd)
        if "-v" in cmd:
            raise voice.VoiceError("'say' failed: voice not found", 502)
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"RIFF-default")
        return b"", b""

    with patch.object(voice, "_run_binary", _fake_run):
        audio = await voice.synthesize("hello")

    assert audio == b"RIFF-default"
    assert len(calls) == 2, "must retry exactly once, on the default voice"
    assert "-v" not in calls[1]


def test_premium_voice_suffix_is_not_dropped():
    """Bare "Ava" is a DIFFERENT, compact voice that co-exists with the premium
    one, so a stripped suffix downgrades quality silently instead of erroring.
    Pin the exact string rather than a substring match."""
    assert voice.TTS_VOICE == "Ava (Premium)"


# ── the dead-host regression guard ───────────────────────────────────────

def test_no_dead_voice_host_references_remain():
    """`raspberrypi.local` / `disorder` are gone; no EXECUTABLE line may reach
    for them again.

    Comments are stripped before the check on purpose: both files describe the
    dead hosts in prose so the next reader knows why the proxies vanished, and
    that documentation must not trip a guard aimed at live code.
    """
    src = (_ROOT / "interface" / "server.py").read_text()
    code = "\n".join(ln for ln in src.splitlines()
                     if not ln.lstrip().startswith("#"))
    for dead in ("PI_VOICE_URL", "raspberrypi.local", "disorder:8000"):
        assert dead not in code, f"server.py still references {dead} in live code"

    # voice.py describes the dead hosts in its module DOCSTRING (a string, not
    # a comment), so assert on resolved behaviour rather than on text: the
    # configured backend must be a real node, and no PI_VOICE_URL setting may
    # have survived as a module attribute.
    assert not hasattr(voice, "PI_VOICE_URL")
    for dead in ("raspberrypi", "disorder"):
        assert dead not in voice.AUDIO_NODE_URL


def test_docs_do_not_advertise_the_dead_voice_knob():
    """The code stopped reading PI_VOICE_URL on 2026-08-02, but capabilities.html
    and installation.html kept listing it as a working setting for another day —
    a reader configuring voice from the docs would have set a variable nothing
    consumes and concluded the feature was broken. Docs are part of the
    interface; a retired knob must not survive in them.

    Only the NAME COLUMN of a settings table is checked — the first <td> of a
    row is what actually advertises "set this". A description or a line of
    prose that names the variable to explain that it was retired must stay
    legal: that is the same why-it-vanished documentation the comment-stripping
    in test_no_dead_voice_host_references_remain deliberately protects, and a
    reader who still has the old variable exported needs to find it.
    """
    first_cell = re.compile(r"<tr>\s*<td>\s*<code>(.*?)</code>")
    for page in ("capabilities.html", "installation.html", "index.html"):
        text = (_ROOT / "docs" / page).read_text()
        for name in first_cell.findall(text):
            assert "PI_VOICE_URL" not in name, (
                f"docs/{page} still lists PI_VOICE_URL as a settable knob"
            )


def test_retired_gpu_voice_external_and_its_orphan_pin_stay_gone():
    """interface/externals/tts_stt/ (faster-whisper + Kokoro on the Orin) was
    deleted 2026-08-03. It had been unreferenced since 2026-08-02 and could not
    start here, yet it kept `soundfile` on the dependency list — a package
    installed on every deploy for a service that could not run.

    Both halves are asserted together on purpose: restoring the external
    without its pin gives an ImportError, and restoring the pin without the
    external re-creates the orphan. If a GPU voice node is ever reinstated,
    this test is the place that should fail and be updated deliberately.
    """
    assert not (_ROOT / "interface" / "externals" / "tts_stt").exists(), (
        "the retired GPU voice external is back — see docs/interfaces/"
        "voice_server.html#retired-external before restoring it"
    )

    reqs = (_ROOT / "requirements.txt").read_text()
    declared = [ln.split("#")[0].strip() for ln in reqs.splitlines()]
    declared = [ln for ln in declared if ln]
    assert not any(ln.lower().startswith("soundfile") for ln in declared), (
        "soundfile is pinned again with no importer in the tree"
    )


def test_mic_button_is_restored_in_the_input_area():
    """The mic was deleted 2026-08-01 as "unused" — unused because its backend
    was dead. It is restored in the INPUT AREA, not the header: the header was
    trimmed to six controls for a single-row mobile layout, and putting the mic
    back there would undo that."""
    html = (_ROOT / "interface" / "static" / "index.html").read_text()
    assert 'id="mic-btn"' in html, "push-to-talk button missing"
    footer_idx = html.find('id="input-area-wrapper"')
    assert footer_idx != -1
    assert html.find('id="mic-btn"') > footer_idx, (
        "mic-btn must live in the input area, not the header (mobile layout)")


def test_voice_in_gives_voice_out_and_typing_turns_it_off():
    """Holding the mic enables spoken replies (and unlocks autoplay, which
    browsers only allow from a user gesture); a typed message disables them."""
    js = (_ROOT / "interface" / "static" / "app.js").read_text()
    start = js.find("const startRecording")
    assert start != -1, "push-to-talk engine missing"
    body = js[start:start + 1600]
    assert "isTTSActive = true" in body, "mic press must enable spoken replies"
    assert "createBuffer" in body, "mic press must unlock autoplay (iOS needs it)"
    assert "function sendTypedMessage" in js
    typed = js[js.find("function sendTypedMessage"):][:200]
    assert "isTTSActive = false" in typed, "typing must turn speech back off"


def test_binaries_resolve_without_a_usable_path(tmp_path, monkeypatch):
    """The interface runs under launchd, which hands a service a MINIMAL PATH
    (/usr/bin:/bin:/usr/sbin:/sbin) that excludes Homebrew. `say` is in
    /usr/bin so it worked, but ffmpeg lives in /opt/homebrew/bin — a bare
    shutil.which() returned None and EVERY STT request 503'd under the daemon
    while working perfectly from a shell.
    """
    fake = tmp_path / "ffmpeg"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o755)

    monkeypatch.setenv("PATH", "/nonexistent")
    monkeypatch.delenv("GHOST_FFMPEG_BIN", raising=False)
    monkeypatch.setattr(voice, "_BIN_PREFIXES", (str(tmp_path),))

    assert voice.resolve_binary("ffmpeg") == str(fake), (
        "must fall back to known install prefixes when PATH is unusable")


def test_explicit_binary_override_wins(tmp_path, monkeypatch):
    fake = tmp_path / "custom-ffmpeg"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o755)
    monkeypatch.setenv("GHOST_FFMPEG_BIN", str(fake))
    assert voice.resolve_binary("ffmpeg") == str(fake)


@pytest.mark.asyncio
async def test_missing_binary_explains_the_launchd_path_trap(monkeypatch):
    monkeypatch.setattr(voice, "resolve_binary", lambda name: None)
    with pytest.raises(voice.VoiceError) as exc:
        await voice._run_binary(["ffmpeg", "-version"], timeout=5)
    assert exc.value.status == 503
    msg = str(exc.value)
    assert "launchd" in msg and "GHOST_FFMPEG_BIN" in msg, (
        "the error must name the actual cause, not just 'not installed'")


def test_push_to_talk_block_calls_no_undefined_helpers():
    """Dormant code is not dead code — it is code whose references stop being
    checked.

    The mic engine sat behind `if (micBtn)` from 2026-08-01 (button deleted)
    until 2026-08-02 (restored). In between, one of its calls rotted:
    `updateActivityIcon` was removed along with the center-stage
    `#activity-icon` on 2026-07-29, but TWO call sites survived. Reviving the
    button revived the ReferenceError and broke STT with "can't find variable"
    — and the second site had been failing every SUCCESSFUL file upload the
    whole time (its catch reported "Upload Error" for uploads that worked).
    """
    import re

    js = (_ROOT / "interface" / "static" / "app.js").read_text()
    start = js.find("// --- Push-To-Talk Audio Engine (Mic) ---")
    assert start != -1, "push-to-talk block missing"
    block = js[start:start + 6000]
    # Strip comments: prose that NAMES a removed helper must not count as a call.
    code = re.sub(r"//.*", "", block)

    called = set(re.findall(r"(?<![\w.$])([a-z][A-Za-z0-9_]{3,})\s*\(", code))
    defined = (set(re.findall(r"function\s+([A-Za-z0-9_]+)", js))
               | set(re.findall(r"(?:const|let|var)\s+([A-Za-z0-9_]+)\s*=", js)))
    browser_globals = {"setTimeout", "clearTimeout", "setInterval", "fetch",
                       "parseInt", "parseFloat", "isNaN", "requestAnimationFrame",
                       "encodeURIComponent", "decodeURIComponent", "queueMicrotask"}
    # `foo(` also matches JS keywords that take a parenthesised head
    # (`async (e) =>`, `catch (err)`) and locally-bound names the file-wide
    # `defined` scan cannot see (Promise executor params). Neither is a helper.
    keywords_and_locals = {"async", "await", "catch", "if", "for", "while",
                           "switch", "return", "typeof", "function",
                           "resolve", "reject"}

    missing = called - defined - browser_globals - keywords_and_locals
    assert not missing, f"push-to-talk calls undefined helper(s): {sorted(missing)}"


def test_removed_activity_icon_api_has_no_callers():
    """`updateActivityIcon` was deleted with #activity-icon on 2026-07-29."""
    import re

    js = (_ROOT / "interface" / "static" / "app.js").read_text()
    code = re.sub(r"//.*", "", js)
    assert "updateActivityIcon" not in code, (
        "updateActivityIcon was removed in 2026-07; it must have no call sites")


def test_clockwork_client_voice_is_repointed_and_authenticated():
    """The uConsole client had the SAME dead-backend bug as the web UI: its
    STT/TTS pointed at `http://192.168.0.24:8000` — the retired Pi voice
    server — so voice on that device silently did nothing. It now targets the
    interface's endpoints and must send the key, exactly like its /api/chat
    calls already do."""
    src = (_ROOT / "interface" / "externals" / "clockwork_ghost"
           / "client.py").read_text()
    code = "\n".join(ln for ln in src.splitlines()
                     if not ln.lstrip().startswith("#"))

    assert "192.168.0.24" not in code, "dead Pi voice server still referenced"
    assert "VOICE_BASE_URL" in code
    assert "/api/stt" in code and "/api/tts" in code

    # Both voice calls must carry the shared key; an unauthenticated call to
    # the interface is a flat 401 and would look like "voice is broken again".
    stt_call = code[code.find("STT_SERVER_URL, files="):][:300]
    assert "X-Ghost-Key" in stt_call, "STT upload missing the auth header"
    tts_call = code[code.find("TTS_SERVER_URL, json="):][:300]
    assert "X-Ghost-Key" in tts_call, "TTS fetch missing the auth header"


def test_stt_and_tts_still_require_auth():
    """Both endpoints stay behind the shared-key dependency after the rewrite."""
    src = (_ROOT / "interface" / "server.py").read_text()
    for route in ("/api/stt", "/api/tts"):
        idx = src.find(f'"{route}"')
        assert idx != -1, f"{route} vanished from server.py"
        assert "verify_interface_key" in src[max(0, idx - 200): idx + 200]
