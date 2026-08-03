"""Local voice loop for the web interface — STT on the Gemma 4 audio node,
TTS via the local macOS speech synthesiser.

**Why this module exists.** The interface has shipped a complete push-to-talk
UI since 2026-07 (mic button, MediaRecorder, full touch lifecycle) whose two
endpoints proxied to a Raspberry-Pi "voice server" at ``PI_VOICE_URL``. That
host no longer exists — the default ``raspberrypi.local`` does not resolve and
the launcher's override ``http://disorder:8000`` resolves on the tailnet but
listens on nothing. So the mic button was a no-op against a dead node: the
classic silent-inoperative-subsystem shape. Both halves now run on
infrastructure that is actually live:

* **STT** — nova's llama-server (the same worker node the agent already uses),
  serving Gemma 4 E4B with an audio projector (``-mm``). Verified live
  2026-08-02: 25.0 audio tokens/sec, ~11x faster than real-time, no encoder
  window limit up to at least 6.3 minutes.
* **TTS** — the macOS ``say`` binary already on the box (184 voices).

Neither half needs extra hardware, an API key, or any egress: audio never
leaves the tailnet. That matters beyond convenience — keyless local
processing is the only speech path compatible with the operator's
no-identity-egress rule, so this capability cannot be bought, only built.

**Two traps encoded here, both measured rather than guessed:**

1. *The empty-response trap.* Gemma 4 emits thinking blocks that its chat
   template strips. With ``max_tokens`` too low the whole budget goes to
   thinking and the API returns EMPTY content with ``finish_reason="length"``
   — no error, just nothing. Measured: 256 tokens → ``''``; 1024 → correct.
   :func:`transcribe` detects that exact shape and raises instead of
   returning a silent empty transcript.
2. *Dialect.* Audio requires the OpenAI ``input_audio`` content part. The
   ``audio_url`` data-URI form that ``tools/vision.py`` uses for images is
   rejected by the node with ``400 unsupported content[].type``, so this is
   deliberately NOT a copy of the vision payload builder.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import shutil
import tempfile
import wave
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────
# Nova, addressed by TAILNET ip: macOS Tahoe's Local Network privacy silently
# drops a system daemon's packets to 192.168.x addresses before they hit the
# wire, and a dotless/mDNS hostname (`disorder`, `raspberrypi.local`) is what
# stranded the previous voice backend. utun/tailnet is exempt.
AUDIO_NODE_URL = os.environ.get("GHOST_AUDIO_NODE_URL", "http://100.83.184.117:8088")
AUDIO_NODE_MODEL = os.environ.get("GHOST_AUDIO_NODE_MODEL", "gemma")

# Per-slot context on nova is `--ctx-size 131072 / -np 4` = 32,768 tokens, and
# audio costs a measured 25.0 tok/s, so ~21 min is the hard ceiling. 15 min
# leaves room for the prompt and the transcript itself. Push-to-talk clips are
# seconds long; this bound exists so a stray large upload fails fast and
# legibly instead of wedging a slot.
STT_MAX_SECONDS = float(os.environ.get("GHOST_STT_MAX_SECONDS", "900"))
# Must clear the thinking-token budget — see trap 1 in the module docstring.
STT_MAX_TOKENS = int(os.environ.get("GHOST_STT_MAX_TOKENS", "2048"))
STT_TIMEOUT_S = float(os.environ.get("GHOST_STT_TIMEOUT_S", "180"))

# Female voice (operator preference, 2026-08-03; was "Samantha", itself a
# 2026-08-02 replacement for "Daniel", en_GB male). Ava (Premium) is a
# neural voice, a clear step up in naturalness from the compact synthesiser
# every other name here uses.
#
# ⚠ Unlike Samantha, Ava is NOT present on a stock macOS — it is a ~1 GB
# download (System Settings → Accessibility → Spoken Content → System Voice
# → Manage Voices…). The parenthesised suffix is part of the name and MUST
# be exact: bare "Ava" is a DIFFERENT, compact voice that also exists once
# the download completes, so a dropped suffix silently downgrades quality
# rather than erroring. On a box without the download, synthesize() falls
# back to the system default (see the VoiceError handler below) — speech
# survives, quality degrades, and the warning names the cause.
#
# Measured on this box 2026-08-03: 0.58s to synthesize a two-sentence chunk
# vs 0.48s for Samantha, so the premium model costs no meaningful latency
# against TTS_TIMEOUT_S.
#
# Alternatives, all verified working through synthesize(): "Samantha"
# (en_US, stock everywhere), "Sandy (English (UK))", "Karen" (en_AU),
# "Moira" (en_IE). List them with `say -v '?'`.
TTS_VOICE = os.environ.get("GHOST_TTS_VOICE", "Ava (Premium)")
TTS_RATE = os.environ.get("GHOST_TTS_RATE", "")  # words/min; "" = voice default
# app.js already splits replies into chunks and queues them, so a single
# request is a sentence or two. The cap stops a pathological chunk from
# turning into a multi-minute synthesis job.
TTS_MAX_CHARS = int(os.environ.get("GHOST_TTS_MAX_CHARS", "2000"))
TTS_TIMEOUT_S = float(os.environ.get("GHOST_TTS_TIMEOUT_S", "60"))

# Verbatim transcription, NOT intent extraction. The transcript goes straight
# to the 35B agent, which is the thing that should be interpreting intent —
# summarising or "extracting the request" on the small model would discard
# information before the smart model ever sees it. Verbatim also keeps the
# spoken language intact (no forced translation), which matters for a
# bilingual operator: the agent handles either language fine.
# The phrasing is measured, not arbitrary — "Transcribe this audio exactly"
# truncated mid-sentence in testing, while this form returned complete text.
_STT_PROMPT = (
    "Write out every word spoken in this audio, from the first word to the "
    "last. Transcribe verbatim in whatever language is spoken — do not "
    "translate, summarise, or add commentary. Output only the spoken words. "
    "If there is no intelligible speech, reply with exactly: (no speech)"
)
_NO_SPEECH_SENTINEL = "(no speech)"


class VoiceError(RuntimeError):
    """A voice-pipeline failure with an HTTP status to surface to the client.

    ``status`` distinguishes "the caller sent something unusable" (4xx) from
    "our backend broke" (5xx) so the endpoint doesn't collapse both into a
    misleading 502 — the same client-vs-server confusion the 2026-07 audit
    fixed for the text-field-named-"file" case.
    """

    def __init__(self, message: str, status: int = 502) -> None:
        super().__init__(message)
        self.status = status


# Absolute fallbacks for binary lookup. THIS IS LOAD-BEARING: the interface
# runs under launchd, which hands a service a MINIMAL PATH
# (/usr/bin:/bin:/usr/sbin:/sbin) that does NOT include Homebrew. `say` lives
# in /usr/bin so it resolved fine, but ffmpeg/ffprobe live in
# /opt/homebrew/bin — so a bare shutil.which("ffmpeg") returned None and every
# STT request 503'd under the daemon while working perfectly from a shell.
# Same defect class as the node binary that was invisible under launchd PATH.
_BIN_PREFIXES = ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin")


def resolve_binary(name: str) -> str | None:
    """Find an executable without trusting the inherited PATH.

    Order: explicit ``GHOST_<NAME>_BIN`` override → PATH → known install
    prefixes. Returns an absolute path, or None if genuinely absent.
    """
    override = os.environ.get(f"GHOST_{name.upper()}_BIN")
    if override and os.path.isfile(override) and os.access(override, os.X_OK):
        return override
    found = shutil.which(name)
    if found:
        return found
    for prefix in _BIN_PREFIXES:
        candidate = os.path.join(prefix, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


async def _run_binary(cmd: list[str], *, timeout: float, stdin: bytes | None = None
                      ) -> tuple[bytes, bytes]:
    """Run a binary with no shell involved, enforcing a wall-clock timeout.

    Every argument is passed as a list element, so operator/model text can
    never be interpreted as shell syntax.
    """
    resolved = resolve_binary(cmd[0])
    if not resolved:
        raise VoiceError(
            f"'{cmd[0]}' not found on PATH or in {', '.join(_BIN_PREFIXES)}. "
            f"Under launchd the PATH is minimal and excludes Homebrew — set "
            f"GHOST_{cmd[0].upper()}_BIN to an absolute path.", 503)
    cmd = [resolved, *cmd[1:]]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(stdin), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        # Reap so the killed child doesn't linger as a zombie — the same
        # mechanism that made dead sandbox services look alive to `kill -0`.
        try:
            await proc.wait()
        except Exception:  # noqa: BLE001 — best effort reap
            pass
        raise VoiceError(f"'{cmd[0]}' timed out after {timeout:.0f}s.", 504) from None
    if proc.returncode != 0:
        detail = (err or b"").decode("utf-8", "replace").strip().splitlines()
        tail = detail[-1][:200] if detail else f"exit {proc.returncode}"
        raise VoiceError(f"'{cmd[0]}' failed: {tail}", 502)
    return out, err


async def transcode_to_wav16k(raw: bytes) -> bytes:
    """Convert ANY browser-produced audio container to 16 kHz mono PCM WAV.

    MediaRecorder emits WebM/Opus on most browsers and MP4/AAC on Safari;
    the audio encoder wants 16 kHz mono PCM. ffmpeg reads/writes via temp
    FILES rather than pipes on purpose: a WAV written to a non-seekable pipe
    carries a placeholder length in its header, which some decoders reject.
    """
    if not raw:
        raise VoiceError("Empty audio upload.", 400)
    with tempfile.TemporaryDirectory(prefix="ghost-stt-") as td:
        src = Path(td) / "in.bin"
        dst = Path(td) / "out.wav"
        src.write_bytes(raw)
        await _run_binary(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
             "-i", str(src), "-ac", "1", "-ar", "16000", "-f", "wav", str(dst)],
            timeout=STT_TIMEOUT_S,
        )
        if not dst.exists() or dst.stat().st_size == 0:
            raise VoiceError("Audio transcode produced no output.", 502)
        return dst.read_bytes()


def wav_duration_seconds(wav_bytes: bytes) -> float:
    """Duration of a PCM WAV, parsed from its header rather than estimated."""
    try:
        with wave.open(io.BytesIO(wav_bytes)) as w:
            frames, rate = w.getnframes(), w.getframerate()
        return (frames / rate) if rate else 0.0
    except Exception as exc:  # noqa: BLE001 — malformed audio is a client error
        raise VoiceError(f"Unreadable WAV data: {exc}", 400) from exc


def _audio_message_payload(wav_bytes: bytes) -> dict:
    """Build the chat payload. Audio rides an ``input_audio`` part — see
    trap 2 in the module docstring for why this is not the vision shape."""
    b64 = base64.b64encode(wav_bytes).decode("ascii")
    return {
        "model": AUDIO_NODE_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": _STT_PROMPT},
            {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
        ]}],
        "temperature": 0.0,
        "max_tokens": STT_MAX_TOKENS,
    }


async def transcribe(raw: bytes, *, client) -> str:
    """Transcode an uploaded clip and transcribe it on the audio node.

    ``client`` is an httpx.AsyncClient supplied by the caller so the endpoint
    reuses the server's shared connection pool.

    Returns "" for genuinely speechless audio (the caller reports "transcribed
    nothing"); raises :class:`VoiceError` for every real failure, so a broken
    backend can never masquerade as silence.
    """
    wav = await transcode_to_wav16k(raw)
    duration = wav_duration_seconds(wav)
    if duration <= 0.05:
        return ""
    if duration > STT_MAX_SECONDS:
        raise VoiceError(
            f"Audio is {duration / 60:.1f} min; the limit is "
            f"{STT_MAX_SECONDS / 60:.0f} min per request "
            f"(raise GHOST_STT_MAX_SECONDS, but nova's per-slot context caps "
            f"a single request near 21 min of audio).", 413)

    resp = await client.post(f"{AUDIO_NODE_URL}/v1/chat/completions",
                             json=_audio_message_payload(wav), timeout=STT_TIMEOUT_S)
    if resp.status_code != 200:
        body = (resp.text or "")[:200]
        raise VoiceError(f"Audio node returned HTTP {resp.status_code}: {body}", 502)
    try:
        data = resp.json()
        choice = data["choices"][0]
        # `or ""` rather than .get(default): the API sends an explicit
        # content: null, for which .get's default never fires.
        text = (choice["message"].get("content") or "").strip()
        finish = choice.get("finish_reason")
    except (KeyError, IndexError, ValueError) as exc:
        raise VoiceError(f"Malformed response from audio node: {exc}", 502) from exc

    if not text:
        # THE trap: thinking tokens ate the whole budget. Fail loudly — a
        # silent "" here would be indistinguishable from silence and would
        # send an empty prompt to the agent.
        if finish == "length":
            raise VoiceError(
                "Transcription returned no text: the model's thinking tokens "
                f"consumed the entire {STT_MAX_TOKENS}-token budget "
                "(finish_reason=length). Raise GHOST_STT_MAX_TOKENS.", 502)
        return ""
    if text.lower().startswith(_NO_SPEECH_SENTINEL):
        return ""
    logger.info("STT: %.1fs audio -> %d chars", duration, len(text))
    return text


async def synthesize(text: str) -> bytes:
    """Render text to WAV bytes with the local macOS speech synthesiser.

    Text is passed via a FILE (``say -f``), never argv or a shell string:
    agent replies routinely contain quotes, backticks and newlines, and this
    keeps all of it inert.
    """
    text = (text or "").strip()
    if not text:
        raise VoiceError("No text to speak.", 400)
    if len(text) > TTS_MAX_CHARS:
        text = text[:TTS_MAX_CHARS].rsplit(" ", 1)[0] + "…"
    with tempfile.TemporaryDirectory(prefix="ghost-tts-") as td:
        txt = Path(td) / "say.txt"
        out = Path(td) / "say.wav"
        txt.write_text(text, encoding="utf-8")
        base = ["say", "-f", str(txt), "-o", str(out), "--data-format=LEI16@22050"]
        if TTS_RATE:
            base[1:1] = ["-r", TTS_RATE]
        try:
            await _run_binary([*base[:1], "-v", TTS_VOICE, *base[1:]],
                              timeout=TTS_TIMEOUT_S)
        except VoiceError:
            # An unknown/uninstalled voice name made `say` exit non-zero, which
            # killed ALL speech — a one-character typo in GHOST_TTS_VOICE, or a
            # voice absent from a given macOS build, took the whole feature
            # down. Fall back to the system default voice: degraded output
            # beats silence, and the warning names the real cause.
            logger.warning("TTS voice %r unavailable — falling back to the "
                           "system default. Check `say -v '?'`.", TTS_VOICE)
            await _run_binary(base, timeout=TTS_TIMEOUT_S)
        if not out.exists() or out.stat().st_size == 0:
            raise VoiceError("Speech synthesis produced no audio.", 502)
        return out.read_bytes()
