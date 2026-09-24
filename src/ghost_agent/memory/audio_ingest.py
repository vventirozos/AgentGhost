"""Long-form audio → knowledge base.

The sibling of :mod:`pdf_ingest` for spoken material: conference talks,
podcast interviews, panels, recorded meetings. It follows that module's shape
exactly — ``iter_*_chunks`` yields breadcrumb-prefixed pieces, and
``ingest_*_streaming`` batches them into ``memory_system.ingest_document`` in
bounded memory — so audio lands in the SAME store, with the same retrieval
behaviour, as every other document.

**Why this exists.** A large slice of AI research discourse is audio-first and
was previously invisible to the agent: nothing in the stack could read a
``.wav``/``.mp3`` at all. Transcription runs on nova's Gemma 4 E4B with an
audio projector — the agent's own node over the tailnet, keyless, no
internet egress — so ingesting a talk costs nothing but idle time on a node
the agent already runs.

**The breadcrumb is a TIMESTAMP RANGE.** PDF ingest uses TOC sections as the
retrieval unit because that is a document's natural structure; audio has no
table of contents, so each window stamps ``[12:00–24:00]`` onto every chunk it
produces. That makes a retrieved passage *citable*: the operator can jump
straight to the moment in the recording. Per-sentence timestamps are
deliberately NOT claimed — the model returns text, not an alignment, and
inventing finer offsets would be fabricated precision.

**Sizing (measured live 2026-08-02).** Audio costs a constant 25.0 tokens per
second. nova serves ``--ctx-size 131072`` across ``-np 4`` slots = 32,768
tokens per slot, so a 12-minute window is ~18k audio tokens plus its
transcript — comfortably inside one slot. Windows overlap slightly so a
sentence spanning a boundary survives in at least one of them; the cost is a
little duplicated text at the seams, which retrieval tolerates far better
than a truncated sentence.

**§4KE (2026-09-24) — what the review changed.**

* Thinking is OFF for transcription (``chat_template_kwargs``). Measured on
  nova: a ~5-minute window cost 2425 completion tokens with thinking (1346 of
  them reasoning) against 1079 without, 90 s against 54 s, identical text. A
  12-minute Greek window with thinking on was within reach of the 8192-token
  cap — and a window cut at the cap WITH text was stored as complete.
* Every non-``stop`` finish is recorded as a TRUNCATED window with its
  timestamp; the old code raised only for the empty+length shape.
* Coverage is the union of transcribed windows, and failed windows are
  reported as GAPS (timestamp range + cause) — not folded into a max
  endpoint that read "1:00:00 transcribed" after a middle window died.
* One retry per window on transient node faults, an abort after three
  consecutive failures (a starved node used to cost windows × 900 s), split
  connect/read timeouts, progress per WINDOW, and ingests serialised because
  the node is shared with the critic.
* The ``(no speech)`` sentinel is matched by SHAPE (a short bracketed reply,
  quotes/markdown/labels stripped, a small multilingual set), and a sentinel
  followed by real speech keeps the speech.
* ffmpeg takes the FIRST audio stream explicitly; the probe requires an
  audio stream and falls back to stream durations when the container header
  says ``N/A``; the ordered transcript is kept on the stats for the caller.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Tuple

from ..utils.helpers import env_positive, semantic_split_text

logger = logging.getLogger(__name__)

# Transcription backend. Tailnet IP on purpose: macOS Tahoe silently drops a
# system daemon's packets to 192.168.x, and mDNS/dotless hostnames are exactly
# what stranded the previous (now-deleted) voice server.
AUDIO_NODE_URL = os.environ.get("GHOST_AUDIO_NODE_URL", "http://100.83.184.117:8088")
AUDIO_NODE_MODEL = os.environ.get("GHOST_AUDIO_NODE_MODEL", "gemma")

# 12 min: ~18k audio tokens at the measured 25 tok/s, leaving room in a
# 32,768-token slot for the transcript. All four knobs go through
# `env_positive`: "0", "abc" and negatives fall back to the default instead
# of raising at import or producing zero windows (§4KE).
WINDOW_SECONDS = env_positive("GHOST_AUDIO_WINDOW_S", 720.0)
# Enough to carry a sentence across a seam without meaningful duplication.
WINDOW_OVERLAP_SECONDS = env_positive("GHOST_AUDIO_WINDOW_OVERLAP_S", 15.0)
if WINDOW_OVERLAP_SECONDS >= WINDOW_SECONDS:
    # An overlap ≥ the window makes the step 1 s → 720 node calls per window.
    WINDOW_OVERLAP_SECONDS = WINDOW_SECONDS / 4.0
# A safety rail, not a judgement: 6 h is longer than any talk, and an
# accidental multi-day recording should fail fast instead of occupying a slot
# for hours.
MAX_AUDIO_SECONDS = env_positive("GHOST_AUDIO_MAX_S", float(6 * 3600))
# Output budget per window. With thinking off (below) a dense 12-minute
# window is ~3–4k tokens; the cap is a rail, and hitting it is REPORTED.
WINDOW_MAX_TOKENS = int(env_positive("GHOST_AUDIO_MAX_TOKENS", 8192))
# Read timeout per window (the node's generation time), and the connect
# timeout — a starved node that accepts TCP and never answers must not cost
# the whole read budget just to be diagnosed as down.
WINDOW_TIMEOUT_S = env_positive("GHOST_AUDIO_TIMEOUT_S", 900.0)
CONNECT_TIMEOUT_S = env_positive("GHOST_AUDIO_CONNECT_TIMEOUT_S", 10.0)
# ffmpeg cuts a window in seconds; it used to borrow the 900 s network budget.
CUT_TIMEOUT_S = env_positive("GHOST_AUDIO_CUT_TIMEOUT_S", 180.0)
# Window-level fault policy.
WINDOW_RETRIES = int(env_positive("GHOST_AUDIO_WINDOW_RETRIES", 1))
MAX_CONSECUTIVE_FAILURES = int(env_positive("GHOST_AUDIO_MAX_CONSECUTIVE_FAILURES", 3))
RETRY_SLEEP_S = env_positive("GHOST_AUDIO_RETRY_SLEEP_S", 5.0)
# Wall budget for ONE recording (resolve + every window). Past it the rest of
# the timeline is reported as a gap and what was transcribed is KEPT — a
# budget is a deadline, not a duration.
TOTAL_BUDGET_S = env_positive("GHOST_AUDIO_TOTAL_BUDGET_S", 2 * 3600.0)
# How long a second ingest may wait for the running one before it is told
# to come back later, instead of blocking a worker thread for hours.
LOCK_WAIT_S = env_positive("GHOST_AUDIO_LOCK_WAIT_S", 600.0)
# Gemma 4's thinking is stripped from the reply but still billed against
# max_tokens and wall time. `GHOST_AUDIO_THINKING=1` re-enables it.
THINKING_ENABLED = os.environ.get("GHOST_AUDIO_THINKING", "").strip() in ("1", "true", "yes")

# Match pdf_ingest so both document kinds chunk identically downstream.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
BATCH_CHUNKS = 256

_TRANSCRIBE_PROMPT = (
    "Write out every word spoken in this audio, from the first word to the "
    "last. Transcribe verbatim in whatever language is spoken — do not "
    "translate, summarise, or add commentary. Output only the spoken words. "
    "If there is no intelligible speech, reply with exactly: (no speech)"
)
_NO_SPEECH = "(no speech)"
#: Honest "nothing here" replies seen or expected from the model, after
#: normalisation (lower-case, quotes/markdown/labels stripped). The SHAPE
#: rule in `_split_no_speech` covers forms not listed.
_NO_SPEECH_PHRASES = (
    "(no speech)", "no speech", "no speech.", "no intelligible speech",
    "no intelligible speech.", "there is no intelligible speech in this audio.",
    "(silence)", "silence", "(music)", "music", "(no audio)", "[no speech]",
    "(χωρίς ομιλία)", "χωρίς ομιλία", "καμία ομιλία", "δεν υπάρχει ομιλία",
    "(kein sprechen)", "(sin voz)", "(pas de parole)",
)
_SENTINEL_MAX_CHARS = 40
#: Text after a leading sentinel that is long enough to be real speech.
_SPEECH_AFTER_SENTINEL_MIN = 80

# One transcription at a time: the node is shared with the critic, and a
# second ingest doubles the slot pressure without doubling the throughput.
_INGEST_LOCK = threading.Lock()


class AudioNodeUnavailable(RuntimeError):
    """Kept for callers that import it; since §4KE round 2 a failing node no
    longer raises out of the generator — it sets ``stats.aborted`` so the
    windows already transcribed are kept."""


@dataclass
class AudioIngestStats:
    """Mirrors :class:`pdf_ingest.IngestStats` so callers can report either."""

    windows: int = 0
    seconds: float = 0.0            # audio actually TRANSCRIBED (union of windows)
    total_seconds: float = 0.0      # what the recording holds (after the cap)
    chunks: int = 0
    chars: int = 0
    skipped_windows: int = 0
    truncated: bool = False         # the recording exceeded MAX_AUDIO_SECONDS
    truncated_windows: int = 0      # windows the token cap cut short
    retries: int = 0
    errors: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)          # "12:00–24:00 (cause)"
    transcript: List[Tuple[float, float, str]] = field(default_factory=list)
    aborted: str = ""               # why the loop stopped early ("" = it did not)


def format_timestamp(seconds: float) -> str:
    """``754.2`` → ``12:34``; past an hour → ``1:02:34``."""
    total = int(max(0.0, seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# Absolute fallbacks for binary lookup — LOAD-BEARING under launchd, which
# gives a daemon a minimal PATH (/usr/bin:/bin:/usr/sbin:/sbin) that excludes
# Homebrew. ffmpeg/ffprobe live in /opt/homebrew/bin, so a bare lookup fails in
# the deployed process while working from a shell. (Cost the interface every
# STT request as a 503 on 2026-08-02; this path had the same latent bug.)
# Deliberately duplicated from interface/voice.py: the interface is a separate
# deployable that cannot import the agent package.
_BIN_PREFIXES = ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin")


def resolve_binary(name: str) -> Optional[str]:
    """Find an executable without trusting the inherited PATH.

    Order: explicit ``GHOST_<NAME>_BIN`` override → PATH → known prefixes.
    """
    import shutil

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


def _run(cmd: List[str], *, timeout: float) -> subprocess.CompletedProcess:
    """Run a binary with no shell. Raises with the tool's own last line."""
    resolved = resolve_binary(cmd[0])
    if not resolved:
        raise RuntimeError(
            f"'{cmd[0]}' not found on PATH or in {', '.join(_BIN_PREFIXES)}. "
            f"Under launchd the PATH is minimal and excludes Homebrew — set "
            f"GHOST_{cmd[0].upper()}_BIN to an absolute path.")
    cmd = [resolved, *cmd[1:]]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise RuntimeError(f"'{cmd[0]}' is not installed or not on PATH.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"'{cmd[0]}' timed out after {timeout:.0f}s.") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or b"").decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError(f"'{cmd[0]}' failed: {tail[-1][:200] if tail else proc.returncode}")
    return proc


def probe_duration_seconds(file_path: Path) -> float:
    """Total duration via ffprobe, for any container ffmpeg can read.

    Reads the FORMAT duration and the STREAM durations, and requires an
    audio stream: a video-only file used to pass here and then fail once
    per window in ffmpeg, and a container whose header says ``N/A`` (a
    stream cut mid-write, a raw ADTS file) crashed on ``float("N/A")``.
    """
    proc = _run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration:stream=codec_type,duration",
        "-of", "json", str(file_path),
    ], timeout=60)
    try:
        data = json.loads(proc.stdout or b"{}")
    except ValueError as exc:
        raise RuntimeError(f"Could not read {file_path.name}: ffprobe output was not JSON ({exc})") from exc
    streams = data.get("streams") or []
    audio = [s for s in streams if isinstance(s, dict) and s.get("codec_type") == "audio"]
    if streams and not audio:
        raise RuntimeError(
            f"{file_path.name} has no audio stream (found: "
            f"{', '.join(str(s.get('codec_type')) for s in streams)}) — nothing to transcribe.")
    candidates: List[float] = []
    for src in ([data.get("format") or {}] + audio):
        try:
            v = float(src.get("duration"))
            if v > 0:
                candidates.append(v)
        except (TypeError, ValueError):
            continue
    if not candidates:
        raise RuntimeError(
            f"Could not read a duration from {file_path.name}: the container reports none "
            f"(a recording cut mid-write or a raw stream). Remux it first, e.g. "
            f"ffmpeg -i in -c copy out.m4a.")
    return max(candidates)


def extract_window_wav(file_path: Path, start: float, duration: float) -> bytes:
    """Cut one window and normalise it to 16 kHz mono PCM WAV.

    ``-ss`` precedes ``-i`` so ffmpeg seeks before decoding — on a 3-hour file
    that is the difference between a fast seek and decoding everything up to
    the window each time. ``-map 0:a:0`` takes the FIRST audio stream
    explicitly: ffmpeg's default prefers the stream with the most channels,
    which on a screen recording is the stereo system audio, not the mic.
    """
    with tempfile.TemporaryDirectory(prefix="ghost-audio-") as td:
        out = Path(td) / "window.wav"
        _run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(file_path),
            "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(out),
        ], timeout=CUT_TIMEOUT_S)
        if not out.exists() or out.stat().st_size == 0:
            raise RuntimeError(f"ffmpeg produced no audio for window at {start:.0f}s")
        return out.read_bytes()


class TransientNodeError(RuntimeError):
    """A fault a second attempt may not hit: transport, timeout, 5xx, 429."""


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    """POST JSON and return the decoded body.

    httpx is imported lazily so this module stays import-safe in environments
    that never ingest audio — the same reason pdf_ingest defers ``import fitz``.
    The timeout is split: ``timeout`` is the READ budget (the node generating),
    the connect budget is short — a node that does not accept the connection
    is down, not slow.
    """
    import httpx

    limits = httpx.Timeout(connect=CONNECT_TIMEOUT_S, read=timeout, write=60.0, pool=CONNECT_TIMEOUT_S)
    try:
        with httpx.Client(timeout=limits) as client:
            resp = client.post(url, json=payload)
    except httpx.TransportError as exc:  # timeouts, resets, refused, DNS
        raise TransientNodeError(f"audio node unreachable: {type(exc).__name__}: {exc}") from exc
    if resp.status_code in (429, 500, 502, 503, 504):
        raise TransientNodeError(
            f"audio node returned HTTP {resp.status_code}: {(resp.text or '')[:200]}")
    if resp.status_code != 200:
        raise RuntimeError(
            f"audio node returned HTTP {resp.status_code}: {(resp.text or '')[:200]}")
    try:
        return resp.json()
    except ValueError as exc:
        raise TransientNodeError("audio node returned a non-JSON body") from exc


def build_payload(wav_bytes: bytes) -> dict:
    import base64

    payload = {
        "model": AUDIO_NODE_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": _TRANSCRIBE_PROMPT},
            # `input_audio`, NOT the `audio_url` data-URI shape used for
            # images — the node rejects that with 400 unsupported content type.
            {"type": "input_audio", "input_audio": {
                "data": base64.b64encode(wav_bytes).decode("ascii"), "format": "wav"}},
        ]}],
        "temperature": 0.0,
        "max_tokens": WINDOW_MAX_TOKENS,
    }
    if not THINKING_ENABLED:
        # llama.cpp forwards this to the chat template; verified on nova's
        # Gemma 4 build 2026-09-24 (reasoning_content empty, same text).
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    return payload


def _normalise_reply(text: str) -> str:
    t = text.strip()
    # Labels the model sometimes prepends.
    for label in ("transcript:", "transcription:", "output:", "μεταγραφή:"):
        if t.lower().startswith(label):
            t = t[len(label):].strip()
    # Quote / markdown wrapping.
    t = t.strip("`*_\"'“”‘’ \n\t")
    return t


def _split_no_speech(text: str) -> Tuple[bool, str]:
    """``(is_sentinel, remaining_speech)``.

    A reply is a "no speech" sentinel when, normalised, it is one of the
    known phrases, OR it is short and bracketed (``(…)``/``[…]``) — no
    12-minute window of real speech transcribes to 40 characters in
    brackets, in any language. When a sentinel LEADS a reply that then
    carries real text, the text is kept: the old prefix match threw away
    nine minutes of speech behind a stray ``(no speech)``.
    """
    norm = _normalise_reply(text)
    low = norm.lower()
    if not low:
        return True, ""
    if low in _NO_SPEECH_PHRASES:
        return True, ""
    if len(low) <= _SENTINEL_MAX_CHARS and low[0] in "([" and low[-1] in ")]":
        return True, ""
    for phrase in _NO_SPEECH_PHRASES:
        if low.startswith(phrase):
            rest = norm[len(phrase):].strip(" .:-\n")
            if len(rest) >= _SPEECH_AFTER_SENTINEL_MIN:
                return False, rest
            return True, ""
    return False, norm


def transcribe_window_full(wav_bytes: bytes, *, post_fn: Optional[Callable] = None) -> Tuple[str, bool]:
    """Transcribe one prepared 16 kHz mono WAV window.

    Returns ``(text, truncated)`` — ``truncated`` when the node stopped for
    any reason other than the end of the transcript (``finish_reason`` not
    ``stop``). ``post_fn`` is injectable so tests never touch the network.
    """
    post = post_fn or _post_json
    data = post(f"{AUDIO_NODE_URL}/v1/chat/completions", build_payload(wav_bytes), WINDOW_TIMEOUT_S)
    try:
        choice = data["choices"][0]
        # `or ""` rather than .get(default): the API sends explicit nulls.
        text = (choice["message"].get("content") or "").strip()
        finish = choice.get("finish_reason")
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"malformed response from audio node: {exc}") from exc

    if not text and finish == "length":
        # The measured silent-failure shape: Gemma 4's thinking blocks are
        # stripped by its chat template, so an exhausted budget yields EMPTY
        # content rather than an error. Never let that pass as "silence" —
        # a whole window would vanish from the transcript unnoticed.
        raise RuntimeError(
            f"transcription returned no text: thinking tokens (or a repetition loop) "
            f"consumed the entire {WINDOW_MAX_TOKENS}-token budget (finish_reason=length). "
            f"Raise GHOST_AUDIO_MAX_TOKENS.")
    truncated = bool(text) and finish is not None and finish != "stop"
    is_sentinel, speech = _split_no_speech(text)
    if is_sentinel:
        return "", False
    return speech, truncated


def transcribe_window(wav_bytes: bytes, *, post_fn: Optional[Callable] = None) -> str:
    """Text-only form of :func:`transcribe_window_full` (kept for callers
    that do not track truncation)."""
    return transcribe_window_full(wav_bytes, post_fn=post_fn)[0]


def _transcribe_with_retry(wav: bytes, *, post_fn, st: AudioIngestStats, sleep=time.sleep) -> Tuple[str, bool]:
    attempt = 0
    while True:
        try:
            return transcribe_window_full(wav, post_fn=post_fn)
        except TransientNodeError:
            if attempt >= WINDOW_RETRIES:
                raise
            attempt += 1
            st.retries += 1
            sleep(RETRY_SLEEP_S)


def iter_audio_chunks(
    file_path: Path,
    filename: str,
    *,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    window_seconds: float = WINDOW_SECONDS,
    window_overlap: float = WINDOW_OVERLAP_SECONDS,
    max_seconds: float = MAX_AUDIO_SECONDS,
    stats: Optional[AudioIngestStats] = None,
    post_fn: Optional[Callable] = None,
    window_done: Optional[Callable[[AudioIngestStats], None]] = None,
    max_consecutive_failures: int = MAX_CONSECUTIVE_FAILURES,
    sleep: Callable[[float], None] = time.sleep,
    total_budget_s: float = TOTAL_BUDGET_S,
) -> Iterator[str]:
    """Yield timestamp-stamped transcript chunks from a recording.

    One bad window is retried once, then skipped and recorded as a GAP in
    ``stats.gaps``; it never sinks the whole recording. Three consecutive
    NODE failures (ffmpeg failures are the file's, and do not count) or an
    exhausted wall budget STOP the loop: ``stats.aborted`` says why, the
    rest of the timeline is one more gap, and everything yielded so far is
    kept by the caller — a 2-hour talk that lost its node at window 9 keeps
    windows 1–8 instead of throwing them away.
    """
    st = stats if stats is not None else AudioIngestStats()
    total = probe_duration_seconds(file_path)
    if total > max_seconds:
        st.truncated = True
        logger.warning("audio %s is %.1f min; ingesting only the first %.1f min",
                       filename, total / 60, max_seconds / 60)
        total = max_seconds
    st.total_seconds = total

    # Advance by (window - overlap) so consecutive windows share a seam.
    step = max(1.0, window_seconds - window_overlap)
    start = 0.0
    consecutive = 0
    covered_to = 0.0
    deadline = time.monotonic() + total_budget_s

    def _stop(reason: str) -> None:
        # Report the REST of the timeline as one gap and stop. Nothing
        # already yielded is lost: the caller flushes what it holds and the
        # record says exactly what is missing and why.
        st.aborted = reason
        st.gaps.append(f"{format_timestamp(start)}–{format_timestamp(total)} (not attempted: {reason[:80]})")
        logger.warning("audio ingest %s stopped at %s: %s", filename, format_timestamp(start), reason)

    while start < total:
        duration = min(window_seconds, total - start)
        if duration < 0.5:  # a sliver at the tail carries nothing
            break
        end = start + duration
        if end <= covered_to:
            # The tail already lies inside the previous window's overlap:
            # transcribing it again is a full node round-trip for text the
            # store already holds (15 s of audio, ~60 s of nova).
            break
        if time.monotonic() > deadline:
            _stop(f"time budget of {format_timestamp(total_budget_s)} exhausted")
            break
        crumb = f"[{filename}] [{format_timestamp(start)}–{format_timestamp(end)}]"
        try:
            wav = extract_window_wav(file_path, start, duration)
        except Exception as exc:  # noqa: BLE001 — the FILE, not the node: never counts toward the abort
            st.skipped_windows += 1
            cause = f"ffmpeg: {exc}"
            st.errors.append(f"{format_timestamp(start)}: {cause}")
            st.gaps.append(f"{format_timestamp(start)}–{format_timestamp(end)} ({cause[:80]})")
            logger.warning("audio window at %s failed: %s", format_timestamp(start), cause)
            if window_done:
                _safe(window_done, st)
            start += step
            continue
        try:
            text, cut = _transcribe_with_retry(wav, post_fn=post_fn, st=st, sleep=sleep)
        except Exception as exc:  # noqa: BLE001 — skip the window, not the file
            st.skipped_windows += 1
            consecutive += 1
            cause = str(exc)
            st.errors.append(f"{format_timestamp(start)}: {cause}")
            st.gaps.append(f"{format_timestamp(start)}–{format_timestamp(end)} ({cause[:80]})")
            logger.warning("audio window at %s failed: %s", format_timestamp(start), cause)
            if window_done:
                _safe(window_done, st)
            start += step
            if consecutive >= max_consecutive_failures:
                _stop(f"{consecutive} consecutive windows failed (last: {cause[:120]}) — "
                      f"the audio node is not answering")
                break
            continue

        consecutive = 0
        st.windows += 1
        # Coverage = union of transcribed windows, NOT the sum of window
        # lengths (overlap over-reports) and NOT the max endpoint (a failed
        # middle window would vanish from the number).
        st.seconds += end - max(start, covered_to)
        covered_to = max(covered_to, end)
        if cut:
            st.truncated_windows += 1
            st.errors.append(f"{format_timestamp(start)}: transcript cut short at the "
                             f"{WINDOW_MAX_TOKENS}-token budget")
            st.gaps.append(f"{format_timestamp(start)}–{format_timestamp(end)} (tail cut at the token budget)")
        if text:
            st.chars += len(text)
            st.transcript.append((start, end, text))
            # Stamp the breadcrumb on EVERY piece so the embedded text itself
            # carries the timestamp, not just the metadata around it.
            for piece in semantic_split_text(text, chunk_size, chunk_overlap):
                piece = piece.strip()
                if piece:
                    st.chunks += 1
                    yield f"{crumb}\n{piece}"
        if window_done:
            _safe(window_done, st)
        start += step


def _safe(cb, st) -> None:
    try:
        cb(st)
    except Exception:  # noqa: BLE001 — progress must never break ingest
        pass


def ingest_audio_streaming(
    file_path: Path,
    filename: str,
    memory_system,
    *,
    progress: Optional[Callable[[AudioIngestStats], None]] = None,
    **kwargs,
) -> AudioIngestStats:
    """Stream a recording into the vector store in bounded memory.

    Batches of ``BATCH_CHUNKS`` are flushed to ``ingest_document`` so peak RAM
    is one batch, not one recording. ``progress`` fires after EVERY window
    (the batch boundary is hours away for a talk). Returns the stats; raises
    on a fatal condition (unreadable file, embedding failure, the node down),
    never on one bad window. Ingests are serialised process-wide.
    """
    stats = AudioIngestStats()
    batch: List[str] = []
    flushed = 0

    def _flush() -> None:
        nonlocal batch, flushed
        if not batch:
            return
        ok, msg = memory_system.ingest_document(filename, batch, _batch=True)
        if not ok:
            raise RuntimeError(f"embedding failed at chunk {flushed}: {msg}")
        flushed += len(batch)
        batch = []

    if not _INGEST_LOCK.acquire(timeout=0):
        logger.info("audio ingest %s: waiting for the running transcription to finish", filename)
        if not _INGEST_LOCK.acquire(timeout=LOCK_WAIT_S):
            raise RuntimeError(
                f"another transcription has held the audio node for over "
                f"{format_timestamp(LOCK_WAIT_S)}; nothing was stored for '{filename}' — "
                f"try again when it finishes.")
    try:
        for chunk in iter_audio_chunks(file_path, filename, stats=stats,
                                       window_done=progress, **kwargs):
            batch.append(chunk)
            if len(batch) >= BATCH_CHUNKS:
                _flush()
        _flush()
    finally:
        _INGEST_LOCK.release()

    logger.info("audio ingest %s: %d windows, %.1f of %.1f min, %d chunks, %d skipped, %d cut",
                filename, stats.windows, stats.seconds / 60, stats.total_seconds / 60,
                stats.chunks, stats.skipped_windows, stats.truncated_windows)
    return stats
