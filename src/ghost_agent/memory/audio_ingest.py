"""Long-form audio → knowledge base.

The sibling of :mod:`pdf_ingest` for spoken material: conference talks,
podcast interviews, panels, recorded meetings. It follows that module's shape
exactly — ``iter_*_chunks`` yields breadcrumb-prefixed pieces, and
``ingest_*_streaming`` batches them into ``memory_system.ingest_document`` in
bounded memory — so audio lands in the SAME store, with the same retrieval
behaviour, as every other document.

**Why this exists.** A large slice of AI research discourse is audio-first and
was previously invisible to the agent: nothing in the stack could read a
``.wav``/``.mp3`` at all. Transcription now runs on nova's Gemma 4 E4B with an
audio projector — local, keyless, no egress — so ingesting a talk costs
nothing but idle time on a node the agent already runs.

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
transcript — comfortably inside one slot, with headroom for the model's
thinking tokens. Windows overlap slightly so a sentence spanning a boundary
survives in at least one of them; the cost is a little duplicated text at the
seams, which retrieval tolerates far better than a truncated sentence.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, List, Optional

from ..utils.helpers import semantic_split_text

logger = logging.getLogger(__name__)

# Transcription backend. Tailnet IP on purpose: macOS Tahoe silently drops a
# system daemon's packets to 192.168.x, and mDNS/dotless hostnames are exactly
# what stranded the previous (now-deleted) voice server.
AUDIO_NODE_URL = os.environ.get("GHOST_AUDIO_NODE_URL", "http://100.83.184.117:8088")
AUDIO_NODE_MODEL = os.environ.get("GHOST_AUDIO_NODE_MODEL", "gemma")

# 12 min: ~18k audio tokens at the measured 25 tok/s, leaving room in a
# 32,768-token slot for the transcript AND the model's stripped thinking.
WINDOW_SECONDS = float(os.environ.get("GHOST_AUDIO_WINDOW_S", "720"))
# Enough to carry a sentence across a seam without meaningful duplication.
WINDOW_OVERLAP_SECONDS = float(os.environ.get("GHOST_AUDIO_WINDOW_OVERLAP_S", "15"))
# A safety rail, not a judgement: 6 h is longer than any talk, and an
# accidental multi-day recording should fail fast instead of occupying a slot
# for hours.
MAX_AUDIO_SECONDS = float(os.environ.get("GHOST_AUDIO_MAX_S", str(6 * 3600)))
# Must clear the thinking-token budget — with too low a cap the node returns
# EMPTY content and finish_reason="length". See _transcribe_window.
WINDOW_MAX_TOKENS = int(os.environ.get("GHOST_AUDIO_MAX_TOKENS", "8192"))
from ..utils.helpers import env_positive

WINDOW_TIMEOUT_S = env_positive("GHOST_AUDIO_TIMEOUT_S", 900.0)

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


@dataclass
class AudioIngestStats:
    """Mirrors :class:`pdf_ingest.IngestStats` so callers can report either."""

    windows: int = 0
    seconds: float = 0.0
    chunks: int = 0
    chars: int = 0
    skipped_windows: int = 0
    truncated: bool = False
    errors: List[str] = field(default_factory=list)


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
    """Total duration via ffprobe. Works for any container ffmpeg can read."""
    proc = _run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", str(file_path),
    ], timeout=60)
    try:
        return float(json.loads(proc.stdout)["format"]["duration"])
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f"Could not read a duration from {file_path.name}: {exc}") from exc


def extract_window_wav(file_path: Path, start: float, duration: float) -> bytes:
    """Cut one window and normalise it to 16 kHz mono PCM WAV.

    ``-ss`` precedes ``-i`` so ffmpeg seeks before decoding — on a 3-hour file
    that is the difference between a fast seek and decoding everything up to
    the window each time.
    """
    with tempfile.TemporaryDirectory(prefix="ghost-audio-") as td:
        out = Path(td) / "window.wav"
        _run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(file_path),
            "-ac", "1", "-ar", "16000", "-f", "wav", str(out),
        ], timeout=WINDOW_TIMEOUT_S)
        if not out.exists() or out.stat().st_size == 0:
            raise RuntimeError(f"ffmpeg produced no audio for window at {start:.0f}s")
        return out.read_bytes()


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    """POST JSON and return the decoded body.

    httpx is imported lazily so this module stays import-safe in environments
    that never ingest audio — the same reason pdf_ingest defers ``import fitz``.
    """
    import httpx

    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(
                f"audio node returned HTTP {resp.status_code}: {(resp.text or '')[:200]}")
        return resp.json()


def transcribe_window(wav_bytes: bytes, *, post_fn: Optional[Callable] = None) -> str:
    """Transcribe one prepared 16 kHz mono WAV window.

    ``post_fn`` is injectable so tests never touch the network.
    """
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
    post = post_fn or _post_json
    data = post(f"{AUDIO_NODE_URL}/v1/chat/completions", payload, WINDOW_TIMEOUT_S)
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
            f"transcription returned no text: thinking tokens consumed the entire "
            f"{WINDOW_MAX_TOKENS}-token budget (finish_reason=length). "
            f"Raise GHOST_AUDIO_MAX_TOKENS.")
    if text.lower().startswith(_NO_SPEECH):
        return ""
    return text


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
) -> Iterator[str]:
    """Yield timestamp-stamped transcript chunks from a recording.

    One bad window is skipped and recorded in ``stats.errors``; it never sinks
    the whole recording — the same failure policy pdf_ingest applies to pages.
    """
    st = stats if stats is not None else AudioIngestStats()
    total = probe_duration_seconds(file_path)
    if total > max_seconds:
        st.truncated = True
        logger.warning("audio %s is %.1f min; ingesting only the first %.1f min",
                       filename, total / 60, max_seconds / 60)
        total = max_seconds

    # Advance by (window - overlap) so consecutive windows share a seam.
    step = max(1.0, window_seconds - window_overlap)
    start = 0.0
    while start < total:
        duration = min(window_seconds, total - start)
        if duration < 0.5:  # a sliver at the tail carries nothing
            break
        end = start + duration
        crumb = f"[{filename}] [{format_timestamp(start)}–{format_timestamp(end)}]"
        try:
            wav = extract_window_wav(file_path, start, duration)
            text = transcribe_window(wav, post_fn=post_fn)
        except Exception as exc:  # noqa: BLE001 — skip the window, not the file
            st.skipped_windows += 1
            st.errors.append(f"{format_timestamp(start)}: {exc}")
            logger.warning("audio window at %s failed: %s", format_timestamp(start), exc)
            start += step
            continue

        st.windows += 1
        # Timeline COVERAGE, not the sum of window lengths. Windows overlap by
        # design, so summing durations over-reports: a 6:17 recording came out
        # as "6.8 min transcribed". The number is shown to the operator, so it
        # has to mean how much of the RECORDING was ingested.
        st.seconds = max(st.seconds, end)
        if text:
            st.chars += len(text)
            # Stamp the breadcrumb on EVERY piece so the embedded text itself
            # carries the timestamp, not just the metadata around it.
            for piece in semantic_split_text(text, chunk_size, chunk_overlap):
                piece = piece.strip()
                if piece:
                    st.chunks += 1
                    yield f"{crumb}\n{piece}"
        start += step


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
    is one batch, not one recording. Returns the stats; raises only on a fatal
    condition (unreadable file, embedding failure), never on one bad window.
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
        if progress:
            try:
                progress(stats)
            except Exception:  # noqa: BLE001 — progress must never break ingest
                pass

    for chunk in iter_audio_chunks(file_path, filename, stats=stats, **kwargs):
        batch.append(chunk)
        if len(batch) >= BATCH_CHUNKS:
            _flush()
    _flush()

    logger.info("audio ingest %s: %d windows, %.1f min, %d chunks, %d skipped",
                filename, stats.windows, stats.seconds / 60, stats.chunks,
                stats.skipped_windows)
    return stats
