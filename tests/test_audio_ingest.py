"""Unit tests for memory/audio_ingest.py — long-form audio → knowledge base.

Hermetic: ffmpeg/ffprobe and the audio node are all stubbed, so these run
anywhere. The live end-to-end path was verified separately against nova
(6.3-minute recording → 3 overlapping windows → 8 chunks, codewords from the
start, middle and very end all recovered).

The behaviours pinned here are the ones whose failure would be SILENT:
a window that vanishes from a transcript, a thinking-token exhaustion that
looks like silence, or a breadcrumb that stops being citable.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
for p in (str(_ROOT), str(_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from ghost_agent.memory import audio_ingest as ai  # noqa: E402


def _post_returning(content, finish_reason="stop"):
    def _post(url, payload, timeout):
        return {"choices": [{"message": {"content": content},
                             "finish_reason": finish_reason}]}
    return _post


# ── timestamps ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("secs,expected", [
    (0, "0:00"), (5, "0:05"), (65, "1:05"), (754.2, "12:34"),
    (3600, "1:00:00"), (3754, "1:02:34"),
])
def test_format_timestamp(secs, expected):
    assert ai.format_timestamp(secs) == expected


# ── windowing ────────────────────────────────────────────────────────────

def _run_chunks(duration, **kw):
    """Drive iter_audio_chunks over a stubbed recording of `duration` secs."""
    stats = ai.AudioIngestStats()
    with patch.object(ai, "probe_duration_seconds", return_value=duration), \
         patch.object(ai, "extract_window_wav", return_value=b"WAVE"):
        chunks = list(ai.iter_audio_chunks(
            Path("talk.mp3"), "talk.mp3", stats=stats,
            post_fn=_post_returning("spoken words here"), **kw))
    return chunks, stats


def test_windows_overlap_so_a_sentence_cannot_fall_through_a_seam():
    chunks, stats = _run_chunks(600, window_seconds=300, window_overlap=15)
    crumbs = [c.split("\n")[0] for c in chunks]
    # step = 300 - 15 = 285 → windows start at 0, 285, 570
    assert "[0:00–5:00]" in crumbs[0]
    assert any("[4:45–9:45]" in c for c in crumbs), "second window must start at the seam"
    assert stats.windows == 3


def test_single_short_recording_is_one_window():
    chunks, stats = _run_chunks(90, window_seconds=720)
    assert stats.windows == 1
    assert "[0:00–1:30]" in chunks[0]


def test_every_chunk_carries_a_citable_breadcrumb():
    chunks, _ = _run_chunks(300, window_seconds=300)
    assert chunks
    for c in chunks:
        head = c.split("\n")[0]
        assert head.startswith("[talk.mp3] ["), f"missing breadcrumb: {head!r}"
        assert "–" in head, "breadcrumb must carry a timestamp RANGE"


def test_duration_cap_truncates_rather_than_running_forever():
    _, stats = _run_chunks(10_000, window_seconds=300, max_seconds=600)
    assert stats.truncated is True
    assert stats.seconds <= 600 + 1


def test_tail_sliver_is_not_transcribed():
    """A sub-second remainder carries nothing and must not cost a node call."""
    _, stats = _run_chunks(300.2, window_seconds=300, window_overlap=0)
    assert stats.windows == 1


# ── failure isolation ────────────────────────────────────────────────────

def test_one_bad_window_is_skipped_not_fatal():
    """A failed window must not sink the whole recording — the same policy
    pdf_ingest applies to an unreadable page."""
    calls = {"n": 0}

    def _flaky(url, payload, timeout):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("node exploded")
        return {"choices": [{"message": {"content": "good text"},
                             "finish_reason": "stop"}]}

    stats = ai.AudioIngestStats()
    with patch.object(ai, "probe_duration_seconds", return_value=900), \
         patch.object(ai, "extract_window_wav", return_value=b"WAVE"):
        chunks = list(ai.iter_audio_chunks(Path("t.mp3"), "t.mp3", stats=stats,
                                           window_seconds=300, window_overlap=0,
                                           post_fn=_flaky))
    assert stats.skipped_windows == 1
    assert stats.windows == 2
    assert chunks, "surviving windows must still produce chunks"
    assert any("node exploded" in e for e in stats.errors)


def test_thinking_token_exhaustion_raises_instead_of_dropping_a_window():
    """Empty content + finish_reason=length is the measured silent-failure
    shape; treating it as silence would delete a whole window of speech."""
    with pytest.raises(RuntimeError) as exc:
        ai.transcribe_window(b"WAVE", post_fn=_post_returning("", "length"))
    assert "thinking" in str(exc.value).lower()
    assert "GHOST_AUDIO_MAX_TOKENS" in str(exc.value)


def test_no_speech_sentinel_is_empty_not_an_error():
    assert ai.transcribe_window(b"WAVE", post_fn=_post_returning("(no speech)")) == ""


def test_malformed_node_response_raises():
    with pytest.raises(RuntimeError):
        ai.transcribe_window(b"WAVE", post_fn=lambda u, p, t: {"nope": 1})


# ── dialect guard ────────────────────────────────────────────────────────

def test_payload_uses_input_audio_dialect():
    """The node rejects the vision-style audio_url shape with HTTP 400."""
    seen = {}

    def _capture(url, payload, timeout):
        seen.update(payload)
        return {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}

    ai.transcribe_window(b"WAVE", post_fn=_capture)
    kinds = [p["type"] for p in seen["messages"][0]["content"]]
    assert "input_audio" in kinds
    assert "audio_url" not in kinds
    assert seen["max_tokens"] >= 512, "must clear the thinking-token budget"


# ── streaming into the store ─────────────────────────────────────────────

def test_streaming_batches_into_ingest_document():
    mem = MagicMock()
    mem.ingest_document.return_value = (True, "ok")
    with patch.object(ai, "probe_duration_seconds", return_value=600), \
         patch.object(ai, "extract_window_wav", return_value=b"WAVE"):
        stats = ai.ingest_audio_streaming(
            Path("t.mp3"), "t.mp3", mem,
            window_seconds=300, window_overlap=0,
            post_fn=_post_returning("some transcribed speech"))
    assert stats.chunks > 0
    assert mem.ingest_document.called
    name, batch = mem.ingest_document.call_args[0][:2]
    assert name == "t.mp3"
    assert all(c.startswith("[t.mp3] [") for c in batch)


def test_embedding_failure_is_fatal():
    """A store that refuses chunks must raise — silently returning stats
    would report success for a document that was never indexed."""
    mem = MagicMock()
    mem.ingest_document.return_value = (False, "embedder down")
    with patch.object(ai, "probe_duration_seconds", return_value=300), \
         patch.object(ai, "extract_window_wav", return_value=b"WAVE"):
        with pytest.raises(RuntimeError, match="embedding failed"):
            ai.ingest_audio_streaming(
                Path("t.mp3"), "t.mp3", mem, window_seconds=300,
                post_fn=_post_returning("words"))


# ── wiring guard ─────────────────────────────────────────────────────────

def test_audio_is_routed_before_the_plain_text_branch():
    """Regression guard: audio must be claimed by the transcription path. If
    this route is ever removed, a .wav silently decodes to replacement-char
    noise in text memory instead of failing."""
    src = (_SRC / "ghost_agent" / "tools" / "memory.py").read_text()
    assert "_AUDIO_INGEST_EXTS" in src
    assert "ingest_audio_streaming" in src
    audio_idx = src.find("_AUDIO_INGEST_EXTS)")
    text_idx = src.find("def _extract_text")
    assert audio_idx != -1 and text_idx != -1
    assert audio_idx < text_idx, "audio branch must precede the plain-text branch"


def test_binaries_resolve_without_a_usable_path(tmp_path, monkeypatch):
    """Same launchd-PATH trap that 503'd the interface's STT: the agent also
    runs under a LaunchDaemon, so ffmpeg/ffprobe in /opt/homebrew/bin are
    invisible to a bare lookup. This path had the identical latent bug — it
    simply had not been exercised yet."""
    fake = tmp_path / "ffprobe"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o755)

    monkeypatch.setenv("PATH", "/nonexistent")
    monkeypatch.delenv("GHOST_FFPROBE_BIN", raising=False)
    monkeypatch.setattr(ai, "_BIN_PREFIXES", (str(tmp_path),))

    assert ai.resolve_binary("ffprobe") == str(fake)


def test_missing_binary_error_names_the_launchd_cause(monkeypatch):
    monkeypatch.setattr(ai, "resolve_binary", lambda name: None)
    with pytest.raises(RuntimeError) as exc:
        ai._run(["ffmpeg", "-version"], timeout=5)
    assert "launchd" in str(exc.value) and "GHOST_FFMPEG_BIN" in str(exc.value)


def test_large_recordings_are_exempt_from_the_byte_cap():
    """A 45-minute talk video is 700 MB–1.3 GB — far over the 100 MB ingest
    cap that guards RAM for text/PDF. Audio is streamed window-by-window and
    never resident, so the cap must not apply or the primary use case fails at
    the door with advice ("split it into chunks") that makes no sense for a
    recording. Duration is the real bound, enforced in audio_ingest."""
    src = (_SRC / "ghost_agent" / "tools" / "memory.py").read_text()
    assert "is_audio_video" in src
    assert "MAX_INGEST_FILE_BYTES and not is_audio_video" in src, (
        "the byte cap must skip audio/video")
    # The exemption must be evaluated where the cap is, i.e. BEFORE the branch
    # that would otherwise never be reached.
    assert src.find("is_audio_video =") < src.find("ingest_audio_streaming")


def test_video_containers_are_claimed_too():
    """A recorded talk is usually an .mp4; ffmpeg takes the audio track."""
    for ext in (".mp4", ".mov", ".mkv", ".webm"):
        assert ext in ai_exts(), f"{ext} should route to audio ingest"


def ai_exts():
    from ghost_agent.tools import memory as memtool
    return memtool._AUDIO_INGEST_EXTS
