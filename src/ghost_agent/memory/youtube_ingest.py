"""YouTube → knowledge base, over Tor, in-process (§4KE, 2026-09-24).

The route a YouTube link takes when it reaches
``knowledge_base(action='transcribe', filename='<url>')``:

1. **Resolve** the video on a fresh Tor circuit with yt-dlp, asking the local
   PO-token helper for a proof-of-origin token. Without that token YouTube
   answers every Tor exit with "Sign in to confirm you're not a bot"
   (measured 0/9 circuits on 2026-09-24); with it roughly 4 in 10 circuits
   pass and the rest fail in 2–9 s, so a handful of attempts on distinct
   circuits (SOCKS-auth isolation, the same trick ``tools/search.py`` uses)
   is what makes the route reliable. The helper's own BotGuard fetch rides
   the SAME circuit — yt-dlp forwards the per-attempt proxy to it.
2. **Captions tier.** When the video carries subtitles (manual first, then
   the ORIGINAL-language automatic track), fetch that one track as json3 and
   index it. Exact words, real timestamps, no audio node involved, seconds
   instead of minutes.
3. **Audio tier.** Otherwise download the audio track into the sandbox as
   ``yt-<id>.<ext>`` and hand it to :mod:`audio_ingest` (windowed
   transcription on the private audio node).

Either way the document is keyed by the VIDEO ID (``yt-<id>.…``), never by
a constant name — the previous macro wrote every video to ``yt_audio.m4a``
and the knowledge base's name-keyed dedup then answered every later video
from the first one. The ordered transcript is kept beside the store
(``set_document_text``) so ``action='transcript'`` can page through it in
order, the outline record lists one entry per passage, and the summary row
carries the title and the first words, not just counts.

Why this is not a shell step in a composed skill any more: the sandbox
``execute`` tool promotes anything still running at 90 s to a background
job, and a sequential macro books that as a failed step — the old
``youtube_transcribe`` macro could not complete by construction (live: 3
uses, 0 successes). Here the tool owns its whole duration.

Everything egresses through Tor (``socks5h`` — DNS through the proxy). No
cookies, no accounts, no keyed APIs: the no-identity-egress rule stands.

Overrides (env):
  GHOST_YT_POT_URL         PO-token helper (default http://127.0.0.1:4416)
  GHOST_YT_POT_DIR         its install dir, for the on-demand spawn fallback
  GHOST_YT_ATTEMPTS        circuits to try per stage (default 8)
  GHOST_YT_INFO_TIMEOUT_S  per-attempt budget for resolve/captions (75)
  GHOST_YT_AUDIO_TIMEOUT_S per-attempt budget for the audio download (900)
  GHOST_YT_PASSAGE_S       caption passage length in seconds (90)
  GHOST_YT_POT_LOG         log file for the on-demand helper spawn
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from ..utils.helpers import env_positive, socks_url_with_identity
from .audio_ingest import (
    MAX_AUDIO_SECONDS, format_timestamp, ingest_audio_streaming, resolve_binary,
)

logger = logging.getLogger("GhostAgent")   # §4NV: __name__ reached neither the stream nor the durable log

POT_URL = os.environ.get("GHOST_YT_POT_URL", "http://127.0.0.1:4416").rstrip("/")
POT_DIR = Path(os.environ.get("GHOST_YT_POT_DIR",
                              str(Path.home() / "Data" / "AI" / "PotProvider")))
ATTEMPTS = int(env_positive("GHOST_YT_ATTEMPTS", 8))
#: §4PI (R1-M1): the chosen caption track is tried on the circuit that resolved
#: the video, then on fresh circuits — it used to be ONE try, and any 429 on
#: that circuit demoted a captioned video to the audio tier
CAPTION_ATTEMPTS = int(env_positive("GHOST_YT_CAPTION_ATTEMPTS", 3))
#: §4PI (R1-M4): a wall clock over the FETCH stages (resolve + captions + audio
#: download) — 8 × 75 s + 8 × 900 s used to be the only bound on the owner's turn
FETCH_BUDGET_S = env_positive("GHOST_YT_FETCH_BUDGET_S", 1200.0)
#: §4PI (R3-M2): the helper is launchd's (com.local.ghost-pot); the in-process
#: spawn is for a dev shell ONLY — opt in. Spawned beside the daemon it took
#: port 4416 and launchd's child died EADDRINUSE every 15 s (seen 2026-09-24).
POT_SPAWN = os.environ.get("GHOST_YT_POT_SPAWN", "").strip().lower() in ("1", "true", "yes")
#: §4PI (R3-M3): the environment yt-dlp / its plugins / deno run with — an
#: allowlist, never the agent's own (which carries GHOST_API_KEY)
_ENV_KEEP = ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "TERM", "SSL_CERT_FILE",
             "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "DENO_DIR", "NO_COLOR")
INFO_TIMEOUT_S = env_positive("GHOST_YT_INFO_TIMEOUT_S", 75.0)
AUDIO_TIMEOUT_S = env_positive("GHOST_YT_AUDIO_TIMEOUT_S", 900.0)
PASSAGE_SECONDS = env_positive("GHOST_YT_PASSAGE_S", 90.0)
PASSAGE_MAX_CHARS = 1200          # matches audio_ingest.CHUNK_SIZE
PREVIEW_CHARS = 3000              # transcript preview returned to the model
SUMMARY_PREVIEW_CHARS = 600       # first words carried by the summary row
BATCH_CHUNKS = 256
#: What yt-dlp's ``ba[ext=m4a]/ba`` can leave in the sandbox.
_AUDIO_EXTS = ("m4a", "webm", "opus", "mp3", "ogg", "aac", "mp4")

_YT_HOSTS = {
    "youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com",
    "youtu.be", "www.youtu.be", "youtube-nocookie.com", "www.youtube-nocookie.com",
}
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_PATH_ID_RE = re.compile(r"^/(?:shorts|live|embed|v|e)/([A-Za-z0-9_-]{11})(?:[/?#]|$)")

#: Faults a DIFFERENT circuit may not hit (bot wall, rate limit, a SABR-only
#: session, transport errors). Everything else stops the loop at once.
_RETRYABLE_RE = re.compile(
    r"sign in to confirm|not a bot|\b429\b|too many requests|http error 403|forbidden"
    r"|requested format is not available|unable to download (?:api page|webpage|json metadata)"
    r"|timed out|timeout|connection (?:reset|refused|aborted)|remote end closed"
    r"|temporarily|page needs to be reloaded|proxy|tunnel|\bssl\b|\btls\b|socket|got error"
    # Under Tor the "country" is the EXIT's, and a format list is per session
    # (SABR): both are circuit faults, not video faults.
    r"|blocked it in your country|available in your country|no video formats found",
    re.I,
)
#: A wall no exit rotation can pass: the video, not the network.
_TERMINAL_RE = re.compile(
    r"private video|has been removed|does not exist|is not a valid url|unsupported url"
    r"|members-only|join this channel|confirm your age|age-restricted|premieres in"
    r"|this live event|account associated with this video has been terminated",
    re.I,
)
#: The dead giveaway that the PO-token helper was not consulted.
_BOT_WALL_RE = re.compile(r"sign in to confirm|not a bot", re.I)


# ── URL handling ────────────────────────────────────────────────────────────

def is_youtube_url(url) -> bool:
    """True for a YouTube watch/short/live/embed URL or a youtu.be link."""
    return youtube_video_id(url) is not None


def youtube_video_id(url) -> Optional[str]:
    """The 11-character video id, or ``None`` when ``url`` is not a YouTube
    video URL (channel and playlist pages are not videos)."""
    if not isinstance(url, str):
        return None
    s = url.strip()
    if not s:
        return None
    if "://" not in s:
        s = "https://" + s
    try:
        p = urlparse(s)
    except ValueError:
        return None
    host = (p.hostname or "").lower()
    if host not in _YT_HOSTS:
        return None
    if host.endswith("youtu.be"):
        seg = p.path.strip("/").split("/")[0]
        return seg if _ID_RE.match(seg) else None
    if p.path in ("/watch", "/watch/"):
        v = (parse_qs(p.query).get("v") or [""])[0]
        return v if _ID_RE.match(v) else None
    m = _PATH_ID_RE.match(p.path)
    return m.group(1) if m else None


def document_stem(video_id: str) -> str:
    return f"yt-{video_id}"


def existing_document_for(video_id: str, library) -> Optional[str]:
    """The knowledge-base name already holding this video, if any — the
    dedup key is the VIDEO, whichever tier ingested it."""
    stem = document_stem(video_id) + "."
    for name in library or ():
        if str(name).startswith(stem):
            return str(name)
    return None


def _socks5h(proxy: Optional[str]) -> Optional[str]:
    """Force remote DNS: ``socks5://`` resolves names LOCALLY, which leaks
    the video host lookup outside Tor."""
    if not proxy:
        return proxy
    low = proxy.lower()
    if low.startswith("socks5://"):
        return "socks5h://" + proxy[len("socks5://"):]
    if low.startswith("socks4://"):
        return "socks4a://" + proxy[len("socks4://"):]
    return proxy


def circuit_proxy(tor_proxy: str, tag: str) -> str:
    """A ``socks5h`` proxy URL carrying a per-attempt identity so Tor's
    ``IsolateSOCKSAuth`` hands this attempt its own circuit. §4PI (R1-m3): a
    configured proxy that already carries credentials is returned UNCHANGED
    by the shared helper — every "fresh circuit" would be the same one — so
    the configured identity is replaced by the attempt's."""
    base = str(tor_proxy or "")
    try:
        from urllib.parse import urlunparse
        p = urlparse(base)
        if p.hostname and p.username:
            _hostport = p.netloc.rsplit("@", 1)[-1]          # r2 (reader m5): keeps IPv6 brackets
            base = urlunparse((p.scheme, _hostport, p.path, p.params, p.query, p.fragment))
    except Exception:  # noqa: BLE001
        pass
    return _socks5h(socks_url_with_identity(base, tag))


# ── the PO-token helper ─────────────────────────────────────────────────────

def pot_ping(timeout: float = 3.0) -> Optional[str]:
    """The helper's version string when it answers ``/ping``, else ``None``."""
    import httpx

    try:
        r = httpx.get(f"{POT_URL}/ping", timeout=timeout)
        if r.status_code == 200:
            try:
                return str(r.json().get("version") or "ok")
            except ValueError:
                return "ok"
    except Exception:  # noqa: BLE001 — a down helper is a normal state
        return None
    return None


def ensure_pot_server(*, spawn: Optional[bool] = None, wait_s: float = 10.0) -> bool:
    """True when the helper answers. When it does not and ``spawn`` is set
    (default: ``GHOST_YT_POT_SPAWN``, off), start it from ``POT_DIR`` — a
    dev-shell fallback only: in production ``com.local.ghost-pot`` owns the
    port and a second helper makes launchd's child crash-loop (§4PI R3-M2)."""
    if pot_ping():
        return True
    if spawn is None:
        spawn = POT_SPAWN
    if not spawn:
        return False
    main_js = POT_DIR / "build" / "main.js"
    node = resolve_binary("node")
    if not main_js.exists() or not node:
        logger.warning("PO-token helper down and not spawnable (main.js=%s node=%s)",
                       main_js.exists(), bool(node))
        return False
    port = str(urlparse(POT_URL).port or 4416)
    try:
        log_path = Path(os.environ.get("GHOST_YT_POT_LOG",
                                       str(Path.home() / "Data" / "AI" / "Logs" / "ghost-pot.log")))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "ab") as log:
            subprocess.Popen(
                [node, str(main_js), "--port", port, "--host", "127.0.0.1"],
                cwd=str(POT_DIR), stdout=log, stderr=log,
                stdin=subprocess.DEVNULL, start_new_session=True,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not spawn the PO-token helper: %s", exc)
        return False
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if pot_ping(1.0):
            return True
        time.sleep(0.5)
    return False


# ── yt-dlp plumbing ─────────────────────────────────────────────────────────

RunFn = Callable[[List[str], float, Optional[str]], Tuple[int, str, str]]


def _subprocess_env() -> Dict[str, str]:
    """§4PI (R3-M3): yt-dlp, its plugins and deno are third-party code that
    the update script pip-upgrades over the clear; they get an allowlisted
    environment, never the agent's (which carries GHOST_API_KEY)."""
    return {k: v for k, v in os.environ.items() if k in _ENV_KEEP}


class _StoppedByUser(Exception):
    pass


def _communicate_stoppable(proc, timeout: float):
    """`proc.communicate(timeout)` that a §4RZ step-5 owner tool stop ends
    early (polled each second; retrying communicate loses no output). With
    no stop bound it is the plain call."""
    from ..utils import tool_stop
    if tool_stop.current() is None:
        return proc.communicate(timeout=timeout)
    import time as _time
    deadline = _time.monotonic() + timeout
    while True:
        if tool_stop.requested():
            raise _StoppedByUser()
        left = deadline - _time.monotonic()
        if left <= 0:
            raise subprocess.TimeoutExpired(proc.args, timeout)
        try:
            return proc.communicate(timeout=min(1.0, left))
        except subprocess.TimeoutExpired:
            continue


def _run_subprocess(cmd: List[str], timeout: float, cwd: Optional[str]) -> Tuple[int, str, str]:
    """Run yt-dlp in its own session so a timeout kills the whole tree."""
    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL, start_new_session=True, env=_subprocess_env(),
    )
    from ..utils import tool_stop as _ts
    _armed = _ts.arm("knowledge_base")          # §4RZ step 5: a stop may still kill it
    try:
        return _run_and_wait(proc, timeout)
    finally:
        _ts.disarm(_armed, "knowledge_base")


def _run_and_wait(proc, timeout: float) -> Tuple[int, str, str]:
    try:
        out, err = _communicate_stoppable(proc, timeout)
    except _StoppedByUser:
        try:
            os.killpg(proc.pid, 9)
        except Exception:  # noqa: BLE001
            proc.kill()
        out, err = proc.communicate()
        try:
            from ..utils.tool_stop import note_killed
            note_killed("knowledge_base")
        except Exception:  # noqa: BLE001
            pass
        return 130, out.decode("utf-8", "replace"), (
            err.decode("utf-8", "replace") + "\nERROR: stopped by the user")
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, 9)
        except Exception:  # noqa: BLE001
            proc.kill()
        out, err = proc.communicate()
        return 124, out.decode("utf-8", "replace"), (
            err.decode("utf-8", "replace") + f"\nERROR: timed out after {timeout:.0f}s")
    return proc.returncode, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")


def ytdlp_base_command(proxy: str) -> List[str]:
    """yt-dlp as a module of THIS interpreter (the venv carries yt-dlp and
    the PO-token plugin), Tor proxy, PO-token helper, deno for the JS
    challenge. ``--no-playlist``: a watch URL inside a playlist is one video."""
    cmd = [
        sys.executable, "-m", "yt_dlp", "--no-playlist", "--no-progress",
        "--socket-timeout", "30", "--proxy", proxy,
        "--extractor-args", f"youtubepot-bgutilhttp:base_url={POT_URL}",
    ]
    deno = resolve_binary("deno")
    if deno:
        cmd += ["--js-runtimes", f"deno:{deno}"]
    return cmd


#: §4PI (R1-m4): faults of the STACK, read from the whole stderr (they are
#: WARNING lines above the final ERROR): no circuit changes them, and the
#: report must name them instead of "format not available" eight times
_INFRA_RES = (
    # §4RR: ONLY yt-dlp's missing-runtime line ("No supported JavaScript runtime
    # could be found"). The bare "javascript runtime" also matched the warning
    # yt-dlp 2026.10 prints on EVERY fetch ("n challenge solving failed …
    # Ensure you have a supported JavaScript runtime …"), so one bot-walled
    # circuit ended the whole fetch as "repair the stack" instead of rotating.
    (re.compile(r"no supported javascript runtime", re.I), "terminal",
     "no JavaScript runtime (deno) for yt-dlp — repair the stack (bin/update-youtube-stack.sh --verify)"),
    (re.compile(r"bgutil|pot server|pot provider|\bpo token|potoken", re.I), "retry",
     "PO-token helper error"),
)


def classify_failure(stderr: str) -> Tuple[str, str]:
    """``("terminal" | "retry", reason)`` from yt-dlp's stderr. The reason is
    the last ERROR line (or the last line) so the report names the cause."""
    lines = [ln.strip() for ln in (stderr or "").splitlines() if ln.strip()]
    errors = [ln for ln in lines if ln.startswith("ERROR")]
    reason = (errors[-1] if errors else (lines[-1] if lines else "yt-dlp failed"))[:300]
    text = "\n".join(errors) if errors else "\n".join(lines[-5:])
    if _TERMINAL_RE.search(text):
        return "terminal", reason        # r2 (reader M3): a private/removed video is terminal
                                         # whatever the plugin warned above it
    for rx, kind, label in _INFRA_RES:
        if rx.search(stderr or ""):
            return kind, f"{label}: {reason}"[:300]
    if _RETRYABLE_RE.search(text):
        return "retry", reason
    return "terminal", reason


@dataclass
class FetchStats:
    attempts: int = 0
    errors: List[str] = field(default_factory=list)
    bot_walls: int = 0
    stage: str = ""              # the stage the last rotation ran
    stage_attempts: int = 0      # …and its own counts, so a clean resolve
    stage_walls: int = 0         # cannot hide an all-walled download
    remaining_s: float = 0.0     # §4PI: seconds left of the fetch budget (0 = unbounded)


def _attempt_tag(video_id: str, stage: str, i: int) -> str:
    return f"yt{stage}{video_id}{i}{random.randint(0, 999999)}"


def _rotate(video_id: str, stage: str, tor_proxy: str, attempts: int,
            do: Callable[[str], Tuple[bool, str]], stats: FetchStats,
            progress: Optional[Callable[[str], None]] = None,
            deadline: Optional[float] = None):
    """Run ``do(proxy)`` on up to ``attempts`` fresh circuits. ``do`` returns
    ``(ok, stderr_or_result)``; a terminal fault stops early. ``deadline``
    (a monotonic instant, §4PI R1-M4) stops the rotation when the fetch
    budget is spent — ``do`` may read the remaining seconds from
    ``stats.remaining_s`` to shorten its own timeout."""
    last = ""
    stats.stage, stats.stage_attempts, stats.stage_walls = stage, 0, 0
    for i in range(1, attempts + 1):
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining < 5.0:
                reason = f"fetch budget of {FETCH_BUDGET_S:.0f}s exhausted after {i - 1} circuit(s) of the {stage} stage"
                stats.errors.append(f"{stage}: {reason}")
                return False, (last or reason)
            stats.remaining_s = remaining
        proxy = circuit_proxy(tor_proxy, _attempt_tag(video_id, stage, i))
        stats.attempts += 1
        stats.stage_attempts += 1
        ok, payload = do(proxy)
        if ok:
            return True, payload
        kind, reason = classify_failure(payload)
        if _BOT_WALL_RE.search(payload or ""):
            stats.bot_walls += 1
            stats.stage_walls += 1
        stats.errors.append(f"{stage} attempt {i}: {reason}")
        last = reason
        if progress:
            progress(f"{stage}: circuit {i}/{attempts} refused ({reason[:80]})"
                     + (" — rotating" if kind == "retry" and i < attempts else ""))
        if kind == "terminal":
            return False, reason
    return False, last


# ── captions ────────────────────────────────────────────────────────────────

def _base_lang(code: str) -> str:
    """``en-US`` → ``en``; ``el-orig`` → ``el``; ``zh-Hans`` stays ``zh-hans``
    (a script suffix is a different language for a reader, a region is not)."""
    c = str(code or "").lower()
    if c.endswith("-orig"):
        c = c[:-5]
    parts = c.split("-")
    if len(parts) > 1 and len(parts[1]) == 4:      # script tag: zh-Hans / sr-Latn
        return "-".join(parts[:2])
    return parts[0]


def _manual_match(subs: Dict[str, list], want: str) -> Optional[str]:
    """Exact key first, then the same base language (``en`` ↔ ``en-US``)."""
    for k in subs:
        if k.lower() == want.lower():
            return k
    for k in subs:
        if _base_lang(k) == _base_lang(want):
            return k
    return None


def choose_caption_track(info: dict, requested: Optional[str] = None) -> Optional[Tuple[str, str]]:
    """``(lang_key, kind)`` with kind ``manual`` or ``auto`` — the track to
    fetch, or ``None`` when the video has no usable captions.

    A REQUESTED language is honoured only by a manual track or the original
    automatic track in that language — never by one of the automatic
    machine TRANSLATIONS yt-dlp also lists (the registry promises as much),
    so an English request against a Greek talk falls through to the talk's
    own language and the reply says so. Then the video's own language
    (manual, ``<lang>-orig``, an exact automatic key), any manual track, any
    original automatic track. ``live_chat`` is not a caption."""
    subs = {k: v for k, v in (info.get("subtitles") or {}).items() if k != "live_chat" and v}
    auto = {k: v for k, v in (info.get("automatic_captions") or {}).items() if v}
    origs = {_base_lang(k): k for k in auto if k.lower().endswith("-orig")}
    if requested:
        w = str(requested)
        k = _manual_match(subs, w)
        if k:
            return k, "manual"
        if _base_lang(w) in origs:
            return origs[_base_lang(w)], "auto"
    lang = str(info.get("language") or "")
    if lang:
        k = _manual_match(subs, lang)
        if k:
            return k, "manual"
        if _base_lang(lang) in origs:
            return origs[_base_lang(lang)], "auto"
        for k in auto:
            if k.lower() == lang.lower() or _base_lang(k) == _base_lang(lang) and not k.lower().endswith("-orig"):
                # the video's OWN language is never a translation
                return k, "auto"
    if subs:
        return next(iter(subs)), "manual"
    if origs:
        return next(iter(origs.values())), "auto"
    return None


def caption_language_note(requested: Optional[str], chosen: Optional[Tuple[str, str]]) -> str:
    """The sentence the reply carries when a requested language could not be
    honoured without a machine translation."""
    if not requested or not chosen:
        return ""
    if _base_lang(chosen[0]) == _base_lang(requested):
        return ""
    return (f" Requested '{requested}' captions exist only as a machine translation, so the "
            f"original '{chosen[0].replace('-orig', '')}' track was used.")


def parse_json3(text: str) -> List[Tuple[float, float, str]]:
    """YouTube json3 → ``[(start_s, end_s, text)]``. Word-level ``segs`` are
    joined per event; events without text (window markers) are dropped."""
    try:
        data = json.loads(text)
    except ValueError:
        return []
    out: List[Tuple[float, float, str]] = []
    for ev in data.get("events") or []:
        segs = ev.get("segs") or []
        txt = "".join(str(s.get("utf8", "")) for s in segs)
        txt = re.sub(r"\s+", " ", txt).strip()
        if not txt:
            continue
        start = float(ev.get("tStartMs", 0)) / 1000.0
        dur = float(ev.get("dDurationMs", 0)) / 1000.0
        out.append((start, start + max(dur, 0.0), txt))
    return out


_VTT_TIME_RE = re.compile(
    r"(?:(\d+):)?(\d{2}):(\d{2})[.,](\d{3})\s+-->\s+(?:(\d+):)?(\d{2}):(\d{2})[.,](\d{3})")


def _vtt_secs(h, m, s, ms) -> float:
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_vtt(text: str) -> List[Tuple[float, float, str]]:
    """WebVTT → ``[(start_s, end_s, text)]`` with tags stripped and the
    rolling duplicate lines automatic captions repeat collapsed."""
    out: List[Tuple[float, float, str]] = []
    cur: Optional[Tuple[float, float]] = None
    buf: List[str] = []
    last_line = ""

    def flush():
        nonlocal buf, cur
        if cur and buf:
            txt = re.sub(r"\s+", " ", " ".join(buf)).strip()
            if txt:
                out.append((cur[0], cur[1], txt))
        buf, cur = [], None

    for raw in text.splitlines():
        line = raw.strip()
        m = _VTT_TIME_RE.search(line)
        if m:
            flush()
            g = m.groups()
            cur = (_vtt_secs(*g[:4]), _vtt_secs(*g[4:]))
            continue
        if not line or line.startswith(("WEBVTT", "NOTE", "Kind:", "Language:")) or line.isdigit():
            if not line:
                flush()
            continue
        if cur is None:
            continue
        clean = re.sub(r"<[^>]+>", "", line).replace("&nbsp;", " ").strip()
        if clean and clean != last_line:
            buf.append(clean)
            last_line = clean
    flush()
    return out


def segments_to_passages(segments: List[Tuple[float, float, str]], *,
                         passage_seconds: float = PASSAGE_SECONDS,
                         max_chars: int = PASSAGE_MAX_CHARS) -> List[Tuple[float, float, str]]:
    """Group caption segments into ~90-second passages (or ``max_chars``),
    the retrieval unit — a single caption line is too small to embed and a
    whole talk is too big."""
    passages: List[Tuple[float, float, str]] = []
    start = None
    end = 0.0
    parts: List[str] = []
    size = 0
    for s, e, t in segments:
        if start is None:
            start = s
        if parts and (e - start >= passage_seconds or size + len(t) + 1 > max_chars):
            passages.append((start, end, " ".join(parts)))
            start, parts, size = s, [], 0
        parts.append(t)
        size += len(t) + 1
        end = max(end, e)
    if parts and start is not None:
        passages.append((start, end, " ".join(parts)))
    return passages


def passage_chunk(filename: str, start: float, end: float, text: str) -> str:
    """The stored form — the same ``[file] [h:mm:ss–h:mm:ss]`` breadcrumb
    the audio path stamps, so retrieval and citation behave identically."""
    return f"[{filename}] [{format_timestamp(start)}–{format_timestamp(end)}]\n{text}"


def render_transcript(passages: List[Tuple[float, float, str]]) -> str:
    return "\n".join(f"[{format_timestamp(s)}–{format_timestamp(e)}] {t}" for s, e, t in passages)


# ── the route ───────────────────────────────────────────────────────────────

@dataclass
class YoutubeIngestResult:
    ok: bool
    message: str
    video_id: str = ""
    title: str = ""
    filename: str = ""
    route: str = ""              # captions | audio
    language: str = ""
    duration: float = 0.0
    passages: List[Tuple[float, float, str]] = field(default_factory=list)
    fetch: FetchStats = field(default_factory=FetchStats)


def _pot_hint(stats: FetchStats, pot_up: bool) -> str:
    if not pot_up:
        return (f" The PO-token helper at {POT_URL} is NOT running — without it YouTube "
                f"refuses every Tor exit ('not a bot'). Start com.local.ghost-pot.")
    if stats.stage_walls and stats.stage_walls == stats.stage_attempts:
        return (f" Every circuit of the {stats.stage} stage hit the bot wall even with a PO "
                f"token; YouTube may have rotated its challenge — update yt-dlp and the helper.")
    return ""


def fetch_info(url: str, video_id: str, tor_proxy: str, *, run: RunFn, stats: FetchStats,
               attempts: int = ATTEMPTS, timeout: float = INFO_TIMEOUT_S,
               progress=None) -> Tuple[Optional[dict], str, Optional[str]]:
    """Resolve the video: returns ``(info, reason, proxy_that_worked)``."""
    winner: Dict[str, str] = {}

    def do(proxy: str):
        rc, out, err = run(ytdlp_base_command(proxy) + ["--skip-download", "--dump-single-json", url],
                           _bounded(timeout, stats), None)
        if rc == 0 and out.strip():
            try:
                info = json.loads(out.strip().splitlines()[-1])
            except ValueError:
                return False, "ERROR: unparseable metadata from yt-dlp"
            winner["proxy"] = proxy
            return True, info
        return False, err

    ok, payload = _rotate(video_id, "resolve", tor_proxy, attempts, do, stats, progress,
                          deadline=getattr(stats, "deadline", None))
    if ok:
        return payload, "", winner.get("proxy")
    return None, str(payload), None


def _bounded(timeout: float, stats: FetchStats) -> float:
    """A per-attempt timeout that never outlives the fetch budget."""
    rem = float(getattr(stats, "remaining_s", 0.0) or 0.0)
    return max(5.0, min(timeout, rem)) if rem > 0 else timeout


def fetch_captions(url: str, video_id: str, lang_key: str, proxy: str, workdir: str, *,
                   run: RunFn, timeout: float = INFO_TIMEOUT_S) -> Tuple[bool, List[Tuple[float, float, str]], str]:
    """Fetch ONE caption track on the circuit that just resolved the video
    (one language per request — a second language on the same circuit drew
    a 429 in measurement). Returns ``(ok, segments, error)``."""
    out_tpl = os.path.join(workdir, document_stem(video_id))
    cmd = ytdlp_base_command(proxy) + [
        "--skip-download", "--write-subs", "--write-auto-subs",
        "--sub-langs", lang_key, "--sub-format", "json3/vtt/best",
        "-o", out_tpl, url,
    ]
    rc, _out, err = run(cmd, timeout, workdir)
    # yt-dlp names the track ``<tpl>.<lang>.<ext>``; read it through the
    # symlink-safe walker (the directory is a fresh tempdir, but the ratchet
    # in tests/test_4gi_symlink_class applies to every tree read in src/).
    from ..tools.file_system import read_text_nofollow, walk_nofollow
    stem = document_stem(video_id)
    chosen, text = "", ""
    for _dirpath, filenames, dir_fd in walk_nofollow(Path(workdir)):
        cands = sorted(f for f in filenames
                       if f.startswith(stem) and f.endswith((".json3", ".vtt")))
        cands.sort(key=lambda f: (not f.endswith(".json3"), f))
        if cands:
            chosen = cands[0]
            text = read_text_nofollow(chosen, dir_fd=dir_fd, encoding="utf-8", errors="replace")
        break  # the tempdir is flat; only its top level matters
    if not chosen:
        return False, [], (err.strip().splitlines()[-1] if err.strip() else f"no caption file written (rc={rc})")
    try:
        segs = parse_json3(text) if chosen.endswith(".json3") else parse_vtt(text)
    except Exception as exc:  # noqa: BLE001 — §4PI (R1-m1): a malformed track is a caption failure, not a route crash
        return False, [], f"unparseable caption file ({type(exc).__name__}: {exc})"[:200]
    if not segs:
        return False, [], "caption track was empty"
    return True, segs, ""


def fetch_captions_rotating(url: str, video_id: str, lang_key: str, first_proxy: str, tor_proxy: str, *,
                            run: RunFn, stats: FetchStats, attempts: int = CAPTION_ATTEMPTS,
                            progress=None, timeout: float = INFO_TIMEOUT_S) -> Tuple[bool, List[Tuple[float, float, str]], str]:
    """§4PI (R1-M1): the track on the circuit that resolved the video first
    (the measured shape), then fresh circuits — a 429 on that one circuit
    used to demote a captioned video to the audio tier."""
    last = ""
    stats.stage, stats.stage_attempts, stats.stage_walls = "captions", 0, 0
    for i in range(max(1, attempts)):
        _dl = float(getattr(stats, "deadline", 0) or 0)
        if _dl:
            stats.remaining_s = max(0.0, _dl - time.monotonic())      # r2 (reader m1): refreshed per attempt
            if i and stats.remaining_s < 5.0:
                stats.errors.append("captions: fetch budget exhausted")
                break
        proxy = first_proxy if i == 0 else circuit_proxy(tor_proxy, _attempt_tag(video_id, "captions", i))
        stats.stage_attempts += 1
        if i:
            stats.attempts += 1
        with tempfile.TemporaryDirectory(prefix="ghost-yt-") as td:
            ok, segs, err = fetch_captions(url, video_id, lang_key, proxy, td, run=run,
                                           timeout=_bounded(timeout, stats))
        if ok:
            return True, segs, ""
        last = err
        stats.errors.append(f"captions attempt {i + 1}: {err}")
        if _BOT_WALL_RE.search(err or ""):
            stats.bot_walls += 1
            stats.stage_walls += 1
        # a fault of the TRACK (empty, unparseable) is not a circuit fault
        _final = bool(_TERMINAL_RE.search(err)) or err.startswith(("caption track was empty", "unparseable caption file"))
        if progress:
            progress(f"captions {lang_key}: attempt {i + 1}/{attempts} failed ({err[:80]})"
                     + (" — rotating" if i + 1 < attempts and not _final else ""))
        if _final:
            break
    return False, [], last


def download_audio(url: str, video_id: str, tor_proxy: str, dest_dir: Path, *, run: RunFn,
                   stats: FetchStats, attempts: int = ATTEMPTS, timeout: float = AUDIO_TIMEOUT_S,
                   progress=None) -> Tuple[Optional[Path], str]:
    """The audio tier's download: ``yt-<id>.<ext>`` in ``dest_dir``."""
    tpl = str(dest_dir / (document_stem(video_id) + ".%(ext)s"))
    found: Dict[str, str] = {}

    def do(proxy: str):
        rc, out, err = run(ytdlp_base_command(proxy) + [
            "-f", "ba[ext=m4a]/ba", "--force-overwrites", "-o", tpl,
            "--print", "after_move:filepath", url,
        ], _bounded(timeout, stats), str(dest_dir))
        path = out.strip().splitlines()[-1].strip() if out.strip() else ""
        if rc == 0 and path and os.path.isfile(path) and os.path.getsize(path) > 0:
            found["path"] = path
            return True, path
        return False, err

    ok, payload = _rotate(video_id, "audio", tor_proxy, attempts, do, stats, progress,
                          deadline=getattr(stats, "deadline", None))
    if ok:
        return Path(found["path"]), ""
    return None, str(payload)


def _store_passages(memory_system, filename: str,
                    passages: List[Tuple[float, float, str]], progress=None) -> int:
    chunks = [passage_chunk(filename, s, e, t) for s, e, t in passages]
    try:
        for i in range(0, len(chunks), BATCH_CHUNKS):
            ok, msg = memory_system.ingest_document(filename, chunks[i:i + BATCH_CHUNKS], _batch=True)
            if not ok:
                raise RuntimeError(f"embedding failed at chunk {i}: {msg}")
        _done = getattr(memory_system, "clear_ingest_marker", None)
        if callable(_done):  # KB review 10-30 (I5): the document is complete
            _done(filename)
    except BaseException:
        # no half-transcript behind (§4MD M13)
        _rb = getattr(memory_system, "rollback_partial_document", None)
        if callable(_rb):
            _rb(filename)
        raise
    return len(chunks)


def clean_title(title) -> str:
    """§4PI (R1-m10): the title is attacker text that lands in the ingest
    reply, the summary row and the transcript head — one line, bounded,
    and never a line that reads as a tool result."""
    t = re.sub(r"[\r\n\t]+", " ", str(title or "")).strip()[:200]
    try:
        from ..tools.memory import neutralise_kb_text
        t = neutralise_kb_text(t)
    except Exception:  # noqa: BLE001
        pass
    return t


def _persist_structure(memory_system, filename: str, *, title: str, url: str, video_id: str,
                       language: str, route: str, duration: float,
                       passages: List[Tuple[float, float, str]], gaps: List[str], now: str,
                       stopped: str = "") -> None:
    """Outline record (one entry per passage), the ordered transcript
    sidecar, and a CONTENT-bearing summary row. Each is best-effort and
    independent: losing the nice-to-have never costs the ingest."""
    title = clean_title(title)
    entries = [[1, f"{format_timestamp(s)}–{format_timestamp(e)}  {t[:80]}"] for s, e, t in passages]
    try:
        memory_system.set_document_outline(filename, {
            "filename": filename, "source": route, "title": title, "url": url,
            "video_id": video_id, "language": language, "duration_s": duration,
            "entries": entries, "chunks": len(passages), "gaps": gaps, "at": now,
        })
    except Exception as exc:  # noqa: BLE001
        logger.debug("outline not stored for %s: %s", filename, exc)
    try:
        memory_system.set_document_text(filename, {
            "filename": filename, "title": title, "url": url, "video_id": video_id,
            "language": language, "route": route, "duration_s": duration,
            "gaps": gaps, "stopped": stopped,      # §4PI (R1-M3): a partial run says so on re-request
            "passages": [[s, e, t] for s, e, t in passages], "at": now,
        })
    except Exception as exc:  # noqa: BLE001
        logger.debug("transcript sidecar not stored for %s: %s", filename, exc)
    try:
        preview = render_transcript(passages)[:SUMMARY_PREVIEW_CHARS]
        summary = (
            f"[Document Summary: {filename}] YouTube video \"{title}\" ({url}), "
            f"{format_timestamp(duration)}, language {language or 'unknown'}, transcript from "
            f"{'captions' if route == 'captions' else 'audio transcription'}, {len(passages)} passages"
            + (f", gaps: {'; '.join(gaps)}" if gaps else "") +
            f". Opening words: {preview}… Read it in order with knowledge_base(action='transcript', "
            f"filename='{filename}'); ask about it with action='query'."
        )
        memory_system.add(summary, {"type": "document_summary", "source": filename, "timestamp": now})
    except Exception as exc:  # noqa: BLE001
        logger.debug("summary row not stored for %s: %s", filename, exc)


def ingest_youtube(url: str, *, sandbox_dir: Path, memory_system, tor_proxy: Optional[str],
                   language: Optional[str] = None, progress: Optional[Callable[[str], None]] = None,
                   run: Optional[RunFn] = None, now: Optional[str] = None,
                   attempts: int = ATTEMPTS, audio_ingest=None) -> YoutubeIngestResult:
    """Resolve → captions or audio → knowledge base. Blocking; the caller
    runs it in a thread. Returns a result whose ``message`` is the tool's
    reply (``SUCCESS: …`` / ``Error: …`` / ``Skipped: …``)."""
    from ..utils.helpers import get_utc_timestamp

    run = run or _run_subprocess
    now = now or get_utc_timestamp()
    say = progress or (lambda _m: None)
    vid = youtube_video_id(url)
    if not vid:
        return YoutubeIngestResult(False, f"Error: '{url}' is not a YouTube video URL.")
    if not tor_proxy:
        return YoutubeIngestResult(False, (
            "Error: the YouTube route is Tor-only and no Tor proxy is configured; "
            "refusing to fetch it in the clear."), video_id=vid)
    if memory_system is None:
        return YoutubeIngestResult(False, "Error: Memory system is disabled.", video_id=vid)

    have = existing_document_for(vid, memory_system.get_library() or [])
    if have:
        # §4PI (R1-M3): a PARTIAL / aborted prior run is not "already there"
        _partial = ""
        try:
            _rec = memory_system.get_document_text(have) or {}
            _gaps = [str(g) for g in (_rec.get("gaps") or [])]
            _stopped = str(_rec.get("stopped") or "")
            if _gaps or _stopped:
                _partial = (" NOTE: the stored transcript is PARTIAL — not transcribed: "
                            + "; ".join(_gaps[:4]) + (f"; stopped early: {_stopped}" if _stopped else "")
                            + ". To redo the whole video, forget it first (the user confirms), then transcribe again.")
        except Exception:  # noqa: BLE001
            pass
        return YoutubeIngestResult(True, (
            f"Skipped: this video is already in the knowledge base as '{have}'.{_partial} Read it in order "
            f"with knowledge_base(action='transcript', filename='{have}') or ask about it with "
            f"action='query'. To transcribe it afresh it must be forgotten first, which the USER confirms: "
            f"knowledge_base(action='forget', target='{have}') shows them what would go; after their yes, "
            f"transcribe again."), video_id=vid, filename=have)

    stats = FetchStats()
    stats.deadline = time.monotonic() + FETCH_BUDGET_S       # §4PI (R1-M4)
    pot_up = ensure_pot_server()
    if not pot_up:
        say("PO-token helper is down — YouTube will refuse every Tor exit")

    canonical = f"https://www.youtube.com/watch?v={vid}"
    say(f"resolving {vid} over Tor")
    info, reason, proxy = fetch_info(canonical, vid, tor_proxy, run=run, stats=stats,
                                     attempts=attempts, progress=say)
    if not info:
        return YoutubeIngestResult(False, (
            f"Error: could not fetch YouTube video {vid} over Tor after {stats.attempts} "
            f"circuit(s). Last: {reason}.{_pot_hint(stats, pot_up)} No file was written and nothing "
            f"was stored; if the video is private or removed, only its owner can help."),
            video_id=vid, fetch=stats)

    title = clean_title(info.get("title") or vid)
    duration = float(info.get("duration") or 0.0)
    lang = str(info.get("language") or "")
    say(f"resolved \"{title[:60]}\" ({format_timestamp(duration)})")
    # §4PI (R1-M2): a live broadcast has no transcript to fetch — it used to
    # go to the audio tier (duration None → 0 bypassed the cap) and run `-f ba`
    # on the live stream to the timeout, eight circuits long, reported as "timed out"
    _live = str(info.get("live_status") or "")
    if info.get("is_live") or _live in ("is_live", "is_upcoming"):
        return YoutubeIngestResult(False, (
            f"Error: \"{title}\" is a live stream ({_live or 'is_live'}) — there is no transcript to fetch "
            f"until the broadcast ends and YouTube publishes the recording. Nothing was stored."),
            video_id=vid, title=title, fetch=stats)

    # ── captions tier ──
    track = choose_caption_track(info, language)
    lang_note = caption_language_note(language, track)
    passages: List[Tuple[float, float, str]] = []
    route = ""
    gaps: List[str] = []
    filename = ""
    stopped = ""
    caption_err = ""
    if track and proxy:
        lang_key, kind = track
        ok, segs, err = fetch_captions_rotating(canonical, vid, lang_key, proxy, tor_proxy, run=run,
                                                stats=stats, progress=say)
        if ok:
            passages = segments_to_passages(segs)
            route = "captions"
            lang = lang_key.replace("-orig", "") or lang
            filename = f"{document_stem(vid)}.captions.{lang or 'und'}"
            say(f"captions ({kind}, {lang_key}): {len(segs)} lines → {len(passages)} passages")
        else:
            caption_err = err
            say(f"captions {lang_key} could not be fetched ({err[:80]}) — falling back to audio")

    # ── audio tier ──
    if not passages:
        _cap_phrase = (f"has captions ({track[0]}) that could not be fetched over Tor" if track
                       else "has no captions")
        if duration and duration > MAX_AUDIO_SECONDS:
            return YoutubeIngestResult(False, (
                f"Error: \"{title}\" is {format_timestamp(duration)} long, {_cap_phrase}, and the "
                f"audio path is capped at {format_timestamp(MAX_AUDIO_SECONDS)} "
                f"(GHOST_AUDIO_MAX_S). Nothing was stored."), video_id=vid, title=title, fetch=stats)
        dest = Path(sandbox_dir)
        dest.mkdir(parents=True, exist_ok=True)
        # A previous attempt may have left the audio behind (the node was down
        # after the download): reuse it rather than spend eight circuits again.
        # Named candidates only (no glob/iterdir — the symlink-class ratchet):
        # a planted link in the sandbox must never be "reused" as audio.
        leftover = [c for c in (dest / f"{document_stem(vid)}.{ext}" for ext in _AUDIO_EXTS)
                    if not c.is_symlink() and c.is_file() and c.stat().st_size > 0]
        if leftover:
            path, reason = leftover[0], ""
            say(f"reusing the audio already in the sandbox: {path.name}")
        else:
            say(f"{_cap_phrase} — downloading the audio track over Tor")
            path, reason = download_audio(canonical, vid, tor_proxy, dest, run=run, stats=stats,
                                          attempts=attempts, progress=say)
        if not path:
            _cap = (f"{_cap_phrase} (last: {caption_err[:120]}) " if track else "has no captions ")
            return YoutubeIngestResult(False, (
                f"Error: \"{title}\" {_cap}and its audio could not be fetched over Tor "
                f"after {stats.attempts} circuit(s). Last: {reason}.{_pot_hint(stats, pot_up)} "
                f"Nothing was stored."), video_id=vid, title=title, fetch=stats)
        filename = path.name
        say(f"audio saved as {filename} ({path.stat().st_size // 1024} KB) — transcribing")
        do_ingest = audio_ingest or ingest_audio_streaming
        st = do_ingest(path, filename, memory_system,
                       progress=lambda s: say(f"{filename}: {s.windows} window(s), "
                                              f"{format_timestamp(s.seconds)} transcribed"))
        passages = list(getattr(st, "transcript", []) or [])
        gaps = list(getattr(st, "gaps", []) or [])
        stopped = str(getattr(st, "aborted", "") or "")
        route = "audio"
        if not passages:
            return YoutubeIngestResult(False, (
                f"Error: no speech was transcribed from \"{title}\" ({filename}). "
                + ("Windows failed: " + "; ".join(st.errors[:3]) if st.errors else
                   "The recording may be silent or music-only.")),
                video_id=vid, title=title, filename=filename, route=route, fetch=stats)
        # The audio path already stored its chunks; only the structure follows.
    else:
        say(f"indexing {len(passages)} passages as {filename}")
        try:
            _store_passages(memory_system, filename, passages, progress=say)
        except Exception as exc:  # noqa: BLE001
            return YoutubeIngestResult(False, f"Ingest Error: {exc}", video_id=vid, title=title,
                                       filename=filename, route=route, fetch=stats)

    _persist_structure(memory_system, filename, title=title, url=canonical, video_id=vid,
                       language=lang, route=route, duration=duration, passages=passages,
                       gaps=gaps, now=now, stopped=stopped)

    transcript = render_transcript(passages)
    preview = transcript[:PREVIEW_CHARS]
    try:                                   # §4PI (R2-m6): the preview is content, like the paged route
        from ..tools.memory import neutralise_kb_text
        preview = neutralise_kb_text(preview)
    except Exception:  # noqa: BLE001
        pass
    more = ""
    if len(transcript) > PREVIEW_CHARS:
        more = (f"\n… [{len(transcript) - PREVIEW_CHARS} more characters — read on with "
                f"knowledge_base(action='transcript', filename='{filename}', offset={PREVIEW_CHARS})]")
    gap_note = f" Gaps (not transcribed): {'; '.join(gaps)}." if gaps else ""
    if stopped:
        gap_note += (f" STOPPED EARLY: {stopped}. What was transcribed is kept; to redo the whole "
                     f"video later: knowledge_base(action='forget', target='{filename}') (a preview the user "
                     f"confirms), then transcribe again.")
    head = "SUCCESS (partial)" if stopped else "SUCCESS"
    msg = (
        f"{head}: Transcribed YouTube video \"{title}\" ({format_timestamp(duration)}, "
        f"language {lang or 'unknown'}, via {'captions' if route == 'captions' else 'audio transcription'}) "
        f"into the knowledge base as '{filename}' — {len(passages)} timestamped passages.{lang_note}{gap_note}\n"
        f"TRANSCRIPT ({min(len(transcript), PREVIEW_CHARS)} of {len(transcript)} chars):\n{preview}{more}\n"
        f"Ask about it with knowledge_base(action='query', filename='{filename}', question='...'); "
        f"the whole text in order is action='transcript'."
    )
    return YoutubeIngestResult(True, msg, video_id=vid, title=title, filename=filename, route=route,
                               language=lang, duration=duration, passages=passages, fetch=stats)
