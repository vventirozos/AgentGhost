import asyncio
import datetime
import json
import logging
import os
import shutil
import sys
import textwrap
import threading
import time
import unicodedata
import contextvars
from typing import Any, Optional

request_id_context = contextvars.ContextVar("request_id", default="SYSTEM")

# WHY a verify call is being routed ("turn gate", "reflection plan-verify",
# …). Read by the llm-routing log lines so a burst of otherwise-identical
# "Routing verification …" entries says which subsystem is asking. A
# contextvar (not an argument) because the log site sits several call
# layers below the code that knows the purpose, and it must survive
# awaits on the async path. Empty string = untagged, renders unchanged.
verify_purpose_context = contextvars.ContextVar("verify_purpose", default="")


from contextlib import contextmanager


@contextmanager
def verify_purpose(label: str):
    """Tag verifier LLM calls made inside this block with ``label``."""
    token = verify_purpose_context.set(str(label or ""))
    try:
        yield
    finally:
        try:
            verify_purpose_context.reset(token)
        except Exception:  # noqa: BLE001 — cross-context reset; never break a caller
            pass
LOG_TRUNCATE_LIMIT = 60
DEBUG_MODE = False
VERBOSE_MODE = False  # When True, raw streamed thinking tokens are printed.

# File-only logger that receives pretty_log's FULL untruncated content, so the
# durable log becomes a complete plain-text record (see pretty_log / _mirror /
# setup_logging). None until setup_logging wires it; _mirror no-ops meanwhile.
_MIRROR_LOGGER: Optional[logging.Logger] = None

# level-name -> numeric, for the durable mirror (getLevelName is unreliable
# for the short "WARN" form across versions).
_LEVELNO = {
    "DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARN": logging.WARNING,
    "WARNING": logging.WARNING, "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def spawn_task(coro):
    """Spawn an asyncio task that inherits the CURRENT contextvars.

    On Python 3.11+ this passes an explicit ``context=`` to
    ``loop.create_task``. On 3.10 (which doesn't accept the ``context``
    kwarg) we fall back to plain ``create_task`` — which already
    snapshots the current context at task-construction time — so the
    behaviour is the same in either case. Use this helper instead of
    ``asyncio.create_task(...)`` for any background work that should
    log under the spawning request's id.

    NOTE: prefer :func:`spawn_bg` for real fire-and-forget work — it adds
    the strong-ref + exception-logging guarantees this bare helper lacks.
    """
    ctx = contextvars.copy_context()
    loop = asyncio.get_event_loop()
    try:
        return loop.create_task(coro, context=ctx)
    except TypeError:
        # Python <3.11: create_task doesn't accept `context`. The default
        # Task constructor still copies the current context, so calling
        # this from inside the spawning coroutine yields equivalent
        # propagation. (We can't easily reapply an arbitrary `ctx` to a
        # coroutine on 3.10 without subclassing Task.)
        return loop.create_task(coro)


# Process-wide strong-ref registry for fire-and-forget tasks. asyncio keeps
# only a WEAK ref to a bare create_task, so an un-stored task can be garbage-
# collected mid-flight and the work silently never lands. Every spawn_bg task
# is held here until it finishes; the lifespan shutdown drains this set.
_BG_TASKS: set = set()


def spawn_bg(coro, *, name: str = "bg"):
    """The one fire-and-forget primitive. Composes the three guarantees the
    four ad-hoc conventions each had only some of:

    1. contextvars propagation — the task logs under the spawning request id
       (same as :func:`spawn_task`).
    2. a strong reference held in a module registry until the task completes,
       so it can't be GC'd mid-flight and CAN be drained at shutdown.
    3. a done-callback that logs any non-Cancelled exception via
       ``logger.warning`` (auto-renders on the operator's live stream) —
       instead of the error vanishing into a swallowed background coroutine.

    Use this for background memory writes, retractions, reflections, PRM
    updates, graph extraction — anything that must not block a turn but whose
    silent death would be a real loss. Returns the Task.
    """
    task = spawn_task(coro)
    _BG_TASKS.add(task)

    def _done(t: "asyncio.Task"):
        _BG_TASKS.discard(t)
        if t.cancelled():
            return
        exc = t.exception()
        if exc is not None:
            logger.warning("background task %r failed: %s: %s",
                           name, type(exc).__name__, exc)

    task.add_done_callback(_done)
    return task


async def drain_background_tasks(timeout: float = 5.0):
    """Await outstanding spawn_bg tasks at shutdown (best-effort, bounded).
    Called from the lifespan finally-block so in-flight memory writes get a
    chance to land before the process exits."""
    pending = [t for t in list(_BG_TASKS) if not t.done()]
    if not pending:
        return
    try:
        await asyncio.wait_for(
            asyncio.gather(*pending, return_exceptions=True), timeout=timeout)
    except (asyncio.TimeoutError, Exception):
        pass

# Serializes stdout writes so concurrent requests can never interleave a line.
# Without this, two requests streaming `print(token, end="")` will splice into
# the same physical line and produce unparseable logs.
_STDOUT_LOCK = threading.Lock()


def atomic_print(line: str) -> None:
    """Print one complete log line atomically. Always appends a newline."""
    with _STDOUT_LOCK:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Color & per-request state
# ---------------------------------------------------------------------------

# Honor the NO_COLOR convention (https://no-color.org/) and FORCE_COLOR.
# Auto-disable when stdout is not a TTY so file logs and piped output stay
# clean (no escape codes leaking into grep/sed/jq).
def _color_enabled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


_USE_COLOR = _color_enabled()


def _ansi(code: str) -> str:
    return f"\033[{code}m" if _USE_COLOR else ""


RESET = _ansi("0")
DIM = _ansi("2")
BOLD = _ansi("1")

# Level colors. Maps both short and full forms.
_LEVEL_COLOR = {
    "INFO": _ansi("36"),       # cyan
    "WARN": _ansi("33"),       # yellow
    "WARNING": _ansi("33"),
    "ERROR": _ansi("31"),      # red
    "CRITICAL": _ansi("1;31"), # bold red
    "DEBUG": _ansi("2"),       # dim
}

# Twelve high-contrast 256-color codes for per-request tags. Picked so that
# adjacent palette entries are visually distinct.
_REQ_PALETTE = [39, 45, 51, 81, 117, 141, 178, 208, 213, 198, 159, 222]


def _req_color(req_id: str) -> str:
    if not _USE_COLOR or req_id == "SYSTEM":
        return ""
    h = sum(ord(c) for c in req_id) % len(_REQ_PALETTE)
    return f"\033[38;5;{_REQ_PALETTE[h]}m"


def _req_tag(req_id: str) -> str:
    """Two-char visual tag derived from the request id."""
    if req_id == "SYSTEM":
        return "**"
    return req_id[:2].upper()


# Per-request lifecycle state (start time, monotonic). Lives only between
# BEGIN and END markers so we don't leak memory across long-running daemons.
_REQ_STATE_LOCK = threading.Lock()
_REQ_STATE: dict = {}  # req_id -> {"started": float}

# Console repeat-collapse state (see pretty_log). One pending run at a
# time, guarded by its own lock — atomic_print takes _STDOUT_LOCK
# internally, so nesting that here would deadlock.
_COLLAPSE_LOCK = threading.Lock()
_COLLAPSE_STATE: dict | None = None


def _collapse_enabled() -> bool:
    return os.getenv("GHOST_LOG_COLLAPSE", "1").strip().lower() not in (
        "0", "false", "no")


def _print_collapse_summary(run: dict) -> None:
    """One dim console line closing a swallowed repeat-run. Console only
    (every occurrence already hit the mirror); never raises."""
    try:
        req_id, icon, title_str, _full, _lvl = run["key"]
        n = run["count"]
        span = max(0.0, run["last"] - run["first"])
        tag = _req_tag(req_id)
        rcol = _req_color(req_id)
        line = (
            f"{rcol}│{RESET}  {rcol}{tag}{RESET}  {_icon_cell(icon)}  "
            f"{DIM}{_format_delta(req_id)}{RESET}  "
            f"{DIM}{_fit_title(title_str)}  ⤷ repeated ×{n} in {span:.0f}s{RESET}"
        )
        atomic_print(line)
    except Exception:  # noqa: BLE001 — a summary must never break logging
        pass


def _req_started(req_id: str) -> Optional[float]:
    with _REQ_STATE_LOCK:
        s = _REQ_STATE.get(req_id)
        return s["started"] if s else None


def request_elapsed_s(req_id: str) -> Optional[float]:
    """Seconds since the request's BEGIN marker, or ``None`` when the
    request isn't being tracked (already closed, or never opened —
    e.g. sim/ablation contexts). Public twin of the pretty-log delta
    so end-of-turn writers (trajectory corpus) can stamp the same
    wall-clock the operator sees in the stream."""
    started = _req_started(req_id)
    if started is None:
        return None
    return time.monotonic() - started


def _fmt_secs(delta: float) -> str:
    if delta < 10:
        return f"+{delta:4.2f}s"
    if delta < 100:
        return f"+{delta:4.1f}s"
    return f"+{int(delta):4d}s"


# Anchor for SYSTEM-scoped ("**") lines. Those carry no request, so they had
# no delta at all — six blank spaces — which meant post-turn and idle work
# (metacog, hippocampus, dream, reaping, the stream drain) was the only part
# of the agent whose cost was invisible in the stream the operator actually
# watches. The reading is time SINCE THE PREVIOUS ** LINE, i.e. how long the
# step that just finished took; there is no request start to measure from,
# and "since process boot" would be useless on a daemon that runs for weeks.
_SYSTEM_ANCHOR_LOCK = threading.Lock()
_SYSTEM_ANCHOR: dict = {"t": None}


def _delta_from(prev: Optional[float], now: float) -> str:
    """Rendered gap between two anchors, or blanks when there is no previous
    one. PURE — split out from the stateful reader below so the semantics can
    be tested deterministically: the anchor is process-global and any
    background thread that logs (a watchdog, a scheduler, `spawn_bg` work)
    can advance it between two statements, which made a test asserting on the
    global pass alone and fail in a full run."""
    if prev is None:
        return "      "
    return _fmt_secs(max(0.0, now - prev))


def _system_delta() -> str:
    """`+12.3s` since the previous SYSTEM line; 6 spaces for the first one
    (nothing to measure against yet). Advances the anchor."""
    now = time.monotonic()
    with _SYSTEM_ANCHOR_LOCK:
        prev = _SYSTEM_ANCHOR["t"]
        _SYSTEM_ANCHOR["t"] = now
    return _delta_from(prev, now)


def reset_system_delta_anchor() -> None:
    """Drop the anchor so the next SYSTEM line starts a fresh measurement.

    Called when a request OPENS: the gap across a whole user turn is not the
    duration of any background step, and reporting it as one would be the
    same category error the request frame made."""
    with _SYSTEM_ANCHOR_LOCK:
        _SYSTEM_ANCHOR["t"] = None


def _format_delta(req_id: str) -> str:
    """`+12.3s` since the request began, or 6 spaces if not tracked."""
    started = _req_started(req_id)
    if started is None:
        return _system_delta() if req_id == "SYSTEM" else "      "
    return _fmt_secs(time.monotonic() - started)


# ---------------------------------------------------------------------------
# Icons
# ---------------------------------------------------------------------------

class Icons:
    # --- Lifecycle ---
    SYSTEM_BOOT  = "⚡"
    SYSTEM_READY = "🚀"
    SYSTEM_SHUT  = "💤"

    # --- Request Flow ---
    REQ_START    = "🎬"
    REQ_DONE     = "🏁"
    REQ_WAIT     = "⏳"

    # --- Brain ---
    BRAIN_THINK  = "💭"   # live streaming thought
    BRAIN_SUM    = "🧠"   # post-stream thought summary
    BRAIN_PLAN   = "📋"
    BRAIN_CTX    = "🧩"
    BRAIN_ROUTE  = "🧭"   # semantic routing / skill selection
    BRAIN_AIM    = "🎯"   # self-play frontier targeting
    LLM_ASK      = "💬"   # LLM request (wide-base; was 🗣️, a narrow-base+VS16 glyph)
    LLM_REPLY    = "🤖"

    # --- Specialized Tools ---
    TOOL_SEARCH  = "🌐"
    TOOL_DEEP    = "🔬"
    TOOL_CODE    = "🐍"
    TOOL_SHELL   = "🐚"
    TOOL_FILE_W  = "💾"
    TOOL_FILE_R  = "📖"
    TOOL_FILE_M  = "📙"   # MULTI-path batch read (distinct from the single 📖)
    TOOL_FILE_S  = "🔍"
    TOOL_FILE_I  = "👀"
    TOOL_DOWN    = "📥"   # download / incoming (wide-base; was ⬇️)
    TOOL_BROWSER = "🌎"   # headless browser automation (distinct from shell 🐚)
    TOOL_DARKWEB = "🧅"   # dark-web (.onion) search over Tor (onion = 🧅, distinct from clearnet 🌐)
    IMAGE_GEN    = "🎨"
    REPORT_PDF   = "📄"
    NODE_WORKER  = "🔧"   # background / edge worker-node compute (wide-base; was ⚙️)
    NODE_EDGE    = "🛸"   # swarm edge-node compute (wide-base; was 🛰️)

    # --- Memory & Identity ---
    MEM_SAVE     = "📝"
    MEM_READ     = "🔎"
    MEM_MATCH    = "📍"
    MEM_INGEST   = "📚"
    MEM_SPLIT    = "📑"   # chunk split (distinct from CUT 🔻)
    MEM_EMBED    = "🧬"
    MEM_WIPE     = "🧹"
    MEM_SCRATCH  = "📔"   # scratchpad memory (wide-base; was 🗒️)
    MEM_REINFORCE = "💪"  # an existing memory/skill strengthened or merged (NOT 🔄 RETRY)
    USER_ID      = "👤"
    SELF_STATE   = "🪞"   # selfhood state / mood transition
    SKILL_GRADUATE = "🎓" # a lesson graduates into a reusable skill
    DREAM        = "🌙"   # REM / dream consolidation cycle (NOT 💤 — that's SYSTEM_SHUT)
    SKIP         = "⏩"   # a step/cycle deliberately skipped
    ACTIVITY     = "📡"   # autonomous-activity ledger / background digest surfaced
    NOTIFY_OUT   = "📣"   # outbound push notification to the operator

    # --- Status ---
    OK           = "✅"
    FAIL         = "❌"
    WARN         = "🔶"   # warning / caution — amber (wide-base; was ⚠️, a narrow-base+VS16 glyph)
    STOP         = "🛑"
    RETRY        = "🔄"
    IDEA         = "💡"
    BUG          = "🐛"
    SHIELD       = "🔒"   # security / guard / fail-closed (wide-base; was 🛡️)
    CUT          = "🔻"   # context compaction / trim (wide-base; was ✂️)
    CONSTRAINT   = "🔗"   # explicit-user-constraint capture/steer/gate (wide-base; was ⛓️)
    GAME_MOVE    = "🎮"   # participant-mode game turn (/api/game/move) (wide-base; was ♟️)
    FEEDBACK_POS = "👍"   # human outcome label, positive (/api/feedback → corrections sidecar)
    FEEDBACK_NEG = "👎"   # human outcome label, negative (same channel; distinct so the
                          # stream shows the label's direction at a glance)

    # --- Custom Modes ---
    MODE_GHOST   = "🫥"
    POSTGRES     = "🐘"

    # --- Boot-phase icons ---
    # Each startup component gets a distinct glyph so the first page of
    # the log is scannable and each subsystem's state jumps out. These
    # supplement (do not replace) the generic lifecycle icons above —
    # SYSTEM_BOOT / SYSTEM_READY stay 🌅-style brackets around the whole
    # boot sequence, while the specific lines in between use these.
    BOOT_AWAKE       = "🌅"   # process spark — the very first line
    SANDBOX_BOX      = "📦"   # sandbox container mount
    GRAPH_WEB        = "🪢"   # triplet / knowledge-graph store (wide-base; was 🕸️)
    VECTOR_EMBED     = "🧮"   # vector DB + sentence embeddings (was 🧬, colliding with MEM_EMBED)
    MEM_INDEX        = "📇"   # memory system fully loaded with items (wide-base; was 🗃️)
    MEM_LIBRARY      = "📓"   # indexed fragments ready for recall (distinct from MEM_INGEST 📚)
    BELIEF_SCALES    = "🧾"   # contradiction log / belief versioning (wide-base; was ⚖️)
    THRESHOLD_TUNE   = "📶"   # adaptive recall threshold (wide-base; was 🎚️)
    EPISODE_REEL     = "🎥"   # episodic memory (sessions = frames) (wide-base; was 🎞️)
    EVENT_BUS        = "🔀"   # memory-bus pub/sub fan-out (was 📡, colliding with ACTIVITY)
    VERIFIER_LAB     = "🧪"   # self-evaluation gate
    UNCERTAINTY_DIE  = "🎲"   # uncertainty tracker
    MCTS_TREE        = "🌳"   # deep-reason MCTS search tree
    FORESIGHT        = "🔮"   # shadow world model — tool-outcome prediction
    HEARTBEAT        = "🫀"   # biological watchdog heartbeat
    JOB_PROMOTE      = "🏃"   # a sandbox command outran its budget and was
                              # DETACHED as a supervised job instead of killed
                              # (sandbox/jobs.py). Kills/expiries use STOP 🛑.


logger = logging.getLogger("GhostAgent")


# Every first-party logger in the codebase. Only "GhostAgent" used to be
# configured; the others (selfhood/workspace/optim/distill/reflection) had
# NO handlers, so their debug/info was dropped and their warning/error
# leaked to bare stderr — never the log file, never the monitored stream.
_GHOST_LOGGERS = (
    "GhostAgent", "GhostSelfhood", "GhostWorkspace",
    "GhostOptim", "GhostDistill", "GhostReflect",
)


class _PrettyLogHandler(logging.Handler):
    """Render stdlib WARNING+ records into the pretty_log console stream so
    failures show up in the SAME aligned/iconed channel the operator watches
    — instead of as bare, unframed plain lines interleaved with it.
    WARNING → ⚠️, ERROR/CRITICAL → ❌. INFO/DEBUG are NOT rendered here (they
    still reach the file handler); skipping them also prevents recursion with
    pretty_log's own DEBUG mirror.
    """
    def emit(self, record):
        if record.levelno < logging.WARNING:
            return
        try:
            name = record.name
            # "GhostSelfhood" → "Selfhood", "GhostAgent" → "Agent".
            subsystem = name[5:] if name.startswith("Ghost") else name
            icon = Icons.FAIL if record.levelno >= logging.ERROR else Icons.WARN
            pretty_log(subsystem or "log", record.getMessage(),
                       icon=icon, level=record.levelname)
        except Exception:
            self.handleError(record)


def setup_logging(log_file: str, debug: bool = False, daemon: bool = False, verbose: bool = False):
    global DEBUG_MODE, LOG_TRUNCATE_LIMIT, VERBOSE_MODE
    DEBUG_MODE = debug
    VERBOSE_MODE = verbose
    if verbose:
        LOG_TRUNCATE_LIMIT = 1000000
    level = logging.DEBUG if debug else logging.INFO
    # File log keeps a plain, grep-friendly format — now WITH the logger name
    # so subsystem lines (GhostSelfhood vs GhostAgent) are distinguishable.
    # DATE INCLUDED. The durable log is the instrument several §4F watch
    # readings are taken from, and a time-only stamp made every window label
    # a guess — the journal records one reading whose window was wrong for
    # exactly this reason, across a log that had also rotated mid-window.
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # A bare filename has no dirname and os.makedirs("") raises even with
    # exist_ok=True — only create the directory when there is one.
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # Console handler: route WARNING+ through pretty_log (only when not daemon).
    pretty_handler = _PrettyLogHandler()
    pretty_handler.setLevel(logging.WARNING)

    # Configure GhostAgent AND every subsystem logger with the SAME handlers.
    # Clearing first keeps repeat calls (hot reload, restart, test fixtures)
    # from accumulating handlers / leaking file descriptors.
    for name in _GHOST_LOGGERS:
        lg = logging.getLogger(name)
        for old in list(lg.handlers):
            try:
                old.close()
            except Exception:
                pass
            lg.removeHandler(old)
        lg.setLevel(level)
        # NB: leave propagate=True (the default). Now that every Ghost*
        # logger HAS handlers, Python's `lastResort` stderr fallback never
        # fires (it only triggers when zero handlers are found in the
        # chain), so there's no double-emit — and propagation keeps pytest's
        # caplog (a root-level handler) working.
        lg.addHandler(fh)
        if not daemon:
            lg.addHandler(pretty_handler)

    for lib in ["httpx", "uvicorn", "docker", "chromadb", "urllib3", "pypdf"]:
        logging.getLogger(lib).setLevel(logging.WARNING)

    # Durable-mirror logger: pretty_log writes its FULL untruncated content
    # here so the on-disk log is a COMPLETE plain-text, restart-surviving
    # record of what the agent did (the stdout pretty stream is truncated and
    # wiped each boot). File-only — it carries ONLY the shared file handler and
    # propagate=False, so a mirror line NEVER double-prints to the operator's
    # stdout stream (the pretty_handler is not attached here).
    global _MIRROR_LOGGER
    _MIRROR_LOGGER = logging.getLogger("GhostStream")
    for old in list(_MIRROR_LOGGER.handlers):
        try:
            old.close()
        except Exception:
            pass
        _MIRROR_LOGGER.removeHandler(old)
    _MIRROR_LOGGER.setLevel(logging.DEBUG)
    _MIRROR_LOGGER.addHandler(fh)
    _MIRROR_LOGGER.propagate = False

    # Surface third-party WARNING+ that used to escape only to raw stderr —
    # notably transformers' "Token indices sequence length is longer than the
    # specified maximum" (a real context-overflow → silent truncation risk)
    # and any warnings.warn(...) deprecations. Route them to BOTH the durable
    # file and the operator's pretty stream so they can't hide.
    logging.captureWarnings(True)
    for extra in ("transformers", "py.warnings"):
        lg = logging.getLogger(extra)
        for old in list(lg.handlers):
            try:
                old.close()
            except Exception:
                pass
            lg.removeHandler(old)
        lg.setLevel(logging.WARNING)
        lg.addHandler(fh)
        if not daemon:
            lg.addHandler(pretty_handler)
        lg.propagate = False


# ---------------------------------------------------------------------------
# pretty_log — the new layout
# ---------------------------------------------------------------------------
#
# Output anatomy for a normal log line:
#
#   │  R7  💭  +0.42s  thinking          128 tokens · 0.9s
#   ^   ^    ^   ^         ^                  ^
#   │   │    │   │         │                  └── content (one line, truncated)
#   │   │    │   │         └── title (lowercased, bold, level-colored)
#   │   │    │   └── delta from request start
#   │   │    └── icon (emoji, picks itself)
#   │   └── 2-char request tag (deterministic color)
#   └── left frame edge (matches BEGIN/END box)
#
# BEGIN frame:
#   ┌─ R7 a8a93a27  request started  11:02:33 ─────────────────────────
# END frame:
#   └─ R7  request finished  +12.3s ──────────────────────────────────
#
# Concurrent requests still interleave line-by-line, but each line carries
# the colored 2-char tag so the eye can group them by stream.


def _truncate(content_str: str, limit: int) -> str:
    if len(content_str) <= limit:
        return content_str
    cut = content_str[:limit]
    last_space = cut.rfind(" ")
    if last_space > limit * 0.6:
        cut = cut[:last_space]
    return cut + "…"


# --- Icon-column normalization ------------------------------------------
# The whole layout assumes the icon occupies a FIXED 2-cell field. Whether a
# glyph actually renders as 2 cells is decided by its BASE codepoint's
# East-Asian width: a Wide/Fullwidth base (💬 🔶 🐍 …) renders as 2 on every
# terminal; a NARROW base — even with a VS16 (U+FE0F) emoji-presentation
# selector, e.g. ⚠️ 🗣️ 🛡️ — is genuinely ambiguous (2 cells on some
# terminals, 1 on others), which shifted the columns after it on exactly
# those lines. So the Icons registry is kept entirely wide-base (enforced by
# test_every_registry_icon_is_wide_base), and this helper pads any *stray*
# narrow glyph up to 2 cells as a best-effort backstop. Measuring by base
# width — NOT assuming VS16 ⇒ 2 — is what makes that measurement correct.
ICON_CELLS = 2


def _icon_display_width(icon: str) -> int:
    """Display width of ``icon`` in terminal cells, by base codepoint width.
    VS16 selectors, ZWJ, and combining marks are zero-advance."""
    w = 0
    for ch in icon:
        if unicodedata.combining(ch) or ord(ch) in (0x200D, 0xFE0F):
            continue
        w += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return w


def _icon_cell(icon: str) -> str:
    """Pad an icon out to the fixed 2-cell icon column."""
    pad = ICON_CELLS - _icon_display_width(icon)
    return icon + " " * pad if pad > 0 else icon


# The column where content begins (and where wrapped continuation lines are
# indented to). Derived from ONE formula so the layout can't silently drift:
#   │ ·· TG ·· <icon> ·· +12.3s ·· <title>·· <content>
#   1  2  2  2   2     2    6     2    18    2   →  content starts at col 39
TITLE_WIDTH = 18
_CONTENT_COL = (
    1            # left frame edge │
    + 2 + 2      # 2 spaces + 2-char request tag
    + 2 + ICON_CELLS  # 2 spaces + icon field
    + 2 + 6      # 2 spaces + fixed 6-char delta
    + 2 + TITLE_WIDTH  # 2 spaces + padded title
    + 2          # separator before content
)
_CONTINUATION_INDENT = " " * _CONTENT_COL


def _term_cols() -> int:
    try:
        return shutil.get_terminal_size((120, 24)).columns
    except Exception:
        return 120


def _fill_rule(visible_len: int, cap: int = 120) -> str:
    """A ``─`` rule that fills the rest of the terminal line after a frame's
    visible prefix, so each request's BEGIN/END spans the full console width
    — a much stronger visual separator than the old fixed 40 dashes."""
    cols = min(_term_cols(), cap)
    return "─" * max(4, cols - visible_len)


def _fit_title(title_str: str) -> str:
    if len(title_str) > TITLE_WIDTH:
        title_str = title_str[: TITLE_WIDTH - 1] + "…"
    return f"{title_str:<{TITLE_WIDTH}}"


def _wrap_content(content_str: str) -> str:
    """Wrap content to terminal width, indenting continuation lines to the
    content column so long details never spill back to column 0."""
    if not content_str:
        return content_str
    try:
        cols = shutil.get_terminal_size((120, 24)).columns
    except Exception:
        cols = 120
    width = max(40, cols - _CONTENT_COL)
    if len(content_str) <= width:
        return content_str
    lines = textwrap.wrap(
        content_str,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
        drop_whitespace=True,
    )
    if not lines:
        return content_str
    return ("\n" + _CONTINUATION_INDENT).join(lines)


# Operator-stream redaction. The operator MONITORS the live log stream,
# which historically was the single largest cleartext sink in the system:
# secrets, .onion addresses, full URLs, and absolute home paths flowed to
# the console verbatim (redaction was applied ONLY at the JSONL trajectory
# boundary, never at the log boundary). On by default (fail-safe toward
# privacy); flip via set_log_redaction() / --no-redact-logs. redact_text
# only rewrites known sensitive patterns, so ordinary log lines are
# untouched and stay readable.
_REDACT_LOGS = True


def set_log_redaction(enabled: bool) -> None:
    global _REDACT_LOGS
    _REDACT_LOGS = bool(enabled)


def _redact_log(s: str) -> str:
    if not _REDACT_LOGS or not s:
        return s
    try:
        from ..distill.redact import redact_text
        return redact_text(s)
    except Exception:
        # Never let a redaction failure break the monitored stream.
        return s


def _mirror(req_id: str, title: str, content: str, level: str = "INFO") -> None:
    """Write pretty_log's FULL (untruncated) content to the durable file sink.

    This is what makes ``$GHOST_HOME/system/ghost-agent.log`` a COMPLETE,
    plain-text, restart-surviving record of everything the agent did — the log
    you can grep or hand to another reader to reconstruct a turn. File-only
    (the GhostStream logger carries no stdout handler; see setup_logging), so a
    mirror line never double-prints on the operator's monitored stream. Never
    raises — logging must not break the app.
    """
    lg = _MIRROR_LOGGER
    if lg is None:
        return
    try:
        delta = _format_delta(req_id).strip()
        prefix = f"{req_id[:8]} {delta}".strip()
        sep = " — " if content else ""
        lg.log(_LEVELNO.get(level.upper(), logging.INFO),
               "[%s] %s%s%s", prefix, title, sep, content)
    except Exception:
        pass


def pretty_log(title: str, content: Any = None, icon: str = "🔹", level: str = "INFO", special_marker: str = None, no_truncate: bool = False, origin: str = None):
    req_id = request_id_context.get()
    tag = _req_tag(req_id)
    rcol = _req_color(req_id)

    # ---- Lifecycle frames ------------------------------------------------
    if special_marker == "BEGIN":
        with _REQ_STATE_LOCK:
            _REQ_STATE[req_id] = {"started": time.monotonic()}
        reset_system_delta_anchor()
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        # Measure the plain (ANSI-free) prefix, then fill the rest of the
        # line so the frame spans the full console width.
        visible = len(f"┌─ {tag} {req_id[:8]}  request started  {ts} ")
        rule = _fill_rule(visible)
        line = (
            f"{rcol}┌─ {BOLD}{tag}{RESET}{rcol} {req_id[:8]}{RESET}  "
            f"{DIM}request started  {ts}{RESET} "
            f"{rcol}{rule}{RESET}"
        )
        # ORIGIN STAMP (2026-08-11). Self-play/dream turns enter through the
        # SAME handle_chat as a human request, so the durable log could not
        # tell them apart — and `liveness._count_user_turns` counted all of
        # them as user turns. On 08-11 that made 28 self-play turns read as
        # 28 user turns while the true count was ZERO, which inverts the
        # meaning of every turn-driven silence below it. Stamped on the
        # MIRROR line only: the console frame is parsed by the uConsole
        # turnstatus + Slack owner-lock clients, and this is not worth a
        # cross-client sync. Absent `origin` emits the pre-08-11 shape, and
        # the probe reports that as UNCLASSIFIED rather than guessing.
        suffix = f" origin={origin}" if origin else ""
        _mirror(req_id, "request started", f"{req_id[:8]} at {ts}{suffix}",
                "INFO")
        atomic_print(line)
        return

    if special_marker == "END":
        delta = _format_delta(req_id).strip()
        # delta already carries its leading "+" (e.g. "+22.3s") — don't double it.
        _mirror(req_id, "request finished", delta, "INFO")
        with _REQ_STATE_LOCK:
            _REQ_STATE.pop(req_id, None)
        visible = len(f"└─ {tag}  request finished  {delta} ")
        rule = _fill_rule(visible)
        line = (
            f"{rcol}└─ {BOLD}{tag}{RESET}  "
            f"{DIM}request finished  {delta}{RESET} "
            f"{rcol}{rule}{RESET}"
        )
        atomic_print(line)
        return

    if special_marker == "SECTION_START":
        delta = _format_delta(req_id)
        title_str = title.lower().replace("_", " ")
        line = (
            f"{rcol}│{RESET}  {rcol}{tag}{RESET}  {_icon_cell(icon)}  "
            f"{DIM}{delta}{RESET}  "
            f"{BOLD}▼ {title_str}{RESET}"
        )
        _mirror(req_id, f"section start: {title_str}", "", "INFO")
        atomic_print(line)
        return

    if special_marker == "SECTION_END":
        delta = _format_delta(req_id)
        title_str = title.lower().replace("_", " ")
        line = (
            f"{rcol}│{RESET}  {rcol}{tag}{RESET}  {_icon_cell(icon)}  "
            f"{DIM}{delta}{RESET}  "
            f"{BOLD}▲ {title_str}{RESET}"
        )
        _mirror(req_id, f"section end: {title_str}", "", "INFO")
        atomic_print(line)
        return

    # ---- Normal log line -------------------------------------------------
    delta = _format_delta(req_id)
    title_str = title.lower().replace("_", " ")

    if content is None:
        raw = ""
    elif isinstance(content, (dict, list)):
        try:
            raw = repr(content) if len(content) > 50 else json.dumps(content, default=str)
        except Exception:
            raw = str(content)
    else:
        raw = str(content)

    # Flatten + redact ONCE to the full content, then derive two views:
    #   • FULL → the durable file mirror (_mirror): the complete, untruncated,
    #     restart-surviving record used to reconstruct what the agent did.
    #   • truncated + column-wrapped → the operator's scannable stdout stream.
    # Failures get a larger stream budget (240) so the *why* survives the live
    # view; the mirror always has the whole thing. ``no_truncate`` exempts the
    # stream line from the budget (💭 thinking blocks); the mirror is unaffected.
    full = _redact_log(raw.replace("\n", " ").replace("\r", ""))
    _mirror(req_id, title_str, full, level)

    # ── Console repeat-collapse (2026-08-05). Consecutive IDENTICAL lines
    # (same request, icon, title, content, level) print once; when a
    # DIFFERENT line arrives, the swallowed run is summarised in one dim
    # "repeated ×N" line. Motivating burst: 33 identical "Routing
    # verification to Critic Node (Nova)" lines during one idle cycle.
    # CONSOLE ONLY — the `_mirror` call above already recorded every
    # occurrence, because several instruments COUNT mirror lines (the
    # escalation-overturn double-count lesson: one logger, complete).
    # Known small imperfection: a pending summary flushes on the next
    # normal line, so it can appear after an intervening frame marker.
    # GHOST_LOG_COLLAPSE=0 disables.
    if _collapse_enabled():
        _ck = (req_id, icon, title_str, full, level.upper())
        _pending = None
        with _COLLAPSE_LOCK:
            global _COLLAPSE_STATE
            st = _COLLAPSE_STATE
            if st is not None and st["key"] == _ck:
                st["count"] += 1
                st["last"] = time.monotonic()
                return
            if st is not None and st["count"] > 1:
                _pending = st
            _COLLAPSE_STATE = {"key": _ck, "count": 1,
                               "first": time.monotonic(),
                               "last": time.monotonic()}
        if _pending is not None:
            _print_collapse_summary(_pending)

    stream_content = full
    if not no_truncate:
        _limit = LOG_TRUNCATE_LIMIT
        if level.upper() in ("WARNING", "WARN", "ERROR", "CRITICAL"):
            _limit = max(LOG_TRUNCATE_LIMIT, 240)
        stream_content = _truncate(stream_content, _limit)
    stream_content = _wrap_content(stream_content)

    lvl_col = _LEVEL_COLOR.get(level.upper(), "")
    sep = "  " if stream_content else ""
    line = (
        f"{rcol}│{RESET}  {rcol}{tag}{RESET}  {_icon_cell(icon)}  "
        f"{DIM}{delta}{RESET}  "
        f"{lvl_col}{BOLD}{_fit_title(title_str)}{RESET}{sep}{stream_content}"
    )
    atomic_print(line)
