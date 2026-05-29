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
import contextvars
from typing import Any, Optional

request_id_context = contextvars.ContextVar("request_id", default="SYSTEM")
LOG_TRUNCATE_LIMIT = 60
DEBUG_MODE = False
VERBOSE_MODE = False  # When True, raw streamed thinking tokens are printed.


def spawn_task(coro):
    """Spawn an asyncio task that inherits the CURRENT contextvars.

    On Python 3.11+ this passes an explicit ``context=`` to
    ``loop.create_task``. On 3.10 (which doesn't accept the ``context``
    kwarg) we fall back to plain ``create_task`` — which already
    snapshots the current context at task-construction time — so the
    behaviour is the same in either case. Use this helper instead of
    ``asyncio.create_task(...)`` for any background work that should
    log under the spawning request's id.
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


def _req_started(req_id: str) -> Optional[float]:
    with _REQ_STATE_LOCK:
        s = _REQ_STATE.get(req_id)
        return s["started"] if s else None


def _format_delta(req_id: str) -> str:
    """`+12.3s` since the request began, or 6 spaces if not tracked."""
    started = _req_started(req_id)
    if started is None:
        return "      "
    delta = time.monotonic() - started
    if delta < 10:
        return f"+{delta:4.2f}s"
    if delta < 100:
        return f"+{delta:4.1f}s"
    return f"+{int(delta):4d}s"


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
    LLM_ASK      = "🗣️"
    LLM_REPLY    = "🤖"

    # --- Specialized Tools ---
    TOOL_SEARCH  = "🌐"
    TOOL_DEEP    = "🔬"
    TOOL_CODE    = "🐍"
    TOOL_SHELL   = "🐚"
    TOOL_FILE_W  = "💾"
    TOOL_FILE_R  = "📖"
    TOOL_FILE_S  = "🔍"
    TOOL_FILE_I  = "👀"
    TOOL_DOWN    = "⬇️"
    TOOL_BROWSER = "🌎"   # headless browser automation (distinct from shell 🐚)
    IMAGE_GEN    = "🎨"
    REPORT_PDF   = "📄"
    NODE_WORKER  = "⚙️"   # background / edge worker-node compute
    NODE_EDGE    = "🛰️"   # swarm edge-node compute (NOT ⚡ — that's SYSTEM_BOOT)

    # --- Memory & Identity ---
    MEM_SAVE     = "📝"
    MEM_READ     = "🔎"
    MEM_MATCH    = "📍"
    MEM_INGEST   = "📚"
    MEM_SPLIT    = "📑"   # chunk split (distinct from CUT ✂️)
    MEM_EMBED    = "🧬"
    MEM_WIPE     = "🧹"
    MEM_SCRATCH  = "🗒️"
    USER_ID      = "👤"
    SELF_STATE   = "🪞"   # selfhood state / mood transition
    SKILL_GRADUATE = "🎓" # a lesson graduates into a reusable skill

    # --- Status ---
    OK           = "✅"
    FAIL         = "❌"
    WARN         = "⚠️"
    STOP         = "🛑"
    RETRY        = "🔄"
    IDEA         = "💡"
    BUG          = "🐛"
    SHIELD       = "🛡️"
    CUT          = "✂️"

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
    GRAPH_WEB        = "🕸️"   # triplet / knowledge-graph store
    VECTOR_EMBED     = "🧬"   # vector DB + sentence embeddings
    MEM_INDEX        = "🗃️"   # memory system fully loaded with items
    MEM_LIBRARY      = "📓"   # indexed fragments ready for recall (distinct from MEM_INGEST 📚)
    BELIEF_SCALES    = "⚖️"   # contradiction log / belief versioning
    THRESHOLD_TUNE   = "🎚️"   # adaptive recall threshold
    EPISODE_REEL     = "🎞️"   # episodic memory (sessions = frames)
    EVENT_BUS        = "📡"   # memory-bus pub/sub fan-out
    VERIFIER_LAB     = "🧪"   # self-evaluation gate
    UNCERTAINTY_DIE  = "🎲"   # uncertainty tracker
    MCTS_TREE        = "🌳"   # deep-reason MCTS search tree
    HEARTBEAT        = "🫀"   # biological watchdog heartbeat


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
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S'
    )

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
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


# Width of the visible prefix before the content column, i.e. the column
# where wrapped continuation lines should be indented to. Computed from:
# `│  TG  🙂  +12.3s  title_padded  ` with TITLE_WIDTH=18 and a 2-space
# separator. Emoji display width varies across terminals so this is a
# best-effort alignment, not a guarantee.
TITLE_WIDTH = 18
_CONTENT_COL = 39
_CONTINUATION_INDENT = " " * _CONTENT_COL


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


def pretty_log(title: str, content: Any = None, icon: str = "🔹", level: str = "INFO", special_marker: str = None):
    req_id = request_id_context.get()
    tag = _req_tag(req_id)
    rcol = _req_color(req_id)

    # ---- Lifecycle frames ------------------------------------------------
    if special_marker == "BEGIN":
        with _REQ_STATE_LOCK:
            _REQ_STATE[req_id] = {"started": time.monotonic()}
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        rule = "─" * 40
        line = (
            f"{rcol}┌─ {BOLD}{tag}{RESET}{rcol} {req_id[:8]}{RESET}  "
            f"{DIM}request started  {ts}{RESET} "
            f"{rcol}{rule}{RESET}"
        )
        atomic_print(line)
        return

    if special_marker == "END":
        delta = _format_delta(req_id).strip()
        with _REQ_STATE_LOCK:
            _REQ_STATE.pop(req_id, None)
        rule = "─" * 40
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
            f"{rcol}│{RESET}  {rcol}{tag}{RESET}  {icon}  "
            f"{DIM}{delta}{RESET}  "
            f"{BOLD}▼ {title_str}{RESET}"
        )
        atomic_print(line)
        return

    if special_marker == "SECTION_END":
        delta = _format_delta(req_id)
        title_str = title.lower().replace("_", " ")
        line = (
            f"{rcol}│{RESET}  {rcol}{tag}{RESET}  {icon}  "
            f"{DIM}{delta}{RESET}  "
            f"{BOLD}▲ {title_str}{RESET}"
        )
        atomic_print(line)
        return

    # ---- Normal log line -------------------------------------------------
    delta = _format_delta(req_id)
    title_str = title.lower().replace("_", " ")

    if content is None:
        content_str = ""
    elif isinstance(content, (dict, list)):
        try:
            content_str = repr(content) if len(content) > 50 else json.dumps(content, default=str)
        except Exception:
            content_str = str(content)
    else:
        content_str = str(content)

    # Failures get a larger content budget so the *why* (exception text, the
    # cause) survives on the monitored stream — not just in --verbose / the
    # file. INFO/DEBUG stay tight (60) to keep the stream scannable.
    _limit = LOG_TRUNCATE_LIMIT
    if level.upper() in ("WARNING", "WARN", "ERROR", "CRITICAL"):
        _limit = max(LOG_TRUNCATE_LIMIT, 240)
    content_str = _truncate(content_str, _limit).replace("\n", " ").replace("\r", "")
    content_str = _wrap_content(content_str)

    lvl_col = _LEVEL_COLOR.get(level.upper(), "")
    sep = "  " if content_str else ""
    line = (
        f"{rcol}│{RESET}  {rcol}{tag}{RESET}  {icon}  "
        f"{DIM}{delta}{RESET}  "
        f"{lvl_col}{BOLD}{_fit_title(title_str)}{RESET}{sep}{content_str}"
    )
    atomic_print(line)

    if DEBUG_MODE:
        # Lazy formatting via %-args so the logger only stringifies + slices
        # when the DEBUG handler is actually going to emit. The previous
        # f-string built the full string (and the full `repr(content)`)
        # eagerly, on EVERY call, which became a measurable cost when
        # `content` was a giant payload. `%.1000s` truncates at format time.
        logger.debug("[%s] %s: %.1000s", req_id, title, str(content))
