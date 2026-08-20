"""Unified structured logging for the metacognition uplift.

Single concern: every uplift event reaches the operator's terminal in
ONE consistent shape so monitoring greps and log parsers don't have to
juggle five different formats.

Shape::

    metacog <subsystem>    key=value key=value ...

Why this shape:
  * ``metacog`` as the universal grep prefix. ``grep "metacog "`` shows
    every uplift event; ``grep "metacog conf"`` narrows to confidence,
    ``grep "metacog host"`` to telemetry, etc.
  * Space-separated ``key=value`` pairs play nicely with awk, jq-style
    log shippers, and the human eye.
  * Strings with spaces are auto-quoted; ints and floats are formatted
    with a fixed precision so a regex on ``C=0.\d{2}`` actually works.

Severity rule (so monitoring dashboards bucket cleanly):
  * INFO    — successful state transitions (boot ok, validator passed,
              confidence above threshold-in-debug-mode).
  * WARNING — soft anomalies (validator block, host signal warning,
              arbiter ask_user, replan rejected).
  * ERROR   — hard anomalies (host signal critical, bundle init
              failure, irrecoverable shutdown error).
  * DEBUG   — high-volume signals that are noise during normal
              operation (per-turn confidence above τ, competence write,
              entropy stash).
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

logger = logging.getLogger("GhostAgent")


# Subsystem labels — keep these short, ALL identical case, no spaces in
# the label part itself. Operators grep them, so stability matters more
# than verbosity.
class Subsystem:
    BOOT = "boot"          # lifecycle: enable / disable / shutdown
    CONF = "conf"          # composite confidence reading
    CALIB = "calib"        # calibration spine: Brier / ECE / refit
    ARBITER = "arbiter"    # dual-solver arbiter decision
    VALID = "valid"        # pre-execution validator verdict
    HOST = "host"          # host telemetry signal
    REPLAN = "replan"      # replan bridge attempt
    GATE = "gate"          # gate skip reasons (debug only)
    SUMMARY = "summary"    # shutdown / periodic rollup


# Per-subsystem icons — REGISTERED in ``utils.logging.Icons`` since §LOG-7
# (2026-08-20). The previous literal map was a registry bypass with three
# glyph COLLISIONS against registry meanings (🧮 arbiter vs VECTOR_EMBED,
# 🔀 replan vs EVENT_BUS, 🧠 default vs BRAIN_SUM), and its unregistered
# glyphs were unknown to the interface ICON_CLASS maps, so every metacog
# line degraded to the generic "think" class on the uConsole/web tickers.
# One source of truth: the registry; this map only picks per subsystem.
from ..utils.logging import Icons as _Icons

_SUBSYSTEM_ICONS = {
    Subsystem.BOOT:    _Icons.METACOG_BOOT,      # 🌱 sprout — lifecycle
    Subsystem.SUMMARY: _Icons.METACOG_SUMMARY,   # 📊 bar chart — rollup
    Subsystem.CONF:    _Icons.METACOG_CONF,      # 📈 trend — confidence
    Subsystem.CALIB:   _Icons.METACOG_CALIB,     # 📐 ruler — Brier refit
    Subsystem.ARBITER: _Icons.METACOG_ARBITER,   # 🥇 winner — arbiter pick
    Subsystem.VALID:   _Icons.METACOG_VALID,     # 🚧 barrier — pre-exec block
    Subsystem.HOST:    _Icons.METACOG_HOST,      # 💻 laptop — host signal
    Subsystem.REPLAN:  _Icons.METACOG_REPLAN,    # 🚦 signals — replan bridge
    Subsystem.GATE:    _Icons.METACOG_GATE,      # 🚪 door — gate skip (debug)
}
_DEFAULT_ICON = _Icons.METACOG   # 🫧 generic metacog (unknown subsystem)


# Levels that map directly to pretty_log's `level` kwarg.
LEVEL_INFO = "INFO"
LEVEL_WARN = "WARNING"
LEVEL_ERROR = "ERROR"
LEVEL_DEBUG = "DEBUG"


def emit(
    subsystem: str,
    *,
    level: str = LEVEL_INFO,
    icon: Optional[str] = None,
    **fields: Any,
) -> None:
    """Emit one structured uplift log line.

    Routes through ``pretty_log`` when available (so the line lines up
    with the rest of the agent's UI) and falls back to ``logger`` when
    not. Never raises — a logging failure must not break a turn.

    Example::

        emit(Subsystem.ARBITER,
             level=LEVEL_WARN,
             tool="execute", action="ask_user",
             sim=0.42, reason="diverged plans")

    Renders as (roughly)::

        metacog arbiter    tool=execute action=ask_user sim=0.42 reason="diverged plans"
    """
    try:
        title = f"Metacog {subsystem.capitalize()}"
        content = _format_fields(fields)
        try:
            from ..utils.logging import pretty_log
            # Pick per-subsystem icon so each line is visually distinct
            # from other metacog subsystems AND from the rest of the
            # agent's log stream. Caller may override via ``icon=``.
            _icon = icon or _SUBSYSTEM_ICONS.get(subsystem, _DEFAULT_ICON)
            pretty_log(title, content, icon=_icon, level=level)
        except Exception:
            # Fallback when pretty_log can't be imported (e.g. early
            # bootstrap, isolated tests).
            getattr(logger, level.lower(), logger.info)(
                "%s  %s", title.lower(), content,
            )
    except Exception as exc:  # pragma: no cover — defensive
        # Last-ditch: never let a logging crash propagate.
        try:
            logger.debug("metacog log emit failed: %s", exc)
        except Exception:
            pass


def _format_fields(fields: Mapping[str, Any]) -> str:
    """Format ``key=value`` pairs deterministically.

    * Floats round to 2 decimals (operators don't need more precision
      and a regex like ``C=0\\.\\d{2}`` should match every confidence
      line). Pass a pre-formatted string if you need different
      precision.
    * Strings with spaces get double-quoted so a parser can split on
      whitespace without losing words. Embedded quotes get escaped.
    * Booleans render as ``yes`` / ``no`` (cheaper to scan than
      ``True`` / ``False`` and matches the "is this happening" framing
      operators use).
    * ``None`` values render as ``-``.

    Field order is preserved — Python 3.7+ dicts are insertion-ordered,
    so the call site controls log column order.
    """
    parts = []
    for k, v in fields.items():
        parts.append(f"{k}={_fmt_value(v)}")
    return " ".join(parts)


def _fmt_value(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        if v != v:  # NaN
            return "nan"
        return f"{v:.2f}"
    s = str(v)
    if not s:
        return '""'
    # Quote strings that contain whitespace or quotes
    needs_quote = any(c in s for c in (" ", "\t", "\n", '"'))
    if needs_quote:
        return '"' + s.replace('"', '\\"') + '"'
    return s


__all__ = [
    "emit",
    "Subsystem",
    "LEVEL_INFO",
    "LEVEL_WARN",
    "LEVEL_ERROR",
    "LEVEL_DEBUG",
]
