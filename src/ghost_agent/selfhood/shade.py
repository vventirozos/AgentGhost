"""The mood's shade (§4UG): what a mood is ABOUT, for one moment — relieved, chastened, proud.

The mood is two numbers and one of nine labels. "content" reads the same whether a thumb made it, a task
got finished, or something that had failed three times finally went through; the cause is known when the
event arrives and is then lost. A shade keeps it for a moment, BESIDE the label (the nine labels, their
looks on the face and everything that reads them stay as they are):

* ``relieved`` — a turn that went well within ``RELIEF_WINDOW_S`` of a turn of the agent's own that went
  badly (failed calls, a loop guard). A sequence, not one event;
* ``proud`` — a turn in which a project task was finished;
* ``chastened`` — the owner corrected one of its replies or gave one a thumbs-down.

A shade is set only by an event that MOVED the mood (an event held by the owner's set mood sets none), it
lasts ``TTL_S``, and a later shade replaces it. It is heard ONCE: the next reply to the owner on a private
surface carries one short line (``line``), and no reply after it does — a moment, not a state. The face
shows the same moment (``/api/health`` carries the shade and when it was set). It never moves the pair, the
reaction ledger or what anyone else hears.

Storage: ``$GHOST_HOME/system/selfhood/shade.json`` (the current one; a restart keeps it) and
``shades.jsonl`` (every shade set, for the day record). Best-effort: a failed write costs the shade only.
"""

from __future__ import annotations

import datetime
import json
import logging
import threading
from pathlib import Path
from typing import Dict, Optional

from . import mood as M

logger = logging.getLogger("GhostSelfhood")

RELIEVED, CHASTENED, PROUD = "relieved", "chastened", "proud"
SHADES = (RELIEVED, CHASTENED, PROUD)
#: how long a shade is there to be heard or shown
TTL_S = 1800.0
#: a good turn this long after the agent's own bad one is relief
RELIEF_WINDOW_S = 1800.0
SHADE_FILENAME = "shade.json"
SHADES_LOG = "shades.jsonl"
SHADES_LOG_MAX_ROWS = 2000
SHADES_LOG_KEEP_ROWS = 1500

#: the one line the next reply hears. Voice only: it changes no fact, no tool use, no answer.
_WHAT = {
    RELIEVED: "relief — something of yours that had gone wrong has just gone through",
    PROUD: "a quiet pride — a piece of work you were carrying is finished",
    # §4UH-e2e review: not "your LAST reply" — a correction is judged in the background after the reply to it
    # has started (its own reply cannot hear it), so the reply that does is one later; a thumb may be on any reply
    CHASTENED: "you are a little chastened — the owner has just marked one of your replies down",
}
LINE = ("RIGHT NOW, FOR THIS REPLY ONLY: {what}. Let it show lightly in HOW you phrase this reply — a word, "
        "a turn of phrase — and do not announce it or explain it. WHAT you say stays exactly as it would be: "
        "the same facts, checks, tool use and every point the answer needs.")


def line(shade: str) -> str:
    """The state-block line for ``shade`` ("" for anything that is not one)."""
    return LINE.format(what=_WHAT[shade]) + "\n\n" if shade in _WHAT else ""


def for_turn(signals, *, moved_dp: float, own_trouble_age_s: Optional[float]) -> str:
    """The shade a finished owner turn sets, or "". ``moved_dp``: what the turn's events added to the
    pleasantness; ``own_trouble_age_s``: seconds since the agent's own last bad turn (None = none)."""
    try:
        if int(getattr(signals, "tasks_done", 0) or 0) > 0 and moved_dp > 0:
            return PROUD
        if (moved_dp > 0 and own_trouble_age_s is not None
                and 0.0 <= own_trouble_age_s <= RELIEF_WINDOW_S):
            return RELIEVED
    except (TypeError, ValueError):
        return ""
    return ""


def for_reaction(kind: str) -> str:
    """The shade an owner's reaction sets, or ""."""
    return CHASTENED if kind in (M.KIND_THUMB_DOWN, M.KIND_CORRECTION) else ""


class ShadeStore:
    """The current shade. One per self-model; thread-safe; never raises."""

    def __init__(self, root: Optional[Path], *, enabled: bool = True):
        self.root = Path(root) if root else None
        self.enabled = bool(enabled and self.root is not None)
        self._lock = threading.Lock()
        self._cur: Optional[Dict[str, str]] = None
        self._heard_at = ""              # the `at` of the shade a reply already carried
        self._loaded = False

    @staticmethod
    def _iso(now=None) -> str:
        now = now or datetime.datetime.now(datetime.timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=datetime.timezone.utc)
        # microseconds: the stamp is the shade's identity (review r1: two shades in one second, the first
        # already heard, and the second was never heard)
        return now.astimezone(datetime.timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")

    def _load_locked(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.enabled:
            return
        try:
            raw = json.loads((self.root / SHADE_FILENAME).read_text(encoding="utf-8"))
            if isinstance(raw, dict) and raw.get("shade") in SHADES and raw.get("at"):
                self._cur = {"shade": str(raw["shade"]), "at": str(raw["at"]), "cause": str(raw.get("cause") or "")}
                self._heard_at = str(raw.get("heard_at") or "")
        except FileNotFoundError:
            pass
        except Exception as e:  # noqa: BLE001
            logger.debug("shade: %s unreadable (%s)", SHADE_FILENAME, type(e).__name__)

    def _save_locked(self) -> None:
        if not self.enabled:
            return
        try:
            from ..utils.json_store import write_json_atomic
            self.root.mkdir(parents=True, exist_ok=True)
            write_json_atomic(self.root / SHADE_FILENAME, {**(self._cur or {}), "heard_at": self._heard_at})
        except Exception as e:  # noqa: BLE001
            logger.debug("shade: not saved (%s)", type(e).__name__)

    def set(self, shade: str, cause: str = "", *, now=None) -> bool:
        """Set the shade (replacing any). False for a name that is not a shade."""
        if shade not in SHADES:
            return False
        with self._lock:
            self._load_locked()
            self._cur = {"shade": shade, "at": self._iso(now), "cause": str(cause or "")[:160]}
            self._save_locked()
            if self.enabled:
                try:
                    from ..utils.json_store import open_append
                    from .schema import read_jsonl_rows
                    path = self.root / SHADES_LOG
                    with open_append(path) as f:
                        f.write(json.dumps(self._cur, ensure_ascii=False) + "\n")
                    # the file is read back only when it is big enough to need cutting (~110 bytes a row)
                    rows = (read_jsonl_rows(path, what="shades")
                            if path.stat().st_size > SHADES_LOG_MAX_ROWS * 110 else [])
                    if len(rows) > SHADES_LOG_MAX_ROWS:
                        tmp = path.with_name(path.name + ".tmp")
                        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                               for r in rows[-SHADES_LOG_KEEP_ROWS:]), encoding="utf-8")
                        tmp.replace(path)
                except Exception as e:  # noqa: BLE001
                    logger.debug("shade: log row not written (%s)", type(e).__name__)
        return True

    def current(self, *, now=None) -> Optional[Dict[str, str]]:
        """{"shade", "at", "cause"} while it is at most ``TTL_S`` old, else None."""
        with self._lock:
            self._load_locked()
            cur = dict(self._cur) if self._cur else None
        if not cur:
            return None
        at = M.parse_iso_utc(cur["at"])
        now = now or datetime.datetime.now(datetime.timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=datetime.timezone.utc)
        if at is None or not (-60.0 <= (now - at).total_seconds() <= TTL_S):
            return None
        return cur

    def take(self, *, now=None) -> str:
        """The shade for ONE reply: the current shade the first time it is asked for, "" after that."""
        cur = self.current(now=now)
        if not cur:
            return ""
        with self._lock:
            if self._heard_at == cur["at"]:
                return ""
            self._heard_at = cur["at"]
            self._save_locked()
        return cur["shade"]
