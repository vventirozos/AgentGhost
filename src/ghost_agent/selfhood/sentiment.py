"""The sentiment of the owner's messages, read by the separate sentiment-service on the worker node (§4TW).

The score FEEDS THE MOOD (operator 11-09, "option 1": the mood follows the owner's sentiment whatever it is
about): a confident positive or negative message is one small mood event (``mood.sentiment_event``, a third of a
thumb). Every score is also written to a journal file next to the mood, with what it did to the mood, so the
effect can be read back. (It ran as a shadow, journal only, for its first hours on 11-09.)

Everything here is best-effort and off the reply path: the call runs as a background task after the owner's
message is read, with a short timeout. A failing service answers nothing, and after ``BREAKER_FAILS`` failures
in a row the agent stops asking for ``BREAKER_PAUSE_S`` seconds, then tries again by itself. Nothing here raises
into a turn.

The service reads English only and answers ``unknown`` for anything else (Greek letters, Greek typed in Latin
letters, a word too short to tell); that answer is written too (it says the message was seen), with no
probabilities. Only the owner's own words are sent: a row a channel member wrote is never sent, and the notes the
clients add to a message (an attached file, a Slack mention) are taken out first.
"""
from __future__ import annotations

import asyncio
import datetime
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Dict, Optional

from ..utils.logging import logger

DEFAULT_KEY_FILE = "~/Data/AI/.sentiment_service_key"
KEY_HEADER = "X-Sentiment-Service-Key"
#: the service answers a short message in well under 100 ms; a slow answer is a sick service. httpx applies
#: TIMEOUT_S to each stage (connect, write, read), so the whole call has its own cap (review r3)
TIMEOUT_S = 3.0
TOTAL_TIMEOUT_S = 5.0
BREAKER_FAILS = 3
BREAKER_PAUSE_S = 300.0
#: the service's own limit (sentiment_service.MAX_TEXT_CHARS); longer text is cut, not refused
MAX_TEXT_CHARS = 4000
JOURNAL_FILENAME = "sentiment.jsonl"
#: the journal keeps the start of each message, enough to read a misjudged score back
JOURNAL_TEXT_CHARS = 300
#: past this the journal is moved to ``.1`` (one old file kept) — ~50 messages a day is ~20 KB
JOURNAL_MAX_BYTES = 5 * 1024 * 1024
LABELS = ("negative", "neutral", "positive", "unknown")


def settings() -> tuple:
    return (os.environ.get("GHOST_SENTIMENT_URL", "").strip().rstrip("/"),
            os.environ.get("GHOST_SENTIMENT_KEY_FILE", DEFAULT_KEY_FILE).strip())


def _read_key(key_file: str) -> str:
    try:
        return Path(os.path.expanduser(key_file)).read_text().strip()
    except OSError:
        return ""


def configured() -> bool:
    """True when the URL is set and the key file holds a key. Configuration only, never reachability."""
    url, key_file = settings()
    return bool(url and _read_key(key_file))


class _Breaker:
    """What the agent last knew of the service, process-wide. Written from background tasks on the event loop
    and read by the health check, so a lock guards it."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self.fails = 0
        self.paused_until = 0.0
        self.scored = 0
        self.moved = 0
        self.last_ok = 0.0
        self.last_fail = 0.0
        self.last_fail_why = ""

    def may_ask(self, now: float) -> bool:
        with self.lock:
            return now >= self.paused_until

    def ok(self, now: float) -> None:
        with self.lock:
            self.fails = 0
            self.paused_until = 0.0
            self.scored += 1
            self.last_ok = now

    def failed(self, why: str, now: float) -> None:
        with self.lock:
            self.fails += 1
            self.last_fail = now
            self.last_fail_why = why
            if self.fails >= BREAKER_FAILS:
                self.paused_until = now + BREAKER_PAUSE_S
                self.fails = 0           # the next round after the pause gets its own BREAKER_FAILS tries


BREAKER = _Breaker()


def _answer(body) -> Optional[Dict[str, object]]:
    """The service's answer as {label, score, probs, reason}, or None when it isn't one."""
    if not isinstance(body, dict) or body.get("ok") is not True:
        return None
    label = body.get("label")
    score = body.get("score")
    probs = body.get("probs")
    if label not in LABELS or isinstance(score, bool) or not isinstance(score, (int, float)):
        return None
    if not isinstance(probs, dict) or not all(
            k in LABELS and isinstance(v, (int, float)) and not isinstance(v, bool) for k, v in probs.items()):
        return None
    return {"label": label, "score": float(score), "probs": {k: float(v) for k, v in probs.items()},
            "reason": str(body.get("reason") or "")[:40]}


async def classify(text: str, *, tor_proxy: Optional[str] = None,
                   now_fn=time.monotonic) -> Optional[Dict[str, object]]:
    """Ask the service. None = no answer (not configured, paused after failures, or a failed call). Never
    raises."""
    url, key_file = settings()
    if not url or not isinstance(text, str):
        return None
    if not BREAKER.may_ask(now_fn()):
        return None
    key = _read_key(key_file)
    if not key:       # review r2: said on the health line, not silently nothing
        BREAKER.failed("no key in the key file", now_fn())
        return None
    try:
        import httpx
        from ..core.llm import compute_tor_proxy
        target = f"{url}/classify"
        # a tailnet address stays direct and anything public goes through Tor: the agent's normal rule
        proxy = compute_tor_proxy(target, tor_proxy)
        from ..utils.aio import wait_for
        async with httpx.AsyncClient(proxy=proxy, timeout=TIMEOUT_S, follow_redirects=False) as client:
            response = await wait_for(client.post(target, json={"text": text[:MAX_TEXT_CHARS]},
                                                  headers={KEY_HEADER: key}), TOTAL_TIMEOUT_S)
        answer = _answer(response.json()) if response.status_code == 200 else None
        why = "" if answer is not None else f"HTTP {response.status_code}"
    except Exception as exc:  # noqa: BLE001 — the type only: a message could quote the text
        answer, why = None, type(exc).__name__
    if answer is None:
        BREAKER.failed(why or "unreadable answer", now_fn())
        logger.debug("sentiment: no answer (%s)", why or "unreadable answer")
        return None
    BREAKER.ok(now_fn())
    return answer


#: what the clients add to the owner's message: the Slack bot's file notes, the web client's upload line, a Slack
#: user mention (``<@U0123>``, ``<@U0123|name>``)
_CLIENT_NOTES = (
    # anchored on the notes' fixed endings: a file name may itself hold ".]" (review r2)
    re.compile(r"\[SYSTEM NOTE: The user attached a file named .*?(?:"
               r"to interact with it\.\]|vision_analysis \(target='.*?'\)\.\]|"
               r"could NOT be ingested \(too large or the fetch failed\)\.\]|found inside it\.\])", re.DOTALL),
    re.compile(r"\(File just uploaded to the sandbox: [^\n]*\)"),
    re.compile(r"<@[A-Z0-9]+(?:\|[^>]*)?>"),
)


def message_text(content) -> str:
    """The owner's words in a chat message: a string as it is, the text parts of a multimodal list, with the
    clients' notes taken out. "" for a row another person wrote (a channel member's, labelled by the bot)."""
    from ..utils.logging import FOREIGN_MESSAGE_LABELS
    if isinstance(content, list):
        content = "\n".join(str(p.get("text") or "") for p in content
                            if isinstance(p, dict) and p.get("type") == "text")
    if not isinstance(content, str) or content.lstrip().startswith(FOREIGN_MESSAGE_LABELS):
        return ""
    for note in _CLIENT_NOTES:
        content = note.sub(" ", content)
    return content.strip()


_JOURNAL_LOCK = threading.Lock()


def _append(path: Path, row: Dict[str, object]) -> None:
    with _JOURNAL_LOCK:      # review r2: two writers at the size limit moved the full file over ".1" twice
        _append_locked(path, row)


def _append_locked(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if path.stat().st_size > JOURNAL_MAX_BYTES:
            os.replace(path, path.with_name(path.name + ".1"))
    except FileNotFoundError:
        pass
    line = json.dumps(row, ensure_ascii=False) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)


HELD_PRAISED = "the praise check counted it"
HELD_NO_MOOD = "no mood to move"


def _feed_mood(answer: Dict[str, object], self_model, praised: bool, arrived: Optional[datetime.datetime] = None,
               request_id: str = ""):
    """(the pleasantness nudge applied, or 0.0; why none, or ""). Never raises.

    §4TX: through the reaction ledger (``SelfModel.note_owner_sentiment``) — the unhappy tone of a message that
    also corrected the reply is one verdict with the correction, not two — and held while the owner's set mood
    holds, or when the owner set the mood after this message arrived (the reasons are the state's own)."""
    from . import mood as M
    try:
        ev = M.sentiment_event(answer.get("label"), answer.get("score"))
        if ev is None:
            label = answer.get("label")
            if label == "unknown":
                return 0.0, str(answer.get("reason") or "unknown")
            if label == "neutral":
                return 0.0, "neutral"
            return 0.0, f"below {M.SENTIMENT_MIN_SCORE:.2f}"
        if praised:              # "thanks, perfect" is already one praise event
            return 0.0, HELD_PRAISED
        if self_model is None:
            return 0.0, HELD_NO_MOOD
        after, dp, why = self_model.note_owner_sentiment(
            answer["label"], answer["score"], request_id=str(request_id or ""),
            arrived=(arrived.isoformat().replace("+00:00", "Z") if arrived is not None else ""))
        if why:
            return 0.0, why
        if after is None:
            return 0.0, HELD_NO_MOOD
        with BREAKER.lock:
            BREAKER.moved += 1
        return dp, ""
    except Exception as exc:  # noqa: BLE001
        logger.debug("sentiment mood event skipped: %s", type(exc).__name__)
        return 0.0, HELD_NO_MOOD


async def score_message(content, root, *, request_id: str = "", tor_proxy: Optional[str] = None,
                        self_model=None, praised: bool = False) -> bool:
    """Score one OWNER message (the caller decides that), feed the mood, write the journal row, log the rating.
    ``praised``: the praise check already counted this message. True when a row was written. Never raises."""
    try:
        arrived = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
        text = message_text(content).strip()
        if not text:
            return False
        answer = await classify(text, tor_proxy=tor_proxy)
        if answer is None:
            return False
        nudge, held = _feed_mood(answer, self_model, praised, arrived, request_id)
        row = {"ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
               "request_id": str(request_id or "")[:64], **answer, "nudge": nudge, "held": held,
               "chars": len(text), "text": text[:JOURNAL_TEXT_CHARS]}
        await asyncio.to_thread(_append, Path(root) / JOURNAL_FILENAME, row)
        log_rating(answer, nudge, held)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("sentiment scoring skipped: %s", type(exc).__name__)
        return False


def rating_line(answer: Dict[str, object], nudge: float = 0.0, held: str = "") -> str:
    """The log line's content: the rating and what it did to the mood, never the words.
    ``positive 0.95 (neg 0.01 · neu 0.04) · mood +0.08``, ``neutral 0.89 (…) · mood unmoved (neutral)``,
    ``unknown — not English · mood unmoved``."""
    label = str(answer.get("label") or "")
    if nudge:
        effect = f"mood {nudge:+.2f}".replace("-", "−")
    else:
        effect = "mood unmoved" + (f" ({held})" if held and label != "unknown" else "")
    if label == "unknown":
        return f"unknown — {answer.get('reason') or 'no reason given'} · {effect}"
    probs = answer.get("probs") or {}
    others = " · ".join(f"{k[:3]} {float(v):.2f}" for k, v in probs.items() if k != label)
    return f"{label} {float(answer.get('score') or 0):.2f}" + (f" ({others})" if others else "") + f" · {effect}"


def log_rating(answer: Dict[str, object], nudge: float = 0.0, held: str = "") -> None:
    """One line in the agent's log, with its own icon. Never raises."""
    try:
        from ..utils.logging import Icons, pretty_log
        pretty_log("Sentiment", rating_line(answer, nudge, held), icon=Icons.SENTIMENT)
    except Exception as exc:  # noqa: BLE001
        logger.debug("sentiment log line skipped: %s", type(exc).__name__)


def journal_rows(root, *, limit: int = 0):
    """The journal's rows, oldest first (the newest ``limit`` when given). Unreadable lines are skipped."""
    path = Path(root) / JOURNAL_FILENAME
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows = []
    for line in lines[-limit:] if limit else lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _ago(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 90:
        return f"{int(seconds)} s ago"
    if seconds < 5400:
        return f"{int(seconds // 60)} min ago"
    return f"{seconds / 3600:.1f} h ago"


def health_text(*, now_fn=time.monotonic) -> Optional[str]:
    """One part for the health check: what the agent last knew, never a live test. None when no URL is set.
    Reads no file: the health check calls it on the event loop."""
    if not settings()[0]:
        return None
    now = now_fn()
    with BREAKER.lock:
        scored, moved, last_ok, last_fail = BREAKER.scored, BREAKER.moved, BREAKER.last_ok, BREAKER.last_fail
        why, paused_until = BREAKER.last_fail_why, BREAKER.paused_until
    text = f"sentiment service: {scored} scored since start, {moved} moved the mood"
    if now < paused_until:
        return text + f", PAUSED after failures (last: {why}, {_ago(now - last_fail)}; retries by itself)"
    if last_fail and last_fail > last_ok:
        return text + f", last call failed ({why}, {_ago(now - last_fail)})"
    return text
