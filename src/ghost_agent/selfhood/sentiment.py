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
        #: §4UB (evaluation: an outage was logged at DEBUG only): the log was told the service is down
        self.down = False

    def may_ask(self, now: float) -> bool:
        with self.lock:
            return now >= self.paused_until

    def ok(self, now: float) -> bool:
        """True when this answer ends an outage the log was told about (§4UB)."""
        with self.lock:
            self.fails = 0
            self.paused_until = 0.0
            self.scored += 1
            self.last_ok = now
            back, self.down = self.down, False
            return back

    def failed(self, why: str, now: float) -> bool:
        """True when this failure pauses the asking (§4UB: said once, at WARNING)."""
        with self.lock:
            self.fails += 1
            self.last_fail = now
            self.last_fail_why = why
            if self.fails >= BREAKER_FAILS:
                self.paused_until = now + BREAKER_PAUSE_S
                self.fails = 0           # the next round after the pause gets its own BREAKER_FAILS tries
                opened, self.down = not self.down, True
                return opened
            return False


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
        _failed("no key in the key file", now_fn())
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
        _failed(why or "unreadable answer", now_fn())
        logger.debug("sentiment: no answer (%s)", why or "unreadable answer")
        return None
    if BREAKER.ok(now_fn()):
        logger.info("sentiment service answering again; the owner's tone feeds the mood again")
    return answer


def _failed(why: str, now: float) -> None:
    """One failed call. The pause it starts is said ONCE, at WARNING — never the text."""
    if BREAKER.failed(why, now):
        logger.warning("sentiment service: %d calls in a row failed (last: %s); not asked for %d min, then "
                       "retried by itself — the mood runs without the owner's tone meanwhile",
                       BREAKER_FAILS, why, int(BREAKER_PAUSE_S // 60))


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


#: §4UB (evaluation: a pasted log, code or an article was scored as the owner's tone): what is not the owner's
#: own words. A fenced block (an unclosed fence runs to the end), a quoted line, a log or traceback line
_FENCE = re.compile(r"```.*?(?:```|\Z)", re.DOTALL)
_PASTED_LINE = re.compile(
    r"^\s*(?:>|\$ |>>> |Traceback \(most recent call last\)|File \"[^\"]+\", line \d|at [\w$<>]+(?:\.[\w$<>]+)+\(|"
    # review r2: a level word opens a log line in its log forms only ("[ERROR] x", "ERROR:root:x", "ERROR 12:"),
    # not "ERROR: nothing works and I'm furious" or "DEBUG this please"
    r"\[(?:ERROR|WARN|WARNING|INFO|DEBUG|FATAL|CRITICAL|TRACE)\]|"
    r"(?:ERROR|WARN|WARNING|INFO|DEBUG|FATAL|CRITICAL|TRACE)(?::[\w.]+:|\s+\d{4}-\d\d-\d\d|\s+\d\d:\d\d:\d\d)|"
    # review r1: a time of day ("10:30 works for me", "2026-11-10 10:00 sounds perfect") is the owner's;
    # a log stamp has seconds AND a level, a fraction or a bracket after it
    r"(?:\d{4}-\d\d-\d\d[T ])?\d\d:\d\d:\d\d(?:[.,]\d+|\S*\s+\[?(?:ERROR|WARN|WARNING|INFO|DEBUG|FATAL|"
    r"CRITICAL|TRACE)\b|Z?\s*\[)|[A-Za-z_][\w.]*(?:Error|Exception|Warning):)")
#: "summarise this: <an article>" — what follows the colon is somebody else's text. Review r1: only verbs that
#: hand text over to be worked on ("check this out: amazing news" and "I read your summary: brilliant" are the
#: owner's own words)
_PASTE_AFTER = re.compile(
    # review r2: a short lead ("hey ghost, can you"), up to 80 characters of instruction, then a colon — or a
    # line break ("summarise this\n\n<article>")
    r"^(.{0,30}?\b(?:summari[sz]e|translate|proofread|rewrite|rephrase|paraphrase|tl;?dr)\b[^:\n]{0,80})"
    r"(?::\s*\S|[ \t]*\n[ \t]*\n\s*\S)", re.IGNORECASE | re.DOTALL)
#: all capitals read as shouting: the service answered "language unclear" for "WHY IS THIS STILL BROKEN"
_SHOUT_MIN_LETTERS = 8
_SHOUT_SHARE = 0.7


def prepare_text(text: str) -> str:
    """§4UB: the part of an owner message that is the owner's own tone — pasted code, logs, quoted lines and
    text handed over to work on ("summarise this: …") are taken out, and a message in capitals is read in
    lower case (shouting is a tone the model reads; capitals it could not). "" when nothing is left."""
    if not isinstance(text, str):
        return ""
    text = _FENCE.sub("\n", text)
    text = "\n".join(line for line in text.splitlines() if not _PASTED_LINE.match(line))
    m = _PASTE_AFTER.match(text)
    if m:
        text = m.group(1)
    letters = [c for c in text if c.isalpha()]
    if len(letters) >= _SHOUT_MIN_LETTERS and sum(c.isupper() for c in letters) >= _SHOUT_SHARE * len(letters):
        text = text.lower()
    return re.sub(r"\n{2,}", "\n", re.sub(r"[ \t]+", " ", text)).strip()


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
#: §4UB (evaluation: "great, you broke it again" scored positive 0.94; "delete the failed jobs" negative 0.87)
HELD_SARCASM = "upbeat words about trouble (it may be sarcasm)"
HELD_TASK = "the wording of a task, not a feeling"
#: words that turn an upbeat score next to them into a doubtful one
_TROUBLE_WORDS = re.compile(
    r"\b(?:broke|broken|breaks?|breaking|crash\w*|wrong|fail\w*|ruin\w*|wasted?|worse|worst|useless|"
    r"mess(?:ed)?|bug(?:s|gy)?|errors?)\b", re.IGNORECASE)
#: review r1: good news about a problem gone ("you fixed the bug, awesome", "no errors this time, perfect")
_RESOLVED = re.compile(
    r"\b(?:fixed|fixes|fixing|found|solved|gone|went away|no longer|it works|it'?s working|everything works|"
    r"works like|(?:tests?|build|checks?) pass(?:es|ed)?(?: now)?|clearer|nothing (?:went wrong|failed|broke)|resolved|sorted|no (?:more )?(?:errors?|bugs?|crash\w*|failures?|problems?|"
    r"issues?)|not (?:broken|failing) any ?more|"
    r"works? (?:now|again|fine|great)|working (?:now|again|fine|great))\b", re.IGNORECASE)
#: review r1: a message that corrects the reply while sounding upbeat ("nice try, but that's not what I asked")
#: — the correction judge marks the trouble only after the score arrives
_CORRECTING = re.compile(
    r"(?:\b(?:not what i (?:asked|meant|wanted|said)|that'?s not (?:it|right|correct|true)|nice try)|"
    r"(?:^|[.!,;]\s*)(?:no,?\s*)?try again\b(?!\s+(?:later|tomorrow|soon)))|(?:"
    r"\byou missed|\byou misunderstood|\bnot quite)\b", re.IGNORECASE)
#: a message that opens with an instruction: its words are the task's ("kill the failed jobs"). Review r1: the
#: verb must lead into its object — "Look, this is awful" and "Stop, this is terrible" are interjections
_TASK_OPENING = re.compile(
    r"^\W*(?:(?:please|pls|ok|okay|now|so|then|and|also|can you|could you|would you|will you|go ahead and|"
    r"just)[\s,]+)*(?:delete|kill|fix|remove|clean|check|terminate|find|debug|purge|write|stop|cancel|drop|"
    r"reset|restart|clear|undo|revert|run|retry|list|show|search|look|block|ban|abort)\b(?!\s*[,.!?;:—-])",
    re.IGNORECASE)
#: … and an instruction that carries a feeling of its own is the owner's feeling ("check the date, it is wrong
#: again, I am fed up")
_FEELING = re.compile(
    r"\b(?:hate|awful|terrible|horrible|fed up|sick of|tired of|annoy\w*|frustrat\w*|ugh+|wtf|ridiculous|"
    r"furious|angry|upset\w*|disappoint\w*|useless|sucks?|sad|heartbreak\w*|stress\w*|worr\w*|depress\w*|"
    r"scar(?:y|ed)|afraid|mess|ruin\w*|desperate|losing|nothing works|keeps? (?:crash|fail|break)\w*|"
    r"again\W*$)", re.IGNORECASE)

def doubtful(label: str, text: str, *, after_trouble: bool = False) -> str:
    """§4UB: why a confident score is NOT the owner's feeling, or "". An upbeat score on a message that talks
    about trouble that is not gone, that corrects the reply, or that comes right after the agent's own trouble
    may be sarcasm; an unhappy score on a message that opens with an instruction and carries no feeling of its
    own is the task's wording."""
    text = text or ""
    if label == "positive":
        if _CORRECTING.search(text):
            return HELD_SARCASM
        # review r2: good news about a problem gone counts, even right after the agent's trouble (a bug-fixing
        # turn usually has a failed call in it)
        if not _RESOLVED.search(text) and (after_trouble or _TROUBLE_WORDS.search(text)):
            return HELD_SARCASM
    if label == "negative" and _TASK_OPENING.match(text) and not _FEELING.search(text):
        return HELD_TASK
    return ""


def _feed_mood(answer: Dict[str, object], self_model, praised: bool, arrived: Optional[datetime.datetime] = None,
               request_id: str = "", text: str = ""):
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
        after_trouble = False
        if answer.get("label") == "positive":
            try:
                after_trouble = bool(self_model.recent_trouble(arrived))
            except Exception:  # noqa: BLE001
                after_trouble = False
        doubt = doubtful(str(answer.get("label") or ""), text, after_trouble=after_trouble)
        if doubt:
            return 0.0, doubt
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
        text = prepare_text(message_text(content))
        if not text:
            return False
        answer = await classify(text, tor_proxy=tor_proxy)
        if answer is None:
            return False
        nudge, held = _feed_mood(answer, self_model, praised, arrived, request_id, text)
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
    ``positive 0.95 (neg 0.01 · neu 0.04) · mood +0.08``, ``neutral 0.89 (…) · mood unmoved — message neutral``,
    ``unknown — not English · mood unmoved``. A neutral message says so in words: a bare ``(neutral)`` read as the
    MOOD being neutral while the owner's set mood held."""
    label = str(answer.get("label") or "")
    if nudge:
        effect = f"mood {nudge:+.2f}".replace("-", "−")
    elif held == "neutral" and label == "neutral":
        effect = "mood unmoved — message neutral"
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
