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

§4VE: the answer may also carry what the message is DOING (``intent``) and whether it is urgent — see
``selfhood/intent.py`` for the switches. The one exception to "the owner's words only" is ``read_only``: a
member's own message, read for the pitch of the reply to them, only once ``GHOST_INTENT_MEMBERS=1`` (off by
default; nothing of it is kept or journalled).
"""
from __future__ import annotations

import asyncio
import collections
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
#: §4UH-warmth: the emotion model's labels (GoEmotions: 27 emotions + neutral). The service answers ``emotions``
#: {label: probability} for those at 0.05 or more; an unknown label or a bad number drops the whole field
EMOTION_LABELS = frozenset((
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire",
    "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy",
    "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise",
    "neutral"))
#: what the character's WARMTH reads (``days`` counts a message for a side when one of its labels is at
#: ``EMOTION_COUNT_MIN`` or more): how the owner talks to it. Not curiosity or confusion (questions), not sadness,
#: fear, grief or nervousness (the owner's own news, not how they are with Ghost)
#: §4UP: ONE list, owned by the mood (its tone gate reads the same sides at the same cut) — re-exported here
from .mood import SENTIMENT_COLD as COLD_EMOTIONS, SENTIMENT_EMOTION_MIN as EMOTION_COUNT_MIN  # noqa: E402
from .mood import SENTIMENT_WARM as WARM_EMOTIONS  # noqa: E402
#: the log line shows emotions from here (weaker readings are noise); only EMOTION_COUNT_MIN ones are counted
EMOTION_SHOWN_MIN = 0.2


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
        #: §4VE: answers that carried an intent reading
        self.intents = 0
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
    out = {"label": label, "score": float(score), "probs": {k: float(v) for k, v in probs.items()},
           "reason": str(body.get("reason") or "")[:40]}
    emo = body.get("emotions")
    if isinstance(emo, dict) and label != "unknown" and all(
            k in EMOTION_LABELS and isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= v <= 1.0
            for k, v in emo.items()):
        out["emotions"] = {k: round(float(v), 4) for k, v in emo.items()}
    # §4VE: the intent and the urgency — kept for an unknown tone too (operator decision 8: a multilingual intent
    # model reads Greek the tone model does not); a bad field drops only itself. Not even read with GHOST_INTENT=0
    try:
        from . import intent as _intent
        if _intent.mode() != _intent.MODE_OFF:
            got = _intent.parse(body)
            if got:
                out.update(got)
    except Exception as exc:  # noqa: BLE001
        logger.debug("sentiment: intent field skipped: %s", type(exc).__name__)
    return out


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
    # §4UY P5 (review r2 N-5): every span is BOUNDED — the open ones were quadratic on a text that repeats a note's
    # opening and never closes it (500k chars: 25 s and 130 s). A note is far shorter: its fixed text is under
    # 200 chars and a file name under 256
    re.compile(r"\[SYSTEM NOTE: The user attached a file named .{0,700}?(?:"
               r"to interact with it\.\]|vision_analysis \(target='.{0,300}?'\)\.\]|"
               r"could NOT be ingested \(too large or the fetch failed\)\.\]|found inside it\.\]|"
               r"as if you had read it\.\])", re.DOTALL),
    re.compile(r"\(File just uploaded to the sandbox: [^\n]{0,700}\)"),
    re.compile(r"<@[A-Z0-9]{1,40}(?:\|[^>\n]{0,200})?>"),
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
#: §4UP: the emotion model read the message and found no feeling on the tone's side ("good morning ghost!")
HELD_UNBACKED = "no emotion backs the tone (politeness, not a feeling)"
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
    got = _feed(answer, self_model, praised, arrived, request_id, text)
    return got["nudge"], got["held"]


#: §4VE: why the intent's mood event moved nothing (beside the state's and the ledger's own reasons)
INTENT_HELD_JOURNAL = "journal only"
INTENT_HELD_OFF = "the intent's mood effect is switched off"
#: the message's tone (the sentiment's event) already went to the ledger: the intent adds nothing (one verdict)
INTENT_HELD_TONE = "the message's tone already counted"
#: §4VE: the row's ``intent_judge`` — the correction judge's NOMINATION by the intent (``intent.says_correcting``)
#: and nothing more: this module cannot know whether the judge then ran or what it said (the request path asks it).
#: "" = not nominated; else one of these (``intent.acts``/``would("judge")`` as for the other effects)
INTENT_JUDGE_NOMINATED = "nominated to the correction judge"
INTENT_JUDGE_OFF = "the intent's judge nomination is switched off"


def _after_trouble(self_model, arrived) -> bool:
    try:
        return bool(self_model.recent_trouble(arrived)) if self_model is not None else False
    except Exception:  # noqa: BLE001
        return False


def _feed(answer: Dict[str, object], self_model, praised: bool, arrived: Optional[datetime.datetime] = None,
          request_id: str = "", text: str = "") -> Dict[str, object]:
    """``_feed_mood``'s work, with the intent's part: ``{"nudge", "held", "intent_effect", "intent_held"}``.
    ``held`` is the SENTIMENT's reason (as before §4VE); ``nudge`` what the message moved, by either reading.

    §4VE: the sentiment's event first. Only when it has none (neutral, under its gate, unbacked, not English)
    may the intent's event (``intent.mood_event``) be booked — through the SAME ledger call, as the one tone of
    this message, so every cap, the owner's hold and "set after the message arrived" apply unchanged. When the
    sentiment's event exists but was doubted (praised, possible sarcasm, a task's wording) the intent stays
    silent for the same reason; when it went to the ledger the intent adds nothing (``INTENT_HELD_TONE``).
    In journal mode the intent's event is never booked (``INTENT_HELD_JOURNAL``). Never raises."""
    from . import intent as _intent
    from . import mood as M
    out: Dict[str, object] = {"nudge": 0.0, "held": "", "intent_effect": "", "intent_held": ""}
    try:
        emotions = answer.get("emotions")
        has_intent = _intent.mode() != _intent.MODE_OFF and isinstance(answer.get("intent"), dict)
        after_trouble = _after_trouble(self_model, arrived) if (has_intent or answer.get("label") == "positive") \
            else False
        iev, iwhy = (_intent.mood_event(answer, text, after_trouble=after_trouble) if has_intent else (None, ""))
        out["intent_effect"] = iev.cause if iev is not None else ""
        out["intent_held"] = iwhy
        arrived_s = arrived.isoformat().replace("+00:00", "Z") if arrived is not None else ""
        ev = M.sentiment_event(answer.get("label"), answer.get("score"), emotions)
        if ev is None:
            label = answer.get("label")
            if label == "unknown":
                out["held"] = str(answer.get("reason") or "unknown")
            elif label == "neutral":
                out["held"] = "neutral"
            else:
                gate = M.sentiment_gate(label, emotions)
                score = answer.get("score")
                if isinstance(score, (int, float)) and not isinstance(score, bool) and score >= gate:
                    out["held"] = HELD_UNBACKED     # §4UP: confident, but no emotion on its side
                else:
                    out["held"] = f"below {gate:.2f}"
            if iev is None:
                return out
            # §4VE: no tone from the sentiment — the intent's event may be the message's one tone
            if praised:
                out["intent_held"] = HELD_PRAISED
            elif self_model is None:
                out["intent_held"] = HELD_NO_MOOD
            elif not _intent.acts("mood"):
                out["intent_held"] = INTENT_HELD_JOURNAL if _intent.would("mood") else INTENT_HELD_OFF
            else:
                after, dp, why = self_model.note_owner_sentiment(
                    str(label or ""), 0.0, request_id=str(request_id or ""), emotions=emotions,
                    arrived=arrived_s, event=iev)
                if why:
                    out["intent_held"] = why
                elif after is None:
                    out["intent_held"] = HELD_NO_MOOD
                else:
                    with BREAKER.lock:
                        BREAKER.moved += 1
                    out["nudge"], out["intent_held"] = dp, ""
            return out
        # the sentiment's event exists: the intent never books a second tone for the same message. (A doubted
        # tone needs no line here: ``intent.refusal`` runs the same ``doubtful`` and the intent is refused there.)
        if praised:              # "thanks, perfect" is already one praise event
            out["held"] = HELD_PRAISED
            if iev is not None:
                out["intent_held"] = HELD_PRAISED
            return out
        if self_model is None:
            out["held"] = HELD_NO_MOOD
            if iev is not None:
                out["intent_held"] = HELD_NO_MOOD
            return out
        doubt = doubtful(str(answer.get("label") or ""), text,
                         after_trouble=after_trouble if answer.get("label") == "positive" else False)
        if doubt:
            out["held"] = doubt
            return out
        after, dp, why = self_model.note_owner_sentiment(
            answer["label"], answer["score"], request_id=str(request_id or ""), emotions=emotions,
            arrived=arrived_s)
        # review r1 (MINOR 6a): the intent's reason is what really happened to the tone it deferred to
        if why or after is None:
            out["held"] = why or HELD_NO_MOOD
            if iev is not None:
                out["intent_held"] = out["held"]
            return out
        with BREAKER.lock:
            BREAKER.moved += 1
        out["nudge"] = dp
        if iev is not None:
            out["intent_held"] = INTENT_HELD_TONE
        return out
    except Exception as exc:  # noqa: BLE001
        logger.debug("sentiment mood event skipped: %s", type(exc).__name__)
        out["nudge"], out["held"] = 0.0, HELD_NO_MOOD
        if out["intent_effect"]:
            out["intent_held"] = HELD_NO_MOOD
        return out


#: ``score_message(register_humour=...)``'s default: read the owner's humour dial as the reply would hear it
REGISTER_HUMOUR_DIAL = object()


def _owner_humour(self_model) -> Optional[int]:
    """The character's humour dial as the owner's reply hears it, or None (no character, no self-model)."""
    try:
        from . import character as _C
        if self_model is None or not _C.enabled():
            return None
        return int(_C.dials(Path(self_model.root).parent.parent).get("humour", 0))
    except Exception:  # noqa: BLE001
        return None


# ── the reading of one request's message, for whoever waits for it (§4UW) ───
#
# The correction judge runs beside the turn and is let in by a phrase at the start of the message. A message
# that does not open like a correction ("the dashboard is still empty") never reached it — yet the emotion
# model has read it by then. The reading of each request's message is kept here for a moment, in memory.

READINGS_MAX = 128
#: how long the judge's gate waits for a reading that is on its way (the service answers in about 0.1 s)
READING_WAIT_S = 3.0
_READING_POLL_S = 0.05
_READINGS: "collections.OrderedDict[str, Optional[dict]]" = collections.OrderedDict()
_READINGS_LOCK = threading.Lock()
_PENDING = None            # the slot of a message whose reading has not come back yet
#: §4VE r3: the request ids whose reply stopped waiting for the reading (``note_reply_stopped_waiting``) — only
#: ids the store holds, dropped with them (so bounded by READINGS_MAX)
_LATE: "collections.OrderedDict[str, bool]" = collections.OrderedDict()


def expect_reading(request_id: str) -> None:
    """A message of request ``request_id`` is about to be scored (called BEFORE the scoring is spawned, so a
    waiter that starts right after it finds the slot)."""
    rid = str(request_id or "")
    if not rid:
        return
    with _READINGS_LOCK:
        _READINGS[rid] = _PENDING
        _READINGS.move_to_end(rid)
        _LATE.pop(rid, None)               # a new message under this id: its reply has not stopped waiting yet
        while len(_READINGS) > READINGS_MAX:
            old, _v = _READINGS.popitem(last=False)
            _LATE.pop(old, None)


def forget_reading(request_id: str) -> None:
    """Drop what is kept for ``request_id`` (a new request under an id that was used before). §4VE: also what
    the END of a request that used ``read_only`` (a member's, a probe's) calls, so its reading does not outlive
    it — safe at any moment: a reading still on its way when this runs is not kept (``_note_reading`` fills only
    a slot that is still expected), and a waiter returns None at once."""
    with _READINGS_LOCK:
        _READINGS.pop(str(request_id or ""), None)
        _LATE.pop(str(request_id or ""), None)


def note_reply_stopped_waiting(request_id: str) -> None:
    """§4VE r3: the reply of request ``request_id`` stopped waiting for its message's reading (the register's
    wait ran out) — the request path calls this when its wait expires. When the scoring then lands,
    ``score_message`` says so: in mode 1 the 💗 line's pitch part reads ``pitch: patient (too late, not
    offered)`` and the row gains ``register_late: true`` (``register`` keeps meaning what the reading implies).

    In journal mode the request path's lateness watcher may call this only AFTER the reading landed — the line
    is written by then, so the journal-mode line never says it (it cannot know, and does not pretend); the row
    carries ``register_late`` only when the mark was already set when the row was written.

    A no-op for an id the store does not hold (none was expected, or it was forgotten); safe to call twice.
    The mark lives and dies with the id's slot (``expect_reading`` / ``forget_reading`` / eviction). Never
    raises."""
    try:
        rid = str(request_id or "")
        with _READINGS_LOCK:
            if rid and rid in _READINGS:
                _LATE[rid] = True
    except Exception:  # noqa: BLE001
        pass


def _reply_stopped_waiting(request_id: str) -> bool:
    with _READINGS_LOCK:
        return bool(_LATE.get(str(request_id or "")))


#: §4VE: what a slot keeps of the answer — the whole reading (the tone, the emotions, the intent, the urgency)
_READING_KEYS = ("label", "emotions", "intent", "urgent")


def _note_reading(request_id: str, emotions, *, reading=None) -> None:
    """The scoring of ``request_id``'s message is over: its emotion reading, or {} when there is none (the
    service is off, failed, or carries no emotion model) — a waiter stops waiting either way. Only a slot that
    was expected is filled, once. §4VE: ``reading`` — the whole parsed answer — is kept beside it
    (``reading_for(..., part="all")``)."""
    rid = str(request_id or "")
    if reading is None and isinstance(emotions, dict):
        reading = {"emotions": emotions}          # a caller with the emotions only (the judge's tests)
    # review r2 MAJOR 1: the whole reading is kept in EXACTLY the parsed answer's shape — an answer with no
    # emotion reading has no "emotions" key (``mood.sentiment_backing`` reads a missing reading as None, "no
    # model", but {} as False, "nothing backs the tone"); the judge's view is kept beside it, as before
    kept: dict = {}
    if isinstance(reading, dict):
        for k in _READING_KEYS:
            v = reading.get(k)
            if v is not None:
                kept[k] = dict(v) if isinstance(v, dict) else v
    slot = {"all": kept, "emotions": dict(emotions) if isinstance(emotions, dict) else {}}
    with _READINGS_LOCK:
        # … and only once: a message the owner sent while the turn ran is scored under the same request id,
        # and the reading the judge waits for is the request's own message's
        if rid and rid in _READINGS and _READINGS[rid] is _PENDING:
            _READINGS[rid] = slot


async def reading_for(request_id: str, *, wait_s: float = READING_WAIT_S, part: str = "emotions") -> Optional[dict]:
    """The emotion reading of ``request_id``'s message: at once when it is in, after at most ``wait_s`` when
    it is on its way, None when none was expected, none came, or the wait ran out. Never raises.

    §4VE: ``part="all"`` — the whole reading instead: ``{"label", "emotions", "intent", "urgent"}``, each key
    ONLY when the parsed answer had it (no ``emotions`` key for an answer without an emotion reading — the same
    shape the journal's row was computed from), None as above or when the scoring had no answer. Any other
    ``part``: None."""
    rid = str(request_id or "")
    if part not in ("emotions", "all"):
        return None
    try:
        deadline = time.monotonic() + max(0.0, float(wait_s))
        while True:
            with _READINGS_LOCK:
                if rid not in _READINGS:
                    return None
                got = _READINGS[rid]
            if got is not _PENDING:
                if part == "all":
                    whole = got.get("all") or {}
                    return {k: (dict(v) if isinstance(v, dict) else v) for k, v in whole.items()} or None
                return dict(got.get("emotions") or {}) or None
            if time.monotonic() >= deadline:
                return None
            await asyncio.sleep(_READING_POLL_S)
    except Exception:  # noqa: BLE001
        return None


async def read_only(content, *, request_id: str, tor_proxy: Optional[str] = None) -> bool:
    """§4VE: read a message for the PITCH of the reply only — a member's own message, a probe's: classify it,
    keep its reading for ``reading_for(request_id, part="all")``, one log line of labels (no words, no
    numbers, no member id). No journal row, no mood, nothing stored past the slot. True when a reading came.

    Only with the register on (``intent.acts("register")``); a member's words only with
    ``intent.acts("members")`` (operator decision 3 — the written rule at the top of this module stands until
    then). The slot is filled on every way out, so a waiter stops waiting. Never raises."""
    rid = str(request_id or "")
    try:
        from . import intent as _intent
        from ..utils.logging import request_kind, requester_is_member
        # review r2: a member's PROBE is kind "probe" — the member test is the request path's own as well, so
        # this gate is complete on its own
        is_member = request_kind() == "member" or requester_is_member()
        if not _intent.acts("register") or (is_member and not _intent.acts("members")):
            _note_reading(rid, None)
            return False
        text = prepare_text(message_text(content))
        answer = (await classify(text, tor_proxy=tor_proxy)) if text else None
        _note_reading(rid, (answer or {}).get("emotions"), reading=answer)
        if answer is None:
            return False
        if "intent" in answer:
            with BREAKER.lock:
                BREAKER.intents += 1
        try:
            from ..utils.logging import Icons, pretty_log
            tone = str(answer.get("label") or "")
            pretty_log("Sentiment", f"read for the reply only: {tone}{_intent.log_part(answer, percent=False)}"
                                    " · nothing kept", icon=Icons.SENTIMENT)
        except Exception as exc:  # noqa: BLE001
            logger.debug("sentiment read-only log line skipped: %s", type(exc).__name__)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("sentiment read-only skipped: %s", type(exc).__name__)
        _note_reading(rid, None)
        return False


async def score_message(content, root, *, request_id: str = "", tor_proxy: Optional[str] = None,
                        self_model=None, praised: bool = False, register_audience: str = "",
                        register_humour=REGISTER_HUMOUR_DIAL) -> bool:
    """Score one OWNER message (the caller decides that), feed the mood, write the journal row, log the rating.
    ``praised``: the praise check already counted this message. True when a row was written. Never raises.

    §4VE: ``register_audience`` / ``register_humour`` — what the reply to this message hears, for the row's
    ``register`` key (what the register would say): the audience (``"owner"``, ``"public"`` for the owner on a
    public channel, …; review r3: the default "" is "not stated" — no register on the row unless the caller
    says who hears the reply) and the humour dial — ``REGISTER_HUMOUR_DIAL`` reads the character's dial, ``None`` = no
    character heard. With the register switched off (``intent.would("register")``) the key is ""."""
    try:
        arrived = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
        text = prepare_text(message_text(content))
        answer = (await classify(text, tor_proxy=tor_proxy)) if text else None
        # §4UW: whoever waits for this message's reading stops waiting here, on every way out
        _note_reading(request_id, (answer or {}).get("emotions"), reading=answer)
        if answer is None:
            return False
        got = _feed(answer, self_model, praised, arrived, request_id, text)
        nudge, held = got["nudge"], got["held"]
        row = {"ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
               "request_id": str(request_id or "")[:64], **answer, "nudge": nudge, "held": held}
        # §4VE: what the intent did — or, in journal mode, would have done. Only on a row whose answer carries
        # an intent or urgency reading (``_answer`` keeps none with GHOST_INTENT=0): every other row is as before
        lines = {}
        if "intent" in answer or "urgent" in answer:
            from . import intent as _intent
            if "intent" in answer:
                with BREAKER.lock:
                    BREAKER.intents += 1
            reg_key = ""
            if _intent.would("register"):
                humour = (_owner_humour(self_model) if register_humour is REGISTER_HUMOUR_DIAL
                          else register_humour)
                reg_key, _line = _intent.register(answer, audience=str(register_audience or ""), humour=humour,
                                                  text=text, after_trouble=_after_trouble(self_model, arrived))
            judge = ""
            if _intent.says_correcting(answer):
                judge = (INTENT_JUDGE_NOMINATED if _intent.acts("judge")
                         else INTENT_HELD_JOURNAL if _intent.would("judge") else INTENT_JUDGE_OFF)
            row.update({"intent_effect": got["intent_effect"], "intent_held": got["intent_held"],
                        "register": reg_key, "intent_judge": judge})
            # review r3: the reply stopped waiting before this reading landed (``note_reply_stopped_waiting``)
            late = _reply_stopped_waiting(request_id)
            if late:
                row["register_late"] = True
            lines = {"intent_effect": got["intent_effect"], "intent_held": got["intent_held"],
                     "register": reg_key, "judge": judge, "register_late": late}
        row.update({"chars": len(text), "text": text[:JOURNAL_TEXT_CHARS]})
        await asyncio.to_thread(_append, Path(root) / JOURNAL_FILENAME, row)
        log_rating(answer, nudge, held, **lines)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("sentiment scoring skipped: %s", type(exc).__name__)
        _note_reading(request_id, None)
        return False


def _mood_part(answer: Dict[str, object], effect: str, nudge: float, held: str, intent_effect: str,
               intent_held: str) -> str:
    """§4VE (operator 11-20: "everything in one 💗 line"): the line's ONE mood part for a message with an intent
    reading — what really happened to the mood, whichever reading caused it. ``effect`` is the tone's own text
    (unchanged); the intent adds to it only when its label HAS a mood effect: the short cause of a nudge it made
    ("mood nudged up +0.02 (asked for my opinion)"), what it would have done in journal mode ("mood unchanged;
    intent would nudge up +0.02 (journal only)"), or why it was held, after a semicolon — never a reason the
    tone's text already gave (the tone's own hold, an unbacked tone), and nothing when the tone moved the mood."""
    from . import intent as _intent
    if nudge:                                       # review r3: a small step is never shown as "+0.00"
        effect = f"mood nudged {'up' if nudge > 0 else 'down'} {_step(nudge)}"
    lab = _intent.top(answer)[0]
    if not lab or not _intent.MOOD_OUTCOMES.get(lab):
        return effect                               # instruction, question, greeting, …, unclear: nothing to add
    if nudge:
        if intent_effect and not intent_held:       # the intent's event was the message's one tone
            return f"{effect} ({_intent.CAUSE_SHORT.get(intent_effect, intent_effect)})"
        return effect                               # the tone moved it; the intent deferred to it
    if intent_held in (INTENT_HELD_TONE, HELD_UNBACKED, held):
        return effect
    if intent_effect and intent_held == INTENT_HELD_JOURNAL:
        # review r3: journal mode cannot know what the ledger's caps would have let through — the EVENT's step,
        # said as such, never as the nudge
        step = _intent.event_nudge(intent_effect)
        return (f"mood unchanged; intent would nudge {'up' if step > 0 else 'down'} ({_step(step)} before caps, "
                "journal only)")
    return f"{effect}; intent held: {intent_held}" if intent_held else effect


def _step(x: float) -> str:
    """A mood step as the 💗 line writes it: two decimals, as the sentiment's own nudge (+0.015 reads "+0.01"),
    with the unicode minus — but never "+0.00" for a step that moved something (then three decimals)."""
    txt = f"{x:+.2f}"
    if x and float(txt) == 0.0:
        txt = f"{x:+.3f}"
    return txt.replace("-", "−")


def _pitch_part(register: str, late: bool = False) -> str:
    """" · pitch: patient" (the register key as the row keeps it), "(journal only)" while the register only
    records; "" with no clause — the reply heard none, or the switch is off. ``late`` (mode 1 only): the reply
    stopped waiting before the reading came — "(too late, not offered)"."""
    from . import intent as _intent
    if not register:
        return ""
    if not _intent.acts("register"):
        return f" · pitch: {register} (journal only)"
    return f" · pitch: {register}" + (" (too late, not offered)" if late else "")


def _judge_part(judge: str) -> str:
    # review r3: the intent only NOMINATES — the judge still needs a reply before it and its own check
    if judge == INTENT_JUDGE_NOMINATED:
        return " · nominated to the correction judge"
    if judge == INTENT_HELD_JOURNAL:
        return " · would be nominated to the correction judge (journal only)"
    return ""


def rating_line(answer: Dict[str, object], nudge: float = 0.0, held: str = "", *, intent_effect: str = "",
                intent_held: str = "", register: str = "", judge: str = "", register_late: bool = False) -> str:
    """The log line's content, in plain words, never the message's words (operator 11-12: "make the logs easy to
    read and understand but informative"). Three parts: how the message read, its strongest emotions (and whether
    the day record counts them for warmth), and what it did to the mood:
    ``positive 95% (neutral 4%, negative 1%) · emotion: gratitude 99% (counts as warm) · mood nudged up +0.08``,
    ``positive 82% (…) · emotion: curiosity 44% · mood unchanged: a nudge needs 85% or more, and a warm emotion
    behind the tone``, ``negative 81% (…) · emotion: confusion 81% · mood unchanged: no unhappy emotion behind the
    tone (with one, a nudge needs 60% or more)``,
    ``not rated: not English · mood unchanged``.

    §4VE (operator 11-20): a message with an intent reading says it on the SAME line, in this order: tone ·
    emotion · intent · pitch · mood · judge — ``negative 81% (…) · emotion: confusion 81% · intent: advice 97% ·
    pitch: patient · mood nudged up +0.02 (asked for my opinion)``. ``intent_effect`` / ``intent_held`` are the
    journal row's fields, ``register`` its register key (what the reply heard), ``judge`` its ``intent_judge``.
    With ``GHOST_INTENT=0`` (or no intent reading) the line is exactly the line before §4VE."""
    label = str(answer.get("label") or "")
    if nudge:
        effect = f"mood nudged {'up' if nudge > 0 else 'down'} {nudge:+.2f}".replace("-", "−")
    elif label == "unknown":
        effect = "mood unchanged"
    elif held == "neutral" or label == "neutral":
        effect = "mood unchanged: a neutral message"
    elif held.startswith("below "):
        try:
            effect = f"mood unchanged: a nudge needs {float(held.split()[1]) * 100:.0f}% or more"
        except (IndexError, ValueError):
            effect = f"mood unchanged: {held}"
        # operator 11-20 (a live line: "negative 81% · confusion 81% · a nudge needs 85% or more" read as if 85%
        # would have moved the mood): a tone no emotion backs moves nothing at any score, and the line says so
        from . import mood as M
        if M.sentiment_backing(label, answer.get("emotions")) is False:
            if label == "negative":
                effect = ("mood unchanged: no unhappy emotion behind the tone (with one, a nudge needs "
                          f"{M.SENTIMENT_MIN_SCORE_BACKED * 100:.0f}% or more)")
            elif label == "positive":
                effect += ", and a warm emotion behind the tone"
    else:
        effect = "mood unchanged" + (f": {held}" if held else "")
    said = pitch = judged = ""
    try:
        from . import intent as _intent
        if _intent.mode() != _intent.MODE_OFF:
            said = _intent.line_part(answer)
            if said:
                pitch = _pitch_part(str(register or ""), bool(register_late))
                judged = _judge_part(str(judge or ""))
                effect = _mood_part(answer, effect, nudge, str(held or ""), str(intent_effect or ""),
                                    str(intent_held or ""))
    except Exception:  # noqa: BLE001
        said = pitch = judged = ""
    said += pitch
    if label == "unknown":
        return f"not rated: {answer.get('reason') or 'no reason given'}{said} · {effect}{judged}"
    probs = answer.get("probs") or {}
    others = ", ".join(f"{k} {float(v) * 100:.0f}%" for k, v in sorted(probs.items(), key=lambda kv: -float(kv[1]))
                       if k != label)
    # operator 11-12 (req 9cbdf9aa: a question read curiosity 0.44 and the line said nothing): the strongest two
    # at EMOTION_SHOWN_MIN or more are shown; the ones the day record COUNTS (EMOTION_COUNT_MIN, warm or annoyed)
    # say so; a reading with none says "none clear", so a line with no emotion part means no emotion answer
    emo = answer.get("emotions")
    shown = ""
    if isinstance(emo, dict):
        top = sorted(((v, k) for k, v in emo.items() if k != "neutral" and v >= EMOTION_SHOWN_MIN), reverse=True)[:2]

        def _tag(k, v):
            if v < EMOTION_COUNT_MIN:
                return ""
            return " (counts as warm)" if k in WARM_EMOTIONS else " (counts as annoyed)" if k in COLD_EMOTIONS else ""
        shown = (" · emotion: " + ", ".join(f"{k} {v * 100:.0f}%{_tag(k, v)}" for v, k in top)) if top \
            else " · emotion: none clear"
    return (f"{label} {float(answer.get('score') or 0) * 100:.0f}%" + (f" ({others})" if others else "") + shown
            + said + f" · {effect}" + judged)


def log_rating(answer: Dict[str, object], nudge: float = 0.0, held: str = "", *, intent_effect: str = "",
               intent_held: str = "", register: str = "", judge: str = "", register_late: bool = False) -> None:
    """One line in the agent's log, with its own icon. Never raises."""
    try:
        from ..utils.logging import Icons, pretty_log
        pretty_log("Sentiment", rating_line(answer, nudge, held, intent_effect=intent_effect,
                                            intent_held=intent_held, register=register, judge=judge,
                                            register_late=register_late),
                   icon=Icons.SENTIMENT)
    except Exception as exc:  # noqa: BLE001
        logger.debug("sentiment log line skipped: %s", type(exc).__name__)


def journal_rows(root, *, limit: int = 0):
    """The journal's rows, oldest first (the newest ``limit`` when given). Unreadable lines are skipped."""
    # §4UD (F5): the shared reader — one byte that is not UTF-8 raised UnicodeDecodeError here
    from .schema import read_jsonl_rows
    rows = read_jsonl_rows(Path(root) / JOURNAL_FILENAME, what="sentiment journal")
    return rows[-limit:] if limit else rows


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
        intents = BREAKER.intents
    text = f"sentiment service: {scored} scored since start, {moved} moved the mood"
    if intents > 0:          # §4VE: said only once an intent reading was seen (the line is pinned without it)
        text += f", {intents} with an intent reading"
    if now < paused_until:
        return text + f", PAUSED after failures (last: {why}, {_ago(now - last_fail)}; retries by itself)"
    if last_fail and last_fail > last_ok:
        return text + f", last call failed ({why}, {_ago(now - last_fail)})"
    return text
