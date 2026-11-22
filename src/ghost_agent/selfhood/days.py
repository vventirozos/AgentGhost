"""The day record (§4UG): what one day was like, as counts — no model call, no words of a message.

Until now nothing kept a day. The mood halves every two hours, its journal is a list of label changes, and
the only thing that looked back further was the Mirror's note about a mood that lasted three days of seven
(which counted a mood the owner's thumb had made as the agent's own: a journal row carries the kind of the
event that crossed a boundary, not of the events that built the mood). The day record is the layer between
the mood (hours) and the character (weeks): one row per mood day (``mood.mood_day`` — it ends at 04:00),
written once the day is over, from the two files the mood already keeps:

* ``mood.events.jsonl`` — every event that moved or tried to move the pair, with its kind and size. Counted
  BY KIND, so what the agent's own work did and what the owner did stay apart;
* ``mood.history.jsonl`` — the label changes, for how long each mood was held (the arc) and the day's peak;
* ``sentiment.jsonl`` (§4UH-warmth) — the emotion labels of the owner's rated messages, counted as messages
  that sounded warm (gratitude, admiration, approval…) and messages that sounded annoyed (annoyance, anger,
  disapproval…): ``msgs_rated`` / ``msgs_warm`` / ``msgs_cold`` / ``msgs_thanks``. Counts only — never a word of a message.
  §4VE: the same journal's INTENT readings (what a message was doing — advice, banter, confiding, thanks…), on a
  day that has one: ``intent_rated``, ``intent`` {label: n}, ``urgent`` and the ``talk_*`` counts the character
  reads (``owner_intents``). Written whenever a reading exists, whatever the intent switches say, so the
  character's norm on its new scales is there the day they are switched on; a day without one has no such key.

A row holds numbers, labels, manner names and the events' own cause sentences (the system's words: "2 tool
calls failed (of 5) in this turn"). It never holds a message, a task name or a subject, so it can be shown
to the character's rules and, later, to a model without carrying the owner's work along.

Storage: ``$GHOST_HOME/system/selfhood/days.jsonl``, append-only, one row per day, written by
``close_days`` from the idle window. Reading never writes.
"""

from __future__ import annotations

import datetime
import json
import logging
import math
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import mood as M
from .schema import read_jsonl_rows

logger = logging.getLogger("GhostSelfhood")

DAYS_FILENAME = "days.jsonl"
SELFHOOD_REL = "system/selfhood"
#: how far back ``close_days`` goes for days nobody recorded (the agent was off, or this is the first run)
BACKFILL_DAYS = 14
#: the biggest events kept per day (the owner's own set_mood is not one: it is the owner's word, not an event)
BIGGEST = 3
DAYS_MAX_ROWS = 800
DAYS_KEEP_ROWS = 600

_LOCK = threading.Lock()

#: the manner an owner's manner event names — ``state.set_manner`` writes "the owner set the manner: X, n of 3"
_MANNER_SET_RE = re.compile(r"^the owner set the manner: ([a-z]+), \d of 3$")
#: the label an owner's set_mood left — the event's ``after`` reads "cheerful, 2 of 3" or "neutral"
_AFTER_RE = re.compile(r"^([a-z]+)(?:, (\d) of 3)?$")
#: the parts of a turn's cause sentence (``mood.turn_events`` joins them with "; ")
#: Order matters: the clean-turn sentence may itself say "… went through, 1 of them failed" (review r1: it
#: was counted as trouble twice). The trouble sentence is the one that ends "in this turn".
_OWN_PARTS = (
    ("clean_turns", re.compile(r"^a turn of \d+ tool calls? went through")),
    ("research", re.compile(r"^read \d+ sources in a research turn$")),
    ("tasks_done", re.compile(r"^finished \d+ project tasks?(?: \(in a background job\))?$")),
    ("long_turns", re.compile(r"^a very long turn \(")),
    ("trouble", re.compile(r"(?:failed\b.*|a loop guard stepped in .*|the context ran out of room .*) in this turn$")),
)


def _root(home) -> Optional[Path]:
    return Path(home) / SELFHOOD_REL if home else None


def _when(row) -> Optional[datetime.datetime]:
    return M.parse_iso_utc(str((row or {}).get("at") or ""))


def _own_counts(cause: str) -> Dict[str, int]:
    """What one turn event's cause says happened: each part of the sentence counted once."""
    out: Dict[str, int] = {}
    for part in str(cause or "").split("; "):
        for name, rx in _OWN_PARTS:
            if rx.search(part.strip()):
                out[name] = out.get(name, 0) + 1
                break
    return out


def owner_thumbs(home, day: datetime.date, *, tz=None) -> Tuple[int, int]:
    """(👍, 👎) the owner gave on mood day ``day``, from the label store (``corrections.jsonl``): each
    turn counts once, by the owner's LAST thumb on it, on the day that thumb was given. §4UH review r2: the
    mood journal cannot say this — a thumb sent again, re-booked after the character's lean changed, or
    given at the mood's limit is journalled with whatever size it moved, so its size cannot tell a new
    thumb from the same one. Two limits (review r3): a thumb taken back (web, handheld, Slack) writes no
    row here and so still counts; and a thumb changed on a later day counts on both days (a day record is
    written once)."""
    from .tastes import OWNER_FEEDBACK
    if not home:
        return 0, 0
    last: Dict[str, Tuple[str, str]] = {}
    for r in read_jsonl_rows(Path(home) / "system" / "trajectories" / "corrections.jsonl", what="labels"):
        # review: matched as the mood matches it (`feedback.is_owner_label_source` lowercases)
        if str(r.get("source") or "").strip().lower() in OWNER_FEEDBACK and r.get("outcome") in ("passed", "failed"):
            last[str(r.get("trajectory_id") or "")] = (str(r.get("outcome")), str(r.get("timestamp") or ""))
    up = down = 0
    for outcome, ts in last.values():
        at = M.parse_iso_utc(ts)
        if at is None or M.mood_day(at, tz=tz) != day:
            continue
        if outcome == "passed":
            up += 1
        else:
            down += 1
    return up, down


#: held reasons that still say something HAPPENED that day (the owner's hold, the mood at its limit)
#: — the owner's hold, and a tone refused only because sentiment's share was full (a cap on NEW messages, not
#: a repeat: review). Not "already counted for this reply" (a repeat). (The same strings as
#: ``SelfStateThread.HELD_OWNER_HOLD`` / ``HELD_OWNER_LATER`` and ``SelfModel.HELD_SENTIMENT_FULL`` — pinned
#: equal in the tests; literals here so this reader imports neither. An event at the mood's limit is journalled
#: as applied, so it is counted already.) Known: under the owner's hold the reaction ledger is not asked, so a
#: tone on a reply that also has a thumb counts on a held day and not on another — WARMTH_CAP bounds it.
EVIDENCE_HELD = frozenset({"the owner set the mood (held until the day ends)",
                           "the owner set the mood after the message",
                           "sentiment's share of the mood is full"})


def _is_intent_cause(cause: str) -> bool:
    """§4VE: is this a mood event the INTENT reading made (``intent.EVENT_CAUSES``), not the sentiment's tone?"""
    try:
        from .intent import EVENT_CAUSES
    except Exception:  # noqa: BLE001
        return False
    c = str(cause or "").strip()
    return any(c == x or (c.startswith(x) and c[len(x)] in " ,;:.") for x in EVENT_CAUSES)


def _count_evidence(rec: Dict[str, object], kind: str, cause: str, dp: float, ev: dict) -> None:
    """What one event says about the day: corrections, praise, tones, the owner's moods and manners, the
    agent's own work. Thumbs are counted from the label store (``owner_thumbs``)."""
    if kind in (M.KIND_THUMB_UP, M.KIND_THUMB_DOWN, M.KIND_THUMB_OFF):
        pass                                        # counted from the label store below (§4UH review r2)
    elif kind == M.KIND_CORRECTION:
        rec["corrections"] = int(rec["corrections"]) + 1
    elif kind == M.KIND_PRAISE:
        rec["praise"] = int(rec["praise"]) + 1
    elif kind == M.KIND_SENTIMENT:
        # §4VE: an intent's mood event is booked as a tone (the same kind, the private shares) but it is not the
        # tone of the owner's words: venting is the owner's own bad news, not coldness toward Ghost, and banter or
        # confiding are counted from the sentiment journal (``owner_intents``) — counted here too they would be
        # counted twice. Told apart by its cause sentence (the ledger may add to its end: matched at the start)
        if _is_intent_cause(cause):
            return
        if dp:
            rec["tone_up" if dp > 0 else "tone_down"] = int(rec["tone_up" if dp > 0 else "tone_down"]) + 1
    elif kind == M.KIND_OWNER:
        m = _AFTER_RE.match(str(ev.get("after") or ""))
        # "neutral" is the owner taking the mood OFF, not a mood the owner asked for
        if m and m.group(1) in M.LABELS and m.group(1) not in rec["owner_moods"]:
            rec["owner_moods"].append(m.group(1))
    elif kind == M.KIND_MANNER:
        m = _MANNER_SET_RE.match(cause)
        if m and m.group(1) in M.MANNERS and m.group(1) not in rec["manners"]:
            rec["manners"].append(m.group(1))
    elif kind == M.KIND_TURN:
        for name, n in _own_counts(cause).items():
            rec["own"][name] = int(rec["own"].get(name, 0)) + n


def _warm_routes(label, text) -> Tuple[bool, bool]:
    """§4VE functional run 11-21 (the real service: "great, you broke it again" → sentiment NEGATIVE 0.72, emotion
    admiration 0.93 — the emotion model takes "great" at face value): which routes may make a message WARM, as
    (a warm emotion, thanks — gratitude felt or the praise check on the words). THE RULE, on both warmth scales:

    =================  ================  =================
    tone label         trouble not gone  routes open
    =================  ================  =================
    negative           yes               neither
    negative           no                thanks
    positive/neutral   yes               thanks
    positive/neutral   no                both
    =================  ================  =================

    * words about trouble that is not gone (``sentiment._TROUBLE_WORDS`` without ``sentiment._RESOLVED``: the
      test ``sentiment.doubtful`` holds an upbeat score by) — no warm emotion, whatever the tone label says
      ("Oh wonderful, another error. Love that for me." read as admiration): a warm-sounding emotion on a message
      about trouble may be a complaint the emotion model took at face value;
    * … and under a NEGATIVE tone label no thanks either ("great, you broke it again" read as negative with
      admiration: the tone says complaint);
    * trouble under a positive or neutral tone — thanks still count ("thank you for catching that bug!", "you
      saved me from a huge mess, thank you!!"): the owner is a programmer talking to a coding agent, bugs and
      errors are the subject matter, not a sign of sarcasm (independent review 11-21);
    * a NEGATIVE tone label alone — no warm emotion (the tone and the emotion disagree: "hehe you'd make a
      terrible weatherman"), but thanks still count ("I'm so sad but thank you for staying with me": gratitude is
      said to someone, the sadness is the owner's own);
    * otherwise both.

    ACCEPTED COST: praise of the work that mentions trouble and carries admiration but no thanks ("great
    debugging, the error was subtle", "I love how you handled that error message design") is NOT warm. KNOWN LEAK:
    sarcastic thanks about trouble under a positive or neutral tone ("thanks for breaking the build again") are
    open here; they are stopped only where the live path held the row as possible sarcasm (``held`` /
    ``intent_held``, read by both callers) and, on the intent scale, where ``intent.refusal`` doubts a positive
    tone label — not on the old scale for a positive score under the mood's gate, nor on either scale for a
    neutral one.

    Never what makes a message ANNOYED (a cold emotion counts as before), and never a confiding (the owner telling
    of their own trouble is confiding). Pure."""
    from .sentiment import _RESOLVED, _TROUBLE_WORDS
    t = str(text or "")
    if _TROUBLE_WORDS.search(t) and not _RESOLVED.search(t):
        return False, label != "negative"
    return label != "negative", True


def owner_tones(home, day: datetime.date, *, tz=None) -> Tuple[int, int, int, int]:
    """(rated, warm, cold, thanks): the owner's messages of mood day ``day`` that the emotion model read, how many
    of them sounded warm / annoyed (a label of ``sentiment.WARM_EMOTIONS`` / ``COLD_EMOTIONS`` at
    ``EMOTION_COUNT_MIN`` or more; a message can be both), and how many of the warm ones were thanks (gratitude) —
    so the thanks the model did NOT read (Greek, a miss) can be told from the ones it did. Rows without emotions
    (before §4UH-warmth, the service without its emotion model, a message that was not English) are not counted.
    A rating held as the wording of a task is not counted; one held as possible sarcasm counts as read and for
    its annoyance only (review: "great, you broke it again" is annoyed, not warm). §4VE functional run 11-21: a
    message is warm only by the routes ``_warm_routes`` leaves open — the real service read "great, you broke it
    again" as NEGATIVE with admiration 0.93, which nothing held, and it was counted warm. Days closed from then on
    count without those false-warm messages; days closed before are not rewritten, so the old scale's norm moves
    from the old count to the new over its 21 days (it only ever loses warm messages that were not warm). Reads
    the rotated journal (``.1``) too. (0, 0, 0, 0) when unreadable."""
    from .sentiment import COLD_EMOTIONS, EMOTION_COUNT_MIN, HELD_SARCASM, HELD_TASK, JOURNAL_FILENAME, WARM_EMOTIONS
    if not home:
        return 0, 0, 0, 0
    path = Path(home) / SELFHOOD_REL / JOURNAL_FILENAME
    rows = (read_jsonl_rows(path.with_name(path.name + ".1"), what="sentiment journal")
            + read_jsonl_rows(path, what="sentiment journal"))
    rated = warm = cold = thanks = 0
    for r in rows:
        emo = r.get("emotions")
        held = str(r.get("held") or "")
        if not isinstance(emo, dict) or held == HELD_TASK:
            continue
        at = M.parse_iso_utc(str(r.get("ts") or ""))
        if at is None or M.mood_day(at, tz=tz) != day:
            continue

        def felt(names) -> bool:
            return any(isinstance(emo.get(k), (int, float)) and not isinstance(emo.get(k), bool)
                       and emo[k] >= EMOTION_COUNT_MIN for k in names)
        rated += 1
        cold += 1 if felt(COLD_EMOTIONS) else 0
        feeling, thanked = _warm_routes(r.get("label"), r.get("text"))
        grateful = thanked and felt(("gratitude",))
        if held != HELD_SARCASM and (grateful or (feeling and felt(WARM_EMOTIONS))):
            warm += 1
            thanks += 1 if grateful else 0
    return rated, warm, cold, thanks


#: §4VE: what each intent label counts toward in the day record's ``talk_*`` counts (what the character reads).
#: Every label of ``intent.LABELS`` is in exactly one of these (a test enumerates them: a label added without a
#: decision fails it):
#: * WARM — the owner confiding in it: how the owner talks TO it (warmth's intent scale);
#: * BANTER — joking around (humour's banter scale). UNMEASURED (§4VE measurement 11-20: 1–5 real examples,
#:   behind a provisional 0.90 floor in ``intent.LABEL_MIN``) — the humour dial's intent scale rests on it until a
#:   week of journal rows re-fits it; the record keeps the per-label counts (``intent``) to re-check it by;
#: * NONE — what a message asks for (advice, a question, an instruction), a greeting or small talk, a short reply,
#:   the owner's own trouble (venting: not how they are with Ghost — the mood hears it, the character does not),
#:   a correction (the correction judge already books it as a correction, which directness reads: counting the
#:   label too would count the same message twice), and thanks — never a top label (``intent.counted``): thanks
#:   and praise are counted from the praise check on the words and the gratitude emotion instead. §4VE: "asked for
#:   my opinion" as a directness signal was looked at and NOT built: it counts how many opinion questions the
#:   owner's work brings, not how its answers were judged.
TALK_WARM_INTENTS = ("confiding",)
TALK_BANTER_INTENTS = ("banter",)
TALK_NO_SIGNAL_INTENTS = ("advice", "instruction", "question", "greeting", "reply", "thanks", "venting",
                          "correcting")
#: §4VE review: the version of the counting rules behind the ``talk_*`` counts, written into every record that has
#: them (``talk_version``) and part of the character's scale name — so days counted under different rules never
#: share a norm. BUMP IT whenever what ``owner_intents`` counts changes: these rules, the constants above, or
#: anything it calls that decides a count — ``intent.counted`` / ``LABELS`` / ``LABEL_MIN`` / ``URGENT_MIN`` / ``refusal`` /
#: ``mood_event``'s witnesses, ``sentiment``'s emotion sides and cut (``WARM_EMOTIONS``, ``COLD_EMOTIONS``,
#: ``EMOTION_COUNT_MIN``) and held reasons, ``mood.has_praise``. A test pins the counts of a fixed table of rows
#: to this number: it fails when they change and the number does not.
#: 1 = the ten-label rules of the §4VE measurement (independent per-label scores, each label at its own floor —
#: ``intent.counted``), refusal by ``intent.refusal`` alone plus the live path's doubt, the words-only counts and
#: ``talk_readable`` (review r2/r3: the messages the service could read for intent) — shipped 2026-11-21;
#: 2 = §4VE functional run 11-21: warm only by the routes ``_warm_routes`` leaves open (no warm emotion under a
#: negative tone label or about trouble that is not gone; no thanks about trouble that is not gone under a negative
#: tone label — thanks about trouble under a positive or neutral one count) — "great, you broke it again" was warm.
#: The thanks part was refined by the independent review the same day, before version 2 shipped: still 2
TALK_COUNT_VERSION = 2
#: §4VE review r3: the sentiment labels of a message the service read (``sentiment.LABELS`` less "unknown": the
#: language gate's refusal, which never carries an intent reading)
_READABLE = ("positive", "negative", "neutral")


def owner_intents(home, day: datetime.date, *, tz=None) -> Optional[Dict[str, object]]:
    """§4VE: the intent readings of the owner's messages of mood day ``day`` (``sentiment.jsonl`` and its
    rotated ``.1``, the same day boundaries as ``owner_tones``) — or None when no message of that day carries an
    intent or urgency reading (the record then gets no intent key at all). Counts only, never a word.

    Written whenever a reading exists, whatever the switches say: rows the journal mode wrote before the owner
    set ``GHOST_INTENT=0`` still give their day these keys (the readings were made; the character reads them only
    while ``intent.acts("character")`` — review r2, MINOR 4: kept on purpose, a test pins it).

    * ``intent_rated`` — messages with a valid intent reading (every label in ``intent.LABELS``);
    * ``intent`` {label: n} — each message under the one label ``intent.counted`` accepts (the top-scoring label
      at its own floor; never a second-place one, never "reply" or "thanks"), every label listed. What the
      message was READ as — counted whether or not its effects were then refused;
    * ``urgent`` — messages whose urgency reading ``intent.urgent`` accepts (with or without an intent);
    * ``talk_readable`` — the day's messages the service COULD read for intent: a rated sentiment label
      (positive, negative, neutral — not "unknown": the service's language gate refused the message, short English
      and typos among them, and it never reads an intent for those, by design) and not held as the wording of a
      task — read for intent or not. What the readings' cover is measured against (review r3): a day the intent
      model missed falls back to the old scales, a day of short messages the gate refused does not;
    * ``talk_read`` — of those, the ones with an intent reading;
    * ``talk_emotions`` — of ``talk_read``, the ones the emotion model read too;
    * ``talk_warm`` — of ``talk_read``, each message ONCE when it was thanks or praise (the praise check on the
      words, ``mood.has_praise``, or gratitude felt — the intent model's "thanks" is never accepted), confided in
      the agent (the confiding mood event: the live path's verdict, ``intent_effect``, on a row that has it), or
      carried a warm emotion — never on a laugh's emotion alone when the message is counted as a joke, and the
      thanks and the warm emotion only as ``_warm_routes`` allows (no warm emotion about trouble that is not gone
      or under a negative tone label; no thanks about such trouble under a negative tone label);
    * ``talk_thanks`` — of the warm ones, those that were thanks or praise;
    * ``talk_cold`` — of ``talk_read``, the messages that sounded annoyed (a cold emotion felt) — a refused
      message keeps its annoyance;
    * ``talk_banter`` — of ``talk_read``, the messages counted as joking around: banter's mood event (the live
      path's verdict on a row that has it);
    * ``talk_warm_words`` / ``talk_thanks_words`` / ``talk_banter_words`` — the same three counted WITHOUT the
      emotion model, for every message (with emotions or not): thanks from the praise check only, a confiding and
      a joke re-read from the reading and the words alone (``intent.mood_event`` on the reading without its
      emotions), nothing annoyed. A day not wholly read by the emotion model is read from these (review r2,
      MINOR 2: one scale name, one computation);
    * ``talk_version`` — ``TALK_COUNT_VERSION``.

    Refusal — ONE rule, intent.py's (review r2, MAJOR): a message counts as no warmth and no joke exactly when
    ``intent.refusal`` refuses its counted label (the doubt on the tone refuses every label; no backing for the
    tone, a tone of the other sign or annoyance behind an upbeat label refuse only a label that is itself a tone,
    ``intent.INTENT_TONE``). The sentiment's own ``held`` is not that rule and is not read — except the DOUBT the
    live path found with the agent's recent trouble in view, which a re-read cannot see: ``held`` or
    ``intent_held`` being ``sentiment.doubtful``'s answer (possible sarcasm, a task's wording).

    Never raises: an unreadable journal or a damaged row counts as no reading."""
    try:
        from . import intent as I
        from .intent import _LAUGH_EMOTIONS
        from .sentiment import COLD_EMOTIONS, EMOTION_COUNT_MIN, HELD_SARCASM, HELD_TASK, JOURNAL_FILENAME, \
            WARM_EMOTIONS
    except Exception:  # noqa: BLE001 — no intent module: no intent reading
        return None
    if not home:
        return None
    try:
        #: ``sentiment.doubtful``'s answers: what the live path doubted, the agent's trouble in view
        doubts = (HELD_SARCASM, HELD_TASK)
        path = Path(home) / SELFHOOD_REL / JOURNAL_FILENAME
        rows = (read_jsonl_rows(path.with_name(path.name + ".1"), what="sentiment journal")
                + read_jsonl_rows(path, what="sentiment journal"))
        counts = {lab: 0 for lab in I.LABELS}
        out = {"intent_rated": 0, "urgent": 0, "talk_readable": 0, "talk_read": 0, "talk_emotions": 0, "talk_warm": 0,
               "talk_thanks": 0, "talk_cold": 0, "talk_banter": 0, "talk_warm_words": 0, "talk_thanks_words": 0,
               "talk_banter_words": 0}
        seen = False
        for r in rows:
            if not isinstance(r, dict):
                continue
            at = M.parse_iso_utc(str(r.get("ts") or ""))
            if at is None or M.mood_day(at, tz=tz) != day:
                continue
            held = str(r.get("held") or "")
            if held != HELD_TASK and r.get("label") in _READABLE:   # not a task's words; past the language gate
                out["talk_readable"] += 1
            if "intent" not in r and "urgent" not in r:
                continue
            # the journal is trusted for nothing: the reading is re-checked as the service's answer is
            got = I.parse({"intent": r.get("intent"), "urgent": r.get("urgent")}) or {}
            if not got:
                continue
            seen = True
            emo = r.get("emotions") if isinstance(r.get("emotions"), dict) else None
            reading = {"label": r.get("label"), **({"emotions": emo} if emo is not None else {}), **got}
            if "urgent" in got and I.urgent(reading):
                out["urgent"] += 1
            if "intent" not in got:
                continue
            out["intent_rated"] += 1
            lab = I.counted(reading)          # THE rule: the top label at its own floor, never a second one
            if lab in counts:
                counts[lab] += 1
            if held == HELD_TASK:                 # read, and nothing else
                continue
            out["talk_read"] += 1
            out["talk_emotions"] += 1 if emo is not None else 0

            def felt(names) -> bool:
                return emo is not None and any(
                    isinstance(emo.get(k), (int, float)) and not isinstance(emo.get(k), bool)
                    and emo[k] >= EMOTION_COUNT_MIN for k in names)
            text = str(r.get("text") or "")
            if felt(COLD_EMOTIONS):
                out["talk_cold"] += 1
            if held in doubts or str(r.get("intent_held") or "") in doubts:
                continue
            words = {"label": r.get("label"), **got}           # the reading without the emotion model
            feeling, thanked = _warm_routes(r.get("label"), text)
            thanks_w = thanked and M.has_praise(text)
            joke_w, confided_w = _talk_events(I, words, lab, text)
            if not I.refusal(words, lab, text):
                out["talk_warm_words"] += 1 if (thanks_w or confided_w) else 0
                out["talk_thanks_words"] += 1 if thanks_w else 0
                out["talk_banter_words"] += 1 if joke_w else 0
            if I.refusal(reading, lab, text):
                continue
            effect = r.get("intent_effect")
            if isinstance(effect, str) and I.MOOD_OUTCOMES.get(lab):   # the live path's verdict, its trouble read
                banter = lab in TALK_BANTER_INTENTS and effect == I.CAUSE_BANTER
                confided = lab in TALK_WARM_INTENTS and effect == I.CAUSE_CONFIDED
            else:                                  # a row from before the field: its witnesses, read now
                banter, confided = _talk_events(I, reading, lab, text)
            thanks = thanks_w or (thanked and felt(("gratitude",)))
            warm_felt = feeling and (felt(WARM_EMOTIONS - _LAUGH_EMOTIONS) if banter else felt(WARM_EMOTIONS))
            if warm_felt or thanks or confided:
                out["talk_warm"] += 1
                out["talk_thanks"] += 1 if thanks else 0
            out["talk_banter"] += 1 if banter else 0
        if not seen:
            return None
        out["intent"] = counts
        out["talk_version"] = TALK_COUNT_VERSION
        return out
    except Exception as e:  # noqa: BLE001
        logger.debug("day records: intent readings not counted (%s)", type(e).__name__)
        return None


def _talk_events(I, reading, lab: str, text: str) -> Tuple[bool, bool]:
    """(a joke, a confiding): the mood event ``intent.mood_event`` makes of this reading — read now, without the
    agent's recent trouble (only the live path saw it)."""
    if lab not in TALK_BANTER_INTENTS + TALK_WARM_INTENTS:
        return False, False
    ev = I.mood_event(reading, text)[0]
    cause = ev.cause if ev is not None else ""
    return (lab in TALK_BANTER_INTENTS and cause == I.CAUSE_BANTER,
            lab in TALK_WARM_INTENTS and cause == I.CAUSE_CONFIDED)


def build_day(home, day: datetime.date, *, tz=None, now: Optional[datetime.datetime] = None) -> Dict[str, object]:
    """The record of mood day ``day``, computed from the mood's two journals. Pure reading; never raises
    past an unreadable file (an absent journal is an empty day)."""
    root = _root(home)
    events = read_jsonl_rows(root / "mood.events.jsonl", what="mood events") if root else []
    history = read_jsonl_rows(root / "mood.history.jsonl", what="mood journal") if root else []
    rec: Dict[str, object] = {
        "date": day.isoformat(), "events": 0, "held": 0, "by_kind": {},
        "thumbs_up": 0, "thumbs_down": 0, "corrections": 0, "praise": 0, "tone_up": 0, "tone_down": 0,
        "msgs_rated": 0, "msgs_warm": 0, "msgs_cold": 0, "msgs_thanks": 0,
        "owner_moods": [], "manners": [],
        "own": {"trouble": 0, "clean_turns": 0, "research": 0, "tasks_done": 0, "long_turns": 0},
        "arc": {}, "peak": None, "biggest": [], "shades": {},
    }
    by_kind: Dict[str, Dict[str, float]] = {}
    candidates = []
    for ev in events:
        at = _when(ev)
        if at is None or M.mood_day(at, tz=tz) != day:
            continue
        kind = str(ev.get("kind") or "")
        held = ev.get("held")
        try:
            dp, de = float(ev.get("dp") or 0.0), float(ev.get("de") or 0.0)
        except (TypeError, ValueError):
            dp = de = 0.0
        cause = str(ev.get("cause") or "")
        # §4UW: ``dp``/``de`` are the event's SIZE; ``moved`` (when the row has it) is what the pair moved by —
        # a tone fitted to its cap, the ±1 limit, a reaction that replaced another. The day's sums and its biggest events are about what
        # moved; what the event says about the day (``_count_evidence``) reads its size
        mp, me = dp, de
        mv = ev.get("moved")
        if (isinstance(mv, (list, tuple)) and len(mv) == 2
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in mv)):
            mp, me = float(mv[0]), float(mv[1])
        if held:
            rec["held"] = int(rec["held"]) + 1          # it happened, it did not move the mood
            # §4UH-e2e review (MAJOR): a day the owner set the mood held every event until 04:00, and the
            # record skipped them — that day's corrections, tones and turns were lost to the character (a day
            # of corrections could pull the dials UP). What the owner's hold or the mood's limit refused still
            # HAPPENED: it is evidence about the day. What was refused as a repeat (already counted) is not
            # counted twice.
            if str(held) not in EVIDENCE_HELD:
                continue
            _count_evidence(rec, kind, cause, dp, ev)
            continue
        rec["events"] = int(rec["events"]) + 1
        slot = by_kind.setdefault(kind, {"n": 0, "dp": 0.0, "de": 0.0})
        slot["n"] += 1
        slot["dp"] = round(slot["dp"] + mp, 4)
        slot["de"] = round(slot["de"] + me, 4)
        _count_evidence(rec, kind, cause, dp, ev)
        if kind not in (M.KIND_OWNER, M.KIND_MANNER):
            candidates.append((abs(mp) + abs(me), {"at": str(ev.get("at") or ""), "kind": kind,
                                                   "dp": round(mp, 4), "de": round(me, 4), "cause": cause[:160]}))
    rec["by_kind"] = by_kind
    rec["thumbs_up"], rec["thumbs_down"] = owner_thumbs(home, day, tz=tz)
    rec["msgs_rated"], rec["msgs_warm"], rec["msgs_cold"], rec["msgs_thanks"] = owner_tones(home, day, tz=tz)
    # §4VE: only on a day with an intent reading — a record without one is exactly what it was before
    talk = owner_intents(home, day, tz=tz)
    if talk:
        rec.update(talk)
    candidates.sort(key=lambda c: -c[0])
    rec["biggest"] = [c[1] for c in candidates[:BIGGEST] if c[0] > 0]
    rec["arc"], rec["peak"] = _arc(history, day, tz, now)
    shades: Dict[str, int] = {}
    for row in (read_jsonl_rows(root / "shades.jsonl", what="shades") if root else []):
        at = _when(row)
        if at is not None and M.mood_day(at, tz=tz) == day and row.get("shade"):
            shades[str(row["shade"])] = shades.get(str(row["shade"]), 0) + 1
    rec["shades"] = shades
    return rec


def _arc(history: List[dict], day: datetime.date, tz, now: Optional[datetime.datetime] = None):
    """(minutes held per label that day, the day's peak) from the journal's label changes. The label held
    when the day began is the last row before it; the last label runs to the day's end (or to now)."""
    start = M.mood_day_end(day - datetime.timedelta(days=1), tz=tz)
    now = now or datetime.datetime.now(datetime.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=datetime.timezone.utc)
    end = min(M.mood_day_end(day, tz=tz), now)
    rows = []
    for r in history:
        at = _when(r)
        if at is None or str(r.get("kind") or "") == M.KIND_MANNER:      # a manner row repeats the mood as it stood
            continue
        try:
            level = int(r.get("intensity") or 0)
        except (TypeError, ValueError):
            level = 0
        rows.append((at, str(r.get("label") or M.NEUTRAL), level, str(r.get("kind") or "")))
    rows.sort(key=lambda r: r[0])
    # a day is [start, end): a row stamped exactly 04:00 is the NEW day's, as its event is (review r1)
    label, level, kind0 = M.NEUTRAL, 0, ""
    for at, lab, lev, kind in rows:
        if at >= start:
            break
        label, level, kind0 = lab, lev, kind
    arc: Dict[str, int] = {}
    peak: Optional[Dict[str, object]] = None
    peak_level = 0
    cursor, kind_now = start, kind0
    for at, lab, lev, kind in [r for r in rows if start <= r[0] < end] + [(end, "", 0, "")]:
        seconds = max(0.0, (at - cursor).total_seconds())
        if seconds > 0 and label != M.NEUTRAL and level > 0:
            if seconds >= 60:
                arc[label] = arc.get(label, 0) + int(seconds // 60)
            # the peak is the strongest mood HELD that day — the one carried in from the day before
            # included, a label replaced the instant it was written not; the boredom of a long quiet is
            # in the arc, never the peak (it is read over a neutral mood)
            if seconds >= 60 and kind_now != M.KIND_QUIET and level > peak_level:
                peak_level = level
                peak = {"label": label, "intensity": level, "at": cursor.isoformat().replace("+00:00", "Z")}
        cursor, label, level, kind_now = at, lab, lev, kind
    return arc, peak


def load_days(home, *, limit: int = 0) -> List[Dict[str, object]]:
    """The recorded days, oldest first by date (the newest ``limit`` when given); a date recorded twice
    counts once, its first row (§4UH: rows seeded at deploy were written before ones already there).
    [] when there are none."""
    root = _root(home)
    if root is None:
        return []
    seen, rows = set(), []
    for r in read_jsonl_rows(root / DAYS_FILENAME, what="day records"):
        d = str(r.get("date") or "")
        if d and d not in seen:
            seen.add(d)
            rows.append(r)
    rows.sort(key=lambda r: str(r.get("date")))
    return rows[-limit:] if limit else rows


def close_days(home, *, now: Optional[datetime.datetime] = None, tz=None) -> List[str]:
    """Write the record of every mood day that is OVER and not yet recorded (at most ``BACKFILL_DAYS``
    back, and never a day before the mood's journals begin). The dates written, oldest first. Idempotent:
    a recorded day is never written again. Never raises."""
    root = _root(home)
    if root is None:
        return []
    try:
        today = M.mood_day(now, tz=tz)
        with _LOCK:
            have = {str(r.get("date")) for r in load_days(home)}
            first = _first_journal_day(root, tz)
            if first is None:
                return []
            written = []
            for back in range(BACKFILL_DAYS, 0, -1):
                day = today - datetime.timedelta(days=back)
                if day < first or day.isoformat() in have:
                    continue
                rec = build_day(home, day, tz=tz, now=now)
                rec["written_at"] = (now or datetime.datetime.now(datetime.timezone.utc)).astimezone(
                    datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
                _append(root / DAYS_FILENAME, rec)
                written.append(day.isoformat())
            return written
    except Exception as e:  # noqa: BLE001 — the idle window never breaks on a diary
        logger.warning("day records: not written (%s: %s)", type(e).__name__, e)
        return []


def _first_journal_day(root: Path, tz) -> Optional[datetime.date]:
    """The mood day of the oldest event the journal still holds — nothing before it can be told."""
    for row in read_jsonl_rows(root / "mood.events.jsonl", what="mood events"):
        at = _when(row)
        if at is not None:
            return M.mood_day(at, tz=tz)
    return None


def _append(path: Path, row: dict) -> None:
    from ..utils.json_store import open_append
    path.parent.mkdir(parents=True, exist_ok=True)
    with open_append(path) as f:
        f.write(json.dumps(row, ensure_ascii=False))
        f.write("\n")
    rows = read_jsonl_rows(path, what="day records")
    if len(rows) > DAYS_MAX_ROWS:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows[-DAYS_KEEP_ROWS:]),
                       encoding="utf-8")
        tmp.replace(path)


def day_line(rec: Dict[str, object]) -> str:
    """One line a person can read (operator 11-12: easy to read but informative):
    "2026-11-10 · moods: irritable 40 min, content 3 h · my turns: 1 ran into trouble, 4 went fine · from you:
    2 👍, 1 correction(s), asked for a sarcastic manner · your messages: 6 read for emotion, 2 warm, 1 annoyed"."""
    def _span(minutes: int) -> str:
        return f"{minutes} min" if minutes < 90 else f"{minutes / 60:.1f} h".replace(".0 h", " h")
    arc = rec.get("arc") or {}
    moods = ", ".join(f"{lab} {_span(int(m))}" for lab, m in sorted(arc.items(), key=lambda kv: -int(kv[1]))) \
        or "none recorded"
    if rec.get("shades"):
        moods += " (shades: " + ", ".join(f"{k} ×{v}" if v > 1 else k for k, v in sorted(rec["shades"].items())) + ")"
    own = rec.get("own") or {}
    own_bits = [f"{own[k]} {name}" for k, name in (("trouble", "ran into trouble"), ("clean_turns", "went fine"),
                                                   ("tasks_done", "task(s) done"), ("research", "research"))
                if own.get(k)]
    you = []
    for key, name in (("thumbs_up", "👍"), ("thumbs_down", "👎"), ("corrections", "correction(s)"),
                      ("praise", "thank-you(s)")):
        if rec.get(key):
            you.append(f"{rec[key]} {name}")
    if rec.get("owner_moods"):
        you.append("set my mood to " + "/".join(rec["owner_moods"]))
    if rec.get("manners"):
        you.append("asked for a " + "/".join(rec["manners"]) + " manner")
    # §4UH-warmth: how the owner talked (what warmth reads), counts only
    talk = ""
    if rec.get("msgs_rated"):
        talk = (f" · your messages: {rec['msgs_rated']} read for emotion, "
                f"{rec.get('msgs_warm') or 0} warm, {rec.get('msgs_cold') or 0} annoyed")
    # §4VE: what the messages were doing, briefly (only a day with intent readings has these)
    if isinstance(rec.get("intent_rated"), int) and not isinstance(rec.get("intent_rated"), bool) \
            and rec["intent_rated"] > 0:
        got = rec.get("intent") if isinstance(rec.get("intent"), dict) else {}
        bits = [f"{rec.get('talk_banter') or 0} joking", f"{rec.get('talk_warm') or 0} warm (thanks, praise, confiding or a warm feeling)"]
        # review r2 (MINOR 3): what the messages were READ as — counted whether or not their effects were refused
        if got.get("advice"):
            bits.append(f"{got['advice']} read as asking my view")
        if rec.get("urgent"):
            bits.append(f"{rec['urgent']} read as urgent")
        talk += f" · read for intent: {rec['intent_rated']}, " + ", ".join(bits)
    return (f"{rec.get('date')} · moods: {moods}"
            + (" · my turns: " + ", ".join(own_bits) if own_bits else "")
            + (" · from you: " + ", ".join(you) if you else "") + talk)
