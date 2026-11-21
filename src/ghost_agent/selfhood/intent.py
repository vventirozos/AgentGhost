"""§4VE: what the owner's message is DOING — asking advice, giving an instruction, asking a question, confiding,
venting, joking around, correcting the reply, thanking — and whether it is urgent: a third reading from the
sentiment-service (beside the tone and the emotions). Pure: no agent code, no file, no network; the reading
arrives inside the service's answer (``sentiment._answer`` keeps it) and everything here reads that dict.
The witnesses read English (Greek is not a priority, operator 11-20): a Greek message rarely gets an effect.

Switches (all read from the environment at each call):

* ``GHOST_INTENT`` = ``0`` | ``journal`` (the default) | ``1``. ``0``: the field is not even parsed.
  ``journal``: parsed, journalled and logged, and it moves NOTHING — what it would have done is written next
  to the reading. ``1``: the effects run, each with its own switch below.
* ``GHOST_INTENT_REGISTER``, ``GHOST_INTENT_MOOD``, ``GHOST_INTENT_JUDGE``, ``GHOST_INTENT_CHARACTER`` default
  on; ``GHOST_INTENT_MEMBERS`` defaults OFF (a member's words are not sent to the service until the operator
  says so). Effective only with ``GHOST_INTENT=1`` (``acts``); ``would`` says what they would do in journal.

THE RULE (review r1 MAJOR 1/2, corrected by the coordinator 11-20), all of it in ONE place, ``refusal``, which
``mood_event`` and ``register`` run first:

* DOUBT refuses every label: no intent mood event is bookable, and no register clause that can offend when it
  is wrong is offered, for a message ``sentiment.doubtful`` doubts on its tone label (possible sarcasm, upbeat
  words right after the agent's own trouble, a task's wording). The clauses that cannot offend when wrong —
  "answer first, then anything else", "be patient and explain your reasoning", "keep the prose brief"
  (``DOUBT_FREE_CLAUSES``) — skip the doubt: putting the answer first, patience with a sarcastic owner, or brevity
  cost nothing; "a light joke is welcome" to a sarcastic one does.
* LACK OF BACKING (``mood.sentiment_backing`` is False: an upbeat or unhappy tone with no emotion on its side, at
  any score) refuses only the labels that ARE a tone (``INTENT_TONE``: banter is upbeat, venting unhappy) — the
  tone gate refused that very feeling, and the intent must not re-admit it. For every other label an unbacked
  tone is no reason to refuse: it says the tone reading tells nothing about a feeling, which is exactly when
  what the message is DOING (asking advice, confiding, an instruction) is the useful reading (a self-deprecating,
  unsure request to look over a plan first: negative 0.81 with confusion only — advice).
* A label that is a tone is also refused under a tone of the other sign, doubted on its own side, and (banter)
  refused with an annoyed emotion behind it.
* ORDER (review r3): every doubt is checked before the backing, so the reason kept (``intent_held``) is the doubt
  whatever the emotion model read — the day close counts doubted messages from that field and cannot re-derive
  the doubt from the row.

A fuzzy classifier is a witness, never the only gate: after
``refusal``, each effect needs a second witness from the text or the emotion reading. Two of them rest on the
classifier plus the FORM of the message only (advice: a question; brief: not a question) — so they carry the
weakest effects (``EV_ASKED`` is the smallest event; "brief" is never said beside another clause).

Labels come from the 11-20 measurement (``LABELS``, ``LABEL_MIN``): everything takes them from there, and a
label the service sends that is not there drops the whole intent field. The scores are INDEPENDENT (not a
softmax); the top label counts only at its own floor (``top``), and a top label below it is no reading — the
second label never stands in. The service sends an intent only for English (a tone of ``unknown`` has none).

The reading every function takes is a dict with any of the keys ``label`` (the tone), ``emotions``,
``intent`` ({label: score}, any subset of ``LABELS``) and ``urgent`` (a score) — the service's parsed answer
(``sentiment._answer``) or ``sentiment.reading_for(..., part="all")``. A missing or malformed key is no reading.

``register(reading, *, audience, humour=None, text="", after_trouble=False)``:

* ``audience``: who reads the reply — ``"owner"``, ``"others"`` (a member) or ``"public"`` (the agent's
  ``_mood_audience()``); anything else gives no line.
* ``humour``: for the owner, the character's humour dial (−2..2) when the character is heard; ``None`` when it
  is not (no character, a public reply). The light-joke clause is not offered to the owner at −1 or lower (the
  character wins, operator decision 7); for members and public the dial is ignored.
* ``text``: the words of the message the reply answers, as the service read them
  (``sentiment.prepare_text(sentiment.message_text(content))``) — the doubt and the witnesses read it.
* ``after_trouble``: for the owner, ``self_model.recent_trouble()`` — upbeat words right after the agent's own
  trouble may be sarcasm.

Returns ``(key, line)``: ``key`` names the clauses ("urgent+patient", "" for none); ``line`` ends in a blank
line, or is "" for none. The line never carries a number, a classifier's name, the mood or anything about the
owner."""
from __future__ import annotations

import math
import os
import re
from typing import Dict, Optional, Tuple

from . import mood as M

#: §4VE measurement 11-20 (deberta-v3-large zero-shot, 166 real owner messages): the service scores each label ON
#: ITS OWN, in [0, 1] — not a softmax: scores do not sum to 1 and several can be high at once (a greeting reads
#: greeting 1.00 AND banter 0.83). "greeting" (a greeting or small talk) and "reply" (a short answer or a
#: confirmation) are there because without them the other labels absorbed those messages wrongly. Nothing counts
#: the labels: everything reads this tuple
LABELS = ("instruction", "question", "advice", "greeting", "reply", "thanks", "correcting", "banter", "confiding",
          "venting")
#: a label that is NEVER accepted as the top label, for any effect or count
NEVER: Optional[float] = None
#: the top-scoring label is accepted only at or above its OWN floor (``top``); below it, or for a label that is
#: ``NEVER``, the message has no intent reading — the second-highest label is never used instead (this model's
#: second labels are unreliable). Measured (precision of the top label; the floors were fitted on the same rows,
#: so they are optimistic): instruction 0.98 at ANY score it reached — it legitimately wins at 0.11 (a short
#: "delete <the project>"; 0.11 was the LOWEST real instruction score of the 166), hence a floor of only 0.10:
#: under every real instruction, above a reading where nothing scores (all ≈ 0.01: review r2, the catch-all);
#: greeting 0.91 → 0.85; advice 0.80 → 0.90; question 0.76 → 0.6. Together ≈ 0.95 precision at
#: ≈ 0.55 coverage. UNMEASURED (1–5 real examples each): banter, confiding, venting, correcting — a deliberately
#: high provisional 0.90 until a week of journal rows can re-fit them. "reply" is NEVER (0.39 precise: it
#: swallows short thanks and chat); "thanks" is NEVER (the model never predicted it; the praise check is better,
#: 8 of 8)
LABEL_MIN: Dict[str, Optional[float]] = {
    "instruction": 0.10, "greeting": 0.85, "advice": 0.90, "question": 0.6,
    "banter": 0.90, "confiding": 0.90, "venting": 0.90, "correcting": 0.90,
    "reply": NEVER, "thanks": NEVER}
#: unmeasured (no urgent message in the 166)
URGENT_MIN = 0.7
#: how long the reply waits for the reading before its first state block (the request path). Measured 11-20: the
#: intent model takes ~620 ms median and ~980 ms at the 90th percentile ON TOP of the sentiment and emotion
#: models — 0.8 s would miss most readings; 1.5 s covers the 90th percentile with the other two models' time
REGISTER_WAIT_S = 1.5

#: the mood events (operator decision 5: banter through the tone path and its caps, no cap raised). Advice is
#: the SMALLEST: its second witness is only the form of the message (a question) — see THE RULE above
EV_ASKED = (+0.02, +0.005)
EV_CONFIDED = (+0.05, +0.0125)
EV_BANTER = (+0.03, +0.03)
EV_VENTING = (-0.05, -0.0125)
CAUSE_ASKED = "the owner asked for my opinion"
CAUSE_CONFIDED = "the owner confided in me"
CAUSE_BANTER = "the owner was joking around"
CAUSE_VENTED = "the owner vented"
#: the intent's events are TONES (kind ``mood.KIND_SENTIMENT``: the ledger's one tone slot, the private shares
#: members never hear) — their cause sentence is what tells them from the sentiment's own on the events log
EVENT_CAUSES = frozenset((CAUSE_ASKED, CAUSE_CONFIDED, CAUSE_BANTER, CAUSE_VENTED))
#: the 💗 line's short form of each cause (``sentiment.rating_line``: "mood nudged up +0.02 (asked for my
#: opinion)"). Every cause of EVENT_CAUSES has one (a test enumerates them)
CAUSE_SHORT = {CAUSE_ASKED: "asked for my opinion", CAUSE_CONFIDED: "confided in me",
               CAUSE_BANTER: "joking around", CAUSE_VENTED: "vented"}

ENV = "GHOST_INTENT"
MODE_OFF, MODE_JOURNAL, MODE_ON = "0", "journal", "1"
EFFECTS = ("register", "mood", "judge", "character", "members")
_DEFAULT_ON = {"register": True, "mood": True, "judge": True, "character": True, "members": False}

#: why no mood event (the ``why`` of ``mood_event``)
WHY_NO_READING = "no intent reading"
WHY_UNCLEAR = "no clear intent"
WHY_NO_EFFECT = "this intent moves nothing"
#: a label in LABELS no outcome was decided for — a test fails on it
WHY_UNDECIDED = "no effect decided for this intent"
WHY_NO_WITNESS = "nothing else in the message backs the intent"
#: ``refusal``: a label that is a tone, read under a tone of the other sign ("haha this is awful")
WHY_TONE_AGAINST = "the message's tone says otherwise"
#: … and an upbeat label with an annoyed emotion behind it
WHY_COLD = "an annoyed emotion behind it"

#: what each label does to the mood (None: nothing). Every label of LABELS has an entry (a test enumerates it)
MOOD_OUTCOMES: Dict[str, Optional[str]] = {
    "advice": CAUSE_ASKED, "confiding": CAUSE_CONFIDED, "banter": CAUSE_BANTER, "venting": CAUSE_VENTED,
    "instruction": None, "question": None, "correcting": None, "thanks": None, "greeting": None, "reply": None}
_EVENTS = {CAUSE_ASKED: EV_ASKED, CAUSE_CONFIDED: EV_CONFIDED, CAUSE_BANTER: EV_BANTER, CAUSE_VENTED: EV_VENTING}
#: the labels that ARE a tone, and its side ("positive" / "negative"); None: not a tone. Every label has an entry
INTENT_TONE: Dict[str, Optional[str]] = {
    "banter": "positive", "venting": "negative", "advice": None, "confiding": None, "instruction": None,
    "question": None, "correcting": None, "thanks": None, "greeting": None, "reply": None}

#: the register's clauses, in the order they are said
REG_URGENT, REG_PATIENT, REG_ACKNOWLEDGE, REG_JOKE, REG_BRIEF = "urgent", "patient", "acknowledge", "joke", "brief"
CLAUSES = {
    REG_URGENT: "answer first, then anything else",
    REG_PATIENT: "be patient and explain your reasoning",
    REG_ACKNOWLEDGE: "acknowledge it before anything else",
    REG_JOKE: "a light joke is welcome",
    REG_BRIEF: "keep the prose brief",
}
#: what each label may add to the register (None: nothing); every label has an entry (a test enumerates it)
REGISTER_OUTCOMES: Dict[str, Optional[str]] = {
    "advice": REG_PATIENT, "instruction": REG_BRIEF, "confiding": REG_ACKNOWLEDGE, "venting": REG_ACKNOWLEDGE,
    "banter": REG_JOKE, "question": None, "correcting": None, "thanks": None, "greeting": None, "reply": None}
#: THE RULE: the clauses that cannot offend when the reading is wrong — they skip the doubt. Only clauses that are
#: not a tone label's may be here (a tone label's refusal is about whether the feeling is there at all). "answer
#: first, then anything else" is here: putting the answer first costs nothing to a sarcastic owner either. Every
#: clause of ``CLAUSES`` is either here or refused under doubt (a test enumerates them)
DOUBT_FREE_CLAUSES = frozenset((REG_URGENT, REG_PATIENT, REG_BRIEF))
REGISTER_HEAD = "HOW TO PITCH THIS REPLY (read from the message you are answering): "
REGISTER_TAIL = (". A MANNER or MOOD line above comes first. This changes how you pitch the reply only: the same "
                 "facts, checks, tool use and every point the answer needs.")
AUDIENCES = ("owner", "others", "public")
#: at or below this humour dial the owner is not offered the joke clause
HUMOUR_NO_JOKE = -1

# ── the second witnesses (English) ──────────────────────────────────────────
#: a laugh in the text, or a laughing / winking emoji. Not "kidding" / "joking" ("I'm not kidding")
_LAUGH = re.compile(
    r"(?:\b(?:l+o+l+z?|lmf?ao+|rofl|ha(?:ha)+h?|he(?:he)+|jk|j/k)\b|χα(?:χα)+|"
    r"[😂🤣😄😆😅😜😉😛😝😁]|(?:^|\s)[:;]-?[)DPp](?=\s|$))", re.IGNORECASE)
_LAUGH_EMOTIONS = frozenset(("amusement", "joy"))
_ASK_EMOTIONS = frozenset(("curiosity", "confusion"))
_UNSURE_EMOTIONS = frozenset(("confusion", "nervousness"))
_HURRY_EMOTIONS = frozenset(("nervousness", "fear"))
#: the owner as the SUBJECT ("I", "I'm", "me") — a possessive alone ("send my report") is an instruction's
_FIRST_PERSON = re.compile(r"\b(?:i|i'm|im|i've|i'd|i'll|me|myself)\b", re.IGNORECASE)
#: hurry in the words — whole words only ("quick question", "quicksort", "nowhere" are not hurry)
_HURRY = re.compile(r"\b(?:asap|urgent|urgently|immediately|right now|right away|hurry|emergency)\b|\bnow!+",
                    re.IGNORECASE)


def _env_on(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in ("0", "off", "false", "no")


def mode() -> str:
    """``"0"``, ``"journal"`` or ``"1"`` from ``GHOST_INTENT``. Anything unreadable is ``journal`` (the safe
    default: it reads and moves nothing)."""
    v = str(os.environ.get(ENV, MODE_JOURNAL) or "").strip().lower()
    if v in ("0", "off", "false", "no"):
        return MODE_OFF
    if v in ("1", "on", "true", "yes"):
        return MODE_ON
    return MODE_JOURNAL


def would(effect: str) -> bool:
    """Would ``effect`` run once the mode is ``1`` — read (mode not ``0``) and its own switch on? What journal
    mode writes down. Unknown effect: False."""
    if effect not in EFFECTS or mode() == MODE_OFF:
        return False
    return _env_on(f"{ENV}_{effect.upper()}", _DEFAULT_ON[effect])


def acts(effect: str) -> bool:
    """Does ``effect`` ("register" | "mood" | "judge" | "character" | "members") run now? Only with
    ``GHOST_INTENT=1`` and its own switch on. Unknown effect: False."""
    return mode() == MODE_ON and would(effect)


def _prob(v) -> Optional[float]:
    """A probability: a real number (not a bool, not a string) in [0, 1]; None otherwise. Never raises."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    try:
        v = float(v)
    except (OverflowError, ValueError):          # 10**400
        return None
    if not math.isfinite(v) or not 0.0 <= v <= 1.0:
        return None
    return v


def _parse_intent(raw) -> Optional[Dict[str, float]]:
    """A non-empty dict of ``LABELS`` → probability, rounded; None when any key or value is not one."""
    if not isinstance(raw, dict) or not raw:
        return None
    if not all(isinstance(k, str) and k in LABELS for k in raw):
        return None
    vals = {k: _prob(v) for k, v in raw.items()}
    if any(v is None for v in vals.values()):
        return None
    return {k: round(v, 4) for k, v in vals.items()}


def parse(body) -> Optional[dict]:
    """The intent part of the service's answer: ``{"intent": {label: p}, "urgent": p}``, each key present only
    when its field is good. An intent field that is not a non-empty dict of ``LABELS`` → probability in [0, 1]
    is dropped whole (as an unknown emotion label is); ``urgent`` the same, on its own. None when neither is
    there. Never raises."""
    if not isinstance(body, dict):
        return None
    out: dict = {}
    try:
        got = _parse_intent(body.get("intent"))
        if got is not None:
            out["intent"] = got
    except Exception:  # noqa: BLE001
        pass
    u = _prob(body.get("urgent"))
    if u is not None:
        out["urgent"] = round(u, 4)
    return out or None


def _intent_of(reading) -> Optional[Dict[str, float]]:
    """The reading's intent, validated as ``parse`` does (a reading may come from anywhere)."""
    if not isinstance(reading, dict):
        return None
    try:
        return _parse_intent(reading.get("intent"))
    except Exception:  # noqa: BLE001
        return None


def top(reading) -> Tuple[str, float]:
    """``(label, p)``: the highest-scoring label, accepted only at or above its own floor (``LABEL_MIN``, compared
    at 4 decimals); ``("", 0.0)`` otherwise — no reading, a malformed one, a top label under its floor or one that
    is ``NEVER``, or two labels tied at the top. Never the second-highest label instead (the scores are
    independent, and this model's second labels are unreliable)."""
    got = _intent_of(reading)
    if not got:
        return "", 0.0
    ranked = sorted(((v, k) for k, v in got.items()), reverse=True)
    p, lab = ranked[0]
    if len(ranked) > 1 and round(ranked[1][0], 4) == round(p, 4):
        return "", 0.0                         # a tie at the top is no reading
    floor = LABEL_MIN.get(lab, NEVER)
    if floor is NEVER or round(p, 4) < floor:
        return "", 0.0
    return lab, p


def counted(reading) -> str:
    """The label a day record counts this message under, or "" — THE same rule as every effect (``top``: the
    top label at its own floor; ``reply`` and ``thanks`` never). What ``days`` calls (no second cut)."""
    return top(reading)[0]


def urgent(reading) -> bool:
    """Does the urgency reading reach ``URGENT_MIN``?"""
    u = _prob(reading.get("urgent")) if isinstance(reading, dict) else None
    return u is not None and round(u, 4) >= URGENT_MIN


def _felt(reading, names) -> bool:
    emo = reading.get("emotions") if isinstance(reading, dict) else None
    if not isinstance(emo, dict):
        return False
    return any(k in names and _prob(v) is not None and v >= M.SENTIMENT_EMOTION_MIN for k, v in emo.items())


def says_correcting(reading) -> bool:
    """Is "correcting" the message's TOP label at its floor (0.90, ``LABEL_MIN``)? It only nominates the message
    to the existing correction judge (the judge is the second witness); it decides nothing else."""
    return top(reading)[0] == "correcting"


def refusal(reading, lab: str, text: str = "", *, after_trouble: bool = False) -> str:
    """THE RULE (module head): why label ``lab``'s effects are refused for this message, or "". The DOUBT first,
    for every label (review r3: it is the stronger reason — the text and the agent's state — and the only one a
    day close cannot re-derive from the row): the doubt on the tone label (``sentiment.doubtful``), and for a
    label that is itself a tone (``INTENT_TONE``) the doubt on its own side. Then, for a tone label only: no
    backing for the tone (``mood.sentiment_backing`` is False, at any score), a tone of the other sign, and
    (upbeat) an annoyed emotion. Never raises (refuses on error)."""
    try:
        from .sentiment import HELD_UNBACKED, doubtful
        tone = reading.get("label") if isinstance(reading, dict) else None
        emotions = reading.get("emotions") if isinstance(reading, dict) else None
        text = text or ""
        side = INTENT_TONE.get(lab)
        if tone in ("positive", "negative"):
            doubt = doubtful(tone, text, after_trouble=after_trouble)
            if doubt:
                return doubt
        if side:
            doubt = doubtful(side, text, after_trouble=after_trouble)
            if doubt:
                return doubt
            if tone in ("positive", "negative"):
                if M.sentiment_backing(tone, emotions) is False:
                    return HELD_UNBACKED
                if tone != side:
                    return WHY_TONE_AGAINST
            if side == "positive" and _felt(reading, M.SENTIMENT_COLD):
                return WHY_COLD
        return ""
    except Exception:  # noqa: BLE001
        return WHY_NO_READING


def _unhappy(reading) -> bool:
    """Something unhappy backs the message: with an emotion reading, an unhappy emotion at the tone gate's own
    bar (``mood.sentiment_backing``, the same rule); only with NO emotion reading may the tone label stand in."""
    if not isinstance(reading, dict):
        return False
    backed = M.sentiment_backing("negative", reading.get("emotions"))
    if backed is None:
        return reading.get("label") == "negative"
    return backed


def _witness(lab: str, reading, text: str) -> Tuple[bool, str]:
    """The second witness of each label's mood event (True, "") or why there is none (after ``refusal``)."""
    from .sentiment import _FEELING, _TASK_OPENING
    text = text or ""
    if lab == "advice":
        # the classifier plus the FORM only (a question) — hence the smallest event, EV_ASKED
        ok = M._is_question(text) or _felt(reading, _ASK_EMOTIONS)
    elif lab == "confiding":
        # the owner speaking of themselves as the subject, or a feeling in the words; never a task's wording
        ok = bool(text) and not (_TASK_OPENING.match(text) and not _FEELING.search(text)) and bool(
            _FIRST_PERSON.search(text) or _FEELING.search(text))
    elif lab == "banter":
        ok = bool(_LAUGH.search(text)) or _felt(reading, _LAUGH_EMOTIONS)
    elif lab == "venting":
        ok = _unhappy(reading)
    else:
        return False, WHY_NO_EFFECT
    return ok, "" if ok else WHY_NO_WITNESS


def mood_event(reading, text: str = "", *, after_trouble: bool = False) -> Tuple[Optional["M.Event"], str]:
    """``(event, "")`` — the mood event the message's intent makes, of kind ``mood.KIND_SENTIMENT`` (a TONE:
    booked as the message's one tone, ``SelfModel.note_owner_sentiment(event=...)``) — or ``(None, why)``.
    ``text``: the message's words as read; ``after_trouble``: the agent had trouble the owner saw just now.
    ``refusal`` first, then the label's second witness. Never raises."""
    try:
        if _intent_of(reading) is None:
            return None, WHY_NO_READING
        lab, _p = top(reading)
        if not lab:
            return None, WHY_UNCLEAR
        if lab not in MOOD_OUTCOMES:
            return None, WHY_UNDECIDED
        cause = MOOD_OUTCOMES[lab]
        if cause is None:
            return None, WHY_NO_EFFECT
        why = refusal(reading, lab, text, after_trouble=after_trouble)
        if why:
            return None, why
        ok, why = _witness(lab, reading, text)
        if not ok:
            return None, why
        dp, de = _EVENTS[cause]
        return M.Event(dp, de, cause, M.KIND_SENTIMENT), ""
    except Exception:  # noqa: BLE001
        return None, WHY_NO_READING


def _past_doubt(key: str, reading, lab: str, text: str, after_trouble: bool) -> bool:
    """THE RULE for every clause, the urgent one included: a clause listed in ``DOUBT_FREE_CLAUSES`` (it cannot
    offend when wrong) skips ``refusal``; any other is offered only when ``refusal`` finds nothing (``lab`` is
    "" for a clause tied to no label: then only the doubt on the tone applies)."""
    return key in DOUBT_FREE_CLAUSES or not refusal(reading, lab, text, after_trouble=after_trouble)


def _clause(lab: str, reading, text: str, after_trouble: bool, audience: str, humour) -> str:
    """The register clause of the top label, or "" (no clause, refused, or its witness is missing)."""
    key = REGISTER_OUTCOMES.get(lab)
    if key is None:
        return ""
    if not _past_doubt(key, reading, lab, text, after_trouble):
        return ""
    if key == REG_PATIENT:
        return key if _felt(reading, _UNSURE_EMOTIONS) else ""
    if key == REG_BRIEF:
        # the classifier plus the FORM only (not a question), and no distress behind it — the weakest clause:
        # never said beside another one (``register``)
        return key if text and not M._is_question(text) and not _felt(reading, M.SENTIMENT_UNHAPPY) else ""
    if key == REG_ACKNOWLEDGE:
        return key if _unhappy(reading) else ""
    if key == REG_JOKE:
        if audience == "owner" and humour is not None:
            # review r2: only a real dial (an int in −2..2, not a bool) is the character; anything else (NaN, a
            # string, True) is "the character is not known" — the joke is withheld rather than guessed
            if isinstance(humour, bool) or not isinstance(humour, int) or not -2 <= humour <= 2:
                return ""
            if humour <= HUMOUR_NO_JOKE:
                return ""
        return key if _witness("banter", reading, text)[0] else ""
    return ""


def register(reading, *, audience: str, humour: Optional[int] = None, text: str = "",
             after_trouble: bool = False) -> Tuple[str, str]:
    """``(key, line)`` — how to pitch the reply to this message (see the module's head for the arguments), or
    ``("", "")``. Never raises."""
    try:
        if audience not in AUDIENCES or not isinstance(reading, dict):
            return "", ""
        keys = []
        if urgent(reading) and (_HURRY.search(text or "") or _felt(reading, _HURRY_EMOTIONS)) and _past_doubt(
                REG_URGENT, reading, "", text, bool(after_trouble)):
            keys.append(REG_URGENT)
        lab, _p = top(reading)
        if lab:
            k = _clause(lab, reading, text, bool(after_trouble), audience, humour)
            if k and not (k == REG_BRIEF and keys):
                keys.append(k)
        if not keys:
            return "", ""
        said = "; ".join(CLAUSES[k] for k in keys)
        return "+".join(keys), REGISTER_HEAD + said[0].upper() + said[1:] + REGISTER_TAIL + "\n\n"
    except Exception:  # noqa: BLE001
        return "", ""


def event_nudge(cause: str) -> float:
    """The mood step of the intent event with this cause (``EV_*``), 0.0 for any other."""
    return float(_EVENTS.get(cause, (0.0, 0.0))[0])


def _pct(p: float) -> str:
    return f"{p * 100:.0f}%"


def _under(p: float, floor: float) -> str:
    """``p`` as a percentage that never reads as the floor itself (0.8996 under 0.90 is "89.9%", not "90%")."""
    if _pct(p) != _pct(floor):
        return _pct(p)
    return f"{math.floor(p * 1000) / 10:.1f}%"


def unclear_why(reading) -> str:
    """Why no label is taken, in plain words, with the next score: ``"advice 89% is under the 90% it needs; next
    question 40%"``, ``"reply is never taken; next thanks 60%"``, ``"advice and question tie at 60%"``. "" when a
    label IS taken or there is no (good) reading. Never raises."""
    try:
        got = _intent_of(reading)
        if not got or top(reading)[0]:
            return ""
        ranked = sorted(((v, k) for k, v in got.items()), key=lambda vk: (-vk[0], vk[1]))
        p, lab = ranked[0]
        if len(ranked) > 1 and round(ranked[1][0], 4) == round(p, 4):
            tied = [k for v, k in ranked if round(v, 4) == round(p, 4)]
            return f"{', '.join(tied[:-1])} and {tied[-1]} tie at {_pct(p)}"
        floor = LABEL_MIN.get(lab, NEVER)
        why = (f"{lab} is never taken" if floor is NEVER
               else f"{lab} {_under(p, floor)} is under the {_pct(floor)} it needs")
        if len(ranked) > 1:
            why += f"; next {ranked[1][1]} {_pct(ranked[1][0])}"
        return why
    except Exception:  # noqa: BLE001
        return ""


def line_part(reading) -> str:
    """The 💗 line's intent part for an owner's scored message, never the words: ``" · intent: advice 97%"``,
    ``" · intent: instruction 80%, urgent 90%"``, ``" · intent: unclear (advice 89% is under the 90% it needs;
    next question 40%)"`` — the runner-up scores ONLY when no label is taken. Urgency only at ``URGENT_MIN``.
    "" with no (or a malformed) reading. Never raises."""
    try:
        parts = []
        if _intent_of(reading):
            lab, p = top(reading)
            parts.append(f"{lab} {_pct(p)}" if lab else f"unclear ({unclear_why(reading)})")
        if urgent(reading):
            parts.append(f"urgent {_pct(_prob(reading['urgent']))}")
        return (" · intent: " + ", ".join(parts)) if parts else ""
    except Exception:  # noqa: BLE001
        return ""


def log_part(reading, *, percent: bool = True) -> str:
    """The log line's intent part, never the words: ``" · intent: advice 93%, urgent 90%"``, ``" · intent:
    unclear (advice 89%, question 40%)"`` (the two highest when the top one is not taken: under its floor, never
    taken, a tie); "" with no (or a malformed) reading. ``percent=False``: labels only
    (a member's line)."""
    try:
        got = _intent_of(reading)
        parts = []
        if got:
            lab, p = top(reading)
            if lab:
                parts.append(f"{lab} {p * 100:.0f}%" if percent else lab)
            else:
                ranked = sorted(((v, k) for k, v in got.items()), reverse=True)[:2]
                parts.append("unclear" + (" (" + ", ".join(f"{k} {v * 100:.0f}%" for v, k in ranked) + ")"
                                          if percent else ""))
        if urgent(reading):
            parts.append(f"urgent {_prob(reading['urgent']) * 100:.0f}%" if percent else "urgent")
        return (" · intent: " + ", ".join(parts)) if parts else ""
    except Exception:  # noqa: BLE001
        return ""
