"""The mood — two numbers, what moves them, and how they are read (§4TC).

Until 2026-11-06 the mood was a LABEL: derived from verifier verdict streaks
and the idle clock, or written as free text through ``set_mood``. In three
months the derived one changed 148 times and only three labels ever fired
(idle 101, satisfied 30, stuck 12); it keyed on verdicts that have mostly
been off since mid-October, test traffic made it "stuck", and nothing in a
reply depended on it. That updater is gone (``derive_mood``, the acute /
standing / stale machinery, the six-hour hold of a self-set label).

The mood is now a PAIR — pleasantness ``P`` and energy ``E``, each in
[-1, 1] — and everything else is read off it:

* both DECAY toward 0 with ``HALF_LIFE_S``, applied when the mood is read or
  moved (no background job);
* the INTENSITY (0–3) is the magnitude ``max(|P|, |E|)`` against
  ``INTENSITY_THRESHOLDS``; below the first one there is no mood ("neutral");
* the LABEL is the region the pair lies in (``region``): eight of them, by
  the signs of ``P`` and ``E`` and by whether the smaller of the two is
  "near 0" next to the larger (``AXIS_RATIO``). "curious" is a TAG, not a
  ninth region: a mood moved by a novelty event reads "curious" for as long
  as it is pleasant and energetic (P > 0 and E > 0);
* a label or an intensity is given up only once the pair is clearly past a
  boundary (``AXIS_MARGIN``, ``INTENSITY_MARGIN``), so decay does not make
  it flicker. Decay scales both numbers alike, so it never changes the
  region at all — only the intensity steps down.

What moves the pair is a closed list of EVENTS, each with a named size:
what happened in an owner's turn (``turn_events``), the owner's thumbs, a
correction, thanks, and the owner saying so in chat (``anchor_for`` —
``introspect(action="set_mood")``). A long quiet moves nothing: boredom is
what a reader other than the owner sees over a neutral mood
(``boredom_level``), never a value in the pair.

This module is pure: numbers in, numbers and text out, no I/O and no clock
read. ``SelfStateThread`` (state.py) owns the stored pair, the journal and
the read-modify-write; ``tone_line`` is the ONE sentence a reply's prompt
carries, and it depends on the label, the intensity and the manner only —
never on a time, a decimal or a cause — so it is byte-stable between changes.
"""

from __future__ import annotations

import datetime
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ── the pair ────────────────────────────────────────────────────────────────

#: both numbers halve in this many seconds
HALF_LIFE_S = 2 * 3600.0

#: ``max(|P|, |E|)`` at or above these → intensity 1, 2, 3; below the first
#: there is no mood. The automatic mood has no cap: it can reach 3.
INTENSITY_THRESHOLDS = (0.20, 0.45, 0.75)
#: an intensity already held is given up only below its threshold MINUS this
INTENSITY_MARGIN = 0.04

#: the smaller of |P|, |E| divided by the larger: under this the smaller one
#: is "near 0" and the label is an axis label (content / down / restless /
#: bored); at or over it the label is a diagonal one (cheerful / calm /
#: irritable / tired)
AXIS_RATIO = 0.40
#: a label already held is kept until the ratio is past ``AXIS_RATIO`` by this
AXIS_MARGIN = 0.10

NEUTRAL = "neutral"
#: the tagged label: the P+ E+ region reached through a novelty event
CURIOUS = "curious"
_CURIOUS_BASE = "cheerful"
#: the eight regions, then the tagged one
LABELS = ("cheerful", "content", "calm", "restless", "bored",
          "irritable", "down", "tired", CURIOUS)

SOURCE_AUTO = "auto"
SOURCE_OWNER = "owner"


def clamp(x) -> float:
    """A number into [-1, 1]; anything that is not a finite number is 0."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if v != v or v in (float("inf"), float("-inf")):
        return 0.0
    return max(-1.0, min(1.0, v))


def decay(value: float, elapsed_s: float) -> float:
    """``value`` after ``elapsed_s`` seconds: halved every ``HALF_LIFE_S``.
    A negative or unreadable span is no time at all."""
    try:
        t = float(elapsed_s)
    except (TypeError, ValueError):
        return float(value)
    if not (t > 0.0):
        return float(value)
    return float(value) * (0.5 ** (t / HALF_LIFE_S))


def region(p: float, e: float) -> str:
    """The label of the region ``(p, e)`` lies in — signs and relative size
    only, no memory. ``NEUTRAL`` for the origin."""
    ap, ae = abs(p), abs(e)
    major, minor = max(ap, ae), min(ap, ae)
    if major <= 0.0:
        return NEUTRAL
    if minor / major < AXIS_RATIO:
        if ap >= ae:
            return "content" if p > 0 else "down"
        return "restless" if e > 0 else "bored"
    if p > 0:
        return "cheerful" if e > 0 else "calm"
    return "irritable" if e > 0 else "tired"


def _still_in(p: float, e: float, held: str) -> bool:
    """Is ``(p, e)`` still inside ``held``'s region widened by the margin?"""
    ap, ae = abs(p), abs(e)
    major, minor = max(ap, ae), min(ap, ae)
    if major <= 0.0:
        return False
    ratio = minor / major
    if held in ("content", "down"):
        return ap >= ae and (p > 0) == (held == "content") and ratio < AXIS_RATIO + AXIS_MARGIN
    if held in ("restless", "bored"):
        return ae >= ap and (e > 0) == (held == "restless") and ratio < AXIS_RATIO + AXIS_MARGIN
    signs = {"cheerful": (True, True), "calm": (True, False),
             "irritable": (False, True), "tired": (False, False)}.get(held)
    if signs is None:
        return False
    return (p > 0, e > 0) == signs and p != 0 and e != 0 and ratio >= AXIS_RATIO - AXIS_MARGIN


def held_region(p: float, e: float, held: str = "") -> str:
    """``region`` with hysteresis: the region already held stands until the
    pair is clearly outside it."""
    raw = region(p, e)
    base = _CURIOUS_BASE if held == CURIOUS else held
    if base and base != raw and _still_in(p, e, base):
        return base
    return raw


def level_for(magnitude: float, held: int = 0) -> int:
    """The intensity (0–3) for ``max(|P|, |E|)``. Rising is at the threshold;
    coming down from an intensity ``held``, each level is kept down to its
    threshold minus ``INTENSITY_MARGIN`` — so a mood that only decays reads
    the same whenever it is looked at."""
    m = abs(float(magnitude))
    raw = sum(1 for t in INTENSITY_THRESHOLDS if m >= t)
    try:
        h = int(held)
    except (TypeError, ValueError):
        h = 0
    h = max(0, min(len(INTENSITY_THRESHOLDS), h))
    for k in range(h, raw, -1):
        if m >= INTENSITY_THRESHOLDS[k - 1] - INTENSITY_MARGIN:
            return k
    return raw


def decay_step_s(magnitude: float, level: int) -> float:
    """How many seconds of decay it takes ``max(|P|, |E|) = magnitude`` to
    come down TO intensity ``level`` (0–2) — i.e. to fall under the lowered
    bound of ``level + 1``. 0.0 when it is already there. This is what dates
    a mood that wore off while nobody looked."""
    m = abs(float(magnitude))
    try:
        k = max(0, min(len(INTENSITY_THRESHOLDS) - 1, int(level)))
    except (TypeError, ValueError):
        k = 0
    bound = INTENSITY_THRESHOLDS[k] - INTENSITY_MARGIN
    if not (m > bound) or bound <= 0.0:
        return 0.0
    return HALF_LIFE_S * math.log2(m / bound)


def read_off(p: float, e: float, *, held_label: str = "", held_intensity: int = 0,
             novelty: bool = False) -> Tuple[str, int, bool]:
    """``(label, intensity, novelty)`` for a pair, given what was held."""
    n = level_for(max(abs(p), abs(e)), held_intensity)
    if n == 0:
        return NEUTRAL, 0, False
    # the novelty tag: "curious" wherever the pair is pleasant AND energetic.
    # Trouble wins the sign of P — once P is not positive (or the energy is
    # gone) the tag is dropped and the plain region is read. (WHO keeps the
    # tag is the store's rule: it belongs to the move that set it — decay
    # keeps it, any later event that is not a research event clears it.)
    if novelty and p > 0 and e > 0:
        return CURIOUS, n, True
    return held_region(p, e, held_label if held_label != NEUTRAL else ""), n, False


# ── quiet → boredom (read-time only) ────────────────────────────────────────
#
# Boredom is never stored into the pair. A READER that is not the owner sees
# "bored" while the owner has been quiet long enough AND the stored, decayed
# mood is neutral — a real mood that has not worn off is not replaced. When
# the owner returns, the quiet simply ends. (The first rule pulled the energy
# down after 3 quiet hours and wrote it into the pair at the owner's return:
# replayed over a month of the owner's real turns it left 13 % of his turns
# starting neutral and about 40 % starting "bored" — review, §4TC.)

BORED = "bored"
#: quiet for at least this long → bored at intensity 1 …
BOREDOM_AFTER_S = 6 * 3600.0
#: … and for at least this long → intensity 2. Never more.
BOREDOM_DEEP_AFTER_S = 12 * 3600.0
#: the thresholds by level
BOREDOM_THRESHOLDS = {1: BOREDOM_AFTER_S, 2: BOREDOM_DEEP_AFTER_S}
QUIET_ENDED_CAUSE = "the quiet ended"


def boredom_level(quiet_s) -> int:
    """0, 1 or 2 for a quiet of ``quiet_s`` seconds; 0 while the quiet is
    unknown or unreadable."""
    try:
        q = float(quiet_s)
    except (TypeError, ValueError):
        return 0
    if not (q >= BOREDOM_AFTER_S):          # also False for NaN
        return 0
    return 2 if q >= BOREDOM_DEEP_AFTER_S else 1


def quiet_cause(level) -> str:
    """The cause a bored reading carries — the THRESHOLD that was crossed
    ("quiet for 6 hours"), not the running length of the quiet, so the text
    is the same for the whole time the level holds."""
    try:
        t = BOREDOM_THRESHOLDS[int(level)]
    except (KeyError, TypeError, ValueError):
        t = BOREDOM_AFTER_S
    return f"quiet for {int(t // 3600)} hours"


# ── events ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Event:
    """One thing that moves the pair: ``(dp, de)``, why, and its kind."""
    dp: float
    de: float
    cause: str
    kind: str
    novelty: bool = False


KIND_TURN = "turn"
KIND_THUMB_UP = "thumb_up"
KIND_THUMB_DOWN = "thumb_down"
KIND_CORRECTION = "correction"
KIND_PRAISE = "praise"
KIND_OWNER = "owner"          # the owner said so in chat (set_mood)
KIND_QUIET = "quiet"          # a reader first saw the boredom of a long quiet
KIND_QUIET_END = "quiet_end"  # the owner came back
KIND_DECAY = "decay"          # it wore off

#: one failed tool call, or one time a loop guard stepped in
EV_TROUBLE = (-0.12, +0.08)
#: at most this many of them count in one turn
TROUBLE_CAP = 4
#: a turn of at least ``CLEAN_TURN_MIN_CALLS`` tool calls, none failed, no
#: guard, completed, not a long one
EV_CLEAN_TURN = (+0.08, 0.0)
CLEAN_TURN_MIN_CALLS = 3
#: a turn whose research tools read at least ``RESEARCH_MIN_SOURCES`` pages
EV_RESEARCH = (+0.15, +0.20)
RESEARCH_MIN_SOURCES = 3
#: a very long turn — by wall time or by the number of tool calls
EV_LONG_TURN = (-0.15, -0.30)
LONG_TURN_S = 900.0
LONG_TURN_CALLS = 25
#: the owner's reactions
#: thumbs move the pleasantness only (review r2: with an energy component,
#: toggling up / down on one turn pumped the energy by +0.05 per "up")
EV_THUMB_UP = (+0.25, 0.0)
EV_THUMB_DOWN = (-0.30, 0.0)
EV_CORRECTION = (-0.20, 0.0)
EV_PRAISE = (+0.10, 0.0)


@dataclass(frozen=True)
class TurnSignals:
    """What happened in one owner turn, as the loop already knows it."""
    tool_calls: int = 0        # tool calls that got a result
    failures: int = 0          # of those, the ones that failed
    guards: int = 0            # times a loop guard stepped in
    sources: int = 0           # distinct pages the research tools read
    duration_s: float = 0.0
    completed: bool = True     # False for a turn that was cut off
    pressure: bool = False     # the turn ended under the context-pressure lockdown
    owner_set: bool = False    # the owner SET the mood or the manner in this turn


def _n(count: int, one: str, many: str) -> str:
    return f"{count} {one if count == 1 else many}"


def turn_events(sig: TurnSignals) -> List[Event]:
    """The events of one owner turn. A plain chat turn (no tools, no guard,
    not long) gives none — and neither does a turn in which the owner set
    the mood or the manner: his word holds for the turn."""
    out: List[Event] = []
    if bool(sig.owner_set):
        return out
    calls = max(0, int(sig.tool_calls or 0))
    failures = max(0, int(sig.failures or 0))
    guards = max(0, int(sig.guards or 0))
    sources = max(0, int(sig.sources or 0))
    try:
        dur = max(0.0, float(sig.duration_s or 0.0))
    except (TypeError, ValueError):
        dur = 0.0
    pressure = 1 if bool(sig.pressure) else 0
    trouble = failures + guards + pressure
    if trouble > 0:
        k = min(trouble, TROUBLE_CAP)
        parts = []
        if failures:
            parts.append(_n(failures, "tool call", "tool calls") + " failed")
        if guards:
            parts.append("a loop guard stepped in " + ("once" if guards == 1 else f"{guards} times"))
        if pressure:
            parts.append("the context ran out of room (context pressure)")
        out.append(Event(EV_TROUBLE[0] * k, EV_TROUBLE[1] * k,
                         " and ".join(parts) + " in this turn", KIND_TURN))
    if sources >= RESEARCH_MIN_SOURCES:
        out.append(Event(EV_RESEARCH[0], EV_RESEARCH[1],
                         f"read {sources} sources in a research turn", KIND_TURN, novelty=True))
    long_turn = dur >= LONG_TURN_S or calls >= LONG_TURN_CALLS
    if long_turn:
        out.append(Event(EV_LONG_TURN[0], EV_LONG_TURN[1],
                         f"a very long turn ({int(dur // 60)} min, {_n(calls, 'tool call', 'tool calls')})",
                         KIND_TURN))
    if (calls >= CLEAN_TURN_MIN_CALLS and trouble == 0 and bool(sig.completed)
            and not long_turn):
        out.append(Event(EV_CLEAN_TURN[0], EV_CLEAN_TURN[1],
                         f"a turn of {calls} tool calls went through without a failure", KIND_TURN))
    return out


def merge_events(events: List[Event]) -> Optional[Event]:
    """One turn's events as ONE move (their sum, the causes joined)."""
    evs = [ev for ev in (events or []) if ev is not None]
    if not evs:
        return None
    return Event(sum(ev.dp for ev in evs), sum(ev.de for ev in evs),
                 "; ".join(ev.cause for ev in evs), evs[0].kind,
                 novelty=any(ev.novelty for ev in evs))


def reaction_event(kind: str) -> Optional[Event]:
    """The owner's reaction as an event, or None for an unknown kind."""
    if kind == KIND_THUMB_UP:
        return Event(EV_THUMB_UP[0], EV_THUMB_UP[1], "the owner gave a thumbs-up", kind)
    if kind == KIND_THUMB_DOWN:
        return Event(EV_THUMB_DOWN[0], EV_THUMB_DOWN[1], "the owner gave a thumbs-down", kind)
    if kind == KIND_CORRECTION:
        return Event(EV_CORRECTION[0], EV_CORRECTION[1], "the owner corrected my last reply", kind)
    if kind == KIND_PRAISE:
        return Event(EV_PRAISE[0], EV_PRAISE[1], "the owner said thanks", kind)
    return None


# Thanks / praise in the owner's message. NARROW on purpose (review, §4TC:
# of 18 live matches of a plain word list 6 were not praise — "make it look
# awesome", "this wasn't awesome", "a great number of files", "thanks to the
# cache it works"). A message is praise only when it is SHORT and is nothing
# but thanks / praise words and filler. That rule is also what keeps a
# negation out ("no, thanks", "that isn't great", "δεν είναι τέλεια"): a
# negation is neither praise nor filler, so there is no second list of them
# to keep in step.

#: at most this many words
PRAISE_MAX_WORDS = 6
#: one-word thanks / praise
PRAISE_WORDS = frozenset({"thanks", "thx", "cheers", "great", "perfect", "awesome", "nice", "bravo",
                          "ευχαριστώ", "ευχαριστω", "σ'ευχαριστώ", "σ'ευχαριστω",
                          "μπράβο", "μπραβο", "τέλεια", "τελεια"})
#: two-word thanks / praise
PRAISE_PAIRS = frozenset({("thank", "you"), ("thank", "u"), ("many", "thanks"), ("well", "done"),
                          ("good", "job"), ("great", "job"), ("nice", "work"), ("great", "work"),
                          ("good", "work")})
#: words that may stand beside them. "is", "it's" and "that" are NOT among
#: them ("is that so great", "is that perfect" are questions, not praise).
PRAISE_FILLER = frozenset({"ok", "okay", "ghost", "mate", "that's", "thats", "very", "so",
                           "much", "a", "lot", "really", "πολύ", "πολυ", "σε", "σου", "ρε"})
_PRAISE_TOKEN_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*")
#: a message that ENDS with one of these is a question (";" is also what the
#: Greek question mark normalises to)
_QUESTION_MARKS = "?;"


def _is_question(text: str) -> bool:
    """Does the message end with a question mark — once what may follow one
    (closing quotes, brackets, "!", emoji, spaces) is set aside?"""
    for ch in reversed(text):
        if ch in _QUESTION_MARKS:
            return True
        if ch.isalnum():
            return False
    return False


def has_praise(text) -> bool:
    """Is the owner's message thanks or praise — and nothing else? At most
    ``PRAISE_MAX_WORDS`` words; every word is a praise word, half of a praise
    pair, or filler; at least one of them is praise; and it is not a
    question ("great?", "τέλεια;"). Any other word — a negation included —
    makes it not praise. Punctuation, digits and emoji are ignored; a curly
    apostrophe is a plain one ("that’s great")."""
    if not isinstance(text, str):
        return False
    t = unicodedata.normalize("NFKC", text).replace("\u2019", "'").replace("\u2018", "'").lower()
    if _is_question(t):
        return False
    words = _PRAISE_TOKEN_RE.findall(t)
    if not words or len(words) > PRAISE_MAX_WORDS:
        return False
    found, i = 0, 0
    while i < len(words):
        if tuple(words[i:i + 2]) in PRAISE_PAIRS:
            found, i = found + 1, i + 2
        elif words[i] in PRAISE_WORDS:
            found, i = found + 1, i + 1
        elif words[i] in PRAISE_FILLER:
            i += 1
        else:
            return False
    return found > 0


# ── the owner says so ───────────────────────────────────────────────────────

#: ``max(|P|, |E|)`` an owner-set mood starts at, by the intensity asked for
#: (each clearly inside its band)
ANCHOR_MAGNITUDE = {1: 0.32, 2: 0.60, 3: 0.90}
#: the direction of each label's anchor, scaled so its larger number is 1
ANCHOR_DIRECTION = {
    "cheerful": (1.0, 0.75), "content": (1.0, 0.0), "calm": (1.0, -0.75),
    "restless": (0.0, 1.0), "bored": (0.0, -1.0),
    "irritable": (-1.0, 0.75), "down": (-1.0, 0.0), "tired": (-0.75, -1.0),
    CURIOUS: (1.0, 0.75),
}
#: "excited": cheerful, with the energy as the larger number
_ENERGETIC = (0.75, 1.0)
#: obvious other words for a label → (label, an anchor direction or None)
SYNONYMS = {
    "happy": ("cheerful", None), "cheery": ("cheerful", None), "joyful": ("cheerful", None),
    "upbeat": ("cheerful", None), "excited": ("cheerful", _ENERGETIC),
    "satisfied": ("content", None), "pleased": ("content", None),
    "relaxed": ("calm", None), "serene": ("calm", None), "chill": ("calm", None),
    "fidgety": ("restless", None), "impatient": ("restless", None),
    "grumpy": ("irritable", None), "annoyed": ("irritable", None), "cranky": ("irritable", None),
    "angry": ("irritable", None), "irritated": ("irritable", None),
    "sad": ("down", None), "low": ("down", None), "gloomy": ("down", None),
    "sleepy": ("tired", None), "exhausted": ("tired", None), "weary": ("tired", None),
    "interested": (CURIOUS, None), "inquisitive": (CURIOUS, None),
}
#: the ONLY words that take the mood off (matched stripped, case-folded)
CLEAR_WORDS = frozenset({"clear", "none", "neutral"})

#: a manner is a way of talking: set by the owner, never decays
MANNERS = ("sarcastic", "playful", "formal", "terse", "gentle", "dramatic")
DEFAULT_INTENSITY = 2


def resolve_label(word) -> Optional[Tuple[str, Tuple[float, float]]]:
    """``(label, anchor direction)`` for a label or one of its synonyms; None
    for any other word (free text is not a mood)."""
    w = str(word or "").strip().lower()
    if w in ANCHOR_DIRECTION:
        return w, ANCHOR_DIRECTION[w]
    hit = SYNONYMS.get(w)
    if hit is None:
        return None
    label, direction = hit
    return label, (direction or ANCHOR_DIRECTION[label])


def resolve_manner(word) -> str:
    """The manner for ``word``, or "" when it is not one."""
    w = str(word or "").strip().lower()
    return w if w in MANNERS else ""


def parse_intensity(value, default: int = DEFAULT_INTENSITY) -> int:
    """1–3. A missing or unreadable value is the default; a number outside
    the range is brought to its nearest end."""
    if value is None or isinstance(value, bool):
        return default
    try:
        n = int(round(float(str(value).strip())))
    except (TypeError, ValueError, OverflowError):
        return default
    return max(1, min(3, n))


def anchor_for(direction: Tuple[float, float], intensity: int) -> Tuple[float, float]:
    """The pair an owner-set mood starts at."""
    m = ANCHOR_MAGNITUDE[max(1, min(3, int(intensity)))]
    return clamp(direction[0] * m), clamp(direction[1] * m)


# ── the one line a reply's prompt carries ───────────────────────────────────

LEVEL_SENTENCE = {
    1: "At 1 it is a faint tint: word choice only, no remarks about it.",
    2: "At 2 it is clearly noticeable, with a remark or two in character.",
    3: "At 3 it is strong, and you may say outright that you are in this mood.",
}
MANNER_LEVEL_SENTENCE = {
    1: "At 1 it is a faint tint: word choice only, no remarks about it.",
    2: "At 2 it is clearly noticeable, with a remark or two in character.",
    3: "At 3 it is strong throughout.",
}
_LIMITS = ("WHAT you say stays exactly as it would be: the same facts, the same numbers, "
           "the same tool use, the same completeness, no refusing. "
           "This applies to every reply, serious ones included.")


def tone_line(label: str, intensity: int, manner: str = "", manner_intensity: int = 0) -> str:
    """The sentence that makes the mood audible in a reply — "" for a neutral
    mood with no manner. A function of its four arguments and nothing else."""
    try:
        n = int(intensity)
    except (TypeError, ValueError):
        n = 0
    has_mood = bool(label) and label != NEUTRAL and n in LEVEL_SENTENCE
    man = resolve_manner(manner)
    try:
        mn = int(manner_intensity)
    except (TypeError, ValueError):
        mn = 0
    mn = mn if mn in MANNER_LEVEL_SENTENCE else DEFAULT_INTENSITY
    if has_mood:
        return (f"YOUR MOOD RIGHT NOW: {label}, {n} of 3."
                + (f" MANNER (set by the owner): {man}, {mn} of 3." if man else "")
                + " Speak in that mood: it must be audible in HOW you phrase this reply - "
                  "word choice, rhythm - the way a person in that mood would say the same thing. "
                + LEVEL_SENTENCE[n] + " " + _LIMITS)
    if man:
        return (f"MANNER (set by the owner): {man}, {mn} of 3."
                " Speak in that manner: it must be audible in HOW you phrase this reply - "
                "word choice, rhythm - the way a person with that manner would say the same thing. "
                + MANNER_LEVEL_SENTENCE[mn] + " " + _LIMITS)
    return ""


# ── timestamps (shared by the state thread and the tool views) ──────────────

def parse_iso_utc(ts: str) -> Optional[datetime.datetime]:
    """Parse the on-disk timestamp shape (``schema._utcnow_iso``: naive
    ISO + trailing "Z") into an aware UTC datetime. Also tolerates
    offset-aware ISO strings from hand-edited files. None on garbage —
    including non-string garbage (a hand-edited ``"set_at": 1755000000``
    must cost the mood line, not the whole view)."""
    if not isinstance(ts, str):
        return None
    raw = ts.strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1]
    try:
        dt = datetime.datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(datetime.timezone.utc)


def age_seconds(
    ts: str, *, now: Optional[datetime.datetime] = None,
) -> Optional[float]:
    """Seconds elapsed since ``ts``. None when the timestamp doesn't
    parse. Clock skew (future ts) clamps to 0."""
    dt = parse_iso_utc(ts)
    if dt is None:
        return None
    if now is None:
        now = datetime.datetime.now(datetime.timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=datetime.timezone.utc)
    return max(0.0, (now - dt).total_seconds())
