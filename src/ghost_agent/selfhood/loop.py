"""The strange loop (§4UH-loop, 2026-11-12): what makes the agent's "I" a loop, and the owner's ``mirror`` view of it.

The loop, with every arrow real:

    the dials → the character's sentence, heard by every private reply to the owner (``character.section``)
      → the replies → what they bring back: how the owner talks to it (the emotions of the messages), how the
        agent's own work went, how its answers were judged (the mood's journals)
      → the day record (``days``) → the daily rule against the agent's own 21-day norm (``character.daily``)
      → a dial moves → a new sentence.

Beside it, once a day, the agent LOOKS at a few of its own replies and says how they sound on the same
dials (``selfview``). That look is shown, not counted (``GHOST_SELFVIEW_VOTE=1`` lets it vote): a model rating
itself, never checked against blind readers.

Before this, the loop was the Mirror (§4SV–§4TZ): one sentence a day that the model rewrote from its own
replies. Measured blind it changed nothing in them, the character replaced it on 2026-11-11, and on
2026-11-12 the Mirror's code and diary were removed. What is here is the little the new loop still needs from
it (where the home is, which replies a look may read, the resting point's unit and switch, a probe's
self-image), and the reply to ``mirror``: the path the loop has taken — each sentence the character has had,
what moved it, and what the agent sees in its own replies. Owner-only, answered by the system (§4PY).
"""

from __future__ import annotations

import datetime
import hashlib
import logging
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("GhostSelfhood")

# ── where the agent's home is ───────────────────────────────────────────────


def home_from_memory_dir(memory_dir) -> Optional[Path]:
    """GHOST_HOME from the context's ``memory_dir`` ($GHOST_HOME/system/memory), or None.
    (Live bug 11-02: call sites took ``memory_dir.parent`` = system/ as the home.)"""
    if not isinstance(memory_dir, (str, Path)) or not str(memory_dir):
        return None
    return Path(memory_dir).parent.parent


def _mood_date(now: Optional[datetime.datetime] = None) -> datetime.date:
    """The MOOD day of ``now`` (``mood.mood_day``: it ends at 04:00). A naive time is this machine's local time."""
    from . import mood as _M
    if now is not None and now.tzinfo is None:
        now = now.astimezone()
    return _M.mood_day(now)


def _look_epoch(at) -> float:
    """A stored "at" (local ISO, or UTC with "Z") as an epoch, or 0.0. py3.10's ``fromisoformat`` refuses "Z"."""
    try:
        return datetime.datetime.fromisoformat(str(at or "").replace("Z", "+00:00")).timestamp()
    except (ValueError, OverflowError, OSError):
        return 0.0


# ── the replies a look may read (``selfview.look``) ────────────────────────

#: how far back a reply may be picked from (days), when the last look is older or there is none
REPLY_WINDOW_DAYS = 2
#: replies shorter than this say too little about who wrote them ("Done.", "👍")
MIN_REPLY_CHARS = 80
#: replies shown per look: across different subjects, what they share is the manner
REPLIES_PER_LOOK = 6
#: how much of each reply the model sees (its start)
REPLY_MAX_CHARS = 600
#: a reply written under a mood at this intensity or more (or under a manner) is the mood talking, not the
#: character: the look is never shown it
STRONG_MOOD_MIN = 2

#: a runtime abort marker (core/agent.py's _ABORT_MARKER_RE): the reply is the runtime talking
_ABORT_RE = re.compile(r"\[ATTEMPT_ABORTED_[A-Z_]+\]")


def pick_replies(home, now: Optional[datetime.datetime] = None,
                 rng: Optional[random.Random] = None, after="") -> List[dict]:
    """Up to REPLIES_PER_LOOK of the OWNER's delivered replies written since the previous look (``after``: its
    "at") and never more than REPLY_WINDOW_DAYS back, at random, none written under a strong mood or a manner —
    the census's own reader, so members', probes', replays' and jobs' turns are never picked.
    [{"text", "at"}], empty when there is none. Trajectory day folders are UTC dates, so they are read in UTC
    with one day to spare and the turns are kept by their own timestamp."""
    pool = [{"text": text, "at": str(t.get("timestamp") or "")}
            for t, text in _replies_since(home, now, after) if not written_under_strong_mood(t)]
    return (rng or random).sample(pool, min(REPLIES_PER_LOOK, len(pool)))


def _replies_since(home, now: Optional[datetime.datetime], after) -> List[Tuple[dict, str]]:
    """The owner's delivered replies since the previous look and inside the window, with their text, before
    the strong-mood filter. (turn, text) pairs; [] when unreadable."""
    try:
        from ..core.failure_census import _owner_turns, _window, _epoch
    except Exception:  # noqa: BLE001
        return []
    now_utc = (now.astimezone(datetime.timezone.utc) if now is not None
               else datetime.datetime.now(datetime.timezone.utc))
    since = max(now_utc.timestamp() - REPLY_WINDOW_DAYS * 86400, _look_epoch(after))
    try:
        sysd = Path(home) / "system"
        good, _bad = _owner_turns(sysd, _window(now_utc.date(), REPLY_WINDOW_DAYS + 1))
    except Exception as e:  # noqa: BLE001
        logger.debug("loop: owner turns unreadable: %s", e)
        return []
    out = []
    for t in good:
        text = str(t.get("final_response") or "").strip()
        if len(text) < MIN_REPLY_CHARS or _ABORT_RE.search(text):
            continue
        if not (since < _epoch(t.get("timestamp")) <= now_utc.timestamp() + 60):
            continue
        out.append((t, text))
    return out


def written_under_strong_mood(turn: dict) -> bool:
    """Was this reply written under a mood at ``STRONG_MOOD_MIN`` or more, or under a manner? Read off the turn
    record's ``extra.mood_tone`` VALUES (``intensity``, ``manner_name``), never its sentences. A record with no
    tone line, or one from before §4UD that holds only sentences, was written under no strong mood."""
    try:
        tone = (turn.get("extra") or {}).get("mood_tone")
    except AttributeError:
        return False
    if not isinstance(tone, dict):
        return False
    if isinstance(tone.get("manner_name"), str) and tone["manner_name"].strip():
        return True
    n = tone.get("intensity")
    return isinstance(n, int) and not isinstance(n, bool) and n >= STRONG_MOOD_MIN


# ── the resting point's unit and switch (``character.rest_pair``) ───────────

#: each unit of the resting point is this much of the mood's pair; at most ``REST_UNITS`` per axis
REST_UNIT = 0.06
REST_UNITS = 2
REST_ENV = "GHOST_MIRROR_REST"
_REST_WORDS = ({-2: "gloomy", -1: "a little low", 0: "even", 1: "a little sunny", 2: "sunny"},
               {-2: "sluggish", -1: "unhurried", 0: "even", 1: "brisk", 2: "lively"})


def rest_enabled() -> bool:
    """``GHOST_MIRROR_REST=0`` turns the resting point off (the name is kept so a live config still works)."""
    return os.environ.get(REST_ENV, "1").strip() != "0"


# ── a probe's self-image (§4UF) ────────────────────────────────────────────

#: what the turn record's ``extra.self_image`` says when the sentence came from a probe's header
PROBE_DATE = "probe"


def probe_section(sentence: str, rule: str = "") -> str:
    """The state-block section for a self-image a PROBE supplied: the character's header, the probe's sentence,
    and the probe's rule text or (without one) the character's standing rule. "" without a sentence."""
    from . import character as _character
    sentence = " ".join(str(sentence or "").split())
    if not sentence:
        return ""
    rule = " ".join(str(rule or "").split()) or _character.RULE
    return f"{_character.HEADER}\n«{sentence}»\n{rule}\n\n"


def probe_stamp(section: str) -> str:
    """What the turn record says a probe heard — ``probe:<6 hex of the section>``."""
    return f"{PROBE_DATE}:{hashlib.sha256(str(section or '').encode('utf-8')).hexdigest()[:6]}"


# ── the owner's ``mirror`` command: the path the loop has taken ─────────────

_COMMAND_RE = re.compile(r"^\s*/?mirror\s*[.!?]*\s*$", re.IGNORECASE)
#: steps of the path shown (the oldest beyond this are counted, not listed)
SHOW_STEPS = 8


def is_mirror_command(text: str) -> bool:
    return bool(_COMMAND_RE.match(str(text or "")))


def _local_day(ts) -> str:
    """A stored UTC "at" as the local date "YYYY-MM-DD", or its first ten characters."""
    try:
        d = datetime.datetime.fromisoformat(str(ts or "").replace("Z", "+00:00"))
    except ValueError:
        return str(ts or "")[:10]
    if d.tzinfo is None:
        d = d.replace(tzinfo=datetime.timezone.utc)
    return d.astimezone().strftime("%Y-%m-%d")


def _signed(n: int) -> str:
    return f"{n:+d}" if n else "0"


def path_steps(character, rows: List[dict]) -> List[Dict[str, object]]:
    """The character's log as the loop's path: one step per moment something changed (the rows a close writes
    together share their "at"). Each step: {"day", "changes": [words], "by": "itself" | "you" | "start",
    "sentence" (after the step, "" when no trait is heard), "changed" (did the sentence change)}. Rows of one
    kind ("itself" or "you") stamped within the same minute are one step.
    The dials are walked from the ``start`` row (or the starting character); a row it cannot read (an unknown
    dial, a "from"/"to" that is not a whole number) is skipped."""
    dials = dict(character.START)
    steps: List[dict] = []
    prev_sentence = character.sentence(dials)
    for r in rows:
        kind = r.get("kind")
        at = str(r.get("at") or "")
        if kind == character.KIND_START:
            got = r.get("dials") if isinstance(r.get("dials"), dict) else character.START
            dials = {d: character.clamp(got.get(d, 0)) for d in character.DIALS}
            prev_sentence = character.sentence(dials)
            steps.append({"at": at, "day": _local_day(at), "changes": [], "by": "start",
                          "sentence": prev_sentence, "changed": True, "why": []})
            continue
        d = character.resolve_dial(r.get("dial"))
        if not d:
            continue
        if kind in (character.KIND_MOVE, character.KIND_OWNER, character.KIND_VETO):
            got = (r.get("from"), r.get("to"))
            if not all(isinstance(v, int) and not isinstance(v, bool) for v in got):
                continue                                       # a hand-edited row: not a step
            a, b = character.clamp(got[0]), character.clamp(got[1])
            dials[d] = b
            word = f"{d} {_signed(a)} → {_signed(b)}" + (" (your veto)" if kind == character.KIND_VETO else "")
            by = "itself" if kind == character.KIND_MOVE else "you"
            why = str(r.get("why") or "") if kind == character.KIND_MOVE else ""
        elif kind in (character.KIND_PIN, character.KIND_UNPIN):
            word = f"{d} {'pinned' if kind == character.KIND_PIN else 'unpinned'}"
            by, why = "you", ""
        else:
            continue
        sentence = character.sentence(dials)
        # one daily close writes its rows within the same minute (each row has its own stamp)
        if steps and str(steps[-1]["at"])[:16] == at[:16] and steps[-1]["by"] == by:
            last: dict = steps[-1]                             # one close, several dials: one step
            last["changes"].append(word)
            if why:
                last["why"].append(f"{d}: {why}")
            last["changed"] = bool(last["changed"]) or sentence != last["sentence"]
            last["sentence"] = sentence
        else:
            steps.append({"at": at, "day": _local_day(at), "changes": [word], "by": by, "sentence": sentence,
                          "changed": sentence != prev_sentence, "why": [f"{d}: {why}"] if why else []})
        prev_sentence = sentence
    return steps


def _step_line(step: Dict[str, object]) -> str:
    s = str(step.get("sentence") or "")
    said = f"«{s}»" if s else "(every dial at 0 — no trait heard)"
    if step["by"] == "start":
        return f"- **{step['day']}** — started as {said}"
    by = "by itself, from how its days went" if step["by"] == "itself" else "by you"
    out = f"- **{step['day']}** — {', '.join(step['changes'])}, {by}"
    out += f" → {said}" if step.get("changed") else " (the sentence did not change)"
    for w in step.get("why") or []:
        out += f"\n  - {w}"
    return out


def banner(home) -> str:
    """The whole reply to ``mirror``: the loop in one paragraph, the path it has taken (each sentence and what
    moved it), where it is now, and what the latest daily looks saw. No reply text, no tastes, no rough kinds.
    Never raises (a broken part is said, not thrown)."""
    from . import character as C
    head = ("**The strange loop** — my character is a sentence that every reply I write to you hears. What those "
            "replies bring back (how you talk to me, how my own work goes, how my answers are judged) is counted "
            "each day against my own last three weeks, and when the days add up a dial moves, which changes the "
            "sentence the next replies hear: self → replies → days → self.")
    if not C.enabled():
        return (head + "\n\nThe character is switched off (GHOST_CHARACTER=0), so no sentence is heard and the "
                "loop does not run.")
    parts = [head]
    now = ""
    try:
        state = C.load(home)
        s = C.sentence(state["dials"])
        if state.get("unreadable"):
            now = (f"⚠ The character file cannot be read, so replies hear the STARTING character: «{s}». "
                   "Type \"character\" for what to do.")
        else:
            now = (f"**Now:** «{s}» ({C.describe(state['dials'])})." if s else
                   "**Now:** every dial is at 0 — no trait is heard.")
            if state.get("pins"):
                now += " Pinned: " + ", ".join(state["pins"]) + "."
            a, b = C.rest_units(state["dials"])
            if (a, b) != (0, 0) and rest_enabled():
                now += (f" With nothing happening, my mood rests at pleasantness {_REST_WORDS[0][a]}, "
                        f"energy {_REST_WORDS[1][b]}.")
    except Exception:  # noqa: BLE001
        logger.warning("loop: the character could not be read", exc_info=True)
        now = "The character could not be read just now (see the agent log); type \"character\"."
    try:
        steps = path_steps(C, C.log_rows(home))
        if steps:
            shown = steps[-SHOW_STEPS:]
            moved = sum(len(st["changes"]) for st in steps if st["by"] == "itself")
            title = (f"**Where it has gone** ({moved} move{'s' if moved != 1 else ''} by itself so far"
                     + (f"; the oldest {len(steps) - len(shown)} step(s) not shown" if len(shown) < len(steps)
                        else "") + "):")
            parts.append(title + "\n\n" + "\n".join(_step_line(st) for st in shown))
        else:
            parts.append("**Where it has gone:** nowhere yet — this is the starting character.")
    except Exception:  # noqa: BLE001
        logger.warning("loop: the character's path could not be shown", exc_info=True)
        parts.append("The character's path could not be read just now (see the agent log); type \"character\".")
    parts.append(now)
    try:
        from . import selfview as _selfview
        looks = _selfview.history_lines(home)
        how = ("Once a day I read a few of my latest replies to you, without my dials, and rate how they sound. "
               "\"Meant\" is my dial then, \"sounded\" what the replies showed. ")
        how += ("When they sit two steps from a dial, or across 0, that adds a vote to the day's pull."
                if _selfview.vote_enabled() else
                "This is shown, not counted: a model rating itself, never checked against blind readers.")
        if looks:
            body = "\n".join(looks)
            seen = _selfview.line(home)
            if seen:
                body += "\n\n" + seen
        elif _selfview.trying_today(home):
            body = "No look yet — today's could not be read on the first try; one more try comes in an idle window."
        else:
            body = ("No look yet — the first happens in an idle window once there are at least "
                    f"{_selfview.MIN_REPLIES} replies of mine to you to read (from the last two days, not short, "
                    "not written under a strong mood or a manner).")
        parts.append("**What I see when I look at my replies.** " + how + "\n\n" + body)
    except Exception:  # noqa: BLE001
        logger.warning("loop: the looks could not be shown", exc_info=True)
        parts.append("My looks at my own replies could not be read just now (see the agent log).")
    return "\n\n".join(parts)
