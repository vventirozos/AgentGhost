"""§4SZ: the owner's rules — standing instructions the owner dictates.

What is left of the learning system after the 2026-11-05 evaluation. For a
month the only live loop was: a reply is flagged as a suspect failure → an idle
job replays the request → one rule is proposed → the owner says "learn rule N".
Measured over that month: 13 replays, 5 rules adopted (two the same, three a
restatement of "say only what the tool output shows"), and nothing new for the
last 17 days, because 30 of 30 flagged turns were follow-ups or action requests
a replay cannot rebuild. The one part with a measured effect was the rule the
owner approved, shown on every turn. So that part stays, without the machinery
that fed it: the owner says the rule, and it is a rule.

    learn this rule: <text>      (also: remember this rule: <text>)
    show my rules                (also: list my rules, show all my rules)
    show rule N
    forget rule N                → asks for "confirm forget rule N"
    forget all my rules          → asks for "confirm forget all my rules"

A dictated rule is ONE line after a colon, and it holds no question mark — a
message with one is a question, not a rule (review r2: "learn this rule: if
x > 5 then y = 2x. Now, what is y for x = 7?" was stored whole and never
answered; so was a rule followed by a second request on the next line, and a
quoted copy of the command inside a question; r3: "…for x = 7?!"). What is left is inherent to a command
with no model: one line that states a rule AND asks for something else is
stored as written — the reply quotes what was stored and how to remove it.

The system answers these itself (§4PY: no model call). Only an owner turn on a
private surface reaches here (the caller's gate, `_owner_rule_note`).

A COMMAND GETS NO MODEL, so each one is an exact phrase that is the whole
message, and nothing else is a command (review r1: "add rule: allow tcp 443"
was stored as a standing rule, "delete rule 3" about a firewall and "run the
regression tests" in a coding chat were answered by the system). "this rule"
followed by a colon, "my" before "rules", and the one verb "forget"
are what make a phrase the owner's rules and not a chat about some other rule
(§4PZ took delete/remove out of the old commands for the same reason). A short
message ABOUT the rules that is not a command reaches the model with a note
(`rule_command_hint`): nothing changed, and which phrases do. The replay
loop's old commands ("learn rule N", "show me all proposed rules") get that
note too — a matcher that answered them itself (review r2) also answered
"list all the suggested rules" after the assistant had suggested lint rules.

ONE store, ``$GHOST_HOME/system/owner_rules.json``:
``{"next_n": int, "rules": [{"n", "id", "rule", "added_at"}], "pending": {...}}``.
The old design kept a rule in two places (the lesson playbook and the replay
ledger) and needed an integrity check to keep them equal — a merge had once
rewritten rule 1 with rule 4's text (§4OR). One file has nothing to reconcile.
A rule's number is never reused.

THE STORE NEVER REPLACES WHAT IT CANNOT READ (review r1). A damaged or
unreadable file is left exactly as it is: no rule is shown, every command says
the file is damaged and changes nothing, and the health check names it. (The
first version set a damaged file aside and started empty — the next "learn
this rule" was "Rule 1" again and the health check read "0 owner rules, fine";
on a file it could not READ it simply wrote a new one over three rules.)

At most ``RULES_MAX`` rules, and a full list REFUSES a new one: every rule is
shown on every turn, and the old cap dropped the oldest from the prompt without
saying so (§4OR).
"""

from __future__ import annotations

import datetime
import json
import logging
import re
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger("GhostAgent")

STORE_NAME = "owner_rules.json"
RULES_MAX = 10
RULE_MIN_CHARS = 8
RULE_MAX_CHARS = 400
FORGET_CONFIRM_S = 15 * 60

_LOCK = threading.RLock()


class RuleNote(str):
    """A system-command result: ``banner`` is the whole reply (§4PY)."""
    banner = ""


def _note(banner: str) -> RuleNote:
    r = RuleNote("The system answered this owner command itself; do not call any tool.")
    r.banner = banner
    return r


# ── the commands ────────────────────────────────────────────────────────────
# Each is the WHOLE message (a leading "ok," / "please" is allowed).

_POLITE = r"(?:(?:ok|okay|yes|sure|please|pls)[\s,!.]+)*"
_LEAD = r"^[\s\"'“”‘’«»]*" + _POLITE
_END = r"(?:[\s,]+(?:please|pls))?[\s\"'“”‘’«».!?]*$"
#: a rule's number: "rule 3", "rule #3" (one optional mark — two adjacent `\s*`
#: around it made the match quadratic on a run of spaces)
_N = r"rule\s*(?:#\s*)?(\d{1,6})"
# "this rule" + a COLON + one line. No opening quote (a message that opens
# with a quote is quoting the command), no other separator ("Remember this
# rule, it will be on the exam: …" and "Remember this rule. I told you about
# it yesterday…" were stored), nothing after a line break.
_ADD_START = r"^\s*" + _POLITE + r"(?:learn|remember)\s+this\s+rule\b"
_ADD_RE = re.compile(_ADD_START + r"\s*:", re.I)       # (the rule is what follows: `_dictated`)
_ADD_START_RE = re.compile(_ADD_START, re.I)
# "my" is required: "show me the rules" after "what are the rules of chess?"
# is a follow-up (§4PZ r3), and "list all rules" / "show all rules" is what
# one says about a firewall or a linter (reviews r2, r3)
_WHICH = r"(?:all\s+(?:of\s+)?)?my\s+"
_LIST_RE = re.compile(_LEAD + r"(?:show|list)\s+(?:me\s+)?" + _WHICH + r"(?:owner\s+|adopted\s+)?rules" + _END, re.I)
_SHOW_RE = re.compile(_LEAD + r"show\s+(?:me\s+)?" + _N + _END, re.I)
_FORGET_RE = re.compile(_LEAD + r"forget\s+" + _N + _END, re.I)
_CONFIRM_RE = re.compile(_LEAD + r"confirm\s+forget\s+" + _N + _END, re.I)
# "my" is required: "forget all the rules, let's improvise" is not about these
_MY_RULES = r"(?:all\s+(?:of\s+)?)?my\s+rules"
_FORGET_ALL_RE = re.compile(_LEAD + r"forget\s+" + _MY_RULES + _END, re.I)
_CONFIRM_ALL_RE = re.compile(_LEAD + r"confirm\s+forget\s+" + _MY_RULES + _END, re.I)
#: what a thumbs-down is answered with (the web client shows it once per page):
#: the label alone teaches nothing any more — a rule the owner states does
FEEDBACK_HINT = ("Noted. If there is something I should always do differently, state it as a rule — say: "
                 "learn this rule: <the rule>")


#: a question mark anywhere (ASCII, full-width, the Greek sign), or a final
#: ";" — how a Greek question ends. Review r3: only a FINAL "?" was looked
#: for, and "…what is y for x = 7?!", "…x = 7? thanks" and a Greek question
#: were stored as rules. A rule states; it is written without one.
_QUESTION_RE = re.compile(r"[?？\u037e]|;[\s\"'”’»)!.]*$")


def _dictated(text: str) -> str:
    """The rule a "learn this rule: …" message dictates, or "" (not that
    command — or a question)."""
    m = _ADD_RE.match(text)
    if not m:
        return ""
    body = text[m.end():].strip()
    if len(body.splitlines()) != 1 or _QUESTION_RE.search(body):
        return ""
    return body


def is_rule_command(text: str) -> bool:
    t = str(text or "")
    return bool(_dictated(t) or _LIST_RE.match(t) or _SHOW_RE.match(t) or _FORGET_RE.match(t)
                or _CONFIRM_RE.match(t) or _FORGET_ALL_RE.match(t) or _CONFIRM_ALL_RE.match(t))


# A short message ABOUT the owner's rules that is none of the commands: the
# model is told nothing changed and which messages do (it has no tool for it).
# Broad on purpose — a note costs a wrong match nothing ("ignore this note").
_HINT_VERB_RE = re.compile(r"\b(?:forget|delete|remove|erase|drop|clear|reset|disable|undo|learn|remember|add|"
                           r"save|make|adopt|dismiss|reject|change|edit|fix|update|replace|show|list|see|new)\b", re.I)
_NUMBER_WORD = r"(?:\d{1,6}|one|two|three|four|five|six|seven|eight|nine|ten)"
_HINT_NOUN_RE = re.compile(r"\brules?\s*(?:#\s*|number\s+|no\.?\s*)?" + _NUMBER_WORD + r"\b"
                           r"|\b(?:my|your|owner|standing|this)\s+rules?\b"
                           r"|\b(?:the|that|last)\s+rule\b|\brule\s*:"
                           r"|\ball\s+(?:the\s+|my\s+)?rules\b|\ba\s+(?:new\s+)?rule\b"
                           r"|\b(?:proposed|candidate)\s+rules?\b"
                           r"|\brules?\s+(?:i|we)\s+(?:gave|set|taught|told)\b", re.I)
HINT_MAX_WORDS = 20
RULE_HINT = ("If the owner means their own standing rules (the ### OWNER RULES block), this message is none of "
             "the commands that change them, so NOTHING was changed. Only these exact messages show or change a "
             "rule, and the system runs them itself: “learn this rule: <the rule>” (the rule on one line after "
             "the colon, with no question mark), “show my rules”, “show rule N”, “forget rule N” (it then asks for “confirm forget "
             "rule N”). To change a rule: “show my rules” gives its number; the owner sends the new rule, on one line "
             "(a list of steps written as one sentence), with “learn this rule: <the rule>”, then “forget rule N” "
             "for the old one (with a full list, forget first). You have no tool that changes a rule, and a "
             "profile entry is not a rule. There are no proposed or candidate rules any more "
             "(the loop that proposed them was removed on 2026-11-05), so “learn rule N” and “dismiss rule N” do "
             "nothing. If the owner asked a question, answer it; if they asked for a change, tell them in one or "
             "two sentences the exact message to send. If the message is about some other kind of rule, ignore "
             "this note.")

# The owner answers a forget preview with "yes": nothing confirms but the exact
# phrase (a bare yes may be answering something else), so the model is told
# that nothing was removed and which message does it.
_AFFIRM_RE = re.compile(r"^\W*(?:y|yes|yep|yup|yeah|ok|okay|sure|absolutely|proceed|i\s+confirm|confirm(?:ed)?|"
                        r"do\s+it|go\s+ahead|please\s+do|(?:forget|delete|remove)\s+it|ναι|εντάξει|οκ)\b", re.I)
AFFIRM_MAX_WORDS = 6
PENDING_HINT = ("The system showed the owner a preview of forgetting {what} and is waiting for the exact message "
                "“{phrase}”. This message is not it, so NOTHING was removed and you cannot remove it. If the "
                "owner is confirming, tell them in one sentence to send exactly: {phrase}. If they are talking "
                "about something else, ignore this note.")


def _waiting_hint(home) -> str:
    """The note for a preview that is STILL waiting: fresh, and what it showed
    is still there. (Review r3: the preview stays in the file after its
    confirmation, so for 15 minutes "ok thanks" told the model that rule 2 —
    already forgotten — was waiting to be confirmed.)"""
    if _path(home) is None:
        return ""
    d = _read(home)[0]
    pend, have = d["pending"], sorted(r["n"] for r in d["rules"])
    if not pend or not _fresh(pend):
        return ""
    if isinstance(pend.get("all"), list):
        if pend["all"] != have:
            return ""
        return PENDING_HINT.format(what="all their rules", phrase="confirm forget all my rules")
    if type(pend.get("n")) is int and pend["n"] in have:
        return PENDING_HINT.format(what=f"rule {pend['n']}", phrase=f"confirm forget rule {pend['n']}")
    return ""


def rule_command_hint(text: str, home=None) -> str:
    t = str(text or "")
    if is_rule_command(t):
        return ""
    words = len(t.split())
    if words <= AFFIRM_MAX_WORDS and _AFFIRM_RE.match(t):
        waiting = _waiting_hint(home)
        if waiting:
            return waiting             # else it may still be about the rules ("ok, delete rule 2": review r3)
    if _ADD_START_RE.match(t):
        return RULE_HINT               # a dictation the command did not take, whatever its length
    if words > HINT_MAX_WORDS:
        return ""
    return RULE_HINT if (_HINT_VERB_RE.search(t) and _HINT_NOUN_RE.search(t)) else ""


# ── the store ───────────────────────────────────────────────────────────────

def _path(home) -> Optional[Path]:
    return Path(home) / STORE_NAME if isinstance(home, (str, Path)) and str(home) else None


def _empty() -> dict:
    return {"next_n": 1, "rules": [], "pending": {}}


def _why_invalid(d) -> str:
    if not isinstance(d, dict):
        return f"holds a JSON {type(d).__name__}, not an object"
    rows = d.get("rules")
    if not isinstance(rows, list):
        return "has no list of rules"
    if len(rows) > RULES_MAX:
        return f"holds more than {RULES_MAX} rules"
    seen = set()
    for r in rows:
        if not (isinstance(r, dict) and type(r.get("n")) is int and 1 <= r["n"] <= 999_999
                and isinstance(r.get("rule"), str) and r["rule"].strip()):
            return "holds a row that is not a rule"
        if r["n"] in seen:
            return f"holds two rules numbered {r['n']}"
        seen.add(r["n"])
    return ""


def _no_repeated_key(pairs) -> dict:
    d = dict(pairs)
    if len(d) != len(pairs):
        raise ValueError("a key is written twice")
    return d


def _read(home) -> Tuple[dict, str]:
    """(the store, why it cannot be used). A file that is absent is an empty
    store; one that is damaged or cannot be read comes back as an EMPTY store
    WITH the reason — and is left on disk as it is (see the module docstring).
    Never raises."""
    p = _path(home)
    if p is None:
        return _empty(), ""
    try:
        # (an editor's byte-order mark is not damage; a key written twice is — the
        # parser would keep the LAST "rules" and the next save would drop the first)
        d = json.loads(p.read_text(encoding="utf-8-sig"), object_pairs_hook=_no_repeated_key)
    except FileNotFoundError:
        return _empty(), ""
    except (ValueError, UnicodeDecodeError):
        return _empty(), "is not readable JSON"
    except RecursionError:
        return _empty(), "is nested too deeply to read"
    except OSError as e:
        return _empty(), f"cannot be read ({type(e).__name__})"
    why = _why_invalid(d)
    if why:
        return _empty(), why
    pend = d.get("pending")
    if not (isinstance(pend, dict) and type(pend.get("at")) in (int, float)):
        d["pending"] = {}                              # a hand-edited or stale preview is no preview
    highest = max((r["n"] for r in d["rules"]), default=0)
    if type(d.get("next_n")) is not int or d["next_n"] <= highest:
        d["next_n"] = highest + 1                      # a number is never reused
    return d, ""


def _fresh(pend: dict) -> bool:
    """A preview confirms for 15 minutes from when it was shown — not before
    (a time in the future or a NaN in a hand-edited file confirmed for ever)."""
    try:
        return 0 <= time.time() - pend["at"] <= FORGET_CONFIRM_S
    except OverflowError:              # an integer too large for a clock
        return False


def damaged(home) -> str:
    """Why the rules file cannot be used, or "" (fine or absent). Read-only:
    what the health check asks."""
    return _read(home)[1]


def load(home) -> dict:
    """The store as the READERS see it: empty when the file is absent — or
    damaged (then no rule is shown; `damaged` says why)."""
    return _read(home)[0]


def _save(home, d: dict) -> None:
    from ..utils.json_store import write_json_atomic
    p = _path(home)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(p, d)


def _refusal(why: str) -> RuleNote:
    return _note(f"Your rules file {why}, so I left it exactly as it is and changed nothing; no rule is shown to "
                 f"me until it reads again. It is `system/{STORE_NAME}` under my data folder — repair it, or "
                 "remove it to start with no rules.")


def rules(home) -> List[dict]:
    """The rules, oldest first."""
    return sorted(load(home)["rules"], key=lambda r: r["n"])


#: an opening quote and the mark that closes it
_QUOTE_PAIRS = {'"': '"', "'": "'", "“": "”", "‘": "’", "«": "»"}
_CONTROL_RE = re.compile(r"<\|[^|>]{0,40}\|>")


def _clean(text: str) -> str:
    """One line, no chat-template control tokens (§4MB), and ONE pair of
    wrapping quotes removed — only a matching pair around the whole text: a
    rule that ends in a quoted phrase keeps its closing mark ("never 'the
    pet'" lost it to a plain strip). The tokens are removed until none is
    left: one pass turned `<|<|x|>im_start|>` into `<|im_start|>`."""
    t = str(text or "")
    for _ in range(50):
        t, n = _CONTROL_RE.subn("", t)
        if not n:
            break
    t = " ".join(t.replace("<|", "< |").split())       # what 50 passes left cannot open a token
    if len(t) >= 2 and _QUOTE_PAIRS.get(t[0]) == t[-1] and t[0] not in t[1:-1] and t[-1] not in t[1:-1]:
        t = t[1:-1].strip()
    return t


def _key(text: str) -> str:
    """What makes two rules the same rule: their words. (A rule with no word
    character — emoji — is compared as written: all such rules had the key "".)"""
    t = str(text or "").lower()
    return " ".join(re.findall(r"\w+", t)) or " ".join(t.split())


def _fmt(r: dict) -> str:
    return f"**Rule {r['n']}** — {_clean(r['rule'])}"       # (a hand-edited rule is cleaned where it is shown too)


# ── what each command does ──────────────────────────────────────────────────

def _add(home, text: str) -> RuleNote:
    rule = _clean(text)
    if re.fullmatch(r"<[^<>]*>", rule):      # §4TD: the placeholder of the command's own hint, pasted as it stands
        return _note("That is the placeholder, not a rule — nothing was added. Put the rule itself after the colon: "
                     "“learn this rule: when I ask for the news, give me five headlines”.")
    if len(rule) < RULE_MIN_CHARS:
        return _note("That is too short to be a rule — nothing was added. Say it in full: "
                     "“learn this rule: when I ask for the news, give me five headlines”.")
    if len(rule) > RULE_MAX_CHARS:
        return _note(f"That rule is {len(rule)} characters; a rule is shown to me on every turn, so the limit "
                     f"is {RULE_MAX_CHARS}. Nothing was added — say it shorter.")
    with _LOCK:
        d, why = _read(home)
        if why:
            return _refusal(why)
        same = next((r for r in d["rules"] if _key(r["rule"]) == _key(rule)), None)
        if same is not None:
            return _note(f"You already have that rule — nothing was added:\n\n- {_fmt(same)}")
        if len(d["rules"]) >= RULES_MAX:
            return _note(f"You have {RULES_MAX} rules, the most I am shown on every turn — nothing was added. "
                         "Forget one first (“show my rules”, then “forget rule N”).")
        row = {"n": d["next_n"], "id": uuid.uuid4().hex, "rule": rule,
               "added_at": datetime.datetime.now().isoformat(timespec="seconds")}
        d["rules"].append(row)
        d["next_n"] += 1
        _save(home, d)
    return _note(f"✓ Rule {row['n']} added. I am shown it on every turn from now on:\n\n> {rule}\n\n"
                 f"Say “forget rule {row['n']}” to remove it.")


def _list(home) -> RuleNote:
    d, why = _read(home)
    if why:
        return _refusal(why)
    rs = sorted(d["rules"], key=lambda r: r["n"])
    if not rs:
        return _note("You have no rules. Say “learn this rule: …” to add one; I am then shown it on every turn.")
    return _note(f"Your rules ({len(rs)} of at most {RULES_MAX}; I am shown them on every turn):\n\n"
                 + "\n".join(f"- {_fmt(r)}" for r in rs)
                 + "\n\nSay “forget rule N” to remove one, “learn this rule: …” to add one.")


def _show(home, n: int) -> RuleNote:
    d, why = _read(home)
    if why:
        return _refusal(why)
    r = next((r for r in d["rules"] if r["n"] == n), None)
    if r is None:
        return _note(f"There is no rule {n}. Say “show my rules” to see the ones you have.")
    return _note(f"{_fmt(r)}\n\nAdded {str(r.get('added_at') or '?')[:10]}. Say “forget rule {n}” to remove it.")


def _preview_forget(home, n: int) -> RuleNote:
    with _LOCK:
        d, why = _read(home)
        if why:
            return _refusal(why)
        r = next((r for r in d["rules"] if r["n"] == n), None)
        if r is None:
            return _note(f"There is no rule {n} — nothing to forget. Say “show my rules” to see the ones you have.")
        d["pending"] = {"n": n, "at": time.time()}
        _save(home, d)
    return _note(f"Forget this rule?\n\n> {_clean(r['rule'])}\n\nSay “confirm forget rule {n}” within 15 minutes to remove it.")


def _confirm_forget(home, n: int) -> RuleNote:
    with _LOCK:
        d, why = _read(home)
        if why:
            return _refusal(why)
        pend = d["pending"]
        r = next((r for r in d["rules"] if r["n"] == n), None)
        if r is None:
            return _note(f"There is no rule {n} — nothing was removed.")
        if pend.get("n") != n or not _fresh(pend):
            return _note(f"Say “forget rule {n}” first, to see what will be removed; then confirm within 15 minutes.")
        d["rules"] = [x for x in d["rules"] if x["n"] != n]
        _save(home, d)             # the preview stays in the file: it names a number that is never reused
    return _note(f"✓ Rule {n} forgotten. I am no longer shown it:\n\n> {_clean(r['rule'])}")


def _preview_forget_all(home) -> RuleNote:
    with _LOCK:
        d, why = _read(home)
        if why:
            return _refusal(why)
        if not d["rules"]:
            return _note("You have no rules — nothing to forget.")
        rs = sorted(d["rules"], key=lambda r: r["n"])
        # the numbers SHOWN: the confirmation removes these and only if they
        # are still all there is (review r1: a rule added after the preview
        # was removed with the two that were shown)
        d["pending"] = {"all": [r["n"] for r in rs], "at": time.time()}
        _save(home, d)
    return _note(f"Forget all {len(rs)} of your rules?\n\n" + "\n".join(f"- {_fmt(r)}" for r in rs)
                 + "\n\nSay “confirm forget all my rules” within 15 minutes to remove them.")


def _confirm_forget_all(home) -> RuleNote:
    with _LOCK:
        d, why = _read(home)
        if why:
            return _refusal(why)
        pend = d["pending"]
        if not d["rules"]:
            return _note("You have no rules — nothing was removed.")
        if pend.get("all") != sorted(r["n"] for r in d["rules"]) or not _fresh(pend):
            return _note("Say “forget all my rules” first, to see what will be removed; then confirm within 15 minutes.")
        n = len(d["rules"])
        d["rules"] = []
        _save(home, d)
    return _note(f"✓ All {n} rules forgotten. I am no longer shown any.")


def rule_command(text: str, home) -> str:
    """The reply to an owner's rule command (a `RuleNote` whose ``banner`` is
    the reply), or "" when the text is not one. Call ONLY for an owner turn on
    a private surface. A file that cannot be WRITTEN is said, by the system:
    the old file is intact (the save is a rename), and a model left to answer
    "learn this rule: …" would not know it failed (review r2)."""
    if _path(home) is None:
        return ""
    try:
        return _dispatch(str(text or "").strip(), home)
    except OSError as e:
        return _note(f"I could not write your rules file ({type(e).__name__}), so nothing was changed. It is "
                     f"`system/{STORE_NAME}` under my data folder.")


def _dispatch(t: str, home) -> str:
    rule = _dictated(t)
    if rule:
        return _add(home, rule)
    if _LIST_RE.match(t):
        return _list(home)
    m = _SHOW_RE.match(t)
    if m:
        return _show(home, int(m.group(1)))
    if _CONFIRM_ALL_RE.match(t):
        return _confirm_forget_all(home)
    if _FORGET_ALL_RE.match(t):
        return _preview_forget_all(home)
    m = _CONFIRM_RE.match(t)
    if m:
        return _confirm_forget(home, int(m.group(1)))
    m = _FORGET_RE.match(t)
    if m:
        return _preview_forget(home, int(m.group(1)))
    return ""


# ── the prompt block ────────────────────────────────────────────────────────

def rules_withheld() -> bool:
    """§4NA: the compliance measurement's control arm — a labelled probe sent
    with `X-Ghost-Arm: no_lessons` is shown no rule."""
    try:
        from ..utils.logging import prompt_arm
        return prompt_arm() == "no_lessons"
    except Exception:  # noqa: BLE001
        return False


#: §4TI-c, under every rules block. Req 0f263dac: "edit this rule, add … 1. … 4. …" (27 words, so no
#: `RULE_HINT`) — the model wrote a guessed profile key and said "Done" while the rule stayed as it was. A matcher
#: deciding which long message is an edit failed three reviews in a row (it hinted "fix my lint rule…" and
#: missed "can you edit my rule…"), so the operator chose this fixed line: no guess, any wording, any language.
RULES_FIXED_LINE = ("These rules change only by the owner's own commands: “learn this rule: <the rule>” adds one; "
                    "“show my rules” gives the numbers (not the order above); “forget rule N”, then “confirm forget "
                    "rule N”, removes one (with ten rules, forget first). You have no tool that changes them, and a "
                    "profile entry is not a rule. Asked to change one, tell the owner which messages to send; never "
                    "say it is changed.")


def rules_block(home) -> str:
    """The ### OWNER RULES block for a turn's state, or "" (no rule). The
    caller decides WHO may see it (never a member, a public reply or an
    isolated delegate)."""
    try:
        rs = rules(home)[:RULES_MAX]
    except Exception:  # noqa: BLE001 — a malformed store never breaks a turn
        return ""
    if not rs or rules_withheld():
        return ""
    return ("### OWNER RULES (set by the owner — follow them whenever they apply)\n"
            + "\n".join(f"- {_clean(r['rule'])[:RULE_MAX_CHARS]}" for r in rs) + "\n" + RULES_FIXED_LINE + "\n\n")


def rules_mentioning(home, target: str) -> List[dict]:
    """The rules that mention ``target`` — the privacy forget lists them and
    points to "forget rule N" (it never removes one itself, §4OR r2)."""
    from ..memory.profile import mentions          # a whole word, case and accents folded; "" names nothing
    return [r for r in rules(home) if mentions(r["rule"], target)]
