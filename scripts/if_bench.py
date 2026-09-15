#!/usr/bin/env python3
"""§4FF — instruction-following bench: SYSTEM_PROMPT vs SYSTEM_PROMPT_COMPILED.

Drives a fixed bank of format-constrained requests through the LIVE agent as
DIAGNOSTIC PROBES (`X-Ghost-Origin: probe` — probes never teach, never enrol
in arms, and are recorded as `task_kind=probe`), once per prompt variant per
repeat, paired per item. The variant is chosen by the probe-only header
`X-Ghost-Prompt-Variant: control|compiled` (ignored on any other origin).

Each item carries a deterministic checker over the final reply text (exact,
regex, word-count, starts-with, JSON, yes/no, language script), so the score
is mechanical. Secondary counts: narration beats ("Let me…"), tool-call XML in
the reply, reply length. Paired outcomes → exact McNemar per pair of arms.

The items are seeded from REAL requests in the trajectory corpus that carried
an explicit constraint (2026-07 → 2026-09; e.g. 5a90ff10 "count the lines …
reply with just the number" failed; 88d1692d "capital of Australia? One
word."), plus synthetic siblings of the same shapes.

§4GJ: the bank is BANDED (easy / tool / deep) because the §4FF bank sat at
ceiling (0.96 vs 0.92, p=1.0) — measured on 1,883 real trajectories, a
constrained request that runs NO tool fails 14% of the time and one that
runs 1-2 tools fails 48%, and the old bank was 21/27 zero-tool items. The
`deep` band states the constraint once, up front, then buries it under
multi-step sandbox work whose ground truth is deterministic, so the checker
pins the format AND the fact. The summary reports per-band pass rates and
per-band McNemar.

usage: if_bench.py [--repeats 2] [--limit N] [--bands easy,tool,deep]
                   [--variants control,compiled] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GHOST_HOME = Path(os.environ.get("GHOST_HOME", "/Users/vasilis/Data/AI/Data"))
AGENT = os.getenv("GHOST_AGENT_URL", "http://127.0.0.1:8000")
#: ⚠ `.strip() or` — not a bare `or`. A `GHOST_API_KEY` set to "" or to a
#: single space is TRUTHY enough for `or` to keep it, and every call in
#: the run then comes back 403. Scored as missing data (§4GJ round 4,
#: correctly), so the run does not fail — it just produces a ledger of
#: 60 errors and a summary with no pairs. Found by doing exactly that.
KEY = (os.getenv("GHOST_API_KEY") or "").strip() or (
    Path.home() / "Data/AI/.ghost_api_key").read_text().strip()

# ── checkers ─────────────────────────────────────────────────────────────
#: Decoration to peel off a reply before matching: markdown emphasis and
#: quote characters, anywhere in the string.
_DECORATION_RE = re.compile(r"[*_`\"'“”‘’]")


def _strip(s: str) -> str:
    """A reply with its decoration peeled off: emphasis and quotes anywhere,
    sentence punctuation at the EDGES.

    ⚠ A FULL STOP IS NOT DECORATION IN THE MIDDLE OF A NUMBER (§4GJ round
    4). The first version deleted every `.` and `!` wherever they appeared,
    which quietly broke both directions of the number family: "98.6" became
    "986" and passed `ck_number_only` as an integer, while `ck_number_equals`
    could never see a decimal point at all — its own `(\\.0+)?` branch was
    unreachable, and measured, `ck_number_equals(12)("12.0")` -> False and
    `ck_number_equals(12.5)("12.5")` -> False. A correctly formatted answer
    therefore scored as an instruction-following VIOLATION and entered the
    paired McNemar as evidence against the variant that produced it.

    Stripping at the edges keeps what the peeling was for: "The answer is
    68." still fails the number checkers, on the prose, not on the stop.
    """
    return _DECORATION_RE.sub("", str(s or "")).strip().strip(".!").strip()

def ck_exact(want):
    return lambda r: _strip(r).lower() == _strip(want).lower()

def ck_regex(pat):
    rx = re.compile(pat, re.I | re.S)
    return lambda r: bool(rx.fullmatch(str(r or "").strip()))

def ck_one_word():
    return lambda r: len(_strip(r).split()) == 1

def ck_max_words(n):
    return lambda r: 0 < len(_strip(r).split()) <= n

def ck_one_sentence():
    return lambda r: len([s for s in re.split(r"(?<=[.!?])\s+", str(r or "").strip()) if s.strip()]) == 1

def ck_number_only():
    return lambda r: bool(re.fullmatch(r"[-+]?\d[\d,]*(\.\d+)?", _strip(r)))

def ck_yes_no():
    return lambda r: _strip(r).lower() in ("yes", "no")

def ck_json_keys(*keys):
    def f(r):
        txt = str(r or "").strip()
        txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt, flags=re.S)
        try:
            obj = json.loads(txt)
        except Exception:
            return False
        return isinstance(obj, dict) and all(k in obj for k in keys)
    return f

def ck_starts_with(prefix):
    return lambda r: str(r or "").lstrip().lower().startswith(prefix.lower())

def ck_greek():
    return lambda r: bool(re.search(r"[α-ωΑ-Ω]", str(r or ""))) and not re.search(r"\b(the|and|is|are)\b", str(r or "").lower())

def ck_bullets(n):
    return lambda r: len([l for l in str(r or "").splitlines() if re.match(r"^\s*(?:[-*•]|\d+[.)])\s+\S", l)]) == n

def ck_no_tool_syntax():
    return lambda r: "<tool_call" not in str(r or "") and "<tool_response" not in str(r or "")

# ── §4GJ checkers: content-anchored, so a DODGE cannot satisfy the format ──
# A cap-only checker passes on a reply that answered nothing ("Done." is ≤5
# words; "0" is a number). Every hard-band checker below therefore pins the
# format AND the one deterministic fact the task computes, so the only way
# to pass is to do the work and obey the shape.

def _content_lines(r):
    return [l for l in str(r or "").splitlines() if l.strip()]

def ck_number_equals(n):
    """Number-only AND the right number (ground truth is deterministic).

    The fraction part is `\\.\\d+`, not `\\.0+`: an item whose ground truth
    is 12 must accept "12.0", and one whose truth is 12.5 must accept
    "12.5". The old pattern could match neither, because `_strip` had
    already eaten the point — see its docstring. The VALUE comparison, not
    the pattern, is what rejects a wrong number."""
    def f(r):
        t = _strip(r).replace(",", "")
        return bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", t)) and abs(float(t) - n) < 1e-9
    return f

def ck_max_lines(n):
    ls = lambda r: _content_lines(r)
    return lambda r: 0 < len(ls(r)) <= n

def ck_lines_exactly(n):
    return lambda r: len(_content_lines(r)) == n

def ck_one_line_containing(*required):
    """The line_cap family (#2 most violated in production, absent from the
    old bank): exactly one content line, carrying every required token."""
    def f(r):
        lines = _content_lines(r)
        if len(lines) != 1:
            return False
        low = lines[0].lower()
        return all(str(w).lower() in low for w in required)
    return f

def ck_json_values(**pairs):
    """Strict JSON, exactly these keys, exactly these values."""
    def f(r):
        txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(r or "").strip(), flags=re.S)
        try:
            obj = json.loads(txt)
        except Exception:
            return False
        if not isinstance(obj, dict) or set(obj) != set(pairs):
            return False
        for k, want in pairs.items():
            got = obj[k]
            if isinstance(want, (int, float)) and isinstance(got, str):
                try:
                    got = float(got)
                except ValueError:
                    return False
            if isinstance(want, (int, float)) and isinstance(got, (int, float)):
                if abs(float(got) - float(want)) > 1e-9:
                    return False
            elif str(got).strip().lower() != str(want).strip().lower():
                return False
        return True
    return f

def ck_max_words_with(n, *required):
    """Word cap AND every required token present — a one-word dodge fails."""
    def f(r):
        body = _strip(r)
        words = body.split()
        if not (0 < len(words) <= n):
            return False
        low = body.lower()
        return all(str(w).lower() in low for w in required)
    return f

def ck_one_sentence_with(*required):
    def f(r):
        body = str(r or "").strip()
        sents = [x for x in re.split(r"(?<=[.!?])\s+", body) if x.strip()]
        if len(sents) != 1:
            return False
        low = body.lower()
        return all(str(w).lower() in low for w in required)
    return f

def ck_all(*fns):
    """Every checker must hold — used where a cap alone is dodge-able."""
    return lambda r: all(f(r) for f in fns)

def ck_min_words(n):
    """Floor under a cap: a bare "None." must not satisfy a one-line item."""
    return lambda r: len(_strip(r).split()) >= n

def ck_exact_word(want):
    """`ck_exact` after multi-step work — kept distinct so the band split
    reads honestly in the report."""
    return ck_exact(want)

# ── §4GX checkers: the constraint that is CHEAP to state and hard to obey ──
# §4GU measured the §4GJ bank at ceiling in every band (deep 1.000 in BOTH
# arms, overall 0.97 vs 0.97, p=1.0). Exactly one item discriminated —
# `words-1`, "Explain recursion in at most 12 words", failed in both arms in
# both repeats — and its shape is the finding: a budget the model must
# COMPRESS to meet, not a format it reliably emits. A JSON object or a
# yes/no is a shape the model produces correctly by habit; eight words about
# a race condition is a shape it has to give something up for. The checkers
# below are that family: tight caps, exact counts, and bans on the words the
# natural answer reaches for first.

def ck_max_sentences(n):
    def f(r):
        body = str(r or "").strip()
        sents = [x for x in re.split(r"(?<=[.!?])\s+", body) if x.strip()]
        return 0 < len(sents) <= n
    return f

def ck_exact_words(n):
    """EXACTLY n words — not a ceiling. A cap can be met by saying less;
    an exact count cannot be met by saying less OR more."""
    return lambda r: len(_strip(r).split()) == n

def ck_forbidden(*words, floor=4):
    """None of `words` (nor their inflections) appears — with a word FLOOR,
    because the cheapest way to obey a ban is to answer nothing."""
    pats = [re.compile(rf"\b{re.escape(w)}\w*", re.I) for w in words]
    def f(r):
        body = _strip(r)
        if len(body.split()) < floor:
            return False
        return not any(p.search(body) for p in pats)
    return f

def ck_lines_each_max_words(nlines, nwords, floor=2):
    """Exactly `nlines` content lines, every one of them within `nwords` —
    a compound budget, which is where a dropped constraint shows."""
    def f(r):
        lines = _content_lines(r)
        if len(lines) != nlines:
            return False
        for l in lines:
            w = _strip(re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s*", "", l)).split()
            if not (floor <= len(w) <= nwords):
                return False
        return True
    return f

def ck_json_field_max_words(key, n, floor=1):
    """Strict JSON AND a word budget INSIDE one field: two constraints that
    have to survive the same reply."""
    def f(r):
        txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(r or "").strip(), flags=re.S)
        try:
            obj = json.loads(txt)
        except Exception:
            return False
        if not isinstance(obj, dict) or key not in obj:
            return False
        w = _strip(str(obj[key])).split()
        return floor <= len(w) <= n
    return f

# ── the bank ─────────────────────────────────────────────────────────────
# Items are (id, request, checker, needs_tools, band).
#
# BANDS, and why they exist (§4GJ, measured 2026-09-13 on 1,883 real
# user_request trajectories with `core.turn_state_check`):
#
#     tool calls | constrained requests | violated | rate
#     ---------- | -------------------- | -------- | ----
#         0      |         198          |    27    |  14%
#        1-2     |          67          |    32    |  48%
#        3-6     |           8          |     2    |  25%
#        7+      |           0          |     0    |   —
#
# The failure rate MORE THAN TRIPLES the moment a tool runs. The §4FF bank
# was 21 of 27 zero-tool items — i.e. concentrated in the 14% band — which
# is why it scored 0.96/0.92 and could not discriminate (p=1.0). The bands
# below are the fix: `easy` is the old bank kept as a regression anchor,
# `tool` is the 48% band, `deep` buries the constraint under multi-step
# work (the recorded requirement: "constraint buried in a multi-tool task").
#
# Family weights also follow the record rather than the old bank. In
# production the constrained requests are strict_json 189, exact 31,
# line_cap 24, sentence_cap 14, word_cap 8, number_only 7; the most-violated
# are strict_json 49/189 (26%) and line_cap 9/24 (38%). line_cap was absent
# from the old bank entirely and is now represented in both hard bands.
#
# Every `deep` checker is content-anchored on a DETERMINISTIC ground truth
# the task itself computes, so the format cannot be satisfied by a dodge
# ("Done." is within any word cap; "0" is a number). The deep items are
# self-seeding — they create the data they measure and delete it again — so
# the bank does not depend on workspace state that drifts between runs.
ITEMS = [
    # ── easy band: zero-tool format compliance (the §4FF bank, anchor) ────
    ("num-1", "What is 17 times 4? Answer with just the number.", ck_number_only(), False, "easy"),
    ("num-2", "How many days are in a leap year? Reply with just the number, nothing else.", ck_number_only(), False, "easy"),
    ("word-1", "What is the capital of Australia? One word.", ck_one_word(), False, "easy"),
    ("word-2", "Which planet is closest to the Sun? Reply with a single word.", ck_one_word(), False, "easy"),
    ("exact-1", "Reply with exactly the word READY and nothing else.", ck_exact("READY"), False, "easy"),
    ("exact-2", "Reply with exactly: PONG2. Nothing else.", ck_exact("PONG2"), False, "easy"),
    ("exact-3", "Do not investigate or use any tools, just reply with exactly: NOTED", ck_exact("NOTED"), False, "easy"),
    ("sent-1", "In one sentence: what is a hash table?", ck_one_sentence(), False, "easy"),
    ("sent-2", "Reply with one short sentence: what is 2+2?", ck_one_sentence(), False, "easy"),
    ("yn-1", "Is Python dynamically typed? Answer yes or no only.", ck_yes_no(), False, "easy"),
    ("yn-2", "Is 91 a prime number? Yes or no, nothing else.", ck_yes_no(), False, "easy"),
    ("json-1", "Give me the boiling point of water at sea level in Celsius and Fahrenheit as JSON with keys celsius and fahrenheit. Output only the JSON.", ck_json_keys("celsius", "fahrenheit"), False, "easy"),
    ("json-2", "Return a JSON object with keys name and year for the first person on the Moon. JSON only, no prose.", ck_json_keys("name", "year"), False, "easy"),
    ("start-1", "Answer starting with the words 'Short answer:' — why is the sky blue? Keep it under 40 words.", ck_starts_with("Short answer:"), False, "easy"),
    ("words-1", "Explain recursion in at most 12 words.", ck_max_words(12), False, "easy"),
    ("words-2", "Describe the colour of the sea in five words or fewer.", ck_max_words(5), False, "easy"),
    ("greek-1", "Απάντησε μόνο στα ελληνικά: τι είναι ο ήλιος; Μία πρόταση.", ck_greek(), False, "easy"),
    ("bul-1", "List exactly three benefits of unit tests as three bullet points, nothing else.", ck_bullets(3), False, "easy"),
    ("bul-2", "Name four primary colours as a numbered list of four items and nothing more.", ck_bullets(4), False, "easy"),
    ("code-1", "Reply with only a Python one-liner that prints hello, no explanation, no code fence.", ck_regex(r"print\((['\"])hello\1\)"), False, "easy"),
    ("line-1", "Name the three primary additive colours on ONE line, nothing else.", ck_all(ck_max_lines(1), ck_min_words(3)), False, "easy"),

    # ── tool band: one or two calls, then the constraint (the 48% band) ───
    ("tool-num-1", "List the top-level entries of the workspace with file_system, then reply with just the number of entries the listing reported.", ck_number_only(), True, "tool"),
    ("tool-num-2", "quick check: list how many files are in your workspace root with file_system. reply with just the number.", ck_number_only(), True, "tool"),
    ("tool-exact-1", "Run exactly this in the sandbox: echo if-probe-ok — then reply with exactly the word: DONE", ck_exact("DONE"), True, "tool"),
    ("tool-json-1", "Run `python3 -c \"print(6*7)\"` in the sandbox and reply with JSON only: {\"result\": <the number>}.", ck_json_values(result=42), True, "tool"),
    ("tool-json-2", "Run `python3 -c \"print(2**10)\"` in the sandbox, then reply with strict JSON and nothing else: {\"value\": <the number it printed>}.", ck_json_values(value=1024), True, "tool"),
    ("tool-sent-1", "Check the system health with system_utility and reply in exactly one sentence saying whether it is healthy.", ck_one_sentence(), True, "tool"),
    ("tool-yn-1", "Look at the workspace listing with file_system and answer yes or no only: is there a file named README.md at the top level?", ck_yes_no(), True, "tool"),
    ("tool-line-1", "List your acquired skills with manage_skills. Just the names, one line.", ck_all(ck_max_lines(1), ck_min_words(3)), True, "tool"),

    # ── deep band: the constraint is stated ONCE, first, then buried under
    #    multi-step sandbox work; ground truth is deterministic ────────────
    ("deep-num-1",
     "Reply with just the number and nothing else. In the sandbox, do all of this first: "
     "create /workspace/ifb_a.txt containing exactly 3 lines, create /workspace/ifb_b.txt "
     "containing exactly 4 lines, create /workspace/ifb_c.txt containing exactly 5 lines, "
     "then count the lines in each of the three files separately, then delete all three "
     "files. Then give me the total number of lines you counted across the three files.",
     ck_number_equals(12), True, "deep"),
    ("deep-num-2",
     "Reply with just the number and nothing else. In the sandbox: write the three lines "
     "a, b and c into /workspace/ifb_n.txt, then count its lines with wc, then delete the "
     "file, then multiply the line count you measured by 5 and give me that result.",
     ck_number_equals(15), True, "deep"),
    ("deep-json-1",
     "Reply with strict JSON and nothing else. In the sandbox: run "
     "`python3 -c \"print(6*7)\"`, then create /workspace/ifb_j.txt containing the single "
     "word ready, then read that file back, then delete it. Only then answer, with an "
     "object carrying exactly two keys: product (the number python printed) and file_text "
     "(the word you read back).",
     ck_json_values(product=42, file_text="ready"), True, "deep"),
    ("deep-line-1",
     "Answer on exactly one line, nothing else. In the sandbox: create the three files "
     "ifb_x1.txt, ifb_x2.txt and ifb_x3.txt in /workspace (any contents), list the "
     "workspace root to confirm all three exist, then delete all three. Then confirm on "
     "your single line, naming all three file names you created.",
     ck_one_line_containing("ifb_x1", "ifb_x2", "ifb_x3"), True, "deep"),
    ("deep-exact-1",
     "When you are completely finished, reply with exactly the word FINISHED and nothing "
     "else. The work, in the sandbox, is: run `echo step1`, then run `echo step2`, then "
     "run `echo step3`, then create /workspace/ifb_e.txt containing the text ok, then read "
     "it back, then delete it. Do not reply until all six steps are done.",
     ck_exact_word("FINISHED"), True, "deep"),
    ("deep-words-1",
     "Answer in at most six words. In the sandbox: run "
     "`python3 -c \"print(sum(range(1,11)))\"`, then write that number into "
     "/workspace/ifb_w.txt, then read the file back to verify it, then delete it. Then "
     "tell me the number.",
     ck_max_words_with(6, "55"), True, "deep"),
    ("deep-sent-1",
     "Answer in exactly one sentence. In the sandbox: create /workspace/ifb_s.txt "
     "containing the word probe, read it back, then delete it, then list the workspace "
     "root to confirm it is gone. Then tell me in that single sentence what happened to "
     "ifb_s.txt.",
     ck_one_sentence_with("ifb_s.txt", "delet"), True, "deep"),

    # ── hard band: tight budgets, exact counts and bans (§4GX) ───────────
    #    The §4GJ bands measured at ceiling; these are the shapes the model
    #    must give something up to satisfy. Calibrated live before use —
    #    an item that passes 100% or 0% in both arms discriminates nothing.
    #
    #    CALIBRATED 2026-09-14 (30 pairs, 0 errors, ledger 20260914T183729Z):
    #    control 0.767 vs compiled 0.600, McNemar b=7 c=2 p=0.18 — the band
    #    is NOT saturated (§4GU's whole bank produced ZERO discordant pairs;
    #    this one produced NINE) and the direction favours control.
    #      discriminating : h-words-1/2/3, h-exact-2, h-comp-1, h-json-1,
    #                       h-lines-1  — every one of them ZERO-TOOL.
    #      both-arm ceiling: h-ban-1, and ALL FIVE tool/deep items. That
    #                       agrees with §4GU, where the tool and deep BANDS
    #                       read 1.000/1.000 over 72 pairs: for THESE two
    #                       prompts the tool-regime axis does not
    #                       discriminate, twice measured. They are kept as
    #                       a regression anchor, not as evidence.
    #      both-arm floor  : h-exact-1 (exact word counts defeat the model
    #                       in both arms — a capability limit, not a
    #                       compliance one) and h-ban-2 v1 (a bank bug,
    #                       reworded above).
    #
    #    ⚠ The way to power this comparison is MORE DISTINCT ITEMS, never
    #    more repeats of the same ones — [[bench-unit-is-the-distinct-prompt]]:
    #    587 rows that were 264 requests turned p=0.0003 into p=0.88.
    #
    #    §4GY REPLICATION (2026-09-14, ledger 20260914T192742Z): the ten
    #    `h-*-4..7 / -2 / -3` items below were authored and pinned BEFORE
    #    being run, and the §4GX direction did NOT hold — 20 pairs, control
    #    0.650, compiled 0.650, b=2 c=2, p=1.0. Pooled (caveated: §4GX's
    #    items were the ones being selected on) b=9 c=4 over 50 pairs,
    #    p=0.27. So the standing answer is "no measurable compliance
    #    difference", from an instrument that CAN lose: 13 discordant pairs
    #    in 50 here, against ZERO in 72 for the whole §4GJ bank. Do not
    #    quote §4GU's null and this one as the same kind of evidence.
    ("h-words-1", "Explain what a race condition is, in at most 8 words.",
     ck_all(ck_max_words(8), ck_min_words(3)), False, "hard"),
    ("h-words-2", "In at most 10 words, say why TCP uses a three-way handshake.",
     ck_all(ck_max_words(10), ck_min_words(4)), False, "hard"),
    ("h-words-3", "What does a database index do? At most 7 words.",
     ck_all(ck_max_words(7), ck_min_words(3)), False, "hard"),
    ("h-exact-1", "Write one sentence about the sea that is exactly nine words long.",
     ck_exact_words(9), False, "hard"),
    ("h-exact-2", "Answer in exactly five words: what is a queue?",
     ck_exact_words(5), False, "hard"),
    ("h-ban-1", "Explain what DNS does in one sentence, without using the words name, domain or address.",
     ck_all(ck_one_sentence(), ck_forbidden("name", "domain", "address")), False, "hard"),
    # ⚠ The ban list must not contain the SUBJECT the question names, and it
    #   must say what the checker enforces. v1 banned "compile" while asking
    #   "what does a compiler do" — both arms opened "A compiler translates…"
    #   and failed on the subject noun, which is a bank bug, not a violation:
    #   an item whose only failure mode is disputable measures nothing.
    ("h-ban-2", "In at most two sentences, describe what a Python interpreter does, without using the words run, execute or program (or any word starting with those).",
     ck_all(ck_max_sentences(2), ck_forbidden("run", "execute", "program")), False, "hard"),
    ("h-comp-1", "In at most 12 words and without using the word data, say what a database index does.",
     ck_all(ck_max_words(12), ck_forbidden("data", floor=4)), False, "hard"),
    ("h-json-1", "What is a mutex? Reply with strict JSON only, no prose, no fence: an object with one key answer whose value is at most 5 words.",
     ck_json_field_max_words("answer", 5, floor=2), False, "hard"),
    ("h-lines-1", "Give exactly three lines, each at most four words: three uses of a hash table. Nothing else.",
     ck_lines_each_max_words(3, 4), False, "hard"),

    # ── §4GY: ten more DISTINCT items in the family that discriminated ──
    #    Not more repeats of the seven — [[bench-unit-is-the-distinct-prompt]].
    #    Every one is zero-tool with a budget the answer must be compressed
    #    to meet, and every checker carries a floor or a content anchor.
    ("h-words-4", "In at most 9 words, say what a cache is for.",
     ck_all(ck_max_words(9), ck_min_words(4)), False, "hard"),
    ("h-words-5", "Why does merge sort beat bubble sort? At most 8 words.",
     ck_all(ck_max_words(8), ck_min_words(4)), False, "hard"),
    ("h-words-6", "Describe what a firewall does in at most 6 words.",
     ck_all(ck_max_words(6), ck_min_words(3)), False, "hard"),
    ("h-words-7", "In at most 11 words, explain why passwords are salted before hashing.",
     ck_all(ck_max_words(11), ck_min_words(5)), False, "hard"),
    ("h-exact-3", "Answer in exactly four words: what is a compiler for?",
     ck_exact_words(4), False, "hard"),
    ("h-exact-4", "Give me exactly six words describing a thunderstorm.",
     ck_exact_words(6), False, "hard"),
    ("h-comp-2", "In at most 10 words and without using the word memory, say what a stack overflow is.",
     ck_all(ck_max_words(10), ck_forbidden("memory", floor=4)), False, "hard"),
    ("h-comp-3", "In one sentence of at most 14 words, and without using the word network, say what a router does.",
     ck_all(ck_one_sentence(), ck_max_words(14), ck_forbidden("network", floor=5)), False, "hard"),
    ("h-json-2", "What is a deadlock? Reply with strict JSON only, no prose, no fence: an object with one key answer whose value is at most 6 words.",
     ck_json_field_max_words("answer", 6, floor=2), False, "hard"),
    ("h-lines-2", "Give exactly four lines, each at most three words: four things a version control system stores. Nothing else.",
     ck_lines_each_max_words(4, 3), False, "hard"),

    # tool band, same family: the 48% band with a budget on top
    ("h-tool-words-1",
     "Run `python3 -c \"print(sum(range(1,101)))\"` in the sandbox, then tell me the result in at most four words.",
     ck_max_words_with(4, "5050"), True, "hard"),
    ("h-tool-ban-1",
     "List the workspace root with file_system, then answer in one sentence that does not contain the word file or files: what is in there?",
     ck_all(ck_one_sentence(), ck_forbidden("file")), True, "hard"),
    ("h-tool-exact-1",
     "Run `python3 -c \"print(3**5)\"` in the sandbox, then reply with exactly two words: the word result followed by the number it printed.",
     ck_all(ck_exact_words(2), ck_max_words_with(2, "243")), True, "hard"),

    # deep band, same family: the budget stated once, then buried
    ("h-deep-words-1",
     "Answer in at most four words. In the sandbox: create /workspace/ifb_h1.txt containing "
     "the numbers 1 to 4 on four separate lines, count its lines with wc, then delete the "
     "file, then multiply the line count you measured by 3. Then tell me that result.",
     ck_max_words_with(4, "12"), True, "hard"),
    ("h-deep-ban-1",
     "Answer in one sentence that does not use the words file, delete or create. In the "
     "sandbox: write the word probe into /workspace/ifb_h2.txt, read it back, delete it, "
     "then list the workspace root to confirm it is gone. Then tell me what you did.",
     ck_all(ck_one_sentence(), ck_forbidden("file", "delet", "creat")), True, "hard"),
]

#: Bands in report order.
BANDS = ("easy", "tool", "deep", "hard")


def select_items(bands: str = "", no_tools: bool = False, item_ids: str = "",
                 offset: int = 0, limit: int = 0, items=None):
    """The bench's item selection, lifted out of ``main`` as a SEAM (§4GJ
    battery): the band filter was unpinnable while it lived inside an
    argparse-driven ``main``, so a mutant that dropped it survived. The
    filter order is unchanged — no-tools, then bands, then explicit ids
    (which override offset/limit).
    """
    items = list(ITEMS if items is None else items)
    items = [it for it in items if not (no_tools and it[3])]
    if bands:
        want_b = {b.strip() for b in bands.split(",") if b.strip()}
        unknown = want_b - set(BANDS)
        if unknown:
            sys.exit(f"unknown band(s): {sorted(unknown)}; known: {list(BANDS)}")
        items = [it for it in items if it[4] in want_b]
    if item_ids:
        want = {x.strip() for x in item_ids.split(",") if x.strip()}
        items = [it for it in items if it[0] in want]
    else:
        items = items[offset:]
        if limit:
            items = items[:limit]
    return items

def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n))


def _chat(text: str, variant: str, rid: str, timeout=400.0) -> dict:
    body = {"messages": [{"role": "user", "content": text}], "stream": False}
    req = urllib.request.Request(
        f"{AGENT}/api/chat", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "X-Ghost-Key": KEY,
                 "X-Ghost-Origin": "probe", "X-Ghost-Prompt-Variant": variant,
                 "X-Request-ID": rid})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    reply = ((d.get("choices") or [{}])[0].get("message") or {}).get("content")
    if reply is None:
        reply = d.get("response") or d.get("content") or ""
    return {"reply": str(reply or ""), "seconds": round(time.time() - t0, 1),
            "usage": d.get("usage") or {}}


NARRATION_RE = re.compile(r"^\s*(let me|now (?:let|i'll|i will)|i'll (?:now|start)|first, i)", re.I | re.M)


def score_call(result: dict, check) -> dict:
    """Score ONE call: `{"passed", "narration", "tool_syntax_leak"}`.

    ⚠ `passed` IS None WHEN THE CALL FAILED, AND None IS NOT False (§4GJ
    round 4). The transport error was being scored: an exception from
    `_chat` became `{"reply": "", ..., "error": ...}`, and `bool(check(""))`
    is False for every checker in this file, so a connection reset, a 500,
    or — with a 400 s timeout over multi-tool `deep` items, the likeliest of
    the three — a TIMEOUT entered the paired McNemar as evidence that the
    prompt variant had violated the instruction. The `error` key was written
    to the ledger and read by nothing.

    A failed call is MISSING DATA. It is still recorded (a run that silently
    drops its failures lies about its own N), but it is excluded from the
    rates, from the bands and from the pairing, and counted separately. This
    lives outside `main` as a SEAM so it can be driven without a live agent
    — the same reason `select_items` was lifted out in §4GJ.
    """
    if result.get("error"):
        return {"passed": None, "narration": None, "tool_syntax_leak": None}
    reply = str(result.get("reply") or "")
    return {"passed": bool(check(reply)),
            "narration": 1 if NARRATION_RE.search(reply) else 0,
            "tool_syntax_leak": 0 if ck_no_tool_syntax()(reply) else 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0, help="skip the first N items (chunked runs)")
    ap.add_argument("--items", default="", help="comma-separated item ids to run (overrides offset/limit)")
    ap.add_argument("--variants", default="control,compiled")
    ap.add_argument("--no-tools", action="store_true", help="skip tool-using items")
    ap.add_argument("--bands", default="", help="comma-separated bands to run (easy,tool,deep)")
    ap.add_argument("--out", default=str(GHOST_HOME / "system" / "eval" / "if_bench"))
    args = ap.parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    items = select_items(bands=args.bands, no_tools=args.no_tools,
                         item_ids=args.items, offset=args.offset, limit=args.limit)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    ledger = out / f"{stamp}.jsonl"
    ok = {v: 0 for v in variants}
    scored = {v: 0 for v in variants}     # calls that produced a REPLY to judge
    errors = {v: 0 for v in variants}     # calls that failed: missing data
    band_ok = {v: {b: 0 for b in BANDS} for v in variants}
    band_scored = {v: {b: 0 for b in BANDS} for v in variants}
    band_err = {b: 0 for b in BANDS}      # failed calls, by band
    band_n = {b: 0 for b in BANDS}
    band_pair = {b: {"b": 0, "c": 0} for b in BANDS}
    narr = {v: 0 for v in variants}
    leak = {v: 0 for v in variants}
    pair = {"b": 0, "c": 0}   # for the first two variants: b = v0 ok & v1 not; c = v1 ok & v0 not
    total = 0
    unpaired = 0              # item-reps lost because a call failed
    t_start = time.time()
    with ledger.open("w") as f:
        for rep in range(args.repeats):
            for iid, text, check, needs_tools, band in items:
                res = {}
                for v in variants:
                    rid = f"ifb-{stamp}-{iid}-{v}-r{rep}"
                    try:
                        r = _chat(text, v, rid)
                    except Exception as e:  # noqa: BLE001
                        r = {"reply": "", "seconds": None, "usage": {}, "error": str(e)}
                    sc = score_call(r, check)
                    passed = sc["passed"]
                    res[v] = passed
                    if passed is None:
                        errors[v] += 1
                        band_err[band] += 1
                    else:
                        scored[v] += 1
                        ok[v] += passed
                        narr[v] += sc["narration"]; leak[v] += sc["tool_syntax_leak"]
                        band_ok[v][band] += passed
                        band_scored[v][band] += 1
                    # `run` is the STAMP, not the rep: `rep` restarts at 0 on
                    # every invocation, so without this two ledgers from two
                    # runs of the same bank collide in the combiner's
                    # (rep, item) key and half the data is dropped silently
                    # (§4GJ round 4 — see if_bench_combine.py).
                    rec = {"run": stamp, "rep": rep, "item": iid, "band": band,
                           "variant": v, "passed": passed,
                           "narration": sc["narration"], "tool_syntax_leak": sc["tool_syntax_leak"],
                           "seconds": r.get("seconds"), "reply": r["reply"][:400],
                           "error": r.get("error"), "usage": r.get("usage")}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                    verdict = "ERR " if passed is None else ("PASS" if passed else "FAIL")
                    print(f"[r{rep} {iid:<12} {band:<4} {v:<8}] {verdict} "
                          f"{(r.get('seconds') or 0):5.1f}s  "
                          f"{(r.get('error') or r['reply'])[:70]!r}", flush=True)
                if len(variants) >= 2:
                    a, b_ = variants[0], variants[1]
                    # A pair needs BOTH variants SCORED. One failed call does
                    # not make the other variant's reply a win.
                    if res.get(a) is None or res.get(b_) is None:
                        unpaired += 1
                        continue
                    total += 1
                    band_n[band] += 1
                    if res[a] and not res[b_]:
                        pair["b"] += 1; band_pair[band]["b"] += 1
                    if res[b_] and not res[a]:
                        pair["c"] += 1; band_pair[band]["c"] += 1
                else:
                    total += 1
                    band_n[band] += 1
    summary = {"items": len(items), "repeats": args.repeats, "pairs": total,
               # The denominator is what was SCORED, per variant: a failed
               # call is missing data, so it must not deflate a pass rate.
               "scored": scored, "errors": errors, "unpaired": unpaired,
               "pass_rate": {v: ok[v] / scored[v] for v in variants if scored[v]},
               "narration": narr, "tool_syntax_leak": leak,
               "mcnemar": {"b_first_only": pair["b"], "c_second_only": pair["c"],
                           "p": mcnemar_exact(pair["b"], pair["c"])} if len(variants) >= 2 else None,
               # A band is reported when it produced ANYTHING — including a
               # band where every call failed. Keying this on `band_n` alone
               # made a wholly timed-out band disappear from the report,
               # which is the same silent drop one level up.
               "by_band": {b: {"pairs": band_n[b], "errors": band_err[b],
                               "pass_rate": {v: (band_ok[v][b] / band_scored[v][b])
                                             for v in variants if band_scored[v][b]},
                               "mcnemar": {"b_first_only": band_pair[b]["b"],
                                           "c_second_only": band_pair[b]["c"],
                                           "p": mcnemar_exact(band_pair[b]["b"], band_pair[b]["c"])}
                               if len(variants) >= 2 else None}
                           for b in BANDS
                           if band_n[b] or band_err[b]
                           or any(band_scored[v][b] for v in variants)},
               "seconds": round(time.time() - t_start), "ledger": str(ledger)}
    (out / f"{stamp}.summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    sys.exit(main())
