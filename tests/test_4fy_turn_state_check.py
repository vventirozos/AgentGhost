"""§4FY — the state-aware, refute-only judge tier (`core/turn_state_check`).

The property under review: **a reply that violates a mechanically checkable
constraint of its OWN request, or asserts an answer over a turn whose every
web retrieval came back empty, is refuted; anything else produces no verdict
at all.** Trusted: the current request text, the tool results as the tools
wrote them, the delivered reply. Untrusted: the reply's content (the thing
under audit), stored project constraints (never read here), the LLM
verifier's verdict.

Every pin names the world it fails in. Fixtures are the corpus's real shapes
(the chess app's "STRICT JSON on a single line and NOTHING else", "Reply
with exactly the word: PONG. Nothing else.", "Just the names, one line.",
"reply with just the number") and the §4FY review's adversarial inputs —
the requests the first parser turned into constraints the user never meant.
"""
import ast
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ghost_agent.core import turn_state_check as T
from ghost_agent.core.turn_state_check import mechanical_constraints, refute_turn_state
from ghost_agent.core.reply_smoothing import strip_system_notes
from ghost_agent.core.verifier import VerifyResult, VerifyVerdict

CHESS = ("You are playing a live chess game as BLACK against Vasilis and coaching him.\n"
         "Position FEN: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1\n"
         "Choose YOUR move from the legal list. Reply with STRICT JSON on a single line "
         'and NOTHING else: {"move": "<uci>", "comment": "<2-3 sentences>"}')
MOVE = '{"move": "c7c5", "comment": "Sicilian Defense — c5 strikes back at the center."}'
PROSE_THEN_MOVE = ("Looking at the position, White just played e4. The Sicilian is the most "
                   "combative reply.\n\n" + MOVE)
AWAY = "**While you were away** — I advanced 1 task(s).\n\n---\n\n"
ACTIVITY = "**Background activity while you were away:**\n  - [dream] REM cycle ran\n\n---\n\n"
NUDGE = ("\n\n---\n💡 This looks like ongoing work (3 sandbox writes). Want me to promote "
         "it to a tracked project (“Chess”)? Just say the word.")
SELF_CHECK = "\n\n---\n**Self-check (principle):** RAM at 92% is normal."
LABEL_ASK = ("\n\n---\n*This was one of the shakier answers I've given this week and nothing "
             "checked it. Was it right? A 👍/👎 here teaches me more than a hundred I score myself.*")


def _kinds(req):
    return [c.kind for c in mechanical_constraints(req)]


def _rules(req, reply, tools=None):
    return [r for r, _ in refute_turn_state(request=req, reply=reply, tools_run=tools)]


# ── the design constraint §4EP bought ──────────────────────────────────────

def test_a_satisfied_constraint_produces_NOTHING_not_a_confirmation():
    """⚠ THE WHOLE POINT. The world it fails in: someone makes a satisfied
    rule return a CONFIRMED because it feels like free coverage — §4EP
    measured that exact change shrinking the Brier delta 2–16×."""
    assert refute_turn_state(request=CHESS, reply=MOVE) == []
    assert refute_turn_state(request="Reply with exactly: pong", reply="pong") == []
    assert refute_turn_state(request="What is 17 times 4? Answer with just the number.", reply="68") == []
    assert refute_turn_state(request="hello there", reply="Hi Vasilis!") == []
    # and the module offers no way to express a pass: every issue is a
    # (rule, complaint) pair and no exported name says "confirmed"
    assert set(T.__all__) == {"refute_turn_state", "mechanical_constraints", "Constraint"}
    assert not any("confirm" in n.lower() for n in dir(T))


# ── a constraint is an unconditional imperative about the reply ────────────

@pytest.mark.parametrize("req", [
    # review #8: JSON as the DELIVERABLE or the question, not the reply
    "The API should respond with valid JSON. Write the Flask handler for me.",
    'is this valid JSON? {"a": 1,}  explain why or why not',
    "Write me a script that outputs strict JSON to stdout. Show me the code with comments.",
    "generate a raw JSON fixture file at /tmp/x.json with 50 users and tell me what you did",
    "the raw JSON from the API is huge; summarise what's in it",
    "give me the JSON schema for the config file; there's nothing else in the repo that reads it",
    "Only JSON files are affected. Explain the bug.",
    "convert this YAML to json only where the keys are lowercase",
    "what does 'strictly JSON' mean in the spec?",
    "STRICT JSON is what the schema validator wants; my file fails. Diagnose",
    "Export the rows as valid JSON, one object per line.",
    # review #9: "the number of …" is not "just the number"
    "only the number of rows changed, not the schema. what happened?",
    "Answer with the number of CPUs and what each core is doing",
    "the number only matters if it's above 3, right?",
    "the number only shows on hover — how do i make it permanent",
    # review #10: per-item caps are not whole-reply caps
    "Reply in one line per file, for each of the 12 files",
    "Give me a one-line summary for EACH of the 40 commits",
    "no more than 5 words per bullet, at least 10 bullets",
    "describe each of the 3 options in 20 words max",
    "In one sentence per item, summarise these 3 papers.",
    # review #11: two-part requests
    "Reply in one line, then in the next paragraph explain your reasoning in detail",
    "tell me in one sentence what it does, then list every option with a paragraph each",
    "Answer in one word or two, then explain",
    # review #12: "one line" describing the deliverable or a bug
    "Rewrite the function so the return fits in one line",
    "Bug: prints everything in one line. Fix it and show me the full file.",
    "Put the JS in one line, minified, and give me the full HTML page with comments",
    "My editor shows the whole config on one line — can you reformat it and explain each key?",
    "Explain how to write that in one line in a shell",
    "I typed the whole thing in one word processor and it broke",
    "Write a one-line HTML file called fr_probe.html that says HELLO.",
    "Using the sandbox, run a Python one-liner that prints 7 factorial.",
    # review #13: quoted speech and field lists
    "The user said 'reply exactly: foo' and the bot said bar. What went wrong?",
    "Did they reply exactly: 'no'? Search the mail thread and tell me the whole story",
    "Answer with exactly: the name, the price and the URL for each of the top 5",
    "reply exactly: <the sha of HEAD>",
    # review #2: conditional constraints refute the honest other branch
    "1. run the tests\n2. if green, reply exactly: GREEN\n3. otherwise paste the failure",
    "if the build passed, reply with just the number of tests; if not, paste the errors",
    "if you know, answer in one word; if you don't, say so and explain what you'd need",
    "answer in one line if the port is open, otherwise explain what is blocking it",
    "When the file is written, reply with exactly: OK and nothing else",
    # misc phrasings that are not caps
    "Summarise it in 20 words or so.",
    "single line item on the invoice",
    "reply exactly as before",
])
def test_a_request_that_does_not_constrain_the_reply_yields_no_constraint(req):
    """The world it fails in: the parser reads a word, not a clause — the
    first version constrained every one of these (§4FY review, adversarial
    lens) and would have refuted the reply the user actually wanted."""
    assert _kinds(req) == [], (req, mechanical_constraints(req))


@pytest.mark.parametrize("req,kinds", [
    (CHESS, ["strict_json"]),
    ("IMPORTANT: PLAIN TEXT JSON ONLY, no tools.", ["strict_json"]),
    ("Return the result as JSON and nothing else.", ["strict_json"]),
    ("Output valid JSON.", ["strict_json"]),
    ("Respond with strictly JSON.", ["strict_json"]),
    ("Reply with pure JSON, no other text.", ["strict_json"]),
    ("Answer with raw JSON only.", ["strict_json"]),
    ("Reply with exactly the word: PONG. Nothing else.", ["exact"]),
    ("reply with exactly: OK and nothing else", ["exact"]),
    ("Is the port open? Reply with exactly: yes or no", ["exact"]),
    ("count the lines and reply with just the number.", ["number_only"]),
    ("What is 17 times 4? Answer with just the number.", ["number_only"]),
    ("How many files are there? Number only.", ["number_only"]),
    ("What is the capital of Australia? Answer in one word.", ["word_cap"]),
    ("Describe the colour of the sea in five words or fewer.", ["word_cap"]),
    ("Say hello in exactly two words.", ["word_cap"]),
    ("Summarise it in no more than 20 words.", ["word_cap"]),
    ("Summarise it, 20 words max.", ["word_cap"]),
    ("List your acquired skills. Just the names, one line.", ["line_cap"]),
    ("Confirm in one line.", ["line_cap"]),
    ("tell me in one line whether any events appear", ["line_cap"]),
    ("Give me a one-line summary.", ["line_cap"]),
    ("paste back ONLY the two lines that mention prompts.gepa, at most 2 lines", ["line_cap"]),
    ("tell me in one sentence whether the table renders correctly.", ["sentence_cap"]),
])
def test_the_phrasings_users_actually_use_still_constrain(req, kinds):
    """The corpus phrasings (189 / 58 / 21 / 11 / 10 / 4 requests) and the
    reviewer's alternations. The world it fails in: the exclusions above
    are widened until the real population loses its check."""
    assert _kinds(req) == kinds


def test_conditional_else_is_not_the_exclusive_else():
    """"NOTHING else" is exclusivity; a clause-level "else" is a condition.
    The world it fails in: the skip regex matches "else" inside "nothing
    else" and the chess population (189 requests) loses its constraint —
    the first cut of the rewrite did exactly that."""
    assert _kinds("Reply with STRICT JSON and NOTHING else.") == ["strict_json"]
    assert _kinds("Reply with JSON, or else explain in prose.") == []
    assert _kinds("Say OK. Anything more is noise: reply with exactly: OK, nothing more") == ["exact"]


def test_the_exact_phrase_is_a_set_without_its_trailing_instruction():
    """Review C1/#3. The world it fails in: "PONG. Nothing else" or "OK and
    nothing else" or "yes or no" becomes the literal."""
    val = lambda req: [c.value for c in mechanical_constraints(req) if c.kind == "exact"]
    assert val("Reply with exactly the word: PONG. Nothing else.") == [("PONG",)]
    assert val("Reply with exactly: PONG2. Nothing else.") == [("PONG2",)]
    assert val("reply with exactly: OK and nothing else") == [("OK",)]
    assert val("reply with exactly: DONE, nothing more") == [("DONE",)]
    assert val("reply exactly: ACK — no explanation") == [("ACK",)]
    assert val("reply exactly: ACK (nothing else)") == [("ACK",)]
    assert val("Is the port open? Reply with exactly: yes or no") == [("yes", "no")]
    assert val("Reply with exactly: yes / no") == [("yes", "no")]
    assert val("Health probe: reply exactly: PROBE-OK-3117") == [("PROBE-OK-3117",)]
    assert val("Reply with exactly: streaming works") == [("streaming works",)]


@pytest.mark.parametrize("req,reply", [
    ("Is the port open? Reply with exactly: yes or no", "yes"),
    ("Is the port open? Reply with exactly: yes or no", "No."),
    ("reply with exactly: OK and nothing else", "OK"),
    ("Reply with exactly the word: PONG. Nothing else.", "PONG"),
    ("Reply with exactly the word: PONG. Nothing else.", "**PONG**"),
    ("Reply with exactly the word: PONG. Nothing else.", "pong."),
    ("Reply with exactly the word: PONG. Nothing else.", ACTIVITY + "PONG"),
    ("Reply with exactly the word: PONG. Nothing else.", "Sure — PONG"),
])
def test_the_phrase_with_light_decoration_is_not_refuted(req, reply):
    assert _rules(req, reply) == []


def test_the_exact_tolerance_boundary_is_twelve_characters():
    """Identity pin (review C8): +12 passes, +13 refutes."""
    assert _rules("Reply with exactly: PONG", "x" * 11 + " PONG") == []          # 12 extra
    assert _rules("Reply with exactly: PONG", "x" * 12 + " PONG") == ["exact"]   # 13 extra


@pytest.mark.parametrize("reply", [
    "I will not reply with that word.",
    "PONG — and here is a paragraph about why ping-pong is the right metaphor for streaming.",
])
def test_a_reply_that_is_not_the_phrase_is_refuted(reply):
    assert _rules("Reply with exactly the word: PONG. Nothing else.", reply) == ["exact"]


# ── strict JSON ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("reply", [
    MOVE,
    "```json\n" + MOVE + "\n```",          # one fence is tolerated
    "~~~json\n" + MOVE + "\n~~~",          # so is a tilde fence (review #18)
    "````\n" + MOVE + "\n````",
    AWAY + MOVE,                            # the away banner is finalize's, not the model's
    ACTIVITY + MOVE,
    MOVE + NUDGE,                           # the promotion nudge is finalize's too (5ef33e14)
    MOVE + SELF_CHECK,                      # and the principle gate's note (review R1)
    MOVE + LABEL_ASK,                       # and the §4ER label request
    MOVE + "\n\n---\n**⚠ Unverified:** the move was not executed.",
    "42", "true", '"c7c5"',                 # a JSON scalar IS a JSON document
])
def test_a_json_answer_is_not_refuted(reply):
    assert _rules(CHESS, reply) == []


@pytest.mark.parametrize("reply", [
    PROSE_THEN_MOVE,                                      # the corpus's commonest violation
    "c7c5",                                               # a bare move (f2de9cf2)
    "I see the issue — only JSON, nothing else. Here it is:\n\n" + MOVE,
    MOVE + "\n\nCoaching tip: castle early.",
])
def test_prose_around_or_instead_of_the_json_is_refuted(reply):
    """The world it fails in: the check strips everything outside the first
    `{…}` and judges only that — every prose-then-JSON turn passes."""
    assert _rules(CHESS, reply) == ["strict_json"]


def test_the_json_rule_shadows_every_shape_cap():
    """"STRICT JSON on a single line": a fenced object is three lines and
    still the answer, and a JSON reply has no word or sentence count. The
    world it fails in: the caps run beside the JSON rule and a fenced JSON
    reply is refuted for its fence (5 corpus turns)."""
    assert _kinds(CHESS) == ["strict_json"]
    assert _kinds("Reply in one sentence with valid JSON only, in 10 words or fewer.") == ["strict_json"]
    assert _rules(CHESS, "```json\n" + MOVE + "\n```") == []


# ── just the number ────────────────────────────────────────────────────────

@pytest.mark.parametrize("reply", ["68", "**68**", "68.", "26 files.", "3,083 pages", "42%", "68. Done.",
                                   "Zero.", "None.", "Twenty-one"])
def test_a_number_with_a_unit_or_bold_or_a_number_word_is_not_refuted(reply):
    assert _rules("count the lines and reply with just the number.", reply) == []


@pytest.mark.parametrize("req", [
    "count the lines and reply with just the number.",
    "What is 17 times 4? Answer with just the number.",
    "How many files are there? Number only.",
])
@pytest.mark.parametrize("reply,why", [
    ("There are quite a few lines in there. I counted 26,014 lines, which is a lot "
     "for one journal. Let me know if you want a breakdown by section.", "words of explanation"),
    ("It depends on what you count as a line, honestly.", "no number"),
])
def test_an_explanation_instead_of_the_number_is_refuted(req, reply, why):
    issues = refute_turn_state(request=req, reply=reply)
    assert [r for r, _ in issues] == ["number_only"] and why in issues[0][1]


def test_number_only_two_sentence_boundary():
    """Review C2: the two-sentence clause is its own rule — ≤8 words but two
    sentences refutes; the same words in one sentence do not."""
    assert _rules("reply with just the number.", "About 26. Roughly speaking.") == ["number_only"]
    assert _rules("reply with just the number.", "About 26 roughly speaking") == []


# ── an honest inability report gets no constraint verdict ──────────────────

@pytest.mark.parametrize("req", [
    "count the lines in /Users/x/PROJECT_JOURNAL.md and reply with just the number.",
    "Reply with exactly: DONE",
    "What is in the file? Answer in one word.",
    "Confirm in one line.",
    "Reply with STRICT JSON and NOTHING else.",
])
@pytest.mark.parametrize("reply", [
    # the two corpus refutes (48/49), verbatim shape
    "The file `/Users/x/PROJECT_JOURNAL.md` is on your host machine, outside the sandbox — I can't "
    "access absolute macOS paths directly. If you copy it into the sandbox, I'll count the lines.",
    "I can't access that file — permission denied on /tmp/count.txt.",
    "Unable to read the file: the sandbox has no such path.",
    "Δεν μπορώ να διαβάσω το αρχείο — δεν έχω πρόσβαση εκτός sandbox.",
    "Sorry — the lookup failed and nothing came back.",       # short + says so
])
def test_an_honest_inability_report_is_not_refuted_for_its_shape(req, reply):
    """The 2026-07-31 honest-failure rule (§4FY review, consumer lens): the
    first version wrote "checked and WRONG" (0.0) on the honest reply and,
    on a tool turn, told the model to "answer it now with NO new tool
    calls" — fabrication pressure. The world it fails in: the exemption is
    dropped from any one rule."""
    assert _rules(req, reply) == []


def test_the_inability_must_lead_or_the_reply_must_be_short():
    """A chess analysis that says "I can't recapture" in its SECOND sentence
    is not an inability report, and a long reply that merely mentions an
    error is not one either — the exemption must not swallow the JSON
    population it was measured on."""
    prose = ("Looking at the position, White just played Bxf5, capturing my bishop. I can't "
             "recapture immediately, so I need a plan.\n\n" + MOVE)
    assert _rules(CHESS, prose) == ["strict_json"]
    # corpus 2d3fabeb: "can't" in the FIRST sentence, about the game, not the task
    prose2 = ("Looking at the position, White's bishop on d7 is deep in my territory but I can't "
              "directly capture it with the available moves. The best plan is to centralize.\n\n" + MOVE)
    assert _rules(CHESS, prose2) == ["strict_json"]
    long_ack = ("Red, yellow, blue — " + "and here is a long explanation about colour theory " * 8
                + "with one failed attempt at brevity.")
    assert _rules("Answer in one word.", long_ack) == ["word_cap"]


# ── word / line / sentence caps ────────────────────────────────────────────

@pytest.mark.parametrize("req,reply,expect", [
    ("What is the capital of Australia? Answer in one word.", "Canberra", []),
    ("What is the capital of Australia? Answer in one word.", "**Canberra.**", []),
    ("What is the capital of Australia? Answer in one word.",
     "Canberra is correct — it's the capital of Australia. Sydney is the largest city.", ["word_cap"]),
    ("Describe the colour of the sea in five words or fewer.", "Deep restless blue.", []),
    ("Describe the colour of the sea in five words or fewer.", "No sea is visible in the image.", ["word_cap"]),
    ("Describe the colour of the sea in five words or fewer.", "No sea visible in the image.", ["word_cap"]),  # cap+1
    ("Describe the colour of the sea in five words or fewer.", "Deep, restless, endless, cold blue.", []),   # exactly the cap
    ("Say hello in exactly two words.", "Hello Vasilis", []),
    ("Summarise it in no more than 20 words.", " ".join(["w"] * 21), ["word_cap"]),
    ("Summarise it, 20 words max.", " ".join(["w"] * 21), ["word_cap"]),
    ("Summarise it, 20 words tops.", " ".join(["w"] * 20), []),
])
def test_word_caps(req, reply, expect):
    assert _rules(req, reply) == expect


@pytest.mark.parametrize("reply", ["3.12.4", "Open-source.", "2026-09-08", "Twenty-one.", "turn_state_check.py",
                                   "v.ventirozos@evolmonkey.com", "https://example.org/a/b?c=1", "O'Brien"])
def test_one_token_answers_are_one_word(reply):
    """Review #4: `\\w+` split "3.12.4" into three words and refuted it. A
    word is a whitespace token carrying a letter or digit."""
    assert _rules("which python? Answer in one word.", reply) == []
    assert T._words(reply) == [T._plain(reply)]


def test_words_is_an_identity_on_a_fixed_sentence():
    assert T._words("Well-tested, production-ready, don't merge yet — v2.1!") == \
        ["Well-tested,", "production-ready,", "don't", "merge", "yet", "v2.1!"]


@pytest.mark.parametrize("req,reply,expect", [
    ("List your acquired skills. Just the names, one line.", "generate_password, news_headlines", []),
    ("List your acquired skills. Just the names, one line.", "- generate_password\n- news_headlines", ["line_cap"]),
    ("Confirm in one line.", "Task marked DONE — probe.txt verified.", []),
    ("Confirm in one line.", "Task marked DONE.\n\n\n\nprobe.txt verified.", ["line_cap"]),   # blank lines do not count (C1)
    ("Confirm in one line.", "Task marked DONE.\n\n\n\n", []),
    ("tell me in one line whether any events appear", "No — none in the last 2 hours.\n\nWant the full list?", ["line_cap"]),
    ("Give me a one-line summary.", "One.\nTwo.", ["line_cap"]),
    ("Give me a one-line summary of the log.", "```\nall green\n```", []),          # fence lines are not lines (review #14)
    ("at most 2 lines please", "```text\na\nb\n```", []),
    ("paste back ONLY the two lines that mention prompts.gepa, at most 2 lines", "a\nb", []),
])
def test_line_caps(req, reply, expect):
    assert _rules(req, reply) == expect


@pytest.mark.parametrize("reply,expect", [
    ("Yes, the table renders correctly.", []),
    ("Yes, it renders. All elements are visible.", []),                              # cap+1: slack
    ("Yes, it renders. All elements are visible. No files were edited.", ["sentence_cap"]),   # cap+2 (C11)
    ("Config files, i.e. yaml, toml and env, etc. plus the lockfile.", []),            # abbreviations (review #5)
    ("Approx. 3 % faster on p50 vs. the baseline, within noise on p99.", []),
    ("Do 1. install, 2. configure, 3. run.", []),
    ("Dr. J. R. R. Tolkien Jr. wrote it at approx. 3 p.m. on Tue.", []),
    ("It renders — see fig. 2 and e.g. the plunger lane.", []),
    # an abbreviation followed by a CAPITALISED word is where the list is
    # load-bearing (review battery N5 survived without this)
    ("Dr. Smith and Mr. Jones vs. Real Madrid agree: it renders, per Fig. Two.", []),
])
def test_sentence_cap_has_one_sentence_of_slack_and_knows_abbreviations(reply, expect):
    assert _rules("tell me in one sentence whether the table renders correctly.", reply) == expect


def test_sentences_is_an_identity_on_fixed_text():
    assert T._sentences("One. Two! Three? Four… and e.g. not five. Six.") == 5
    assert T._sentences("About 26. Roughly speaking.") == 2
    assert T._sentences("Dr. Smith met Mr. Jones vs. Real Madrid. Fine.") == 2
    assert T._sentences("i.e. one sentence with vs. and etc. inside") == 1


# ── every WEB retrieval empty, the reply asserts anyway ────────────────────

BROWSER_BLANK = ("--- BROWSER RESULT ---\nSTATUS: OK\nOP: extract_text\nURL: http://127.0.0.1:8100\n"
                 "TITLE: WebOS\nLENGTH: 7\n--- TEXT ---\nLoading")
BROWSER_404 = ("--- BROWSER RESULT ---\nSTATUS: OK\nOP: navigate\nURL: http://127.0.0.1:8102/static/index.html\n"
               "HTTP_STATUS: 404\nTITLE: 404 Not Found")
BROWSER_429 = ("--- BROWSER RESULT ---\nSTATUS: OK\nOP: navigate\nURL: https://www.google.com/sorry/index\n"
               "HTTP_STATUS: 429\nTITLE: Sorry")
BROWSER_ERR = "--- BROWSER RESULT ---\nSTATUS: ERROR\nRefused navigation: refused internal host (SSRF guard)."
BROWSER_200 = ("--- BROWSER RESULT ---\nSTATUS: OK\nOP: navigate\nURL: http://127.0.0.1:8102/\n"
               "HTTP_STATUS: 200\nTITLE: Ghost's Study — chess with a coach")
SEARCH_EMPTY = "ERROR: No search results found. The internet might be blocking your request."
SEARCH_FULL = "### 1. Title\nbody text here\n[Source: https://example.org/a]\n"
RECALL_WEAK = ("SYSTEM: Found 3 memories (best match: LOW — these are probably UNRELATED "
               "to the query; do not present them as facts about it).\n\nSOURCE: pg\nCONTENT: notes")
ASSERTIVE = ("WebOS is up and running on port 8100. The desktop is loading with the Start "
             "button visible and all four apps are on the desktop: Browser, Wallpaper, "
             "Minesweeper and Arkanoid. You can open it at http://127.0.0.1:8100 now.")
HONEST = ("The page came back blank — I couldn't extract any text from the desktop, so I "
          "cannot confirm the apps rendered. The service itself reports RUNNING on 8100; "
          "open it in your browser and tell me what you see, and I will take it from there.")
EXEC_OK = "--- EXECUTION RESULT ---\nEXIT CODE: 0\nSTDOUT/STDERR:\n" + "Canberra is the capital " * 5


def test_an_assertive_reply_over_blank_retrievals_is_refuted():
    """A browser ERROR / 4xx / 5xx, or a search that returned nothing, and
    an assertive reply that never says so."""
    for bad in (BROWSER_404, BROWSER_429, BROWSER_ERR):
        tools = [{"name": "browser", "content": bad, "arguments": {"operation": "navigate"}}]
        assert _rules("start the service for webos", ASSERTIVE, tools) == ["empty_evidence"], bad
    assert _rules("look", ASSERTIVE, [{"name": "web_search", "content": SEARCH_EMPTY}]) == ["empty_evidence"]


def test_a_short_page_is_evidence_and_a_later_success_outweighs_an_earlier_error():
    """Review of the replay's own fires: the gate's 40-char page floor is a
    STEER threshold — a 7-char ("Loading") or 16-char ("Start Soundscape")
    page the reply described was refuted; and a 404 on the wrong path
    followed by a 200 on the right one (e2361219), or a 429 followed by a
    200 + extract (e5369464, HUMAN-approved), were refuted because the
    successful load was dropped rather than counted."""
    short = [{"name": "browser", "content": BROWSER_BLANK, "arguments": {"operation": "extract_text"}}]
    assert _rules("start the service for webos", ASSERTIVE, short) == []
    seq = [{"name": "browser", "content": BROWSER_404, "arguments": {"operation": "navigate"}},
           {"name": "browser", "content": BROWSER_200, "arguments": {"operation": "navigate"}}]
    assert _rules("start the chess server", ASSERTIVE, seq) == []
    seq2 = [{"name": "browser", "content": BROWSER_429, "arguments": {"operation": "navigate"}},
            {"name": "browser", "content": BROWSER_200, "arguments": {"operation": "navigate"}},
            {"name": "browser", "content": BROWSER_BLANK, "arguments": {"operation": "extract_text"}}]
    assert _rules("how long to drive home", ASSERTIVE, seq2) == []


@pytest.mark.parametrize("tools,reply", [
    ([{"name": "browser", "content": BROWSER_404}], HONEST),                        # says so
    ([{"name": "web_search", "content": SEARCH_EMPTY}], "Nothing came back, sorry."),  # short + says so
    ([{"name": "browser", "content": BROWSER_404}], "Port 8100 is up."),              # short, no ack
    ([{"name": "web_search", "content": SEARCH_FULL}], ASSERTIVE),                  # had evidence
    ([{"name": "web_search", "content": SEARCH_EMPTY},
      {"name": "browser", "content": BROWSER_BLANK.replace("LENGTH: 7", "LENGTH: 5321") + "words " * 200}], ASSERTIVE),
    ([], ASSERTIVE),                                                                # nothing consulted
    ([{"name": "darkweb_search", "content": SEARCH_EMPTY}],
     "Dark web search for 'gun' returned zero results — onion engines are flaky. " + ASSERTIVE),
    # review #15: a memory miss is not an absent source; absence never refutes
    ([{"name": "recall", "content": RECALL_WEAK}], ASSERTIVE),
    ([{"name": "knowledge_base", "content": ""}], ASSERTIVE),
    # a successful command IS evidence (search failed → curl → answer)
    ([{"name": "web_search", "content": SEARCH_EMPTY}, {"name": "execute", "content": EXEC_OK}], ASSERTIVE),
    # a browser click / screenshot is not a page read
    ([{"name": "browser", "content": "--- BROWSER RESULT ---\nSTATUS: ERROR\nOP: click\nelement not found",
       "arguments": {"operation": "click"}}], ASSERTIVE),
    # a three-character curl verification is evidence (replay 9fa6dc99)
    ([{"name": "browser", "content": BROWSER_ERR, "arguments": {"operation": "navigate"}},
      {"name": "execute", "content": "--- COMMAND RESULT ---\nEXIT CODE: 0\nSTDOUT/STDERR:\n200"}], ASSERTIVE),
    # Greek acknowledgements
    ([{"name": "web_search", "content": SEARCH_EMPTY}],
     "Δεν υπάρχουν διαθέσιμα αποτελέσματα για αυτό το θέμα αυτή τη στιγμή. " + ASSERTIVE),
])
def test_an_honest_short_or_evidenced_reply_is_not_refuted(tools, reply):
    assert _rules("look it up", reply, tools) == []


def test_a_failed_command_is_not_an_absent_source():
    """The world it fails in: `execute` counts as a retrieval — the first
    version refuted three chess moves whose only call was a failed helper
    script, and a correct answer to "print what 0/0 does, I know it
    errors" (20ff0ec8)."""
    tools = [{"name": "execute", "content": "Traceback (most recent call last):\nZeroDivisionError: division by zero\nEXIT CODE: 1",
              "arguments": {"command": "python3 -c 'print(0/0)'"}}]
    reply = ("In Python `0/0` raises a ZeroDivisionError rather than returning NaN the way "
             "JavaScript does — the interpreter throws before any value exists, which is what "
             "the traceback above shows, exactly as you expected it to.")
    assert _rules("print what 0/0 does, i know this produces an error", reply, tools) == []


def test_the_web_tool_set_is_exactly_the_web_tools():
    """Identity pin (review C3): the set, not its complement."""
    assert T._WEB_TOOLS == {"web_search", "darkweb_search", "deep_research", "darkweb_research",
                            "fact_check", "news_headlines", "browser"}


def test_the_assertive_threshold_boundary_is_thirty_words():
    """Identity pin (review C9): 29 words pass, 30 fire."""
    tools = [{"name": "browser", "content": BROWSER_404}]
    assert T._ASSERTIVE_MIN_WORDS == 30
    assert _rules("look", " ".join(["fact"] * 29), tools) == []
    assert _rules("look", " ".join(["fact"] * 30), tools) == ["empty_evidence"]


@pytest.mark.parametrize("ack", ["couldn't", "cannot", "can't", "unable", "no results", "zero results", "0 hits",
                                 "not found", "nothing came back", "unavailable", "failed", "blocked", "timed out",
                                 "error", "refused", "empty", "δεν βρήκα", "δεν υπάρχει", "καμία", "αποτυχία", "σφάλμα"])
def test_every_acknowledgement_token_exempts(ack):
    """Review C10: each token of the acknowledgement lexicon is load-bearing."""
    tools = [{"name": "web_search", "content": SEARCH_EMPTY}]
    assert _rules("look", ASSERTIVE + " Note: " + ack + ".", tools) == []


def test_the_two_tool_row_shapes_tell_one_story():
    """R5: the live loop hands `{"name","content"}`, the trajectory store
    `{"name","result","error"}`; one verdict for one input — and the stored
    `error` string (a copy of the result on 24 substantive corpus rows) is
    not read (review #16)."""
    live = [{"name": "browser", "content": BROWSER_404}]
    stored = [{"name": "browser", "result": BROWSER_404, "error": "", "arguments": {}}]
    assert (_rules("x", ASSERTIVE, live) == _rules("x", ASSERTIVE, stored) == ["empty_evidence"])
    stored_full_with_error = [{"name": "web_search", "result": SEARCH_FULL, "error": SEARCH_FULL}]
    assert _rules("x", ASSERTIVE, stored_full_with_error) == []
    mixed = [{"name": "web_search", "content": None, "result": SEARCH_FULL}]
    assert _rules("x", ASSERTIVE, mixed) == []


# ── the reply as the model wrote it ────────────────────────────────────────

@pytest.mark.parametrize("head", ["**While you were away**", "**Background activity while you were away:**"])
def test_both_banner_heads_are_peeled(head):
    """Review C15: 66 corpus replies carry the first head, fixtures used only
    the second."""
    assert _rules(CHESS, head + " — stuff\n\n---\n\n" + MOVE) == []


def test_strip_system_notes_removes_every_finalize_appended_note():
    """R1: the class is every `f"{final_ai_content}\\n\\n---\\n<head>"` append
    in agent.py plus the §4ER label ask — enumerated from the SOURCE so a
    new appender fails here until its head is in the stripper. The world
    it fails in: a new note is appended and every consumer of the model's
    reply (the LLM judge included) reads it as the model's words."""
    import ghost_agent.core.agent as agent_mod
    src = Path(agent_mod.__file__).read_text()
    appends = re.findall(r'\{final_ai_content\}\\n\\n---\\n', src)
    label_ask = src.count('return ("\\n\\n---\\n*This was one of the shakier answers')
    note_lit = src.count('"\\n\\n---\\n**⚠ Unverified:**')
    heads = ["**⚠ Unverified:** x", "**Plan check:** x", "**Things I'm not certain about:** x",
             "**Assumptions I made:** x", "**Self-check (principle):** x",
             "💡 This looks like ongoing work (x). Want me to promote it?",
             "*This was one of the shakier answers I've given this week. Was it right?*",
             "**Verifier note:** x"]
    # 4 f-string appends (plan note, risk summary, self-check, nudge) + the
    # Unverified literal + the label ask = 6 appenders; add a 7th and this
    # count AND the list above must grow together
    assert (len(appends), note_lit, label_ask) == (4, 1, 1), (len(appends), note_lit, label_ask)
    for h in heads:
        assert strip_system_notes(MOVE + "\n\n---\n" + h) == MOVE, h
    assert strip_system_notes(MOVE) == MOVE
    # a model-authored 💡 paragraph followed by real content survives (fail-open)
    body = MOVE + "\n\n---\n💡 This looks like ongoing work to me.\n\nBut here is the plan."
    assert strip_system_notes(body) == body


# ── the deleted rules stay deleted ─────────────────────────────────────────

def test_the_two_measured_and_deleted_rules_do_not_fire():
    """Pin the deletion. `count_vs_list` (116 fires, 66 on passed turns;
    then 9 with 6 false) and `start_with` (only the originating request of
    the §4FD project, 2 passed / 0 failed). The world it fails in: either
    grows back."""
    assert refute_turn_state(request="give me the news", reply="Here are today's top 10 headlines:\n\n1. a\n2. b\n3. c") == []
    assert refute_turn_state(request="Explore it. Start with: What it means to BE ghost.",
                             reply="You want me to actually do it. I am Ghost.") == []
    assert _kinds("Start with: X. Here are 10 items:") == []


def test_the_checker_never_raises_and_is_linear():
    """Total, and linear (review #17: the first version took 23 s on 50k
    dots, 2.7 s on 33k backticks, 2.7 s on 100k newlines)."""
    assert refute_turn_state(request=None, reply=None) == []
    assert refute_turn_state(request=123, reply=["x"], tools_run=[None, "x", {"name": 3}]) == []
    for req, rep in [("Reply with exactly: PONG", ". " * 50000 + "a"), (CHESS, "```" * 33000),
                     ("hello", "\n" * 100000), ("Answer in one word.", "word " * 50000),
                     (CHESS, "{" * 50000), ("in one sentence: x", "a. " * 50000)]:
        t0 = time.perf_counter()
        refute_turn_state(request=req, reply=rep)
        assert time.perf_counter() - t0 < 0.5, (req[:20], time.perf_counter() - t0)
    t0 = time.perf_counter()
    strip_system_notes("\n" * 100000)
    assert time.perf_counter() - t0 < 0.2


# ── at the REAL site ───────────────────────────────────────────────────────

VERIFY_NAMES = ("verify_response", "verify", "verify_turn", "run", "verify_claim",
                "verify_code_output", "verify_visual")
DUMP = ("Process finished successfully.\n\n### Final Output:\n```text\n"
        "--- EXECUTION RESULT ---\nEXIT CODE: 0\nSTDOUT/STDERR:\nok\n```")


@pytest.fixture
def agent_at_site(mock_context, tmp_path):
    from ghost_agent.core.agent import GhostAgent
    from ghost_agent.distill.collector import TrajectoryCollector
    agent = GhostAgent(mock_context)
    agent.context.sandbox_dir = str(tmp_path)
    agent.context.current_project_id = None
    agent.context.project_store = None
    agent.context.trajectory_collector = TrajectoryCollector(tmp_path / "trajectories")
    verifier = MagicMock()
    verifier.llm_client = MagicMock()
    calls = {"n": 0}

    async def confirm(*a, **k):
        calls["n"] += 1
        return VerifyResult(verdict=VerifyVerdict.CONFIRMED, confidence=0.9, reasoning="ok", issues=[])
    for name in VERIFY_NAMES:
        setattr(verifier, name, confirm)
    agent.context.verifier = verifier
    agent._is_strict_trivial_chat = lambda lc: False
    agent._verify_depth_for_turn = lambda *a, **k: False

    def sidecar():
        return [json.loads(l) for f in (tmp_path / "verdicts").glob("*.jsonl")
                for l in f.read_text().splitlines() if l.strip()]

    return agent, verifier, calls, sidecar


async def _run(agent, reply, rid, *, request=CHESS, tools=None):
    return await agent._compute_verifier_verdict(
        tools_run_this_turn=tools or [], messages=[{"role": "user", "content": request}],
        final_ai_content=reply, last_user_content=request, lc=request.lower(),
        req_id=rid, trajectory_id=rid)


@pytest.mark.asyncio
async def test_tool_free_turn_is_refuted_mechanically_and_recorded(agent_at_site):
    """The tool-free path: the LLM judge is never asked (46% of turns land
    here), the verdict carries override='turn-state', confidence 0.9, and
    reaches the sidecar the override report reads. The world it fails in:
    the state check is computed but the tool-free branch returns before it."""
    agent, verifier, calls, sidecar = agent_at_site
    res, _ = await _run(agent, PROSE_THEN_MOVE, "s1")
    assert res is not None and res.verdict == VerifyVerdict.REFUTED and res.confidence == 0.9
    assert res.issues and res.issues[0].startswith("strict_json: ")
    assert calls["n"] == 0
    rows = sidecar()
    assert rows and rows[-1]["trajectory_id"] == "s1"
    assert str(rows[-1]["verdict"]).lower().startswith("refut")
    assert rows[-1].get("override") == "turn-state" and rows[-1].get("route") == "turn-state"
    # the control leg: a compliant reply gets NO mechanical verdict; with no
    # evidence tool it stays the placeholder (None), never a CONFIRMED
    res_ok, _ = await _run(agent, MOVE, "s1b")
    assert res_ok is None and calls["n"] == 0


@pytest.mark.asyncio
async def test_three_rules_survive_the_issue_cap(agent_at_site):
    """Review C7: `issues[:3]` — three rules firing at once keep three."""
    agent, verifier, calls, sidecar = agent_at_site
    req = "Reply with exactly: PONG. Also reply with just the number."
    tools = [{"name": "web_search", "content": SEARCH_EMPTY}]
    res, _ = await _run(agent, ASSERTIVE, "c3", request=req, tools=tools)
    assert res.verdict == VerifyVerdict.REFUTED
    assert [i.split(":")[0] for i in res.issues] == ["exact", "number_only", "empty_evidence"]


@pytest.mark.asyncio
async def test_tool_turn_override_replaces_a_confirm_and_merges_into_a_refute(agent_at_site):
    """Tool turns: the LLM judge still runs, and the mechanical refute is
    applied as an OVERRIDE ahead of the ground-truth checks — it replaces a
    CONFIRMED (4 corpus turns the judge confirmed with prose around the
    JSON) and merges into a standing REFUTED keeping the grounded issue
    first. The world it fails in: `_state` is computed and never applied."""
    agent, verifier, calls, sidecar = agent_at_site
    tools = [{"name": "execute", "arguments": {"command": "python3 x.py"},
              "content": "EXIT CODE: 0\nout", "result": "EXIT CODE: 0\nout"}]
    res, _ = await _run(agent, PROSE_THEN_MOVE, "t1", tools=tools)
    assert calls["n"] == 1
    assert res.verdict == VerifyVerdict.REFUTED and res.override == "turn-state"
    assert res.issues[0].startswith("strict_json: ")
    assert sidecar()[-1].get("override") == "turn-state"

    async def refute(*a, **k):
        return VerifyResult(verdict=VerifyVerdict.REFUTED, confidence=0.9, reasoning="r",
                            issues=["the page throws a SyntaxError on load", "x", "y"])
    for name in VERIFY_NAMES:
        setattr(verifier, name, refute)
    res_m, _ = await _run(agent, PROSE_THEN_MOVE, "t2", tools=tools)
    assert res_m.issues[0] == "the page throws a SyntaxError on load"
    assert any(i.startswith("strict_json: ") for i in res_m.issues)
    assert res_m.override == "turn-state"
    # a compliant reply: the judge's verdict stands untouched, no override
    res_ok, _ = await _run(agent, MOVE, "t3", tools=tools)
    assert res_ok.verdict == VerifyVerdict.REFUTED and not getattr(res_ok, "override", "")


@pytest.mark.asyncio
async def test_shape_and_state_chain_in_arm_order_on_both_paths(agent_at_site):
    """R5: one input, one story — a reply that is BOTH a raw dump and a
    JSON-constraint violation carries both issues and the chain in the
    order the arms run, on the tool turn AND the tool-free turn (review
    finding 5: the first version exited on the shape alone tool-free)."""
    agent, verifier, calls, sidecar = agent_at_site
    tools = [{"name": "execute", "arguments": {"command": "python3 x.py"},
              "content": "EXIT CODE: 0\nok", "result": "EXIT CODE: 0\nok"}]
    res, _ = await _run(agent, DUMP, "c1", tools=tools)
    assert res.verdict == VerifyVerdict.REFUTED and res.override == "reply-shape+turn-state"
    assert len(res.issues) == 2 and "raw tool output" in res.issues[0] and res.issues[1].startswith("strict_json: ")
    assert sidecar()[-1].get("override") == "reply-shape+turn-state"
    res_free, _ = await _run(agent, DUMP, "c2")
    assert res_free.override == "reply-shape+turn-state"
    assert len(res_free.issues) == 2 and "raw tool output" in res_free.issues[0] and res_free.issues[1].startswith("strict_json: ")
    assert sidecar()[-1].get("route") == "reply-shape+turn-state"


def test_the_merge_helper_keeps_both_arms_and_replaces_a_below_gate_refute():
    """Identity pins on the ONE merge both arms use (review findings 5, 6)."""
    from ghost_agent.core.agent import GhostAgent
    shape = VerifyResult(verdict=VerifyVerdict.REFUTED, confidence=0.9, reasoning="s",
                         issues=["raw tool output pasted as the answer"])
    state = VerifyResult(verdict=VerifyVerdict.REFUTED, confidence=0.9, reasoning="t",
                         issues=["strict_json: the request asked for strict JSON and nothing else, but x"])
    judge = VerifyResult(verdict=VerifyVerdict.REFUTED, confidence=0.9, reasoning="j", issues=["L1", "L2", "L3"])
    out = GhostAgent._merge_mechanical_refute(judge, shape, "reply-shape")
    out = GhostAgent._merge_mechanical_refute(out, state, "turn-state")
    assert out is judge and out.issues == ["L1", "L2", shape.issues[0], state.issues[0]]
    assert out.override == "reply-shape+turn-state"
    weak = VerifyResult(verdict=VerifyVerdict.REFUTED, confidence=0.55, reasoning="w", issues=["W1"])
    out2 = GhostAgent._merge_mechanical_refute(weak, state, "turn-state")
    assert out2 is state and out2.confidence == 0.9 and out2.override == "turn-state"
    assert GhostAgent._merge_mechanical_refute(None, None, "turn-state") is None


@pytest.mark.asyncio
async def test_ablation_and_trivial_chat_get_no_state_verdict(agent_at_site):
    """The guard is the shape check's guard: no verifier / no client / a
    strict greeting → no verdict."""
    agent, verifier, calls, sidecar = agent_at_site
    agent._is_strict_trivial_chat = lambda lc: True
    assert (await _run(agent, PROSE_THEN_MOVE, "g1"))[0] is None
    agent._is_strict_trivial_chat = lambda lc: False
    agent.context.verifier = None
    assert (await _run(agent, PROSE_THEN_MOVE, "g2"))[0] is None
    v2 = MagicMock(); v2.llm_client = None
    agent.context.verifier = v2
    assert (await _run(agent, PROSE_THEN_MOVE, "g3"))[0] is None


# ── consumers of a delivery-shape refute ───────────────────────────────────

def _shape_issue_examples():
    out = [f"{r}: {m}" for r, m in refute_turn_state(request=CHESS, reply=PROSE_THEN_MOVE)]
    out += [f"{r}: {m}" for r, m in refute_turn_state(request="Reply with exactly: PONG", reply="I refuse, sorry.")]
    out += [f"{r}: {m}" for r, m in refute_turn_state(request="reply with just the number.", reply="It depends entirely.")]
    out += [f"{r}: {m}" for r, m in refute_turn_state(request="Answer in one word.", reply="Red, yellow, blue.")]
    out += [f"{r}: {m}" for r, m in refute_turn_state(request="Confirm in one line.", reply="One.\nTwo.")]
    out += [f"{r}: {m}" for r, m in refute_turn_state(request="in one sentence: x",
                                                       reply="Alpha runs. Beta waits. Gamma fails. Delta retries.")]
    out += [f"{r}: {m}" for r, m in refute_turn_state(
        request="look it up", reply=ASSERTIVE, tools_run=[{"name": "browser", "content": BROWSER_404}])]
    return out


def test_every_rule_is_a_delivery_shape_complaint():
    """ONE predicate — `_delivery_shape_only` over `_REFUTE_TASK_ARTIFACT_RE`
    — decides "never a project task", "never a retroactive correction" and
    the shape repair directive. Review C6: the first pin covered 2 of 7
    rules; here every rule's real issue text passes it, and a grounded
    issue (alone or mixed in) does not."""
    from ghost_agent.core.agent import GhostAgent
    issues = _shape_issue_examples()
    assert sorted(i.split(":")[0] for i in issues) == sorted(
        ["strict_json", "exact", "number_only", "word_cap", "line_cap", "sentence_cap", "empty_evidence"])
    for i in issues:
        assert GhostAgent._delivery_shape_only(VerifyResult(
            verdict=VerifyVerdict.REFUTED, confidence=0.9, reasoning="r", issues=[i])), i
    grounded = "the CSV export omits the header row the request asked for"
    assert not GhostAgent._delivery_shape_only(VerifyResult(
        verdict=VerifyVerdict.REFUTED, confidence=0.9, reasoning="r", issues=[grounded]))
    assert not GhostAgent._delivery_shape_only(VerifyResult(
        verdict=VerifyVerdict.REFUTED, confidence=0.9, reasoning="r", issues=[grounded, issues[0]]))
    assert not GhostAgent._delivery_shape_only(VerifyResult(
        verdict=VerifyVerdict.REFUTED, confidence=0.9, reasoning="r", issues=[]))


def test_a_state_issue_is_never_filed_as_a_project_task(mock_context, monkeypatch):
    from ghost_agent.core.agent import GhostAgent
    monkeypatch.delenv("GHOST_REFUTE_FOLLOWUP_TASKS", raising=False)
    agent = GhostAgent(mock_context)
    added = []

    class Store:
        def get_project(self, pid):
            return {"id": pid, "status": "ACTIVE", "metadata": {}}

        def add_task(self, pid, description, **kw):
            added.append(description); return {"id": "t1"}

        def list_tasks(self, pid):
            return []
    agent.context.project_store = Store()
    issues = _shape_issue_examples()
    assert all(len(i) >= agent._REFUTE_TASK_MIN_CHARS for i in issues)
    agent._file_refute_followup_tasks(
        VerifyResult(verdict=VerifyVerdict.REFUTED, confidence=0.9, reasoning="r", issues=issues), "p1")
    assert added == []
    agent._file_refute_followup_tasks(
        VerifyResult(verdict=VerifyVerdict.REFUTED, confidence=0.9, reasoning="r",
                     issues=["the CSV export omits the header row the request asked for"]), "p1")
    assert len(added) == 1


def test_a_shape_refute_queues_no_retroactive_correction(mock_context):
    """Review finding 4: a late shape refute used to queue "Correction to my
    previous answer: word_cap: …" — useless to the user, and one of three
    newest-win slots. A grounded refute still queues."""
    from ghost_agent.core.agent import GhostAgent
    agent = GhostAgent(mock_context)
    agent._human_label_locked = lambda tid: False
    agent._backfill_trajectory_outcome = lambda *a, **k: None
    agent._emit_late_outcome_correction = lambda *a, **k: None
    agent._file_refute_followup_tasks = lambda *a, **k: None
    agent._pending_corrections = []
    shape = VerifyResult(verdict=VerifyVerdict.REFUTED, confidence=0.9, reasoning="r",
                         issues=_shape_issue_examples()[:2])
    agent._record_late_verdict(shape, "t-shape", conv_fp="c1", force_correction=True)
    assert agent._pending_corrections == []
    grounded = VerifyResult(verdict=VerifyVerdict.REFUTED, confidence=0.9, reasoning="r",
                            issues=["Stiva's is not in the evidence"])
    agent._record_late_verdict(grounded, "t-fact", conv_fp="c1", force_correction=True)
    assert len(agent._pending_corrections) == 1 and "Stiva" in agent._pending_corrections[0]["note"]


def test_the_repair_directive_for_a_shape_refute_asks_for_the_same_answer_reshaped():
    """Review finding 3: "do NOT repeat the same claim" is exactly wrong for
    a shape refute — the claim is right and must be repeated in the shape
    asked for, with no tools."""
    from ghost_agent.core.agent import _render_refute_directive
    crit = "strict_json: the request asked for strict JSON and nothing else, but the reply carries prose"
    d = _render_refute_directive(crit, "Reply with STRICT JSON", shape_only=True)
    assert "SHAPE" in d and "Re-send the SAME answer" in d and "NO tool calls" in d
    assert "Do NOT repeat the same claim" not in d
    d0 = _render_refute_directive("Stiva's is not in the evidence", "find a burger")
    assert "Do NOT repeat the same claim" in d0 and "SHAPE" not in d0


def test_the_repair_site_passes_shape_only_from_the_one_predicate():
    """AST: the single repair-directive call site passes
    `shape_only=self._delivery_shape_only(...)` — not a second rule."""
    import ghost_agent.core.agent as agent_mod
    tree = ast.parse(Path(agent_mod.__file__).read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", "") == "_render_refute_directive"]
    assert len(calls) == 1
    kw = {k.arg: ast.unparse(k.value) for k in calls[0].keywords}
    assert kw.get("shape_only", "").startswith("GhostAgent._delivery_shape_only(")


def test_the_streamed_tool_free_turn_reaches_the_mechanical_checks():
    """Review finding 1 (MAJOR): the streamed drain used to skip the verdict
    when no substantive tool ran, so the web UI's tool-free turns never
    reached reply-shape / memory-claim / turn-state. The skip branch is
    gone: the only `_sv_tool is None` handling is a log inside the spawn
    branch. The world it fails in: the `elif _sv_tool is None: … skipped`
    branch grows back."""
    import ghost_agent.core.agent as agent_mod
    src = Path(agent_mod.__file__).read_text()
    gate = src.split("VERIFIER GATE (STREAM)")[1][:14000]
    assert "elif _sv_tool is None:" not in gate
    assert "mechanical checks only" in gate
    spawn_at = gate.index("_sv_task = _glog.spawn_task(")
    assert gate.index("if _sv_tool is None:") < spawn_at


# ── the instrument can fail (R6) ───────────────────────────────────────────

def test_the_replay_script_counts_a_planted_violation_and_joins_human_labels(tmp_path):
    """The replay is the measurement this tier was accepted on. A five-row
    corpus: a violating turn a HUMAN approved, a compliant turn, a probe, a
    reflection row and a violating turn with no label. The world it fails
    in: the script walks the wrong field or partition, counts non-user
    kinds (review C14), or reports the corrections row count as the human
    join (review finding 1)."""
    root = tmp_path / "system" / "trajectories" / "2026-09-10"
    root.mkdir(parents=True)
    base = {"timestamp": "2026-09-10T10:00:00Z", "session_id": "s", "task_kind": "user_request",
            "system_prompt": "", "tool_calls": [], "n_steps": 0, "tokens_in": 0, "tokens_out": 0,
            "duration_s": 1.0, "outcome": "unknown", "failure_reason": "", "validator_signal": {},
            "extra": {}, "model": "m", "temperature": 0.0, "cluster": None, "tier": None,
            "sample_index": None, "batch_id": None, "planning_output": None}
    rows = [dict(base, id="a" * 32, user_request=CHESS, final_response=PROSE_THEN_MOVE, outcome="passed"),
            dict(base, id="b" * 32, user_request=CHESS, final_response=MOVE),
            dict(base, id="c" * 32, user_request=CHESS, final_response=PROSE_THEN_MOVE, task_kind="probe"),
            dict(base, id="d" * 32, user_request=CHESS, final_response=PROSE_THEN_MOVE, task_kind="reflection"),
            dict(base, id="e" * 32, user_request=CHESS, final_response=PROSE_THEN_MOVE)]
    (root / "session-x.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (tmp_path / "system" / "trajectories" / "corrections.jsonl").write_text(
        json.dumps({"trajectory_id": "a" * 32, "outcome": "passed", "source": "human_feedback:web"}) + "\n"
        + json.dumps({"trajectory_id": "z" * 32, "outcome": "passed", "source": "human_feedback:web"}) + "\n"
        + json.dumps({"trajectory_id": "e" * 32, "outcome": "failed", "source": "verifier_late"}) + "\n")
    env = dict(os.environ, GHOST_HOME=str(tmp_path),
               PYTHONPATH=str(Path(__file__).resolve().parent.parent / "src"))
    script = Path(__file__).resolve().parent.parent / "scripts" / "turn_state_replay.py"
    out = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert re.search(r"^3 real user turns", out.stdout, re.M)                      # probe + reflection excluded
    assert "human-labelled turns walked 1 (of 2 human label rows)" in out.stdout   # joined ≠ row count
    assert "of the human-labelled turns: 1" in out.stdout
    rule_line = re.search(r"^strict_json\s+2\s+(.*)$", out.stdout, re.M)
    assert rule_line, out.stdout
    assert "'passed': 1" in rule_line.group(1) and "'failed': 1" in rule_line.group(1)
    assert rule_line.group(1).rstrip().endswith("{'passed': 1, '-': 1}")       # by HUMAN label: 'a' only; 'e' unlabelled
    assert "1 fire(s) on a passed / human-approved turn" in out.stdout
    assert "a" * 32 in out.stdout and "b" * 32 not in out.stdout and "e" * 32 not in out.stdout
