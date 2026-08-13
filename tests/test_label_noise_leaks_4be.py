"""§4BE (2026-08-13) — two leaks that wrote FAKE labels into the corpus.

LEAK 1 — the checklist nudge minted junk lessons.
`has_meta_intent` was a bare substring test over the user's text for
{learn, skill, profile, lesson, playbook, memorize}, so "Define a composed
SKILL", "show me all custom SKILLS" and "what have you LEARNED today" all
armed it. The injected message then told the model it had "not fulfilled
the learning/profile instructions IN THE USER'S REQUEST" — a statement that
was false on every measured turn. Measured on the live trajectory corpus:
**59 of 59** arming turns carried no learning instruction (100% FP), and it
fired 39 times. Production req 32a8101d shows the cost: the model reasoned
"there's no explicit learning or profile instruction in their message… But
the system is requiring it", called `learn_skill` to comply, and minted a
lesson that vector-dedup reinforced to freq=11 — the false nudge writing
into the corpus that retrieval later injects.

LEAK 2 — self-play graded incoherent exercises.
A mined challenge wraps the original request in a FIXED harness ("a fixture
has been written — write a solution.py that performs the operation the user
asked for, APPLIED TO THAT FILE"). `_detect_data_shape` fell back to a
generic `input.txt` for ANY request, so "Run the composed skill
youtube_transcribe on this URL" became a file exercise. Observed 2026-08-12
17:36: the solver ran the REAL macro (8 Tor exit rotations, twice — live
egress from a "synthetic" exercise), then satisfied the validator by
printing the fixture, scoring passed=True and minting a lesson from an
exercise whose task and grader never agreed.

Both fixes are FILTERS, so both get a false-positive audit here: the strings
they must NOT match are pinned alongside the ones they must.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.core.agent import _has_meta_task_directive
from ghost_agent.core.journal_challenges import (
    _detect_data_shape, _is_transferable_challenge, _synthesize_challenge,
)


# ── LEAK 1: the nudge must need a DIRECTIVE, not a keyword ──────────────

DIRECTIVES = [
    "remember this for next time",
    "Remember this: my preferred debugging tool on macOS is dtrace",
    "learn from this mistake",
    "update your profile: I use zsh",
    "please memorize this",
    "call learn_skill when you are done",
    "save this for later",
    "record this for future runs",
    "take a lesson from what just happened",
]

# Every one of these ARMED the old nudge on live traffic and carries no
# instruction to record anything. A hit here re-opens the junk-lesson path.
NOT_DIRECTIVES = [
    "Define a composed skill NOW, exactly as specified",
    "show me all custom skills.",
    "List your acquired skills. Just the names, one line.",
    "what have you learned today ?",
    "have you learned anything today ?",
    "tell me about one mistake that you have learned from.",
    "Create a new skill that will generate a secure password with N characters",
    "run the news_headlines skill",
    "show me the skill playbook",
]

# Storage verbs pointed at a DATA store are not meta-tasks. These pin the
# SECOND guard (added when the first cut of the new predicate still armed
# on the live youtube-project request "transcribe it, save it to the
# knowledge base"), so they deliberately do NOT contain the old keywords.
STORAGE_DESTINATIONS = [
    "transcribe it, save it to the knowledge base",
    "store this in the database",
    "save it to a file",
    "record this in the project notes",
    "save that to disk",
]


@pytest.mark.parametrize("text", DIRECTIVES)
def test_real_directive_arms_the_nudge(text):
    assert _has_meta_task_directive(text.lower()) is True


@pytest.mark.parametrize("text", STORAGE_DESTINATIONS)
def test_storing_data_is_not_a_meta_task(text):
    """"save it to the knowledge base" is a DOCUMENT write, not a lesson."""
    assert _has_meta_task_directive(text.lower()) is False, (
        f"a data-store destination must not arm the nudge: {text!r}")


@pytest.mark.parametrize("text", NOT_DIRECTIVES)
def test_keyword_mention_does_not_arm_the_nudge(text):
    assert _has_meta_task_directive(text.lower()) is False, (
        f"false positive re-opens the junk-lesson path: {text!r}")


def test_old_substring_predicate_would_have_fired_on_all_of_them():
    """Non-vacuity: these strings are only interesting because the OLD
    predicate armed on them. If this ever goes green trivially, the
    fixtures have drifted away from the bug they pin."""
    old_kw = ["learn", "skill", "profile", "lesson", "playbook", "memorize"]
    for text in NOT_DIRECTIVES:
        assert any(k in text.lower() for k in old_kw), (
            f"fixture no longer exercises the old bug: {text!r}")


def test_nudge_message_does_not_assert_an_instruction_that_may_not_exist():
    """The old message stated as fact that the user gave learning
    instructions. That is an INFERENCE, and when it was wrong the model
    complied by inventing a lesson. The replacement must hedge and must
    explicitly forbid inventing one."""
    import inspect
    from ghost_agent.core.agent import GhostAgent
    src = inspect.getsource(GhostAgent.handle_chat)
    i = src.index("Checklist Nudge")
    block = src[i:i + 1600]
    assert "You have not fulfilled the learning/profile instructions" not in block
    assert "do NOT invent a lesson" in block
    assert "If that reading is right" in block or "appears to" in block


# The subject after "remember that" decides, and the first cut had it
# BACKWARDS — it excluded I/we (user facts, exactly what the profile store
# is for) and admitted you/the/it (task constraints). Unpinned, a reviewer
# deleted the whole exclusion and 166 tests stayed green.
CONSTRAINT_NOT_DIRECTIVE = [
    "remember that you use TOR.",                    # the live FP
    # round-4: the exclusion was a closed DENYLIST of subjects, so every
    # subject outside it armed. Now an allowlist: the "remember that X"
    # form needs a FIRST-PERSON subject.
    "remember that TOR must be used",
    "remember that this repo uses ripgrep",
    "remember that Ghost uses TOR",
    "remember that a rebuild takes 8 minutes",
    "remember that your workspace is read-only",
    "remember that the sandbox has no network",
    "remember that it takes 8 rotations",
    "remember that time when the server melted?",     # rhetorical
    "do you remember that we discussed postgres tuning?",   # a question
    "can you remember that we use tabs?",
    "Message 1: 'You remember that your favorite project was the Chess "
    "Coach v3'",                                      # quoted test material
]

USER_FACT_IS_A_DIRECTIVE = [
    "remember that I use zsh",
    "remember that I prefer ripgrep over grep",
    "remember that we agreed to use tabs",
    "please remember that my laptop is offline",
]


@pytest.mark.parametrize("text", CONSTRAINT_NOT_DIRECTIVE)
def test_task_constraints_and_questions_do_not_arm(text):
    assert _has_meta_task_directive(text.lower()) is False, (
        f"a constraint/question must not arm the nudge: {text!r}")


@pytest.mark.parametrize("text", USER_FACT_IS_A_DIRECTIVE)
def test_user_facts_still_arm(text):
    """The over-fix direction: excluding first-person subjects dropped the
    most natural genuine phrasings."""
    assert _has_meta_task_directive(text.lower()) is True


# ── LEAK 2: self-play may only mine TRANSFERABLE, data-shaped work ──────

# The live incident, plus the families that caused real side effects.
NON_TRANSFERABLE = [
    'Run the composed skill youtube_transcribe on this short clip: '
    'url="https://www.youtube.com/watch?v=jNQXAC9IRR0". Report honestly.',
    "invoke the macro with the url and report what happened",
    "download the video as mp4 and transcribe it",
    "restart the service please.",
    "browse to the dashboard and click the login button",
    "deploy the new build to production",
    "show me all custom skills",
    "search what the features of postgresql 19 will be",
    "You are playing a live chess game as BLACK against Vasilis",
    "Use your file_system tool to list the files in your sandbox root",
    "create a new project called 'youtube transcription'",
]

# Honest data work must still be minable — the gate is not allowed to
# starve self-play of legitimate material.
TRANSFERABLE = [
    "parse the CSV and count the rows where status is error",
    "summarize the JSON payload in the file",
    "find duplicate entries in the log file",
    "Write a python script to sort CSV by a column.",
    "extract the failing test names from the log file",
]


@pytest.mark.parametrize("text", NON_TRANSFERABLE)
def test_non_transferable_requests_are_refused(text):
    minable = (_is_transferable_challenge(text)
               and _detect_data_shape(text.lower()).get("explicit"))
    assert not minable, f"would grade an incoherent exercise: {text!r}"


@pytest.mark.parametrize("text", TRANSFERABLE)
def test_data_work_is_still_minable(text):
    assert _is_transferable_challenge(text) is True
    assert _detect_data_shape(text.lower()).get("explicit") is True


# The verb gate is the single biggest blocker in the change (it alone
# accounts for ~39% of refusals) and shipped with NO coverage: a reviewer
# replaced it with `re.compile("")` — disabling it entirely — and all four
# self-play suites stayed green. These pin it in both directions.
VERB_REQUIRED = [
    # declarative statements name no operation; the harness's "perform the
    # operation the user asked for" has nothing to bind to
    "I own a company called EvolMonkey that provides PostgreSQL services.",
    "I am a PostgreSQL engineer with 20+ years of experience.",
]

VERB_PRESENT = [
    # inflected forms must match — the first cut used `parse\b`, which does
    # not match "parsing", and four stash tests went red
    "First stashed failure about json parsing",
    "Second failing task about a broken sqlite database query",
    "how many rows in the csv have status=error?",
    "give me the top 10 IPs in the access log",
    "turn this csv into json",
    "what is the average of the value column in input.csv",
    "search the log file for lines with ERROR and tell me the count",
    # round-4: y-stems and doubled-consonant gerunds were DEAD verbs —
    # `verif`+`y` / `identif`+`y` could never match through the shared
    # suffix group, silently refusing real work (one live corpus entry).
    "verify that the csv has no missing values",
    "identify the bad rows in the csv",
    "debug the parsing in the json file",
    "split the csv into monthly files",
    # irregulars added for the reviewer's phrasings — previously unpinned
    "plot the data from the csv",
    "i need a breakdown of the csv by month",
    "tail the log and show the last errors",
    "grep the log for timeouts",
]


@pytest.mark.parametrize("text", VERB_REQUIRED)
def test_statement_without_an_operation_is_refused(text):
    from ghost_agent.core.journal_challenges import _TASK_VERB_RE
    assert not _TASK_VERB_RE.search(text)
    assert _is_transferable_challenge(text) is False


@pytest.mark.parametrize("text", VERB_PRESENT)
def test_real_data_operations_are_recognised(text):
    from ghost_agent.core.journal_challenges import _TASK_VERB_RE
    assert _TASK_VERB_RE.search(text), f"legitimate data work refused: {text!r}"
    assert _is_transferable_challenge(text) is True


def test_verb_gate_is_load_bearing():
    """Disabling the verb gate must change outcomes — otherwise the whole
    filter is decorative and its refusals come from elsewhere."""
    import ghost_agent.core.journal_challenges as jc
    blocked = [t for t in VERB_REQUIRED if not jc._is_transferable_challenge(t)]
    assert blocked, "the verb gate blocks nothing that the denylist misses"


def test_shape_fallback_is_marked_inexplicit():
    """The generic input.txt fallback is the incoherence generator: it hands
    ANY request a fixture the harness then claims the task is about."""
    assert _detect_data_shape("do something interesting")["kind"] == "text"
    assert _detect_data_shape("do something interesting")["explicit"] is False
    # explicit shapes keep their kind AND are marked
    for text, kind in (("parse the spreadsheet at data.csv", "csv"),
                       ("walk this JSON payload", "json"),
                       ("parse the nginx access log", "log"),
                       ("write a SELECT against postgres", "sql")):
        got = _detect_data_shape(text)
        assert got["kind"] == kind and got["explicit"] is True


def test_synthesize_refuses_the_live_incident_end_to_end():
    """The gate must hold at the real entry point, not just in isolation —
    otherwise the mining path keeps producing the fake PASS."""
    entry = {"kind": "post_mortem", "data": {"user": NON_TRANSFERABLE[0]}}
    assert _synthesize_challenge(entry) is None


def test_synthesize_still_produces_a_challenge_for_data_work():
    """Non-vacuity for the gate: if this returns None the mining path is
    dead, and every 'no incoherent exercise' assertion above is green for
    the wrong reason."""
    entry = {"kind": "post_mortem",
             "data": {"user": "parse the CSV file and count the rows where "
                              "the status column is error"}}
    mined = _synthesize_challenge(entry)
    assert mined is not None
    assert "solution.py" in mined.challenge
    assert mined.setup_script and mined.validation_script


# Round-2 regressions, pinned. Both were introduced BY the fixes for the
# over-blocking finding, and both re-open the incoherent-exercise path.
STEM_MUST_NOT_MATCH = [
    "the csv is ready, what do you think?",            # read  in ready
    "my country is Greece and the csv is from 2019",   # count in country
    "here is the summer csv, no action needed",        # sum   in summer
    "listen, the csv is fine",                         # list  in listen
    "the testament is in the json file",               # test  in testament
    "the log is from the maple street server",         # map   in maple
    "as per usual the csv is attached",                # per \w+ dropped
    "the csv, which is why I asked",                   # which \w+ dropped
]

LOG_WORD_MUST_NOT_MATCH = [
    "fix the login page, the button does nothing",
    # round-4: bare `log` is also a VERB — the third appearance of this
    # class, after the substring and word-boundary versions
    "log in to the box and fix the login page",
    "please log a ticket and check the button",
    "log the request and show me the result",
    # round-5: the PLURAL verb sense — 4th appearance of this class
    "the service logs errors, fix the crash",
    "the user logs a ticket, please check the format",
    "she logs every request and we fix the parser",
    "debug the logic in the parser module",
    "check the logout flow after the release",
    "summarize the blogs I wrote last month",
]


@pytest.mark.parametrize("text", STEM_MUST_NOT_MATCH)
def test_stems_still_require_a_word_boundary(text):
    minable = (_is_transferable_challenge(text)
               and _detect_data_shape(text.lower()).get("explicit"))
    assert not minable, f"a stem matched mid-word: {text!r}"


# The other side of the plural rule: a bare plural governed by a DATA VERB
# is the noun ("could not match logs", "parse logs"), and refusing it broke
# a real stash fixture — caught only by the full suite, after I had already
# committed. Pinned so the noun/verb distinction cannot collapse either way.
LOG_VERB_GOVERNED_MUST_MATCH = [
    "a regex that could not match logs",
    "parse logs and count the errors",
    "count the lines in their logs",
    "tail the log and show the last errors",
]


@pytest.mark.parametrize("text", LOG_VERB_GOVERNED_MUST_MATCH)
def test_verb_governed_plural_is_log_data(text):
    assert _detect_data_shape(text.lower())["kind"] == "log", (
        f"legitimate log work refused: {text!r}")


@pytest.mark.parametrize("text", LOG_WORD_MUST_NOT_MATCH)
def test_log_vocabulary_is_word_bounded(text):
    """"the log" must not match "the login"/"the logic"; "logs" must not
    match "blogs" — otherwise UI-bug prose becomes a log exercise."""
    assert _detect_data_shape(text.lower())["kind"] != "log", (
        f"UI/prose misread as a log task: {text!r}")
