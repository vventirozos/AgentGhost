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
§4NM: LEAK 2 (journal_challenges transferability / data-shape gate) went
with synthetic self-play; its tests were removed.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.core.agent import _has_meta_task_directive


# ── LEAK 1: the nudge must need a DIRECTIVE, not a keyword ──────────────

DIRECTIVES = [
    "remember this for next time",
    "Remember this: my preferred debugging tool on macOS is dtrace",
    "learn from this mistake",
    "update your profile: I use zsh",
    "please memorize this",
    "save this for later",
    "record this for future runs",
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
    src = inspect.getsource(GhostAgent.handle_chat) + inspect.getsource(GhostAgent._run_internal_turn)
    i = src.index("Checklist Nudge")
    block = src[i:i + 1600]
    assert "You have not fulfilled the learning/profile instructions" not in block
    assert "do NOT invent" in block
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
