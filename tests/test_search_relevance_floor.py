"""Search relevance floor (2026-09-15, req 4b518a82 evaluation).

The wave race declared a winner on the first NON-EMPTY batch, which is a
liveness test, not a relevance test: yahoo won one wave in 2.6s with
eleven Microsoft-account support pages, none mentioning Revolut. Measured
over the recorded corpus (557 web_search calls), 38 — 6.8% — came back
with not one query content word anywhere in the result set: Bing-quiz
threads, NHL standings, coloring pages, AZLyrics, Domino's Pizza México.
Those 38 are what this floor rejects; see the reach limit noted below.

The world each pin fails in is named on the test.
"""

import os
import re
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.tools.search import (
    _failure_category,
    _rel_tokens,
    _results_are_off_topic,
)


def _hit(title="", body="", href="https://example.com/x"):
    return {"title": title, "body": body, "href": href}


# Real losing batches, trimmed, reproduced from recorded trajectories
# rather than invented. All three are from this same investigation.
_REVOLUT_QUERY = ("Revolut customer data breach 2026 government request "
                  "fraudulent")
_BING_QUIZ_JUNK = [
    _hit("Bing Homepage Quiz - Reddit",
         "Microsoft Bing Homepage daily quiz questions and their answers",
         "https://www.reddit.com/r/BingHomepageQuiz/hot/"),
    _hit("Quiz Answers for today : r/MicrosoftRewards - Reddit",
         "quiz that was mentioned a month ago and mentioned again more "
         "recently, but never appeared on my dash until today.",
         "https://www.reddit.com/r/MicrosoftRewards/comments/cwya5d/"),
]
_NHL_JUNK = [
    _hit("NHL Hockey Standings | NHL.com",
         "The official standings for the National Hockey League.",
         "https://www.nhl.com/standings"),
]

# KNOWN REACH LIMIT, stated so nobody mistakes the floor for a relevance
# ranker: a batch sharing ONE incidental word survives it. The opening
# turn's yahoo batch (eleven Microsoft-support pages) matched "data" and
# would still pass. Every statistic measured that WOULD reject it —
# per-result token counts, coverage ratios — also rejected genuinely good
# result sets ("PostgreSQL 20 features roadmap 2028" → "PostgreSQL:
# Release Notes" matches exactly one token and is the right answer), so
# the floor is deliberately the weakest rule with no false positives
# across all 38 of its historical firings.


def test_the_live_failure_is_rejected():
    """FAILS IF: the floor is absent, or arms only on an empty batch.

    This exact query returned these exact Reddit Bing-quiz threads live
    (2026-09-14). The batch is non-empty, so the wave counted it a WIN and
    the reformulation wave never ran.
    """
    assert _results_are_off_topic(_REVOLUT_QUERY, _BING_QUIZ_JUNK) is True
    assert _results_are_off_topic(_REVOLUT_QUERY, _NHL_JUNK) is True


def test_one_genuine_hit_is_enough_to_pass():
    """FAILS IF: the floor demands coverage rather than presence.

    A long query whose best result matches only 'Revolut' is a NORMAL
    search, not a junk batch: 45 of 557 recorded calls sit below 20%
    coverage and are fine. The floor must be presence-of-one, not a ratio.
    """
    batch = _BING_QUIZ_JUNK + [
        _hit("Revolut confirms customer data breach",
             "The company said an unauthorised third party used a legitimate "
             "agency domain.",
             "https://techcrunch.com/2026/09/12/revolut-confirms")]
    assert _results_are_off_topic(_REVOLUT_QUERY, batch) is False


def test_space_stripped_snippets_still_match():
    """FAILS IF: matching is word-boundaried instead of substring.

    Engine snippets arrive with whitespace collapsed out of the markup —
    the corpus is full of 'Revolutconfirmscustomerdatabreach'. A \\b-anchored
    rule sees no tokens there and rejects a GOOD batch.
    """
    batch = [_hit("Revolutconfirmscustomerdatabreachthroughfakegovernment",
                  "RevolutLeaksPassportsandBTCHistory",
                  "https://example.com/a")]
    assert _results_are_off_topic(_REVOLUT_QUERY, batch) is False


def test_greek_query_with_greek_results_passes():
    """FAILS IF: the tokenizer is ASCII-only — the first draft of it was.

    `[a-z0-9]+` scores EVERY Greek query 0.0, so an ASCII floor rejects
    every one of the operator's Greek searches. 11 such queries are in the
    recorded corpus.
    """
    q = "Τέμπη 2025 ατύχημα αυτοκίνητο νεκροί κατεύθυνση Θεσσαλονίκη"
    good = [_hit("Τέμπη: το ατύχημα με το αυτοκίνητο",
                 "Επτά νεκροί στην κατεύθυνση προς Θεσσαλονίκη.",
                 "https://example.gr/tempi")]
    assert _rel_tokens(q), "Greek query must produce content tokens"
    assert _results_are_off_topic(q, good) is False


def test_greek_query_with_unrelated_results_is_rejected():
    """FAILS IF: Greek text is tokenised but never compared (fold mismatch)."""
    q = "Τέμπη 2025 ατύχημα αυτοκίνητο νεκροί κατεύθυνση Θεσσαλονίκη"
    junk = [_hit("Premier League Table, Standings & Form Guide",
                 "View the latest standings in the Premier League.",
                 "https://example.com/football")]
    assert _results_are_off_topic(q, junk) is True


def test_accent_and_case_folding_counts_as_a_match():
    """FAILS IF: folding is dropped — 'ΤΈΜΠΗ' must match 'τεμπη'.

    The batch must carry enough prose to ARM the floor, or the test
    passes in both worlds: the first version of it used a 27-character
    fixture that fell under `_REL_MIN_RESULT_CHARS`, so the unfolded
    mutant survived the battery.
    """
    q = "Τέμπη ατύχημα αυτοκίνητο"
    hit = [_hit("ΤΕΜΠΗ",
                "ΑΤΥΧΗΜΑ ΜΕ ΑΥΤΟΚΙΝΗΤΟ ΣΤΗΝ ΕΘΝΙΚΗ ΟΔΟ, ΜΕ ΠΟΛΛΟΥΣ "
                "ΤΡΑΥΜΑΤΙΕΣ ΚΑΙ ΕΚΤΕΤΑΜΕΝΕΣ ΖΗΜΙΕΣ ΣΤΟ ΟΔΟΣΤΡΩΜΑ.",
                "https://example.gr/x")]
    assert _results_are_off_topic(q, hit) is False
    # Control: the same UNRELATED prose at the same length IS rejected,
    # so the pass above is folding, not the arming guard standing down.
    miss = [_hit("PREMIER LEAGUE",
                 "VIEW THE LATEST STANDINGS AND FORM GUIDE FOR THE SEASON, "
                 "WITH FIXTURES, RESULTS AND TOP SCORERS LISTED BY CLUB.",
                 "https://example.com/football")]
    assert _results_are_off_topic(q, miss) is True


def test_punctuation_inside_a_result_term_still_matches():
    """FAILS IF: the haystack keeps its separators.

    Engines split terms with markup and punctuation — 'Postgre-SQL',
    'Re:volut'. Stripping every non-alphanumeric from the haystack is
    what makes a token match those; without it a GOOD batch is rejected.
    """
    # The punctuated term must be the ONLY token that can match, or the
    # pin passes on an incidental word and the un-stripped mutant lives.
    q = "PostgreSQL upgrade downtime"
    batch = [_hit("Postgre-SQL: what the notes say",
                  "The Postgre/SQL global development group published its "
                  "notes for the coming major version this week, covering "
                  "planner work and replication changes in some detail.",
                  "https://example.com/notes")]
    assert _results_are_off_topic(q, batch) is False
    # Control: with the term written plainly it matches either way, so the
    # assertion above is genuinely testing the separator strip.
    plain = [_hit("PostgreSQL: what the notes say",
                  "The PostgreSQL global development group published its "
                  "notes for the coming major version this week, covering "
                  "planner work and replication changes in some detail.",
                  "https://example.com/notes")]
    assert _results_are_off_topic(q, plain) is False


def test_short_query_is_exempt_so_spelling_correction_survives():
    """FAILS IF: the floor arms on any query length.

    Live: `Πεοτρόμπης` (a misspelling) returned `Πετρόμπεης Μαυρομιχάλης`
    — zero token overlap and the CORRECT answer. Under the minimum the
    floor must stand down or typo-correction breaks.
    """
    q = "Πεοτρόμπης"
    corrected = [_hit("Πετρόμπεης Μαυρομιχάλης - Βικιπαίδεια",
                      "Ο Πετρόμπεης Μαυρομιχάλης (πραγματικό όνομα: Πέτρος)",
                      "https://el.wikipedia.org/wiki/x")]
    assert len(_rel_tokens(q)) < 3
    assert _results_are_off_topic(q, corrected) is False


def test_two_token_query_is_exempt():
    """FAILS IF: the minimum is set below 3 content tokens."""
    assert _results_are_off_topic("Petroby biography", [_hit("Unrelated")]) is False


def test_empty_batch_is_not_off_topic():
    """FAILS IF: an empty wave is re-labelled off-topic.

    'Empty' and 'off-topic' are different diagnoses — one says Tor/the
    engine failed, the other says the engine is useless for this query.
    """
    assert _results_are_off_topic(_REVOLUT_QUERY, []) is False


def test_a_batch_with_no_snippet_text_is_never_judged():
    """FAILS IF: the floor condemns a batch for what it does not contain.

    Found by the suite: several harnesses drive the race with bare
    `{"href": ...}` rows, and an engine can return a hit with no snippet.
    Absence of text is absence of evidence — 21 tests went red on the
    first draft, every one of them href-only rows.
    """
    bare = [{"href": "https://a.com/1"}, {"href": "https://a.com/2"}]
    assert _results_are_off_topic(_REVOLUT_QUERY, bare) is False
    tiny = [_hit("t", "b", "h")]
    assert _results_are_off_topic(_REVOLUT_QUERY, tiny) is False


def test_the_arming_guard_does_not_blunt_the_floor():
    """FAILS IF: the minimum-text guard is set so high that real junk
    escapes. The Bing-quiz batch is ~200 characters of prose — well over
    the line — and must still be rejected.
    """
    prose = "".join(f"{h['title']}{h['body']}" for h in _BING_QUIZ_JUNK)
    assert len(prose) >= 80
    assert _results_are_off_topic(_REVOLUT_QUERY, _BING_QUIZ_JUNK) is True


def test_url_alone_can_carry_the_match():
    """FAILS IF: only title+body are searched.

    Some engines return a bare URL row with an empty snippet; the slug
    still identifies the topic.
    """
    # Enough prose to ARM the floor (otherwise this would pass for the
    # wrong reason — both worlds agree on an empty batch), with the only
    # query term anywhere living in a URL slug.
    batch = [
        _hit("Quiz Answers for today : r/MicrosoftRewards - Reddit",
             "quiz that was mentioned a month ago and mentioned again more "
             "recently, but never appeared on my dash until today.",
             "https://www.reddit.com/r/MicrosoftRewards/comments/cwya5d/"),
        _hit("", "", "https://example.com/revolut-breach-2026"),
    ]
    assert _results_are_off_topic(_REVOLUT_QUERY, batch[:1]) is True, (
        "control: without the slug row this batch IS off-topic")
    assert _results_are_off_topic(_REVOLUT_QUERY, batch) is False


def test_stopwords_alone_cannot_satisfy_the_floor():
    """FAILS IF: stopwords are left in the token set.

    'for'/'the'/'data' appear in almost any English page; if stopwords
    counted, no batch would ever be off-topic.
    """
    toks = _rel_tokens("the of and for with this that")
    assert toks == []


def test_off_topic_is_its_own_failure_category():
    """FAILS IF: 'off-topic' collapses into 'empty' or 'error'.

    The operator reads these categories to tell a dead circuit from a
    useless engine; merging them destroys that signal.
    """
    assert _failure_category("off-topic") == "off-topic"
    assert _failure_category("empty") == "empty"
    assert _failure_category("timed out after 30s") == "timeout"


@pytest.mark.asyncio
async def test_race_keeps_going_after_an_off_topic_win(monkeypatch):
    """FAILS IF: the off-topic batch is returned as the wave winner.

    The end-to-end property: a fast junk engine must LOSE, and a slower
    on-topic engine must still be able to win the same wave.
    """
    import ghost_agent.tools.search as S

    class _FakeDDGS:
        def __init__(self, **kw):
            self._backend = None

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def text(self, query, **kw):
            # `backend` is passed positionally-by-keyword by the caller.
            backend = kw.get("backend")
            if backend == "yahoo":
                return list(_BING_QUIZ_JUNK)
            if backend == "mojeek":
                # The junk engine MUST finish first or this test proves
                # nothing: the wave is a genuine FIRST_COMPLETED race, so
                # without this delay mojeek can win on its own and the
                # assertions below pass whether or not the floor exists.
                # (Caught by the mutation battery — two mutants that died
                # on one run survived the next.)
                time.sleep(0.4)
                return [_hit("Revolut confirms customer data breach",
                             "Revolut said an unauthorised third party used a "
                             "legitimate government agency domain.",
                             "https://techcrunch.com/revolut")]
            return []

    import types
    fake_mod = types.ModuleType("ddgs")
    fake_mod.DDGS = _FakeDDGS
    monkeypatch.setitem(sys.modules, "ddgs", fake_mod)
    monkeypatch.setattr(S, "_RACE_ENGINES", ("yahoo", "mojeek"), raising=False)

    out = await S._race_search_wave(_REVOLUT_QUERY, max_results=8,
                                    tor_proxy=None, wave=0)
    assert out, "an on-topic engine was available and must win the wave"
    blob = " ".join(str(r.get("title", "")) for r in out).lower()
    assert "revolut" in blob
    assert "microsoft" not in blob
