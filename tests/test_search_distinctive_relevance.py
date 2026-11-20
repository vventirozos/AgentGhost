"""§4IL — an engine that answers a different question does not win the wave.

THE LIVE FAILURE (probe ifs21133…, 2026-09-17): `web_search("eckit Grid
reduced_gg spec format nxacc pl python")` returned "How to get help in
Windows - Microsoft Support" as result 1 and the agent read it as its
look-it-up attempt. The batch rule ("any content word anywhere") passed
it on a shared common word. Replayed over 786 recorded searches
(2026-08/09): 52 batches (6.6%) were engines answering a different
question — Doha for a Greek gastronomy award, Control Panel for a
restaurant, NHL standings and XVideos for a Revolut breach — every one
with a common word in common and no DISTINCTIVE token of the query.

Now: `distinctive_tokens` (the leading content word unless it is query
framing, digit/underscore tokens, words the system dictionary does not
know — stem-aware; never a year), `result_on_topic` (substring or fuzzy
≥0.8 for tokens of 6+ chars, so a one-letter typo still finds its page),
`rank_on_topic` (on-topic first, nothing dropped), and the wave's
off-topic check uses them; a query with no distinctive token keeps the
old batch rule.

World where each pin fails: the Windows page wins again, a typo loses its
page, a dictionary word masquerades as distinctive (plurals, years), the
subject word stops counting, results get dropped instead of re-ranked, or
the wave stops re-ranking its winner.
"""
import ast
import inspect

import pytest

from ghost_agent.tools import search as S

ECKIT = "eckit Grid reduced_gg spec format nxacc pl python"
WINDOWS = [{"title": "How to get help in Windows - Microsoft Support",
            "body": "Search for help on the taskbar, use the Tips app, select the Get help link "
                    "in the Settings app, or go to support.microsoft.com/windows.",
            "href": "https://support.microsoft.com/en-us/windows/how-to-get-help"},
           # the live batch's second row shared ONE common word with the query
           # ("format") — the old any-word batch rule passed the whole batch on it
           {"title": "Format a drive in Windows - Microsoft Support",
            "body": "How to format a USB drive or disk with the Settings app; choose a file system.",
            "href": "https://support.microsoft.com/en-us/windows/format-a-drive"}]
ECKIT_HIT = {"title": "eckit-geo: Grid — ECMWF", "body": "Grid specs: type=reduced_gg, N or pl …",
             "href": "https://github.com/ecmwf/eckit"}


@pytest.mark.skipif(not S._dict_words(), reason="no system dictionary on this host")
@pytest.mark.parametrize("query,expected", [
    (ECKIT, ["eckit", "reduced_gg", "nxacc"]),
    ("best pizza place Paralia Epydos Greece", ["paralia", "epydos", "greece"]),
    ("Tempi 2003 bus accident children died", ["tempi", "died"]),          # leading subject; year dropped
    ("tiny AI neural network train 5 minutes minimal resources Python", ["tiny"]),  # plurals are dictionary words
    ("number one supplement for anti aging most researched", []),          # weak leading word, all dictionary
    ("Qwen3.5-9B-A3B model specs context window", ["qwen3", "a3b"]),
    ("Χρυσοί Σκούφοι Βραβεία Γαστρονομίας", ["χρυσοι", "σκουφοι", "βραβεια", "γαστρονομιασ"]),
])
def test_distinctive_tokens(query, expected):
    assert S.distinctive_tokens(query) == expected


def test_years_and_weak_leading_words_never_count():
    assert S.distinctive_tokens("2025 Tempi road crash") == ["tempi"]
    assert "2025" not in S.distinctive_tokens("best watch 500 700 euros 2025 recommendation")
    assert S.distinctive_tokens("best watch 500 700 euros 2025 recommendation")[:2] == ["500", "700"]


def test_the_windows_page_is_off_topic_for_the_eckit_query():
    assert S._results_are_off_topic(ECKIT, WINDOWS) is True
    assert S._results_are_off_topic(ECKIT, WINDOWS + [ECKIT_HIT]) is False


def test_a_typo_still_finds_its_page():
    dist = S.distinctive_tokens("Πεοτρόμπης")                    # ONE misspelt token: only the fuzzy path can match
    assert dist == ["πεοτρομπησ"]
    page = {"title": "Πετρόμπεης Μαυρομιχάλης - Βικιπαίδεια", "body": "", "href": ""}
    assert S.result_on_topic(page, dist) is True
    assert S.result_on_topic({"title": "Doha - Wikipedia", "body": "Qatar's capital", "href": ""}, dist) is False


def test_short_tokens_do_not_fuzzy_match():
    assert S.result_on_topic({"title": "nxabc things", "body": "", "href": ""}, ["nxacc"]) is False   # 5 chars: substring only


def test_bare_rows_are_not_evidence_and_a_url_slug_still_counts():
    assert S._results_are_off_topic(ECKIT, [{"href": "https://x"}, {"href": "https://y"}]) is False
    batch = WINDOWS + [{"title": "", "body": "", "href": "https://example.com/eckit-reduced_gg-spec"}]
    assert S._results_are_off_topic(ECKIT, batch) is False


def test_rank_on_topic_partitions_stably_and_drops_nothing():
    a = {"title": "Windows help", "body": "taskbar tips", "href": "https://a"}
    b = dict(ECKIT_HIT)
    c = {"title": "Another eckit page", "body": "reduced_gg", "href": "https://c"}
    d = {"title": "unrelated", "body": "x", "href": "https://d"}
    out = S.rank_on_topic(ECKIT, [a, b, c, d])
    assert out == [b, c, a, d]
    assert S.rank_on_topic("number one supplement anti aging", [a, b]) == [a, b]   # no distinctive token: untouched


def test_no_distinctive_token_falls_back_to_the_batch_rule():
    q = "number one supplement for anti aging most researched"
    assert S.distinctive_tokens(q) == []
    junk = [{"title": "Doha - Wikipedia", "body": "Doha is Qatar's capital and its fastest growing city, "
             "with over 80% of the nation's population living in it or its suburbs.", "href": "https://w"}]
    assert S._results_are_off_topic(q, junk) is True                     # no content word at all
    assert S._results_are_off_topic(q, [{"title": "Taurine supplements slow aging", "body": "research on supplements and aging", "href": ""}]) is False


def test_the_wave_reranks_its_winner(monkeypatch):
    """The winning batch comes back with the rows that carry a distinctive token first. (§4VA: was an
    AST pin on `valid = rank_on_topic(…)` followed by `return valid`; the wave now returns a `WaveBatch`,
    so the behaviour is pinned instead of the statement.)"""
    import asyncio
    import sys
    from unittest.mock import MagicMock
    other = {"title": "Release notes of another database", "body": "release notes, version 18", "href": "https://other.example/notes"}
    wanted = {"title": "PostgreSQL 18 release notes", "body": "PostgreSQL 18 release notes", "href": "https://db.example/pg18"}
    mod = MagicMock()
    mod.DDGS.return_value.__enter__.return_value.text.side_effect = lambda q, **kw: [dict(other), dict(wanted)]
    monkeypatch.setitem(sys.modules, "ddgs", mod)
    rows = asyncio.run(S._race_search_wave("postgresql 18 release notes", None, 0))
    assert [r["href"] for r in rows] == [wanted["href"], other["href"]]
    assert S.rank_on_topic("postgresql 18 release notes", [other, wanted]) == [wanted, other]
