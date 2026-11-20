"""§4VD-D — a hyphenated query term is one distinctive token.

Open since §4UN (2026-11-13): "write-ahead" split into "write" + "ahead";
"write" (the lead word) was distinctive alone, so any page saying "write"
was on topic and yahoo's generic results won a deep_research wave. World
where each test fails: the compound is split again, page focus looks for a
joined form the page never contains, or the keyword trim stops seeing the
compound as a must-keep word.
"""
import pytest

from ghost_agent.tools import search as S

Q = "what is write-ahead logging"


def test_the_compound_is_one_distinctive_token():
    assert S.distinctive_tokens(Q) == ["write-ahead"]
    assert "write" not in S.distinctive_tokens("write-ahead log postgres")


@pytest.mark.parametrize("title,on", [
    ("How to write a CV that gets read", False),
    ("Write-Ahead Logging — PostgreSQL documentation", True),
    ("write ahead logging explained", True),
    ("WriteAhead log internals", True),
])
def test_a_page_is_on_topic_only_with_the_compound(title, on):
    assert S.result_on_topic({"title": title}, S.distinctive_tokens(Q)) is on


def test_the_batch_of_generic_write_pages_is_off_topic():
    rows = [{"title": "How to write a CV", "body": "Tips to write a better resume for any job " * 3},
            {"title": "Write for us", "body": "We accept guest posts; write about anything you like " * 3}]
    assert S._results_are_off_topic("sqlite write-ahead checkpoint behaviour", rows)


def test_page_focus_looks_for_the_compound_as_written():
    dist, rest = S.focus_terms(Q)
    assert dist == ["write-ahead"]
    page = ("Intro text. " * 400) + "Write-ahead logging (WAL) records changes before applying them. " \
        + ("Other text. " * 400)
    out = S.focus_page_text(page, (dist, rest), 1200, window=400)
    assert "Write-ahead logging" in out
    # a joined-only form is not in the page: focus would have fallen back to the head
    assert "Write-ahead logging" not in S.focus_page_text(page, (["writeahead"], []), 1200, window=400)


def test_the_keyword_trim_keeps_the_compound():
    q = "how does the sqlite engine handle write-ahead checkpoint starvation under load"
    assert "write-ahead" in S._keyword_trim(q).split()


def test_a_query_without_compounds_is_unchanged():
    assert S.distinctive_tokens("postgres replication lag") == ["postgres"]
    assert S._compound_forms("postgres replication lag") == {}


@pytest.mark.parametrize("q,dist", [
    ("Qwen3.5-9B-A3B model specs context window", ["qwen3", "a3b"]),
    ("COVID-19 vaccine trial results", ["covid"]),
    ("end-to-end encryption in signal", ["end-to-end"]),
])
def test_only_letter_compounds_are_joined(q, dist):
    assert S.distinctive_tokens(q) == dist


@pytest.mark.parametrize("text", ["mp3-player-reviews", "end-to-end-3d", "v2.write-ahead", "x-ray-9"])
def test_a_compound_touching_digits_or_dots_is_not_joined(text):
    assert S._COMPOUND_RE.findall(text) == []


def test_the_keyword_trim_keeps_a_late_compound():
    q = "please explain in some detail how the database engine handles its write-ahead log"
    assert "write-ahead" in S._keyword_trim(q).split()


@pytest.mark.parametrize("q,title,on", [
    ("Greece-Turkey border crossings", "Greece and Turkey border tensions rise", True),
    ("Tesla-Ford merger rumor", "Ford and Tesla deny merger talks", True),
    ("Greece-Turkey border crossings", "Turkey election results", False),
    ("Russia-Ukraine ceasefire talks", "Russia-Ukraine talks resume in Istanbul", True),
])
def test_a_compound_of_names_matches_its_parts_together(q, title, on):
    """Review r2 MAJOR 2: joined-only, "Greece-Turkey" made every "Greece and Turkey" page
    off-topic; one part alone is still not enough."""
    assert S.result_on_topic({"title": title}, S.distinctive_tokens(q)) is on


def test_the_batch_rule_keeps_a_parts_match_on_topic():
    rows = [{"title": "Greece and Turkey border tensions rise", "body": "Officials from Greece and Turkey met " * 4}]
    assert not S._results_are_off_topic("Greece-Turkey border crossings", rows)


def test_a_wiki_title_with_both_parts_carries_the_compound():
    assert S._wiki_token_found("greece-turkey", {"greece", "turkey", "relations"}, "greeceturkeyrelations")
    assert not S._wiki_token_found("write-ahead", {"write", "a", "cv"}, "writeacv")
