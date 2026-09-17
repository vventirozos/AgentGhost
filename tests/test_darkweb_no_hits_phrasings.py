"""A zero-hit answer is not a broken engine (2026-09-15, §4HB).

torch runs Xapian Omega, whose zero-hit page reads "No documents match your
query". `_NO_HITS_RE` knew "no results", "no matches", "nothing found" —
not that — so every torch miss fell through to the last branch and was
reported at WARNING as "the engine's result format may have drifted".
The fixture is the real page, fetched over Tor on 2026-09-15 (1,848 bytes;
the anti-scraping form token scrubbed).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.tools.darkweb_search import _NO_HITS_RE, _diagnose_empty_body

_FIX = Path(__file__).resolve().parent / "fixtures" / "torch_zero_hits_2026-09-15.html"
_TORCH = ("http://xmh57jrknzkhv6y3ls3ubitzfqnkrwxhopf5aygthi7d6rplyvk3noyd.onion"
          "/cgi-bin/omega/omega?P=x&HITSPERPAGE=100")


def test_the_real_torch_miss_is_a_no_hits_answer():
    """FAILS IF: Omega's phrasing is absent — the shipped world, where this
    exact page was 'parser … format may have drifted' at WARNING.
    """
    body = _FIX.read_text(encoding="utf-8")
    kind, msg = _diagnose_empty_body(body, _TORCH, _TORCH)
    assert kind == "no-hits", (kind, msg)
    assert "drifted" not in msg


@pytest.mark.parametrize("phrase", [
    "No documents match your query",
    "no document matches",
    "NO DOCUMENTS MATCH",
])
def test_omega_phrasings(phrase):
    """FAILS IF: the pattern is anchored to one casing or one number."""
    assert _NO_HITS_RE.search(phrase)


def test_the_previous_phrasings_still_match():
    """FAILS IF: widening the pattern narrowed something that worked."""
    for p in ("no results", "No matches", "nothing was found", "0 results found",
              "could not find", "didn't find"):
        assert _NO_HITS_RE.search(p), p


def test_a_genuinely_unparseable_page_is_still_reported_as_parser():
    """FAILS IF: the no-hits branch swallows real drift.

    A page with results-looking markup and none of the zero-hit wording
    must keep reaching the last branch — that IS the drift signal.
    """
    body = "<html><body><div class='result'>something new</div></body></html>" * 30
    kind, _ = _diagnose_empty_body(body, _TORCH, _TORCH)
    assert kind == "parser"


def test_documents_match_in_a_positive_sentence_is_not_a_miss():
    """FAILS IF: the pattern fires on 'documents match' without the 'no'."""
    assert not _NO_HITS_RE.search("All 3 documents match your query")
