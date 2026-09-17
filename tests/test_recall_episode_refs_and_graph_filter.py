"""recall: an episode hit offers its answer, and graph edges stay on topic
(2026-09-15, §4HB).

Live (req 6a7882f5): `recall` found the 09-14 turn that had READ the
notification card, at distance 0.20 — the right memory — and rendered its
REQUEST text, because that is all the vector store holds for an episode.
The finding lives in episode 374, and the renderer only ever emitted the
`ep:<id>` drill-down for `source_refs`, never for `episode_id`. Above it,
fifteen graph edges about July news headlines and the operator's family —
matched through the year token — sat at the top of the result. Corpus: 111
of 336 graph edges shown (33%) share no non-numeric content word with
their query.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.tools import memory as mem
from ghost_agent.tools.memory import _graph_edges_on_topic, _graph_query_terms

_Q = ("Revolut September 2026 fraudulent government request email domain "
      "sender address ZachXBT")

# The live graph section, verbatim shape.
_LIVE_EDGES = [
    "- (User) -[INTERACTED_WITH]-> (Assistant) -[PROVIDED]-> (News_Headlines) -[DATED]-> (July 27, 2026) [Score 24]",
    "- (User) -[REQUESTED]-> (Custom Skill) -[HAS_NAME]-> (News_Headlines) -[DATED]-> (July 27, 2026) [Score 11]",
    "- (Ai) -[RESPONDED_TO]-> (User) -[HAS_SON]-> (Leonidas) -[HAS_BIRTHDATE]-> (March 12, 2026) [Score 11]",
    "- (User) -[HAS_SON]-> (Leonidas) -[BORN_ON]-> (March 12, 2026) [Score 9]",
    "- (User) -[HAS_SKILL]-> (News_Headlines) -[DATED]-> (July 27, 2026) [Score 8]",
]


class _Mem:
    def __init__(self, rows):
        self._rows = rows

    def search_advanced(self, query, limit=10):
        return list(self._rows)


class _Graph:
    def __init__(self, edges):
        self._edges = edges
        self.calls = []

    def get_neighborhood(self, words, n):
        self.calls.append(list(words))
        return list(self._edges)


def _recall(rows, graph=None, q=_Q):
    return asyncio.run(mem.tool_recall(q, memory_system=_Mem(rows), graph_memory=graph))


# ------------------------------------------------------------ G2 episode refs

def test_an_episode_hit_offers_its_drilldown_ref():
    """FAILS IF: `episode_id` metadata does not produce an EVIDENCE REFS line.

    The world it fails in is the shipped one: the request text is shown,
    the answer is one unoffered call away.
    """
    rows = [{"score": 0.20, "text": "Find the Revolut customer-notification screenshot …",
             "metadata": {"type": "episode", "episode_id": 374}}]
    out = _recall(rows)
    assert "EVIDENCE REFS: ep:374" in out
    assert "past REQUEST" in out
    # …and the expand affordance that hangs off EVIDENCE REFS is offered.
    assert "knowledge_base(action='expand'" in out


def test_source_refs_still_win_and_are_not_duplicated():
    """FAILS IF: an entry carrying both keys gets two EVIDENCE REFS lines."""
    rows = [{"score": 0.3, "text": "lesson", "metadata": {
        "type": "skill", "episode_id": 9, "source_refs": "ep:12,ep:15"}}]
    out = _recall(rows)
    assert out.count("EVIDENCE REFS:") == 1
    assert "ep:12,ep:15" in out


def test_a_non_episode_hit_without_refs_gets_no_line():
    """FAILS IF: the ref line is emitted unconditionally."""
    rows = [{"score": 0.3, "text": "plain fact", "metadata": {"type": "auto"}}]
    assert "EVIDENCE REFS" not in _recall(rows)


def test_an_empty_episode_id_is_not_rendered():
    """FAILS IF: `episode_id: ""` renders as `ep:` — a ref that expands nothing."""
    rows = [{"score": 0.3, "text": "x", "metadata": {"type": "episode", "episode_id": ""}}]
    assert "ep:" not in _recall(rows)


# ------------------------------------------------------------ G3 graph filter

def test_the_live_graph_section_keeps_one_edge_and_drops_fourteen():
    """FAILS IF: edges are filtered per section, or not at all.

    Exactly one live edge shares a content word with the query
    ("REQUESTED" ⊃ "request"); the family-profile and news-headline edges
    share nothing but the year.
    """
    kept = _graph_edges_on_topic(_Q, _LIVE_EDGES)
    assert len(kept) == 1
    assert "REQUESTED" in kept[0]
    assert not any("Leonidas" in e for e in kept)


def test_a_year_alone_cannot_carry_an_edge():
    """FAILS IF: digit tokens count as content words — the live mechanism."""
    assert "2026" not in _graph_query_terms(_Q)
    only_year = ["- (User) -[HAS_SON]-> (Leonidas) -[BORN_ON]-> (March 12, 2026) [Score 9]"]
    assert _graph_edges_on_topic("meeting notes 2026", only_year) == []


def test_greek_query_matches_greek_edges():
    """FAILS IF: folding is ASCII-only (the §4HA search-floor mistake, again)."""
    q = "Τέμπη ατύχημα αυτοκίνητο νεκροί"
    edges = ["- (Τέμπη) -[SITE_OF]-> (Ατύχημα) -[DATED]-> (2025) [Score 3]",
             "- (User) -[HAS_SON]-> (Leonidas) [Score 9]"]
    kept = _graph_edges_on_topic(q, edges)
    assert len(kept) == 1 and "Τέμπη" in kept[0]


def test_a_query_with_no_content_words_passes_edges_through():
    """FAILS IF: an unjudgeable query drops everything.

    Nothing to compare against means nothing to reject — the tier keeps
    today's behaviour rather than going silent.
    """
    edges = ["- (A) -[B]-> (C) [Score 1]"]
    assert _graph_edges_on_topic("of the and", edges) == edges


def test_the_section_disappears_when_no_edge_survives():
    """FAILS IF: an empty section header is still inserted at the top."""
    rows = [{"score": 0.3, "text": "Revolut breach details", "metadata": {"type": "auto"}}]
    g = _Graph(["- (User) -[HAS_SON]-> (Leonidas) -[BORN_ON]-> (March 12, 2026) [Score 9]"])
    out = _recall(rows, graph=g)
    assert g.calls, "control: the graph tier was consulted"
    assert "TOPOLOGICAL GRAPH EDGES" not in out


def test_an_on_topic_edge_still_leads_the_result():
    """FAILS IF: filtering also moved the section off the top."""
    rows = [{"score": 0.3, "text": "Revolut breach details", "metadata": {"type": "auto"}}]
    g = _Graph(["- (Revolut) -[DISCLOSED]-> (Breach) -[DATED]-> (Sep 12, 2026) [Score 7]"])
    out = _recall(rows, graph=g)
    assert out.index("TOPOLOGICAL GRAPH EDGES") < out.index("SOURCE:")
    assert "(Revolut) -[DISCLOSED]" in out
