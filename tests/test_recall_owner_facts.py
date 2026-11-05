"""§4KX r8 probe follow-up (2026-10-03): recall reaches the owner's facts by
their KIND. The graph tier seeded on NODE names, so "what health conditions do
I have" never reached `user HAS_CONDITION asthma`. Each test names the
world it fails in."""
import asyncio

import pytest

from ghost_agent.memory.graph import GraphMemory
from ghost_agent.tools import memory as M


def _owner_graph(tmp_path):
    g = GraphMemory(tmp_path)
    g.add_triplets([{"subject": "user", "predicate": "HAS_CONDITION", "object": "asthma"},
                    {"subject": "user", "predicate": "TAKES_MEDICATION", "object": "ventolin"},
                    {"subject": "user", "predicate": "HAS_COMPANION", "object": "elena"},
                    {"subject": "user", "predicate": "WORKS_AT", "object": "orbitdata"},
                    {"subject": "user", "predicate": "WORKS_ON", "object": "pinball.html"},
                    {"subject": "user", "predicate": "HAS_SON", "object": "kosmas"},
                    {"subject": "postgres", "predicate": "HAS_CONDITION", "object": "replication lag"}])
    return g


# §4SN: names are shown as stored (lowercase), not re-cased by .title()
@pytest.mark.parametrize("query,expect", [
    ("what health conditions do I have", ["asthma", "ventolin"]),
    ("what medication do I take", ["ventolin"]),
    ("τι φάρμακα παίρνω;", ["ventolin"]),
    ("η υγεία μου", ["asthma"]),
    ("which conditions are on file", ["asthma"]),
    ("who is my companion", ["elena"]),
    ("what is my job", ["orbitdata"]),
    ("tell me about my family", ["kosmas"]),
])
def test_an_owner_fact_is_found_by_its_kind(tmp_path, query, expect):
    out = " ".join(_owner_graph(tmp_path).owner_facts_matching(query))
    assert all(e in out for e in expect)


@pytest.mark.parametrize("query", ["postgres replication lag", "latest news about bitcoin",
                                   "how do I fix the pinball game"])
def test_an_unrelated_question_gets_no_owner_facts(tmp_path, query):
    assert _owner_graph(tmp_path).owner_facts_matching(query) == []


def test_a_category_never_reaches_a_task_in_progress(tmp_path):
    """Fails where "what is my job" listed every WORKS_ON project."""
    assert "pinball" not in " ".join(_owner_graph(tmp_path).owner_facts_matching("what is my job"))


def test_only_the_owners_facts(tmp_path):
    assert "replication" not in " ".join(_owner_graph(tmp_path).owner_facts_matching("health conditions"))


class _NoVec:
    def search_advanced(self, *a, **k):
        return []


def test_recall_reports_the_owners_facts_by_kind(tmp_path):
    """Fails in the world of the live probe: asthma and Ventolin were
    stored and recall never returned them."""
    out = asyncio.run(M.tool_recall("what health conditions and medication do I have", memory_system=_NoVec(),
                                    graph_memory=_owner_graph(tmp_path)))
    assert "asthma" in out and "ventolin" in out
