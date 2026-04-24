"""Tests for contradiction resolution logging (#11).

Verifies that:
- ContradictionLog records and retrieves belief revisions
- explain_belief_change searches the log
- Entries are properly persisted and capped
"""

import pytest
import json
from pathlib import Path
from ghost_agent.memory.contradiction_log import ContradictionLog


@pytest.fixture
def contradiction_log(tmp_path):
    return ContradictionLog(tmp_path)


class TestContradictionLog:
    def test_record_creates_entry(self, contradiction_log):
        contradiction_log.record(
            new_fact="User works at CompanyB",
            old_facts=[{"id": "abc", "text": "User works at CompanyA"}],
            deleted_ids=["abc"],
            reason="LLM belief revision"
        )

        recent = contradiction_log.get_recent(limit=1)
        assert len(recent) == 1
        assert recent[0]["new_fact"] == "User works at CompanyB"
        assert len(recent[0]["superseded"]) == 1
        assert recent[0]["deleted_ids"] == ["abc"]

    def test_get_recent_returns_newest_first(self, contradiction_log):
        for i in range(5):
            contradiction_log.record(
                new_fact=f"Fact {i}",
                old_facts=[{"id": f"old{i}", "text": f"Old fact {i}"}],
                deleted_ids=[f"old{i}"]
            )

        recent = contradiction_log.get_recent(limit=3)
        assert len(recent) == 3
        assert recent[0]["new_fact"] == "Fact 4"
        assert recent[2]["new_fact"] == "Fact 2"

    def test_max_entries_enforced(self, contradiction_log):
        for i in range(250):
            contradiction_log.record(
                new_fact=f"Fact {i}",
                old_facts=[],
                deleted_ids=[]
            )

        all_entries = contradiction_log.get_recent(limit=300)
        assert len(all_entries) <= ContradictionLog.MAX_ENTRIES

    def test_explain_belief_change_finds_match(self, contradiction_log):
        contradiction_log.record(
            new_fact="User drives a Tesla Model 3",
            old_facts=[{"id": "car1", "text": "User drives a Toyota Camry"}],
            deleted_ids=["car1"]
        )

        explanation = contradiction_log.explain_belief_change("drives")
        assert "BELIEF REVISION HISTORY" in explanation
        assert "Tesla" in explanation

    def test_explain_belief_change_no_match(self, contradiction_log):
        contradiction_log.record(
            new_fact="User lives in Athens",
            old_facts=[{"id": "loc1", "text": "User lives in London"}],
            deleted_ids=["loc1"]
        )

        explanation = contradiction_log.explain_belief_change("programming")
        assert explanation == ""

    def test_explain_searches_old_facts(self, contradiction_log):
        contradiction_log.record(
            new_fact="User prefers dark mode",
            old_facts=[{"id": "pref1", "text": "User prefers light mode"}],
            deleted_ids=["pref1"]
        )

        # Should find by searching in superseded texts too
        explanation = contradiction_log.explain_belief_change("light mode")
        assert "BELIEF REVISION HISTORY" in explanation

    def test_clear_empties_log(self, contradiction_log):
        contradiction_log.record(
            new_fact="test", old_facts=[], deleted_ids=[]
        )
        assert len(contradiction_log.get_recent()) > 0

        contradiction_log.clear()
        assert len(contradiction_log.get_recent()) == 0

    def test_empty_query_returns_empty(self, contradiction_log):
        assert contradiction_log.explain_belief_change("") == ""

    def test_persistence_across_instances(self, tmp_path):
        log1 = ContradictionLog(tmp_path)
        log1.record(
            new_fact="Persistent fact",
            old_facts=[{"id": "p1", "text": "Old persistent"}],
            deleted_ids=["p1"]
        )

        # Create new instance pointing to same directory
        log2 = ContradictionLog(tmp_path)
        recent = log2.get_recent()
        assert len(recent) == 1
        assert recent[0]["new_fact"] == "Persistent fact"
