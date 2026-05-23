"""Unit tests for ghost_agent.memory.competence."""

import json
from pathlib import Path

import pytest

from ghost_agent.memory.competence import CompetenceProfile, KNOWN_DOMAINS


@pytest.fixture
def memdir(tmp_path):
    return tmp_path


class TestCompetenceProfile:
    def test_neutral_prior_for_unseen(self, memdir):
        cp = CompetenceProfile(memdir)
        assert cp.estimate("shell") == 0.5

    def test_record_and_estimate(self, memdir):
        cp = CompetenceProfile(memdir)
        for _ in range(8):
            cp.record("shell", "ls", success=True)
        for _ in range(2):
            cp.record("shell", "ls", success=False)
        # 8 successes + 2 failures + Beta(1,1) prior = 9/12 ≈ 0.75
        est = cp.estimate("shell", "ls")
        assert 0.7 < est < 0.8

    def test_tool_fallback_to_domain(self, memdir):
        cp = CompetenceProfile(memdir)
        # Record under "ls" only
        for _ in range(5):
            cp.record("shell", "ls", success=True)
        # Asking about an unseen tool in same domain → domain roll-up
        est = cp.estimate("shell", "grep")
        assert est > 0.5

    def test_domain_fallback_to_global(self, memdir):
        cp = CompetenceProfile(memdir)
        # Populate one domain; ask about another
        for _ in range(5):
            cp.record("shell", "ls", success=True)
        est = cp.estimate("vision", "describe")
        # Should fall back to global, which is shell's history → >0.5
        assert est > 0.5

    def test_observations_count(self, memdir):
        cp = CompetenceProfile(memdir)
        for _ in range(3):
            cp.record("sql", "select", success=True)
        assert cp.observations("sql", "select") == 3
        assert cp.observations("sql") == 3  # roll-up

    def test_persistence_round_trip(self, memdir):
        cp = CompetenceProfile(memdir)
        for _ in range(4):
            cp.record("shell", "ls", success=True)
        cp.record("shell", "ls", success=False)
        # Re-open
        cp2 = CompetenceProfile(memdir)
        assert cp2.observations("shell", "ls") == 5
        assert cp2.estimate("shell", "ls") == pytest.approx(
            cp.estimate("shell", "ls"))

    def test_domain_canonicalisation(self, memdir):
        cp = CompetenceProfile(memdir)
        cp.record("postgres", "select", success=True)
        cp.record("postgresql", "select", success=True)
        # Both should land in "sql"
        assert cp.observations("sql", "select") == 2

    def test_unknown_domain_lands_in_other(self, memdir):
        cp = CompetenceProfile(memdir)
        cp.record("alien", "weird", success=True)
        assert cp.observations("other", "weird") == 1

    def test_by_domain_rollup(self, memdir):
        cp = CompetenceProfile(memdir)
        for _ in range(3):
            cp.record("shell", "ls", success=True)
        for _ in range(2):
            cp.record("sql", "select", success=False)
        roll = cp.by_domain()
        assert "shell" in roll and "sql" in roll
        assert roll["shell"][0] > roll["sql"][0]

    def test_reset(self, memdir):
        cp = CompetenceProfile(memdir)
        cp.record("shell", "ls", success=True)
        cp.reset()
        assert cp.observations("shell", "ls") == 0

    def test_get_context_string_empty(self, memdir):
        assert CompetenceProfile(memdir).get_context_string() == ""

    def test_get_context_string_populated(self, memdir):
        cp = CompetenceProfile(memdir)
        cp.record("shell", "ls", success=True)
        s = cp.get_context_string()
        assert "shell" in s
        assert "n=1" in s

    def test_known_domains_list(self):
        assert "shell" in KNOWN_DOMAINS
        assert "sql" in KNOWN_DOMAINS
        assert "other" in KNOWN_DOMAINS

    def test_corrupted_json_does_not_crash(self, memdir):
        # Pre-seed a corrupted file
        (memdir / "competence_profile.json").write_text("{not json")
        cp = CompetenceProfile(memdir)
        # Should silently load empty and accept new writes
        cp.record("shell", "ls", success=True)
        assert cp.observations("shell", "ls") == 1
