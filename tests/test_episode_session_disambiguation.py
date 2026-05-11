"""Episodic-memory header must make temporal scope unambiguous.

When the user asks a metacognitive query like "what occurred between my
previous message and this one", `bus.hydrate_context` semantically
retrieves prior-session episodes (URL extractions, weather lookups,
etc.) and prepends them as context. Before this fix the header was a
bare `### RELEVANT PAST EPISODES:` with no temporal marker, so the
model narrated those entries as if they had happened in the current
conversation (observed during the consciousness-probe run on
2026-04-30, self-report turn 2).

The fix is one line: extend the header to explicitly state that the
entries are from prior sessions and are NOT current-conversation
events. Existing pinned-substring tests (`"RELEVANT PAST EPISODES" in
formatted`) still pass because we keep that substring as a prefix.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ghost_agent.memory.episodes import EpisodicMemory


def _ep_memory(tmp_path) -> EpisodicMemory:
    return EpisodicMemory(tmp_path)


def test_header_disambiguates_prior_session_from_current(tmp_path):
    em = _ep_memory(tmp_path)
    em.record_episode(
        trigger="extract text from URL",
        outcome="200 OK, 4kb extracted",
        success=True,
        cluster_id="web",
    )
    eps = em.get_recent_episodes()
    formatted = em.format_for_context(eps)

    # Header must say "prior sessions" and "NOT events in the current
    # conversation" — both phrases are required so the model has no
    # plausible reading that lands these entries inside this turn.
    assert "prior sessions" in formatted
    assert "NOT events in the current conversation" in formatted


def test_legacy_substring_still_present(tmp_path):
    """Three existing tests assert ``"RELEVANT PAST EPISODES" in formatted``;
    keep that exact substring at the front of the header so we don't
    break them."""
    em = _ep_memory(tmp_path)
    em.record_episode(
        trigger="anything",
        outcome="ok",
        success=True,
        cluster_id="x",
    )
    formatted = em.format_for_context(em.get_recent_episodes())
    assert "RELEVANT PAST EPISODES" in formatted
    # And the bus-wiring test asserts "PAST EPISODES" — still satisfied.
    assert "PAST EPISODES" in formatted
