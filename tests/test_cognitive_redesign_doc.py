"""Guard: the cognitive-layer redesign rationale must not silently vanish
again (IMPROVEMENTS.md #25).

The original COGNITIVE_LAYER_REDESIGN.md was lost with no VCS; core/agent.py
comments and docs still cite it. This pins that the doc exists and that the
toggle constants it documents match their default-OFF state, so the doc and
code can't drift apart.
"""
from pathlib import Path

import ghost_agent.core.agent as agent_mod


_REPO = Path(__file__).resolve().parents[1]


def test_redesign_doc_exists_and_covers_toggles():
    doc = _REPO / "COGNITIVE_LAYER_REDESIGN.md"
    assert doc.exists(), "COGNITIVE_LAYER_REDESIGN.md was lost again"
    text = doc.read_text()
    for tog in ("_MCTS_TURNSTART_ENABLED", "_SELFHOOD_PREFIX_ENABLED",
                "_METACOG_ARBITER_ENABLED"):
        assert tog in text, f"redesign doc must document {tog}"
    # Re-enable criteria must be present (not just a status dump).
    assert "Re-enable criteri" in text


def test_off_toggles_match_documented_default():
    # The doc describes these as OFF on the request path.
    assert agent_mod._MCTS_TURNSTART_ENABLED is False
    assert agent_mod._SELFHOOD_PREFIX_ENABLED is False
    assert agent_mod._METACOG_ARBITER_ENABLED is False
    # The grounded hypothesis loop is the one kept ON.
    assert agent_mod._HYPOTHESIS_GROUNDING_ENABLED is True
