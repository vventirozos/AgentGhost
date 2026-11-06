"""Provenance links on abstractions (NapMem-inspired, 2026-07-14).

Abstractions previously kept no pointer to the evidence they were generalized
from — a stale or hallucinated abstraction was unfalsifiable after the fact.
Now (§4NM: the dream synthesis / episode-consolidation writers are gone;
lessons come only from ``learn_skill`` and the operator's repairs):

- syntheses carry ``{"provenance": [{"id", "excerpt"}, ...]}`` metadata;
- lessons carry ``source_refs`` (e.g. ``["ep:12", "ep:15"]``), unioned on
  dedup-merge, mirrored onto the vector twin as a comma-joined string;
- ``tool_recall`` surfaces both as EVIDENCE lines so the model (and the
  operator) can drill from an abstraction back to its sources.
"""

import json

import pytest
from unittest.mock import MagicMock



# ----------------------------------------------------------------- lessons


# ------------------------------------------------------------ recall surface


class TestRecallSurfacesEvidence:
    async def test_recall_shows_provenance_and_refs(self):
        from ghost_agent.tools.memory import tool_recall
        vm = MagicMock()
        vm.search_advanced.return_value = [
            {
                "score": 0.2,
                "text": "The operator's deploys fail on permissions.",
                "metadata": {
                    "type": "synthesis",
                    "provenance": json.dumps([
                        {"id": "id1", "excerpt": "deploy v1 failed EACCES"},
                        {"id": "id2", "excerpt": "deploy v2 failed EACCES"},
                    ]),
                },
            },
            {
                "score": 0.25,
                "text": "Always verify the publish path before retrying.",
                "metadata": {"type": "skill", "source_refs": "ep:1,ep:2"},
            },
        ]

        out = await tool_recall(query="deploy failures", memory_system=vm)

        assert "EVIDENCE (synthesized from 2 fragments)" in out
        assert "deploy v1 failed EACCES" in out
        assert "EVIDENCE REFS: ep:1,ep:2" in out
