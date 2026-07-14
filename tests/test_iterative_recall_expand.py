"""Iterative recall: evidence drill-down + recall-routing fix (2026-07-14).

Generalizes tool_query_document's "read → refine → read again" loop to the
memory stores:

- `knowledge_base(action='expand', ref='ep:12' | 'session:<id>')` resolves an
  EVIDENCE REF (from task-3 provenance) to the raw record — full episode with
  its action chain, or a stored conversation's tail.
- `tool_recall` output now carries an expand TIP when hits have evidence, and
  a retry-with-different-wording hint on zero results.
- journal §4C routing variance: a `manage_projects` `get` miss runs a memory
  recall before giving up, returning hits in a NON-error payload ("when does
  project Kestrel ship?" previously dead-ended on `project not found` while
  the answer sat in vector memory).
"""

import json
from types import SimpleNamespace

import pytest
from unittest.mock import MagicMock

from ghost_agent.memory.episodes import EpisodicMemory
from ghost_agent.core.sessions import SessionStore
from ghost_agent.tools.memory import tool_expand_evidence, tool_knowledge_base, tool_recall
from ghost_agent.tools.projects import tool_manage_projects


@pytest.fixture
def epi(tmp_path):
    e = EpisodicMemory(tmp_path)
    e.record_episode(
        trigger="deploy chess-v4 failed on permissions",
        context="sandbox publish",
        actions=[
            {"tool": "execute", "args": {"cmd": "cp"}, "result": "EACCES", "success": False},
            {"tool": "execute", "args": {"cmd": "sudo cp"}, "result": "ok", "success": True},
        ],
        outcome="recovered after path fix",
        success=True,
        lesson="check publish path first",
        cluster_id="deploy",
    )
    return e


class TestExpandEvidence:
    async def test_episode_ref_resolves_full_record(self, epi):
        out = await tool_expand_evidence(ref="ep:1", episodic_memory=epi)
        assert "EPISODE 1 [deploy]" in out
        assert "TRIGGER: deploy chess-v4 failed on permissions" in out
        assert "[FAILED]" in out and "→ [ok]" in out  # action chain detail
        assert "LESSON: check publish path first" in out

    async def test_missing_episode(self, epi):
        out = await tool_expand_evidence(ref="ep:999", episodic_memory=epi)
        assert "no longer exists" in out

    async def test_malformed_ref(self, epi):
        out = await tool_expand_evidence(ref="ep:banana", episodic_memory=epi)
        assert "malformed" in out

    async def test_session_ref_resolves_tail(self, tmp_path):
        store = SessionStore(tmp_path / "sessions")
        store.append_turn("sessabc", [{"role": "user", "content": "plan the deploy"}],
                          "Deploy planned for friday.")
        out = await tool_expand_evidence(ref="session:sessabc", session_store=store)
        assert "SESSION sessabc" in out
        assert "Deploy planned for friday." in out

    async def test_unknown_scheme_and_missing_ref(self):
        assert "unknown ref scheme" in await tool_expand_evidence(ref="frag:x")
        assert "MANDATORY" in await tool_expand_evidence()

    async def test_dispatcher_routes_expand(self, epi):
        out = await tool_knowledge_base(action="expand", ref="ep:1",
                                        episodic_memory=epi)
        assert "EPISODE 1" in out


class TestRecallIterationHints:
    async def test_zero_results_suggests_retry(self):
        vm = MagicMock()
        vm.search_advanced.return_value = []
        out = await tool_recall(query="nonexistent topic", memory_system=vm)
        assert "ONE more recall with" in out

    async def test_evidence_hits_get_expand_tip(self):
        vm = MagicMock()
        vm.search_advanced.return_value = [{
            "score": 0.5,
            "text": "Always verify the publish path.",
            "metadata": {"type": "skill", "source_refs": "ep:1"},
        }]
        out = await tool_recall(query="publish path", memory_system=vm)
        assert "EVIDENCE REFS: ep:1" in out
        assert "action='expand'" in out

    async def test_plain_hits_get_no_tip(self):
        vm = MagicMock()
        vm.search_advanced.return_value = [{
            "score": 0.5, "text": "The user lives in Athens.", "metadata": {},
        }]
        out = await tool_recall(query="where does the user live", memory_system=vm)
        assert "action='expand'" not in out


class TestProjectRecallFallback:
    def _context(self, tmp_path, hits):
        from ghost_agent.memory.projects import ProjectStore
        vm = MagicMock()
        vm.search_advanced.return_value = hits
        return SimpleNamespace(
            project_store=ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb"),
            memory_system=vm,
            current_project_id=None,
            scratchpad=None,
            graph_memory=None,
            contradiction_log=None,
        )

    async def test_get_miss_with_memory_hits_returns_facts(self, tmp_path):
        ctx = self._context(tmp_path, [
            {"score": 0.76, "text": "Project Kestrel ships in September.", "metadata": {}},
        ])
        out = await tool_manage_projects(ctx, action="get", project_id="kestrel")
        assert not out.startswith("ERROR")
        payload = json.loads(out)
        assert payload["project"] is None
        assert "Kestrel ships in September" in payload["memory_recall"][0]

    async def test_get_miss_without_hits_redirects_to_recall(self, tmp_path):
        ctx = self._context(tmp_path, [])
        out = await tool_manage_projects(ctx, action="get", project_id="kestrel")
        assert out.startswith("ERROR")
        assert "recall" in out

    async def test_low_relevance_hits_are_not_attached(self, tmp_path):
        ctx = self._context(tmp_path, [
            {"score": 1.4, "text": "unrelated noise", "metadata": {}},
        ])
        out = await tool_manage_projects(ctx, action="get", project_id="kestrel")
        assert out.startswith("ERROR")
