"""§4LC (2026-10-03): past requests — what a PUBLIC reply never carries, and
the non-Latin query rule. Each test names the world it fails in. (§4TD: the
lesson tests of this file — writes, retraction, supersession, admission,
credit — went with the lesson store.)"""
import asyncio
import datetime as dt
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.utils.logging import reply_surface_context, request_id_context


def _now(days_ago=0):
    return (dt.datetime.utcnow() - dt.timedelta(days=days_ago)).isoformat() + "Z"


def test_a_mostly_greek_query_is_not_judged_by_the_english_embedder():
    from ghost_agent.memory.request_text import mostly_non_latin as _mostly_non_latin
    assert _mostly_non_latin("Ψάξε στο web για φαρμακεία")
    assert not _mostly_non_latin("search the web for pharmacies")


# ── public replies ───────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_the_owners_digests_never_reach_a_public_reply(monkeypatch, tmp_path):
    """Fails where the "while you were away" digests (project names, waiting
    tasks) opened an owner's channel reply and consumed the DM's watermarks."""
    from tests.test_requester_role import _digest_turn
    t = reply_surface_context.set("public")
    try:
        out, touched = await _digest_turn(monkeypatch, tmp_path, "owner")
    finally:
        reply_surface_context.reset(t)
    assert "OWNER-PROJECT-DIGEST" not in out and "OWNER-ACTIVITY-DIGEST" not in out and touched == []


def test_the_age_check_never_quotes_a_birth_date_into_a_channel():
    from ghost_agent.core.agent import GhostAgent
    pm = MagicMock()
    pm.load.return_value = {"relationships": {"sons": ["Markos (born 2016-11-22)"]}}
    me = SimpleNamespace(context=SimpleNamespace(profile_memory=pm))
    reply = "Markos is 4 years old."
    assert GhostAgent._memory_claim_refutation(me, reply) is not None        # the private control
    t = reply_surface_context.set("public")
    try:
        assert GhostAgent._memory_claim_refutation(me, reply) is None
    finally:
        reply_surface_context.reset(t)


def test_a_dm_and_a_channel_thread_are_different_conversations():
    """Fails where a DM and a channel thread both starting "hi" shared a tag
    — a correction queued for the DM opened the channel reply."""
    from ghost_agent.core.agent import GhostAgent
    msgs = [{"role": "user", "content": "hi"}]
    dm = GhostAgent._conversation_fingerprint(SimpleNamespace(), msgs)
    t = reply_surface_context.set("public")
    try:
        pub = GhostAgent._conversation_fingerprint(SimpleNamespace(), msgs)
    finally:
        reply_surface_context.reset(t)
    assert dm and pub and dm != pub


def _vector_sm(tmp_path, trigger):
    """A playbook with ONE lesson the vector store returns for any query,
    and an embedder that puts every trigger FAR from every query."""
    import numpy as np
    sm = SkillMemory(tmp_path)
    _learn(sm, task=trigger, fix="Run nginx -t before restarting the nginx service.", source="learn_skill")
    vm = MagicMock()
    vm.collection.query.return_value = {"documents": [[trigger]], "distances": [[0.2]],
                                        "metadatas": [[{"type": "skill", "trigger": trigger}]], "ids": [["l1"]]}
    vm.embedding_fn = lambda texts: [np.eye(16)[0 if t == trigger else 1] for t in texts]   # always far
    return sm, vm


