"""§4UK (operator 2026-11-12: "i want all confirmations, everywhere to be removed. i don't wanna be asked for
anything. that's why everything is sandboxed"): knowledge_base forget and reset_all run in the call that asks.
Only the walls on WHO may act stay. Each test names the world it fails in."""
import asyncio
from unittest.mock import MagicMock

import pytest

from ghost_agent.tools import memory as M
from ghost_agent.utils.logging import request_id_context
from tests.test_forget_confirm_r8 import _family, _token, _wipe_store
from tests.test_memory_producers_rereview import _Vec, _edges


def _kb(pm, gm, vec=None, **kw):
    return asyncio.run(M.tool_knowledge_base(memory_system=vec or _Vec(), sandbox_dir=None, profile_memory=pm,
                                             graph_memory=gm, episodic_memory=None, **kw))


@pytest.fixture
def as_rid():
    toks = []

    def _set(rid):
        toks.append(request_id_context.set(rid))
    yield _set
    for t in reversed(toks):
        request_id_context.reset(t)


def test_forget_deletes_at_once_in_the_users_turn(tmp_path, as_rid):
    """Fails in the world where forget still returned a PREVIEW and waited for a yes."""
    as_rid("req-the-user")
    pm, gm = _family(tmp_path)
    out = _kb(pm, gm, action="forget", target="ektoras")
    assert "PREVIEW" not in str(out) and "done" in str(out)
    e = _edges(gm)
    assert ("ektoras", "WORKS_AT", "acme") not in e and ("user", "WORKS_AT", "orbitdata") in e


def test_kept_items_are_listed_and_removable_by_their_token(tmp_path, as_rid):
    """Fails in the world where a related item (an owner fact) went with the default list, or could not be
    removed afterwards."""
    as_rid("req-the-user")
    pm, gm = _family(tmp_path)
    out = _kb(pm, gm, action="forget", target="pista gp rr")
    assert "nothing was deleted" in str(out) and "1. graph user OWNS pista gp rr" in str(out)
    assert ("user", "OWNS", "pista gp rr") in _edges(gm)
    done = _kb(pm, gm, action="forget", confirm=_token(out), items="1")      # same turn: no wait
    assert "✅" in str(done) and ("user", "OWNS", "pista gp rr") not in _edges(gm)


def test_reset_all_wipes_at_once(as_rid):
    """Fails in the world where reset_all still answered with a preview and a token."""
    as_rid("req-the-user")
    vec, gm = _wipe_store(), MagicMock()
    out = _kb(None, gm, vec=vec, action="reset_all")
    assert "Wiped" in str(out) and vec.collection.delete.called and gm.wipe_all.called


@pytest.mark.parametrize("rid", ["job-1", "sched-2", "probe-3", "SYSTEM", ""])
def test_a_turn_that_is_not_the_user_deletes_nothing(tmp_path, as_rid, monkeypatch, rid):
    """Fails in the world where removing the confirmation also let a background, scheduled or probe turn
    forget or wipe the owner's memory."""
    as_rid(rid)
    pm, gm = _family(tmp_path)
    out = _kb(pm, gm, action="forget", target="ektoras")
    assert ("ektoras", "WORKS_AT", "acme") in _edges(gm), out
    vec, gm2 = _wipe_store(), MagicMock()
    out = _kb(None, gm2, vec=vec, action="reset_all")
    assert str(out).startswith("NOT executed") and not vec.collection.delete.called and not gm2.wipe_all.called


def test_a_request_that_read_outside_content_deletes_nothing(tmp_path, as_rid, monkeypatch):
    """Fails in the world where a page the agent read could order a forget or a wipe."""
    import ghost_agent.utils.provenance as PV
    monkeypatch.setattr(PV, "untrusted_seen", lambda rid=None: ["web page"])
    as_rid("req-the-user")
    pm, gm = _family(tmp_path)
    out = _kb(pm, gm, action="forget", target="ektoras")
    assert "outside content" in str(out) and ("ektoras", "WORKS_AT", "acme") in _edges(gm)
    vec, gm2 = _wipe_store(), MagicMock()
    out = _kb(None, gm2, vec=vec, action="reset_all")
    assert "outside content" in str(out) and not vec.collection.delete.called


def test_the_kept_token_after_a_deletion_removes_only_its_own_list(tmp_path, as_rid):
    """Review CRIT: the kept-items token printed after a deletion was the spent one, and the fallback then picked
    ANOTHER forget's kept list — "items='1'" deleted the owner's "user OWNS pista gp rr". Fails in that world."""
    from ghost_agent.utils.logging import conversation_key_context
    ct = conversation_key_context.set("conv-A")
    try:
        pm, gm = _family(tmp_path)
        gm.add_triplets([{"subject": "orbitdata", "predicate": "LOCATED_IN", "object": "athens"}])
        as_rid("req-one")
        _kb(pm, gm, action="forget", target="pista gp rr")                  # leaves a kept list in conv-A
        as_rid("req-two")
        out = str(_kb(pm, gm, action="forget", target="orbitdata"))
        assert "done" in out and "KEPT" in out
        kept_head = out.split("KEPT", 1)[1]
        first = next(l for l in kept_head.splitlines() if l.startswith("1. "))[3:]
        done = str(_kb(pm, gm, action="forget", confirm=_token(kept_head), items="1"))
        assert "✅" in done and first.split(" ", 1)[1] in done
        assert ("user", "OWNS", "pista gp rr") in _edges(gm)                 # the other list is untouched
        spent = str(_kb(pm, gm, action="forget", confirm=_token(kept_head), items="1"))
        assert "✅" not in spent and ("user", "OWNS", "pista gp rr") in _edges(gm)
    finally:
        conversation_key_context.reset(ct)


def test_items_list_only_lists_and_its_token_deletes_the_pick(tmp_path, as_rid):
    """Fails in the world where the only forget deleted the whole default list (a document's file with its
    stored copy) and the "forget the stored copy first" advice could not be followed."""
    as_rid("req-the-user")
    pm, gm = _family(tmp_path)
    out = str(_kb(pm, gm, action="forget", target="ektoras", items="list"))
    assert out.startswith("LIST — nothing was deleted") and ("ektoras", "WORKS_AT", "acme") in _edges(gm)
    assert "No confirmation is needed" in out and "USER confirms" not in out
    done = str(_kb(pm, gm, action="forget", confirm=_token(out), items="all"))
    assert "✅" in done and ("ektoras", "WORKS_AT", "acme") not in _edges(gm)


def test_a_refused_forget_cannot_run_later_with_a_placeholder_token(tmp_path, as_rid, monkeypatch):
    """Review r2 MAJOR: a forget refused because its request read a page stayed planned, and a later clean
    turn's confirm='yes' ran it. Fails in that world."""
    import ghost_agent.utils.provenance as PV
    from ghost_agent.utils.logging import conversation_key_context
    ct = conversation_key_context.set("conv-B")
    try:
        pm, gm = _family(tmp_path)
        monkeypatch.setattr(PV, "untrusted_seen", lambda rid=None: ["web page"] if rid == "req-page" else [])
        as_rid("req-page")
        assert "outside content" in str(_kb(pm, gm, action="forget", target="ektoras"))
        as_rid("req-clean")
        for tok in ("yes", "all", "", "deadbeef"):
            out = str(_kb(pm, gm, action="forget", confirm=tok or None, target=None if tok else "x-none"))
            assert "✅" not in out, (tok, out)
        assert ("ektoras", "WORKS_AT", "acme") in _edges(gm)
    finally:
        conversation_key_context.reset(ct)
