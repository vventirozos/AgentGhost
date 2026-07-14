"""Turn-registry req_id collision handling (bug-hunt 2026-07-14).

`req_id` can be a CLIENT-supplied `X-Request-ID` (routes reads the header),
so two overlapping requests can carry the same id. The old registry
overwrote the live entry unconditionally and popped by key on unregister,
so: /api/turns hid the first turn, cancel("X") hit the SECOND turn's flag
(cancelling B killed running A), and A's finally popped B's entry.

Fix: register() never clobbers a live entry (uniquifies the new key);
unregister() is identity-checked.
"""

import asyncio

import pytest

from ghost_agent.core.turns import TurnRegistry


class TestCollision:
    def test_second_register_does_not_clobber_first(self):
        reg = TurnRegistry()
        a = reg.register("X", preview="turn A")
        b = reg.register("X", preview="turn B")
        # Both turns are tracked and visible.
        assert a.req_id == "X"
        assert b.req_id != "X"          # uniquified
        assert b.req_id.startswith("X#")
        ids = {t.req_id for t in reg.list()}
        assert ids == {a.req_id, b.req_id}
        # Each id resolves to its OWN turn.
        assert reg.get(a.req_id) is a
        assert reg.get(b.req_id) is b

    def test_cancel_hits_the_right_turn(self):
        reg = TurnRegistry()
        a = reg.register("X", preview="A")
        b = reg.register("X", preview="B")
        reg.mark_running(a.req_id)
        reg.mark_running(b.req_id)
        # Cancelling B's id must not flag A.
        reg.cancel(b.req_id)
        assert reg.is_cancelled(b.req_id) is True
        assert reg.is_cancelled(a.req_id) is False

    def test_identity_checked_unregister(self):
        reg = TurnRegistry()
        a = reg.register("X")
        b = reg.register("X")
        # A finishes and unregisters with its OWN object — must not evict B.
        reg.unregister(a.req_id, a)
        assert reg.get(b.req_id) is b
        # A stale unregister of B's key with A's object is a no-op.
        reg.unregister(b.req_id, a)
        assert reg.get(b.req_id) is b
        # B unregisters itself correctly.
        reg.unregister(b.req_id, b)
        assert reg.get(b.req_id) is None

    def test_unregister_without_turn_is_unconditional(self):
        # Back-compat: unregister(key) with no identity still pops.
        reg = TurnRegistry()
        reg.register("solo")
        reg.unregister("solo")
        assert reg.get("solo") is None

    def test_no_collision_keeps_plain_id(self):
        reg = TurnRegistry()
        t = reg.register("plain")
        assert t.req_id == "plain"
