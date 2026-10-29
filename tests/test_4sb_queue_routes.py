"""§4SB: the queued-messages ROUTES and REGISTRY after the 2026-10-29
fresh-eye review of §4RZ.

1. a typed "stop" (and the Stop button) interrupts the running tool batch
   BEFORE the cooperative cancel — the flag alone let an `execute` run to
   its budget; the reply lists the interrupted tools;
2. a cancel that found nothing to cancel answers `queue`, not `cancel`;
3. "stop" / interrupt reach EVERY running turn of the sender (a draining
   tail and a newer turn), oldest first;
4. the Stop button holds a waiting yes only when queued messages are ON;
5. the leftovers cap says what it dropped;
6. hand-backs are logged;
7. the inbox strips "\\r" like /api/chat;
+ `TurnRegistry.add_leftovers` for rows a turn drained but never read.
"""

import time

import pytest

from ghost_agent.core import queued_messages as qm
from ghost_agent.core.turns import TurnRegistry
from ghost_agent.utils import tool_stop as TS
from tests.test_queued_messages import _running

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402
from tests.test_queued_messages import _make_app  # noqa: E402


def _tool_running(turn, name="execute"):
    turn.tool_stop, turn.tools_now = TS.ToolStop(), [name]
    turn.tool_stop.enter(0, name)
    return turn.tool_stop


@pytest.fixture
def logged(monkeypatch):
    """Every pretty_log line of the routes and of the registry."""
    lines = []

    def _pl(title, text, *a, **k):
        lines.append((title, text))
    monkeypatch.setattr("ghost_agent.api.routes.pretty_log", _pl)
    monkeypatch.setattr("ghost_agent.utils.logging.pretty_log", _pl)
    return lines


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv("GHOST_QUEUED_MESSAGES", "1")


# ── 1: a stop interrupts the running tool batch ─────────────────────────

class TestStopInterruptsTools:
    def test_typed_stop_interrupts_the_running_tool(self, tmp_path, on, logged):
        app, agent = _make_app(tmp_path)
        reg = agent.turn_registry
        stop = _tool_running(_running(reg))
        with TestClient(app) as c:
            d = c.post("/api/turn/inbox", json={"session_id": "s1", "text": "stop"}).json()
        assert d["path"] == "cancel" and d["request_id"] == "r1"
        assert d["interrupted"] == ["execute"]
        assert stop.requested() and reg.is_cancelled("r1")
        assert any(t == "Queued Message" and "interrupted: execute" in x for t, x in logged)

    def test_typed_stop_with_no_tool_running(self, tmp_path, on):
        app, agent = _make_app(tmp_path)
        _running(agent.turn_registry)
        with TestClient(app) as c:
            d = c.post("/api/turn/inbox", json={"session_id": "s1", "text": "stop"}).json()
        assert d["path"] == "cancel" and d["interrupted"] == []

    @pytest.mark.parametrize("kind", ["owner", "probe"])
    def test_stop_button_interrupts_the_running_tool(self, tmp_path, on, kind):
        app, agent = _make_app(tmp_path)
        reg = agent.turn_registry
        stop = _tool_running(_running(reg, kind=kind), "browser")
        with TestClient(app) as c:
            d = c.post("/api/turn/cancel", json={"request_id": "r1"}).json()
        assert d["cancelled"] and d["interrupted"] == ["browser"]
        assert stop.requested() and reg.is_cancelled("r1")

    def test_stop_button_without_a_tool_lists_nothing(self, tmp_path, on):
        app, agent = _make_app(tmp_path)
        _running(agent.turn_registry)
        with TestClient(app) as c:
            d = c.post("/api/turn/cancel", json={}).json()
        assert d["cancelled"] and d["interrupted"] == []

    def test_stop_button_on_a_member_turn_interrupts_nothing(self, tmp_path, on):
        app, agent = _make_app(tmp_path)
        stop = _tool_running(_running(agent.turn_registry, kind="member"))
        with TestClient(app) as c:
            d = c.post("/api/turn/cancel", json={"request_id": "r1"}).json()
        assert d["cancelled"] and d["interrupted"] == [] and not stop.requested()


# ── 2: nothing cancelled → queue ────────────────────────────────────────

def test_a_turn_gone_between_find_and_cancel_is_queue_not_cancel(tmp_path, on, monkeypatch):
    app, agent = _make_app(tmp_path)
    reg = agent.turn_registry
    _running(reg)
    monkeypatch.setattr(reg, "cancel", lambda *a, **k: {"cancelled": False})
    with TestClient(app) as c:
        d = c.post("/api/turn/inbox", json={"session_id": "s1", "text": "stop"}).json()
    assert d == {"path": "queue", "reason": "the turn is finishing"}


def test_tools_stopped_before_a_turn_vanished_are_still_reported(tmp_path, on, monkeypatch):
    """r2 review: the interrupt ran (the tools ARE stopped) though the cancel found
    nothing — the client must learn that, not re-send "stop" as a request."""
    app, agent = _make_app(tmp_path)
    reg = agent.turn_registry
    turn = _running(reg)
    stop = _tool_running(turn)
    monkeypatch.setattr(reg, "cancel", lambda *a, **k: {"cancelled": False})
    with TestClient(app) as c:
        d = c.post("/api/turn/inbox", json={"session_id": "s1", "text": "stop"}).json()
    assert d["path"] == "queue" and d["interrupted"] == ["execute"] and stop.requested()


# ── 3: every running turn of the sender ─────────────────────────────────

class TestEveryRunningTurn:
    def test_find_running_all_is_oldest_first_and_find_running_the_newest(self):
        reg = TurnRegistry()
        old = _running(reg, "old")
        old.started_at -= 10
        new = _running(reg, "new")
        _running(reg, "p", kind="probe")
        _running(reg, "other", session="s2")
        reg.register("q", session_id="s1", kind="owner")        # waiting, not running
        assert reg.find_running_all("s1", "owner") == [old, new]
        assert reg.find_running("s1", "owner") is new
        assert reg.find_running_all("", "owner") == [] and reg.find_running_all("s9", "owner") == []

    def test_stop_cancels_the_tail_and_the_newer_turn(self, tmp_path, on):
        app, agent = _make_app(tmp_path)
        reg = agent.turn_registry
        tail = _running(reg, "tail")
        tail.started_at -= 10
        reg.post("tail", "a")
        reg.close_inbox("tail", tail)                 # its streamed final is draining
        reg.post("tail", "never")                     # refused: closed
        new = _running(reg, "new")
        reg.post("new", "b")
        stop_tail, stop_new = _tool_running(tail, "web_search"), _tool_running(new, "execute")
        with TestClient(app) as c:
            d = c.post("/api/turn/inbox", json={"session_id": "s1", "text": "stop"}).json()
        assert d["path"] == "cancel" and d["request_id"] == "new"         # r2: the sender's live turn
        assert d["request_ids"] == ["tail", "new"]
        assert d["undelivered"] == ["b"]              # the tail's inbox was already closed → leftovers
        assert d["interrupted"] == ["web_search", "execute"]
        assert stop_tail.requested() and stop_new.requested()
        assert reg.is_cancelled("tail") and reg.is_cancelled("new")
        assert reg.take_leftovers("s1", kind="owner") == ["a"]

    def test_interrupt_reaches_every_running_turn(self, tmp_path, on):
        app, agent = _make_app(tmp_path)
        reg = agent.turn_registry
        tail = _running(reg, "tail")
        tail.started_at -= 10
        new = _running(reg, "new")
        s1, s2 = _tool_running(tail, "browser"), _tool_running(new, "execute")
        with TestClient(app) as c:
            r = c.post("/api/turn/interrupt", json={"session_id": "s1"})
        assert r.status_code == 200
        assert r.json() == {"request_id": "new", "interrupted": ["browser", "execute"]}
        assert s1.requested() and s2.requested()
        assert not reg.is_cancelled("tail") and not reg.is_cancelled("new")


# ── 4: the Stop button holds only when queued messages are on ───────────

@pytest.mark.parametrize("flag,held", [("1", True), (None, False)])
def test_stop_button_holds_a_waiting_yes_only_when_on(tmp_path, monkeypatch, flag, held):
    from ghost_agent.tools import memory as M
    if flag is None:
        monkeypatch.delenv("GHOST_QUEUED_MESSAGES", raising=False)
    else:
        monkeypatch.setenv("GHOST_QUEUED_MESSAGES", flag)
    monkeypatch.setattr(M, "_HOLDS", {})
    monkeypatch.setattr(M, "_REQ_START", {})
    monkeypatch.setattr(M, "_HANDED_BACK", {})
    app, agent = _make_app(tmp_path)
    _running(agent.turn_registry)
    M.note_request_start("Y", "s1", "owner")          # a yes waiting on the turn lock
    time.sleep(0.01)
    with TestClient(app) as c:
        assert c.post("/api/turn/cancel", json={"request_id": "r1"}).json()["cancelled"]
    assert ("s1" in M._HOLDS) is held
    assert M._held_since_start("Y") is held


# ── 5 + 6: the leftovers cap says what it dropped; hand-backs are logged ─

class TestLeftoverLog:
    def test_the_cap_names_the_dropped_count(self, logged):
        reg = TurnRegistry()
        t1 = _running(reg, "first")
        for k in range(qm.MAX_PENDING):
            reg.post("first", f"m{k}")
        reg.unregister("first", t1)
        t2 = _running(reg, "second")
        reg.post("second", "x")
        reg.post("second", "y")
        reg.unregister("second", t2)
        texts = [x for t, x in logged if t == "Queued Message"]
        assert f"[first] {qm.MAX_PENDING} unread → leftovers" in texts
        assert "[second] 2 unread → leftovers" in texts
        assert "[second] 2 older unread message(s) dropped (leftovers full)" in texts
        assert not any("dropped" in x and x.startswith("[first]") for x in texts)
        got = reg.take_leftovers("s1", kind="owner")
        assert len(got) == qm.MAX_PENDING and got[-2:] == ["x", "y"]

    def test_the_leftovers_route_logs_the_hand_back(self, tmp_path, on, logged):
        app, agent = _make_app(tmp_path)
        reg = agent.turn_registry
        t = _running(reg)
        reg.post("r1", "a")
        reg.post("r1", "b")
        reg.unregister("r1", t)
        with TestClient(app) as c:
            assert c.post("/api/turn/inbox/leftovers", json={"session_id": "s1"}).json() == {"messages": ["a", "b"]}
            assert c.post("/api/turn/inbox/leftovers", json={"session_id": "s1"}).json() == {"messages": []}
        texts = [x for t, x in logged if t == "Queued Message"]
        assert sum(1 for x in texts if x.startswith("handed back 2 to the client")) == 1
        assert not any(x.startswith("handed back 0") for x in texts)

    @pytest.mark.parametrize("via", ["inbox", "button"])
    def test_a_cancels_undelivered_is_logged(self, tmp_path, on, logged, via):
        app, agent = _make_app(tmp_path)
        reg = agent.turn_registry
        _running(reg)
        reg.post("r1", "pending")
        with TestClient(app) as c:
            if via == "inbox":
                d = c.post("/api/turn/inbox", json={"session_id": "s1", "text": "stop"}).json()
            else:
                d = c.post("/api/turn/cancel", json={"request_id": "r1"}).json()
        assert d["undelivered"] == ["pending"] and "undelivered_items" not in d
        assert ("Queued Message", "[r1] 1 undelivered handed back to the client (cancel)") in logged

    def test_a_cancel_with_nothing_undelivered_logs_no_hand_back(self, tmp_path, on, logged):
        app, agent = _make_app(tmp_path)
        _running(agent.turn_registry)
        with TestClient(app) as c:
            c.post("/api/turn/cancel", json={"request_id": "r1"})
        assert not any("handed back" in x for _, x in logged)


# ── 7: the inbox strips \r ──────────────────────────────────────────────

def test_the_inbox_strips_carriage_returns(tmp_path, on):
    app, agent = _make_app(tmp_path)
    reg = agent.turn_registry
    _running(reg)
    with TestClient(app) as c:
        assert c.post("/api/turn/inbox", json={"session_id": "s1", "text": "use\r\nParis\r"}).json()["path"] == "inject"
        assert [m["text"] for m in reg.drain("r1")] == ["use\nParis"]
        assert c.post("/api/turn/inbox", json={"session_id": "s1", "text": "stop\r"}).json()["path"] == "cancel"
    assert reg.is_cancelled("r1")


# ── add_leftovers: drained-but-unread rows ──────────────────────────────

class TestAddLeftovers:
    def test_two_items_kept_even_for_a_cancelled_turn(self, logged):
        reg = TurnRegistry()
        _running(reg)
        reg.cancel("r1")
        now = time.time()
        assert reg.add_leftovers("s1", "owner", [{"text": "a", "ts": now - 2}, {"text": "b", "ts": now - 1}]) == 2
        reg.unregister("r1")
        items = reg.take_leftover_items("s1", kind="owner")
        assert [m["text"] for m in items] == ["a", "b"]
        assert items[0]["ts"] == now - 2 and all(m["kind"] == "owner" for m in items)
        assert ("Queued Message", "[s1] 2 unread → leftovers") in logged

    def test_the_cap_is_respected_and_said(self, logged):
        reg = TurnRegistry()
        now = time.time()
        reg.add_leftovers("s1", "owner", [{"text": f"m{k}", "ts": now} for k in range(qm.MAX_PENDING)])
        assert reg.add_leftovers("s1", "owner", [{"text": "x", "ts": now}, {"text": "y", "ts": now}]) == 2
        got = reg.take_leftovers("s1", kind="owner")
        assert len(got) == qm.MAX_PENDING and got[-2:] == ["x", "y"] and "m0" not in got
        assert ("Queued Message", "[s1] 2 older unread message(s) dropped (leftovers full)") in logged
        # more than the cap in one call: only the newest MAX_PENDING are kept
        n = reg.add_leftovers("s2", "owner", [{"text": f"z{k}", "ts": now} for k in range(qm.MAX_PENDING + 3)])
        assert n == qm.MAX_PENDING
        assert len(reg.take_leftovers("s2", kind="owner")) == qm.MAX_PENDING

    def test_kind_filters_on_take(self):
        reg = TurnRegistry()
        now = time.time()
        reg.add_leftovers("s1", "owner", [{"text": "mine", "ts": now}])
        reg.add_leftovers("s1", "probe", [{"text": "theirs", "ts": now}])
        assert reg.take_leftovers("s1", kind="member") == []
        assert reg.take_leftovers("s1", kind="owner") == ["mine"]
        assert reg.take_leftovers("s1", kind="probe") == ["theirs"]
        assert reg.take_leftovers("s1", kind="probe") == []

    def test_nothing_to_add_adds_nothing(self):
        reg = TurnRegistry()
        assert reg.add_leftovers("", "owner", [{"text": "a", "ts": 1.0}]) == 0
        assert reg.add_leftovers("s1", "owner", []) == 0
        assert reg.add_leftovers("s1", "owner", [{"text": "  ", "ts": 1.0}, "junk"]) == 0
        assert reg._leftovers == {}

    def test_expired_rows_are_pruned_and_sessions_bounded(self, monkeypatch):
        reg = TurnRegistry()
        reg.add_leftovers("old", "owner", [{"text": "stale", "ts": time.time() - qm.LEFTOVER_TTL_S - 1}])
        for i in range(qm.LEFTOVER_MAX_SESSIONS + 2):
            reg.add_leftovers(f"s{i}", "owner", [{"text": "x", "ts": time.time()}])
        assert "old" not in reg._leftovers
        assert len(reg._leftovers) == qm.LEFTOVER_MAX_SESSIONS
        assert reg.take_leftovers("s0", kind="owner") == []            # oldest session evicted
