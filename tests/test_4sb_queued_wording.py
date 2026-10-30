"""§4SB — queued-messages (§4RZ) wording and the cancel-stub turn.

1. The owner-fact refusal names a MID-TURN message when there was one
   (the "the user's own message does not state it" wording was false then).
2. The status-only gate's refusal names the instruction the user added
   mid-turn — still a refusal (a mid-turn message never grants).
3. `_persist_session` stores a turn whose reply is ONLY the cancel note as
   the user row alone: history keeps the request, nothing memorises a stub.
"""
import json

import pytest
from fastapi.testclient import TestClient

from ghost_agent.core import queued_messages as qm
from ghost_agent.utils import provenance as PV
from ghost_agent.utils.logging import request_id_context
from tests.test_sessions_and_cancel import _make_app


# --- 1. owner-fact refusal ---------------------------------------------------

@pytest.fixture
def prov():
    PV._reset_for_tests()
    yield PV
    PV._reset_for_tests()


def _refuse(prov, rid, *, added=None):
    prov.note_user_message("look up the wedding page", rid)
    prov.note_untrusted_content("https://wedding.example", rid)
    if added is not None:
        prov.note_added_user_message(added, rid)
    out = prov.refuse_unless_user_said("the fact", "my wife is Mallory", rid)
    assert out is not None and out.reason_code == "untrusted_content"
    return str(out)


def test_owner_fact_refusal_without_a_mid_turn_message_keeps_its_wording(prov):
    text = _refuse(prov, "sb-1")
    assert "the user's own message does not state it" in text
    assert "pages, files or transcripts is not the user" in text
    assert "while the turn ran" not in text


def test_owner_fact_refusal_names_the_mid_turn_message(prov):
    text = _refuse(prov, "sb-2", added="btw my wife is Mallory")
    assert "a message sent while the turn ran cannot grant this" in text
    assert "this request read outside content: https://wedding.example" in text
    assert "send it again as a new message" in text
    # the false claim is gone — the user DID state it, in a mid-turn message
    assert "does not state it" not in text and "they will say so" not in text


def test_an_unrelated_mid_turn_message_keeps_the_page_is_not_the_user_wording(prov):
    """r2 review MAJOR: only an addition that STATES the value is a mid-turn
    statement — otherwise the page's claim was relayed to the owner as a question."""
    text = _refuse(prov, "sb-2b", added="also use metric units")
    assert "pages, files or transcripts is not the user" in text
    assert "while the turn ran" not in text and "send it again" not in text


def test_owner_fact_refusal_reads_the_root_requests_additions(prov):
    # a sub-agent works for its root's user: the root's mid-turn message counts
    prov.link_child("sub-sb-3", "sb-3")
    prov.note_added_user_message("my wife is Mallory", "sb-3")
    text = _refuse(prov, "sub-sb-3")
    assert "while the turn ran" in text


# --- 2. status-only gate -----------------------------------------------------

def _delivered(*texts):
    tok = qm.begin_request()
    for t in texts:
        qm.note_delivered(t)
    return tok


def test_status_only_text_without_additions():
    from ghost_agent.core.agent import status_only_refusal_text
    tok = qm.begin_request()
    try:
        t = status_only_refusal_text()
    finally:
        qm.end_request(tok)
    assert t.startswith("Not run: the user asked for a STATUS / progress report.")
    assert "added" not in t


def test_status_only_text_names_the_newest_addition_cut_to_80():
    from ghost_agent.core.agent import status_only_refusal_text
    long = "also restart it " * 10                          # 160 chars
    tok = _delivered("first one", long)
    try:
        t = status_only_refusal_text()
    finally:
        qm.end_request(tok)
    assert t.startswith("Not run: this request began as a STATUS question, so no new work starts in it")
    assert f"the user added '{long.strip()[:80]}' while it ran" in t
    assert "first one" not in t                              # the NEWEST message
    assert "send that instruction as a new message" in t


async def test_the_gate_still_refuses_after_a_mid_turn_instruction_and_names_it():
    """The real block: deny stays deny (§4RZ s4), the text changes."""
    from ghost_agent.core.strikes import StrikeLedger
    import tests.test_4jj_search_yield_steer as H
    agent = H._agent()
    calls = []

    async def ex(**kw):
        calls.append(kw)
        return "ran"
    agent.available_tools = {"execute": ex}
    rtok = request_id_context.set("own-sb")
    qtok = _delivered("also restart it")
    agent.context._status_only_req = "own-sb"
    try:
        ts = H._ts([("execute", {"command": "systemctl restart x"})], StrikeLedger(), set())
        await agent._dispatch_and_process_tool_batch(ts)
    finally:
        qm.end_request(qtok)
        request_id_context.reset(rtok)
    assert not calls
    tool_rows = [str(m.get("content")) for m in ts.messages if m.get("role") == "tool"]
    assert len(tool_rows) == 1
    assert "began as a STATUS question" in tool_rows[0]
    assert "also restart it" in tool_rows[0]
    assert "the user asked for a STATUS / progress report" not in tool_rows[0]


async def test_the_gate_keeps_the_plain_wording_without_additions():
    from ghost_agent.core.strikes import StrikeLedger
    import tests.test_4jj_search_yield_steer as H
    agent = H._agent()
    agent.available_tools = {"execute": (lambda **kw: None)}
    rtok = request_id_context.set("own-sb2")
    qtok = qm.begin_request()
    agent.context._status_only_req = "own-sb2"
    try:
        ts = H._ts([("execute", {"command": "make"})], StrikeLedger(), set())
        await agent._dispatch_and_process_tool_batch(ts)
    finally:
        qm.end_request(qtok)
        request_id_context.reset(rtok)
    tool_rows = [str(m.get("content")) for m in ts.messages if m.get("role") == "tool"]
    assert tool_rows and "the user asked for a STATUS / progress report" in tool_rows[0]


# --- 3. the cancel-stub turn -------------------------------------------------

CANCEL = "_(Turn cancelled: the user pressed Stop.)_"


def _chat(c, sid, text):
    r = c.post("/api/chat", json={"model": "test-model", "stream": False, "session_id": sid,
                                  "messages": [{"role": "user", "content": text}]})
    assert r.status_code == 200
    return [(m["role"], m["content"]) for m in c.get(f"/api/sessions/{sid}").json()["messages"]]


@pytest.mark.parametrize("reply", [
    CANCEL,
    CANCEL + " No output was produced before the cancellation.",
    "  " + CANCEL + "\n",
    # §4SD r3: the hard limit's stop adds the files written (the state of work keeps them)
    "_(Turn cancelled: this request reached its time limit (30 min).)_ No output was produced before the "
    "cancellation.\n\nFiles written so far: `app.py`, `index.html`.",
])
def test_a_cancel_note_alone_stores_the_request_only(tmp_path, reply):
    async def cancelled(body, bg, request_id=None):
        return (reply, 1, "r")
    app, _ = _make_app(tmp_path, handle_chat=cancelled)
    with TestClient(app) as c:
        assert _chat(c, "s1", "u1") == [("user", "u1")]


def test_a_real_reply_still_stores_both(tmp_path):
    app, _ = _make_app(tmp_path)
    with TestClient(app) as c:
        assert _chat(c, "s1", "hello") == [("user", "hello"), ("assistant", "the reply")]


def test_a_partial_reply_before_the_cancel_is_kept(tmp_path):
    async def partial(body, bg, request_id=None):
        return ("I found two of them.\n\n" + CANCEL, 1, "r")
    app, _ = _make_app(tmp_path, handle_chat=partial)
    with TestClient(app) as c:
        rows = _chat(c, "s1", "u1")
    assert [r for r, _ in rows] == ["user", "assistant"]
    assert rows[1][1].startswith("I found two of them.")


def test_the_next_turn_after_a_cancel_has_no_stub_in_its_history(tmp_path):
    seen = {}

    async def hc(body, bg, request_id=None):
        seen["messages"] = [(m["role"], m["content"]) for m in body["messages"]]
        return (CANCEL if body["messages"][-1]["content"] == "u1" else "the reply", 1, "r")
    app, _ = _make_app(tmp_path, handle_chat=hc)
    with TestClient(app) as c:
        _chat(c, "s1", "u1")
        rows = _chat(c, "s1", "u2")
    # a thin client's second turn: the server supplied history = [u1, u2]
    assert seen["messages"] == [("user", "u1"), ("user", "u2")]
    assert rows == [("user", "u1"), ("user", "u2"), ("assistant", "the reply")]
    assert not any("Turn cancelled" in json.dumps(r) for r in rows)
