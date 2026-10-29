"""§4RZ step 6a — the web client of queued messages.

The interface proxies the three agent routes (inbox, leftovers, tool
interrupt) and the page acts on the inbox's answer. The page's handlers are
EXECUTED under node against a small fake DOM (tests/helpers.eval_js), not
only matched as text.

World where each test fails: a proxy route is missing, unauthenticated, or
masks the agent's status (a 409 "nothing running" read as 200); the page
sends a mid-turn message to /api/chat instead of the inbox; a message the
turn read is shown twice or out of place; a leftover is dropped; a Stop
re-sends what the user stopped; the feature being off loses the text.
"""
import json
import os
import re
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("GHOST_API_KEY", "test-ghost-key")

import interface.server as server  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from tests.helpers import eval_js, extract_js_function, strip_js_comments  # noqa: E402

STATIC = _ROOT / "interface" / "static"
APP = (STATIC / "app.js").read_text(encoding="utf-8")
INDEX = (STATIC / "index.html").read_text(encoding="utf-8")
CSS = (STATIC / "style.css").read_text(encoding="utf-8")

ROUTES = ["/api/turn/inbox", "/api/turn/inbox/leftovers", "/api/turn/interrupt"]


# ── the proxy routes ─────────────────────────────────────────────────────

def _fake_upstream(status_code=200, payload=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json = MagicMock(return_value=payload if payload is not None else {"ok": True})
    resp.text = "raw"
    client = MagicMock()
    client.request = AsyncMock(return_value=resp)
    return client


@pytest.mark.parametrize("path", ROUTES)
def test_each_route_requires_the_interface_key(path):
    fake = _fake_upstream()
    with patch.object(server, "_get_http_client", return_value=fake):
        r = TestClient(server.app).post(path, json={"session_id": "web-s1"})
    assert r.status_code == 401
    fake.request.assert_not_called()


@pytest.mark.parametrize("path", ROUTES)
def test_each_route_forwards_the_body_verbatim_to_the_same_agent_path(path):
    fake = _fake_upstream(payload={"path": "inject", "request_id": "abcd1234"})
    body = b'{"session_id": "web-s1", "text": "use metric units"}'
    with patch.object(server, "_get_http_client", return_value=fake):
        r = TestClient(server.app).post(
            path, content=body,
            headers={"Content-Type": "application/json", "X-Ghost-Key": server.GHOST_API_KEY})
    assert r.status_code == 200
    args, kwargs = fake.request.call_args
    assert args[0] == "POST" and args[1] == f"http://localhost:8000{path}"
    assert kwargs["content"] == body
    assert kwargs["headers"]["X-Ghost-Key"] == server.GHOST_API_KEY
    # the sender is the owner, as on /api/chat: no requester/origin header
    assert "X-Ghost-Requester" not in kwargs["headers"]
    assert "X-Ghost-Origin" not in kwargs["headers"]


@pytest.mark.parametrize("status,payload", [
    (409, {"request_id": "abcd1234", "interrupted": [], "reason": "no tool is running"}),
    (404, {"interrupted": [], "reason": "queued messages are off"}),
    (200, {"request_id": "abcd1234", "interrupted": ["execute"]}),
])
def test_interrupt_status_and_body_pass_through(status, payload):
    fake = _fake_upstream(status_code=status, payload=payload)
    with patch.object(server, "_get_http_client", return_value=fake):
        r = TestClient(server.app).post(
            "/api/turn/interrupt", json={"session_id": "web-s1"},
            headers={"X-Ghost-Key": server.GHOST_API_KEY})
    assert r.status_code == status
    assert r.json() == payload


def test_an_unreachable_agent_is_an_error_not_a_path():
    """The page reads a non-2xx as "queue it locally"; a proxy failure
    must not come back as a 200 the page could read as an answer."""
    client = MagicMock()
    client.request = AsyncMock(side_effect=ConnectionError("refused"))
    with patch.object(server, "_get_http_client", return_value=client):
        r = TestClient(server.app).post(
            "/api/turn/inbox", json={"session_id": "web-s1", "text": "x"},
            headers={"X-Ghost-Key": server.GHOST_API_KEY})
    assert r.status_code >= 500
    assert "path" not in r.json()


def test_routes_sit_in_the_authenticated_proxy_block():
    """No key (or a wrong one) → refused before anything reaches the agent."""
    client = MagicMock()
    client.request = AsyncMock(side_effect=AssertionError("reached the agent"))
    with patch.object(server, "_get_http_client", return_value=client):
        tc = TestClient(server.app)
        for path in ROUTES:
            for headers in ({}, {"X-Ghost-Key": "wrong"}):
                r = tc.post(path, json={"session_id": "web-s1", "text": "x"}, headers=headers)
                assert r.status_code in (401, 403), (path, headers, r.status_code)
    client.request.assert_not_called()


# ── the page, executed ───────────────────────────────────────────────────

_FAKE_DOM = r"""
class CL {
  constructor() { this.s = new Set(); }
  add(...c) { c.forEach(x => this.s.add(x)); }
  remove(...c) { c.forEach(x => this.s.delete(x)); }
  contains(c) { return this.s.has(c); }
  toggle(c, on) { if (on === undefined ? !this.s.has(c) : on) this.s.add(c); else this.s.delete(c); }
}
class El {
  constructor() { this.children = []; this.dataset = {}; this.classList = new CL(); this.parentNode = null;
                  this.textContent = ''; this.style = {}; this.disabled = false; }
  set className(v) { this.classList = new CL(); String(v).split(/\s+/).filter(Boolean).forEach(c => this.classList.add(c)); }
  set innerHTML(v) { this.textContent = v; this.children.slice().forEach(c => c.remove()); }
  appendChild(c) { if (c.parentNode) c.parentNode._rm(c); this.children.push(c); c.parentNode = this; return c; }
  insertBefore(c, ref) { if (c.parentNode) c.parentNode._rm(c); const i = this.children.indexOf(ref);
                         this.children.splice(i < 0 ? this.children.length : i, 0, c); c.parentNode = this; return c; }
  _rm(c) { const i = this.children.indexOf(c); if (i >= 0) this.children.splice(i, 1); c.parentNode = null; }
  remove() { if (this.parentNode) this.parentNode._rm(this); }
  querySelector(sel) { return this.children.find(c => c.classList.contains('queued-label')) || null; }
}
const document = { createElement: () => new El() };
const chatLog = new El();
const chatInput = { value: '', style: {}, placeholder: '', dispatchEvent() {}, selectionStart: 0, selectionEnd: 0,
                    setSelectionRange(a, b) { this.selectionStart = a; this.selectionEnd = b; } };
const store = {};
const localStorage = { getItem: (k) => (k in store ? store[k] : null), setItem: (k, v) => { store[k] = String(v); },
                       removeItem: (k) => { delete store[k]; }, key: (i) => Object.keys(store)[i] ?? null,
                       get length() { return Object.keys(store).length; } };
// the tray as stored: one localStorage key per item (r6a10), grouped by session
const stored = () => { const out = {};
  for (const k of Object.keys(store)) if (k.startsWith('ghost.queued.restore.item.')) {
    const it = JSON.parse(store[k]); (out[it.sid || ''] = out[it.sid || ''] || []).push(it); }
  for (const k of Object.keys(out)) out[k].sort((a, b) => a.at - b.at);
  return out; };
const sstore = {};
const sessionStorage = { getItem: (k) => (k in sstore ? sstore[k] : null), setItem: (k, v) => { sstore[k] = String(v); },
                         removeItem: (k) => { delete sstore[k]; } };
const interruptBtn = new El(); interruptBtn.classList.add('hidden');
const restoreTray = new El(); restoreTray.classList.add('hidden');
// the hand-back tray of the session on screen (r6a7): what the owner can Insert
const tray = () => restoreTray.children.filter(r => r.classList.contains('restore-item')).map(r => r.children[0].textContent);
const resyncs = [];
const window = { __ghostSessionId: 'web-s1',
                 GhostCore: { events: { dispatchEvent(e) { resyncs.push(e.type); } } } };
const storedSessionId = () => null;
const notes = [];
function addMessage(role, text) { const d = new El(); d.className = `message ${role}`; d.textContent = text;
                                  chatLog.appendChild(d); notes.push(text); return d; }
const renderMarkdown = (t) => t;
const dismissEmptyStateHero = () => {}, scrollToBottomIfPinned = () => {};
let renders = 0;
function renderHistoryToLog(h) { renders++; chatLog.children.slice().forEach(c => c.remove());
  h.forEach(m => { const d = new El(); d.className = `message ${m.role === 'assistant' ? 'agent' : m.role}`;
                   d.textContent = m.content; chatLog.appendChild(d); }); }
const mergeClientLabelKeys = (x) => x;
const saveChatState = () => {};
let toggled = 0; const toggleSendButtonUI = () => { toggled++; };
const sent = []; function sendMessage(resume, text) { sent.push(text); }
let isProcessingRequest = true;
let aborted = 0;
let currentChatController = { abort() { aborted++; } };
let chatHistory = [];
let currentAgentMessageDiv = null;
const calls = [];
let routes = {};
let held = {};       // url -> release functions: the answer waits until released
globalThis.fetch = (url, opts) => {
  calls.push({ url, body: opts && opts.body ? JSON.parse(opts.body) : null });
  const answer = () => {
    const r = routes[url];
    if (!r) throw new TypeError('Failed to fetch');
    const [status, body] = typeof r === 'function' ? r() : r;
    return { ok: status >= 200 && status < 300, status, json: async () => body };
  };
  if (held[url]) return new Promise((res, rej) => held[url].push(() => { try { res(answer()); } catch (e) { rej(e); } }));
  return (async () => answer())();
};
const hold = (url) => { held[url] = []; };
const tick = async () => { for (let i = 0; i < 10; i++) await new Promise(r => setTimeout(r, 0)); };
const release = async (url) => { const fns = held[url] || []; delete held[url]; fns.forEach(f => f()); await tick(); };
"""


def _state_block() -> str:
    m = re.search(r"// §4RZ step 6a — messages sent while a turn runs.*?\n(let _queuedInjects.*?let _leftoversOwed = null;[^\n]*\n)",
                  APP, re.S)
    assert m, "the §4RZ client state block is missing"
    consts = [re.search(rf"^const {n} = [^\n]*", APP, re.M)
              for n in ("INBOX_TIMEOUT_MS", "STOP_SETTLE_TIMEOUT_MS", "LEFTOVER_TTL_MS", "_OWED_KEY", "_CANCEL_NOTE_RE",
                        "RESTORE_TTL_MS", "_RESTORE_ITEM_PREFIX", "_RESTORE_KEY", "_RESTORE_UNSAVED_KEY",
                        "_RESTORE_CONFIRMED_KEY")]
    assert all(consts), "a §4RZ client constant is missing"
    return m.group(1) + "".join(c.group(0) + "\n" for c in consts)


_FUNCS = ["_msgKey", "_lcsPairs", "_reconcileWithLocal", "_adoptionChanges", "toWireMessage",
          "_queuedSid", "_setQueuedFeature", "_isOffReason", "_beginQueuedTurn", "_settleLostTurns",
          "_setQueuedLabel", "_queuedBubble", "_restoreId", "_restoreValid", "_confirmedIds", "_confirm", "_storedItems", "_unsavedList",
          "_saveUnsaved", "_storeItem", "_legacyId", "_migrateRestores", "_restoreMap",
          "_restoreToComposer", "_trayRemove", "_trayInsert", "_onSessionShown", "_renderTray", "_owedMap", "_saveOwed",
          "_markOwed", "_clearOwed", "_isCleanReply", "_reattachLabelKeys", "_fetchLeftovers", "_queueOwedLeftovers",
          "_collectOwedLeftovers", "_resyncFromSession", "_requestSessionResync", "_holdForAfterTurn",
          "_sendWhileBusy", "_settleQueuedTurn", "_settleQueuedTurnNow", "_drainSendAfterTurn",
          "interruptRunningTool"]

S = "/api/sessions/web-s1"
L = "/api/turn/inbox/leftovers"


def _harness(body: str) -> str:
    return (_FAKE_DOM + _state_block()
            + "\n".join(extract_js_function(APP, f) for f in _FUNCS) + "\n"
            # a turn in flight (generation 1): [u1] with the reply bubble streaming
            + "chatHistory.push({ role: 'user', content: 'u1' }); _beginQueuedTurn();\n"
              "const u1 = new El(); u1.className = 'message user'; u1.textContent = 'u1'; chatLog.appendChild(u1);\n"
              "currentAgentMessageDiv = new El(); currentAgentMessageDiv.className = 'message agent'; chatLog.appendChild(currentAgentMessageDiv);\n"
              "const send = async (t) => { chatInput.value = t; await _sendWhileBusy(); };\n"
              "const endTurn = () => { isProcessingRequest = false; currentChatController = null; };\n"
              "const nextTurn = () => { isProcessingRequest = true; currentChatController = { abort() {} };\n"
              "  chatHistory.push({ role: 'user', content: 'u' + (_turnGen + 1) }); _beginQueuedTurn(); };\n"
              f"routes['{L}'] = [200, {{ messages: [] }}];\n"
            + body)


def _log_view():
    return ("chatLog.children.map(c => ({cls: [...c.classList.s].join(' '), text: c.textContent,"
            " label: (c.querySelector() || {}).textContent || null}))")


_INJECT = "routes['/api/turn/inbox'] = [200, { path: 'inject', request_id: 'r' }];\n"
_QUEUE = "routes['/api/turn/inbox'] = [200, { path: 'queue', reason: 'the turn is finishing' }];\n"
_OUT = "{sent, composer: tray().join('\\n'), hist: chatHistory.map(m => m.content), notes, calls: calls.map(c => c.url)}"


def test_a_message_sent_mid_turn_goes_to_the_inbox_and_shows_queued_above_the_reply():
    src = _harness(_INJECT + "await send('  use metric units  ');\n")
    out = eval_js(src, "{calls, log: " + _log_view() + ", composer: chatInput.value, n: _queuedInjects.length}")
    assert out["calls"] == [{"url": "/api/turn/inbox",
                             "body": {"session_id": "web-s1", "text": "use metric units"}}]
    assert out["composer"] == "" and out["n"] == 1
    log = out["log"]
    assert [c["cls"] for c in log] == ["message user", "message user queued", "message agent"], log
    assert log[1]["label"].startswith("queued")


def test_status_is_a_system_note_not_a_bubble_or_history():
    src = _harness("routes['/api/turn/inbox'] = [200, { path: 'status', request_id: 'r', reply: 'Working for 2 min, step 4.' }];\n"
                   "await send('how is it going?');\n")
    out = eval_js(src, "{log: " + _log_view() + ", hist: chatHistory.length, n: _queuedInjects.length + _sendAfterTurn.length}")
    assert out["hist"] == 1 and out["n"] == 0
    assert [c["cls"] for c in out["log"]] == ["message user", "message agent", "message system queued-status"]
    assert out["log"][-1]["text"] == "Working for 2 min, step 4."


def test_feature_off_restores_the_text_and_never_sends_it():
    src = _harness("routes['/api/turn/inbox'] = [200, { path: 'queue', reason: 'queued messages are off' }];\n"
                   "await send('and then the weather');\n"
                   "endTurn(); await _settleQueuedTurn(1, null, 'done');\n")
    out = eval_js(src, "{on: _queuedFeatureOn, held: _sendAfterTurn.length, out: " + _OUT + "}")
    assert out["on"] is False and out["held"] == 0
    o = out["out"]
    assert o["sent"] == [] and o["composer"] == "and then the weather"
    assert o["calls"] == ["/api/turn/inbox"]                 # no leftovers, no session call


@pytest.mark.parametrize("outcome,sent", [("done", ["list my rules"]), ("ended", []), ("unknown", []), ("stopped", [])])
def test_a_held_text_is_sent_only_after_a_turn_that_ended_with_a_reply(outcome, sent):
    """r6a4 rule 3: auto-send needs positive evidence."""
    src = _harness(_QUEUE + "await send('list my rules');\n"
                   f"endTurn(); await _settleQueuedTurn(1, null, '{outcome}');\n")
    out = eval_js(src, _OUT)
    assert out["sent"] == sent
    assert out["composer"] == ("" if sent else "list my rules")


def test_the_tool_button_stays_hidden_until_the_agent_answers_as_on():
    src = _harness("const start = { on: _queuedFeatureOn, hidden: interruptBtn.classList.contains('hidden') };\n"
                   + _INJECT + "await send('A');\n"
                   "const afterOn = interruptBtn.classList.contains('hidden');\n"
                   "routes['/api/turn/inbox'] = [200, { path: 'queue', reason: 'queued messages are off' }];\n"
                   "await send('B');\n")
    out = eval_js(src, "{start, afterOn, afterOff: interruptBtn.classList.contains('hidden')}")
    assert out == {"start": {"on": None, "hidden": True}, "afterOn": False, "afterOff": True}
    fn = extract_js_function(strip_js_comments(APP), "toggleSendButtonUI")
    assert "|| _queuedFeatureOn !== true)" in fn


def test_an_unreachable_inbox_restores_the_text_and_never_sends_it():
    src = _harness("await send('still here');\n"            # no inbox route: fetch throws
                   "endTurn(); await _settleQueuedTurn(1, null, 'done');\n")
    out = eval_js(src, _OUT)
    assert out["sent"] == [] and out["composer"] == "still here"
    assert re.search(r"AbortSignal\.timeout\(INBOX_TIMEOUT_MS\)", extract_js_function(APP, "_sendWhileBusy"))
    assert int(re.search(r"const INBOX_TIMEOUT_MS = (\d+);", APP).group(1)) > 30000   # past the proxy's 30 s


def test_no_session_id_holds_the_text_for_after_the_turn():
    src = _harness("window.__ghostSessionId = null; _turnInfo.get(1).sid = null;\n"
                   "await send('later');\n"
                   "endTurn(); await _settleQueuedTurn(1, null, 'done');\n")
    out = eval_js(src, _OUT)
    assert out["sent"] == ["later"] and out["calls"] == []


def test_a_stop_word_is_handled_like_stop():
    src = _harness("routes['/api/turn/inbox'] = [200, { path: 'cancel', request_id: 'r', undelivered: ['b'] }];\n"
                   "await send('stop');\n")
    out = eval_js(src, "{aborted, und: _inboxStopUndelivered, toggled, log: " + _log_view() + "}")
    assert out["aborted"] == 1 and out["und"] == ["b"] and out["toggled"] == 1
    assert all("queued" not in c["cls"] for c in out["log"])     # the stop word is not a message


def test_after_the_turn_history_is_the_agents_session_copy_and_leftovers_go_to_the_composer():
    """r6a4 rules 1+2: the page writes no queued row itself. The agent stored
    [u1, A, a]; B was accepted, never read → composer, never sent."""
    src = _harness(_INJECT + "await send('A'); await send('B');\n"
                   "chatHistory.push({ role: 'assistant', content: 'a' });\n"
                   f"routes['{S}'] = [200, {{ messages: [{{role: 'user', content: 'u1'}}, {{role: 'user', content: 'A'}}, {{role: 'assistant', content: 'a'}}] }}];\n"
                   f"routes['{L}'] = [200, {{ messages: ['B'] }}];\n"
                   "endTurn(); await _settleQueuedTurn(1, null, 'done');\n")
    out = eval_js(src, "{out: " + _OUT + ", log: " + _log_view() + ", renders}")
    o = out["out"]
    assert o["hist"] == ["u1", "A", "a"] and o["sent"] == [] and o["composer"] == "B"
    assert o["calls"] == ["/api/turn/inbox", "/api/turn/inbox", L, S]   # leftovers, then the session copy
    assert out["renders"] == 1
    assert [c["text"] for c in out["log"] if c["cls"] != "message system"] == ["u1", "A", "a"]
    assert not any("queued" in c["cls"] for c in out["log"])
    assert any("Not read by the agent" in n for n in o["notes"])


def test_a_late_inject_answer_after_the_settle_is_never_history_nor_composer():
    """r6a4 MAJOR 1: B's inject answer lands after the turn settled."""
    src = _harness(_INJECT + "await send('A');\n"
                   "hold('/api/turn/inbox'); const p = send('B');\n"
                   "chatHistory.push({ role: 'assistant', content: 'a' });\n"
                   f"routes['{S}'] = [200, {{ messages: [{{role: 'user', content: 'u1'}}, {{role: 'user', content: 'A'}}, {{role: 'assistant', content: 'a'}}] }}];\n"
                   "endTurn(); await _settleQueuedTurn(1, null, 'done');\n"
                   "await release('/api/turn/inbox'); await p;\n")
    out = eval_js(src, "{out: " + _OUT + ", owed: Object.keys(_owedMap()).join(',') || null, log: " + _log_view() + "}")
    o = out["out"]
    assert o["hist"] == ["u1", "A", "a"] and o["sent"] == []
    assert o["composer"] == "B"                                   # r6a8 MAJOR 2a: in the tray, not only a debt
    assert out["owed"] == "web-s1"                                # and the leftovers are still taken later
    assert not [c for c in out["log"] if c["text"] == "B"]


def test_a_lost_turns_unread_inject_comes_back_once_its_end_is_known():
    """r6a4 MAJOR 2 / r6a7 rule A: no leftovers call while the turn may still
    run; every text the turn accepted comes back at once (may or may not have
    been read), the debt is owed, and the next turn's start pays it — a
    duplicate in the tray, never a loss."""
    src = _harness(_INJECT + "await send('A'); await send('B');\n"
                   "endTurn(); await _settleQueuedTurn(1, null, 'unknown');\n"
                   "const mid = { calls: calls.map(c => c.url), owed: Object.keys(_owedMap()).join(',') || null,"
                   " tray: tray() };\n"
                   f"routes['{L}'] = [200, {{ messages: ['B'] }}];\n"
                   "nextTurn(); await _settleChain; await tick();\n")
    out = eval_js(src, "{mid, owed: Object.keys(_owedMap()).join(',') || null, out: " + _OUT + "}")
    assert out["mid"] == {"calls": ["/api/turn/inbox", "/api/turn/inbox"], "owed": "web-s1", "tray": ["A", "B"]}
    o = out["out"]
    assert o["composer"] == "A\nB\nB" and o["sent"] == [] and out["owed"] is None
    assert any("may or may not" in n for n in o["notes"])


def test_a_confirming_recovery_takes_the_owed_leftovers():
    rec = extract_js_function(strip_js_comments(APP), "reconcileFromSession")
    i = rec.rindex("clearInflightHandle(handle);")
    assert "_settleLostTurns();" in rec[i:i + 300] and "_queueOwedLeftovers(true);" in rec[i:i + 300]
    assert "if (!currentTaskId) { _settleLostTurns(); return; }" in strip_js_comments(APP)


def test_a_queue_answer_after_an_unknown_settle_is_never_sent():
    """r6a4 MINOR (reproduced)."""
    src = _harness(_QUEUE + "hold('/api/turn/inbox'); const p = send('X');\n"
                   "endTurn(); await _settleQueuedTurn(1, null, 'unknown');\n"
                   "await release('/api/turn/inbox'); await p;\n")
    out = eval_js(src, _OUT)
    assert out["sent"] == [] and out["composer"] == "X"


def test_a_failed_leftovers_call_marks_nothing_read_and_retries_next_turn():
    src = _harness(_INJECT + "await send('A');\n"
                   f"delete routes['{L}'];\n"
                   "endTurn(); await _settleQueuedTurn(1, null, 'done');\n"
                   "const mid = { composer: chatInput.value, hist: chatHistory.map(m => m.content), owed: Object.keys(_owedMap()).join(',') || null,"
                   " note: notes.some(n => n.includes('Could not fetch')) };\n"
                   f"routes['{L}'] = [200, {{ messages: ['A'] }}];\n"
                   "nextTurn(); await _settleChain; await tick();\n")
    out = eval_js(src, "{mid, out: " + _OUT + "}")
    assert out["mid"] == {"composer": "", "hist": ["u1"], "owed": "web-s1", "note": True}
    # r6a8 MAJOR 2b: in the tray at once, and the leftover again later (a duplicate, no loss)
    assert out["out"]["composer"] == "A\nA" and out["out"]["sent"] == []


def test_a_stop_puts_unread_messages_back_and_sends_nothing():
    src = _harness(_INJECT + "await send('A'); await send('B');\n"
                   + _QUEUE + "await send('C');\n"
                   "endTurn(); chatInput.value = 'draft';\n"
                   "await _settleQueuedTurn(1, Promise.resolve(['B']), 'stopped');\n")
    out = eval_js(src, "{out: " + _OUT + ", q: _sendAfterTurn.length, draft: chatInput.value}")
    o = out["out"]
    assert o["sent"] == [] and out["q"] == 0 and out["draft"] == "draft"     # the composer is never touched
    assert sorted(o["composer"].split("\n")) == ["A", "B", "B", "C"]         # B twice: undelivered + typed


def test_stop_while_the_inbox_answer_is_in_flight_never_sends_it():
    src = _harness(_QUEUE + "hold('/api/turn/inbox'); const p = send('C');\n"
                   "endTurn(); await _settleQueuedTurn(1, Promise.resolve([]), 'stopped');\n"
                   "await release('/api/turn/inbox'); await p;\n")
    out = eval_js(src, _OUT)
    assert out["sent"] == [] and out["composer"] == "C"


def test_stop_while_an_accepted_inject_is_in_flight_comes_back():
    src = _harness(_INJECT + "hold('/api/turn/inbox'); const p = send('C');\n"
                   "endTurn(); await _settleQueuedTurn(1, Promise.resolve(['C']), 'stopped');\n"
                   "await release('/api/turn/inbox'); await p;\n")
    out = eval_js(src, "{out: " + _OUT + ", log: " + _log_view() + "}")
    assert out["out"]["composer"].split("\n") == ["C", "C"]     # undelivered + the late answer: a duplicate, no loss
    assert out["out"]["hist"] == ["u1"] and out["out"]["sent"] == []
    assert not [c for c in out["log"] if c["text"] == "C"]


def test_a_slow_stop_never_sweeps_the_next_turns_held_messages():
    src = _harness("let undo; const und = new Promise(r => { undo = r; });\n"
                   "endTurn(); const s = _settleQueuedTurn(1, und, 'stopped');\n"
                   "nextTurn();\n" + _QUEUE + "await send('D');\n"
                   "undo([]); await s;\n"
                   "const mid = { held: _sendAfterTurn.map(i => i.text), composer: chatInput.value };\n"
                   "endTurn(); await _settleQueuedTurn(2, null, 'done');\n")
    out = eval_js(src, "{mid, sent}")
    assert out == {"mid": {"held": ["D"], "composer": ""}, "sent": ["D"]}


def test_held_messages_never_go_into_another_conversation():
    src = _harness(_QUEUE + "await send('X');\n"
                   "endTurn(); window.__ghostSessionId = 'web-other';\n"
                   "await _settleQueuedTurn(1, null, 'done'); _renderTray();\n"
                   "const away = tray(); window.__ghostSessionId = 'web-s1'; _renderTray();\n")
    out = eval_js(src, "{away, o: " + _OUT + "}")
    assert out["o"]["sent"] == [] and out["away"] == [] and out["o"]["composer"] == "X"   # its own session's tray
    fn = extract_js_function(strip_js_comments(APP), "sendMessage")
    i = fn.index("if (text === '/clear' && !resuming) {")
    assert "_convEpoch += 1;" in fn[i:i + 200]


def test_a_turn_lost_to_a_recovery_never_blocks_later_held_messages():
    src = _harness(_QUEUE + "await send('typed during turn 1');\n"
                   "endTurn(); _drainSendAfterTurn();\n"
                   "nextTurn(); await _settleChain; await tick(); const back = tray().join('\\n');\n"
                   "await send('typed during turn 2');\n"
                   "endTurn(); await _settleQueuedTurn(2, null, 'done');\n")
    out = eval_js(src, "{out: " + _OUT + ", back, ended: _turnInfo.get(1).ended}")
    assert out["out"]["sent"] == ["typed during turn 2"]
    assert out["back"] == "typed during turn 1" and out["ended"] is True


def test_a_turn_settled_unknown_stays_unknown():
    src = _harness(_QUEUE + "await send('X');\n"
                   "endTurn(); _settleQueuedTurn(1, null, 'unknown'); await _settleQueuedTurn(1, null, 'done');\n")
    out = eval_js(src, "{out: " + _OUT + ", outcome: _turnInfo.get(1).outcome}")
    assert out["outcome"] == "unknown" and out["out"]["sent"] == [] and out["out"]["composer"] == "X"


def test_a_stop_word_answered_after_the_page_lost_the_stream_settles_as_stopped():
    src = _harness(_INJECT + "await send('A');\n" + _QUEUE + "await send('C');\n"
                   "endTurn();\n"
                   "routes['/api/turn/inbox'] = [200, { path: 'cancel', request_id: 'r', undelivered: ['A'] }];\n"
                   "await send('stop'); await _settleChain; await tick();\n")
    out = eval_js(src, "{out: " + _OUT + ", outcome: _turnInfo.get(1).outcome}")
    assert out["outcome"] == "stopped"
    assert out["out"]["sent"] == [] and set(out["out"]["composer"].split("\n")) == {"A", "C"}


def test_no_leftovers_or_session_call_for_a_turn_without_an_accepted_inject():
    src = _harness("endTurn(); await _settleQueuedTurn(1, null, 'done');\n")
    out = eval_js(src, "{calls: calls.map(c => c.url)}")
    assert out == {"calls": []}


def test_a_queue_answer_landing_mid_settle_waits_for_the_settle():
    src = _harness(_INJECT + "await send('A');\n" + _QUEUE + "hold('/api/turn/inbox'); const p = send('C');\n"
                   f"hold('{L}');\n"
                   "endTurn(); const s = _settleQueuedTurn(1, null, 'done');\n"
                   "await release('/api/turn/inbox'); await p;\n"
                   "const mid = [...sent];\n"
                   f"await release('{L}'); await s;\n")
    out = eval_js(src, "{mid, sent}")
    assert out == {"mid": [], "sent": ["C"]}     # held until the settle finished, then sent


def test_wiring_the_page_settles_each_turn_with_its_outcome():
    fn = strip_js_comments(extract_js_function(APP, "sendMessage"))
    assert "_settleQueuedTurn(_myGen, _stopUndeliveredP, _stopUndeliveredP !== null ? 'stopped' : _turnOutcome);" in fn
    assert "_turnOutcome = 'done';" in fn and "_turnOutcome = 'unknown';" in fn
    assert "_beginQueuedTurn();" in fn
    # r6a4: the page never writes a queued row into history itself
    block = APP[APP.index("// ── Messages sent while a turn runs"):APP.index("async function interruptRunningTool")]
    assert "chatHistory.push" not in block and "chatHistory.splice" not in block


def test_a_stop_whose_cancel_never_answers_owes_the_leftovers():
    """No answer from the cancel → unknown: no leftovers call, the next turn takes them."""
    src = _harness(_INJECT + "await send('A');\n"
                   "endTurn(); await _settleQueuedTurn(1, new Promise(() => {}), 'stopped');\n"
                   "const mid = { calls: calls.map(c => c.url), owed: Object.keys(_owedMap()).join(',') || null, outcome: _turnInfo.get(1).outcome };\n"
                   f"routes['{L}'] = [200, {{ messages: ['A'] }}];\n"
                   "nextTurn(); await _settleChain; await tick();\n")
    src = src.replace("const STOP_SETTLE_TIMEOUT_MS = ", "const STOP_SETTLE_TIMEOUT_MS = 5 || ")
    out = eval_js(src, "{mid, tray: tray(), sent}")
    assert out["mid"] == {"calls": ["/api/turn/inbox"], "owed": "web-s1", "outcome": "unknown"}
    assert out["tray"] == ["A", "A"] and out["sent"] == []     # may-or-may-not + the leftover


def test_owed_leftovers_whose_fetch_fails_again_stay_owed():
    src = _harness("_markOwed('web-s1');\n"
                   f"delete routes['{L}'];\n"
                   "nextTurn(); await _settleChain; await tick();\n"
                   "const mid = Object.keys(_owedMap()).join(',') || null;\n"
                   f"routes['{L}'] = [200, {{ messages: ['A'] }}];\n"
                   "nextTurn(); await _settleChain; await tick();\n")
    out = eval_js(src, "{mid, owed: Object.keys(_owedMap()).join(',') || null, composer: tray().join('\\n')}")
    assert out == {"mid": "web-s1", "owed": None, "composer": "A"}


def test_a_held_text_waits_while_the_next_turn_runs_even_after_a_done_settle():
    """A slow settle of a 'done' turn finishing mid next turn: send only once idle."""
    src = _harness(_INJECT + "await send('A');\n" + _QUEUE + "await send('C');\n"
                   f"hold('{L}');\n"
                   "endTurn(); const s = _settleQueuedTurn(1, null, 'done');\n"
                   "isProcessingRequest = true;\n"
                   f"await release('{L}'); await s;\n"
                   "const mid = [...sent];\n"
                   "isProcessingRequest = false; _drainSendAfterTurn();\n")
    out = eval_js(src, "{mid, sent}")
    assert out == {"mid": [], "sent": ["C"]}


# ── r6a5: never lose; a duplicate in the composer is acceptable ────────────

def test_a_clean_end_is_a_finished_stream_with_a_real_reply():
    """Rule A: the agent's cancel note (agent.py / routes.py wording) is not a clean reply."""
    cases = ["a reply", "", "   ", "_(Turn cancelled: stopped by the owner.)_",
             "partial text\n\n_(Turn cancelled: stopped by the owner.)_",
             "_(Turn cancelled: owner pressed stop.)_ No output was produced before the cancellation.",
             "the note _(Turn cancelled: x.)_ quoted mid-reply, then more",
             "glued_(Turn cancelled: x.)_"]
    src = _harness("")
    out = eval_js(src, "%s.map(_isCleanReply)" % json.dumps(cases))
    assert out == [True, False, False, False, False, False, True, False]
    fn = strip_js_comments(extract_js_function(APP, "sendMessage"))
    assert ("if (!_onlyReasoned && streamSawDone && !streamHadError && typeof _isCleanReply === 'function'\n"
            "                    && _isCleanReply(chatHistory[chatHistory.length - 1].content)) _turnOutcome = 'done';") in fn
    assert "const _onlyReasoned = _revealAgentBubble(currentAgentMessageDiv);" in fn


def test_an_inject_read_by_a_failed_turn_that_kept_nothing_comes_back():
    """MAJOR 2 (r6a5): the turn errored after reading A; the session copy lacks it."""
    src = _harness(_INJECT + "await send('A');\n"
                   f"routes['{S}'] = [200, {{ messages: [{{role:'user',content:'u0'}}, {{role:'assistant',content:'a0'}}] }}];\n"
                   "endTurn(); await _settleQueuedTurn(1, null, 'ended');\n")
    out = eval_js(src, _OUT)
    assert out["composer"] == "A" and out["sent"] == []
    assert any("may or may not have read" in n for n in out["notes"])


def test_an_inject_accepted_after_a_turn_that_did_not_end_cleanly_comes_back():
    src = _harness(_INJECT + "hold('/api/turn/inbox'); const p = send('B');\n"
                   "endTurn(); await _settleQueuedTurn(1, null, 'ended');\n"
                   "await release('/api/turn/inbox'); await p;\n")
    out = eval_js(src, "{o: " + _OUT + ", owed: Object.keys(_owedMap())}")
    assert out["o"]["composer"] == "B" and out["o"]["hist"] == ["u1"] and out["owed"] == ["web-s1"]


def test_an_empty_leftovers_answer_keeps_the_debt_while_the_old_turn_may_run():
    """MAJOR 1 (r6a5, probe p1): the old turn still runs when the next starts."""
    src = _harness(_INJECT + "await send('B');\n"
                   "endTurn(); await _settleQueuedTurn(1, null, 'unknown');\n"
                   "nextTurn(); await _settleChain; await tick();\n"          # leftovers: [] (still running)
                   "const mid = Object.keys(_owedMap());\n"
                   f"routes['{L}'] = [200, {{ messages: ['B'] }}];\n"
                   "chatHistory.push({role:'assistant', content:'a2'});\n"
                   "endTurn(); await _settleQueuedTurn(2, null, 'done');\n")
    out = eval_js(src, "{mid, owed: Object.keys(_owedMap()), tray: tray(), sent}")
    assert out == {"mid": ["web-s1"], "owed": [], "tray": ["B", "B"], "sent": []}


def test_a_later_clean_turn_settles_the_debt_even_when_empty():
    src = _harness("_markOwed('web-s1'); _owedMap()['web-s1'] -= 1000;\n"
                   "nextTurn(); await _settleChain; await tick();\n"
                   "const mid = Object.keys(_owedMap());\n"
                   "endTurn(); await _settleQueuedTurn(2, null, 'done');\n"
                   "const clean = Object.keys(_owedMap());\n"
                   "_markOwed('web-s1'); await _settleQueuedTurn(1, null, 'ended');\n")
    out = eval_js(src, "{mid, clean, unclean: Object.keys(_owedMap())}")
    assert out == {"mid": ["web-s1"], "clean": [], "unclean": ["web-s1"]}


def test_a_confirming_recovery_or_the_ttl_settles_the_debt():
    src = _harness("_markOwed('web-s1'); await _collectOwedLeftovers(false);\n"
                   "const open = Object.keys(_owedMap());\n"
                   "await _collectOwedLeftovers(true);\n"
                   "const closed = Object.keys(_owedMap());\n"
                   "_markOwed('web-s1'); _owedMap()['web-s1'] -= LEFTOVER_TTL_MS + 1; await _collectOwedLeftovers(false);\n")
    out = eval_js(src, "{open, closed, ttl: Object.keys(_owedMap())}")
    assert out == {"open": ["web-s1"], "closed": [], "ttl": []}
    assert re.search(r"^const LEFTOVER_TTL_MS = 30 \* 60 \* 1000;", APP, re.M)


def test_debts_are_kept_per_session_and_paid_only_into_that_session():
    """MAJOR 3 (r6a5, probe p4)."""
    src = _harness(_INJECT + "await send('B');\n"
                   "endTurn(); await _settleQueuedTurn(1, null, 'unknown');\n"
                   "window.__ghostSessionId = 'web-s2';\n"
                   "nextTurn(); await send('C'); endTurn(); await _settleQueuedTurn(2, null, 'unknown');\n"
                   "await _collectOwedLeftovers(false);\n"
                   "const away = { owed: Object.keys(_owedMap()).sort(), tray: tray(),"
                   " fetched: calls.filter(c => c.url === '" + L + "').map(c => c.body.session_id) };\n"
                   f"routes['{L}'] = [200, {{ messages: ['B'] }}];\n"
                   "window.__ghostSessionId = 'web-s1'; await _collectOwedLeftovers(false);\n")
    out = eval_js(src, "{away, owed: Object.keys(_owedMap()), tray: tray(), store}")
    assert out["away"]["owed"] == ["web-s1", "web-s2"] and out["away"]["tray"] == ["C"]
    assert "web-s1" not in out["away"]["fetched"]
    assert out["owed"] == ["web-s2"] and out["tray"] == ["B", "B"]
    assert json.loads(out["store"]["ghost.queued.leftoversOwed"]).keys() == {"web-s2"}


def test_debts_survive_a_reload_and_the_page_works_without_storage():
    src = _harness("store['ghost.queued.leftoversOwed'] = JSON.stringify({'web-s1': Date.now()});\n"
                   "const loaded = Object.keys(_owedMap());\n"
                   "localStorage.getItem = () => { throw new Error('denied'); };\n"
                   "localStorage.setItem = () => { throw new Error('denied'); };\n"
                   "_leftoversOwed = null; _markOwed('web-s1');\n")
    out = eval_js(src, "{loaded, after: Object.keys(_owedMap())}")
    assert out == {"loaded": ["web-s1"], "after": ["web-s1"]}


def test_a_new_turn_never_settles_a_turn_a_recovery_may_own():
    """_settleLostTurns(true): no session call, no settle; owed + typed back."""
    src = _harness(_INJECT + "await send('A');\n" + _QUEUE + "await send('H');\n"
                   "endTurn(); calls.length = 0;\n"
                   "nextTurn(); await tick();\n")
    out = eval_js(src, "{calls: calls.map(c => c.url), owed: Object.keys(_owedMap()), composer: tray().join('\\n'),"
                  " pend: _settlesPending, sent, info: _turnInfo.get(1)}")
    assert S not in out["calls"] and out["pend"] == 0 and out["sent"] == []
    assert out["owed"] == ["web-s1"] and out["composer"] == "A\nH"
    assert out["info"]["outcome"] == "unknown" and out["info"]["settled"] is True


def test_the_reply_keeps_its_label_key_after_the_session_copy_is_adopted():
    """MINOR 4 (r6a5, probe p3): real mergeClientLabelKeys; the inject row shifts the reply."""
    src = _harness(_INJECT + "await send('A');\n"
                   "chatHistory.push({ role: 'assistant', content: 'a', reqId: 'rq1', feedback: 'up' });\n"
                   f"routes['{S}'] = [200, {{ messages: [{{role:'user',content:'u1'}}, {{role:'user',content:'A'}}, {{role:'assistant',content:'a'}}] }}];\n"
                   "endTurn(); await _settleQueuedTurn(1, null, 'done');\n")
    src = src.replace("const mergeClientLabelKeys = (x) => x;", extract_js_function(APP, "mergeClientLabelKeys"))
    out = eval_js(src, "chatHistory")
    assert [m["content"] for m in out] == ["u1", "A", "a"]
    assert out[-1].get("reqId") == "rq1" and out[-1].get("feedback") == "up"


def test_a_label_key_is_never_grafted_onto_a_different_reply():
    src = _harness(_INJECT + "await send('A');\n"
                   "chatHistory.push({ role: 'assistant', content: 'a', reqId: 'rq1' });\n"
                   f"routes['{S}'] = [200, {{ messages: [{{role:'user',content:'u1'}}, {{role:'user',content:'A'}}, {{role:'assistant',content:'z'}}] }}];\n"
                   "endTurn(); await _settleQueuedTurn(1, null, 'done');\n")
    src = src.replace("const mergeClientLabelKeys = (x) => x;", extract_js_function(APP, "mergeClientLabelKeys"))
    out = eval_js(src, "chatHistory")
    assert out[-1]["content"] == "z" and "reqId" not in out[-1]


def test_a_turn_of_another_session_hands_back_into_that_sessions_tray():
    """r6a6 MAJOR (probe p2): the owner switched before the settle."""
    src = _harness(_INJECT + "await send('A');\n"
                   f"routes['{L}'] = [200, {{ messages: ['A'] }}];\n"
                   "endTurn(); window.__ghostSessionId = 'web-s2';\n"
                   "await _settleQueuedTurn(1, null, 'ended');\n"
                   "const away = { tray: tray(), composer: chatInput.value, notes: notes.length };\n"
                   "window.__ghostSessionId = 'web-s1'; _renderTray();\n")
    out = eval_js(src, "{away, tray: tray(), notes, stored: stored()}")
    assert out["away"] == {"tray": [], "composer": "", "notes": 0}
    assert out["tray"] == ["A", "A"] and any("in the tray" in n for n in out["notes"])
    assert [i["text"] for i in out["stored"]["web-s1"]] == ["A", "A"]


def test_only_a_clean_turn_started_after_the_mark_settles_an_empty_debt():
    src = _harness(_INJECT + "await send('A');\n"
                   "_markOwed('web-s1'); _owedMap()['web-s1'] -= 1000;\n"
                   "endTurn(); await _settleQueuedTurn(1, null, 'ended');\n"   # unclean, leftovers []
                   "const unclean = Object.keys(_owedMap());\n"
                   "nextTurn(); await _settleChain; _markOwed('web-s1'); _owedMap()['web-s1'] += 1000;\n"
                   "chatInput.value = ''; await send('B');\n"
                   "endTurn(); await _settleQueuedTurn(2, null, 'done');\n")    # marked after it started
    out = eval_js(src, "{unclean, late: Object.keys(_owedMap())}")
    assert out == {"unclean": ["web-s1"], "late": ["web-s1"]}


# ── r6a7: a tray, not the composer; no dedup, no implicit pruning ──────────

def _extra(*names):
    return "\n".join(extract_js_function(APP, n) for n in names) + "\n"


def test_an_errored_turn_hands_back_every_inject_even_with_an_older_identical_ask():
    """r6a7 (probe p1): no 'visibly held' check — an unclean end hands back all."""
    src = _harness(
        "chatHistory = [{role:'user',content:'continue'},{role:'user',content:'yes'},{role:'assistant',content:'a0'},"
        "{role:'user',content:'continue'}]; _turnInfo.clear(); _turnGen = 0; _beginQueuedTurn();\n"
        + _INJECT + "await send('yes');\n"
        f"routes['{S}'] = [200, {{ messages: [{{role:'user',content:'continue'}},{{role:'user',content:'yes'}},{{role:'assistant',content:'a0'}}] }}];\n"
        "endTurn(); await _settleQueuedTurn(1, null, 'ended');\n")
    out = eval_js(src, "{tray: tray(), calls: calls.map(c => c.url)}")
    assert out["tray"] == ["yes"]
    assert S not in out["calls"]           # the copy is read only to adopt after a clean turn
    assert "info.ask" not in APP and "_typings" not in APP


def test_another_clients_identical_leftover_is_shown_too():
    """r6a7 (probe p2): no counters — every server text is shown."""
    src = _harness(_INJECT + "await send('yes');\n"
                   f"routes['{L}'] = [200, {{ messages: ['yes'] }}];\n"
                   "endTurn(); await _settleQueuedTurn(1, null, 'done');\n"
                   "const first = tray(); _trayRemove(first.length ? _restoreMap()['web-s1'][0].id : '');\n"
                   "_restoreToComposer(['yes'], 'Not read by the agent in an earlier turn.', 'web-s1');\n")
    assert eval_js(src, "{first, tray: tray()}") == {"first": ["yes"], "tray": ["yes"]}


def test_two_tabs_never_overwrite_each_others_tray():
    """r6a7 (probe p3) / r6a10: each item is its own key."""
    src = _harness("_restoreMap();\n"
                   "store['ghost.queued.restore.item.a1'] = JSON.stringify({id:'a1', batch:'b', sid:'web-s9', text:'X', why:'w', at: Date.now()});\n"
                   "_restoreToComposer(['Y'], 'why');\n"
                   "const both = stored();\n"
                   "_trayRemove(_restoreMap()['web-s1'][0].id);\n")
    out = eval_js(src, "{both, after: stored()}")
    assert [i["text"] for i in out["both"]["web-s9"]] == ["X"] and [i["text"] for i in out["both"]["web-s1"]] == ["Y"]
    assert list(out["after"]) == ["web-s9"]
    assert ("if (e.key === null || e.key === _RESTORE_KEY || String(e.key).startsWith(_RESTORE_ITEM_PREFIX)) "
            "{ try { _renderTray(); }") in APP


def test_retry_or_edit_never_touches_the_tray():
    """r6a7 (probe p4): setting the composer no longer prunes anything."""
    src = _harness(_extra("_userTurnIndexFor", "_historyEntryText", "_truncateToUserTurn")
        + "chatInput.focus = () => {}; const _endRecoveryForNewTurn = () => {}; const sendTypedMessage = () => {};\n"
        + _INJECT + "await send('also do Y');\n"
        f"routes['{S}'] = [500, {{}}];\n"
        "endTurn(); await _settleQueuedTurn(1, null, 'ended');\n"
        "chatHistory = [{role:'user',content:'u1'}];\n"
        "_truncateToUserTurn(0, false);\n")
    out = eval_js(src, "{c: chatInput.value, tray: tray()}")
    assert out == {"c": "u1", "tray": ["also do Y"]}
    assert "_pruneRestores" not in APP and "_imeComposing" not in APP


def test_a_voice_or_upload_send_never_touches_the_tray():
    """r6a7 (probe p5)."""
    src = _harness(_extra("_sendComposed")
        + "isProcessingRequest = false;\n"
          "sendMessage = () => { chatInput.value = ''; return Promise.resolve(); };\n"
          "_restoreToComposer(['keep me'], 'why');\n"
          "await _sendComposed('voice words');\n")
    assert eval_js(src, "tray()") == ["keep me"]


def test_insert_puts_the_text_at_the_caret_and_dismiss_removes_it():
    src = _harness("chatInput.value = 'ab'; chatInput.selectionStart = 1; chatInput.selectionEnd = 1;\n"
                   "_restoreToComposer(['X', 'Z'], 'why');\n"
                   "const rows = restoreTray.children.length, hidden = restoreTray.classList.contains('hidden');\n"
                   "restoreTray.children[0].children[1].onclick();\n"          # Insert X
                   "const afterInsert = { c: chatInput.value, caret: chatInput.selectionStart, tray: tray() };\n"
                   "restoreTray.children[0].children[2].onclick();\n")         # Dismiss Z
    out = eval_js(src, "{rows, hidden, afterInsert, c: chatInput.value, tray: tray(), hiddenNow: restoreTray.classList.contains('hidden'),"
                  " stored: JSON.stringify(stored()), notes}")
    assert out["rows"] == 2 and out["hidden"] is False
    assert out["afterInsert"] == {"c": "aXb", "caret": 2, "tray": ["Z"]}
    assert out["c"] == "aXb" and out["tray"] == [] and out["hiddenNow"] is True and out["stored"] == "{}"
    assert sum("in the tray" in n for n in out["notes"]) == 1      # one note per batch


def test_the_tray_survives_a_reload_and_follows_the_session_on_screen():
    src = _harness("_restoreToComposer(['A'], 'why', 'web-s1'); _restoreToComposer(['B'], 'why', 'web-s2');\n"
                   "const s1 = tray(); window.__ghostSessionId = 'web-s2'; _renderTray(); const s2 = tray();\n"
                   "_restoreUnsaved = null; restoreTray.innerHTML = ''; window.__ghostSessionId = 'web-s1'; _renderTray();\n")
    out = eval_js(src, "{s1, s2, reloaded: tray()}")
    assert out == {"s1": ["A"], "s2": ["B"], "reloaded": ["A"]}
    pid = extract_js_function(APP, "persistSessionId")
    assert "if (typeof _onSessionShown === 'function') setTimeout(_onSessionShown, 0);" in pid
    assert re.search(r"^setTimeout\(_onSessionShown, 0\);", APP, re.M)


def test_the_tray_expires_after_a_week_and_works_without_storage():
    src = _harness("store['ghost.queued.restore'] = JSON.stringify({'web-s1': ["
                   "{id:'o', batch:'b', text:'old', why:'w', at: Date.now() - RESTORE_TTL_MS - 1},"
                   "{id:'n', batch:'b', text:'new', why:'w', at: Date.now()}]});\n"
                   "_renderTray(); const ttl = tray();\n"
                   "localStorage.getItem = () => { throw new Error('denied'); };\n"
                   "localStorage.setItem = () => { throw new Error('denied'); };\n"
                   "_restoreToComposer(['C'], 'why');\n")
    out = eval_js(src, "{ttl, without: tray()}")
    assert out == {"ttl": ["new"], "without": ["C"]}
    assert re.search(r"^const RESTORE_TTL_MS = 7 \* 24 \* 60 \* 60 \* 1000;", APP, re.M)


def test_a_full_storage_keeps_the_tray_in_memory():
    src = _harness("localStorage.setItem = () => { throw new Error('quota'); };\n"
                   "_restoreToComposer(['A'], 'why'); _restoreToComposer(['B'], 'why');\n")
    assert eval_js(src, "tray()") == ["A", "B"]


def test_owed_leftovers_fetched_across_a_switch_go_to_their_own_session():
    src = _harness("await _settleChain; _markOwed('web-s1');\n"
                   f"routes['{L}'] = [200, {{ messages: ['B'] }}];\n"
                   f"hold('{L}'); const p = _collectOwedLeftovers(false); await tick();\n"
                   "window.__ghostSessionId = 'web-s2';\n"
                   f"await release('{L}'); await p; _renderTray(); const away = tray();\n"
                   "window.__ghostSessionId = 'web-s1'; _renderTray();\n")
    assert eval_js(src, "{away, home: tray()}") == {"away": [], "home": ["B"]}


# ── r6a8: storage merges, every text reaches a tray ────────────────────────

def test_a_full_storage_keeps_the_tray_across_a_reload_and_says_so():
    """r6a8 MINOR 1 (probe p1): sessionStorage keeps what localStorage refused."""
    src = _harness("localStorage.setItem = () => { throw new Error('quota'); };\n"
                   "_restoreToComposer(['A'], 'why', 'web-s1');\n"
                   "const warn = restoreTray.children.some(r => r.classList.contains('restore-warn'));\n"
                   "_restoreUnsaved = null; restoreTray.innerHTML = ''; _renderTray();\n"   # a reload
                   "const warnAfter = restoreTray.children.some(r => r.classList.contains('restore-warn'));\n")
    assert eval_js(src, "{warn, warnAfter, after: tray()}") == {"warn": True, "warnAfter": True, "after": ["A"]}


def test_a_tab_back_from_a_full_storage_never_overwrites_another_tabs_items():
    """r6a8 MAJOR 1 (probe p1b)."""
    src = _harness("let fail = 1; const realSet = localStorage.setItem;\n"
                   "localStorage.setItem = (k, v) => { if (fail-- > 0) throw new Error('quota'); realSet(k, v); };\n"
                   "_restoreToComposer(['A'], 'why', 'web-s1');\n"
                   "store['ghost.queued.restore.item.t2'] = JSON.stringify({id: 't2', batch: 't2', sid: 'web-s2', text: 'TAB2 TEXT', why: 'w', at: Date.now()});\n"
                   "_restoreToComposer(['B'], 'why', 'web-s1');\n")
    out = eval_js(src, "{stored: stored(), unsaved: sstore}")
    assert [i["text"] for i in out["stored"]["web-s2"]] == ["TAB2 TEXT"]
    assert [i["text"] for i in out["stored"]["web-s1"]] == ["A", "B"]
    assert out["unsaved"] == {}                       # saved now: nothing left only with this tab


def test_items_kept_before_a_session_id_show_with_the_bound_session():
    """r6a8 MINOR 2 (probe p2)."""
    src = _harness("await _settleChain; endTurn(); await _settleQueuedTurn(1, null, 'done');\n"
                   "window.__ghostSessionId = undefined; nextTurn();\n"
                   "await send('later please');\n"
                   "window.__ghostSessionId = 'web-new';\n"
                   "endTurn(); await _settleQueuedTurn(2, null, 'ended'); await tick(); _renderTray();\n"
                   "const shown = tray(); restoreTray.children[0].children[2].onclick();\n")
    out = eval_js(src, "{shown, sent, stored: JSON.stringify(stored())}")
    assert out == {"shown": ["later please"], "sent": [], "stored": "{}"}


def test_a_late_inject_after_a_clean_settle_reaches_the_tray():
    """r6a8 MAJOR 2a (probe p3)."""
    src = _harness(_INJECT + f"routes['{S}'] = [200, {{ messages: [] }}];\n"
                   "hold('/api/turn/inbox'); const p = send('one more thing'); await tick();\n"
                   "endTurn(); await _settleQueuedTurn(1, null, 'done');\n"
                   "await release('/api/turn/inbox'); await p; await tick();\n")
    out = eval_js(src, "{tray: tray(), owed: Object.keys(_owedMap()), hist: chatHistory.map(m => m.content)}")
    assert out == {"tray": ["one more thing"], "owed": ["web-s1"], "hist": ["u1"]}


def test_retry_or_edit_puts_the_draft_in_the_tray_before_refilling():
    """r6a8 MINOR 3 (probe p4)."""
    src = _harness(_extra("_userTurnIndexFor", "_historyEntryText", "_truncateToUserTurn")
        + "chatInput.focus = () => {}; const _endRecoveryForNewTurn = () => {}; const sendTypedMessage = () => {};\n"
          "_restoreToComposer(['A'], 'why'); chatInput.value = 'draft '; chatInput.selectionStart = chatInput.selectionEnd = 6;\n"
          "restoreTray.children[0].children[1].onclick();\n"
          "const inserted = chatInput.value;\n"
          "endTurn(); chatHistory = [{role:'user',content:'u1'}]; _truncateToUserTurn(0, false);\n")
    out = eval_js(src, "{inserted, c: chatInput.value, tray: tray(), notes}")
    assert out["inserted"] == "draft A" and out["c"] == "u1" and out["tray"] == ["draft A"]
    assert any("Your draft before Retry/Edit." in n for n in out["notes"])


def test_a_clean_end_whose_leftovers_call_failed_hands_its_texts_back():
    """r6a8 MAJOR 2b (probe p5)."""
    src = _harness(_INJECT + f"routes['{S}'] = [200, {{ messages: [] }}];\n"
                   f"routes['{L}'] = [500, {{}}];\n"
                   "await send('unread one');\n"
                   "endTurn(); await _settleQueuedTurn(1, null, 'done'); await tick();\n")
    out = eval_js(src, "{tray: tray(), owed: Object.keys(_owedMap())}")
    assert out == {"tray": ["unread one"], "owed": ["web-s1"]}


def test_owed_leftovers_are_collected_when_a_session_is_shown():
    """r6a8 MAJOR 2: at load and on a switch, not only at the next turn."""
    src = _harness("await _settleChain; endTurn(); _markOwed('web-s2');\n"
                   f"routes['{L}'] = [200, {{ messages: ['B'] }}];\n"
                   "window.__ghostSessionId = 'web-s2'; _onSessionShown(); await _settleChain; await tick();\n")
    out = eval_js(src, "{tray: tray(), owed: Object.keys(_owedMap())}")
    assert out == {"tray": ["B"], "owed": []}


def test_notes_wait_for_a_bound_session_and_the_tray_is_announced():
    src = _harness("window.__ghostSessionId = null; _restoreToComposer(['A'], 'why');\n"
                   "const before = notes.length; window.__ghostSessionId = 'web-s1'; _renderTray();\n")
    out = eval_js(src, "{before, after: notes.length, tray: tray()}")
    assert out == {"before": 0, "after": 1, "tray": ["A"]}
    assert '<div id="restore-tray" class="hidden" role="region" aria-live="polite"' in INDEX


# ── r6a9: random ids, removals stick under quota, duplicated tabs ──────────

_QUOTA = ("const realSet = localStorage.setItem;\n"
          "localStorage.setItem = (k, v) => { if (String(v).length > 200 && k.startsWith('ghost.queued.restore.item.')) throw new Error('quota'); realSet(k, v); };\n")


def test_a_dismiss_sticks_while_storage_is_full():
    """r6a9 MINOR (probe p1)."""
    src = _harness("_restoreToComposer(['STORED'], 'w', 'web-s1');\n" + _QUOTA +
                   "_restoreToComposer(['UNSAVED ' + 'x'.repeat(150)], 'w', 'web-s1');\n"
                   "const before = tray();\n"
                   "const id = stored()['web-s1'][0].id;\n"
                   "_trayRemove(id); const after1 = tray(); _trayRemove(id);\n")
    out = eval_js(src, "{before, after1, after2: tray(), stored: JSON.stringify(stored())}")
    assert out["before"][0] == "STORED" and len(out["before"]) == 2
    assert [t[:7] for t in out["after1"]] == ["UNSAVED"] and out["after2"] == out["after1"]
    assert out["stored"] == "{}"


def test_two_tabs_in_the_same_millisecond_get_distinct_ids():
    """r6a9 MAJOR (probe p2)."""
    src = _harness("const now = Date.now(); Date.now = () => now;\n"
                   "_restoreToComposer(['MINE'], 'w', 'web-s1');\n"
                   "_restoreSeq = 0; _restoreToComposer(['OTHER TAB'], 'w', 'web-s1');\n"   # a second tab's counter
                   "const m = stored();\n"
                   "_trayInsert(m['web-s1'][0].id);\n")
    out = eval_js(src, "{after: tray(), composer: chatInput.value, ids: new Set(m['web-s1'].map(i => i.id)).size}")
    assert out == {"after": ["OTHER TAB"], "composer": "MINE", "ids": 2}
    assert re.fullmatch(r"[0-9a-f-]{36}", eval_js(_harness(""), "_restoreId()"))      # crypto.randomUUID when present
    fallback = "const crypto = undefined;\n" + _harness("const a = _restoreId(), b = _restoreId();\n")
    assert eval_js(fallback, "a !== b && /-[a-z0-9]+-\\d+$/.test(a)") is True


def test_a_duplicated_tab_never_brings_back_a_dismissed_item():
    """r6a9 MINOR (probe p4): unsaved items localStorage once held are dropped on load."""
    src = _harness("let full = true; const realSet = localStorage.setItem;\n"
                   "localStorage.setItem = (k, v) => { if (full && k.startsWith('ghost.queued.restore.item.')) throw new Error('quota'); realSet(k, v); };\n"
                   "_restoreToComposer(['U'], 'w', 'web-s1');\n"
                   "const copied = sstore['ghost.queued.restore.unsaved'];\n"          # Duplicate tab copies this
                   "full = false; _restoreToComposer(['X'], 'w', 'web-s1');\n"         # A saves U and X
                   "_trayRemove(stored()['web-s1'].find(i => i.text === 'U').id);\n"
                   "const afterDismiss = tray();\n"
                   "sstore['ghost.queued.restore.unsaved'] = copied; _restoreUnsaved = null;\n"   # now tab B
                   "_restoreToComposer(['Y'], 'w', 'web-s1');\n")
    out = eval_js(src, "{copied: !!copied, afterDismiss, stored: stored()['web-s1'].map(i => i.text), tray: tray()}")
    assert out == {"copied": True, "afterDismiss": ["X"], "stored": ["X", "Y"], "tray": ["X", "Y"]}


def test_a_dismiss_of_an_unsaved_item_sticks_and_other_keys_are_left_alone():
    src = _harness("store['someone.else'] = 'not json {';\n"
                   "localStorage.setItem = (k, v) => { if (k.startsWith('ghost.queued.restore.item.')) throw new Error('quota'); store[k] = String(v); };\n"
                   "_restoreToComposer(['U'], 'w', 'web-s1'); const before = tray();\n"
                   "restoreTray.children.find(r => r.classList.contains('restore-item')).children[2].onclick();\n")
    out = eval_js(src, "{before, after: tray(), unsaved: 'ghost.queued.restore.unsaved' in sstore, other: store['someone.else']}")
    assert out == {"before": ["U"], "after": [], "unsaved": False, "other": "not json {"}


def test_a_reload_keeps_an_unsaved_item_that_storage_never_held():
    src = _harness("localStorage.setItem = () => { throw new Error('quota'); };\n"
                   "_restoreToComposer(['A'], 'w', 'web-s1');\n"
                   "store['ghost.queued.restore.confirmed'] = JSON.stringify(['someone-else']);\n"
                   "_restoreUnsaved = null; _renderTray();\n")
    assert eval_js(src, "tray()") == ["A"]


# ── r6a10: one localStorage key per item ───────────────────────────────────

def test_a_stale_cached_read_never_loses_another_tabs_item():
    """r6a10 MINOR (probe): Chrome's per-renderer cache — tab A reads before
    tab B's write shows up; A's add writes only its own key."""
    src = _harness("_restoreToComposer(['A0'], 'w', 'web-s1');\n"
                   "const staleKeys = Object.keys(store);\n"
                   "store['ghost.queued.restore.item.t2'] = JSON.stringify({id: 't2', batch: 't2', sid: 'web-s2', text: 'TAB B TEXT', why: 'w', at: Date.now()});\n"
                   "const realKey = localStorage.key; let stale = true;\n"
                   "Object.defineProperty(localStorage, 'length', { get: () => (stale ? staleKeys : Object.keys(store)).length });\n"
                   "localStorage.key = (i) => (stale ? staleKeys : Object.keys(store))[i] ?? null;\n"
                   "_restoreToComposer(['A1'], 'w', 'web-s1'); _trayRemove(stored()['web-s1'][0].id); stale = false;\n")
    out = eval_js(src, "stored()")
    assert [i["text"] for i in out["web-s2"]] == ["TAB B TEXT"]
    assert [i["text"] for i in out["web-s1"]] == ["A1"]


def test_the_old_single_map_is_migrated_to_one_key_per_item():
    src = _harness("store['ghost.queued.restore'] = JSON.stringify({'web-s1': [{id:'o1', batch:'b', text:'OLD', why:'w', at: Date.now()}],"
                   " '': [{id:'o2', batch:'b', text:'NOSID', why:'w', at: Date.now()}]});\n"
                   "_renderTray();\n")
    out = eval_js(src, "{tray: tray(), legacy: 'ghost.queued.restore' in store, stored: stored()}")
    assert out["tray"] == ["OLD", "NOSID"] and out["legacy"] is False
    assert [i["id"] for i in out["stored"]["web-s1"]] == ["o1"] and [i["id"] for i in out["stored"][""]] == ["o2"]


def test_an_unsaved_item_is_written_once_storage_has_room():
    src = _harness("let full = true; const realSet = localStorage.setItem;\n"
                   "localStorage.setItem = (k, v) => { if (full && k.startsWith('ghost.queued.restore.item.')) throw new Error('quota'); realSet(k, v); };\n"
                   "_restoreToComposer(['A'], 'w', 'web-s1');\n"
                   "const mid = { stored: Object.keys(stored()), unsaved: 'ghost.queued.restore.unsaved' in sstore };\n"
                   "full = false; _renderTray();\n")
    out = eval_js(src, "{mid, stored: stored()['web-s1'].map(i => i.text), unsaved: 'ghost.queued.restore.unsaved' in sstore,"
                  " warn: restoreTray.children.some(r => r.classList.contains('restore-warn'))}")
    assert out == {"mid": {"stored": [], "unsaved": True}, "stored": ["A"], "unsaved": False, "warn": False}


def test_the_full_storage_warning_names_this_session_and_points_at_others():
    """r6a10 NIT."""
    src = _harness("localStorage.setItem = () => { throw new Error('quota'); };\n"
                   "_restoreToComposer(['OTHER'], 'w', 'web-s2');\n"
                   "const warns = () => restoreTray.children.filter(r => r.classList.contains('restore-warn')).map(r => r.textContent);\n"
                   "const onS1 = { warns: warns(), tray: tray(), hidden: restoreTray.classList.contains('hidden') };\n"
                   "_restoreToComposer(['MINE'], 'w', 'web-s1');\n")
    out = eval_js(src, "{onS1, warns: warns(), tray: tray()}")
    assert out["onS1"] == {"warns": ["Other conversations have unsaved items: open them before closing this tab."],
                           "tray": [], "hidden": False}
    assert out["warns"] == ["Browser storage is full: these are kept only until this tab closes.",
                            "Other conversations have unsaved items: open them before closing this tab."]
    assert out["tray"] == ["MINE"]


def test_an_expired_item_key_is_removed():
    src = _harness("store['ghost.queued.restore.item.x'] = JSON.stringify({id:'x', batch:'b', sid:'web-s1', text:'OLD', why:'w',"
                   " at: Date.now() - RESTORE_TTL_MS - 1});\n"
                   "_renderTray();\n")
    assert eval_js(src, "{tray: tray(), left: 'ghost.queued.restore.item.x' in store}") == {"tray": [], "left": False}


# ── r6a11: the confirmed list on every retry and in the migration ──────────

_FULL = ("let full = true; const realSet = localStorage.setItem;\n"
         "localStorage.setItem = (k, v) => { if (full && k.startsWith('ghost.queued.restore.item.')) throw new Error('quota'); realSet(k, v); };\n")


def test_an_already_loaded_duplicated_tab_never_brings_back_a_dismissed_item():
    """r6a11 (probe p1): the confirmed check runs on every retry, not only at load."""
    src = _harness(_FULL +
        "_restoreToComposer(['U'], 'w', 'web-s1');\n"
        "const copiedB = JSON.parse(sstore['ghost.queued.restore.unsaved']);\n"
        "full = false; _renderTray();\n"
        "_trayRemove(stored()['web-s1'][0].id);\n"
        "const afterA = tray();\n"
        "_restoreUnsaved = copiedB; _renderTray();\n")
    out = eval_js(src, "{afterA, trayB: tray(), stored: (stored()['web-s1'] || []).map(i => i.text)}")
    assert out == {"afterA": [], "trayB": [], "stored": []}


def test_old_map_items_are_normalised_never_dropped():
    """r6a11 NIT (probe p2)."""
    src = _harness("store['ghost.queued.restore'] = JSON.stringify({'web-s1': [{batch:'b', text:'NOID', why:'w', at: Date.now()},"
                   " {id:'x', batch:'b', text:'STRAT', why:'w', at: String(Date.now())}, {id:'e', text:'  '}]});\n_renderTray();\n")
    out = eval_js(src, "{tray: tray(), legacy: 'ghost.queued.restore' in store, ids: stored()['web-s1'].map(i => typeof i.id + typeof i.at)}")
    assert out == {"tray": ["NOID", "STRAT"], "legacy": False, "ids": ["stringnumber", "stringnumber"]}


def test_the_old_key_stays_until_every_item_is_kept():
    """r6a11 (probe p3): localStorage and sessionStorage both full."""
    src = _harness(_FULL + "const realSS = sessionStorage.setItem; sessionStorage.setItem = () => { throw new Error('quota'); };\n"
        "store['ghost.queued.restore'] = JSON.stringify({'web-s1': [{id:'o1', batch:'b', text:'OLD', why:'w', at: Date.now()}]});\n"
        "_renderTray(); const mid = { tray: tray(), legacy: 'ghost.queued.restore' in store };\n"
        "full = false; sessionStorage.setItem = realSS; _renderTray(); _renderTray();\n")
    out = eval_js(src, "{mid, tray: tray(), legacy: 'ghost.queued.restore' in store, stored: stored()['web-s1'].map(i => i.text)}")
    assert out == {"mid": {"tray": ["OLD"], "legacy": True}, "tray": ["OLD"], "legacy": False, "stored": ["OLD"]}


def test_a_partly_failed_migration_keeps_the_old_key_without_duplicating():
    src = _harness("const realSet = localStorage.setItem; const realSS = sessionStorage.setItem;\n"
        "localStorage.setItem = (k, v) => { if (k.endsWith('.o2')) throw new Error('quota'); realSet(k, v); };\n"
        "sessionStorage.setItem = () => { throw new Error('quota'); };\n"
        "store['ghost.queued.restore'] = JSON.stringify({'web-s1': [{id:'o1', batch:'b', text:'ONE', why:'w', at: Date.now()},"
        " {id:'o2', batch:'b', text:'TWO', why:'w', at: Date.now()}]});\n"
        "_renderTray(); _renderTray();\n")
    out = eval_js(src, "{tray: tray(), legacy: 'ghost.queued.restore' in store}")
    assert out == {"tray": ["ONE", "TWO"], "legacy": True}


def test_an_old_code_tab_rewriting_the_map_never_brings_back_a_dismissed_item():
    """r6a11 (probe p4)."""
    src = _harness("const oldMap = JSON.stringify({'web-s1': [{id:'o1', batch:'b', text:'OLD', why:'w', at: Date.now()}]});\n"
        "store['ghost.queued.restore'] = oldMap; _renderTray(); _trayRemove('o1');\n"
        "const after = tray(); store['ghost.queued.restore'] = oldMap; _renderTray();\n")
    out = eval_js(src, "{after, again: tray(), legacy: 'ghost.queued.restore' in store}")
    assert out == {"after": [], "again": [], "legacy": False}


def test_a_migration_into_session_storage_counts_as_kept():
    """localStorage full, sessionStorage fine: the old key goes, the item stays (unsaved, warned)."""
    src = _harness(_FULL + "store['ghost.queued.restore'] = JSON.stringify({'web-s1': [{id:'o1', batch:'b', text:'OLD', why:'w', at: Date.now()}]});\n"
                   "_renderTray();\n")
    out = eval_js(src, "{tray: tray(), legacy: 'ghost.queued.restore' in store, unsaved: JSON.parse(sstore['ghost.queued.restore.unsaved']).map(i => i.text)}")
    assert out == {"tray": ["OLD"], "legacy": False, "unsaved": ["OLD"]}


# ── r6a12: an id-less old item gets the same id on every run ───────────────

def test_a_retried_migration_never_copies_an_id_less_item_again():
    """r6a12 MINOR 1 (probe a1): one store full for one item, so the old key stays."""
    src = _harness("const realSet = localStorage.setItem;\n"
        "localStorage.setItem = (k, v) => { if (k.startsWith('ghost.queued.restore.item.') && v.includes('BIG')) throw new Error('quota'); realSet(k, v); };\n"
        "sessionStorage.setItem = () => { throw new Error('quota'); };\n"
        "store['ghost.queued.restore'] = JSON.stringify({'web-s1': [{batch:'b', text:'NOID', why:'w'},"
        " {id:'o2', batch:'b', text:'BIG', why:'w', at: Date.now()}]});\n"
        "_renderTray(); const first = JSON.stringify(stored());\n"
        "for (let i = 0; i < 5; i++) _renderTray();\n")
    out = eval_js(src, "{same: first === JSON.stringify(stored()), tray: tray(), legacy: 'ghost.queued.restore' in store,"
                  " id: stored()['web-s1'][0].id}")
    assert out["same"] is True and out["tray"] == ["NOID", "BIG"] and out["legacy"] is True
    assert re.fullmatch(r"legacy-[0-9a-f]+", out["id"])


def test_a_retried_migration_never_grows_the_unsaved_list():
    """r6a12 MINOR 1 (probe a2): both stores full."""
    src = _harness(_FULL + "sessionStorage.setItem = () => { throw new Error('quota'); };\n"
        "store['ghost.queued.restore'] = JSON.stringify({'web-s1': [{batch:'b', text:'NOID', why:'w', at: 'x'}]});\n"
        "_renderTray(); const at = _unsavedList()[0].at;\n"
        "const realNow = Date.now; let skew = 0; Date.now = () => realNow() + (skew += 1000);\n"
        "for (let i = 0; i < 5; i++) _renderTray();\n")
    out = eval_js(src, "{n: _unsavedList().length, sameAt: _unsavedList()[0].at === at, tray: tray()}")
    assert out == {"n": 1, "sameAt": True, "tray": ["NOID"]}


def test_legacy_ids_are_stable_and_distinct():
    src = _harness("const it = {text: 'same', batch: 'b'};\n"
        "const a = _legacyId('web-s1', it, 0), b = _legacyId('web-s1', it, 0), c = _legacyId('web-s1', it, 1), d = _legacyId('web-s2', it, 0);\n"
        "const Z = String.fromCharCode(0);\n"   # a NUL-joined key made these two the same
        "const e = _legacyId('a', {text: 'b' + Z + 'c', batch: ''}, 0), f = _legacyId('a' + Z + 'b', {text: 'c', batch: ''}, 0);\n")
    out = eval_js(src, "{same: a === b, distinct: new Set([a, c, d]).size, sep: e !== f}")
    assert out == {"same": True, "distinct": 3, "sep": True}
    assert "legacyAt" not in APP


def test_an_item_without_a_time_first_seen_after_any_delay_is_kept():
    """r6a13: a missing time is the migration time, never an old stand-in."""
    src = _harness("const realNow = Date.now; Date.now = () => realNow() + 30 * 24 * 3600 * 1000;\n"
        "store['ghost.queued.restore'] = JSON.stringify({'web-s1': [{id:'o1', batch:'b', text:'LATE', why:'w'}]});\n"
        "_renderTray();\n")
    out = eval_js(src, "{tray: tray(), legacy: 'ghost.queued.restore' in store, fresh: stored()['web-s1'][0].at >= realNow() + 30 * 24 * 3600 * 1000 - 1000}")
    assert out == {"tray": ["LATE"], "legacy": False, "fresh": True}


def test_two_tabs_never_trade_writes_over_a_stuck_migration():
    """r6a12 MINOR 2: a re-read after a storage event writes nothing to localStorage."""
    src = _harness("let writes = 0; const realSet = localStorage.setItem;\n"
        "localStorage.setItem = (k, v) => { writes++; if (k.startsWith('ghost.queued.restore.item.') && v.includes('BIG')) throw new Error('quota'); realSet(k, v); };\n"
        "sessionStorage.setItem = () => { throw new Error('quota'); };\n"
        "store['ghost.queued.restore'] = JSON.stringify({'web-s1': [{batch:'b', text:'NOID', why:'w'},"
        " {id:'o2', batch:'b', text:'BIG', why:'w', at: Date.now()}]});\n"
        "_renderTray(); const keysAfterFirst = Object.keys(store).sort().join(); writes = 0;\n"
        "for (let i = 0; i < 5; i++) _renderTray();\n")
    out = eval_js(src, "{same: keysAfterFirst === Object.keys(store).sort().join(), succeeded: writes}")
    # retries of BIG still attempt (and fail) its own key; nothing new ever lands in localStorage
    assert out["same"] is True
