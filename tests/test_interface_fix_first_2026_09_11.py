"""Web client "fix first" batch (2026-09-11) — eight user-visible defects.

Each section names the WORLD in which its test fails (the pre-fix code, or
the obvious regression), and the behavioural ones are EXECUTED under node
or against the FastAPI app rather than matched as text — a text pin over
this file's own vocabulary would pass with `if (false)` around every fix.

  1. Blank bubble while reasoning   — the typing dots stayed until the first
     VISIBLE character, not the first chunk (which opens a <think> block).
  2. Retry / regenerate / edit      — cut the conversation at a known
     history index, refill the composer, optionally resend.
  3. IME Enter                      — a composing Enter must not send.
  4. Global dblclick preventDefault — deleted (pin the deletion).
  5. Scroll-jacking                 — only the user's own message scrolls
     unconditionally; other appends follow only while pinned.
  6. Notification permission        — no prompt on the first click anywhere;
     an explicit control reads a three-world state.
  7. SSE heartbeat                  — a parked reader emits a comment frame
     every SSE_PING_S, which also recovers a lost wake.
  8. Stop is one call               — the proxy mints X-Request-ID, sends it
     upstream and on the response, and /api/chat/cancel cancels the agent.
"""

import asyncio
import re
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.helpers import eval_js, extract_js_function, strip_js_comments

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import interface.server as server  # noqa: E402

_STATIC = _ROOT / "interface" / "static"


@pytest.fixture(scope="module")
def app_js() -> str:
    return (_STATIC / "app.js").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def app_js_nc(app_js) -> str:
    return strip_js_comments(app_js)


@pytest.fixture(scope="module")
def index_html() -> str:
    return (_STATIC / "index.html").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def workspace_js() -> str:
    return strip_js_comments((_STATIC / "workspace.js").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def notifications_js() -> str:
    return strip_js_comments((_STATIC / "notifications.js").read_text(encoding="utf-8"))


def _extract_listener(source: str, needle: str) -> str:
    """Body of `<needle>(e) => { … })`, brace-matched, as `function handler(e)`."""
    i = source.index(needle)
    j = source.index("{", source.index("=>", i))
    depth = 0
    for k in range(j, len(source)):
        if source[k] == "{":
            depth += 1
        elif source[k] == "}":
            depth -= 1
            if depth == 0:
                return "function handler(e) " + source[j:k + 1]
    raise AssertionError(f"unbalanced braces after {needle!r}")


# A minimal element: classList, dataset, children, textContent/innerHTML.
_DOM_SHIM = """
function _el(tag) {
    const cls = new Set();
    return {
        tag, dataset: {}, children: [], _text: '', innerHTML: '', parent: null,
        classList: {
            add: (...c) => c.forEach(x => cls.add(x)),
            remove: (...c) => c.forEach(x => cls.delete(x)),
            contains: (c) => cls.has(c),
            toggle: (c, on) => { on ? cls.add(c) : cls.delete(c); },
        },
        get className() { return [...cls].join(' '); },
        set className(v) { cls.clear(); v.split(/\\s+/).filter(Boolean).forEach(x => cls.add(x)); },
        get textContent() { return this._text; },
        set textContent(v) { this._text = String(v); this.children = []; },
        appendChild(c) { c.parent = this; this.children.push(c); return c; },
        remove() { if (this.parent) this.parent.children = this.parent.children.filter(x => x !== this); },
        querySelector() { return null; },
        querySelectorAll() { return []; },
        setAttribute() {},
        addEventListener() {},
    };
}
globalThis.document = {
    createElement: _el,
    createTextNode: (t) => ({ text: t, remove() {} }),
    getElementById: () => null,
    body: { classList: { contains: () => false, toggle() {}, add() {}, remove() {} } },
};
globalThis.requestAnimationFrame = (fn) => { fn(); return 1; };
"""


# ═══════════════════════════════════════════════════════════════════════════
# 1. The bubble reveals on the first VISIBLE character
# ═══════════════════════════════════════════════════════════════════════════

class TestBubbleRevealsOnVisibleText:
    def _run(self, app_js, acc):
        fn = (extract_js_function(app_js, "_stripInternalTags")
              + extract_js_function(app_js, "_revealAgentBubble")
              + extract_js_function(app_js, "_renderStreamingContent"))
        pre = _DOM_SHIM + f"""
const status = [];
let currentAgentMessageDiv = _el('div');
currentAgentMessageDiv.className = 'message agent thinking';
currentAgentMessageDiv.appendChild(_el('span'));   // the typing indicator
let currentAccumulatedContent = {acc!r};
function renderMarkdown(t) {{ return '<p>' + t + '</p>'; }}
function setTurnStatusDesc(t, i) {{ status.push(t); }}
const activeFace = {{ setPhase(p) {{ status.push('phase:' + p); }} }};   // 2026-09-11: the reveal sets the write gait
const chatLog = {{ scrollHeight: 100, scrollTop: 0, clientHeight: 100 }};
function scrollToBottomDuringStream() {{}}
function _noteNewContentBelow() {{}}
function decorateCodeBlocks() {{}}
"""
        return eval_js(pre + fn, """(() => { _renderStreamingContent(); return {
            thinking: currentAgentMessageDiv.classList.contains('thinking'),
            html: currentAgentMessageDiv.innerHTML,
            kids: currentAgentMessageDiv.children.length,
            status,
        }; })()""")

    def test_open_think_block_keeps_the_dots(self, app_js):
        """FAILS in the pre-fix world: the first chunk cleared the bubble and
        rendered the empty string, so the user watched an empty bubble for
        the whole reasoning phase."""
        out = self._run(app_js, "<think>let me consider the")
        assert out["thinking"] is True, out
        assert out["kids"] == 1, "the typing indicator was torn down"
        assert out["html"] == "", "an empty render replaced the indicator"
        assert out["status"] == [], "'writing the reply' was claimed with nothing to show"

    def test_first_visible_character_reveals_once(self, app_js):
        out = self._run(app_js, "<think>a</think>Hi")
        assert out["thinking"] is False
        assert out["html"] == "<p>Hi</p>"
        assert out["kids"] == 0, "the indicator survived the reveal"
        assert out["status"] == ["writing the reply…", "phase:write"]

    def test_reveal_is_idempotent(self, app_js):
        fn = extract_js_function(app_js, "_revealAgentBubble")
        out = eval_js(_DOM_SHIM + fn, """(() => {
            const d = _el('div'); d.className = 'message agent thinking';
            return [_revealAgentBubble(d), _revealAgentBubble(d), _revealAgentBubble(null)];
        })()""")
        assert out == [True, False, False]

    def test_first_chunk_no_longer_tears_down_the_indicator(self, app_js_nc):
        """The old site: `if (currentAccumulatedContent === "")` removed
        `.thinking` and cleared textContent. Both must be gone from there."""
        i = app_js_nc.index('if (currentAccumulatedContent === "") {')
        window = app_js_nc[i:i + 600]
        assert "classList.remove('thinking')" not in window
        assert 'textContent = ""' not in window

    def test_a_reasoning_only_reply_still_becomes_something(self, app_js_nc):
        i = app_js_nc.index("if (currentAccumulatedContent) {\n            chatHistory.push")
        window = app_js_nc[i:i + 900]
        assert "_revealAgentBubble(currentAgentMessageDiv)" in window
        assert "No reply text" in window


# ═══════════════════════════════════════════════════════════════════════════
# 2. Retry / regenerate / edit-and-resend
# ═══════════════════════════════════════════════════════════════════════════

class TestRetryRegenerateEdit:
    HISTORY = ("[{role:'user', content:'q1'}, {role:'assistant', content:'a1'},"
               " {role:'user', content:'q2'}, {role:'assistant', content:'a2'}]")

    def _run(self, app_js, expr, history=None, processing="false", log_hidx=(0, 1, 2, 3)):
        fn = "".join(extract_js_function(app_js, n) for n in (
            "_stampHistoryIndex", "_historyEntryText", "_userTurnIndexFor",
            "_truncateToUserTurn", "retryLastTurn"))
        kids = ", ".join(f"Object.assign(_el('div'), {{dataset: {{hidx: '{h}'}}}})" for h in log_hidx)
        pre = _DOM_SHIM + f"""
let isProcessingRequest = {processing};
let chatHistory = {history or self.HISTORY};
const saved = [];
function saveChatState() {{ saved.push(chatHistory.length); }}
const sent = [];
function sendTypedMessage() {{ sent.push(chatInput.value); }}
const chatInput = {{ value: '', style: {{}}, focused: false, sel: null,
    dispatchEvent() {{}}, focus() {{ this.focused = true; }},
    setSelectionRange(a, b) {{ this.sel = [a, b]; }} }};
const chatLog = _el('div');
[{kids}].forEach(k => chatLog.appendChild(k));
"""
        return eval_js(pre + fn, expr)

    def test_regenerate_from_an_agent_bubble_cuts_at_its_question(self, app_js):
        out = self._run(app_js, """(() => {
            const r = _truncateToUserTurn(3, true);
            return { r, hist: chatHistory.map(m => m.content), sent,
                     left: chatLog.children.map(c => c.dataset.hidx), saved };
        })()""")
        assert out["r"]["ok"] is True and out["r"]["text"] == "q2"
        assert out["hist"] == ["q1", "a1"], "history was not cut before the question"
        assert out["left"] == ["0", "1"], "the transcript kept bubbles from the redone turn"
        assert out["sent"] == ["q2"], "Regenerate did not resend the question"
        assert out["saved"] == [2], "the cut was not persisted"

    def test_edit_from_a_user_bubble_refills_without_sending(self, app_js):
        out = self._run(app_js, """(() => {
            const r = _truncateToUserTurn(2, false);
            return { r, hist: chatHistory.length, sent, value: chatInput.value,
                     focused: chatInput.focused, sel: chatInput.sel };
        })()""")
        assert out["r"]["ok"] is True
        assert out["hist"] == 2 and out["sent"] == []
        assert out["value"] == "q2" and out["focused"] is True
        assert out["sel"] == [2, 2], "the caret was not placed at the end"

    def test_retry_last_resends_the_last_user_turn(self, app_js):
        out = self._run(app_js, "(() => { const r = retryLastTurn(); return {r, sent, n: chatHistory.length}; })()")
        assert out["r"]["ok"] is True and out["sent"] == ["q2"] and out["n"] == 2

    def test_retry_after_a_stop_with_a_trailing_system_bubble(self, app_js):
        """The Stop path pushes an aborted assistant entry and a system pill
        with NO history index; the cut must still take the pill with it."""
        out = self._run(app_js, "(() => { const r = retryLastTurn(); return {r, left: chatLog.children.length}; })()",
                        history="[{role:'user', content:'q1'}, {role:'assistant', content:'partial\\n\\n*[Aborted]*'}]",
                        log_hidx=("0", "1", "undefined"))
        assert out["r"]["ok"] is True
        assert out["left"] == 0, "the 'Stopped by user' pill outlived the retried turn"

    def test_refuses_while_a_turn_is_running(self, app_js):
        out = self._run(app_js, "(() => { const r = retryLastTurn(); return {r, sent, n: chatHistory.length}; })()",
                        processing="true")
        assert out["r"]["ok"] is False and "running" in out["r"]["reason"]
        assert out["sent"] == [] and out["n"] == 4, "a running turn was cut underneath"

    def test_refuses_to_drop_an_attachment_silently(self, app_js):
        out = self._run(app_js, "(() => { const r = retryLastTurn(); return {r, n: chatHistory.length}; })()",
                        history="[{role:'user', content:[{type:'text', text:'what is this'}, {type:'image_url', image_url:{url:'data:,x'}}]}]")
        assert out["r"]["ok"] is False and "attachment" in out["r"]["reason"]
        assert out["n"] == 1

    def test_empty_history_has_nothing_to_resend(self, app_js):
        out = self._run(app_js, "retryLastTurn()", history="[]", log_hidx=())
        assert out["ok"] is False

    def test_bubbles_carry_their_history_index(self, app_js_nc):
        """Every site that pushes a user/assistant entry stamps its bubble;
        renderHistoryToLog stamps by position. Without the stamp the menu
        offers nothing (see the workspace test) — a silent feature death."""
        assert app_js_nc.count("_stampHistoryIndex(") >= 5, app_js_nc.count("_stampHistoryIndex(")
        i = app_js_nc.index("function renderHistoryToLog")
        assert "_stampHistoryIndex(div, hidx)" in app_js_nc[i:i + 2600]

    def test_error_and_stop_bubbles_are_retryable(self, app_js_nc):
        for phrase in ("Stopped by user.", "Network Error: ${_m}", "Error${_type}: ${_msg}${_eid}"):
            i = app_js_nc.index(phrase)
            assert "addRetryableSystemMessage(" in app_js_nc[i - 60:i], phrase

    def test_retryable_message_builds_a_button_that_retries(self, app_js):
        fn = (extract_js_function(app_js, "addRetryableSystemMessage"))
        out = eval_js(_DOM_SHIM + """
const made = [];
function addMessage(role, text) { const d = _el('div'); d.role = role; d.textContent = text; made.push(d); return d; }
const retried = [];
function retryLastTurn() { retried.push(1); return { ok: true }; }
globalThis.window = {};
""" + fn.replace("btn.addEventListener('click',", "btn.onclick = ("), """(() => {
            const d = addRetryableSystemMessage('Network Error: x');
            const btn = d.children.find(c => c.tag === 'button');
            btn.onclick({ stopPropagation() {} });
            return { role: d.role, label: btn.textContent, retried: retried.length };
        })()""")
        assert out == {"role": "system", "label": "Retry", "retried": 1}

    def test_menu_offers_regenerate_and_edit_by_history_index(self, workspace_js):
        i = workspace_js.index("function openMessageMenu")
        body = workspace_js[i:i + 3000]
        assert "msgDiv.dataset.hidx" in body
        assert "Core.resendFromIndex(hidx)" in body and "Core.editFromIndex(hidx)" in body
        assert "'Regenerate'" in body and "'Edit & resend'" in body
        assert "!Core.isProcessing()" in body, "the menu offers a cut mid-turn"

    def test_core_exports_the_three_entry_points(self, app_js_nc):
        i = app_js_nc.index("window.GhostCore = {")
        core = app_js_nc[i:]
        for name in ("resendFromIndex:", "editFromIndex:", "retryLastTurn,"):
            assert name in core, name


# ═══════════════════════════════════════════════════════════════════════════
# 3. Composer keydown: IME guard + ArrowUp recall
# ═══════════════════════════════════════════════════════════════════════════

class TestComposerKeydown:
    def _run(self, app_js, events, history="[{role:'user', content:'last one'}]", value="''"):
        handler = _extract_listener(app_js, "chatInput.addEventListener('keydown'")
        helpers = (extract_js_function(app_js, "_historyEntryText")
                   + extract_js_function(app_js, "_userTurnIndexFor"))
        pre = f"""
let isProcessingRequest = false;
let chatHistory = {history};
const sent = [], prevented = [];
function sendTypedMessage() {{ sent.push(1); }}
const chatInput = {{ value: {value}, dispatchEvent() {{}}, setSelectionRange() {{}} }};
"""
        return eval_js(pre + helpers + handler, f"""(() => {{
            for (const ev of {events}) {{ ev.preventDefault = () => prevented.push(ev.key); handler(ev); }}
            return {{ sent: sent.length, prevented, value: chatInput.value }};
        }})()""")

    def test_a_composing_enter_does_not_send(self, app_js):
        """FAILS pre-fix: the IME candidate-confirm Enter sent half a word."""
        out = self._run(app_js, "[{key:'Enter', shiftKey:false, isComposing:true}]")
        assert out["sent"] == 0 and out["prevented"] == []

    def test_legacy_keycode_229_does_not_send(self, app_js):
        out = self._run(app_js, "[{key:'Enter', shiftKey:false, isComposing:false, keyCode:229}]")
        assert out["sent"] == 0

    def test_a_plain_enter_still_sends(self, app_js):
        out = self._run(app_js, "[{key:'Enter', shiftKey:false, isComposing:false}]")
        assert out["sent"] == 1 and out["prevented"] == ["Enter"]

    def test_shift_enter_is_a_newline(self, app_js):
        out = self._run(app_js, "[{key:'Enter', shiftKey:true, isComposing:false}]")
        assert out["sent"] == 0 and out["prevented"] == []

    def test_arrow_up_in_an_empty_composer_recalls_the_last_message(self, app_js):
        out = self._run(app_js, "[{key:'ArrowUp'}]")
        assert out["value"] == "last one" and out["prevented"] == ["ArrowUp"]

    def test_arrow_up_with_text_in_the_composer_is_left_alone(self, app_js):
        out = self._run(app_js, "[{key:'ArrowUp'}]", value="'typing'")
        assert out["value"] == "typing" and out["prevented"] == []

    def test_composer_declares_its_intent_to_the_keyboard(self, index_html):
        m = re.search(r'<textarea id="chat-input"[^>]*>', index_html)
        assert m and 'enterkeyhint="send"' in m.group(0) and 'aria-label="Message"' in m.group(0)


# ═══════════════════════════════════════════════════════════════════════════
# 4. The document-wide dblclick guard is gone (pin the deletion)
# ═══════════════════════════════════════════════════════════════════════════

def test_no_global_dblclick_preventdefault(app_js_nc):
    assert "addEventListener('dblclick'" not in app_js_nc, (
        "the double-click guard is back: it breaks word selection everywhere")


# ═══════════════════════════════════════════════════════════════════════════
# 5. Reading position: follow only while pinned
# ═══════════════════════════════════════════════════════════════════════════

class TestScrollFollowsOnlyWhenPinned:
    def _run(self, app_js, role, dist):
        fn = (extract_js_function(app_js, "_isNearBottom")
              + extract_js_function(app_js, "scrollToBottomIfPinned")
              + extract_js_function(app_js, "addMessage"))
        pre = _DOM_SHIM + f"""
const scrolled = [], pill = [];
function scrollToBottom() {{ scrolled.push(1); }}
function _noteNewContentBelow() {{ pill.push(1); }}
function dismissEmptyStateHero() {{}}
function _maybeInsertDaySeparator() {{}}
function renderMarkdown(t) {{ return t; }}
function decorateCodeBlocks() {{}}
function decorateMessageActions() {{}}
const chatLog = Object.assign(_el('div'), {{ scrollHeight: 2000, clientHeight: 500, scrollTop: 2000 - 500 - {dist} }});
"""
        return eval_js(pre + fn, f"(() => {{ addMessage({role!r}, 'x'); return {{scrolled: scrolled.length, pill: pill.length}}; }})()")

    def test_an_agent_message_does_not_yank_a_reader_who_scrolled_up(self, app_js):
        """FAILS pre-fix: addMessage called scrollToBottom unconditionally."""
        out = self._run(app_js, "agent", dist=800)
        assert out == {"scrolled": 0, "pill": 1}, out

    def test_a_system_notice_does_not_yank_either(self, app_js):
        assert self._run(app_js, "system", dist=800) == {"scrolled": 0, "pill": 1}

    def test_pinned_near_the_bottom_still_follows(self, app_js):
        assert self._run(app_js, "agent", dist=40) == {"scrolled": 1, "pill": 0}

    def test_your_own_message_always_scrolls(self, app_js):
        assert self._run(app_js, "user", dist=800) == {"scrolled": 1, "pill": 0}

    def test_the_other_unconditional_sites_are_gated(self, app_js_nc):
        assert "setTimeout(scrollToBottom, 100)" not in app_js_nc, "turn-end still yanks"
        assert app_js_nc.count("setTimeout(scrollToBottomIfPinned, 100)") == 2
        i = app_js_nc.index("const syncBodyHeight = () => {")
        assert app_js_nc[i:i + 120].count("scrollToBottomIfPinned()") == 1, (
            "a keyboard show/hide (viewport resize) still yanks the reader")

    def test_streaming_lights_the_pill_when_not_at_bottom(self, app_js_nc):
        i = app_js_nc.index("function _renderStreamingContent")
        body = app_js_nc[i:i + 1200]
        assert "else _noteNewContentBelow();" in body

    def test_the_pill_exists_and_returning_to_the_bottom_hides_it(self, index_html, app_js_nc):
        assert 'id="chat-resume"' in index_html
        i = app_js_nc.index("document.body.classList.toggle('is-reading', isReading);")
        assert "if (!isReading) _hideNewContentPill();" in app_js_nc[i:i + 120]
        i = app_js_nc.index("function scrollToBottom()")
        assert "_hideNewContentPill();" in app_js_nc[i:i + 220]


# ═══════════════════════════════════════════════════════════════════════════
# 6. Notification permission behind an explicit control
# ═══════════════════════════════════════════════════════════════════════════

class TestNotificationPermissionIsDeliberate:
    def test_no_prompt_on_the_first_click_anywhere(self, app_js_nc):
        """Pin the deletion: the old listener asked on any click, once."""
        for i in [m.start() for m in re.finditer(r"document\.addEventListener\('click'", app_js_nc)]:
            assert "requestPermission" not in app_js_nc[i:i + 500], (
                "the OS permission prompt is still wired to a click anywhere")

    def _state(self, app_js, ios, standalone, notification):
        fn = extract_js_function(app_js, "pushPermissionState")
        pre = f"""
const isIOS = {ios}; const isStandalonePWA = {standalone};
globalThis.Notification = {notification};
"""
        return eval_js(pre + fn, "pushPermissionState()")

    def test_ios_safari_tab_needs_an_install_not_a_prompt(self, app_js):
        st = self._state(app_js, "true", "false", "{permission: 'default'}")
        assert st["needsInstall"] is True and st["supported"] is False

    def test_installed_ios_pwa_is_supported(self, app_js):
        st = self._state(app_js, "true", "true", "{permission: 'default'}")
        assert st == {"supported": True, "needsInstall": False, "permission": "default"}

    def test_no_notification_api_is_unsupported(self, app_js):
        st = self._state(app_js, "false", "false", "undefined")
        assert st["supported"] is False and st["permission"] == "unsupported"

    def test_granted_is_reported_verbatim(self, app_js):
        assert self._state(app_js, "false", "false", "{permission: 'granted'}")["permission"] == "granted"

    def test_request_subscribes_on_grant_and_never_prompts_when_settled(self, app_js):
        fn = (extract_js_function(app_js, "pushPermissionState")
              + extract_js_function(app_js, "requestNotificationPermission"))
        out = eval_js("""
const isIOS = false, isStandalonePWA = false;
const subs = [], prompts = [];
function ensurePushSubscription() { subs.push(1); }
globalThis.Notification = { permission: 'default',
    requestPermission: async () => { prompts.push(1); Notification.permission = 'granted'; return 'granted'; } };
""" + fn, """await (async () => {
            const a = await requestNotificationPermission();      // default → prompt → granted
            const b = await requestNotificationPermission();      // granted → no prompt, re-subscribe
            Notification.permission = 'denied';
            const c = await requestNotificationPermission();      // denied → no prompt
            return { a, b, c, prompts: prompts.length, subs: subs.length };
        })()""")
        assert out == {"a": "granted", "b": "granted", "c": "denied", "prompts": 1, "subs": 2}

    def test_the_bell_panel_renders_the_control(self, notifications_js):
        i = notifications_js.index("function pushRow()")
        body = notifications_js[i:i + 2600]
        assert "Core.pushPermissionState()" in body
        assert "Core.requestNotificationPermission()" in body
        assert "Add to Home Screen" in body, "the iOS install path is not explained"
        j = notifications_js.index("function renderList()")
        assert "pushRow()" in notifications_js[j:j + 300], "the row is built but never shown"

    def test_core_exports_both(self, app_js_nc):
        core = app_js_nc[app_js_nc.index("window.GhostCore = {"):]
        assert "pushPermissionState," in core and "requestNotificationPermission," in core


# ═══════════════════════════════════════════════════════════════════════════
# 7. SSE heartbeat on a parked reader
# ═══════════════════════════════════════════════════════════════════════════

def _task(**over):
    t = {"buffer": [], "buffer_size": 0, "done": False, "error": None,
         "truncated": False, "new_data_event": asyncio.Event()}
    t.update(over)
    return t


async def _next(gen, timeout=2.0):
    return await asyncio.wait_for(gen.__anext__(), timeout)


class TestHeartbeat:
    @pytest.mark.asyncio
    async def test_a_silent_task_emits_a_ping_within_the_interval(self, monkeypatch):
        """FAILS pre-fix: the reader awaited the event with no timeout, so
        the test's own 2s bound expired with NO bytes on the wire."""
        monkeypatch.setattr(server, "SSE_PING_S", 0.05)
        t = _task()
        gen = server._relay_task_stream(t, 0)
        first = await _next(gen)
        assert first == b": ping\n\n"
        # Then real data, then the terminal break.
        t["buffer"].append(b"data: {}\n\n")
        t["new_data_event"].set()
        assert await _next(gen) == b"data: {}\n\n"
        t["done"] = True
        t["new_data_event"].set()
        with pytest.raises(StopAsyncIteration):
            await _next(gen)

    @pytest.mark.asyncio
    async def test_ping_is_an_sse_comment_the_client_skips(self):
        """The bundled client keeps only `data:` lines; a comment frame must
        parse to nothing. Mirrors app.js: `if (!trimmedLine.startsWith("data: ")) continue`."""
        frame = b": ping\n\n".decode()
        kept = [ln for ln in frame.split("\n") if ln.strip() and ln.strip().startswith("data: ")]
        assert kept == []

    @pytest.mark.asyncio
    async def test_a_lost_wake_is_recovered_by_the_next_ping(self, monkeypatch):
        """Two readers share ONE Event. If reader B clears it between reader
        A's clear and wait, A misses the set. Pre-fix A parked forever (a
        permanent spinner); now the ping timeout re-checks the buffer."""
        monkeypatch.setattr(server, "SSE_PING_S", 0.05)
        t = _task()
        gen = server._relay_task_stream(t, 0)
        # Let A reach its wait().
        pending = asyncio.ensure_future(gen.__anext__())
        await asyncio.sleep(0.01)
        # Producer appends and sets; "reader B" clears immediately after.
        t["buffer"].append(b"data: late\n\n")
        t["new_data_event"].set()
        t["new_data_event"].clear()
        got = await asyncio.wait_for(pending, 2.0)
        if got == b": ping\n\n":          # a ping may land first
            got = await _next(gen)
        assert got == b"data: late\n\n"

    @pytest.mark.asyncio
    async def test_wait_helper_reports_timeout_vs_wake(self, monkeypatch):
        t = _task()
        assert await server._wait_for_new_data(t, timeout=0.02) is False
        t["new_data_event"].set()
        assert await server._wait_for_new_data(t, timeout=0.5) is True

    def test_both_readers_use_the_shared_relay(self):
        import inspect
        src = inspect.getsource(server.chat_proxy) + inspect.getsource(server.chat_resume_proxy)
        assert src.count("_relay_task_stream(task, ") == 2
        assert 'await task["new_data_event"].wait()' not in src, "an unbounded wait is back"

    def test_default_interval_is_env_tunable_and_sane(self):
        assert 5.0 <= server.SSE_PING_S <= 30.0

    @pytest.mark.asyncio
    async def test_resume_offset_is_clamped_not_trusted(self):
        t = _task(buffer=[b"a", b"b"], done=True)
        out = [c async for c in server._relay_task_stream(t, 99)]
        assert out == []
        out = [c async for c in server._relay_task_stream(t, -5)]
        assert out == [b"a", b"b"]

    @pytest.mark.asyncio
    async def test_terminal_markers_still_ride_the_relay(self):
        t = _task(done=True, truncated=True, truncated_reason="per-task buffer cap exceeded")
        out = b"".join([c async for c in server._relay_task_stream(t, 0)])
        assert b"BufferCapExceeded" in out
        t = _task(done=True, error="agent returned HTTP 503")
        out = b"".join([c async for c in server._relay_task_stream(t, 0)])
        assert b"503" in out


# ═══════════════════════════════════════════════════════════════════════════
# 8. Stop is one call: proxy-minted request id, cancel reaches the agent
# ═══════════════════════════════════════════════════════════════════════════

def _fake_stream_client(chunks=(b"data: {}\n\n",)):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()

    async def _aiter(*a, **k):
        for c in chunks:
            yield c
    resp.aiter_bytes = MagicMock(side_effect=_aiter)
    ctx = AsyncMock()
    ctx.__aenter__.return_value = resp
    ctx.__aexit__.return_value = None
    client = MagicMock()
    client.stream = MagicMock(return_value=ctx)
    return client


class TestStopIsOneCall:
    @pytest.mark.asyncio
    async def test_proxy_mints_the_id_sends_it_upstream_and_exposes_it(self):
        """FAILS pre-fix: no X-Request-ID anywhere — the id was born inside
        the agent and surfaced on the first content frame, after the
        thinking phase, i.e. exactly when Stop gets pressed."""
        client = _fake_stream_client()
        req = MagicMock()
        req.json = AsyncMock(return_value={"stream": True, "messages": [{"role": "user", "content": "hi"}]})
        req.headers = {}
        with patch.object(server, "_get_http_client", return_value=client):
            response = await server.chat_proxy(req)
            async for _ in response.body_iterator:
                pass
        task_id = response.headers["x-task-id"]
        rid = response.headers["x-request-id"]
        try:
            assert re.fullmatch(r"[0-9a-f]{8}", rid), rid
            assert "X-Request-ID" in response.headers["access-control-expose-headers"]
            _, kwargs = client.stream.call_args
            assert kwargs["headers"]["X-Request-ID"] == rid, "the agent was not told the id"
            assert server.active_chat_tasks[task_id]["request_id"] == rid
        finally:
            server.active_chat_tasks.pop(task_id, None)

    @pytest.mark.asyncio
    async def test_cancel_cancels_the_agent_turn_by_the_minted_id(self):
        tid = "fix-first-cancel"
        server.active_chat_tasks[tid] = _task(request_id="abcd1234", background_task=MagicMock())
        post = AsyncMock(return_value=MagicMock(status_code=200, json=lambda: {"cancelled": True, "request_id": "abcd1234"}))
        client = MagicMock(post=post)
        try:
            with patch.object(server, "_get_http_client", return_value=client):
                out = await server.chat_cancel_proxy(tid)
            assert out["status"] == "cancelled" and out["request_id"] == "abcd1234"
            assert out["agent"]["cancelled"] is True
            args, kwargs = post.call_args
            assert args[0].endswith("/api/turn/cancel")
            assert kwargs["json"] == {"request_id": "abcd1234"}
            assert kwargs["headers"]["X-Ghost-Key"] == server.GHOST_API_KEY
            t = server.active_chat_tasks[tid]
            assert t["done"] and t["cancelled"] and t["new_data_event"].is_set()
        finally:
            server.active_chat_tasks.pop(tid, None)

    @pytest.mark.asyncio
    async def test_a_refused_cancel_is_reported_not_swallowed(self):
        tid = "fix-first-refused"
        server.active_chat_tasks[tid] = _task(request_id="abcd1234", background_task=None)
        post = AsyncMock(return_value=MagicMock(status_code=404, json=lambda: {"cancelled": False, "detail": "no such turn"}))
        try:
            with patch.object(server, "_get_http_client", return_value=MagicMock(post=post)):
                out = await server.chat_cancel_proxy(tid)
            assert out["agent"]["cancelled"] is False
            assert out["agent"]["status_code"] == 404
            assert out["agent"]["detail"] == "no such turn"
        finally:
            server.active_chat_tasks.pop(tid, None)

    @pytest.mark.asyncio
    async def test_a_transport_failure_is_reported_not_raised(self):
        tid = "fix-first-down"
        server.active_chat_tasks[tid] = _task(request_id="abcd1234", background_task=None)
        post = AsyncMock(side_effect=ConnectionError())
        try:
            with patch.object(server, "_get_http_client", return_value=MagicMock(post=post)):
                out = await server.chat_cancel_proxy(tid)
            assert out["agent"]["cancelled"] is False
            assert out["agent"]["error"], "an empty str(e) left the client with no reason"
        finally:
            server.active_chat_tasks.pop(tid, None)

    @pytest.mark.asyncio
    async def test_a_legacy_task_without_an_id_keeps_the_old_shape(self):
        tid = "fix-first-legacy"
        server.active_chat_tasks[tid] = _task(background_task=None)
        post = AsyncMock()
        try:
            with patch.object(server, "_get_http_client", return_value=MagicMock(post=post)):
                out = await server.chat_cancel_proxy(tid)
            assert out == {"status": "cancelled"}
            post.assert_not_awaited()
        finally:
            server.active_chat_tasks.pop(tid, None)

    def test_task_state_exposes_the_id(self):
        from fastapi.testclient import TestClient
        tid = "fix-first-state"
        server.active_chat_tasks[tid] = _task(request_id="abcd1234")
        try:
            r = TestClient(server.app).get(f"/api/chat/task/{tid}/state",
                                           headers={"X-Ghost-Key": server.GHOST_API_KEY})
            assert r.json()["request_id"] == "abcd1234"
        finally:
            server.active_chat_tasks.pop(tid, None)

    def test_client_reads_the_id_from_the_response_header(self, app_js_nc):
        i = app_js_nc.index("response.headers.has('X-Task-ID')")
        window = app_js_nc[i:i + 700]
        assert "response.headers.has('X-Request-ID')" in window
        assert "currentReqId = _rid.replace(/^chatcmpl-/, '')" in window

    def test_client_stop_falls_back_only_when_the_proxy_could_not(self, app_js_nc):
        """One call first; the /api/turns resolution is the FALLBACK, and a
        proxy answer of agent.cancelled === true ends it. The captured ids
        (`_rid`, `_txt`) matter: the live fields are reset before the
        promise settles."""
        i = app_js_nc.index("/api/chat/cancel/${currentTaskId}")
        body = app_js_nc[i - 200:i + 700]
        assert "const _rid = currentReqId, _txt = _lastSentUserText;" in body
        assert "d.agent.cancelled === true" in body
        assert body.count("_cancelAgentTurn(_rid, _txt)") == 3, body.count("_cancelAgentTurn(_rid, _txt)")
        assert "_cancelAgentTurn(currentReqId," not in body, "a reset field is read after the await"


# ═══════════════════════════════════════════════════════════════════════════
# Cache-bust: every touched module moved (the manifest test pins content)
# ═══════════════════════════════════════════════════════════════════════════

def test_touched_modules_bumped(index_html, app_js):
    assert "app.js?v=12.1" in index_html and "style.css?v=6.3" in index_html
    assert "workspace.js?v=8.5" in app_js and "matrix_graph.js?v=12.1" in app_js
    ws = (_STATIC / "workspace.js").read_text(encoding="utf-8")
    assert "notifications.js?v=7.0" in ws
