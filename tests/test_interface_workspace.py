"""Workspace client features (2026-07-28): sessions rail, notifications,
status strip, command palette, message actions, design-system additions.

Static contract tests in the style of test_interface_chat_layout /
test_interface_log_console: pin the load-bearing wiring so a refactor
can't silently disconnect a feature while the page still renders.
Server-side proxy behavior is covered by test_interface_agent_proxies.
"""
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_STATIC = _ROOT / "interface" / "static"


def _read(name):
    """CODE ONLY.

    ⚠ This suite asserted over RAW text, so all 85 of its assertions were
    satisfiable by the COMMENTS beside the code. Demonstrated (R2 lens B):
    repointing `status.js`'s `fetch('/api/turn/cancel')` at `/api/DEAD` —
    every per-turn cancel button dead — still passed, because the string
    also appears in that file's header comment. Same for `/api/health`,
    `/api/turns` and `/api/notifications/ack`. Worse, one test sliced from
    the first `pointerleave` occurrence, which is INSIDE a comment, so
    `ev.pointerType === 'mouse' || true` restored the 2026-08-01
    "mobile delete-all impossible" bug and stayed green.
    """
    from tests.helpers import strip_js_comments
    text = (_STATIC / name).read_text(encoding="utf-8")
    return strip_js_comments(text) if name.endswith(".js") else text


# ---------------------------------------------------------------------------
# index.html — workspace chrome exists and cache busters moved
# ---------------------------------------------------------------------------

def test_workspace_markup_present():
    html = _read("index.html")
    for el_id in (
        "session-rail", "session-list", "session-search", "new-chat-btn",
        "rail-toggle", "rail-scrim", "notif-btn", "notif-badge",
        "notif-panel", "notif-list", "status-panel", "status-body",
        "model-pill", "cmd-palette", "palette-input", "palette-results",
        "msg-menu", "memory-modal", "memory-match", "memory-replacement",
        "toast-stack",
    ):
        assert f'id="{el_id}"' in html, f"missing #{el_id}"


def test_cache_busters_bumped_for_workspace():
    html = _read("index.html")
    assert "style.css?v=3." not in html
    assert "app.js?v=3." not in html


def test_existing_chrome_untouched():
    # The workspace must not have displaced the original surfaces.
    # mic-btn removed from the list 2026-08-01: DELIBERATE operator removal
    # (hold-to-talk unused); the recording engine stays `if (micBtn)`-guarded
    # in app.js, same dormancy pattern as the removed TTS toggle.
    html = _read("index.html")
    for el_id in ("chat-container", "chat-log", "log-console", "render-window",
                  "send-btn", "chat-input"):
        assert f'id="{el_id}"' in html, f"missing legacy #{el_id}"


# ---------------------------------------------------------------------------
# app.js — bridge seam + session binding
# ---------------------------------------------------------------------------

def test_chat_payload_binds_session_id():
    js = _read("app.js")
    # The payload gains session_id only when the workspace holds one —
    # sessions-disabled agents keep the legacy fully-client-carried mode.
    assert "window.__ghostSessionId" in js
    # The binding falls back to the STORED id: sessions.js publishes the
    # global only after the workspace import chain plus three agent round
    # trips, and the composer is live throughout — a turn sent in that
    # window carried no session_id and was never persisted (R3 lens B).
    assert re.search(r"payload\.session_id\s*=\s*_sid", js)
    i = js.index("const _sid = window.__ghostSessionId")
    assert "safeStorage.get('ghost_session_id')" in js[i:i + 200], (
        "an unbound early turn is still sent with no session id")


def test_ghostcore_bridge_published_before_workspace_import():
    js = _read("app.js")
    bridge_at = js.find("window.GhostCore = {")
    import_at = js.find("import('./workspace.js")
    assert bridge_at != -1, "GhostCore bridge missing"
    assert import_at != -1, "workspace dynamic import missing"
    assert bridge_at < import_at, "bridge must be published before the import"
    for member in ("renderHistoryToLog", "clearConversation", "setChatHistory",
                   "isProcessing", "events: new EventTarget()"):
        assert member in js, f"bridge member missing: {member}"


def test_turn_lifecycle_events_dispatched():
    js = _read("app.js")
    assert "new CustomEvent('turn-complete')" in js
    assert "new CustomEvent('conversation-cleared')" in js


def test_message_actions_guard_survives_streaming_rebuilds():
    js = _read("app.js")
    # innerHTML rebuilds wipe children but keep element attributes, so the
    # guard must check for the ROW, not a dataset flag.
    assert ":scope > .msg-actions" in js
    assert "decorateMessageActions" in js


def test_day_separator_is_live_only():
    js = _read("app.js")
    sep = js.split("function _maybeInsertDaySeparator", 1)[1].split("function ", 1)[0]
    assert "_restoringHistory" in sep, "restored history must not get separators"


def test_long_code_collapses_with_expander():
    js = _read("app.js")
    assert "code-collapsed" in js
    assert "code-expand" in js
    # 2026-07-28: actions live in a BOTTOM bar and the collapsed window
    # scrolls internally instead of clipping.
    assert "code-footer" in js
    css = _read("style.css")
    block = css.split(".message pre.code-collapsed code {", 1)[1].split("}", 1)[0]
    assert "overflow-y: auto" in block, "long code must scroll, not clip"
    assert "code-footer" in css


# ---------------------------------------------------------------------------
# sessions.js
# ---------------------------------------------------------------------------

def test_sessions_module_contract():
    js = _read("sessions.js")
    assert "'/api/sessions'" in js or '"/api/sessions"' in js
    assert "/api/sessions/${encodeURIComponent(" in js, "session id must be URL-encoded"
    assert "ghost_session_id" in js, "current session must persist across reloads"
    assert "crypto.randomUUID" in js
    assert "turn-complete" in js, "rail must refresh after turns (titles/counts)"
    assert "conversation-cleared" in js, "/clear must mint a fresh session"
    assert "data.enabled" in js, "sessions-disabled agents must degrade gracefully"


def test_sessions_delete_confirms_first():
    js = _read("sessions.js")
    fn = js.split("async function deleteSession", 1)[1].split("async function", 1)[0]
    assert "confirm(" in fn, "destructive delete must confirm"
    assert "method: 'DELETE'" in fn


def test_turn_status_line():
    """2026-07-29 (v3, final after two placement iterations): while a
    user turn runs, a caption line sits directly UNDER the waiting reply
    bubble — `timer : description - icon`, with a TEXT-sized icon of its
    own (v2's 2.2rem stage icon dwarfed the caption). Parsed live from
    the WS log frames."""
    html = _read("index.html")
    assert 'id="turn-status"' in html
    for el_id in ("turn-status-clock", "turn-status-desc", "turn-status-icon"):
        assert f'id="{el_id}"' in html, f"missing #{el_id}"
    # Format order: clock, then desc, then icon.
    ts = html.split('id="turn-status"', 1)[1].split("</div>", 1)[0]
    assert ts.index("turn-status-clock") < ts.index("turn-status-desc") \
        < ts.index("turn-status-icon")
    js = _read("app.js")
    # Re-parented under the current waiting bubble each turn.
    assert "startTurnTicker(currentAgentMessageDiv)" in js
    assert "insertAdjacentElement('afterend', turnStatusEl)" in js
    assert "noteTickerLine(data.content)" in js, "WS log frames must feed it"
    # Corridor filtering: only OUR request's lines update the caption —
    # background self-play/dream corridors stream on the same socket.
    assert "request started" in js and "tickerReqId" in js
    assert "parts[1] !== tickerReqId" in js
    # Plumbing lines never narrate progress; common steps read human.
    assert "TICKER_NOISE" in js and "'prefill cache'" in js
    assert "TICKER_VERBS" in js
    # Reasoning stretches + streaming get honest captions + icons too.
    assert "setTurnStatusDesc('thinking…', '💭')" in js
    assert "setTurnStatusDesc('writing the reply…', '💬')" in js
    # One idempotent teardown covers every request path's end.
    assert "if (!isProcessing) stopTurnTicker();" in js
    css = _read("style.css")
    assert "#turn-status" in css and "#turn-status-icon" in css
    # The caption's icon is TEXT-sized — the whole point of v3 (the old
    # stage icon was 2.2rem). Pin the invariant, not the exact value.
    icon_block = css.split("#turn-status-icon {", 1)[1].split("}", 1)[0]
    m = re.search(r"font-size:\s*([\d.]+)rem", icon_block)
    assert m and float(m.group(1)) < 1.0, "status icon must stay text-sized"
    # Integration pass (operator: "the timer doesn't belong there"): the
    # clock is a quiet tabular-nums chip so its width never jitters.
    clock_block = css.split("#turn-status-clock {", 1)[1].split("}", 1)[0]
    assert "tabular-nums" in clock_block


def test_delete_all_sessions_button():
    # 2026-07-29 operator request: bulk destructive control at the
    # bottom of the rail.
    html = _read("index.html")
    assert 'id="rail-footer"' in html
    assert 'id="delete-all-sessions-btn"' in html
    js = _read("sessions.js")
    # Two-step armed confirm — no browser dialog, and a stray click can
    # never wipe the strata: first click arms (danger state + explicit
    # count), second click within the window executes, timeout/leaving
    # the rail disarms.
    assert "deleteAllBtn.classList.contains('armed')" in js
    assert "disarmDeleteAll" in js
    assert "pointerleave" in js, "leaving the rail must disarm"
    fn = js.split("async function deleteAllSessions", 1)[1].split(
        "deleteAllBtn?.addEventListener", 1)[0]
    # Deletion iterates the ENUMERATED per-id proxy — deliberately no
    # bulk endpoint (the LAN-reachable surface stays enumerated).
    assert "method: 'DELETE'" in fn
    assert "encodeURIComponent(s.id)" in fn
    assert "Core.isProcessing()" in fn, "mid-turn delete-all must be refused"
    assert "mintId()" in fn, "must land on a fresh session afterwards"
    # Footer only exists when there is something to destroy.
    assert "sessions.length > 0" in js
    css = _read("style.css")
    assert "#delete-all-sessions-btn.armed" in css


# ---------------------------------------------------------------------------
# notifications.js
# ---------------------------------------------------------------------------

def test_notifications_watermark_contract():
    js = _read("notifications.js")
    assert "CONSUMER = 'web-ui'" in js, \
        "consumer name must be ours — watermarks are per-consumer (Slack keeps its own)"
    assert "/api/notifications/pending" in js
    assert "/api/notifications/ack" in js
    assert "data.baseline" in js, "first contact must ack the baseline silently"
    # Ack must come AFTER the fresh-records handling so a crash re-serves
    # instead of drops. Anchor on the fresh-path marker (the disabled-path
    # renderList() earlier in poll() would make a bare renderList() index
    # vacuous).
    poll = js.split("async function poll", 1)[1].split("function openPanel", 1)[0]
    assert poll.index("fresh.length") < poll.rindex("ack(data.watermark)")
    assert poll.index("records = [") < poll.rindex("ack(data.watermark)")


# ---------------------------------------------------------------------------
# status.js
# ---------------------------------------------------------------------------

def test_status_module_contract():
    js = _read("status.js")
    assert "/api/health" in js
    assert "/api/turns" in js
    assert "/api/turn/cancel" in js
    assert "hard" in js, "hard cancel (guaranteed lock release) must be reachable"
    # The two silent-failure detectors must be surfaced, not dropped.
    assert "memory_system_loaded" in js
    assert "biological_watchdog_alive" in js
    # ActiveTurn.to_dict() emits request_id (NOT req_id/id) — a review
    # found the per-turn cancel buttons dead on this exact mismatch.
    assert "t.request_id" in js
    assert "arg.model" in js, "live health config flattens args as arg.model"


# ---------------------------------------------------------------------------
# palette.js / workspace.js
# ---------------------------------------------------------------------------

def test_palette_keybinding():
    js = _read("palette.js")
    assert "e.metaKey || e.ctrlKey" in js
    assert "'k'" in js


def test_workspace_rail_and_hue_contract():
    js = _read("workspace.js")
    assert "min-width: 900px" in js, "docked/drawer boundary must be width-gated"
    assert "min-height: 500px" in js, "landscape phones must stay in drawer mode"
    assert "rail-open" in js
    assert "JEWEL_WHEEL" in js, "session hues ride the face's jewel wheel"


# ---------------------------------------------------------------------------
# style.css — mobile-safe workspace chrome
# ---------------------------------------------------------------------------

def test_css_rail_has_drawer_mode():
    css = _read("style.css")
    m = re.search(r"@media \(max-width: 899\.98px\), \(max-height: 499\.98px\)"
                  r"\s*\{(.*?)\n\}", css, re.DOTALL)
    assert m, "drawer-mode media query missing"
    body = m.group(1)
    assert "position: fixed" in body
    assert "translateX(-105%)" in body


def test_css_panels_become_bottom_sheets_on_phones():
    css = _read("style.css")
    m = re.search(r"@media \(max-width: 640px\)\s*\{(.*?)\n\}", css, re.DOTALL)
    assert m, "phone bottom-sheet media query missing"
    assert "bottom: 0" in m.group(1)


def test_css_touch_and_focus_affordances():
    css = _read("style.css")
    # Hover-revealed actions must stay reachable on touch devices.
    coarse = re.findall(r"@media \(pointer: coarse\)\s*\{(.*?)\n\}", css, re.DOTALL)
    assert any(".msg-actions" in b for b in coarse), \
        "msg actions must be visible without hover on touch"
    assert ":focus-visible" in css, "keyboard focus must be visible"


def test_css_palette_input_never_triggers_ios_zoom():
    css = _read("style.css")
    box = css.split("#palette-input {", 1)[1].split("}", 1)[0]
    assert "max(1rem, 16px)" in box, "sub-16px inputs make iOS Safari auto-zoom"


def test_delete_all_disarm_is_mouse_only():
    """Touch pointers are transient: EVERY tap ends in a pointerleave, so an
    unconditional rail pointerleave-disarm made the second tap of the
    delete-all confirm re-arm instead of execute — mobile delete-all was
    impossible (2026-08-01). The disarm must check pointerType === 'mouse';
    touch relies on the 4s timeout."""
    js = _read("sessions.js")
    handler = js.split("pointerleave", 1)[1].split(");", 1)[0]
    assert "pointerType === 'mouse'" in handler
