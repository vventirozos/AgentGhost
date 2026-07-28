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
    return (_STATIC / name).read_text(encoding="utf-8")


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
    html = _read("index.html")
    for el_id in ("chat-container", "chat-log", "log-console", "render-window",
                  "mic-btn", "send-btn", "chat-input"):
        assert f'id="{el_id}"' in html, f"missing legacy #{el_id}"


# ---------------------------------------------------------------------------
# app.js — bridge seam + session binding
# ---------------------------------------------------------------------------

def test_chat_payload_binds_session_id():
    js = _read("app.js")
    # The payload gains session_id only when the workspace holds one —
    # sessions-disabled agents keep the legacy fully-client-carried mode.
    assert "window.__ghostSessionId" in js
    assert re.search(r"payload\.session_id\s*=\s*window\.__ghostSessionId", js)


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
