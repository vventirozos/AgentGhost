// ═══════════════════════════════════════════════════════════════
//  Workspace entry module (2026-07-28)
//
//  Loaded dynamically by app.js AFTER it publishes window.GhostCore.
//  Everything new lives out here in ES modules; app.js stays the chat
//  core and is only touched at the bridge seam. Sub-modules:
//    sessions.js       — durable server-side conversations (left rail)
//    notifications.js  — notify-severity autonomous activity (bell)
//    status.js         — /api/health poll → DEGRADED tag on the ONLINE chip
//    palette.js        — ⌘K command palette
//  This module owns the shared chrome: rail open/close, density,
//  toasts, the per-message action menu, and the memory-correction
//  modal.
// ═══════════════════════════════════════════════════════════════
import { initSessions } from './sessions.js?v=7.7';
import { initNotifications } from './notifications.js?v=6.9';
import { initStatus } from './status.js?v=7.2';
import { initPalette } from './palette.js?v=7.0';

const Core = window.GhostCore;

// --- tiny DOM helpers ------------------------------------------------
export function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
}

export function relTime(ts) {
    // Accepts epoch seconds, epoch ms, or an ISO string.
    let ms = Number(ts);
    if (!Number.isFinite(ms)) ms = Date.parse(ts);
    if (!Number.isFinite(ms)) return '';
    if (ms < 1e12) ms *= 1000;   // epoch seconds → ms
    const d = Date.now() - ms;
    if (d < 60_000) return 'now';
    if (d < 3_600_000) return `${Math.floor(d / 60_000)}m ago`;
    if (d < 86_400_000) return `${Math.floor(d / 3_600_000)}h ago`;
    return `${Math.floor(d / 86_400_000)}d ago`;
}

// Session-stripe wheel — the same THERMAL family as the face's palette
// (matrix_graph COLORS.palette, blue↔red through violet) so the rail
// reads as strata of the agent's episodic memory rather than a generic
// chat list. Hues are stable per session id. (Name kept for the pinned
// contract; the family changed with the 2026-07-28 thermal redesign.)
const JEWEL_WHEEL = ['#1e2a78', '#0c1c50', '#4a165a', '#701226', '#8f1226'];
export function jewelHue(str) {
    let h = 0;
    for (const ch of String(str || '')) h = ((h * 31) + ch.charCodeAt(0)) >>> 0;
    return JEWEL_WHEEL[h % JEWEL_WHEEL.length];
}

// --- toasts ----------------------------------------------------------
const TOAST_CAP = 4;
export function toast(text, kind = 'info', onClick = null) {
    const stack = document.getElementById('toast-stack');
    if (!stack) return;
    while (stack.childElementCount >= TOAST_CAP) stack.removeChild(stack.firstChild);
    const t = el('div', `toast toast-${kind}`);
    t.appendChild(el('span', 'toast-text', String(text).slice(0, 220)));
    if (onClick) {
        t.classList.add('toast-clickable');
        t.addEventListener('click', () => { try { onClick(); } catch (e) {} dismiss(); });
    }
    let timer = setTimeout(dismiss, 5200);
    function dismiss() {
        clearTimeout(timer);
        t.classList.add('toast-out');
        setTimeout(() => t.remove(), 250);
    }
    if (!onClick) t.addEventListener('click', dismiss);
    stack.appendChild(t);
}

// --- sessions rail open/close ---------------------------------------
// Docked ≥900px wide AND ≥500px tall (landscape phones stay in drawer
// mode — a docked 272px rail would cramp a 430px-tall viewport).
const mqDocked = window.matchMedia('(min-width: 900px) and (min-height: 500px)');
export function isDocked() { return mqDocked.matches; }

function setRailOpen(open, persist = true) {
    document.body.classList.toggle('rail-open', open);
    const scrim = document.getElementById('rail-scrim');
    if (scrim) scrim.classList.toggle('hidden', !(open && !isDocked()));
    if (persist) Core.safeStorage.set('ghost_rail_open', open ? '1' : '0');
}
export function isRailOpen() { return document.body.classList.contains('rail-open'); }
export function toggleRail() { setRailOpen(!isRailOpen()); }

function initRail() {
    const stored = Core.safeStorage.get('ghost_rail_open');
    // Default: open when docked, closed in drawer mode.
    const open = stored === null ? isDocked() : stored === '1';
    setRailOpen(open && (isDocked() || stored === '1') ? open : false, false);
    if (!isDocked()) setRailOpen(false, false);   // drawers never start open

    document.getElementById('rail-toggle')?.addEventListener('click', toggleRail);
    document.getElementById('rail-scrim')?.addEventListener('click', () => setRailOpen(false));
    // Crossing the docked/drawer boundary (device rotation, split view):
    // drawers must not stay open as overlays.
    mqDocked.addEventListener('change', () => {
        if (!isDocked()) setRailOpen(false, false);
        else setRailOpen(Core.safeStorage.get('ghost_rail_open') !== '0', false);
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !isDocked() && isRailOpen()) setRailOpen(false);
    });
}

// --- density ---------------------------------------------------------
export function toggleDensity() {
    const compact = document.body.classList.toggle('compact');
    Core.safeStorage.set('ghost_density', compact ? 'compact' : 'comfortable');
    toast(compact ? 'Compact density' : 'Comfortable density');
}
function initDensity() {
    if (Core.safeStorage.get('ghost_density') === 'compact') {
        document.body.classList.add('compact');
    }
}

// --- per-message action menu ----------------------------------------
const msgMenu = document.getElementById('msg-menu');
let _menuAnchoredTo = null;

function closeMsgMenu() {
    if (msgMenu) msgMenu.classList.add('hidden');
    _menuAnchoredTo = null;
}

function menuItem(label, handler) {
    const b = el('button', 'msg-menu-item', label);
    b.type = 'button';
    b.setAttribute('role', 'menuitem');
    b.addEventListener('click', () => { closeMsgMenu(); handler(); });
    return b;
}

function messageText(msgDiv) {
    // Prefer the user's text selection inside this message (lets the
    // operator target the exact false claim); fall back to full text.
    const sel = window.getSelection();
    if (sel && !sel.isCollapsed && msgDiv.contains(sel.anchorNode)) {
        const s = sel.toString().trim();
        if (s) return s;
    }
    const clone = msgDiv.cloneNode(true);
    clone.querySelectorAll('.msg-actions').forEach(n => n.remove());
    return (clone.textContent || '').trim();
}

function openMessageMenu(msgDiv, anchorBtn) {
    if (!msgMenu) return;
    if (_menuAnchoredTo === anchorBtn) { closeMsgMenu(); return; }
    msgMenu.replaceChildren();
    const isAgent = msgDiv.classList.contains('agent');
    const text = messageText(msgDiv);

    msgMenu.appendChild(menuItem('Copy message', async () => {
        try { await navigator.clipboard.writeText(text); toast('Copied'); }
        catch (e) { toast('Copy failed', 'error'); }
    }));
    msgMenu.appendChild(menuItem('Quote in reply', () => {
        const input = Core.elements.chatInput;
        const quoted = text.split('\n').map(l => `> ${l}`).join('\n');
        input.value = (input.value ? input.value + '\n' : '') + quoted + '\n\n';
        input.focus();
        input.dispatchEvent(new Event('input'));
    }));
    if (isAgent) {
        msgMenu.appendChild(menuItem('Correct a memory…', () =>
            openMemoryModal('correct', text.slice(0, 400))));
        msgMenu.appendChild(menuItem('Forget a memory…', () =>
            openMemoryModal('delete', text.slice(0, 400))));
    }

    msgMenu.classList.remove('hidden');
    // Position near the anchor, clamped to the viewport.
    const r = anchorBtn.getBoundingClientRect();
    const mw = msgMenu.offsetWidth || 220;
    const mh = msgMenu.offsetHeight || 160;
    msgMenu.style.left = `${Math.max(8, Math.min(r.right - mw, window.innerWidth - mw - 8))}px`;
    msgMenu.style.top = `${Math.min(r.bottom + 6, window.innerHeight - mh - 8)}px`;
    _menuAnchoredTo = anchorBtn;
}

document.addEventListener('click', (e) => {
    if (msgMenu && !msgMenu.classList.contains('hidden')
        && !msgMenu.contains(e.target)
        && !(e.target.closest && e.target.closest('.msg-menu-btn'))) {
        closeMsgMenu();
    }
});

// --- memory correction modal ----------------------------------------
// The safe surgical path for a poisoned auto-memory: /api/memory/correct
// rewrites ONE vector fragment matched by exact text / unique substring;
// /api/memory/delete removes one wholly-false fragment.
const memModal = document.getElementById('memory-modal');
let _memMode = 'correct';

export function openMemoryModal(mode, prefill = '') {
    if (!memModal) return;
    _memMode = mode;
    const isCorrect = mode === 'correct';
    document.getElementById('memory-modal-title').textContent =
        isCorrect ? 'CORRECT MEMORY' : 'FORGET MEMORY';
    document.getElementById('memory-modal-hint').textContent = isCorrect
        ? 'Finds one stored memory fragment containing this exact text and rewrites it. Trim the match down to the false part before applying.'
        : 'Finds one stored memory fragment containing this exact text and deletes it. Trim the match down so it is unique.';
    document.getElementById('memory-replacement').classList.toggle('hidden', !isCorrect);
    document.getElementById('memory-replacement-label').classList.toggle('hidden', !isCorrect);
    document.getElementById('memory-modal-apply').textContent =
        isCorrect ? 'Apply correction' : 'Forget it';
    document.getElementById('memory-match').value = prefill;
    document.getElementById('memory-replacement').value = '';
    memModal.classList.remove('hidden');
    document.getElementById('memory-match').focus();
}
function closeMemoryModal() { memModal?.classList.add('hidden'); }

async function applyMemoryEdit() {
    const match = document.getElementById('memory-match').value.trim();
    const replacement = document.getElementById('memory-replacement').value.trim();
    if (!match) { toast('Match text is required', 'error'); return; }
    if (_memMode === 'correct' && !replacement) {
        toast('Corrected text is required', 'error'); return;
    }
    const url = _memMode === 'correct' ? '/api/memory/correct' : '/api/memory/delete';
    const body = _memMode === 'correct' ? { match, replacement } : { match };
    const applyBtn = document.getElementById('memory-modal-apply');
    applyBtn.disabled = true;
    try {
        const res = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const data = await res.json().catch(() => ({}));
        if (res.ok && data.ok !== false) {
            toast(_memMode === 'correct' ? 'Memory corrected' : 'Memory forgotten', 'ok');
            closeMemoryModal();
        } else {
            // 409 = no match / ambiguous match — the detail says which.
            toast(data.error || data.detail || `Failed (HTTP ${res.status})`, 'error');
        }
    } catch (e) {
        toast(`Memory edit failed: ${e.message}`, 'error');
    } finally {
        applyBtn.disabled = false;
    }
}

function initMemoryModal() {
    document.getElementById('memory-modal-close')?.addEventListener('click', closeMemoryModal);
    document.getElementById('memory-modal-cancel')?.addEventListener('click', closeMemoryModal);
    document.getElementById('memory-modal-apply')?.addEventListener('click', applyMemoryEdit);
    memModal?.addEventListener('click', (e) => { if (e.target === memModal) closeMemoryModal(); });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && memModal && !memModal.classList.contains('hidden')) {
            closeMemoryModal();
        }
    });
}

// --- boot ------------------------------------------------------------
Core.openMessageMenu = openMessageMenu;

const ctx = { Core, el, toast, jewelHue, relTime, toggleRail, toggleDensity, openMemoryModal, isDocked, setRailOpen };
initRail();
initDensity();
initMemoryModal();
const sessions = initSessions(ctx);
const notifications = initNotifications(ctx);
const status = initStatus(ctx);
initPalette({ ...ctx, sessions, notifications });

// Decorate whatever history app.js already restored before we loaded.
Core.decorateMessageActions();

// Debug / test hook (page is key-gated; exposes nothing the modules
// don't already drive).
window.__ghostWorkspace = { toast, toggleRail, toggleDensity, sessions, notifications, status, jewelHue };
