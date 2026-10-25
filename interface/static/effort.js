// §4RB: the reasoning-effort selector beside Send.
//
// Shown only while the agent's /api/health says the switch is on
// (GHOST_REASONING_EFFORT=1 — the 27B); hidden, it is reset to the default so a
// stale choice is never sent. status.js announces every health result as a
// `ghost-health` window event (detail = the health object, null when the
// agent did not answer — then the selector is left as it is).
//
// The choice rides the chat body as `reasoning_effort`; the interface turns
// it into the X-Ghost-Effort header. Default ("") sends nothing: the model
// server's own level (its --chat-template-kwargs).

export const EFFORT_LEVELS = ['low', 'medium', 'xhigh'];

export function applyEffortSelector(sel, health) {
    if (!sel || !health) return;
    const on = health.reasoning_effort === true;
    sel.classList.toggle('hidden', !on);
    if (!on) sel.value = '';
}

export function effortForPayload(sel) {
    if (!sel || sel.classList.contains('hidden')) return '';
    const v = String(sel.value || '');
    return EFFORT_LEVELS.includes(v) ? v : '';
}

export function initEffortSelector(doc = document, win = window) {
    const sel = doc.getElementById('effort-select');
    win.addEventListener('ghost-health', ev => applyEffortSelector(sel, ev && ev.detail));
    applyEffortSelector(sel, win.__ghostHealth || null);   // a poll that landed before this ran
    return sel;
}
