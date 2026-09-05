// ═══════════════════════════════════════════════════════════════
//  Command palette (2026-07-28)
//
//  ⌘K / Ctrl+K. Static commands + a dynamic entry per session, filtered
//  by substring tokens, keyboard-first (arrows + Enter, Escape closes).
// ═══════════════════════════════════════════════════════════════

export function initPalette(ctx) {
    const { Core, el, toast, toggleRail, toggleDensity, sessions, notifications } = ctx;

    const overlay = document.getElementById('cmd-palette');
    const input = document.getElementById('palette-input');
    const results = document.getElementById('palette-results');
    if (!overlay || !input || !results) return { open: () => {}, close: () => {} };

    let items = [];
    let filtered = [];
    let selected = 0;

    function commandList() {
        const cmds = [
            { label: 'New chat', hint: 'start a fresh session', run: () => sessions.newChat() },
            { label: 'Toggle sessions rail', hint: '', run: toggleRail },
            { label: 'Toggle live log', hint: 'pretty-log stream drawer', run: () => Core.toggleLogConsole() },
            { label: 'Toggle zen mode', hint: 'hide the chrome', run: () => document.body.classList.toggle('zen-mode') },
            { label: 'Notifications', hint: 'agent activity', run: () => notifications.togglePanel() },
            // ⚠ Stop is scoped to THIS tab's turn on purpose: a null
            // request id means "cancel whatever holds the semaphore"
            // (core/turns.py::current), which may be a background dream or
            // self-play turn — the composer's Stop had that defect and was
            // fixed (R3 lens B). 'Agent status' and 'Turn queue…' (the
            // status panel with per-turn Stop/Force) were removed
            // 2026-09-05 at the operator's request, along with the panel.
            { label: 'Stop MY turn', hint: 'cooperative cancel of this tab\'s turn', run: () => Core.cancelOwnTurn(false) },
            { label: 'Toggle density', hint: 'compact / comfortable', run: toggleDensity },
            {
                label: 'Copy last reply', hint: '', run: async () => {
                    const history = Core.getChatHistory();
                    const last = [...history].reverse().find(m => m.role === 'assistant');
                    if (!last) { toast('No reply to copy', 'error'); return; }
                    const text = Core.stripInternalTags(
                        typeof last.content === 'string' ? last.content : '');
                    try { await navigator.clipboard.writeText(text); toast('Copied last reply'); }
                    catch (e) { toast('Copy failed', 'error'); }
                },
            },
            { label: 'Save workspace', hint: 'download sandbox + chat as zip', run: () => document.getElementById('workspace-save-btn')?.click() },
            { label: 'Load workspace', hint: 'restore sandbox + chat from a zip', run: () => document.getElementById('workspace-load-btn')?.click() },
            { label: 'Download file from sandbox', hint: '', run: () => document.getElementById('download-btn')?.click() },
        ];
        for (const s of sessions.list()) {
            cmds.push({
                label: `Open: ${s.title || 'untitled'}`,
                hint: s.message_count != null ? `${s.message_count} msgs` : 'session',
                run: () => sessions.switchTo(s.id),
            });
        }
        return cmds;
    }

    function open() {
        items = commandList();
        input.value = '';
        overlay.classList.remove('hidden');
        applyFilter('');
        input.focus();
    }
    function close() {
        overlay.classList.add('hidden');
        input.blur();
    }
    function isOpen() { return !overlay.classList.contains('hidden'); }

    function applyFilter(q) {
        const tokens = q.toLowerCase().split(/\s+/).filter(Boolean);
        filtered = items.filter(it =>
            tokens.every(t => (it.label + ' ' + it.hint).toLowerCase().includes(t)));
        selected = 0;
        renderResults();
    }

    function renderResults() {
        results.replaceChildren();
        if (!filtered.length) {
            results.appendChild(el('div', 'palette-empty', 'No matching command.'));
            return;
        }
        filtered.slice(0, 12).forEach((it, i) => {
            const row = el('div', 'palette-item' + (i === selected ? ' selected' : ''));
            row.setAttribute('role', 'option');
            row.appendChild(el('span', 'palette-label', it.label));
            if (it.hint) row.appendChild(el('span', 'palette-hint', it.hint));
            row.addEventListener('click', () => { close(); it.run(); });
            row.addEventListener('pointermove', () => {
                if (selected !== i) { selected = i; renderResults(); }
            });
            results.appendChild(row);
        });
    }

    function runSelected() {
        const it = filtered[selected];
        if (!it) return;
        close();
        it.run();
    }

    document.addEventListener('keydown', (e) => {
        if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
            e.preventDefault();
            if (isOpen()) close(); else open();
            return;
        }
        if (!isOpen()) return;
        if (e.key === 'Escape') { e.preventDefault(); close(); }
        else if (e.key === 'ArrowDown') {
            e.preventDefault();
            selected = Math.min(selected + 1, Math.min(filtered.length, 12) - 1);
            renderResults();
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            selected = Math.max(selected - 1, 0);
            renderResults();
        } else if (e.key === 'Enter') {
            e.preventDefault();
            runSelected();
        }
    });
    input.addEventListener('input', () => applyFilter(input.value));
    overlay.addEventListener('click', (e) => { if (e.target === overlay) close(); });

    return { open, close };
}
