// Token counter under the composer's buttons (2026-10-27, operator:
// "discreet", "just up and down, for context / generation", "on the right
// side, under the buttons … similar in mobile").
//
// The agent sends one `{"ghost": {"usage": {...}}}` frame just before a
// stream's [DONE] (owner turns only): the turn's prompt / output / cached
// tokens summed over every model call it made. Shown: `↑29.6k ↓840`, the
// last turn's context and generation; the title adds the cached part.
//
// Kept per session in memory, mirrored to localStorage so an iOS tab reload
// or a session load shows that session's last turn (the last SESSIONS_MAX
// sessions). Memory is the truth: a full store (the chat histories fill it)
// must not freeze the counter on old numbers (review r2). A resumed stream
// replays its frame — it just writes the same turn again. Without a session
// id (sessions off, fresh browser) the turns go under NO_SESSION.
// No frame for the latest turn → hidden: an absent number is honest, a
// zero (or the previous turn's numbers) is not.

const STORE_KEY = 'ghost_token_counts';
const SESSIONS_MAX = 30;
const NO_SESSION = '_';

export function fmtTokens(n) {
    n = Math.max(0, Math.floor(Number(n) || 0));
    if (n < 1000) return String(n);
    // 99,950+ would round to "100.0k": switch to whole thousands there
    return n < 99950 ? (n / 1000).toFixed(1) + 'k' : Math.round(n / 1000) + 'k';
}

// {prompt, completion, cached, approx} from a frame's ghost.usage, or null.
export function readUsage(u) {
    if (!u || typeof u !== 'object') return null;
    const num = v => (Number.isFinite(Number(v)) && Number(v) >= 0 ? Math.floor(Number(v)) : 0);
    const prompt = num(u.prompt_tokens);
    return {
        prompt,
        completion: num(u.completion_tokens),
        cached: Math.min(num(u.cached_tokens), prompt),
        approx: num(u.unmetered_calls) > 0,
    };
}

export function createTokenCounter(el, storage, getSessionId) {
    let all;
    try {
        const v = JSON.parse(storage.get(STORE_KEY) || '{}');
        all = v && typeof v === 'object' && !Array.isArray(v) ? v : {};
    } catch (e) { all = {}; }
    const sidNow = () => getSessionId() || NO_SESSION;
    const at = sid => (all[sid] && typeof all[sid] === 'object' && Number(all[sid].at)) || 0;
    const save = () => {
        const sids = Object.keys(all).sort((a, b) => at(b) - at(a));
        for (const sid of sids.slice(SESSIONS_MAX)) delete all[sid];
        try { storage.set(STORE_KEY, JSON.stringify(all)); } catch (e) { /* quota / private mode */ }
    };
    const hide = () => {
        if (!el) return;
        el.textContent = '';
        el.removeAttribute('title');
        el.classList.add('hidden');
    };

    function render() {
        if (!el) return;
        const s = all[sidNow()];
        // re-read through readUsage: the stored copy is untrusted (old/edited)
        const u = s && s.last && typeof s.last === 'object' ? readUsage({
            prompt_tokens: s.last.prompt, completion_tokens: s.last.completion,
            cached_tokens: s.last.cached, unmetered_calls: s.last.approx ? 1 : 0 }) : null;
        if (!u) return hide();
        const ap = u.approx ? '~' : '';
        el.textContent = `↑${ap}${fmtTokens(u.prompt)} ↓${fmtTokens(u.completion)}`;
        el.title = `Last reply — context: ${ap}${u.prompt.toLocaleString()} tokens`
            + (u.cached ? ` (${u.cached.toLocaleString()} cached)` : '')
            + `, generated: ${u.completion.toLocaleString()}`
            + (u.approx ? ' (a call ended without its usage report: context incomplete)' : '');
        el.classList.remove('hidden');
    }

    function note(usage) {
        const u = readUsage(usage);
        if (!u) return;
        all[sidNow()] = { last: u, at: Date.now() };
        save();
        render();
    }

    // Drop this session's last turn: a new turn was sent (until its frame
    // arrives the last turn's numbers are not this one's), or /clear (with
    // sessions off the id does not change, and a reload must not bring the
    // cleared turn back — review r2).
    function forget() {
        const s = all[sidNow()];
        if (s && typeof s === 'object' && s.last) {
            s.last = null;
            save();
        }
        render();
    }

    render();
    return { note, render, hide, startTurn: forget, forget };
}
