// What the web turn status line under the reply SHOWS (2026-10-27,
// operator: show the user what the agent is doing, hide what they don't
// need to know — tool calls, image generation, web / dark-web searches;
// never errors, warnings, head-disk or other housekeeping).
//
// An allowlist, read by app.js's noteTickerLine: a line whose title is not
// here leaves the caption as it is (reasoning lines read "💭 thinking…").
// Each caption wears its own icon `i` (a 🔶-flavoured line of a shown step
// must not put a warning glyph under the reply), words `t`, and `d` picks
// the detail worth showing from the line's content, or none. Keys are
// titles the agent prints (pinned in tests/test_web_turn_caption.py). Web
// only: the live log, Slack and the device client (which mirrors app.js's
// TICKER_VERBS) are unchanged.
const _capFile = c => {                      // "/webos/index.html [190:210]" → "index.html"
    const m = String(c).match(/[^\s/]+\.[A-Za-z0-9]{1,8}(?=\s|$|\s*[\[(])/);
    return m && !m[0].startsWith('.ephemeral') ? m[0] : '';   // the agent's own temp scripts
};
const _capHost = c => {
    const m = String(c).match(/https?:\/\/([^\/\s?#]+)/);
    return m ? m[1].replace(/^www\./, '') : '';
};
export const TURN_CAPTIONS = {
    'tool call':        { i: '🧰', t: 'using a tool', d: c => String(c).split(' · ')[0].replace(/_/g, ' ').trim() },
    // the query line only, never the engine race ("yandex won wave 0 in
    // 1.4s", "no winner after 6s — hedging …")
    'web search':       { i: '🌐', t: 'searching the web', d: c => (/\b(won|wave|winner|hedg\w*|results?|engines?)\b|\d\s?m?s\b/.test(c) ? '' : c) },
    'darkweb search':   { i: '🧅', t: 'searching the dark web', d: c => String(c).split(' | ')[0] },
    'darkweb engine':   { i: '🧅', t: 'searching the dark web' },
    'browser':          { i: '🌎', t: 'browsing', d: _capHost },
    'image generation': { i: '🎨', t: 'generating an image' },
    'image compute':    { i: '🎨', t: 'generating an image' },
    'subject photo':    { i: '📥', t: 'fetching a reference photo', d: c => String(c).split(' → ')[0] },
    'youtube':          { i: '📚', t: 'reading a YouTube video' },
    'system weather':   { i: '⛅', t: 'checking the weather', d: c => String(c).replace(/^Location:\s*/, '') },
    'vision':           { i: '🔬', t: 'looking at an image' },
    'vision compute':   { i: '🔬', t: 'looking at an image' },
    'file read':        { i: '📖', t: 'reading', d: _capFile },
    'file write':       { i: '💾', t: 'writing', d: _capFile },
    'file edit':        { i: '💾', t: 'editing', d: _capFile },
    'file append':      { i: '💾', t: 'writing', d: _capFile },
    'sandbox tree':     { i: '👀', t: 'looking through the workspace' },
    'sandbox exec':     { i: '🐚', t: 'running a command' },
    'shell command':    { i: '🐚', t: 'running a command' },
    'execution task':   { i: '🐍', t: 'running code', d: _capFile },
    'executing skill':  { i: '🐍', t: 'running a skill', d: c => String(c).replace(/^Running custom tool:\s*/, '') },
    'memory recall':    { i: '🔎', t: 'recalling memories' },
    'memory save':      { i: '📝', t: 'saving to memory' },
    'project tool':     { i: '📋', t: 'working on a project' },
    'verifier':         { i: '🧪', t: 'checking the answer' },
    'worker compute':   { i: '🔧', t: 'delegating to the worker node' },
};
// {text, icon} for the status line, or null (keep the caption). Pure.
// `Download [<mode>]` (file_system) is the one title built at runtime.
const _CAP_DOWNLOAD = { i: '📥', t: 'downloading', d: _capHost };
export function turnCaption(title, icon, detail) {
    const key = String(title || '').toLowerCase();
    const c = TURN_CAPTIONS[key] || (/^download \[/.test(key) ? _CAP_DOWNLOAD : null);
    if (!c) return null;
    let d = '';
    try { d = c.d ? String(c.d(String(detail || '')) || '').trim() : ''; } catch (e) { d = ''; }
    if (d.length > 30) d = d.slice(0, 30).trimEnd() + '…';
    // the caption's own icon: the line's may be a 🔶 / ✅ / 📌 flavour
    return { text: d ? `${c.t} · ${d}` : c.t, icon: c.i || icon };
}
