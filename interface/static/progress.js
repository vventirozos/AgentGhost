// §4UT — the agent's steps while it works (operator 2026-11-16: "show the progress in a request like claude
// code does … we need to be able to hide the steps in the ui somehow, we need to be smart on how to do it").
//
// The stream carries `ghost.progress` frames: a step `{seq, tools, summary}` right after each tool batch, and
// its sentence `{seq, text}` about a second later (written by the side model). This module keeps one list
// per reply, placed just ABOVE the reply bubble (the bubble's content is replaced on every render; a sibling
// survives that).
//
// Hiding, without asking the user to manage it:
//   * while the agent works the list shows the LAST few steps; older ones sit behind "N earlier steps";
//   * the moment the answer starts it folds to one line, "▸ 7 steps · 42 s" — the reply is what matters now;
//     a click opens it again;
//   * collapsing it while the agent works is remembered: the next turns start folded (one live line);
//     opening it brings the live list back. The mode is per browser (localStorage; it works without it);
//   * "hide" on the list minimizes it to that one line (and later turns start folded);
//   * ⌘K → "Agent steps" can turn them off; the newest reply then keeps one "steps hidden · show" line.

const MODE_KEY = 'ghost.progress.mode';          // 'live' | 'folded' | 'off'
export const LIVE_SHOWN = 4;                     // steps shown while working; the rest behind "N earlier"

function _readMode() {
    try {
        const v = localStorage.getItem(MODE_KEY);
        return v === 'folded' || v === 'off' ? v : 'live';
    } catch (_) { return 'live'; }
}
function _writeMode(v) {
    try { localStorage.setItem(MODE_KEY, v); } catch (_) { /* private window: this page only */ }
}
let _mode = null;
export function progressMode() { if (_mode === null) _mode = _readMode(); return _mode; }
// the lists on this page: a mode change shows at once (r1: the palette's "off" left the open list up)
const _lists = new Set();
let _latest = null;          // the newest reply's list: in "off" mode it alone keeps the way back
export function setProgressMode(v) {
    _mode = v === 'folded' || v === 'off' ? v : 'live';
    _writeMode(_mode);
    for (const l of [..._lists]) {
        if (!l._anchor() || !l._anchor().isConnected) { l.remove(); continue; }
        if (_mode === 'off') l.open = null;     // r2 NIT: a mode change keeps lists the owner opened
        l.render();
    }
}

// ── pure helpers (tested under node) ──────────────────────────────────────
/** What a step shows: its sentence, else the fixed summary, else the tool names. */
export function stepLabel(step) {
    if (!step || typeof step !== 'object') return '';
    const t = typeof step.text === 'string' ? step.text.trim() : '';
    if (t) return t;
    const s = typeof step.summary === 'string' ? step.summary.trim() : '';
    if (s) return s;
    const tools = step.tools && typeof step.tools === 'object' ? Object.keys(step.tools) : [];
    return tools.length ? 'Used ' + tools.map((k) => k.replace(/_/g, ' ')).join(', ') : '';
}

/** The folded line: "7 steps · 42 s" (1 step, minutes past 90 s). */
export function foldLabel(count, elapsedMs) {
    const n = Math.max(0, count | 0);
    let out = n === 1 ? '1 step' : `${n} steps`;
    const s = Math.round((Number(elapsedMs) || 0) / 1000);
    if (s > 0) out += ' · ' + (s < 90 ? `${s} s` : `${Math.round(s / 60)} min`);
    return out;
}

/** Merge a frame into the ordered step list (a sentence attaches to its seq; a replayed step is no
 * duplicate). Returns the list. */
export function mergeStep(steps, frame) {
    if (!frame || typeof frame !== 'object' || !Number.isFinite(Number(frame.seq))) return steps;
    const seq = Number(frame.seq);
    let s = steps.find((x) => x.seq === seq);
    if (!s) {
        if (!frame.summary && !frame.tools) return steps;     // a sentence for a step never seen
        s = { seq, tools: {}, summary: '', text: '' };
        steps.push(s);
        steps.sort((a, b) => a.seq - b.seq);
    }
    if (frame.tools && typeof frame.tools === 'object') s.tools = { ...frame.tools };
    if (typeof frame.summary === 'string' && frame.summary) s.summary = frame.summary;
    if (typeof frame.text === 'string' && frame.text.trim()) s.text = frame.text.trim();
    return steps;
}

// ── the list ──────────────────────────────────────────────────────────────
function _el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
}

/** One reply's step list. `anchor()` returns the reply bubble it sits above. */
export class StepList {
    constructor(anchor, startedAt) {
        this._anchor = anchor;
        this.startedAt = Number(startedAt) || Date.now();
        this.steps = [];
        this.answering = false;      // the answer has started: folded, no new steps
        this.open = null;            // the user's click on THIS list wins over the mode
        this.showEarlier = false;
        this.el = null;
        this.endedAt = 0;
        _lists.add(this);
        _latest = this;
    }

    note(frame) {
        if (this.answering) return;
        const before = this.steps.length;
        mergeStep(this.steps, frame);
        if (!this.steps.length) return;
        if (this.steps.length !== before || frame.text) this.render();
    }

    /** The answer's first visible text: fold. */
    answer() {
        if (this.answering) return;
        this.answering = true;
        this.endedAt = Date.now();
        if (this.open !== true) this.open = false;
        this.render();
    }

    /** The answer was taken back (a tool call followed it): the steps go on. */
    resume() {
        if (!this.answering) return;
        this.answering = false;
        this.open = null;
        this.render();
    }

    _isOpen() {
        if (this.open !== null) return this.open;
        return !this.answering && progressMode() === 'live';
    }

    render() {
        const bubble = this._anchor();
        const off = progressMode() === 'off';
        // "off": only the newest reply keeps one line — the way back (r2: it went with the answer, and a
        // "hide" pressed on a finished reply left nothing to undo it with)
        if (!bubble || !bubble.parentNode || !this.steps.length || (off && this !== _latest)) {
            if (this.el) { this.el.remove(); this.el = null; }
            return;
        }
        // r1: the focused control survives the rebuild (every step re-renders the list)
        const _ae = this.el && typeof document !== 'undefined' && this.el.contains(document.activeElement)
            ? document.activeElement.className.split(' ')[0] : '';
        if (!this.el) {
            this.el = _el('div', 'progress-steps');
            this.el.setAttribute('role', 'group');
            this.el.setAttribute('aria-label', 'What the agent did');
        }
        if (this.el.nextSibling !== bubble) bubble.parentNode.insertBefore(this.el, bubble);
        if (off) {
            // r1 M2: steps are hidden, but the way back is one tap away — the palette (⌘K) is not
            // reachable on a phone
            this.el.className = 'progress-steps is-hidden';
            this.el.replaceChildren();
            const b = _el('button', 'progress-show', 'steps hidden · show');
            b.type = 'button';
            b.addEventListener('click', () => setProgressMode('live'));
            this.el.appendChild(b);
            if (_ae) { const f = this.el.querySelector('.' + _ae); if (f) f.focus(); }
            return;
        }
        this.el.className = 'progress-steps';
        const open = this._isOpen();
        this.el.classList.toggle('is-open', open);
        this.el.classList.toggle('is-working', !this.answering);
        this.el.replaceChildren();

        // header row: [▸/▾ count (latest step)] [hide] — the hide control sits with the list's title,
        // not under it (operator 11-16: "hide steps is at the wrong place")
        const bar = _el('div', 'progress-bar');
        const head = _el('button', 'progress-head');
        head.type = 'button';
        head.setAttribute('aria-expanded', open ? 'true' : 'false');
        head.appendChild(_el('span', 'progress-caret', open ? '▾' : '▸'));
        const elapsed = (this.endedAt || Date.now()) - this.startedAt;
        head.appendChild(_el('span', 'progress-count',
            this.answering ? foldLabel(this.steps.length, elapsed) : foldLabel(this.steps.length, 0)));
        if (!open && !this.answering) {
            head.appendChild(_el('span', 'progress-latest', stepLabel(this.steps[this.steps.length - 1])));
        }
        head.addEventListener('click', () => {
            const next = !open;
            this.open = next;
            // collapsing a live list is a preference: later turns start folded; opening one undoes it
            if (!this.answering) setProgressMode(next ? 'live' : 'folded');
            this.render();
        });
        bar.appendChild(head);
        // operator 11-16: "hide steps should minimize" — fold to the one-line header (a click on it opens the
        // list again); folded is remembered for later turns. Off entirely stays in ⌘K → Agent steps.
        if (open) {
            const hide = _el('button', 'progress-off', 'hide');
            hide.type = 'button';
            hide.title = 'Minimize the steps to one line (click the line to open them again)';
            hide.setAttribute('aria-label', 'Minimize the agent\'s steps');
            hide.addEventListener('click', () => {
                this.open = false;
                setProgressMode('folded');
                this.render();
            });
            bar.appendChild(hide);
        }
        this.el.appendChild(bar);

        if (open) {
            const list = _el('ol', 'progress-list');
            let shown = this.steps;
            const hidden = !this.answering && !this.showEarlier ? Math.max(0, this.steps.length - LIVE_SHOWN) : 0;
            if (hidden) {
                shown = this.steps.slice(hidden);
                const more = _el('li', 'progress-earlier');
                const b = _el('button', 'progress-earlier-btn', hidden === 1 ? '1 earlier step' : `${hidden} earlier steps`);
                b.type = 'button';
                b.addEventListener('click', () => { this.showEarlier = true; this.render(); });
                more.appendChild(b);
                list.appendChild(more);
            }
            for (const s of shown) {
                const li = _el('li', 'progress-step' + (s.text ? '' : ' is-summary'));
                li.appendChild(_el('span', 'progress-dot', '●'));
                const body = _el('span', 'progress-body');
                body.appendChild(_el('span', 'progress-text', stepLabel(s)));
                if (s.text && s.summary) body.appendChild(_el('span', 'progress-sub', '⎿ ' + s.summary));
                li.appendChild(body);
                list.appendChild(li);
            }
            this.el.appendChild(list);
        }
        if (_ae) { const f = this.el.querySelector('.' + _ae); if (f) f.focus(); }
    }

    remove() { _lists.delete(this); if (this.el) { this.el.remove(); this.el = null; } }
}
