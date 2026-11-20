// §4UT — the agent's steps while it works (operator 2026-11-16: "show the progress in a request like claude
// code does … we need to be able to hide the steps in the ui somehow, we need to be smart on how to do it").
//
// The stream carries `ghost.progress` frames for each step (`seq`; core/progress.py has the protocol):
//   {seq, tools, label, running: true}      the batch was dispatched — the step shows at once (§4UY S3)
//   {seq, detail}                           what a task turn inside the running step is doing
//   {seq, tools, summary, failed?, cut?}    the batch is done; `failed` = calls that did not do it (S13)
//   {seq, gone: true}                       nothing in it ran — the step goes
//   {seq, text} | {seq, none: reason}       its sentence (from the side model), or why there is none (S15)
// This module keeps one list per reply, placed just ABOVE the reply bubble (the bubble's content is replaced
// on every render; a sibling survives that). The list outlives its bubble: a Stop or an error before any
// reply text removes the bubble, and the steps are what is left of the turn (§4UY S1).
//
// Hiding, without asking the user to manage it:
//   * while the agent works the list shows the LAST few steps; older ones sit behind "N earlier steps";
//   * the moment the answer starts it folds to one line, "▸ 7 steps · 42 s" — the reply is what matters now;
//     a click opens it again;
//   * collapsing it while the agent works is remembered: the next turns start folded (one live line);
//     opening it brings the live list back. The mode is per browser (localStorage; it works without it);
//   * "hide" on the list minimizes it to that one line (and later turns start folded);
//   * ⌘K → "Agent steps" can turn them off; the newest reply then keeps one "steps hidden · show" line (the
//     agent then sends the steps without asking its side model for sentences — app.js, §4UY S10).

const MODE_KEY = 'ghost.progress.mode';          // 'live' | 'folded' | 'off'
export const LIVE_SHOWN = 4;                     // steps shown while working; the rest behind "N earlier"

/** Why a step has no sentence (core/progress.py NO_SENTENCE), as the step's tooltip. */
export const NO_SENTENCE = {
    off: 'No sentence: sentences are off, or there is no side model',
    busy: 'No sentence: a newer step finished while the side model was writing',
    slow: 'No sentence: the side model did not answer in time',
    unavailable: 'No sentence: the side model could not be reached',
    rejected: 'No sentence: the side model\'s line was not one clean sentence about this step',
    cut: 'No sentence: the step was stopped before it finished',
};

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
// the lists on this page: a mode change shows at once (r1: the palette's "off" left the open list up). A list
// whose element left the page (a cleared chat, a session switch) leaves this set (§4UY S17).
const _lists = new Set();
let _latest = null;          // the newest reply's list: in "off" mode it alone keeps the way back
function _prune() {
    for (const l of [..._lists]) if (l.el && !l.el.isConnected) _lists.delete(l);
}
export function liveListCount() { _prune(); return _lists.size; }
export function setProgressMode(v) {
    _mode = v === 'folded' || v === 'off' ? v : 'live';
    _writeMode(_mode);
    _prune();
    for (const l of [..._lists]) {
        if (_mode === 'off') l.open = null;     // r2 NIT: a mode change keeps lists the owner opened
        l.render();
    }
}

// ── pure helpers (tested under node) ──────────────────────────────────────
/** What a step shows: its sentence, else the fixed summary, else its running label, else the tool names. */
export function stepLabel(step) {
    if (!step || typeof step !== 'object') return '';
    for (const k of ['text', 'summary', 'label']) {
        const v = typeof step[k] === 'string' ? step[k].trim() : '';
        if (v) return v;
    }
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
 * duplicate; a done frame ends a running step; `gone` removes it). Returns the list. */
export function mergeStep(steps, frame) {
    if (!frame || typeof frame !== 'object' || !Number.isFinite(Number(frame.seq))) return steps;
    const seq = Number(frame.seq);
    const at = steps.findIndex((x) => x.seq === seq);
    if (frame.gone === true) {
        if (at >= 0) steps.splice(at, 1);
        return steps;
    }
    let s = at >= 0 ? steps[at] : null;
    if (!s) {
        if (!frame.summary && !frame.tools && !frame.label) return steps;   // a sentence for a step never seen
        s = { seq, tools: {}, summary: '', label: '', text: '', detail: '', why: '', failed: 0, cut: false,
              running: false };
        steps.push(s);
        steps.sort((a, b) => a.seq - b.seq);
    }
    if (frame.tools && typeof frame.tools === 'object') s.tools = { ...frame.tools };
    if (typeof frame.label === 'string' && frame.label) s.label = frame.label;
    if (frame.running === true && !s.summary) s.running = true;
    if (typeof frame.summary === 'string' && frame.summary) {
        s.summary = frame.summary;
        s.running = false;                        // done: a replayed "started" frame never revives it
        s.detail = '';
        s.failed = Math.max(0, Number(frame.failed) || 0);
        s.cut = frame.cut === true;
    }
    if (typeof frame.detail === 'string' && frame.detail.trim() && s.running) s.detail = frame.detail.trim();
    if (typeof frame.text === 'string' && frame.text.trim()) { s.text = frame.text.trim(); s.why = ''; }
    if (typeof frame.none === 'string' && frame.none && !s.text) s.why = frame.none;
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
        this.answering = false;      // the answer has started (or the turn ended): folded, no new steps
        this.open = null;            // the user's click on THIS list wins over the mode
        this.showEarlier = false;
        this.el = null;
        this.endedAt = 0;
        this._queued = false;
        this._said = '';
        // §4UY S4: one list works at a time — a list a lost turn left working stops when the next one starts
        _prune();
        for (const l of _lists) if (!l.answering) l.answer();
        _lists.add(this);
        _latest = this;
    }

    note(frame) {
        if (!frame || typeof frame !== 'object') return;
        const before = JSON.stringify(this.steps);
        if (this.answering) {
            // the answer has started: no new step — but a sentence (or its reason) for a step already shown
            // still lands on the folded list (§4UY S16: a turn that ended within a second lost its last line)
            if (!this.steps.some((x) => x.seq === Number(frame.seq))) return;
            if (!('text' in frame) && !('none' in frame) && !('summary' in frame)) return;
        }
        mergeStep(this.steps, frame);
        if (JSON.stringify(this.steps) !== before) this._schedule();
    }

    // §4UY S17: one render per burst of frames (a resume replays the whole stream in one read)
    _schedule() {
        if (this._queued) return;
        this._queued = true;
        queueMicrotask(() => { this._queued = false; this.render(); });
    }

    /** The answer's first visible text, or the end of the turn: fold, nothing works any more. */
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
        // §4UY S4: only the turn's own list resumes — an older list stays ended
        if (_latest !== this) return;
        this.answering = false;
        this.open = null;
        this.render();
    }

    _isOpen() {
        if (this.open !== null) return this.open;
        return !this.answering && progressMode() === 'live';
    }

    /** Where the list goes: above its bubble while there is one; where it stands once the bubble is gone. */
    _place() {
        const bubble = this._anchor();
        if (bubble && bubble.parentNode) {
            if (this.el.nextSibling !== bubble) bubble.parentNode.insertBefore(this.el, bubble);
            return true;
        }
        return this.el.isConnected;
    }

    _current() {
        // the step that is running now — never a finished one (§4UY S3: the last done step pulsed as current)
        if (this.answering) return null;
        const last = this.steps[this.steps.length - 1];
        return last && last.running ? last : null;
    }

    _announce() {
        // §4UY S17: a screen reader hears each new step once (the list itself is rebuilt, a live region is not)
        if (!this._live) {
            this._live = _el('span', 'progress-live');
            this._live.setAttribute('aria-live', 'polite');
            this._live.setAttribute('role', 'status');
        }
        const last = this.steps[this.steps.length - 1];
        const say = this.answering ? '' : (last ? stepLabel(last) : '');
        if (say !== this._said) { this._said = say; this._live.textContent = say; }
        return this._live;
    }

    render() {
        const off = progressMode() === 'off';
        if (!this.steps.length || (off && this !== _latest)) {
            if (this.el) this.el.hidden = true;        // hidden in place: it can come back where it was
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
        if (!this._place()) return;
        this.el.hidden = false;
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
        const current = this._current();
        this.el.classList.toggle('is-open', open);
        this.el.classList.toggle('is-working', !this.answering);
        this.el.replaceChildren(this._announce());

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
            head.appendChild(_el('span', 'progress-latest', stepLabel(current || this.steps[this.steps.length - 1])));
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
                const li = _el('li', 'progress-step' + (s.text ? '' : ' is-summary')
                    + (s === current ? ' is-running' : '') + (s.failed ? ' is-failed' : '')
                    + (s.cut ? ' is-cut' : ''));
                li.appendChild(_el('span', 'progress-dot', s.failed ? '✕' : '●'));
                const body = _el('span', 'progress-body');
                body.appendChild(_el('span', 'progress-text', stepLabel(s)));
                if (s.text && s.summary) body.appendChild(_el('span', 'progress-sub', '⎿ ' + s.summary));
                if (s === current && s.detail) body.appendChild(_el('span', 'progress-sub progress-detail', '⎿ ' + s.detail));
                if (s.cut) body.appendChild(_el('span', 'progress-sub', '⎿ stopped before it finished'));
                if (!s.text && s.why && NO_SENTENCE[s.why]) li.title = NO_SENTENCE[s.why];
                li.appendChild(body);
                list.appendChild(li);
            }
            if (!this.answering && !current) {
                // between steps the model is thinking: say so, rather than make the last done step look current
                const li = _el('li', 'progress-step is-summary is-running is-thinking');
                li.appendChild(_el('span', 'progress-dot', '●'));
                li.appendChild(_el('span', 'progress-body', 'Thinking…'));
                list.appendChild(li);
            }
            this.el.appendChild(list);
        }
        if (_ae) { const f = this.el.querySelector('.' + _ae); if (f) f.focus(); }
    }

    remove() { _lists.delete(this); if (this.el) { this.el.remove(); this.el = null; } }
}
