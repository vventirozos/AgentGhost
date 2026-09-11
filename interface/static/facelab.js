// ═══════════════════════════════════════════════════════════════
//  Face lab (2026-09-11)
//
//  A tuning aid for the face's signal layer. Everything the face can be
//  told, fired by hand: gaits, tool kick, recall comet, the three
//  verdicts, error kinds, background busy, mood, gaze, an idle twitch,
//  the audio level — plus a slider per TUNE amplitude, a live readout
//  of getDebugState(), and "Copy JSON" so a tuned table can be pasted
//  back into matrix_graph.js. Opens from the command palette ("Face
//  lab") or Alt+Shift+F. Lazy-loaded: costs nothing until asked for.
// ═══════════════════════════════════════════════════════════════

// What each button does — a table so the test can prove every action
// names an export the face module actually has.
export function labActions() {
    return [
        { group: 'events', label: 'tool kick', call: 'noteToolCall', args: ['lab_tool'] },
        { group: 'events', label: 'recall comet', call: 'noteRecall', args: [] },
        { group: 'events', label: 'verdict: pass', call: 'noteVerdict', args: ['pass'] },
        { group: 'events', label: 'verdict: refute', call: 'noteVerdict', args: ['refute'] },
        { group: 'events', label: 'verdict: stop', call: 'noteVerdict', args: ['stop'] },
        { group: 'events', label: 'idle twitch', call: 'fireIdleTwitch', args: [] },
        { group: 'errors', label: 'error: network', call: 'noteError', args: ['network'] },
        { group: 'errors', label: 'error: refusal', call: 'noteError', args: ['refusal'] },
        { group: 'errors', label: 'error: timeout', call: 'noteError', args: ['timeout'] },
        { group: 'errors', label: 'error: generic', call: 'noteError', args: ['generic'] },
        { group: 'gait', label: 'none', call: 'setPhase', args: [null] },
        { group: 'gait', label: 'search', call: 'setPhase', args: ['search'] },
        { group: 'gait', label: 'read', call: 'setPhase', args: ['read'] },
        { group: 'gait', label: 'tool', call: 'setPhase', args: ['tool'] },
        { group: 'gait', label: 'verify', call: 'setPhase', args: ['verify'] },
        { group: 'gait', label: 'write', call: 'setPhase', args: ['write'] },
    ];
}

// Slider range for a tunable: 0 .. 4× its default (a decay-per-frame
// value stays below 1), with a step that gives ~200 positions.
export function tuneSliderSpec(key, def) {
    const max = key === 'passHold' ? 0.999 : def * 4;
    const min = key === 'passHold' ? 0.9 : 0;
    const step = Math.max((max - min) / 200, 0.0001);
    return { min, max, step };
}

const MOODS = ['', 'satisfied', 'curious', 'stuck', 'overloaded', 'idle'];

export function openFaceLab(ctx) {
    const { Core, el, toast } = ctx;
    const face = Core && Core.activeFace;
    if (!face || typeof face.getTune !== 'function') {
        toast('The face module predates the lab — hard-reload', 'error');
        return null;
    }
    let panel = document.getElementById('face-lab');
    if (panel) { panel.classList.toggle('hidden'); return panel; }

    panel = el('div', '');
    panel.id = 'face-lab';
    panel.setAttribute('role', 'dialog');
    panel.setAttribute('aria-label', 'Face lab');

    const head = el('div', 'panel-header');
    head.appendChild(el('span', 'panel-title', 'FACE LAB'));
    const copyBtn = el('button', 'render-btn', 'Copy JSON');
    copyBtn.type = 'button';
    copyBtn.title = 'Copy the current TUNE table';
    copyBtn.addEventListener('click', async () => {
        try {
            await navigator.clipboard.writeText(JSON.stringify(face.getTune(), null, 2));
            toast('TUNE copied');
        } catch (e) { toast('Copy failed', 'error'); }
    });
    head.appendChild(copyBtn);
    const resetBtn = el('button', 'render-btn', 'Reset');
    resetBtn.type = 'button';
    resetBtn.addEventListener('click', () => { face.resetTune(); rebuildSliders(); toast('TUNE reset'); });
    head.appendChild(resetBtn);
    const closeBtn = el('button', 'render-btn', '✕');
    closeBtn.type = 'button';
    closeBtn.setAttribute('aria-label', 'Close the face lab');
    closeBtn.addEventListener('click', () => panel.classList.add('hidden'));
    head.appendChild(closeBtn);
    panel.appendChild(head);

    const body = el('div', 'face-lab-body');
    panel.appendChild(body);

    // ── form + auto ──
    const formRow = el('div', 'face-lab-row');
    formRow.appendChild(el('span', 'face-lab-label', 'form'));
    const formSel = document.createElement('select');
    for (const name of face.getForms()) {
        const o = document.createElement('option');
        o.value = name; o.textContent = name;
        if (name === face.getForm()) o.selected = true;
        formSel.appendChild(o);
    }
    formSel.addEventListener('change', () => { face.setForm(formSel.value); refresh(); });
    formRow.appendChild(formSel);
    body.appendChild(formRow);

    // ── action groups ──
    const groups = {};
    for (const a of labActions()) {
        if (!groups[a.group]) {
            const row = el('div', 'face-lab-row');
            row.appendChild(el('span', 'face-lab-label', a.group));
            groups[a.group] = el('div', 'face-lab-buttons');
            row.appendChild(groups[a.group]);
            body.appendChild(row);
        }
        const b = el('button', 'face-lab-btn', a.label);
        b.type = 'button';
        b.addEventListener('click', () => {
            if (typeof face[a.call] === 'function') face[a.call](...a.args);
            else toast(`${a.call} missing on the face module`, 'error');
            refresh();
        });
        groups[a.group].appendChild(b);
    }

    // ── toggles ──
    const togRow = el('div', 'face-lab-row');
    togRow.appendChild(el('span', 'face-lab-label', 'state'));
    const togs = el('div', 'face-lab-buttons');
    const mkToggle = (label, get, set) => {
        const b = el('button', 'face-lab-btn', label);
        b.type = 'button';
        const paint = () => b.classList.toggle('on', !!get());
        b.addEventListener('click', () => { set(!get()); paint(); refresh(); });
        paint();
        togs.appendChild(b);
        return paint;
    };
    let bgBusy = false, gaze = false;
    mkToggle('background busy', () => bgBusy, (v) => { bgBusy = v; face.setBackgroundBusy(v); });
    mkToggle('gaze', () => gaze, (v) => { gaze = v; face.setComposerGaze(v); });
    if (typeof face.getAutoForm === 'function') {
        mkToggle('auto form', () => face.getAutoForm(), (v) => face.setAutoForm(v));
    }
    togRow.appendChild(togs);
    body.appendChild(togRow);

    // ── mood + audio ──
    const moodRow = el('div', 'face-lab-row');
    moodRow.appendChild(el('span', 'face-lab-label', 'mood'));
    const moodSel = document.createElement('select');
    for (const m of MOODS) {
        const o = document.createElement('option');
        o.value = m; o.textContent = m || 'neutral';
        moodSel.appendChild(o);
    }
    moodSel.addEventListener('change', () => face.setMoodHue(moodSel.value || null));
    moodRow.appendChild(moodSel);
    moodRow.appendChild(el('span', 'face-lab-label', 'audio'));
    const audio = document.createElement('input');
    audio.type = 'range'; audio.min = '0'; audio.max = '1'; audio.step = '0.01'; audio.value = '0';
    let audioTimer = null;
    audio.addEventListener('input', () => {
        // The face low-passes and decays the level; keep feeding it while
        // the slider is held so the value is what you see.
        if (audioTimer) clearInterval(audioTimer);
        audioTimer = setInterval(() => face.setAudioLevel(Number(audio.value)), 40);
    });
    audio.addEventListener('change', () => { if (audioTimer) { clearInterval(audioTimer); audioTimer = null; } });
    moodRow.appendChild(audio);
    body.appendChild(moodRow);

    // ── tunables ──
    const tuneWrap = el('div', 'face-lab-tune');
    body.appendChild(tuneWrap);
    function rebuildSliders() {
        tuneWrap.replaceChildren();
        const tune = face.getTune();
        const defs = face.TUNE_DEFAULTS || tune;
        for (const key of Object.keys(tune)) {
            const row = el('div', 'face-lab-slider');
            const lab = el('label', 'face-lab-key', key);
            const inp = document.createElement('input');
            const spec = tuneSliderSpec(key, defs[key]);
            inp.type = 'range';
            inp.min = String(spec.min); inp.max = String(spec.max); inp.step = String(spec.step);
            inp.value = String(tune[key]);
            inp.id = `face-lab-tune-${key}`;
            lab.setAttribute('for', inp.id);
            const val = el('span', 'face-lab-val', Number(tune[key]).toFixed(3));
            inp.addEventListener('input', () => {
                const v = face.setTune(key, Number(inp.value));
                val.textContent = Number(v).toFixed(3);
                if (Number(v) !== defs[key]) row.classList.add('changed'); else row.classList.remove('changed');
            });
            if (tune[key] !== defs[key]) row.classList.add('changed');
            row.appendChild(lab); row.appendChild(inp); row.appendChild(val);
            tuneWrap.appendChild(row);
        }
    }
    rebuildSliders();

    // ── live readout ──
    const readout = el('pre', 'face-lab-readout');
    body.appendChild(readout);
    function refresh() {
        try {
            // The form may have changed elsewhere (menu, auto mode, API):
            // the select follows the face, not the other way round.
            if (formSel.value !== face.getForm()) formSel.value = face.getForm();
            const d = face.getDebugState();
            const g = d.gait || {};
            readout.textContent = [
                `form ${d.anatomy}  phase ${d.phase || '-'}  auto ${d.autoForm ? 'on' : 'off'}  hint ${d.taskHint || '-'}`,
                `gait s${g.search?.toFixed(2)} r${g.read?.toFixed(2)} t${g.tool?.toFixed(2)} v${g.verify?.toFixed(2)} w${g.write?.toFixed(2)}`,
                `dialect ${d.dialect ? Object.values(d.dialect).join('/') : '-'}`,
                `flow ${d.gaitFlow?.toFixed(2)} thicken ${d.gaitThicken?.toFixed(2)} flash ${d.gaitFlash?.toFixed(2)} align ${d.gaitAlign?.toFixed(2)}`,
                `tool ${d.toolPulse?.toFixed(2)}  recall ${d.recallSpark?.toFixed(2)}  verdict ${d.verdict || '-'} ${d.verdictEnv?.toFixed(2)}`,
                `working ${d.workingState?.toFixed(2)}  userTurn ${d.userTurn?.toFixed(2)}  activity ${d.activity?.toFixed(2)}  immersion ${d.immersion?.toFixed(2)}`,
                `bg ${d.backgroundBusy?.toFixed(2)}  mood ${d.moodHue?.toFixed(3)}  gaze ${d.gazeY?.toFixed(2)}  error ${d.errorKind || '-'} ${d.errorKindEnv?.toFixed(2)}`,
                `conversation ${d.conversation}  tools ${d.tools}  camZ ${d.cameraZ?.toFixed(2)}`,
            ].join('\n');
        } catch (e) { readout.textContent = String(e); }
    }
    const timer = setInterval(() => { if (!panel.classList.contains('hidden') && panel.isConnected) refresh(); }, 125);
    panel.addEventListener('face-lab-destroy', () => clearInterval(timer));
    refresh();

    document.body.appendChild(panel);
    return panel;
}
