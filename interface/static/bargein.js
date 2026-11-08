// Talk over a spoken reply (2026-11-07, operator: "interrupt the ai mid sentence").
// The microphone listens only while a reply to a voice message is being spoken.
// Hearing speech pauses the reply; words cut it and go to the agent, while a cough
// or an empty transcript lets it carry on.
//
// No DOM and no browser audio here: app.js hands in the microphone and the audio
// frames, so all of this runs under node in tests/test_interface_bargein.py.

export const BARGE_DEFAULTS = {
    calibrateMs: 300,    // measure the room (and any speaker bleed) before listening
    startMs: 220,        // voiced this long before it counts as someone talking
    endMs: 900,          // silence this long ends the utterance
    maxMs: 60000,        // one utterance, at most
    prerollMs: 500,      // kept up to the trigger, its own frames included: the first word
    minVoicedMs: 280,    // a trigger alone (256 ms in a browser) is a cough; "wait" measured 341
    minRms: 0.012,       // never trigger below this, however quiet the room
    maxFloor: 0.02,      // someone already talking at open must not read as "the room"
    ratio: 3.5,          // this far above the room counts as voiced
    endRatio: 0.5,       // the bar drops once talking: the reply's bleed stopped at the pause
    maxEndBar: 0.02,     // however loud the room was while the reply played
    stallMs: 1500,       // no frames this long while capturing: the browser suspended audio
};

export function rms(samples) {
    if (!samples || !samples.length) return 0;
    let sum = 0;
    for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
    return Math.sqrt(sum / samples.length);
}

// feed() one frame's level and length; returns 'start', 'end' or null.
export function createSpeechDetector(options = {}) {
    const o = { ...BARGE_DEFAULTS, ...options };
    let phase = 'calibrate', elapsed = 0, levelTime = 0, floor = 0;
    let run = 0, silence = 0, voiced = 0, total = 0, bar = 0;
    const threshold = () => Math.max(o.minRms, floor * o.ratio);
    return {
        voicedMs: () => voiced,
        reachedLimit: () => total >= o.maxMs,
        feed(level, ms) {
            if (phase === 'calibrate') {
                levelTime += level * ms;
                elapsed += ms;
                if (elapsed >= o.calibrateMs) {
                    floor = Math.min(o.maxFloor, levelTime / elapsed);
                    phase = 'listen';
                }
                return null;
            }
            if (phase === 'listen') {
                if (level > threshold()) {
                    run += ms;
                    if (run >= o.startMs) {
                        phase = 'speech';
                        bar = Math.min(threshold() * o.endRatio, o.maxEndBar);
                        voiced = run;
                        total = run;
                        silence = 0;
                        return 'start';
                    }
                } else {
                    run = Math.max(0, run - ms);
                    floor = Math.min(o.maxFloor, floor + (level - floor) * 0.02);
                }
                return null;
            }
            if (phase === 'speech') {
                total += ms;
                if (level > bar) {
                    voiced += ms;
                    silence = 0;
                } else {
                    silence += ms;
                }
                if (silence >= o.endMs || total >= o.maxMs) {
                    phase = 'done';
                    return 'end';
                }
            }
            return null;
        },
    };
}

// One mono PCM clip as a 16-bit WAV: /api/stt takes it as it takes a recording.
export function encodeWav(frames, sampleRate) {
    const count = frames.reduce((n, f) => n + f.length, 0);
    const buffer = new ArrayBuffer(44 + count * 2);
    const view = new DataView(buffer);
    const text = (at, s) => { for (let i = 0; i < s.length; i++) view.setUint8(at + i, s.charCodeAt(i)); };
    text(0, 'RIFF');
    view.setUint32(4, 36 + count * 2, true);
    text(8, 'WAVE');
    text(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    text(36, 'data');
    view.setUint32(40, count * 2, true);
    let at = 44;
    for (const frame of frames) {
        for (let i = 0; i < frame.length; i++, at += 2) {
            const s = Math.max(-1, Math.min(1, frame[i]));
            view.setInt16(at, s < 0 ? s * 0x8000 : s * 0x7fff, true);
        }
    }
    return buffer;
}

// Twin of core/queued_messages.py is_stop_word: a stop said with no turn running
// only ends the speech, so it must not start a new turn. The table in the test
// runs both.
const _STOP_RE = /^(?:stop|cancel|abort|σταματα|σταματησε|ακυρο|ακυρωσε)[\s.!]*$/;
export function isStopWord(text) {
    const folded = String(text || '').normalize('NFD').replace(/\p{M}/gu, '').toLowerCase().trim();
    return _STOP_RE.test(folded);
}

// One listening session at a time. Callbacks, all optional except where used:
//   openMic(onFrame) -> Promise<{close()}>   onFrame(samples, sampleRate)
//   onListening(bool)       the microphone opened or closed
//   onSpeechStart()         someone is talking: pause the reply
//   onUtterance(wav, reachedLimit)  they stopped, or hit maxMs: transcribe this
//   onShort()               too short to be words: resume the reply
//   onMicError(error)       no microphone: the reply simply carries on
// now() is the clock for finishIfStalled.
export function createBargeIn({ openMic, onListening, onSpeechStart, onUtterance, onShort, onMicError, options, now = Date.now }) {
    const o = { ...BARGE_DEFAULTS, ...(options || {}) };
    let state = 'closed', session = 0, mic = null, detector = null;
    let preroll = [], capture = [], lastFrameAt = 0, rate = 0;

    const shut = () => {
        session += 1;
        const was = state;
        state = 'closed';
        preroll = [];
        capture = [];
        detector = null;
        if (mic) {
            try { mic.close(); } catch (_) { /* already gone */ }
            mic = null;
        }
        if (was !== 'closed' && onListening) onListening(false);
    };

    const finish = (dropShort = false) => {
        const voiced = detector.voicedMs();
        const reachedLimit = detector.reachedLimit();
        const frames = capture;
        const sampleRate = rate;
        shut();
        if (voiced < o.minVoicedMs) {
            if (onShort && !dropShort) onShort();
        } else if (onUtterance) {
            onUtterance(encodeWav(frames, sampleRate), reachedLimit);
        }
    };

    const onFrame = (samples, sampleRate) => {
        if (state !== 'listening' && state !== 'capturing') return;
        lastFrameAt = now();
        rate = sampleRate;
        const ms = samples.length / sampleRate * 1000;
        if (state === 'capturing') {
            capture.push(samples);
        } else {
            preroll.push(samples);
            let held = preroll.reduce((n, f) => n + f.length, 0) / sampleRate * 1000;
            while (preroll.length > 1 && held - preroll[0].length / sampleRate * 1000 >= o.prerollMs) {
                held -= preroll[0].length / sampleRate * 1000;
                preroll.shift();
            }
        }
        const event = detector.feed(rms(samples), ms);
        if (event === 'start') {
            capture = preroll;
            preroll = [];
            state = 'capturing';
            if (onSpeechStart) onSpeechStart();
        } else if (event === 'end') {
            finish();
        }
    };

    return {
        state: () => state,
        isCapturing: () => state === 'capturing',
        // The operator pressed the button mid-sentence: what was said so far still counts.
        // dropShort: too short to be words, and the caller wants no resume from it.
        finishNow({ dropShort = false } = {}) {
            if (state === 'capturing') finish(dropShort);
        },
        finishIfStalled() {
            if (state === 'capturing' && now() - lastFrameAt >= o.stallMs) finish();
        },
        async open() {
            if (state !== 'closed') return;
            state = 'opening';
            const mine = ++session;
            let opened = null;
            try {
                opened = await openMic(onFrame);
            } catch (error) {
                if (mine !== session) return;
                state = 'closed';
                if (onMicError) onMicError(error);
                return;
            }
            if (mine !== session) {
                // closed while the permission or the stream was pending
                try { opened.close(); } catch (_) { /* already gone */ }
                return;
            }
            mic = opened;
            detector = createSpeechDetector(o);
            state = 'listening';
            if (onListening) onListening(true);
        },
        close: shut,
    };
}
