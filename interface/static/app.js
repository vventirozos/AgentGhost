import * as matrixGraphFace from './matrix_graph.js?v=10.2';

// --- Voice Globals ---
let isTTSActive = false;
let ttsTextQueue = [];
let ttsAudioQueue = [];
let isFetchingTTS = false;
let isPlayingTTS = false;
let audioCtx = null;
let currentAudioSource = null;
let ttsBuffer = "";
let mediaRecorder;
let audioChunks = [];

let activeFace = matrixGraphFace;

const chatLog = document.getElementById('chat-log');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const fullscreenBtn = document.getElementById('fullscreen-btn');
const statusText = document.getElementById('status-text');
const connectionDot = document.getElementById('connection-dot');
const plannerMonologue = document.getElementById('planner-monologue');

let ws;
let monologueTimeout;
let chatHistory = [];
const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
// The /ws log stream is auth-gated server-side. Browsers can't set custom
// headers on a WebSocket, so pass the injected key as a query param.
const wsUrl = `${wsProtocol}//${window.location.host}/ws?key=${encodeURIComponent(window.GHOST_API_KEY || '')}`;
const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent) ||
    (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
const isSafari = /^((?!chrome|android).)*safari/i.test(navigator.userAgent);

// Wrap window.fetch so every request to our own /api/* automatically
// includes the X-Ghost-Key header injected into the page by the server.
(function installAuthFetch() {
    const apiKey = window.GHOST_API_KEY;
    if (!apiKey) return;
    const origFetch = window.fetch.bind(window);
    window.fetch = function (input, init) {
        try {
            const url = typeof input === 'string' ? input : (input && input.url) || '';
            const isApi = url.startsWith('/api/') || url.includes(window.location.host + '/api/');
            if (isApi) {
                init = init || {};
                const headers = new Headers(init.headers || (typeof input !== 'string' ? input.headers : undefined) || {});
                if (!headers.has('X-Ghost-Key')) headers.set('X-Ghost-Key', apiKey);
                init.headers = headers;
            }
        } catch (e) { /* fall through */ }
        return origFetch(input, init);
    };
})();

// Safe localStorage wrapper (iOS private mode throws QuotaExceededError)
const safeStorage = {
    get(key) {
        try { return localStorage.getItem(key); } catch (e) { return null; }
    },
    set(key, value) {
        try { localStorage.setItem(key, value); } catch (e) {
            console.warn('localStorage unavailable (private mode?)', e.name);
        }
    },
    remove(key) {
        try { localStorage.removeItem(key); } catch (e) {}
    }
};

// ═══════════════════════════════════════════════════════════════
//  Icon vocabulary — aligned 1:1 with src/ghost_agent/utils/logging.py
//  `class Icons` so every log line from the agent maps to a known
//  category, color, and dwell-time. Without this the UI sees
//  `<icon> <+delta> <title> <content>` but only the handful of icons
//  in the old WORKING_ICONS Set got any special treatment — the rest
//  (🐍 🐚 🌐 🔬 💭 🧭 🎯 🗣️ 🤖 📖 👀 ⬇️ 📝 🔎 📍 📚 🧬 🗒️ 🎨 etc.) were
//  treated as generic "non-working" emoji and faded out after 2s,
//  letting the ubiquitous post-LLM 🧠 summary log dominate.
// ═══════════════════════════════════════════════════════════════

// Priority classes. Higher numbers override lower when a log arrives.
//   accent  — one-off lifecycle / status flashes, always visible briefly
//   tool    — external action: code, shell, search, web, image gen
//   memory  — filesystem + memory stores
//   plan    — planning, routing, context assembly
//   think   — raw LLM thinking / replies (the floor; easily overwritten)
//   idle    — system-idle states
// Keys mirror utils/logging.py's Icons class. The Python glyphs were
// migrated to wide-base (non-VS16) forms — WARN ⚠️→🔶, SHIELD 🛡️→🔒,
// CUT ✂️→🔻, etc. This map was regenerated to match (2026-07-15); the old
// two-codepoint VS16 keys ('⚠️','🛡️',…) were unreachable anyway because
// extractIcon's \p{Extended_Pictographic} match strips the trailing U+FE0F.
const ICON_CLASS = {
    // --- lifecycle / status (accent) ---
    '⚡': 'accent',   // SYSTEM_BOOT
    '🚀': 'accent',   // SYSTEM_READY
    '🌅': 'accent',   // BOOT_AWAKE
    '💤': 'idle',     // SYSTEM_SHUT
    '😴': 'idle',
    '🎬': 'accent',   // REQ_START
    '🏁': 'accent',   // REQ_DONE
    '⏳': 'accent',   // REQ_WAIT
    '✅': 'accent',   // OK
    '❌': 'accent',   // FAIL
    '🔶': 'accent',   // WARN
    '🛑': 'accent',   // STOP
    '🔄': 'accent',   // RETRY
    '💡': 'accent',   // IDEA
    '🐛': 'accent',   // BUG
    '🔒': 'accent',   // SHIELD
    '📣': 'accent',   // NOTIFY_OUT
    '🪞': 'accent',   // SELF_STATE
    '🎓': 'accent',   // SKILL_GRADUATE
    '👍': 'accent',   // FEEDBACK_POS — human outcome label (/api/feedback)
    '👎': 'accent',   // FEEDBACK_NEG

    // --- tools (external action) ---
    '🌐': 'tool',     // TOOL_SEARCH
    '🌎': 'tool',     // TOOL_BROWSER
    '🧅': 'tool',     // TOOL_DARKWEB
    '🔬': 'tool',     // TOOL_DEEP
    '🐍': 'tool',     // TOOL_CODE
    '🐚': 'tool',     // TOOL_SHELL
    '📥': 'tool',     // TOOL_DOWN / Docker pull
    '🎨': 'tool',     // IMAGE_GEN
    '📄': 'tool',     // REPORT_PDF
    '🎮': 'tool',     // GAME_MOVE
    '🎲': 'tool',     // UNCERTAINTY_DIE
    '🧪': 'tool',     // VERIFIER_LAB
    '📦': 'tool',     // SANDBOX_BOX
    '🏃': 'tool',     // JOB_PROMOTE — long sandbox command detached as a job
    '🔧': 'tool',     // NODE_WORKER
    '🛸': 'tool',     // NODE_EDGE

    // --- memory / filesystem ---
    '💾': 'memory',   // TOOL_FILE_W
    '📖': 'memory',   // TOOL_FILE_R
    '📙': 'memory',   // TOOL_FILE_M — multi-path batch read
    '🔍': 'memory',   // TOOL_FILE_S
    '👀': 'memory',   // TOOL_FILE_I
    '📝': 'memory',   // MEM_SAVE
    '🔎': 'memory',   // MEM_READ
    '📍': 'memory',   // MEM_MATCH
    '📚': 'memory',   // MEM_INGEST
    '📑': 'memory',   // MEM_SPLIT
    '🔻': 'memory',   // CUT
    '🧬': 'memory',   // MEM_EMBED
    '🧮': 'memory',   // VECTOR_EMBED (de-collided from 🧬 2026-07-24)
    '🧹': 'memory',   // MEM_WIPE
    '📔': 'memory',   // MEM_SCRATCH
    '💪': 'memory',   // MEM_REINFORCE
    '📇': 'memory',   // MEM_INDEX
    '📓': 'memory',   // MEM_LIBRARY
    '🪢': 'memory',   // GRAPH_WEB
    '🧾': 'memory',   // BELIEF_SCALES
    '📶': 'memory',   // THRESHOLD_TUNE
    '🎥': 'memory',   // EPISODE_REEL
    '📡': 'memory',   // ACTIVITY
    '🔀': 'memory',   // EVENT_BUS (de-collided from 📡 2026-07-24)
    '👤': 'memory',   // USER_ID
    '🐘': 'memory',   // POSTGRES

    // --- planning / routing ---
    '📋': 'plan',     // BRAIN_PLAN
    '🧩': 'plan',     // BRAIN_CTX
    '🧭': 'plan',     // BRAIN_ROUTE
    '🎯': 'plan',     // BRAIN_AIM
    '🌳': 'plan',     // MCTS_TREE
    '🔮': 'plan',     // FORESIGHT — shadow world model
    '🔗': 'plan',     // CONSTRAINT

    // --- raw thinking (the floor) ---
    '🧠': 'think',    // BRAIN_SUM — the one that was drowning everything
    '💭': 'think',    // BRAIN_THINK
    '💬': 'think',    // LLM_ASK
    '🤖': 'think',    // LLM_REPLY

    // --- idle / skipped ---
    '⏩': 'idle',     // SKIP
    '🌙': 'plan',     // DREAM
    '🫀': 'idle',     // HEARTBEAT — biological watchdog, routine

    // --- metacognition (§LOG-7 — registered in Icons; were literal
    // glyphs in metacog_log.py that fell through to the 'think' floor) ---
    '🫧': 'plan',     // METACOG (generic uplift)
    '🌱': 'plan',     // METACOG_BOOT
    '📊': 'plan',     // METACOG_SUMMARY
    '📈': 'plan',     // METACOG_CONF
    '📐': 'plan',     // METACOG_CALIB
    '🥇': 'plan',     // METACOG_ARBITER
    '🚧': 'plan',     // METACOG_VALID
    '💻': 'plan',     // METACOG_HOST
    '🚦': 'plan',     // METACOG_REPLAN
    '🚪': 'plan',     // METACOG_GATE

    // --- misc ---
    '🫥': 'accent',   // MODE_GHOST
    '🔥': 'accent',
};

// (The dwell/priority tables that arbitrated the center-stage
// #activity-icon left with it, 2026-07-29 — the turn status line shows
// the CURRENT corridor step verbatim, no arbitration needed.)

function _iconClass(icon) { return ICON_CLASS[icon] || 'think'; }

// Used elsewhere (setWorkingState gating). "Working" = the agent is
// actively doing something a user cares to watch; true for everything
// except idle / SYSTEM_SHUT — and except the human-feedback receipts
// (👍/👎), which are terminal bookkeeping: labeling yesterday's reply
// must not spin the face into a 60s "working" animation on an idle agent.
const _NON_WORKING_ICONS = new Set(['👍', '👎']);
const WORKING_ICONS = new Set(Object.keys(ICON_CLASS).filter(i =>
    ICON_CLASS[i] !== 'idle' && !_NON_WORKING_ICONS.has(i)
));
const IDLE_ICONS = new Set(Object.keys(ICON_CLASS).filter(i =>
    ICON_CLASS[i] === 'idle' || i === '✅' || i === '❌' || i === '🛑'
));

let isProcessingRequest = false;
// True only while we're rebuilding the chat log from saved/loaded history.
// Auto-opening the visualizer (for renderable code blocks and images) is
// desirable for a fresh live reply, but jarring when restoring history —
// it would pop the floating window open on every page load. The image
// auto-open runs from an async MutationObserver microtask, so a plain
// flag cleared on a macrotask (setTimeout 0) reliably stays set through
// the observer pass that processes the restored nodes.
let _restoringHistory = false;
function _withHistoryRestore(fn) {
    _restoringHistory = true;
    try { fn(); }
    finally { setTimeout(() => { _restoringHistory = false; }, 0); }
}
let currentChatController = null;
let currentTaskId = null;
let currentChunkIndex = 0;
let currentAgentMessageDiv = null;
let currentAccumulatedContent = "";
// The AGENT's request id for the in-flight turn (from the SSE frames'
// "chatcmpl-<id>" envelope) — the handle /api/feedback labels ride on.
// Distinct from currentTaskId, which is the interface proxy's stream handle.
let currentReqId = null;
// The text of the turn this tab most recently sent — the only identifier
// available when durable sessions are off. See `_resolveOwnTurnId`.
let _lastSentUserText = '';
let currentThinkingInterval = null;
let currentTTSMutedLength = 0;

const SEND_SVG = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>`;
const CANCEL_SVG = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="9" y1="9" x2="15" y2="15"></line><line x1="15" y1="9" x2="9" y2="15"></line></svg>`;

function toggleSendButtonUI(isProcessing, stoppable = true) {
    // Every request path (chat, voice, resume) ends through here with
    // false — one idempotent hook covers all ticker teardown.
    if (!isProcessing) stopTurnTicker();
    if (sendBtn) {
        if (isProcessing) {
            // ⚠ `stoppable` distinguishes a turn Stop can abort from an
            // upload/workspace operation that merely holds the turn flag.
            // Making those visible (F14) gave them a live-looking Stop
            // button wired to nothing, because the handler needs
            // `currentChatController` (R3 lens B). Show BUSY, not STOP.
            sendBtn.innerHTML = stoppable ? CANCEL_SVG : SEND_SVG;
            sendBtn.classList.toggle('send-btn-cancel', !!stoppable);
            sendBtn.disabled = !stoppable;
        } else {
            sendBtn.innerHTML = SEND_SVG;
            sendBtn.classList.remove('send-btn-cancel');
            sendBtn.disabled = false;
        }
    }
}

// --- Notification & Service Worker Setup ---
// iOS Safari: Notification API and Web Push are only available for installed PWAs.
// Skip on iOS unless running in standalone (home-screen) mode.
const isStandalonePWA = window.matchMedia('(display-mode: standalone)').matches ||
    window.navigator.standalone === true;
const notificationsSupported = ('Notification' in window) && (!isIOS || isStandalonePWA);

// iOS standalone (installed PWA) MUST register the SW too — it is the
// only iOS context that can hold a push subscription. The old blanket
// `!isIOS` skip kept even installed PWAs pushless.
if ('serviceWorker' in navigator && (!isIOS || isStandalonePWA)) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js')
            .then(reg => console.log('Service Worker registered successfully'))
            .catch(err => console.error('Service Worker registration failed:', err));
    });
}

// ── Web push subscription (2026-08-01) ─────────────────────────────
// Subscribes this device for locked-phone delivery: reply-ready pushes
// (the ack-grace contract in server.py) + agent notify events. Runs
// after permission lands; re-runs each boot (subscribe is idempotent —
// the server upserts by endpoint).
function _b64urlToUint8(base64url) {
    const pad = '='.repeat((4 - (base64url.length % 4)) % 4);
    const b64 = (base64url + pad).replace(/-/g, '+').replace(/_/g, '/');
    const raw = atob(b64);
    return Uint8Array.from(raw, (c) => c.charCodeAt(0));
}

async function ensurePushSubscription() {
    try {
        if (!('serviceWorker' in navigator) || !('PushManager' in window)) return;
        if (isIOS && !isStandalonePWA) return;   // Safari tab: no push context
        if (Notification.permission !== 'granted') return;
        const res = await fetch('/api/push/vapid');
        if (!res.ok) return;
        const { enabled, key } = await res.json();
        if (!enabled || !key) return;
        const registration = await navigator.serviceWorker.ready;
        let sub = await registration.pushManager.getSubscription();
        // VAPID rotation: a subscription bound to a RETIRED server key
        // looks alive on both ends (server upserts it forever, pushes
        // fail 401/403) and never delivers again. Compare and re-bind.
        if (sub && sub.options && sub.options.applicationServerKey) {
            const cur = new Uint8Array(sub.options.applicationServerKey);
            const want = _b64urlToUint8(key);
            const same = cur.length === want.length
                && cur.every((v, i) => v === want[i]);
            if (!same) {
                try { await sub.unsubscribe(); } catch (e) { /* re-bind anyway */ }
                sub = null;
            }
        }
        if (!sub) {
            sub = await registration.pushManager.subscribe({
                userVisibleOnly: true,
                applicationServerKey: _b64urlToUint8(key),
            });
        }
        await fetch('/api/push/subscribe', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ subscription: sub.toJSON() }),
        });
    } catch (e) {
        console.warn('Push subscription unavailable:', e);
    }
}

document.addEventListener('click', () => {
    if (notificationsSupported && Notification.permission === "default") {
        try {
            const result = Notification.requestPermission();
            if (result && typeof result.then === 'function') {
                result.then(() => ensurePushSubscription())
                      .catch(e => console.error('Notification permission error:', e));
            } else {
                ensurePushSubscription();
            }
        } catch (e) {
            console.error('Notification permission error:', e);
        }
    }
}, { once: true });

// Already-granted permission (returning visitor): subscribe on boot.
window.addEventListener('load', () => { ensurePushSubscription(); });

async function notifyUser(message) {
    if (!notificationsSupported) return;
    if (Notification.permission !== "granted" || !document.hidden) return;

    try {
        if ('serviceWorker' in navigator) {
            const registration = await navigator.serviceWorker.ready;
            registration.showNotification("Ghost System", {
                body: message,
                icon: "/static/icons/icon-192.png?v=2",
                // Keyed URL: the SW click handler opens data.url; a bare
                // '/' would land a fresh window on the 401 page.
                data: { url: location.pathname + location.search }
            });
        } else {
            new Notification("Ghost System", {
                body: message, icon: "/static/icons/icon-192.png?v=2" });
        }
    } catch (e) {
        console.error("Notification failed:", e);
    }
}
// ------------------------------------

function setConnectionState(state, label) {
    // state: 'online' | 'pending' | 'busy' | 'error'
    // Clearing inline styles so the CSS class wins — the old onopen/onclose
    // handlers set inline background-color, which would override classes.
    if (connectionDot) {
        connectionDot.style.backgroundColor = '';
        connectionDot.style.boxShadow = '';
        connectionDot.classList.remove('state-online', 'state-pending', 'state-busy', 'state-error');
        connectionDot.classList.add(`state-${state}`);
    }
    if (statusText && label) statusText.textContent = label;
}

let _wsReconnectTimer = null;
function connectWebSocket() {
    // Coalesce overlapping triggers (the 3s onclose timer races the
    // visibilitychange/pageshow handlers). Without this, a second call
    // overwrote `ws` while the first socket stayed open and un-torn-down —
    // an orphan that kept firing onmessage (every log line processed twice)
    // and spawned its own onclose reconnect chain, multiplying sockets.
    if (ws && (ws.readyState === WebSocket.CONNECTING
               || ws.readyState === WebSocket.OPEN)) {
        return;
    }
    if (_wsReconnectTimer) { clearTimeout(_wsReconnectTimer); _wsReconnectTimer = null; }
    // Detach the old socket's handlers before replacing it so a late
    // onclose from the previous socket can't schedule a duplicate reconnect.
    if (ws) { ws.onopen = ws.onmessage = ws.onclose = ws.onerror = null; }
    setConnectionState('pending', 'CONNECTING…');
    ws = new WebSocket(wsUrl);
    ws.onopen = () => {
        setConnectionState('online', 'ONLINE');
    };
    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.type === 'log') {
                const icon = extractIcon(data.content);
                const flashColor = getIconColor(icon);

                // Feed the live log console's ring buffer (renders only
                // while the drawer is open; collects regardless).
                pushLogEntry(data.content, data.is_error);
                // …and the in-turn activity ticker (no-op unless a turn
                // is showing its typing indicator right now).
                noteTickerLine(data.content);

                // Log lines feed the face's activity ENVELOPE — a small
                // energy contribution each, smoothed inside matrix_graph —
                // instead of firing a full pulse per line, which strobed
                // the scene several times a second on a busy agent.
                const isMonologue = data.content.includes("PLANNER MONOLOGUE");
                if (activeFace.noteActivity) {
                    activeFace.noteActivity(isMonologue ? 0.08 : 0.16, flashColor);
                } else if (isMonologue) {
                    activeFace.triggerSmallPulse();
                } else {
                    activeFace.triggerPulse(flashColor);
                }

                if (icon) {
                    updateStateFromIcon(icon);
                    // Genuine failures only: ⚠️ warnings are routine (node
                    // fallbacks, heals) and kept tinting the face on
                    // ordinary healthy turns.
                    if (['❌', '🛑', '🔥'].includes(icon)) activeFace.triggerSpike();
                }
                if (data.is_error) activeFace.triggerSpike();

                // Check for Planner Monologue
                const plannerMatch = data.content.match(/PLANNER MONOLOGUE\s*:\s*(.*)/);
                if (plannerMatch && plannerMatch[1]) {
                    showPlannerMonologue(plannerMatch[1]);
                }
            }
        } catch (e) { console.error("WebSocket Error:", e); }
    };
    ws.onclose = () => {
        setConnectionState('error', 'DISCONNECTED');
        if (_wsReconnectTimer) clearTimeout(_wsReconnectTimer);
        _wsReconnectTimer = setTimeout(connectWebSocket, 3000);
    };
}

// iOS Safari suspends background timers; force reconnect when the tab comes back.
document.addEventListener('visibilitychange', () => {
    if (document.visibilityState !== 'visible') return;
    if (!ws || ws.readyState === WebSocket.CLOSED || ws.readyState === WebSocket.CLOSING) {
        connectWebSocket();
    }
});
window.addEventListener('pageshow', (e) => {
    if (e.persisted && (!ws || ws.readyState !== WebSocket.OPEN)) {
        connectWebSocket();
    }
    // Re-arm the artifact observer on a bfcache restore. It is only
    // observed once at module init, so any teardown left it dead for the
    // life of the restored page (see the pagehide handler).
    if (e.persisted) {
        try {
            const el = document.getElementById('chat-log');
            if (el) chatObserver.observe(el, { childList: true, subtree: true });
        } catch (err) { /* ignore */ }
    }
});

// VisualViewport: track virtual keyboard height so the input area stays visible on iOS.
//
// "Is a keyboard open?" can NOT be derived from innerHeight vs vv.height:
// in standalone-PWA contexts innerHeight SHRINKS with the keyboard (unlike
// Safari-browser, where it stays at layout height) — that comparison read
// keyboard-open as no-keyboard, cleared the body pin, and the user typed
// blind behind the keyboard (operator repro 2026-08-01). The reliable
// baseline is the MAXIMUM vv height seen in this orientation: a focused
// editable + a vv meaningfully below that maximum = keyboard.
let _vvMaxH = 0;
function _vvEditing() {
    const ae = document.activeElement;
    return !!ae && (ae.tagName === 'TEXTAREA' || ae.tagName === 'INPUT'
        || ae.isContentEditable);
}
function _vvNoteHeight() {
    const vv = window.visualViewport;
    if (vv && Math.abs(window.innerWidth - vv.width) <= 2
            && vv.height > _vvMaxH) {
        _vvMaxH = vv.height;
    }
}
function _vvKeyboardOpen() {
    const vv = window.visualViewport;
    return !!vv && _vvEditing() && (_vvMaxH - vv.height) > 80;
}
function _vvResetMax() { _vvMaxH = 0; }   // rotation: re-learn the baseline

if ('visualViewport' in window) {
    let kbSettleTimer = null;
    let kbGuardRetries = 0;
    let lastWinWidth = window.innerWidth;

    const setKb = (px) => document.documentElement.style.setProperty(
        '--keyboard-height', `${px}px`);
    const isEditing = _vvEditing;

    const updateKeyboardOffset = () => {
        const vv = window.visualViewport;
        _vvNoteHeight();
        let kb = Math.max(0, window.innerHeight - vv.height - vv.offsetTop);
        // Mid-rotation, innerHeight and vv.height describe DIFFERENT
        // orientations (bogus half-screen "keyboard"). Defer while the
        // widths disagree — but only a few times: real iOS Safari can
        // report slightly incoherent widths indefinitely, and an
        // ever-deferring guard would freeze keyboard tracking entirely.
        if (Math.abs(window.innerWidth - vv.width) > 2 && kbGuardRetries < 8) {
            kbGuardRetries += 1;
            clearTimeout(kbSettleTimer);
            kbSettleTimer = setTimeout(updateKeyboardOffset, 250);
            return;
        }
        kbGuardRetries = 0;
        // No focused editable = no keyboard; anything left over is stale.
        if (!isEditing()) kb = 0;
        // ── ONE MECHANISM AT A TIME ────────────────────────────────
        // `syncBodyHeight` pins document.body to the visual viewport
        // whenever _vvKeyboardOpen(), which ALREADY lifts the footer to
        // sit just above the keyboard. Translating #ui-layer on top of
        // that applies the same correction twice and drives the composer
        // clean off the TOP of the screen, overlapping the status bar
        // (operator screenshot, iOS, after a long reply).
        //
        // The transform stays for the window the body pin does not
        // cover: _vvKeyboardOpen() additionally requires this
        // orientation's max height to be learned and the drop to exceed
        // 80px, so a just-focused input or an unlearned baseline still
        // needs it. Whichever is active, exactly one of them is.
        if (_vvKeyboardOpen()) kb = 0;
        // A real mobile keyboard never exceeds ~55% of the screen.
        kb = Math.min(kb, Math.round(window.innerHeight * 0.55));
        setKb(kb);
    };
    window.visualViewport.addEventListener('resize', updateKeyboardOffset);
    window.visualViewport.addEventListener('scroll', updateKeyboardOffset);

    const scheduleSettle = (delay) => {
        clearTimeout(kbSettleTimer);
        kbSettleTimer = setTimeout(updateKeyboardOffset, delay);
    };

    // Rotation is the state we can never trust on iOS Safari: it dismisses
    // the keyboard but can KEEP the element focused, and visualViewport
    // may report stale geometry until the next user interaction — every
    // derived value is suspect (the "input box at the top of the screen"
    // repro: focus → rotate → rotate back). Don't repair the lie — make
    // the state deterministic: blur (the keyboard is gone anyway), zero
    // the translate NOW, re-measure once settled.
    const onRotate = () => {
        if (isEditing()) { try { document.activeElement.blur(); } catch (e) {} }
        setKb(0);
        _vvResetMax();   // the old orientation's max height is meaningless now
        scheduleSettle(400);
    };
    window.addEventListener('orientationchange', onRotate);
    window.addEventListener('resize', () => {
        if (window.innerWidth !== lastWinWidth) {
            // Width change == rotation (mobile viewports don't resize
            // horizontally otherwise; desktop windows don't run this UI
            // with a virtual keyboard).
            lastWinWidth = window.innerWidth;
            onRotate();
            return;
        }
        scheduleSettle(250);
    });

    // Keyboard closed via blur: zero the translate immediately — the vv
    // resize event can arrive late or (WebKit quirk) with stale numbers.
    // Small delay so focus hopping between two editables doesn't flicker.
    document.addEventListener('focusout', () => {
        setTimeout(() => { if (!isEditing()) setKb(0); }, 50);
    });
}

// ═══════════════════════════════════════════════════════════════
//  Live log console (2026-07-13)
//
//  The pretty-log stream already arrives over the WebSocket (it drives
//  the face's activity envelope + the planner monologue) — this just
//  makes it READABLE: a bottom-drawer console toggled from the header.
//  A ring buffer collects even while the drawer is closed, so opening
//  shows recent history instead of starting blind. Lines carry the same
//  icon→jewel-tone accent mapping as the face (left border), with plain
//  dim text for readability on the dark theme.
// ═══════════════════════════════════════════════════════════════

const LOG_BUFFER_CAP = 500;
const logBuffer = [];
const logsBtn = document.getElementById('logs-btn');
const logConsole = document.getElementById('log-console');
const logConsoleBody = document.getElementById('log-console-body');
const logClearBtn = document.getElementById('log-clear');
const logCloseBtn = document.getElementById('log-close');
const logResumeBtn = document.getElementById('log-resume');
const logNewCount = document.getElementById('log-new-count');
let logPinned = true;    // auto-scroll follows the tail until the user scrolls up
let logUnseen = 0;

// CSI colour/cursor sequences from the agent's pretty stream.
const ANSI_ESCAPE_RE = /\x1b\[[0-9;]*[A-Za-z]/g;

function cleanLogLine(raw) {
    return String(raw || '').replace(ANSI_ESCAPE_RE, '').replace(/\s+$/, '');
}

function pushLogEntry(rawContent, isError) {
    const text = cleanLogLine(rawContent);
    if (!text.trim()) return;
    const icon = extractIcon(text);
    const entry = {
        text,
        accent: icon ? getIconColor(icon) : '',
        isError: !!isError,
    };
    logBuffer.push(entry);
    if (logBuffer.length > LOG_BUFFER_CAP) logBuffer.shift();
    if (logConsole && !logConsole.classList.contains('hidden')) {
        logConsoleBody.appendChild(buildLogLine(entry));
        while (logConsoleBody.childElementCount > LOG_BUFFER_CAP) {
            logConsoleBody.removeChild(logConsoleBody.firstChild);
        }
        if (logPinned) {
            logConsoleBody.scrollTop = logConsoleBody.scrollHeight;
        } else {
            logUnseen++;
            if (logNewCount) logNewCount.textContent = String(logUnseen);
            if (logResumeBtn) logResumeBtn.classList.remove('hidden');
        }
    }
}

function buildLogLine(entry) {
    const div = document.createElement('div');
    div.className = 'log-line' + (entry.isError ? ' log-line-error' : '');
    if (entry.accent) div.style.borderLeftColor = entry.accent;
    div.textContent = entry.text;
    return div;
}

function openLogConsole() {
    if (!logConsole) return;
    const frag = document.createDocumentFragment();
    for (const entry of logBuffer) frag.appendChild(buildLogLine(entry));
    logConsoleBody.innerHTML = '';
    logConsoleBody.appendChild(frag);
    logConsole.classList.remove('hidden');
    if (logsBtn) logsBtn.classList.add('active');
    logPinned = true;
    logUnseen = 0;
    if (logResumeBtn) logResumeBtn.classList.add('hidden');
    logConsoleBody.scrollTop = logConsoleBody.scrollHeight;
}

function closeLogConsole() {
    if (!logConsole) return;
    logConsole.classList.add('hidden');
    if (logsBtn) logsBtn.classList.remove('active');
}

if (logsBtn) {
    logsBtn.addEventListener('click', () => {
        if (logConsole.classList.contains('hidden')) openLogConsole();
        else closeLogConsole();
    });
}
if (logCloseBtn) logCloseBtn.addEventListener('click', closeLogConsole);
if (logClearBtn) {
    logClearBtn.addEventListener('click', () => {
        logBuffer.length = 0;
        logConsoleBody.innerHTML = '';
        logUnseen = 0;
        if (logResumeBtn) logResumeBtn.classList.add('hidden');
    });
}
if (logConsoleBody) {
    logConsoleBody.addEventListener('scroll', () => {
        const nearBottom = logConsoleBody.scrollTop + logConsoleBody.clientHeight
            >= logConsoleBody.scrollHeight - 40;
        if (nearBottom) {
            logPinned = true;
            logUnseen = 0;
            if (logResumeBtn) logResumeBtn.classList.add('hidden');
        } else {
            logPinned = false;
        }
    }, { passive: true });
}
if (logResumeBtn) {
    logResumeBtn.addEventListener('click', () => {
        logPinned = true;
        logUnseen = 0;
        logResumeBtn.classList.add('hidden');
        logConsoleBody.scrollTop = logConsoleBody.scrollHeight;
    });
}

function showPlannerMonologue(text) {
    if (!plannerMonologue) return;
    plannerMonologue.textContent = text.trim();
    plannerMonologue.classList.add('visible');

    // Clear any existing hide timer so it stays visible while updating
    clearTimeout(monologueTimeout);

    // If we are NOT currently processing a request (e.g. late logs after reply),
    // ensure we still auto-hide after a short delay so it doesn't get stuck open.
    if (!isProcessingRequest) {
        monologueTimeout = setTimeout(hidePlannerMonologue, 2000);
    }
}

function hidePlannerMonologue() {
    if (!plannerMonologue) return;
    plannerMonologue.classList.remove('visible');
}

function extractIcon(logLine) {
    const match = logLine.match(/(\p{Extended_Pictographic})/u);
    return match ? match[0] : null;
}

// Map each class to a distinct accent color for the face's slow mood
// tint. THERMAL family (2026-07-28, matching matrix_graph's blue↔red
// two-pole scheme): cold blues for memory/planning/success, hot reds
// for external action and failure, violet for raw thought — the bridge
// hue between the poles. These BLEND into the sphere's accent, so a
// full neon here would drag the whole theme bright again.
const _ICON_CLASS_COLOR = {
    accent_ok:   '#16418c',  // ✅ settled blue — success = cooled
    accent_err:  '#a3123a',  // ❌ 🛑 🔥 hot crimson (WARN 🔶 is routine, stays accent)
    accent:      '#1f3fa8',  // ⚡ 🚀 🎬 🏁 electric blue (dark)
    tool:        '#6e1530',  // dark arterial red — external action = heat
    memory:      '#14297a',  // deep indigo — stored, cold
    plan:        '#2a2470',  // blue-violet
    think:       '#3a1a5e',  // violet — the bridge between the poles
    idle:        '#10131f',  // blue-black slate
};

function getIconColor(icon) {
    // Error/success get special sub-colors — they're visually the
    // most meaningful accent events the user can see flash by.
    if (icon === '✅') return _ICON_CLASS_COLOR.accent_ok;
    if (icon === '❌' || icon === '🛑' || icon === '🔥') {
        return _ICON_CLASS_COLOR.accent_err;
    }
    return _ICON_CLASS_COLOR[_iconClass(icon)] || _ICON_CLASS_COLOR.think;
}

// The center-stage #activity-icon was REMOVED 2026-07-29 (operator:
// "remove the icon from the middle") — the turn status line under the
// waiting reply bubble carries the per-turn icon now, and ambient
// activity still animates the face itself. updateStateFromIcon stays:
// it drives the FACE's working state from the log stream, icon element
// or not.

let workTimer;
function updateStateFromIcon(icon) {
    if (isProcessingRequest) return; // Prevent logs from turning off the active state during a request

    if (WORKING_ICONS.has(icon)) {
        activeFace.setWorkingState(true);
        clearTimeout(workTimer);
        workTimer = setTimeout(() => {
            if (!isProcessingRequest) {
                activeFace.setWorkingState(false);
            }
        }, 60000);
    } else if (IDLE_ICONS.has(icon)) {
        activeFace.setWorkingState(false);
        clearTimeout(workTimer);
    }
}

// Languages that get a "Visualize" button instead of inline display.
// Declared here (not next to attachRenderButtons further down) because
// decorateCodeBlocks references it, and loadChatState() — which calls
// decorateCodeBlocks — runs at module top-level *before* the render
// section below would otherwise initialize this const (TDZ).
const RENDERABLE_LANGS = new Set([
    'language-html', 'language-css', 'language-javascript', 'language-js',
    'language-mermaid', 'language-csv'
]);

// Decorate <pre><code> blocks inside a rendered message with a small
// language pill + copy button. Runs on each marked.parse() reassignment;
// the `data-decorated` marker makes it idempotent so repeated streaming
// updates don't stack buttons.
function decorateCodeBlocks(root) {
    if (!root) return;
    const pres = root.querySelectorAll('pre');
    pres.forEach(pre => {
        if (pre.dataset.decorated === '1') return;
        const code = pre.querySelector('code');
        if (!code) return;
        // Skip renderable blocks — `attachRenderButtons` owns those; we
        // don't want two buttons stacking in the same corner.
        if (pre.classList.contains('renderable-hidden')) return;
        // During streaming a renderable block (html/css/js/mermaid/csv)
        // isn't `.renderable-hidden` yet, so also skip by language. Else
        // we'd decorate it with a lang badge + Copy header that ends up
        // orphaned above the hidden code once the Visualize button lands.
        const langClass = [...code.classList].find(c => c.startsWith('language-'));
        if (langClass && RENDERABLE_LANGS.has(langClass)) return;

        pre.dataset.decorated = '1';
        pre.classList.add('decorated-code');

        const lang = langClass ? langClass.replace('language-', '') : 'text';

        // Header carries only the language badge; the ACTION buttons
        // (copy, expand) live in a footer bar at the BOTTOM of the block
        // (operator request) — which also keeps them reachable while the
        // collapsed window is scrolled.
        const header = document.createElement('div');
        header.className = 'code-header';

        const badge = document.createElement('span');
        badge.className = 'code-lang';
        badge.textContent = lang;
        header.appendChild(badge);

        const footer = document.createElement('div');
        footer.className = 'code-footer';

        const copyBtn = document.createElement('button');
        copyBtn.className = 'code-copy';
        copyBtn.type = 'button';
        copyBtn.textContent = 'Copy';
        copyBtn.addEventListener('click', async () => {
            try {
                await navigator.clipboard.writeText(code.textContent || '');
                copyBtn.textContent = 'Copied';
                copyBtn.classList.add('copied');
                setTimeout(() => {
                    copyBtn.textContent = 'Copy';
                    copyBtn.classList.remove('copied');
                }, 1400);
            } catch (e) {
                copyBtn.textContent = 'Failed';
                setTimeout(() => { copyBtn.textContent = 'Copy'; }, 1400);
            }
        });
        footer.appendChild(copyBtn);

        // Long outputs render as a fixed-height window that SCROLLS
        // internally (right-hand scrollbar) — the old version clipped the
        // overflow entirely until expanded. The expander still toggles to
        // full inline height. Threshold is generous so ordinary snippets
        // never shrink. (Streaming rebuilds reset to collapsed; the final
        // decorate pass after the stream settles it.)
        const lineCount = (code.textContent.match(/\n/g) || []).length + 1;
        if (lineCount > 34) {
            pre.classList.add('code-collapsed');
            const expand = document.createElement('button');
            expand.className = 'code-expand';
            expand.type = 'button';
            expand.textContent = `Expand · ${lineCount} lines`;
            expand.addEventListener('click', () => {
                const collapsed = pre.classList.toggle('code-collapsed');
                expand.textContent = collapsed ? `Expand · ${lineCount} lines` : 'Collapse';
            });
            footer.appendChild(expand);
        }

        pre.insertBefore(header, pre.firstChild);
        pre.appendChild(footer);
    });
}

// Empty-state hero. Operator direction 2026-08-16: the suggestion chips
// ("What are you capable of?" …) are GONE, and the wordmark is quiet.
//
// The reasoning: this sits on top of a live animated constellation that is
// the most striking thing on the screen. A heavy 700-weight, 10px-tracked,
// double-glow title plus three buttons competed with it and won — which is
// why the result read as cluttered. An empty state is a threshold, not a
// billboard: a hairline wordmark, a rule, and one line of orientation.
function renderEmptyStateHero() {
    if (!chatLog) return;
    // Don't stack heroes if one's already there.
    if (chatLog.querySelector('.empty-hero')) return;

    const hero = document.createElement('div');
    hero.className = 'empty-hero';

    const title = document.createElement('div');
    title.className = 'empty-hero-title';
    // Per-letter spans so the reveal can cascade; the trailing letter's
    // tracking is trimmed in CSS so the word stays optically centred.
    for (const ch of 'GHOST') {
        const span = document.createElement('span');
        span.textContent = ch;
        title.appendChild(span);
    }

    const rule = document.createElement('div');
    rule.className = 'empty-hero-rule';

    const sub = document.createElement('div');
    sub.className = 'empty-hero-sub';
    sub.textContent = window.matchMedia('(pointer: coarse)').matches
        ? 'Ask anything'
        : 'Ask anything · ⌘K for commands';

    hero.appendChild(title);
    hero.appendChild(rule);
    hero.appendChild(sub);
    chatLog.appendChild(hero);
}
function dismissEmptyStateHero() {
    if (!chatLog) return;
    const hero = chatLog.querySelector('.empty-hero');
    if (hero) hero.remove();
}

// Render untrusted markdown safely.
//
// marked.parse() passes raw HTML through by default (GFM spec). Agent
// responses are untrusted — a tool output containing <script>…</script>
// or `<img onerror=…>` outside a code fence would execute in-page.
// DOMPurify strips dangerous nodes/attributes without mangling the rest
// of the markdown. If either library failed to load (CDN down, offline
// deploy), fall back to plain text rather than silently leaking unsafe
// HTML into the DOM.
function renderMarkdown(text) {
    const raw = String(text ?? "");
    if (window.marked && window.DOMPurify) {
        try {
            return window.DOMPurify.sanitize(window.marked.parse(raw));
        } catch (e) {
            console.warn('markdown render failed, falling back to text', e);
        }
    }
    // Fallback: HTML-escape and wrap in <p> so line breaks survive.
    const escaped = raw
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
    return escaped;
}

// Strip the agent's internal-only markup before display: <tool_call>
// blocks AND <think> reasoning blocks (including an unclosed trailing
// one mid-stream). This keeps the visible transcript consistent with
// what the TTS engine speaks — previously the display stripped only
// <tool_call>, so raw reasoning leaked into the chat bubble while being
// suppressed in audio.
function _stripInternalTags(text) {
    return String(text ?? "")
        .replace(/<think>[\s\S]*?<\/think>/gi, '')   // closed reasoning
        .replace(/<think>[\s\S]*$/i, '')             // unclosed (still streaming)
        .replace(/<tool_call[\s\S]*?(?:<\/tool_call>|$)/gi, '')
        .trim();
}

// Day separators: a live conversation spanning midnight gets a quiet
// mono date rule between the two days. Live messages only — restored
// history renders without them (no reliable per-message timestamps).
function _maybeInsertDaySeparator(ts) {
    if (_restoringHistory || !chatLog) return;
    const prior = chatLog.querySelectorAll('.message[data-ts]');
    const last = prior.length ? prior[prior.length - 1] : null;
    const lastTs = last ? Number(last.dataset.ts || 0) : 0;
    if (!lastTs) return;
    if (new Date(lastTs).toDateString() !== new Date(ts).toDateString()) {
        const sep = document.createElement('div');
        sep.className = 'day-separator';
        sep.textContent = new Date(ts).toLocaleDateString(undefined,
            { weekday: 'short', month: 'short', day: 'numeric' });
        chatLog.appendChild(sep);
    }
}

function addMessage(role, text) {
    dismissEmptyStateHero();
    const ts = Date.now();
    _maybeInsertDaySeparator(ts);
    const div = document.createElement('div');
    div.className = `message ${role}`;
    div.dataset.ts = String(ts);
    // System messages are ours (connection notices, upload progress,
    // error strings), not agent output — render them as plain text so a
    // crafted upstream error message cannot inject markup.
    if (role === 'system') {
        div.textContent = text;
    } else {
        div.innerHTML = renderMarkdown(text);
        if (role === 'agent') decorateCodeBlocks(div);
    }
    chatLog.appendChild(div);
    if (role !== 'system') decorateMessageActions();
    scrollToBottom();
    return div;
}

// Per-message affordance: a small overlay row (timestamp + ⋯ menu) on
// finished user/agent bubbles. The menu itself (copy / quote / memory
// correction) is owned by the workspace module via GhostCore. Guarded by
// the presence of the row, not a dataset flag — streaming rebuilds a
// bubble's innerHTML, which wipes children while element attributes
// survive, so a dataset guard would leave completed replies bare.
function decorateMessageActions() {
    if (!chatLog) return;
    chatLog.querySelectorAll('.message.agent, .message.user').forEach(div => {
        if (div.classList.contains('thinking') || div.classList.contains('streaming')) return;
        _ensureFeedbackRow(div);
        if (div.querySelector(':scope > .msg-actions')) return;
        const row = document.createElement('div');
        row.className = 'msg-actions';
        const time = document.createElement('span');
        time.className = 'msg-time';
        if (div.dataset.ts) {
            time.textContent = new Date(Number(div.dataset.ts))
                .toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        }
        row.appendChild(time);
        const menuBtn = document.createElement('button');
        menuBtn.className = 'msg-menu-btn';
        menuBtn.type = 'button';
        menuBtn.textContent = '⋯';
        menuBtn.title = 'Message actions';
        menuBtn.setAttribute('aria-label', 'Message actions');
        menuBtn.addEventListener('click', (ev) => {
            ev.stopPropagation();
            if (window.GhostCore && window.GhostCore.openMessageMenu) {
                window.GhostCore.openMessageMenu(div, menuBtn);
            }
        });
        row.appendChild(menuBtn);
        div.appendChild(row);
    });
}

// Inline SVG thumbs (Material outlines rendered as currentColor fills) so
// the feedback controls take the theme's color system — emoji glyphs can't.
// Built via DOM APIs to keep the file free of innerHTML='<...>' patterns.
const _THUMB_PATHS = {
    up: 'M1 21h4V9H1v12zM23 10c0-1.1-.9-2-2-2h-6.31l.95-4.57.03-.32c0-.41-.17-.79-.44-1.06L14.17 1 7.58 7.59C7.22 7.95 7 8.45 7 9v10c0 1.1.9 2 2 2h9c.83 0 1.54-.5 1.84-1.22l3.02-7.05c.09-.23.14-.47.14-.73v-2z',
    down: 'M15 3H6c-.83 0-1.54.5-1.84 1.22l-3.02 7.05c-.09.23-.14.47-.14.73v2c0 1.1.9 2 2 2h6.31l-.95 4.57-.03.32c0 .41.17.79.44 1.06L9.83 23l6.59-6.59c.36-.36.58-.86.58-1.41V5c0-1.1-.9-2-2-2zm4 0v12h4V3h-4z'
};

function _thumbSvg(kind) {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('aria-hidden', 'true');
    const p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    p.setAttribute('fill', 'currentColor');
    p.setAttribute('d', _THUMB_PATHS[kind]);
    svg.appendChild(p);
    return svg;
}

// Outcome-label footer for agent bubbles with a known request id: a quiet
// right-aligned row INSIDE the bubble (hover-revealed on desktop, dim-always
// on touch, latched visible once a verdict is chosen). Lives in normal flow
// — the old placement inside the floating .msg-actions overlay crowded the
// timestamp/menu and could overlap the reply text (operator: "really bad").
function _ensureFeedbackRow(div) {
    if (!div.classList.contains('agent') || !div.dataset.reqId) return;
    if (div.querySelector(':scope > .msg-feedback')) return;
    const fbRow = document.createElement('div');
    fbRow.className = 'msg-feedback';
    if (div.dataset.feedback) fbRow.classList.add('fb-decided');
    for (const sig of ['positive', 'negative']) {
        const fb = document.createElement('button');
        fb.className = `msg-fb-btn msg-fb-${sig}`;
        fb.type = 'button';
        fb.title = sig === 'positive'
            ? 'Good answer (labels this turn passed)'
            : 'Bad answer (labels this turn failed)';
        fb.setAttribute('aria-label', fb.title);
        fb.appendChild(_thumbSvg(sig === 'positive' ? 'up' : 'down'));
        if (div.dataset.feedback === sig) fb.classList.add('fb-active');
        fb.addEventListener('click', (ev) => {
            ev.stopPropagation();
            sendFeedback(div, sig);
        });
        fbRow.appendChild(fb);
    }
    div.appendChild(fbRow);
}

// ── Client-only history keys (reqId/feedback — the 👍/👎 label plumbing) ──
// Stripped from EVERY wire payload; preserved across server-history
// adoption. One canonical mapper so a new chatHistory sender can't forget
// the strip (R1 review: the sessions resync compared fat-local vs
// clean-server JSON and read "drifted" on every focus, wiping the labels).
function toWireMessage(m) {
    if (!m || typeof m !== 'object') return m;
    const { reqId, feedback, ...rest } = m;
    return rest;
}

// Carry the client-only label keys forward onto a server-fetched history.
// Server messages are canonical for role/content but never carry reqId, so
// adopting them verbatim silently stripped every label (and persisted the
// loss via saveChatState). Index-aligned and gated on role+content
// equality, so adopting a genuinely different conversation is a no-op.
function mergeClientLabelKeys(serverMsgs) {
    const out = Array.isArray(serverMsgs)
        ? serverMsgs.map(m => (m && typeof m === 'object' ? { ...m } : m))
        : [];
    for (let i = 0; i < out.length && i < chatHistory.length; i++) {
        const loc = chatHistory[i];
        if (!loc || !loc.reqId || !out[i] || out[i].role !== loc.role) continue;
        // Content gate, with ONE relaxation: an aborted reply's local copy
        // is the server's full reply truncated + "*[Aborted]*" — exact
        // equality would drop the label from precisely the reply most
        // likely to be 👎'd (the sessions resync re-adopts ~2s after every
        // abort, deterministically). Scoped tight (R3 review): only at the
        // LOCAL TAIL, where the abort stub always lives, and only with a
        // substantial prefix — a short generic stub ("Sure") prefix-matching
        // an index-shifted different reply would graft the wrong reqId.
        let same = out[i].content === loc.content;
        if (!same && i === chatHistory.length - 1
                && typeof out[i].content === 'string'
                && typeof loc.content === 'string') {
            const abortIdx = loc.content.lastIndexOf('\n\n*[Aborted]*');
            if (abortIdx >= 32) {
                same = out[i].content.startsWith(loc.content.slice(0, abortIdx));
            }
        }
        if (same) {
            out[i].reqId = loc.reqId;
            if (loc.feedback) out[i].feedback = loc.feedback;
        }
    }
    return out;
}

// Stamp a bubble with its agent request id. A REUSED bubble (resume path)
// keeps its dataset across innerHTML rewrites, so a changed id must clear
// the stale feedback latch + row or the new reply renders a verdict nobody
// gave — and the same-signal no-op guard would then make that thumb
// permanently untappable.
function _stampReqId(div, id) {
    if (!div) return;
    // Stale-clear runs even when the NEW id is unknown (R3: a reused
    // bubble whose new turn captured no id — the "No response" path —
    // kept the previous turn's reqId, latch, and thumbs, so a tap there
    // labeled the wrong trajectory).
    if (div.dataset.reqId && div.dataset.reqId !== (id || '')) {
        delete div.dataset.reqId;
        delete div.dataset.feedback;
        // The retry latch and any armed retry belong to the OLD turn — a
        // reused bubble must not fire the old label at the new id, nor
        // start the new turn with its one 404 retry already burned.
        delete div.dataset.fbRetried;
        if (div._fbTimer) { clearTimeout(div._fbTimer); div._fbTimer = null; }
        const old = div.querySelector(':scope > .msg-feedback');
        if (old) old.remove();
    }
    if (id) div.dataset.reqId = id;
}

// POST a human outcome label for an agent reply (Track-1a, 2026-08-13).
// The agent resolves request_id → trajectory and writes its corrections
// sidecar, turning this turn's outcome=unknown into a real label for the
// learning stack. Fire-and-forget with a visual latch on the tapped thumb.
async function sendFeedback(div, signal, retryReqId) {
    const reqId = div.dataset.reqId;
    // Same-signal repeats are a no-op CLIENT-side too (the server also
    // dedupes — first live test: three 👍 in 2s appended three sidecar
    // rows). Switching to the opposite thumb re-labels.
    if (!reqId || div.dataset.feedback === signal) return;
    // An ARMED RETRY fires blind 4s later — it must stand down if the
    // operator labeled in the meantime (either thumb: a 👎 landed after
    // the pending 👍 would otherwise be silently overwritten server-side)
    // or if the bubble was re-stamped with a different turn's id.
    if (retryReqId && (div.dataset.feedback || reqId !== retryReqId)) return;
    // A fresh tap supersedes any armed retry from a previous tap — and
    // returns the un-consumed retry budget (R3: a superseding tap during
    // the flush window burned the latch without a retry ever running, so
    // the tap that would have succeeded at t+4s turned into an error).
    if (!retryReqId && div._fbTimer) {
        clearTimeout(div._fbTimer);
        div._fbTimer = null;
        delete div.dataset.fbRetried;
    }
    const btns = div.querySelectorAll('.msg-fb-btn');
    btns.forEach(b => { b.disabled = true; });
    try {
        const r = await fetch('/api/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ request_id: reqId, signal: signal, source: 'web' })
        });
        if (!r.ok) {
            // 404 = the trajectory can flush moments AFTER the stream ends
            // (the agent records it post-semaphore) — one delayed retry
            // recovers exactly the tap-the-instant-it-lands case.
            //
            // 429/5xx too (review C3): a 503 is the agent RESTARTING, i.e.
            // the exact window a deploy creates, and this path produced
            // one transient chat bubble and nothing durable — while this
            // client is the source of HALF the live labels. Slack already
            // retries these; the web UI silently did not.
            const _retryable = (r.status === 404 || r.status === 429
                                || r.status >= 500);
            if (_retryable && !div.dataset.fbRetried) {
                div.dataset.fbRetried = '1';
                div._fbTimer = setTimeout(() => {
                    div._fbTimer = null;
                    sendFeedback(div, signal, reqId);
                }, 4000);
                return;
            }
            const err = await r.json().catch(() => ({}));
            addMessage('system', `Feedback not recorded: ${err.error || err.detail || ('HTTP ' + r.status)}`);
            return;
        }
        delete div.dataset.fbRetried;
        // Latch on the LIVE bubble: a drift re-render can detach `div`
        // between the tap (or an armed retry) and this response — latching
        // the detached node showed no verdict and invited a duplicate tap
        // (R3 review). The history loop below is the durable half either way.
        let target = div;
        if (!target.isConnected) {
            // CSS.escape: reqIds are server hex today, but X-Request-ID is
            // client-controlled on other surfaces — an unescaped quote
            // would throw here AFTER the POST succeeded and misreport a
            // recorded label as a network error.
            target = chatLog.querySelector(
                `.message.agent[data-req-id="${CSS.escape(reqId)}"]`) || div;
        }
        target.dataset.feedback = signal;
        const fbRow = target.querySelector(':scope > .msg-feedback');
        if (fbRow) fbRow.classList.add('fb-decided');
        target.querySelectorAll('.msg-fb-btn').forEach(b => {
            b.classList.toggle('fb-active',
                b.classList.contains(`msg-fb-${signal}`));
        });
        // Persist the latch so a reload re-renders the chosen thumb.
        for (let i = chatHistory.length - 1; i >= 0; i--) {
            if (chatHistory[i].reqId === reqId) {
                chatHistory[i].feedback = signal;
                break;
            }
        }
        if (typeof saveChatState === 'function') saveChatState();
    } catch (e) {
        console.warn('feedback failed:', e);
        addMessage('system', 'Feedback not recorded: network error — tap again.');
    } finally {
        btns.forEach(b => { b.disabled = false; });
    }
}

function scrollToBottom() {
    requestAnimationFrame(() => { chatLog.scrollTo({ top: chatLog.scrollHeight, behavior: 'smooth' }); });
}

// During streaming we get 50+ chunks/sec. Smooth-scrolling each one
// stutters and animates the scrollbar through every partial frame.
// Use an rAF-throttled `auto` scroll: feels instant, burns one frame.
let _streamScrollScheduled = false;
function scrollToBottomDuringStream() {
    if (_streamScrollScheduled) return;
    _streamScrollScheduled = true;
    requestAnimationFrame(() => {
        _streamScrollScheduled = false;
        chatLog.scrollTo({ top: chatLog.scrollHeight, behavior: 'auto' });
    });
}

// Re-render the in-flight agent message from the accumulated content.
// Re-parsing the full markdown + re-sanitizing on every network read is
// O(n²) over a long reply; coalescing renders to one-per-animation-frame
// (scheduleStreamRender) keeps it smooth. _renderStreamingContent does
// the actual work and is also called once synchronously after the stream
// ends to flush the final tokens.
function _renderStreamingContent() {
    if (!currentAgentMessageDiv || currentAccumulatedContent === "") return;
    const displayContent = _stripInternalTags(currentAccumulatedContent);
    currentAgentMessageDiv.innerHTML = renderMarkdown(displayContent);
    // Streaming-cursor glyph; removed in sendMessage's finally.
    currentAgentMessageDiv.classList.add('streaming');
    const isAtBottom = Math.abs(chatLog.scrollHeight - chatLog.scrollTop - chatLog.clientHeight) <= 50;
    if (isAtBottom) scrollToBottomDuringStream();
    decorateCodeBlocks(currentAgentMessageDiv);
}
let _streamRenderRafId = null;
function scheduleStreamRender() {
    if (_streamRenderRafId !== null) return;
    _streamRenderRafId = requestAnimationFrame(() => {
        _streamRenderRafId = null;
        _renderStreamingContent();
    });
}
// Cancel a pending throttled render. Without this, a frame scheduled
// during streaming can fire AFTER sendMessage's finally removed the
// `streaming` class and re-add it — leaving the blinking cursor glyph
// stuck on a completed reply.
function _cancelScheduledStreamRender() {
    if (_streamRenderRafId !== null) {
        cancelAnimationFrame(_streamRenderRafId);
        _streamRenderRafId = null;
    }
}

// Is-reading dim: when the user has scrolled up to review earlier
// history, fade the background sphere back so the text takes focus.
// When they return to the bottom, the sphere returns. Debounced via
// a simple "near-bottom" threshold so it doesn't flicker with
// rubber-banding on iOS Safari.
(function installReadingDimListener() {
    if (!chatLog) return;
    let scheduled = false;
    const update = () => {
        scheduled = false;
        const dist = chatLog.scrollHeight - chatLog.scrollTop - chatLog.clientHeight;
        // 150px: bigger than iOS rubber-band overscroll, smaller than a
        // single message — reliably detects intentional scrollback.
        const isReading = dist > 150;
        document.body.classList.toggle('is-reading', isReading);
    };
    chatLog.addEventListener('scroll', () => {
        if (scheduled) return;
        scheduled = true;
        requestAnimationFrame(update);
    }, { passive: true });
})();

function saveChatState() {
    safeStorage.set('ghost_chat_history', JSON.stringify(chatHistory));
}

// Render a full message history into the (cleared) chat log. Shared by
// the localStorage restore, workspace load, and the sessions rail
// (server-side conversations). body.restoring suppresses the entrance
// animations so a re-render doesn't replay every slide-in.
function renderHistoryToLog(history) {
    chatLog.innerHTML = '';
    document.body.classList.add('restoring');
    _withHistoryRestore(() => {
        for (const msg of history) {
            // Only conversational roles get bubbles — a stored session can
            // carry tool/function messages (other clients), which would
            // otherwise masquerade as USER bubbles.
            if (!['user', 'assistant', 'system'].includes(msg.role)) continue;
            const roleClass = msg.role === 'assistant' ? 'agent' : (msg.role === 'system' ? 'system' : 'user');
            let displayContent = msg.content;

            if (Array.isArray(displayContent)) {
                const texts = displayContent.filter(c => c.type === "text").map(c => c.text);
                displayContent = texts.join(" ") + " \n*[Image Attached]*";
            }
            if (typeof displayContent === 'string') {
                let cleanContent = _stripInternalTags(displayContent);
                if (cleanContent) {
                    const div = document.createElement('div');
                    div.className = `message ${roleClass}`;
                    if (roleClass === 'system') {
                        div.textContent = cleanContent;
                    } else {
                        div.innerHTML = renderMarkdown(cleanContent);
                        if (roleClass === 'agent') decorateCodeBlocks(div);
                    }
                    // Restore the feedback handle + chosen thumb so the
                    // 👍/👎 buttons survive a reload (local history only —
                    // server-side sessions don't carry reqId).
                    if (roleClass === 'agent' && msg.reqId) {
                        div.dataset.reqId = msg.reqId;
                        if (msg.feedback) div.dataset.feedback = msg.feedback;
                    }
                    chatLog.appendChild(div);
                }
            }
        }
        decorateMessageActions();
        scrollToBottom();
        if (typeof attachRenderButtons === 'function') attachRenderButtons();
    });
    setTimeout(() => document.body.classList.remove('restoring'), 60);
}

function loadChatState() {
    const saved = safeStorage.get('ghost_chat_history');
    if (saved) {
        try {
            // Shape-validate: JSON.parse happily yields a string, a number
            // or an object, and `chatHistory.push` then throws BEFORE
            // sendMessage's try block — the composer dies with a
            // console-only error and no visible cause (R2 lens B). The
            // bridge setter has always guarded this; the two direct
            // assignments did not.
            const _parsed = JSON.parse(saved);
            chatHistory = Array.isArray(_parsed) ? _parsed : [];
            renderHistoryToLog(chatHistory);
        } catch (e) {
            console.error("Failed to load chat history", e);
        }
    }
}

// ═══════════════════════════════════════════════════════════════
//  In-flight turn survival (2026-08-01) — the Claude-app model.
//
//  The proxy buffers every chat stream server-side (task lives while
//  running + 10 min after completion), and the AGENT persists the
//  finished reply into the durable session store. What used to be lost
//  when iOS killed the page mid-request was only the HANDLE — the task
//  id lived in a module `let`. Persist it, and on return:
//    1. proxy replay buffer still there  → reattach the live stream
//    2. buffer gone, turn still running  → placeholder + poll sessions
//    3. buffer gone, turn finished       → adopt the session's history
//  (3) goes through wholesale adoption (server canonical), never an
//  append — appending a diverged fat replay is the merge_history
//  doubling case.
// ═══════════════════════════════════════════════════════════════
const INFLIGHT_KEY = 'ghost_inflight_turn';
let reconcilePollTimer = null;
let resumeLatch = false;        // synchronous re-entry guard (see below)
// Bounded attempts for the NETWORK-error resume path (the fourth resume
// trigger, which used to bypass the latch and could retry forever silently).
let _netResumeTries = 0;
let inflightHeartbeat = null;   // owner-tab liveness while streaming

// Tab identity: sessionStorage survives same-tab reloads but NOT an iOS
// PWA kill — which is exactly right: the reopened app gets a fresh id
// and may adopt the orphaned handle, while a SECOND live tab (whose
// heartbeat is fresh) must keep its hands off another tab's turn.
const TAB_ID = (() => {
    try {
        let id = sessionStorage.getItem('ghost_tab_id');
        if (!id) {
            id = (crypto.randomUUID ? crypto.randomUUID()
                : `tab-${Date.now()}-${Math.random().toString(36).slice(2)}`);
            sessionStorage.setItem('ghost_tab_id', id);
        }
        return id;
    } catch (e) { return `tab-${Math.random().toString(36).slice(2)}`; }
})();

// ⚠ ONE KEY PER TAB. A single shared `ghost_inflight_turn` meant every
// tab wrote over every other tab's handle and `clearInflightHandle()`
// deleted it outright — on turn completion, abort, /clear, the resume
// mismatch drop, the reconcile give-up. Tab 1's heartbeat then correctly
// stood down, so nothing noticed; if tab 1 was killed mid-turn its reply
// was unrecoverable, which is the exact recovery this subsystem exists for
// (R3 lens B). Namespacing is the fix — a tabId GUARD would only move the
// loss into saveInflightHandle, where tab 2's own turn becomes the casualty.
function _inflightKey(tabId) {
    return `${INFLIGHT_KEY}:${tabId || TAB_ID}`;
}

function _inflightKeys() {
    const out = [];
    try {
        for (let i = 0; i < localStorage.length; i++) {
            const k = localStorage.key(i);
            if (k && k.startsWith(INFLIGHT_KEY + ':')) out.push(k);
        }
    } catch (e) { /* private mode — no enumeration */ }
    return out;
}

function saveInflightHandle() {
    if (!currentTaskId) return;
    safeStorage.set(_inflightKey(), JSON.stringify({
        taskId: currentTaskId,
        sessionId: window.__ghostSessionId || safeStorage.get('ghost_session_id') || null,
        tabId: TAB_ID,
        ts: Date.now(),
        beat: Date.now(),
    }));
    // Heartbeat while the owning tab streams: lets other tabs tell a
    // LIVE owner from a killed one.
    if (inflightHeartbeat) clearInterval(inflightHeartbeat);
    inflightHeartbeat = setInterval(() => {
        try {
            const raw = safeStorage.get(_inflightKey());
            if (!raw) return;
            const h = JSON.parse(raw);
            if (h.tabId !== TAB_ID) return;  // paranoia: not ours to beat
            h.beat = Date.now();
            safeStorage.set(_inflightKey(), JSON.stringify(h));
        } catch (e) { /* storage gone — nothing to keep alive */ }
    }, 5000);
}

// `handle` names WHOSE key to drop. Adopting an orphan and then calling the
// bare form would clear our own (empty) slot and leave the orphan on disk to
// be re-adopted on every visibility change — a loop, not a cleanup.
function clearInflightHandle(handle) {
    safeStorage.remove(_inflightKey(handle && handle.tabId));
    safeStorage.remove(INFLIGHT_KEY);            // legacy single-key handle
    if (reconcilePollTimer) { clearTimeout(reconcilePollTimer); reconcilePollTimer = null; }
    if (inflightHeartbeat) { clearInterval(inflightHeartbeat); inflightHeartbeat = null; }
}

function readInflightHandle() {
    const parse = (raw) => {
        try {
            const h = JSON.parse(raw);
            return (h && typeof h.taskId === 'string' && h.taskId) ? h : null;
        } catch (e) { return null; }
    };
    // Ours first — a tab always owns its own turn.
    const mine = safeStorage.get(_inflightKey());
    const ours = mine ? parse(mine) : null;
    if (ours) return ours;
    // Otherwise adopt an ORPHAN: another tab's handle whose heartbeat has
    // gone stale (that tab was killed). A live tab's handle is left alone.
    const STALE_MS = 15000;
    const GC_MS = 10 * 60 * 1000;   // the proxy's replay buffer TTL
    let best = null;
    for (const k of _inflightKeys()) {
        const h = parse(safeStorage.get(k) || '');
        if (!h) { safeStorage.remove(k); continue; }
        const age = Date.now() - (h.beat || h.ts || 0);
        if (age > GC_MS) { safeStorage.remove(k); continue; }
        if (age > STALE_MS && (!best || (h.ts || 0) > (best.ts || 0))) best = h;
    }
    // Legacy single-key handle written before this tab upgraded.
    if (!best) {
        const legacy = safeStorage.get(INFLIGHT_KEY);
        if (legacy) best = parse(legacy);
    }
    return best;
}

// Recreate the thinking bubble + ticker for a resume that outlived the
// page (after a reload nothing of the original turn's DOM exists). No-op
// when the original bubble is still attached (same-page Safari resume).
function ensureAgentBubbleForResume() {
    if (currentAgentMessageDiv && currentAgentMessageDiv.isConnected) return;
    dismissEmptyStateHero();
    currentAgentMessageDiv = document.createElement('div');
    currentAgentMessageDiv.className = 'message agent thinking';
    currentAgentMessageDiv.dataset.ts = String(Date.now());
    const ind = document.createElement('span');
    ind.className = 'typing-indicator';
    ind.setAttribute('aria-label', 'Thinking');
    ind.appendChild(document.createElement('span'));
    ind.appendChild(document.createElement('span'));
    ind.appendChild(document.createElement('span'));
    currentAgentMessageDiv.appendChild(ind);
    chatLog.appendChild(currentAgentMessageDiv);
    startTurnTicker(currentAgentMessageDiv);
    scrollToBottom();
}

// True while `sid` is still the conversation this tab is showing. The
// reconcile poll can outlive a rail switch/new-chat by many seconds —
// adopting session A's history after the user moved to session B would
// bind A's transcript to B and replay it into B's durable store on the
// next turn (R6 review; same corruption class as the sessions resync).
function _sessionStillCurrent(sid) {
    const cur = window.__ghostSessionId;
    // undefined = the sessions module hasn't booted/bound yet → allow.
    // null = EXPLICITLY unbound (/clear with sessions unreachable,
    // setCurrent(null)) → adopting anything would resurrect the cleared
    // conversation (R7 review). Anything else: exact match only.
    if (cur === undefined) return true;
    return cur === sid;
}

// One announcement per recovery, and a poll chain that cannot silently die.
let _reconcileAnnounced = false;
let _reconcileTries = 0;
const _RECONCILE_MAX_TRIES = 120;   // ~10 minutes at 5s

function _scheduleReconcilePoll(sid, handle) {
    if (reconcilePollTimer) return;
    if (_reconcileTries++ >= _RECONCILE_MAX_TRIES) {
        addMessage('system', 'Gave up waiting for this turn. If it finished, '
            + 'switching sessions and back will show the reply.');
        stopTurnTicker();
        clearInflightHandle(handle);
        _reconcileAnnounced = false;
        _reconcileTries = 0;
        return;
    }
    reconcilePollTimer = setTimeout(() => {
        reconcilePollTimer = null;
        reconcileFromSession(sid, handle);
    }, 5000);
}

async function reconcileFromSession(sid, handle) {
    // `handle` is the inflight record this recovery is FOR — which may belong
    // to another (dead) tab we adopted. Terminal cleanups must name it: a
    // bare clear drops our own empty slot and leaves the orphan on disk, to
    // be re-adopted on every visibility change (R4 lens B).
    if (!sid) { clearInflightHandle(handle); return; }
    // A LIVE turn owns the log, and so does a NEWER session identity.
    // Bail at entry AND after every await — adoption calls
    // renderHistoryToLog, which would detach the bubble a new turn is
    // streaming into and drop its user message.
    if (isProcessingRequest || !_sessionStillCurrent(sid)) return;
    let stillRunning = false;
    try {
        const tr = await fetch('/api/turns');
        if (isProcessingRequest || !_sessionStillCurrent(sid)) return;
        if (tr.ok) {
            const td = await tr.json();
            stillRunning = (td.turns || []).some(
                (t) => t.session_id === sid && (t.running || t.queued));
        }
    } catch (e) {
        // A hiccup on /api/turns used to return here with NO poll
        // rescheduled, so the recovery chain died permanently — a thinking
        // bubble and a running clock, forever, on a visible tab (R2 lens B).
        _scheduleReconcilePoll(sid, handle);
        return;
    }
    if (stillRunning) {
        // The proxy's replay buffer is gone (restart) but the agent is
        // still computing. The reply will land in the session store when
        // it finishes — keep a placeholder and poll.
        ensureAgentBubbleForResume();
        // Announce ONCE per recovery, not once per poll. The old guard
        // (`if (!reconcilePollTimer)`) conflated "first time" with "no poll
        // pending", and the timeout body nulls the timer BEFORE recursing —
        // so a 3-minute turn printed ~36 identical "Ghost is still working"
        // bubbles (measured under node: 6 in 30s).
        if (!_reconcileAnnounced) {
            _reconcileAnnounced = true;
            addMessage('system', 'Ghost is still working on your request…');
        }
        _scheduleReconcilePoll(sid, handle);
        return;
    }
    _reconcileAnnounced = false;
    try {
        const res = await fetch(`/api/sessions/${encodeURIComponent(sid)}`);
        if (isProcessingRequest || !_sessionStillCurrent(sid)) return;
        if (res.ok) {
            const data = await res.json();
            if (isProcessingRequest || !_sessionStillCurrent(sid)) return;
            const msgs = Array.isArray(data.messages) ? data.messages : [];
            if (msgs.length > chatHistory.length) {
                // Server is canonical: ADOPT with the FULL message shape
                // ({...m} — re-shaping to {role, content} dropped
                // tool_calls/name and guaranteed one spurious drift
                // re-render on the next resync), carrying the client-only
                // label keys forward (they exist ONLY here; verbatim
                // adoption stripped every 👍/👎 and persisted the loss).
                chatHistory = mergeClientLabelKeys(msgs);
                renderHistoryToLog(chatHistory);
                saveChatState();
                addMessage('system', 'Recovered the reply that finished while you were away.');
                _reconcileAnnounced = false;
                _reconcileTries = 0;
            } else {
                addMessage('system', 'The previous request could not be recovered.');
                // Reset the budget on EVERY terminal exit, not only the
                // successful one: a later recovery in the same page
                // otherwise started with whatever was left and gave up
                // early — or immediately (R3 lens B).
                _reconcileAnnounced = false;
                _reconcileTries = 0;
            }
        } else if (res.status === 404) {
            addMessage('system', 'The previous request could not be recovered.');
            _reconcileAnnounced = false;
            _reconcileTries = 0;
        } else {
            return; // sessions store hiccup — retry on next visibility
        }
    } catch (e) { return; }
    // A placeholder bubble that never streamed has nothing to show.
    if (currentAgentMessageDiv && currentAgentMessageDiv.classList.contains('thinking')) {
        stopTurnTicker();
        currentAgentMessageDiv.remove();
        currentAgentMessageDiv = null;
    }
    clearInflightHandle(handle);
}

async function resumeOrReconcileInflightTurn() {
    // Synchronous latch: boot + visibilitychange + pageshow can all fire
    // in one tick (bfcache restore), and the isProcessingRequest guard
    // alone has an async gap before sendMessage(true) claims the turn —
    // two concurrent resumes each replayed the stream from chunk 0 into
    // one bubble and pushed TWO assistant messages (review CRITICAL).
    if (resumeLatch || isProcessingRequest) return;
    resumeLatch = true;
    try {
        const h = readInflightHandle();
        if (!h) return;
        // Another LIVE tab owns this turn: fresh heartbeat → hands off.
        // A dead owner (killed PWA) stops beating and may be adopted.
        if (h.tabId && h.tabId !== TAB_ID
                && Date.now() - (h.beat || h.ts || 0) < 15000) {
            return;
        }
        // The handle belongs to a conversation this client no longer
        // shows (/clear minted a new session while the turn hung):
        // resurrecting the old transcript into the new session would
        // duplicate it server-side. Drop it.
        const currentSid = window.__ghostSessionId
            || safeStorage.get('ghost_session_id') || null;
        if (h.sessionId && currentSid && h.sessionId !== currentSid) {
            clearInflightHandle(h);
            return;
        }
        let state = null;
        try {
            const r = await fetch(`/api/chat/task/${encodeURIComponent(h.taskId)}/state`);
            if (r.ok) state = await r.json();
        } catch (e) { /* proxy down — nothing to do until it returns */ return; }
        if (isProcessingRequest) return;
        if (state && state.cancelled) {
            // The user stopped this turn; reattaching would replay the
            // stream they ended (R2 lens A).
            clearInflightHandle(h);
            return;
        }
        if (state && state.exists) {
            // Replay buffer alive: reattach the stream exactly like the
            // same-page Safari resume, with a fresh bubble. sendMessage
            // sets isProcessingRequest synchronously before its first
            // await, so the latch can release right after this call.
            //
            // ⚠ RE-HOME THE HANDLE FIRST. `clearInflightHandle`'s own
            // docstring says adopting an orphan and then clearing the bare
            // form "would leave the orphan on disk to be re-adopted on every
            // visibility change — a loop, not a cleanup" — and this branch,
            // the ADOPT branch, did exactly that: the resume path never
            // calls saveInflightHandle(), so all four terminal cleanups
            // dropped OUR empty slot while the dead tab's key survived. Every
            // visibility return re-adopted it and replayed the whole buffer:
            // reply re-rendered, TTS re-spoken, a duplicate assistant pushed
            // into chatHistory, for up to the proxy's 10-minute TTL (R5 lens
            // B, reproduced). Dropping the orphan and re-saving under THIS
            // tab makes the heartbeat and all four terminal paths work.
            clearInflightHandle(h);
            currentTaskId = h.taskId;
            currentChunkIndex = 0;
            currentAccumulatedContent = "";
            saveInflightHandle();
            ensureAgentBubbleForResume();
            sendMessage(true);
            return;
        }
        await reconcileFromSession(h.sessionId, h);
    } finally {
        resumeLatch = false;
    }
}

// Auto-expand textarea height organically. When the field empties,
// CLEAR the inline height so the `rows="1"` attribute regains control
// — `height: auto` on an already-styled textarea computes from content
// and doesn't reliably snap back to the one-row default.
chatInput.addEventListener('input', function () {
    if (this.value === '') {
        this.style.height = '';
        return;
    }
    this.style.height = 'auto';
    this.style.height = this.scrollHeight + 'px';
});

// ═══════════════════════════════════════════════════════════════
//  Turn status HUD (2026-07-29, v2 — operator direction)
//
//  The agent's turns routinely run 30-60s+ of tools + thinking before
//  the first content token. Placement iterated twice with the
//  operator: v1 breadcrumb inside the bubble (rejected), v2 next to
//  the center-stage activity icon (rejected — the 2.2rem icon dwarfed
//  the caption), v3 FINAL: a caption line re-parented directly UNDER
//  the waiting reply bubble, format `timer : description - icon` with
//  a text-sized icon of its own. (The big center-stage #activity-icon
//  was removed entirely right after — same operator session.)
//
//  Corridor filtering: background work (self-play, dreams) streams
//  over the same socket with its own request ids. On send we adopt the
//  id of the FIRST "request started" corridor that opens (ours — the
//  POST fires it within ~a second; a corridor already open when we
//  sent can never be adopted) and only lines carrying that id update
//  the description; until adoption only the clock runs.
// ═══════════════════════════════════════════════════════════════

// Titles that narrate plumbing, not progress — never shown.
const TICKER_NOISE = new Set([
    'prefill cache', 'memory bus', 'llm request',
    'constraint check', 'metacog conf', 'turn outcome', 'agent parser',
]);
// Friendly phrasings for the most common step titles; anything not
// listed falls back to the raw title (already short and readable).
const TICKER_VERBS = {
    'worker compute': 'delegating to the worker node',
    'sandbox tree': 'scanning the workspace',
    'sandbox exec': 'running a command',
    'execution task': 'executing code',
    'file read': 'reading a file',
    'file write': 'writing a file',
    'web search': 'searching the web',
    'web read': 'reading a page',
    'browser': 'driving the browser',
    'verifier': 'verifying the answer',
    'memory search': 'recalling memories',
    'memory save': 'saving a memory',
    'graph updated': 'updating the knowledge graph',
    'belief revision': 'revisiting a belief',
    'hydrated context': 'gathering context',
    'system weather': 'checking the weather',
    'vision': 'looking at an image',
    'delegation': 'delegating a subtask',
};
const turnStatusEl = document.getElementById('turn-status');
const turnStatusDescEl = document.getElementById('turn-status-desc');
const turnStatusClockEl = document.getElementById('turn-status-clock');
const turnStatusIconEl = document.getElementById('turn-status-icon');
let tickerTimer = null, tickerStart = 0, tickerReqId = null;

function startTurnTicker(afterEl) {
    stopTurnTicker();
    if (!turnStatusEl) return;
    tickerStart = Date.now();
    tickerReqId = null;
    turnStatusDescEl.textContent = 'starting…';
    turnStatusClockEl.textContent = '0:00';
    turnStatusIconEl.textContent = '⏳';
    // Re-parent under the waiting reply bubble (the element survives
    // detachment by /clear or history repaints — appendChild re-adopts
    // the same node wherever the current bubble lives).
    if (afterEl && afterEl.insertAdjacentElement) {
        afterEl.insertAdjacentElement('afterend', turnStatusEl);
    }
    turnStatusEl.classList.remove('hidden');
    tickerTimer = setInterval(() => {
        const s = Math.floor((Date.now() - tickerStart) / 1000);
        turnStatusClockEl.textContent =
            `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
    }, 1000);
}

function stopTurnTicker() {
    clearInterval(tickerTimer);
    tickerTimer = null;
    tickerReqId = null;
    if (turnStatusEl) turnStatusEl.classList.add('hidden');
}

function setTurnStatusDesc(text, icon) {
    if (!tickerTimer) return;
    if (turnStatusDescEl) turnStatusDescEl.textContent = text;
    if (icon && turnStatusIconEl) turnStatusIconEl.textContent = icon;
}

function noteTickerLine(raw) {
    if (!tickerTimer) return;                       // no turn in flight
    const clean = cleanLogLine(raw);
    // Corridor adoption: the first corridor that OPENS after our send.
    if (tickerReqId === null) {
        if (/request started/.test(clean)) {
            const m = clean.match(/^\S+\s+(\S+)\s/);
            if (m) tickerReqId = m[1];
        }
        return;
    }
    // Body lines: │  <id>  <icon>  <+delta>  <title>  <content…>
    const parts = clean.split(/\s{2,}/).map(p => p.trim()).filter(Boolean);
    if (parts.length < 3 || parts[1] !== tickerReqId) return;
    const icon = extractIcon(clean);
    if (!icon) return;
    const cls = _iconClass(icon);
    if (cls === 'think') {
        // Long reasoning stretches between tools: say so instead of
        // leaving the last tool's caption to go stale.
        setTurnStatusDesc('thinking…', '💭');
        return;
    }
    if (cls !== 'tool' && cls !== 'memory' && cls !== 'plan' && cls !== 'accent') return;
    // Feedback receipts are terminal bookkeeping, not turn activity — an
    // in-flight turn's status line must not read out a 👍 someone left on
    // yesterday's reply.
    if (_NON_WORKING_ICONS.has(icon)) return;
    const iconIdx = parts.findIndex(p => p.includes(icon));
    if (iconIdx < 0) return;
    let ti = iconIdx + 1;
    if (/^\+[\d.\s]*m?s$/.test(parts[ti] || '')) ti++;
    const title = (parts[ti] || '').toLowerCase().slice(0, 22).trim();
    if (!title || TICKER_NOISE.has(title)) return;
    let desc = TICKER_VERBS[title] || title;
    // Specificity when the line carries it ("reading a file · foo.py").
    const detail = parts.slice(ti + 1).join(' ').trim();
    if (detail) desc += ` · ${detail.slice(0, 30)}${detail.length > 30 ? '…' : ''}`;
    setTurnStatusDesc(desc, icon);
}

// Turn a failed Response into an Error that still says something.
//
// Two ways the previous one-liner destroyed the diagnosis (review R1 M14/M15):
//   * `await response.json()` on a NON-JSON error body throws, and the
//     SyntaxError replaced the status entirely — a front-door 502 with an
//     HTML body read as "Unexpected token <", which names no subsystem.
//   * only `.error` was read, but FastAPI's own HTTPException renders
//     `{"detail": …}` — so a 401 from the key check and a 400 "Invalid
//     session id" both arrived as a bare "HTTP 4xx".
// `detail` can also be an array (validation errors), which stringifies to
// "[object Object]"; carry the status so callers can classify regardless.
async function _httpError(response, prefix) {
    let body = null;
    try { body = await response.json(); } catch (e) { body = null; }
    let msg = '';
    if (body && typeof body === 'object') {
        const d = body.error || body.detail;
        msg = typeof d === 'string' ? d : (d ? JSON.stringify(d) : '');
    }
    if (!msg) msg = `HTTP ${response.status}`;
    const err = new Error(prefix ? `${prefix}: ${msg}` : msg);
    err.status = response.status;
    return err;
}

// Stop means stop. `/api/chat/cancel/<task>` is a PROXY-side teardown (it
// kills the buffered relay task); the agent's own turn only stops via
// `/api/turn/cancel`. Without this, "Stopped by user." was a lie: the turn
// ran to completion, held the global turn lock, and its reply landed in the
// durable session with no client showing it (review R1 M8).
// ⚠ RESOLVE THE TARGET FIRST. A bare `{}` body means "cancel the turn
// holding the semaphore" (core/turns.py::current) — NOT "cancel mine". And
// `currentReqId` is null until the first SSE frame carrying an id, which for
// a tool-heavy turn is the whole 30-60s thinking phase, i.e. exactly when
// Stop gets pressed. So the R1 version of this fix could kill a background
// dream/self-play turn while the user's own queued turn ran on to completion
// and the tab reported "Stopped by user." — the same dishonesty one layer
// deeper (R2 lens B). `/api/turns` carries `session_id` per turn, so the
// client can identify its own turn; if it cannot, it must NOT guess.
async function _resolveOwnTurnId(sentText) {
    const sid = window.__ghostSessionId
        || safeStorage.get('ghost_session_id') || null;
    let data;
    try {
        // BOUNDED. Stop exists for the case where the agent is wedged, and
        // /api/turns queues behind the same lock — an unbounded fetch here
        // meant the "could not identify your turn" warning never printed at
        // all, and every Stop leaked a pending request (R3 lens B).
        const res = await fetch('/api/turns', {
            signal: AbortSignal.timeout(4000),
        });
        if (!res.ok) return null;
        data = await res.json();
    } catch (e) {
        return null;
    }
    const turns = (data && data.turns) || [];
    const pick = (list) => list.length
        ? ((list.find(t => t.running) || list[0]).request_id || null) : null;
    if (sid) {
        const mine = turns.filter(t => t && t.session_id === sid);
        if (mine.length) return pick(mine);
    }
    // SESSIONS DISABLED (or not yet bound): there is no session id to match
    // on — `routes.py` reports `enabled:false` whenever the store is off, and
    // sessions.js then never publishes an id at all. Identify by the turn
    // PREVIEW instead, which the registry derives from the user message
    // (turns.py trims it to 120 chars). Without this the composer's Stop
    // could cancel nothing at all in a supported deployment mode — a
    // regression the tests missed because they all stub a session id.
    const probe = String(sentText || '').split(/\s+/).join(' ').slice(0, 40);
    if (probe) {
        const byText = turns.filter(
            t => t && typeof t.preview === 'string' && t.preview.startsWith(probe));
        if (byText.length === 1) return byText[0].request_id || null;
    }
    return null;
}

async function _cancelAgentTurn(reqId, sentText, hard) {
    const target = reqId || await _resolveOwnTurnId(sentText);
    if (!target) {
        // No bare-body fallback: cancelling "whatever is running" can hit an
        // unrelated background turn.
        addMessage('system', '⚠ Stopped listening, but this tab could not '
            + 'identify its turn on the agent, so nothing was cancelled — it '
            + 'may still be running. The turn panel (status strip) lists '
            + 'in-flight turns and can cancel by id.');
        return false;
    }
    const body = hard ? { request_id: target, hard: true }
                      : { request_id: target };
    return fetch('/api/turn/cancel', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    }).then(async (res) => {
        const data = await res.json().catch(() => ({}));
        if (res.ok && data.cancelled) return true;
        // 404 = nothing to cancel, which is the benign case (the turn had
        // already finished). Anything else means the agent IS still working:
        // actionable, so say it.
        if (res.status !== 404) {
            addMessage('system', '⚠ Stop reached this tab but NOT the agent — '
                + 'the turn is still running (' + (data.detail || data.error
                || ('HTTP ' + res.status)) + '). Its reply will appear in this '
                + 'session when it finishes; the turn panel can hard-cancel.');
        }
        return false;
    }).catch((e) => {
        addMessage('system', '⚠ Stop reached this tab but the cancel call '
            + 'failed (' + (e && e.message || e) + ') — the agent may still '
            + 'be running this turn.');
        return false;
    });
}

async function sendMessage(isResume = false) {
    const resuming = isResume === true;
    const text = chatInput.value.trim();
    if (!resuming && (!text || isProcessingRequest)) return;
    // A resume must not stack on a turn either: the legacy same-page
    // auto-resume (resumeOnVisible) and the persisted-handle path can
    // both fire on the same visibilitychange — the second replay wiped
    // the first's accumulator and double-pushed the assistant message.
    if (resuming && isProcessingRequest) return;

    if (text === '/clear' && !resuming) {
        chatInput.value = '';
        chatInput.style.height = '';
        chatLog.innerHTML = '';
        chatHistory = [];
        safeStorage.remove('ghost_chat_history');
        // A surviving in-flight handle points at the OLD conversation —
        // left alone, the next reconcile would resurrect the cleared
        // transcript into the freshly minted session.
        clearInflightHandle();
        // Close the visualizer BEFORE revoking: it may be displaying one
        // of these blobs right now (blank preview + dead Download button
        // otherwise).
        if (typeof closeRenderWindow === 'function') closeRenderWindow();
        // Flush the authed-image blob cache too — the thumbnails those
        // URLs pointed at are gone from the DOM, so holding on to the
        // blobs is pure leak.
        for (const url of _authedBlobCache.values()) {
            try { URL.revokeObjectURL(url); } catch (e) { /* ignore */ }
        }
        _authedBlobCache.clear();
        if (typeof updateWorkspaceBtnState === 'function') updateWorkspaceBtnState();
        // Let the sessions rail mint a fresh session id — /clear means
        // "new conversation", not "amnesia inside the same session".
        try {
            window.GhostCore?.events?.dispatchEvent(new CustomEvent('conversation-cleared'));
        } catch (e) { /* workspace modules not loaded */ }
        const msg = addMessage('system', 'Context cleared');
        // Hero AFTER the confirmation removes itself (addMessage dismisses
        // any hero, so rendering it first left a blank log 2s later) — and
        // only if the log is still empty by then.
        setTimeout(() => {
            msg.remove();
            if (!chatLog.querySelector('.message')) renderEmptyStateHero();
        }, 2000);
        return;
    }

    if (!resuming) {
        chatInput.value = '';
        chatInput.style.height = ''; // Clear inline height → rows="1" takes over
        addMessage('user', text);
        
        currentTaskId = null;
        currentChunkIndex = 0;
        currentAccumulatedContent = "";
        currentReqId = null;

        chatHistory.push({ role: "user", content: text });
        // Remembered for Stop: with sessions disabled there is no session id
        // to match a turn on, and the turn registry's `preview` is derived
        // from exactly this text (R3 lens B).
        _lastSentUserText = text;
        if (typeof saveChatState === 'function') saveChatState();
        if (typeof updateWorkspaceBtnState === 'function') updateWorkspaceBtnState();
        
        // Render a CSS-driven typing indicator (three staggered dots) so
        // the waiting state reads as deliberate UX rather than text poke.
        // The setInterval-based "Thinking.".repeat() version flickered on
        // fast responses because the first real chunk could race a tick
        // that overwrote the fresh content with "Thinking..".
        dismissEmptyStateHero();
        currentAgentMessageDiv = document.createElement('div');
        currentAgentMessageDiv.className = 'message agent thinking';
        currentAgentMessageDiv.dataset.ts = String(Date.now());
        // Build indicator via DOM APIs (no user-supplied content, but
        // keeps the file free of innerHTML = '<...>' patterns).
        const _ind = document.createElement('span');
        _ind.className = 'typing-indicator';
        _ind.setAttribute('aria-label', 'Thinking');
        _ind.appendChild(document.createElement('span'));
        _ind.appendChild(document.createElement('span'));
        _ind.appendChild(document.createElement('span'));
        currentAgentMessageDiv.appendChild(_ind);
        chatLog.appendChild(currentAgentMessageDiv);
        // Turn status line under the waiting bubble (`timer :
        // description - icon`) — see the Turn status block above.
        startTurnTicker(currentAgentMessageDiv);
        scrollToBottom();
        currentThinkingInterval = null;
    } else {
        // Cross-reload resumes have no surviving bubble; same-page resumes
        // keep theirs (this is a no-op then). Reconnects are silent — the
        // stream resuming is the only signal the user needs.
        ensureAgentBubbleForResume();
        setTimeout(scrollToBottom, 100);
    }

    // Explicitly lock the blob into an active state
    isProcessingRequest = true;
    toggleSendButtonUI(true);
    activeFace.setWorkingState(true);
    // The immersion dive is gated on USER turns only — ambient log
    // activity keeps driving speed/glow via setWorkingState, but only
    // this lifecycle swallows the camera.
    if (typeof activeFace.setUserTurn === 'function') activeFace.setUserTurn(true);
    // Only flip the dot to 'busy' if the WS is actually up — otherwise
    // the 'error' state should stay so the user sees the real problem.
    if (ws && ws.readyState === WebSocket.OPEN) {
        setConnectionState('busy', 'PROCESSING…');
    }
    try {
        // Deliberately omit `model`: the agent validates a supplied model
        // name against its single configured model and returns 404
        // ModelNotFound on any mismatch (see api/routes.py). Pinning a
        // name here means every model rename on the agent silently breaks
        // chat with a 404. With no `model`, the agent uses its configured
        // model (`requested_model or configured_model`), so the client
        // always tracks whatever the agent is running.
        // Strip client-only bookkeeping (reqId/feedback — the 👍/👎 label
        // plumbing) so the wire payload stays clean OpenAI message shapes.
        const payload = {
            messages: chatHistory.map(toWireMessage),
            stream: true
        };
        // Durable sessions: when the workspace module holds an active
        // session id, bind the turn to it. The AGENT is the source of
        // truth for session history and merges tolerantly, so replaying
        // the full local history here can never double it.
        // ⚠ Fall back to the STORED id. `window.__ghostSessionId` is only
        // published by sessions.js, which runs after the dynamic
        // import('./workspace.js') chain plus three agent round trips — and
        // the composer is live throughout. A turn sent in that window
        // carried no session_id, was never persisted, and the next resync
        // (which protects user messages but not assistant ones) then deleted
        // the reply the user was reading. `saveInflightHandle` has had this
        // exact fallback all along (R3 lens B).
        const _sid = window.__ghostSessionId
            || safeStorage.get('ghost_session_id') || null;
        if (_sid) payload.session_id = _sid;
        currentChatController = new AbortController();

        let response;
        if (resuming && currentTaskId) {
            currentTTSMutedLength = currentAccumulatedContent.length;
            currentAccumulatedContent = "";
            currentChunkIndex = 0;
            response = await fetch(`/api/chat/resume/${currentTaskId}?offset=0`, {
                signal: currentChatController.signal
            });
            setTimeout(scrollToBottom, 50);
        } else {
            // A genuinely NEW upstream turn (also reachable on a resume
            // whose task handle was lost) — any captured id belongs to an
            // abandoned turn and must not stamp this one's reply.
            currentReqId = null;
            response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
                signal: currentChatController.signal
            });
            if (response.headers.has('X-Task-ID')) {
                currentTaskId = response.headers.get('X-Task-ID');
                // Survive a page kill: the handle is all a reload needs to
                // reattach this stream (see resumeOrReconcileInflightTurn).
                saveInflightHandle();
            }
        }

        if (!response.ok) {
            throw await _httpError(response);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let streamBuffer = "";
        // Set when an error frame renders; suppresses the phantom
        // "No response" assistant reply that would otherwise be pushed into
        // chatHistory and persisted (see the finalize branch below).
        let streamHadError = false;
        // Reset per turn: the network-resume attempts below are bounded so a
        // down proxy cannot spin a silent retry loop forever.
        _netResumeTries = 0;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            currentChunkIndex++;

            streamBuffer += decoder.decode(value, { stream: true });
            let lines = streamBuffer.split('\n');

            // Keep the last partial line in the buffer
            streamBuffer = lines.pop();

            for (const line of lines) {
                const trimmedLine = line.trim();
                if (!trimmedLine || !trimmedLine.startsWith("data: ")) continue;

                const dataStr = trimmedLine.substring(6).trim();
                if (dataStr === "[DONE]") continue;

                try {
                    const data = JSON.parse(dataStr);
                    let chunkContent = "";

                    // Every frame carries the agent's request id as
                    // "chatcmpl-<id>" — capture once for feedback labels.
                    if (!currentReqId && typeof data.id === 'string' && data.id) {
                        currentReqId = data.id.replace(/^chatcmpl-/, '');
                    }

                    if (data.choices && data.choices[0] && data.choices[0].delta && data.choices[0].delta.content) {
                        chunkContent = data.choices[0].delta.content;
                    } else if (data.message && data.message.content) {
                        // Fallback for non-streaming formats that might be wrapped
                        chunkContent = data.message.content;
                    } else if (data.error) {
                        // The agent and the proxy both send STRUCTURED errors
                        // ({message, type, error_id}); a bare template turned
                        // every one of them into "Error: [object Object]".
                        // That is the string the operator saw for the agent
                        // being down (UpstreamError), a evicted task
                        // (TaskEvicted), a blown buffer (BufferCapExceeded)
                        // and an agent InternalError alike — destroying the
                        // error_id that correlates to the log. Objects only
                        // ever reached this branch, so the bug was total.
                        const _e = data.error;
                        let _msg = (_e && typeof _e === 'object')
                            ? (_e.message || _e.detail || _e.error || JSON.stringify(_e))
                            : String(_e);
                        const _type = (_e && typeof _e === 'object' && _e.type) ? ` [${_e.type}]` : '';
                        const _eid = (_e && typeof _e === 'object' && _e.error_id) ? ` (error_id ${_e.error_id})` : '';
                        addMessage('system', `Error${_type}: ${_msg}${_eid}`);
                        streamHadError = true;
                        activeFace.triggerSpike();
                        continue;
                    }
                    if (chunkContent) {
                        if (currentAccumulatedContent === "") {
                            if (currentThinkingInterval) {
                                clearInterval(currentThinkingInterval);
                                currentThinkingInterval = null;
                            }
                            // Swap off the typing-dots indicator and clear
                            // whatever innerHTML the placeholder had before
                            // content starts streaming in.
                            currentAgentMessageDiv.classList.remove('thinking');
                            currentAgentMessageDiv.textContent = "";
                            // From here the reply itself is the story —
                            // the caption says so until turn end.
                            setTurnStatusDesc('writing the reply…', '💬');
                        }

                        currentAccumulatedContent += chunkContent;

                        // --- Voice Intercept Logic ---
                        if (isTTSActive && currentAccumulatedContent.length > currentTTSMutedLength) {
                            ttsBuffer += chunkContent;
                            
                            let openThink = ttsBuffer.lastIndexOf('<think>') > ttsBuffer.lastIndexOf('</think>');
                            let openTool = ttsBuffer.lastIndexOf('<tool_call') > ttsBuffer.lastIndexOf('</tool_call>');
                            
                            if (!openThink && !openTool) {
                                // Strip markdown, XML tags, and reasoning blocks
                                let cleanBuffer = ttsBuffer.replace(/(<think>[\s\S]*?<\/think>|<tool_call[\s\S]*?(?:<\/tool_call>|$)|<[^>]+>|\*|_|`|#|\[|\]|\(|\)|!\[.*?\]\(.*?\))/gi, "");
                                
                                // Find natural sentence boundaries
                                let boundaryMatch = cleanBuffer.match(/([.?!;]+[\s]+)/);
                                if (boundaryMatch) {
                                    let splitIndex = boundaryMatch.index + boundaryMatch[0].length;
                                    let sentence = cleanBuffer.substring(0, splitIndex).trim();
                                    
                                    // Only queue if it contains actual words or numbers
                                    if (/[\p{L}\p{N}]/u.test(sentence)) {
                                        queueTTS(sentence);
                                    }
                                    
                                    ttsBuffer = cleanBuffer.substring(splitIndex); // Keep remainder
                                }
                            }
                        }
                        // -----------------------------
                    }
                } catch (e) {
                    console.warn("Failed to parse SSE chunk:", dataStr, e);
                }
            }

            // Coalesce re-renders to one per animation frame (see
            // _renderStreamingContent) instead of re-parsing the whole
            // markdown on every network read.
            if (currentAccumulatedContent !== "" && currentAgentMessageDiv) {
                scheduleStreamRender();
            }
        }

        // Flush a final synchronous render so the last tokens are shown
        // even if an rAF was still pending when the stream ended.
        _renderStreamingContent();

        // Push the final concatenated message to chat history. On the
        // resume path the partial assistant turn was never pushed when it
        // dropped, so a plain push is correct in both cases. `reqId` is
        // client-only bookkeeping (stripped from wire payloads) that lets
        // the 👍/👎 label buttons survive a reload.
        if (currentAccumulatedContent) {
            chatHistory.push({ role: "assistant", content: currentAccumulatedContent,
                               reqId: currentReqId || undefined });
            // Stamp only alongside a history entry that carries the id —
            // a "No response" bubble with live thumbs would take labels
            // that can never persist (no matching entry to latch onto).
            _stampReqId(currentAgentMessageDiv, currentReqId);
        } else if (streamHadError) {
            // An error frame already rendered a system message saying what
            // went wrong. Pushing a phantom assistant reply on top of it put
            // "No response" into chatHistory — and from there into the
            // DURABLE session store, where it is indistinguishable from a
            // real (bad) reply and becomes context for later turns. The
            // error is the outcome; there is no assistant turn to record.
            currentAgentMessageDiv.remove();
        } else {
            currentAgentMessageDiv.textContent = "No response";
            chatHistory.push({ role: "assistant", content: "No response" });
            // Clear any stale stamp from a previous turn on a reused
            // bubble — an unlabeled "No response" must carry no thumbs.
            _stampReqId(currentAgentMessageDiv, null);
        }
        if (typeof saveChatState === 'function') saveChatState();
        if (typeof notifyUser === 'function') notifyUser("Response complete.");
        // Delivery ack: proof that live JS rendered the reply. Without it
        // the server assumes the phone is locked and sends the reply-ready
        // push after the grace window.
        if (currentTaskId) {
            fetch(`/api/chat/ack/${encodeURIComponent(currentTaskId)}`,
                { method: 'POST' }).catch(() => {});
        }
        currentTaskId = null; // Clear task id
        clearInflightHandle();

        if (typeof updateWorkspaceBtnState === 'function') updateWorkspaceBtnState();
    } catch (e) {
        if (e.name === 'AbortError') {
            if (currentThinkingInterval) { clearInterval(currentThinkingInterval); currentThinkingInterval = null; }
            if (currentTaskId) {
                fetch(`/api/chat/cancel/${currentTaskId}`, { method: 'POST' }).catch(()=>{});
            }
            // /api/chat/cancel only tears down the PROXY's buffered stream:
            // the agent kept running the whole turn, holding the global turn
            // lock and burning tokens, while this client said "Request
            // cancelled by user." (review R1 M8). Cancel the real turn too —
            // by request_id when the stream got that far, else the
            // currently-running turn, which is ours because the lock is
            // global. Report failure, because then it is NOT cancelled.
            _cancelAgentTurn(currentReqId, _lastSentUserText);
            if (!resuming && currentAccumulatedContent === "" && currentAgentMessageDiv) currentAgentMessageDiv.remove();
            if (currentAccumulatedContent !== "") {
                // Keep the reqId: a reply aborted BECAUSE it was going
                // wrong is a prime 👎 target, and the trajectory exists.
                chatHistory.push({ role: "assistant",
                                   content: currentAccumulatedContent + "\n\n*[Aborted]*",
                                   reqId: currentReqId || undefined });
                if (currentAgentMessageDiv && currentAgentMessageDiv.isConnected) {
                    _stampReqId(currentAgentMessageDiv, currentReqId);
                }
                addMessage('system', 'Stopped by user.');
            } else {
                addMessage('system', 'Stopped by user.');
            }
            currentTaskId = null;
            clearInflightHandle();
            if (typeof saveChatState === 'function') saveChatState();
            // The agent may have persisted the FULL reply while this client
            // kept the aborted stub — let the sessions module re-align from
            // the server so the next replay can't diverge.
            try {
                window.GhostCore?.events?.dispatchEvent(new CustomEvent('turn-aborted'));
            } catch (err) { /* workspace modules not loaded */ }
        } else {
            if (currentThinkingInterval) { clearInterval(currentThinkingInterval); currentThinkingInterval = null; }
            
            const errMsg = e.message.toLowerCase();
            if (errMsg.includes('load failed') || errMsg.includes('networkerror') || errMsg.includes('fetch')) {
                // Safari killed the fetch by suspending the tab. Resume
                // silently — no system bubble; the reply continuing is the
                // only signal.
                // ⚠ GUARDED RESUME. These two callbacks are the FOURTH
                // resume trigger and they bypassed `resumeLatch` entirely.
                // On an iOS unlock the latched path (boot /
                // visibilitychange / pageshow) replays an already-complete
                // buffer in milliseconds and NULLS currentTaskId — so this
                // 1s callback then found no task, fell through the
                // `resuming && currentTaskId` test, and took the POST
                // branch: a whole NEW turn whose last message is an
                // assistant reply. There is also no attempt cap, so a down
                // proxy produced an endless silent 1s retry loop behind a
                // permanent thinking bubble.
                const _resumeAttempt = () => {
                    if (!currentTaskId) return;          // already resumed
                    if (isProcessingRequest) return;     // the latch's job
                    if (_netResumeTries >= 3) {
                        addMessage('system',
                            '⚠ Lost the connection to this turn and could ' +
                            'not reattach after 3 tries. The turn may still ' +
                            'be running — reload to pick it up.');
                        _netResumeTries = 0;
                        // ⚠ RELEASE THE HANDLE. The 5s inflight heartbeat
                        // kept writing `beat`, so `Date.now() - h.beat <
                        // 15000` stayed true forever and no other tab (or
                        // reopened PWA) could ever adopt this orphaned turn
                        // — the exact recovery this subsystem exists for
                        // (R2 lens B).
                        clearInflightHandle();
                        return;
                    }
                    _netResumeTries += 1;
                    sendMessage(true);
                };
                if (document.visibilityState === 'visible') {
                    setTimeout(_resumeAttempt, 1000);
                } else {
                    document.addEventListener('visibilitychange', function resumeOnVisible() {
                        if (document.visibilityState === 'visible') {
                            document.removeEventListener('visibilitychange', resumeOnVisible);
                            setTimeout(_resumeAttempt, 1000);
                        }
                    });
                }
            } else {
                // Not every failure here is a NETWORK failure. The 500-message
                // interface cap surfaces as `Too many messages (501 > 500
                // cap)` — a lock-out that needs /clear or a session switch,
                // not a retry — and labelling it "Network Error" pointed the
                // operator at the wrong problem (review R1 M11).
                const _m = String(e && e.message || e);
                if (/Too many messages/i.test(_m)) {
                    addMessage('system',
                        `This conversation is too long for one request (${_m}). `
                        + 'Start a new chat, or switch sessions and back — the '
                        + 'server keeps the durable copy, so nothing is lost.');
                } else {
                    addMessage('system', `Network Error: ${_m}`);
                }
                activeFace.triggerSpike();
                // The turn is abandoned client-side. If the agent finishes
                // anyway, the sessions resync (visibilitychange) adopts it.
                clearInflightHandle();
            }
        }
    } finally {
        if (currentThinkingInterval) { clearInterval(currentThinkingInterval); currentThinkingInterval = null; }
        // Cancel any throttled render still queued for a future frame —
        // otherwise it fires after this point and re-adds `streaming`.
        _cancelScheduledStreamRender();
        // Drop the streaming cursor glyph and any stale "thinking" class
        // so the message renders as a completed reply.
        //
        // ⚠ THE CLASS IS NOT THE ANIMATION. `.thinking` is styled in no CSS
        // file at all; the bouncing dots belong to `.typing-indicator span`,
        // a CHILD element that only ever disappears when the bubble's
        // content is replaced. So on every zero-content end — a 500, a 401
        // after key rotation, the 413 message cap, an abort during a resume,
        // the resume give-up — this teardown removed a class that styles
        // nothing and left dots animating forever, with a timestamp and a ⋯
        // menu decorating the corpse (R2 lens B).
        if (currentAgentMessageDiv) {
            currentAgentMessageDiv.classList.remove('streaming', 'thinking');
            const _ind = currentAgentMessageDiv.querySelector('.typing-indicator');
            if (_ind) {
                if (currentAccumulatedContent) {
                    _ind.remove();          // real content arrived; the
                                            // renderer normally does this
                } else {
                    // Nothing ever streamed: this bubble has no content to
                    // become. `decorateMessageActions()` below would give it
                    // a timestamp and a ⋯ menu, making an animating stub read
                    // as a finished empty reply.
                    currentAgentMessageDiv.remove();
                    currentAgentMessageDiv = null;
                }
            }
        }

        if (isTTSActive && ttsBuffer.trim().length > 0) {
            let finalClean = ttsBuffer.replace(/(<think>[\s\S]*?<\/think>|<tool_call[\s\S]*?(?:<\/tool_call>|$)|<[^>]+>|\*|_|`|#|\[|\]|\(|\)|!\[.*?\]\(.*?\))/gi, "").trim();
            if (/[\p{L}\p{N}]/u.test(finalClean)) queueTTS(finalClean);
            ttsBuffer = "";
        }

        isProcessingRequest = false;
        currentChatController = null;
        toggleSendButtonUI(false);
        activeFace.setWorkingState(false);
        if (typeof activeFace.setUserTurn === 'function') activeFace.setUserTurn(false);
        if (ws && ws.readyState === WebSocket.OPEN) {
            setConnectionState('online', 'ONLINE');
        }
        setTimeout(scrollToBottom, 100);

        // Auto-hide planner monologue 2 seconds after reply
        monologueTimeout = setTimeout(hidePlannerMonologue, 2000);

        // Attach "Visualize" buttons to any renderable code blocks
        attachRenderButtons();
        // Timestamp + actions row for the completed reply, and let the
        // sessions rail refresh (the agent may have just created the
        // session / derived its title).
        decorateMessageActions();
        try {
            window.GhostCore?.events?.dispatchEvent(new CustomEvent('turn-complete'));
        } catch (e) { /* workspace modules not loaded */ }
    }
}

// A TYPED message leaves voice mode: if you're at a keyboard you're reading,
// not listening. The mic path (which sets isTTSActive) deliberately calls
// sendMessage() directly so a spoken turn still gets a spoken reply.
function sendTypedMessage() {
    isTTSActive = false;
    sendMessage();
}

sendBtn.addEventListener('click', () => {
    if (isProcessingRequest && currentChatController) {
        currentChatController.abort();
        toggleSendButtonUI(false);
    } else if (!isProcessingRequest) {
        sendTypedMessage();
    }
});

chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (!isProcessingRequest) {
            sendTypedMessage();
        }
    }
});

if (fullscreenBtn) {
    fullscreenBtn.addEventListener('click', () => { document.body.classList.toggle('zen-mode'); });
}

// Face-form picker (2026-07-29 — replaced the blind cycle: at 9 forms,
// reaching the one you wanted took up to 8 clicks × 1.4s reorganization
// blends, each through a form you didn't ask for). The button now opens
// a menu built from matrix_graph's own roster (getForms — a form added
// there appears here with no edit), current form highlighted, one click
// jumps straight to it. Falls back to the old cycle if the face module
// predates getForms/setForm (stale cache).
const FACE_FORM_HINTS = {
    abyssal: 'deep-sea creature',
    horizon: 'collapsing orbits',
    cortex: 'neural lobes',
    vortex: 'black hole',
    lattice: 'weight tensor',
    stack: 'transformer',
    embedding: 'latent space',
    descent: 'loss landscape',
    cube: 'infinite monolith',
    empty: 'no face',
};
const faceFormBtn = document.getElementById('face-form-btn');
let faceFormMenu = null;

function closeFaceFormMenu() {
    if (faceFormMenu) faceFormMenu.classList.add('hidden');
}

function buildFaceFormMenu() {
    if (faceFormMenu) return faceFormMenu;
    faceFormMenu = document.createElement('div');
    faceFormMenu.id = 'face-form-menu';
    faceFormMenu.className = 'hidden';
    faceFormMenu.setAttribute('role', 'menu');
    faceFormMenu.setAttribute('aria-label', 'Face form');
    for (const name of activeFace.getForms()) {
        const item = document.createElement('button');
        item.type = 'button';
        item.className = 'face-form-item';
        item.dataset.form = name;
        item.setAttribute('role', 'menuitemradio');
        const label = document.createElement('span');
        label.className = 'face-form-name';
        label.textContent = name;
        item.appendChild(label);
        const hint = document.createElement('span');
        hint.className = 'face-form-hint';
        hint.textContent = FACE_FORM_HINTS[name] || '';
        item.appendChild(hint);
        item.addEventListener('click', () => {
            const picked = activeFace.setForm(name);
            markActiveFaceForm();
            closeFaceFormMenu();
            const toast = window.__ghostWorkspace && window.__ghostWorkspace.toast;
            if (toast) toast(`Face form: ${picked}`);
        });
        faceFormMenu.appendChild(item);
    }
    document.body.appendChild(faceFormMenu);
    // Outside click / Escape closes — standard menu manners.
    document.addEventListener('click', (ev) => {
        if (!faceFormMenu.classList.contains('hidden')
            && !faceFormMenu.contains(ev.target)
            && ev.target !== faceFormBtn && !faceFormBtn.contains(ev.target)) {
            closeFaceFormMenu();
        }
    });
    document.addEventListener('keydown', (ev) => {
        if (ev.key === 'Escape') closeFaceFormMenu();
    });
    return faceFormMenu;
}

function markActiveFaceForm() {
    if (!faceFormMenu) return;
    const current = typeof activeFace.getForm === 'function' ? activeFace.getForm() : '';
    for (const item of faceFormMenu.querySelectorAll('.face-form-item')) {
        item.classList.toggle('active', item.dataset.form === current);
        item.setAttribute('aria-checked', item.dataset.form === current ? 'true' : 'false');
    }
}

if (faceFormBtn) {
    faceFormBtn.addEventListener('click', () => {
        if (typeof activeFace.getForms !== 'function'
            || typeof activeFace.setForm !== 'function') {
            // Stale-cache fallback: the old blind cycle still works.
            if (typeof activeFace.cycleForm !== 'function') return;
            const name = activeFace.cycleForm();
            const toast = window.__ghostWorkspace && window.__ghostWorkspace.toast;
            if (toast) toast(`Face form: ${name}`);
            return;
        }
        const menu = buildFaceFormMenu();
        if (!menu.classList.contains('hidden')) {
            closeFaceFormMenu();
            return;
        }
        markActiveFaceForm();
        // Anchor under the button, right-aligned to it, clamped on-screen
        // on BOTH axes (the #msg-menu manners — workspace.js). The old code
        // clamped only the right offset: on a phone the centered header
        // puts this button near the LEFT edge, so a right-anchored 230px+
        // menu hung ~half off the left of the screen. Must be visible
        // before measuring (hidden = display:none).
        const rect = faceFormBtn.getBoundingClientRect();
        menu.classList.remove('hidden');
        const mw = menu.offsetWidth;
        const mh = menu.offsetHeight;
        const rightMax = Math.max(8, window.innerWidth - mw - 8);
        menu.style.right = `${Math.min(
            Math.max(8, Math.round(window.innerWidth - rect.right)), rightMax)}px`;
        let top = Math.round(rect.bottom + 8);
        if (top + mh > window.innerHeight - 8) {
            top = Math.max(8, window.innerHeight - mh - 8);
        }
        menu.style.top = `${top}px`;
    });
}

// Global toggle for Zen Mode (Persistent Key)
document.addEventListener('keydown', (e) => {
    // Toggle with 'Z' unless the user is typing in ANY text field (chat
    // input, session search, palette, memory modal…) — the original
    // check excluded only #chat-input, so typing "z" in a newer field
    // would flip zen mode mid-word.
    const el = document.activeElement;
    const typing = el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA'
        || el.isContentEditable);
    if (e.key.toLowerCase() === 'z' && !typing && !e.metaKey && !e.ctrlKey) {
        document.body.classList.toggle('zen-mode');
    }
});

// Workspace Save/Load Logic
const workspaceBtn = document.getElementById('workspace-btn');
const workspaceUploadInput = document.getElementById('workspace-upload-input');

function updateWorkspaceBtnState() {
    if (!workspaceBtn) return;
    if (chatHistory.length === 0) {
        workspaceBtn.title = "Load Workspace";
    } else {
        workspaceBtn.title = "Save Workspace";
    }
    workspaceBtn.innerHTML = `
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"></path>
            <polyline points="17 21 17 13 7 13 7 21"></polyline>
            <polyline points="7 3 7 8 15 8"></polyline>
        </svg>
    `; // Save/Load icon
}

if (workspaceBtn && workspaceUploadInput) {
    updateWorkspaceBtnState();

    workspaceBtn.addEventListener('click', async () => {
        if (isProcessingRequest) return;
        if (chatHistory.length === 0) {
            workspaceUploadInput.click();
        } else {
            isProcessingRequest = true;
            activeFace.setWorkingState(true);
            // The composer must SHOW the busy state: without this the
            // button still reads SEND, and a click hits neither branch of
            // the handler while Enter tests !isProcessingRequest — the user
            // is silently ignored (R2 lens B).
            toggleSendButtonUI(true, false);   // no AbortController here

            try {
                const response = await fetch('/api/workspace/save', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    // Same wire-shape strip as /api/chat: a workspace
                    // restored weeks later would otherwise present live
                    // thumbs whose reqIds are past the agent's 8-day
                    // trajectory scan — guaranteed 404s.
                    body: JSON.stringify({ chat_history: chatHistory.map(toWireMessage) })
                });
                
                if (!response.ok) throw await _httpError(response, 'Save failed');
                
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                
                let filename = `workspace_${new Date().toISOString().replace(/[:.]/g, '-')}.zip`;
                const disposition = response.headers.get('content-disposition');
                if (disposition && disposition.indexOf('filename=') !== -1) {
                    filename = disposition.split('filename=')[1].replace(/["']/g, '');
                }
                
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                a.remove();
                window.URL.revokeObjectURL(url);
                
                addMessage('system', 'Workspace saved successfully.');
            } catch (err) {
                addMessage('system', `Save Workspace Error: ${err.message}`);
                activeFace.triggerSpike();
            } finally {
                isProcessingRequest = false;
                activeFace.setWorkingState(false);
                toggleSendButtonUI(false);
            }
        }
    });

    workspaceUploadInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        workspaceUploadInput.value = '';
        
        isProcessingRequest = true;
        activeFace.setWorkingState(true);
        // The composer must SHOW the busy state: without this the
        // button still reads SEND, and a click hits neither branch of
        // the handler while Enter tests !isProcessingRequest — the user
        // is silently ignored (R2 lens B).
        toggleSendButtonUI(true, false);   // no AbortController here
        addMessage('system', `Loading workspace from ${file.name}...`);
        
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch('/api/workspace/load', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) throw await _httpError(response, 'Load failed');
            
            const result = await response.json();
            if (result.error) throw new Error(result.error);
            
            // Shape-validate like the bridge setter does: a non-array here
            // makes the next `chatHistory.push` throw BEFORE sendMessage's
            // try block, killing the composer with a console-only error.
            chatHistory = Array.isArray(result.chat_history)
                ? result.chat_history : [];
            renderHistoryToLog(chatHistory);
            // PERSIST, and take a FRESH session identity. Without the save,
            // a reload silently restored the old localStorage history over
            // the loaded workspace; without the rebind, the next turn
            // appended the workspace's messages into whatever durable
            // session was current — contaminating a conversation the
            // operator never opened (R2 lens B).
            saveChatState();
            try {
                window.GhostCore?.events?.dispatchEvent(
                    new CustomEvent('conversation-cleared'));
            } catch (err) { /* workspace modules not loaded */ }

            addMessage('system', 'Workspace loaded successfully.');
            updateWorkspaceBtnState();

        } catch (error) {
            addMessage('system', `Workspace Load Error: ${error.message}`);
            activeFace.triggerSpike();
        } finally {
            isProcessingRequest = false;
            activeFace.setWorkingState(false);
            toggleSendButtonUI(false);
            scrollToBottom();
        }
    });
}

// File Transfer Logic
const uploadBtn = document.getElementById('upload-btn');
const downloadBtn = document.getElementById('download-btn');
const fileUploadInput = document.getElementById('file-upload-input');

if (uploadBtn && fileUploadInput) {
    uploadBtn.addEventListener('click', () => {
        if (isProcessingRequest) return;
        fileUploadInput.click();
    });

    fileUploadInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        // Reset input immediately so the same file can be uploaded again if needed
        fileUploadInput.value = '';

        isProcessingRequest = true;
        activeFace.setWorkingState(true);
        toggleSendButtonUI(true, false);   // no AbortController here

        addMessage('system', `Uploading ${file.name} to sandbox...`);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                // The server explains WHY (type not allowed, too large, no
                // filename); pasting only the status threw that away.
                throw await _httpError(response, 'Upload failed');
            }

            const result = await response.json();
            if (result.error) {
                addMessage('system', `Upload Failed: ${result.error}`);
                activeFace.triggerSpike();
            } else {
                // NOTE: an `updateActivityIcon('✅')` call sat here until
                // 2026-08-02. The function was deleted with the center-stage
                // #activity-icon on 2026-07-29 but two call sites survived, so
                // this threw a ReferenceError on every SUCCESSFUL upload — the
                // enclosing catch then reported "Upload Error" for an upload
                // that had just worked. The success message above is the
                // status signal; the turn-status line owns icons now.
                addMessage('system', `Successfully uploaded ${file.name}.`);
            }
        } catch (error) {
            addMessage('system', `Upload Error: ${error.message}`);
            activeFace.triggerSpike();
        } finally {
            isProcessingRequest = false;
            activeFace.setWorkingState(false);
            toggleSendButtonUI(false);
            scrollToBottom();
        }
    });
}

if (downloadBtn) {
    downloadBtn.addEventListener('click', () => {
        if (isProcessingRequest) return;
        const filename = prompt("Enter the exact filename to download from the sandbox:");
        if (filename && filename.trim() !== '') {
            addMessage('system', `Starting download for ${filename}...`);
            // Trigger download by creating a hidden anchor tag
            const url = `/api/download/${encodeURIComponent(filename.trim())}`;
            // Must go through fetch (not a plain <a> navigation) so the
            // authed wrapper attaches X-Ghost-Key. A plain-anchor fallback
            // can't carry the header, so it would just 401 — and skipping
            // the res.ok check would "download" the error JSON as the file.
            fetch(url)
                .then(async (res) => {
                    // `await` needs an async callback — and the server's
                    // reason ("no such file: report.pdf") is what the
                    // operator needs, not a bare status (R3 lens B).
                    if (!res.ok) throw await _httpError(res, 'Download failed');
                    return res.blob();
                })
                .then(blob => {
                    const objectUrl = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = objectUrl;
                    a.download = filename.trim();
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    setTimeout(() => URL.revokeObjectURL(objectUrl), 100);
                })
                .catch(err => {
                    addMessage('system', `Download failed for ${filename.trim()}: ${err.message}`);
                    activeFace.triggerSpike();
                });
        }
    });
}

if (window.visualViewport) {
    let bodySettleTimer = null;
    const syncBodyHeight = () => {
        scrollToBottom();
        _vvNoteHeight();
        // Pin the body ONLY while a keyboard is genuinely open (focused
        // editable + vv meaningfully below this orientation's max — see
        // _vvKeyboardOpen; innerHeight comparisons are unusable because
        // standalone PWAs shrink innerHeight WITH the keyboard). Clearing
        // to the CSS 100dvh steady state otherwise keeps one stale WebKit
        // reading (likelier the longer a reply streams) from freezing the
        // body short — footer floating mid-screen (operator repro).
        if (_vvKeyboardOpen()) {
            document.body.style.height = window.visualViewport.height + 'px';
        } else {
            document.body.style.height = '';
        }
        window.scrollTo(0, 0);
    };
    window.visualViewport.addEventListener('resize', syncBodyHeight);
    // Same rotation-transient hazard as the keyboard offset above: if the
    // LAST vv resize fired mid-rotate, body height stuck at the wrong
    // orientation's value. Re-sync once the window geometry settles.
    window.addEventListener('resize', () => {
        clearTimeout(bodySettleTimer);
        bodySettleTimer = setTimeout(syncBodyHeight, 300);
    });

    // ── Geometry watchdog ──────────────────────────────────────────
    // Events cannot be trusted to arrive, arrive in order, or carry
    // fresh numbers on iOS (rotation, scroll-dismiss, WebKit staleness —
    // three distinct producers of the same stuck-layout symptom so far).
    // Instead of chasing each producer, verify the applied geometry
    // against FRESH measurements on a slow tick and correct drift. Only
    // acts when something is measurably wrong (>32px), so the steady
    // state costs two computed-style reads per tick.
    setInterval(() => {
        try {
            const vv = window.visualViewport;
            if (Math.abs(window.innerWidth - vv.width) > 2) return; // mid-rotate
            _vvNoteHeight();
            // 1. Body pin: BOTH directions. A stale pin with no keyboard
            // floats the input mid-screen; a MISSING pin with the keyboard
            // open hides the input behind it (typing blind) — heal each.
            const pinned = document.body.style.height;
            if (_vvKeyboardOpen()) {
                const want = vv.height;
                if (!pinned || Math.abs(parseFloat(pinned) - want) > 32) {
                    document.body.style.height = want + 'px';
                    window.scrollTo(0, 0);
                }
            } else if (pinned) {
                document.body.style.height = '';
                window.scrollTo(0, 0);
            }
            // 2. Keyboard-translate drift.
            const applied = parseFloat(getComputedStyle(document.documentElement)
                .getPropertyValue('--keyboard-height')) || 0;
            if (applied > 0) {
                const ae = document.activeElement;
                const editing = !!ae && (ae.tagName === 'TEXTAREA'
                    || ae.tagName === 'INPUT' || ae.isContentEditable);
                let want = editing
                    ? Math.max(0, window.innerHeight - vv.height - vv.offsetTop)
                    : 0;
                want = Math.min(want, Math.round(window.innerHeight * 0.55));
                if (Math.abs(applied - want) > 32) {
                    document.documentElement.style.setProperty(
                        '--keyboard-height', `${want}px`);
                }
            }
        } catch (e) { /* never let the watchdog take the page down */ }
    }, 800);
}

document.addEventListener('dblclick', function (event) { event.preventDefault(); }, { passive: false });

setTimeout(() => {
    const sysMsg = document.getElementById('init-msg');
    if (sysMsg) {
        sysMsg.style.transition = 'opacity 1s ease';
        sysMsg.style.opacity = '0';
        setTimeout(() => sysMsg.remove(), 1000);
    }
}, 2000);

// Warn once at boot if either markdown dependency didn't load — renderMarkdown
// quietly falls back to escaped text, but silent markdown breakage is a bad
// debugging experience.
if (!window.marked || !window.DOMPurify) {
    console.warn(
        '[Ghost] Markdown/sanitizer CDN did not load. Falling back to plain text.',
        { marked: !!window.marked, DOMPurify: !!window.DOMPurify }
    );
}

// Restore the previously chosen face (default = first) and start it.
activeFace.init();
// Dev/verification hook: lets headless checks (Playwright) drive the
// face's working state and read its debug state. Harmless in prod —
// the page is key-gated and the hook mutates nothing the WS handlers
// don't already mutate.
window.__ghostFace = activeFace;
loadChatState();
// Reattach or reconcile a turn that outlived the page. Ordering vs the
// sessions boot (dynamic import below) is TIMING, not structure: the
// local /state probe normally resolves well before sessions.js finishes
// its two module fetches + agent round trips, so sendMessage(true) has
// already claimed isProcessingRequest (checked by the sessions boot)
// by the time it could reload history. If the probe loses the race the
// cost is a redundant render, not corruption — reconcileFromSession
// re-checks isProcessingRequest after every await.
resumeOrReconcileInflightTurn();
document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') resumeOrReconcileInflightTurn();
});
window.addEventListener('pageshow', (e) => {
    if (e.persisted) resumeOrReconcileInflightTurn();
});
connectWebSocket();
if (typeof updateWorkspaceBtnState === 'function') updateWorkspaceBtnState();
// If the chat log is empty after loading history, show an onboarding hero.
if (!chatHistory || chatHistory.length === 0) renderEmptyStateHero();

// (Reconnect-on-visible is handled by the single visibilitychange
// listener defined earlier — see connectWebSocket wiring above.)

// ═══════════════════════════════════════════════════════════════
//  Render Window – Visualizer Logic
// ═══════════════════════════════════════════════════════════════

// --- Mermaid init ---
// Guarded: if the mermaid CDN failed to load, `mermaid` is undefined.
// This runs at module top-level *before* the render-window element refs
// and listeners below, so an unguarded throw here would leave those
// consts in the TDZ and make `attachRenderButtons()` (called from
// sendMessage's finally) throw on every completed message. Mirrors the
// defensive marked/DOMPurify handling.
if (window.mermaid) {
    try { mermaid.initialize({ startOnLoad: false, theme: 'dark' }); }
    catch (e) { console.warn('mermaid.initialize failed', e); }
} else {
    console.warn('[Ghost] mermaid CDN did not load — diagram rendering disabled.');
}

// --- Element references ---
const renderWindow = document.getElementById('render-window');
const renderHeader = document.getElementById('render-header');
const renderIframe = document.getElementById('render-iframe');
// Opaque-origin iframe for untrusted agent HTML/JS (see index.html).
const renderIframeSandboxed = document.getElementById('render-iframe-sandboxed');
const mermaidContainer = document.getElementById('mermaid-container');
const chartContainer = document.getElementById('chart-container');
const renderChart = document.getElementById('render-chart');
const renderCloseBtn = document.getElementById('render-close');
const renderZoomIn = document.getElementById('render-zoom-in');
const renderZoomOut = document.getElementById('render-zoom-out');
const renderDownloadBtn = document.getElementById('render-download');

let currentChart = null;
let currentZoom = 1.0;
let currentRenderState = null;
let _pdfBlobUrl = null;   // PDFs bypass the image LRU (multi-MB); one live URL

// HTML-attribute escaper for srcdoc interpolation. The old code used
// contentDocument.write + setAttribute to dodge attribute breakout —
// correct, but document.write races the close handler's about:blank
// navigation and cannot follow a PDF src (contentDocument inaccessible).
// srcdoc with PROPERLY ESCAPED values is equally breakout-proof and
// deterministic from any prior frame state.
const _escAttr = (s) => String(s).replace(/[&<>"']/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;',
              '"': '&quot;', "'": '&#39;' }[c]));

// Every open path goes through this FIRST. Hiding a surface is not enough:
// agent code in the sandboxed frame keeps running (audio, timers) while
// display:none, a destroyed chart leaves its canvas, the previous mermaid
// SVG masquerades as the new diagram, and a stale zoom magnifies a blank
// margin. Tear down, then let the caller show its own surface.
function resetRenderSurfaces() {
    renderIframe.style.display = 'none';
    renderIframeSandboxed.style.display = 'none';
    mermaidContainer.style.display = 'none';
    chartContainer.style.display = 'none';
    renderIframeSandboxed.removeAttribute('srcdoc');   // stops agent code
    renderIframe.removeAttribute('srcdoc');
    if (renderIframe.getAttribute('src')) renderIframe.src = 'about:blank';
    mermaidContainer.replaceChildren();
    if (currentChart) {
        currentChart.destroy();
        currentChart = null;
    }
    currentZoom = 1.0;
    applyZoom();
}

// Wrapper-document helper for our OWN markup (never agent content).
function _setMainFrameHTML(html) {
    renderIframe.removeAttribute('src');
    renderIframe.srcdoc = html;   // srcdoc wins over any queued navigation
    renderIframe.style.display = 'block';
}

// Friendly in-window failure panel — a visualizer that pops open and shows
// NOTHING is the defect class this file kept reproducing.
function _showRenderError(text) {
    currentRenderState = null;
    _setMainFrameHTML(
        '<!DOCTYPE html><html><head><style>'
        + 'body{margin:0;height:100vh;display:flex;align-items:center;'
        + 'justify-content:center;background:#0f0505;color:#ff8888;'
        + 'font-family:monospace;font-size:14px;padding:24px;'
        + 'box-sizing:border-box;text-align:center;}'
        + '</style></head><body>' + _escAttr(text) + '</body></html>');
}

function closeRenderWindow() {
    renderWindow.classList.add('hidden');
    resetRenderSurfaces();
    if (_pdfBlobUrl) {
        try { URL.revokeObjectURL(_pdfBlobUrl); } catch (e) { /* ignore */ }
        _pdfBlobUrl = null;
    }
    currentRenderState = null;
}

// --- Close button ---
renderCloseBtn.addEventListener('click', closeRenderWindow);

// --- Download helper ---
if (renderDownloadBtn) {
    renderDownloadBtn.addEventListener('click', () => {
        if (!currentRenderState) return;

        const triggerDownload = async (url, filename) => {
            if (url.startsWith('blob:')) {
                const a = document.createElement('a');
                a.href = url;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            } else {
                try {
                    const response = await fetch(url);
                    const blob = await response.blob();
                    const objectUrl = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = objectUrl;
                    a.download = filename;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    setTimeout(() => URL.revokeObjectURL(objectUrl), 100);
                } catch (e) {
                    console.error("Download fallback", e);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = filename;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                }
            }
        };

        if (currentRenderState.type === 'image') {
            const url = currentRenderState.src;
            // The src is a blob: URL by the time we get here — the real
            // filename was stashed at open time from the /api/download
            // path (the old startsWith('http') branch never matched, so
            // every image saved as image.png).
            let filename = currentRenderState.name || 'image.png';
            if (!currentRenderState.name && url.startsWith('http') && url.includes('/')) {
                const parts = url.split('/');
                filename = parts[parts.length - 1].split('?')[0] || 'image.png';
            }
            triggerDownload(url, filename);
        } else if (currentRenderState.type === 'pdf') {
            // Was a silent no-op: type 'pdf' matched no branch.
            triggerDownload(currentRenderState.src,
                currentRenderState.name || 'report.pdf');
        } else if (currentRenderState.type === 'mermaid') {
            const svgElement = mermaidContainer.querySelector('svg');
            if (svgElement) {
                const serializer = new XMLSerializer();
                let source = serializer.serializeToString(svgElement);
                if (!source.match(/^<svg[^>]+xmlns="http\:\/\/www\.w3\.org\/2000\/svg"/)) {
                    source = source.replace(/^<svg/, '<svg xmlns="http://www.w3.org/2000/svg"');
                }
                const blob = new Blob([source], { type: "image/svg+xml;charset=utf-8" });
                const url = URL.createObjectURL(blob);
                triggerDownload(url, 'diagram.svg');
                setTimeout(() => URL.revokeObjectURL(url), 100);
            }
        } else if (currentRenderState.type === 'chart') {
            const canvas = document.getElementById('render-chart');
            if (canvas) {
                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = canvas.width;
                tempCanvas.height = canvas.height;
                const ctx = tempCanvas.getContext('2d');
                ctx.fillStyle = '#0f0505';
                ctx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
                ctx.drawImage(canvas, 0, 0);
                triggerDownload(tempCanvas.toDataURL("image/png"), 'chart.png');
            }
        } else if (currentRenderState.type === 'html') {
            const ext = currentRenderState.lang.replace('language-', '');
            const filename = `code.${(ext === 'javascript' || ext === 'js') ? 'js' : ext}`;
            let mimeType = 'text/plain;charset=utf-8';
            if (ext === 'html') mimeType = 'text/html;charset=utf-8';
            else if (ext === 'css') mimeType = 'text/css;charset=utf-8';
            else if (ext === 'javascript' || ext === 'js') mimeType = 'application/javascript;charset=utf-8';
            
            const blob = new Blob([currentRenderState.content], { type: mimeType });
            const url = URL.createObjectURL(blob);
            triggerDownload(url, filename);
            setTimeout(() => URL.revokeObjectURL(url), 100);
        }
    });
}

// --- Zoom helpers ---
function applyZoom() {
    const t = `scale(${currentZoom})`;
    renderIframe.style.transform = t;
    renderIframeSandboxed.style.transform = t;
    mermaidContainer.style.transform = t;
    chartContainer.style.transform = t;
    renderIframe.style.transformOrigin = 'center center';
    renderIframeSandboxed.style.transformOrigin = 'center center';
    mermaidContainer.style.transformOrigin = 'center center';
    chartContainer.style.transformOrigin = 'center center';
}

renderZoomIn.addEventListener('click', () => {
    currentZoom = Math.min(currentZoom + 0.1, 3.0);
    applyZoom();
});

renderZoomOut.addEventListener('click', () => {
    currentZoom = Math.max(currentZoom - 0.1, 0.3);
    applyZoom();
});

// --- Drag and Resize logic ---
(function initDragAndResize() {
    let isDragging = false;
    let startX, startY, origLeft, origTop;

    // --- DRAG ---
    renderHeader.addEventListener('mousedown', onDragStart);
    renderHeader.addEventListener('touchstart', onDragStart, { passive: false });

    function onDragStart(e) {
        // Ignore clicks on buttons inside the header
        if (e.target.closest('.render-btn')) return;
        isDragging = true;
        
        const clientX = e.type.includes('mouse') ? e.clientX : e.touches[0].clientX;
        const clientY = e.type.includes('mouse') ? e.clientY : e.touches[0].clientY;

        startX = clientX;
        startY = clientY;
        origLeft = renderWindow.offsetLeft;
        origTop = renderWindow.offsetTop;
        renderHeader.style.cursor = 'grabbing';
        // Prevent either iframe from swallowing the drag's pointer events.
        renderIframe.style.pointerEvents = 'none';
        renderIframeSandboxed.style.pointerEvents = 'none';

        if (e.cancelable) e.preventDefault();

        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', onMouseUp);
        document.addEventListener('touchmove', onMouseMove, { passive: false });
        document.addEventListener('touchend', onMouseUp);
    }

    function onMouseMove(e) {
        if (!isDragging) return;
        const clientX = e.type.includes('mouse') ? e.clientX : e.touches[0].clientX;
        const clientY = e.type.includes('mouse') ? e.clientY : e.touches[0].clientY;
        const dx = clientX - startX;
        const dy = clientY - startY;
        // Clamp so at least a grabbable sliver of the header stays
        // on-screen — a window dragged fully off-viewport had no recovery
        // path (no reset control, position not persisted).
        const w = renderWindow.offsetWidth;
        const newLeft = Math.min(Math.max(origLeft + dx, -(w - 60)),
            window.innerWidth - 60);
        const newTop = Math.min(Math.max(origTop + dy, 0),
            window.innerHeight - 44);
        renderWindow.style.left = newLeft + 'px';
        renderWindow.style.top = newTop + 'px';
        if (e.cancelable) e.preventDefault();
    }

    function onMouseUp() {
        if (isDragging) {
            isDragging = false;
            renderHeader.style.cursor = 'grab';
            renderIframe.style.pointerEvents = 'auto'; // Restore iframe interactions
            renderIframeSandboxed.style.pointerEvents = 'auto';
            document.removeEventListener('mousemove', onMouseMove);
            document.removeEventListener('mouseup', onMouseUp);
            document.removeEventListener('touchmove', onMouseMove);
            document.removeEventListener('touchend', onMouseUp);
        }
    }

    // --- RESIZE ---
    const resizer = document.createElement('div');
    resizer.style.position = 'absolute';
    resizer.style.bottom = '0';
    resizer.style.right = '0';
    resizer.style.width = '16px';
    resizer.style.height = '16px';
    resizer.style.cursor = 'nwse-resize';
    resizer.style.zIndex = '1000';
    // Match visual handle
    resizer.style.background = 'linear-gradient(135deg, transparent 50%, rgba(139, 0, 0, 0.4) 50%)';
    resizer.style.borderRadius = '0 0 12px 0';
    renderWindow.appendChild(resizer);

    let isResizing = false;
    let startW, startH, startResX, startResY;

    resizer.addEventListener('mousedown', onResizeStart);
    resizer.addEventListener('touchstart', onResizeStart, { passive: false });

    function onResizeStart(e) {
        isResizing = true;
        
        const clientX = e.type.includes('mouse') ? e.clientX : e.touches[0].clientX;
        const clientY = e.type.includes('mouse') ? e.clientY : e.touches[0].clientY;

        startResX = clientX;
        startResY = clientY;
        const style = window.getComputedStyle(renderWindow);
        startW = parseInt(style.width, 10);
        startH = parseInt(style.height, 10);
        renderIframe.style.pointerEvents = 'none'; // Prevent iframe event swallowing
        renderIframeSandboxed.style.pointerEvents = 'none';

        if (e.cancelable) e.preventDefault();
        e.stopPropagation();

        document.addEventListener('mousemove', onResizeMove);
        document.addEventListener('mouseup', onResizeUp);
        document.addEventListener('touchmove', onResizeMove, { passive: false });
        document.addEventListener('touchend', onResizeUp);
    }

    function onResizeMove(e) {
        if (!isResizing) return;
        const clientX = e.type.includes('mouse') ? e.clientX : e.touches[0].clientX;
        const clientY = e.type.includes('mouse') ? e.clientY : e.touches[0].clientY;
        const newWidth = startW + clientX - startResX;
        const newHeight = startH + clientY - startResY;
        renderWindow.style.width = Math.max(200, newWidth) + 'px';
        renderWindow.style.height = Math.max(150, newHeight) + 'px';
        if (e.cancelable) e.preventDefault();
    }

    function onResizeUp() {
        if (isResizing) {
            isResizing = false;
            renderIframe.style.pointerEvents = 'auto'; // Restore iframe interactions
            renderIframeSandboxed.style.pointerEvents = 'auto';
            document.removeEventListener('mousemove', onResizeMove);
            document.removeEventListener('mouseup', onResizeUp);
            document.removeEventListener('touchmove', onResizeMove);
            document.removeEventListener('touchend', onResizeUp);
        }
    }
})();

// ═══════════════════════════════════════════════════════════════
//  Render Buttons – "Visualize" on code blocks
// ═══════════════════════════════════════════════════════════════

function attachRenderButtons() {
    const pres = chatLog.querySelectorAll('pre');
    pres.forEach(pre => {
        // Skip if already processed
        if (pre.querySelector('.render-code-btn')) return;

        const code = pre.querySelector('code');
        if (!code) return;

        // Check if the code block has a renderable language class
        const lang = [...code.classList].find(c => RENDERABLE_LANGS.has(c));
        if (!lang) return;

        // Make the <pre> a positioning context
        pre.style.position = 'relative';
        pre.classList.add('renderable-hidden');

        const btn = document.createElement('button');
        btn.className = 'render-code-btn';
        btn.textContent = 'Visualize';
        pre.appendChild(btn);

        btn.addEventListener('click', () => {
            const codeText = code.textContent;

            resetRenderSurfaces();
            renderWindow.classList.remove('hidden');

            if (lang === 'language-mermaid') {
                renderMermaid(codeText);
            } else if (lang === 'language-csv') {
                renderCSV(codeText);
            } else {
                renderHTMLContent(codeText, lang);
            }
        });

        // Auto-open the visualizer for freshly-arrived blocks, but not
        // when we're rebuilding the log from history (that would spring
        // the window open on every page/workspace load).
        if (!_restoringHistory) btn.click();
    });
}

// --- Mermaid renderer --- (caller ran resetRenderSurfaces)
function renderMermaid(codeText) {
    currentRenderState = { type: 'mermaid' };
    mermaidContainer.style.display = 'flex';

    if (!window.mermaid) {
        mermaidContainer.innerHTML = `<pre style="color:#ff4444;">Diagram rendering unavailable (mermaid failed to load).</pre>`;
        return;
    }
    // Visible placeholder while the async render runs — the container was
    // just cleared by resetRenderSurfaces, so the PREVIOUS diagram can no
    // longer masquerade as this one during the wait.
    const wait = document.createElement('pre');
    wait.style.color = 'rgba(158,170,192,0.6)';
    wait.textContent = 'Rendering diagram…';
    mermaidContainer.appendChild(wait);
    mermaid.render('mermaid-graph-' + Date.now(), codeText).then(result => {
        mermaidContainer.innerHTML = result.svg;
    }).catch(err => {
        // textContent (not innerHTML) — the error string can echo fragments
        // of untrusted agent-authored diagram source (reflected XSS sink).
        mermaidContainer.replaceChildren();
        const pre = document.createElement('pre');
        pre.style.color = '#ff4444';
        pre.textContent = String(err);
        mermaidContainer.appendChild(pre);
    });
}

// --- HTML / CSS / JS renderer --- (caller ran resetRenderSurfaces)
function renderHTMLContent(codeText, lang) {
    currentRenderState = { type: 'html', content: codeText, lang: lang };
    // Agent code goes into the SANDBOXED frame only.
    renderIframeSandboxed.style.display = 'block';

    let html = codeText;
    if (lang === 'language-css') {
        html = `<!DOCTYPE html><html><head><style>${codeText}</style></head><body></body></html>`;
    } else if (lang === 'language-javascript' || lang === 'language-js') {
        html = `<!DOCTYPE html><html><head></head><body><script>${codeText}<\/script></body></html>`;
    }

    // srcdoc + sandbox="allow-scripts" (no allow-same-origin) runs the
    // agent code in an opaque origin: it can't reach window.parent, so it
    // can't read the injected GHOST_API_KEY or drive our /api/* calls.
    // Using srcdoc (not contentDocument.write) is what lets us drop
    // allow-same-origin — the parent never needs to touch the doc.
    renderIframeSandboxed.srcdoc = html;
}

// --- CSV / Chart renderer --- (caller ran resetRenderSurfaces)
function renderCSV(codeText) {
    // Vendored libs are local, but a failed load previously threw a raw
    // ReferenceError out of the click handler with the window already
    // open on stale content.
    if (!window.Papa || !window.Chart) {
        _showRenderError('Chart rendering unavailable (vendor library failed to load).');
        return;
    }
    currentRenderState = { type: 'chart' };
    chartContainer.style.display = 'block';

    const parsed = Papa.parse(codeText.trim(), {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true
    });

    const fields = parsed.meta.fields;
    if (!fields || fields.length < 2) {
        // Was a silent `return`: empty black panel, and Download handed
        // out a black PNG of the destroyed chart.
        chartContainer.style.display = 'none';
        _showRenderError('This CSV needs a header row and at least two columns to chart.');
        return;
    }

    const labelKey = fields[0];
    const labelsArray = parsed.data.map(row => row[labelKey]);

    // Palette for multiple datasets
    const palette = [
        'rgba(139, 0, 0, 0.8)',
        'rgba(0, 243, 255, 0.8)',
        'rgba(0, 255, 157, 0.8)',
        'rgba(189, 0, 255, 0.8)',
        'rgba(255, 170, 0, 0.8)',
        'rgba(255, 51, 102, 0.8)'
    ];

    const datasetsArray = fields.slice(1).map((key, i) => ({
        label: key,
        data: parsed.data.map(row => row[key]),
        backgroundColor: palette[i % palette.length],
        borderColor: palette[i % palette.length].replace('0.8', '1'),
        borderWidth: 1
    }));

    currentChart = new Chart(document.getElementById('render-chart'), {
        type: 'bar',
        data: { labels: labelsArray, datasets: datasetsArray },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            color: '#fff',
            plugins: {
                legend: { labels: { color: '#fff' } }
            },
            scales: {
                x: { ticks: { color: '#fff' }, grid: { color: 'rgba(255,255,255,0.08)' } },
                y: { ticks: { color: '#fff' }, grid: { color: 'rgba(255,255,255,0.08)' } }
            }
        }
    });
}

// Zoomed-image viewer via srcdoc. The src is interpolated through
// _escAttr, which closes the `" onload="alert(1)` attribute-breakout the
// old document.write/setAttribute approach existed to prevent — while
// being deterministic from ANY prior frame state (document.write can't
// follow a PDF src and raced the close handler's navigation).
function writeImageIntoIframe(iframe, imgSrc) {
    if (!iframe) return;
    _setMainFrameHTML(
        '<!DOCTYPE html><html><head><style>'
        + 'body{margin:0;display:flex;justify-content:center;align-items:center;'
        + 'height:100vh;background:#0f0505;}'
        + 'img{max-width:100%;max-height:100%;object-fit:contain;'
        + 'border-radius:8px;box-shadow:0 10px 40px rgba(0,0,0,0.5);}'
        + '</style></head><body>'
        + `<img src="${_escAttr(imgSrc)}">`
        + '</body></html>'
    );
}

function openImageInVisualizer(imgEl) {
    resetRenderSurfaces();
    currentRenderState = {
        type: 'image', src: imgEl.src,
        name: imgEl.dataset.ghostName || null,
    };
    writeImageIntoIframe(renderIframe, imgEl.src);
    renderWindow.classList.remove('hidden');
}

chatLog.addEventListener('click', (e) => {
    if (e.target.tagName === 'IMG' && e.target.closest('.message')) {
        openImageInVisualizer(e.target);
    }
});


// Auto-open images in Visualizer (Robust against streaming DOM destruction).
//
// The agent emits images as markdown pointing at `/api/download/<file>`,
// which is gated by `X-Ghost-Key`. Browsers don't attach custom headers to
// plain <img> loads, so the image 401s — both the chat thumbnail and the
// visualizer iframe render empty. Route the load through the authed fetch
// wrapper and swap to a blob URL before handing it to the iframe.
//
// Map is used as a simple LRU: re-setting a key moves it to the end of
// insertion order, and we evict from the front when we exceed the cap.
// Without eviction the cache leaks blob URLs indefinitely (long chats
// with many images exhaust browser memory).
const AUTHED_BLOB_CACHE_MAX = 100;
const _authedBlobCache = new Map();  // raw URL → blob object URL

function _evictAuthedBlobCache() {
    let guard = _authedBlobCache.size;
    while (_authedBlobCache.size > AUTHED_BLOB_CACHE_MAX && guard-- > 0) {
        const oldestKey = _authedBlobCache.keys().next().value;
        const oldestUrl = _authedBlobCache.get(oldestKey);
        _authedBlobCache.delete(oldestKey);
        // Never revoke the URL the visualizer is showing RIGHT NOW —
        // that blanked the open preview and killed its Download button.
        // Re-insert as most-recent and evict the next-oldest instead.
        if (currentRenderState && currentRenderState.src === oldestUrl) {
            _authedBlobCache.set(oldestKey, oldestUrl);
            continue;
        }
        try { URL.revokeObjectURL(oldestUrl); } catch (e) { /* ignore */ }
    }
}

async function _toAuthedBlobUrl(rawSrc) {
    if (!rawSrc || !rawSrc.includes('/api/download/')) return rawSrc;
    if (_authedBlobCache.has(rawSrc)) {
        // Touch: delete + re-set to mark as most recently used.
        const cached = _authedBlobCache.get(rawSrc);
        _authedBlobCache.delete(rawSrc);
        _authedBlobCache.set(rawSrc, cached);
        return cached;
    }
    const res = await fetch(rawSrc);
    if (!res.ok) throw new Error(`image fetch ${res.status}`);
    const blob = await res.blob();
    const blobUrl = URL.createObjectURL(blob);
    _authedBlobCache.set(rawSrc, blobUrl);
    _evictAuthedBlobCache();
    return blobUrl;
}

async function _handleChatImage(img) {
    // Capture the restore state synchronously, BEFORE any await. The
    // auto-open decision below happens after an async blob fetch, by
    // which point _restoringHistory's macrotask reset may already have
    // fired — so we can't read the live flag there.
    const restoringAtStart = _restoringHistory;
    let swappedToBlob = false;
    try {
        const rawSrc = img.getAttribute('src') || '';
        if (rawSrc.includes('/api/download/')) {
            // The original <img src="/api/download/..."> load starts the
            // moment marked inserts it into the DOM; the browser can't
            // attach the X-Ghost-Key header to a plain <img>, so that
            // initial load 401s. We rewrite src to an authed blob URL
            // that succeeds.
            //
            // Deliberately DO NOT attach an error listener for authed
            // images. Safari delivers the original 401 "error" event
            // asynchronously (sometimes after we've swapped src), and
            // blob URLs load from memory — they don't meaningfully
            // fail. Genuine fetch failures are already caught below
            // and surface via _showBrokenImagePlaceholder.
            const blobUrl = await _toAuthedBlobUrl(rawSrc);
            if (blobUrl !== rawSrc) {
                img.src = blobUrl;
                swappedToBlob = true;
                // Real filename for the visualizer's Download button — by
                // open time the src is an anonymous blob: URL.
                img.dataset.ghostName =
                    rawSrc.split('/').pop().split('?')[0] || '';
            }
        } else {
            // Non-API URL: browser loads it natively, no rewrite, so
            // surface genuine load failures up front.
            img.addEventListener('error', () => { _showBrokenImagePlaceholder(img); }, { once: true });
        }
    } catch (e) {
        // Authed fetch itself failed (401/404/network) — render the
        // broken badge directly instead of leaving the unresolved <img>.
        console.warn('Authed image fetch failed', e);
        _showBrokenImagePlaceholder(img);
        return;
    }

    // The img may have been destroyed mid-stream when the agent's
    // streaming content kept rebuilding innerHTML. If so, bail quietly
    // — the NEXT img the observer sees will get its own processing.
    if (!img.parentNode) return;

    const renderWindow = document.getElementById('render-window');
    const renderIframe = document.getElementById('render-iframe');
    const mermaidContainer = document.getElementById('mermaid-container');
    const chartContainer = document.getElementById('chart-container');

    // Only auto-open the visualizer for live images. When restoring
    // history we still swap to the blob URL and add the reopen button
    // below, but we don't pop the floating window open on page load.
    if (!restoringAtStart) {
        openImageInVisualizer(img);
    }

    const placeholder = document.createElement('button');
    placeholder.className = 'icon-btn';
    placeholder.textContent = '🖼️';
    placeholder.title = 'Open Image';
    placeholder.style.width = '100%';
    placeholder.style.borderRadius = '8px';
    placeholder.style.marginTop = '10px';
    placeholder.style.height = '40px';
    placeholder.onclick = () => { openImageInVisualizer(img); };
    // Insert BEFORE hiding the img so a concurrent error event (rare,
    // blob-decode failure) can't run `img.replaceWith(badge)` while
    // the placeholder button is still un-inserted — that would leave
    // the chat with only the badge and no way to reopen the image.
    img.parentNode.insertBefore(placeholder, img);
    img.classList.add('renderable-hidden-img');
    // Stash the button on the img so `_showBrokenImagePlaceholder` can
    // detect "this image was already successfully processed, don't
    // replace it" even if Safari dispatches a late spurious error.
    img._ghostPlaceholderBtn = placeholder;
}

// PDF report support.
//
// The agent's `report_pdf` tool returns markdown of the form
//   [📄 Title (PDF)](/api/download/report_xxx.pdf)
// which marked renders as a plain <a href="/api/download/...">. A plain
// navigation 401s (no X-Ghost-Key header), so we intercept the click and
// fetch the blob ourselves. History of this feature (2026-08-01 review):
// the first version nested an iframe inside a sandbox'd iframe — the
// plugins-sandbox flag is unclearable and inherited, so the PDF viewer
// was blocked in EVERY engine and the window opened onto nothing.
// Now: desktop gets the blob straight into the (unsandboxed, same-origin)
// #render-iframe — browsers render application/pdf blobs natively. iOS
// cannot render PDFs in ANY subframe (platform limit, not fixable with
// attributes), so it gets an in-window panel whose "Open PDF" link
// targets a new top-level tab, where Safari's native viewer works.
function _showPdfPanel(pdfBlobUrl, filename) {
    _setMainFrameHTML(
        '<!DOCTYPE html><html><head><style>'
        + 'body{margin:0;height:100vh;display:flex;flex-direction:column;gap:18px;'
        + 'align-items:center;justify-content:center;background:#0f0505;'
        + 'font-family:-apple-system,sans-serif;}'
        + 'a{color:#0ff;font-size:18px;text-decoration:none;border:1px solid #0aa;'
        + 'border-radius:10px;padding:14px 26px;}'
        + 'p{color:#9eaac0;font-size:13px;margin:0;}'
        + '</style></head><body>'
        + `<a href="${_escAttr(pdfBlobUrl)}" target="_blank" rel="noopener">📄 Open ${_escAttr(filename)}</a>`
        + '<p>PDF preview opens in a new tab on this device.</p>'
        + '</body></html>');
}

async function _handleChatPdfLink(link) {
    // Decorate the anchor with a clear PDF affordance so users know it's
    // a document deliverable, not a generic link. The label is built once
    // on first sight; reopens reuse the same DOM.
    const labelText = link.textContent && link.textContent.trim()
        ? link.textContent.trim()
        : 'Open PDF';
    if (!link.dataset.pdfDecorated) {
        link.dataset.pdfDecorated = '1';
        link.classList.add('pdf-link');
        link.title = 'Open PDF in visualizer (and download)';
    }

    link.addEventListener('click', async (ev) => {
        ev.preventDefault();
        const rawHref = link.getAttribute('href') || '';
        const filename = rawHref.split('/').pop().split('?')[0] || 'report.pdf';
        try {
            // Dedicated blob, NOT the image LRU: PDFs are multi-MB and a
            // 100-entry cache of them is a phone tab-kill. One live URL,
            // revoked on replace and on window close.
            const res = await fetch(rawHref || link.href);
            if (!res.ok) throw new Error(`PDF fetch ${res.status}`);
            const blob = await res.blob();
            if (_pdfBlobUrl) {
                try { URL.revokeObjectURL(_pdfBlobUrl); } catch (e) { /* ignore */ }
            }
            _pdfBlobUrl = URL.createObjectURL(blob);

            resetRenderSurfaces();
            currentRenderState = { type: 'pdf', src: _pdfBlobUrl, name: filename };
            if (isIOS) {
                _showPdfPanel(_pdfBlobUrl, filename);
            } else {
                renderIframe.removeAttribute('srcdoc');
                renderIframe.src = _pdfBlobUrl;   // native viewer
                renderIframe.style.display = 'block';
            }
            renderWindow.classList.remove('hidden');
            // No auto-download anymore: it prompted on every click (iOS)
            // and could fall outside the user-activation window. The
            // window's Download button covers keeping a copy.
        } catch (e) {
            console.warn('PDF fetch failed', e);
            resetRenderSurfaces();
            renderWindow.classList.remove('hidden');
            _showRenderError(`Could not load ${filename}: ${e.message || e}`);
        }
    });

    // If the link was rendered without visible text (rare — agent style
    // is `[📄 Title (PDF)](...)`, which always has text), set a sensible
    // label so it's still clickable.
    if (!link.textContent || !link.textContent.trim()) {
        link.textContent = `📄 ${labelText}`;
    }
}

function _processChatLogArtifacts() {
    document.querySelectorAll('#chat-log img').forEach(img => {
        if (img.classList.contains('placeholder-added')) return;
        img.classList.add('placeholder-added');
        // _handleChatImage owns the error-listener lifecycle — it has
        // to wait until AFTER the authed blob-URL swap to attach it,
        // so the expected 401 on the initial unauthed load doesn't
        // spuriously trigger the broken-image badge.
        _handleChatImage(img);
    });

    // Capture PDF download links emitted by tools/report_pdf.py.
    // Matches `/api/download/*.pdf` (case-insensitive) anywhere in the
    // href so we still hit nested or proxied URLs. Other file types
    // continue to use the browser's default <a> behaviour.
    document.querySelectorAll('#chat-log a[href*="/api/download/"]').forEach(link => {
        if (link.dataset.pdfHandled) return;
        const href = link.getAttribute('href') || '';
        if (!/\.pdf(\?|$)/i.test(href)) return;
        link.dataset.pdfHandled = '1';
        _handleChatPdfLink(link);
    });
}

const chatObserver = new MutationObserver(_processChatLogArtifacts);
const chatLogElement = document.getElementById('chat-log');
if (chatLogElement) {
    chatObserver.observe(chatLogElement, { childList: true, subtree: true });
    // A MutationObserver only reports FUTURE mutations, so chat history
    // restored by loadChatState() (which ran earlier, before this point)
    // was never processed — its images would 401 and show broken. Do one
    // initial pass over the already-present nodes. _restoringHistory is
    // still true here (its reset is a macrotask that runs after this
    // synchronous script), so restored images get the authed-blob swap
    // without auto-opening the visualizer.
    _processChatLogArtifacts();
}

function _showBrokenImagePlaceholder(img) {
    // Replace a failed <img> with a visible badge so users aren't staring
    // at a silent blank. Surfaces auth / 404 / network failures.
    if (img.dataset.brokenHandled) return;
    // Guard 1: blob URLs load synchronously from memory and don't fail
    // in practice. A spurious error on a blob src is almost always
    // Safari dispatching the original unauthed 401 late, after we've
    // already swapped — ignore it.
    if (img.src && img.src.startsWith('blob:')) return;
    // Guard 2: if the image was already decorated with a placeholder
    // button (meaning the authed flow completed successfully), the
    // user has a working way to reopen the visualizer — don't clobber.
    if (img._ghostPlaceholderBtn) return;
    img.dataset.brokenHandled = '1';
    const badge = document.createElement('div');
    badge.className = 'broken-image-badge';
    const name = (img.getAttribute('alt') || img.getAttribute('src') || 'image')
        .split('/').pop().split('?')[0].slice(0, 64);
    badge.textContent = `🖼️ image unavailable — ${name}`;
    img.replaceWith(badge);
}

// Release resources on page unload so zombies don't pile up across
// reloads. Closing the WebSocket lets the server drop the socket from
// `connected_websockets` immediately instead of waiting for the TCP RST.
// Disconnecting the MutationObserver is harmless during unload but helps
// in tests / reused windows.
// ⚠ `pagehide`, guarded on `!persisted` — NOT `beforeunload`.
//
// `beforeunload` does not block bfcache in Safari or Chrome, and this
// handler performs teardown that NOTHING restores: `chatObserver.observe()`
// runs exactly once, at module init, and neither pageshow handler
// re-observes. So opening a PDF and hitting Back left a page where every
// existing <img> pointed at a REVOKED blob (and could not be reprocessed,
// because `_processChatLogArtifacts` bails on the surviving
// `placeholder-added` class) and every FUTURE image or PDF link went
// unprocessed — a plain /api/download/ load that 401s with no broken-image
// badge (R2 lens B). The code's own `e.persisted` handling proves bfcache
// is expected here.
window.addEventListener('pagehide', (e) => {
    if (e.persisted) return;   // going into bfcache — the page comes BACK
    try { chatObserver.disconnect(); } catch (err) { /* ignore */ }
    try { if (ws) ws.close(); } catch (err) { /* ignore */ }
    // Revoke every cached blob URL so the browser releases the memory
    // promptly instead of waiting for GC on a navigated-away document.
    for (const url of _authedBlobCache.values()) {
        try { URL.revokeObjectURL(url); } catch (err) { /* ignore */ }
    }
    _authedBlobCache.clear();
});


// --- TTS Engine ---
const ttsToggleBtn = document.getElementById('tts-toggle-btn');
if (ttsToggleBtn) {
    ttsToggleBtn.addEventListener('click', () => {
        isTTSActive = !isTTSActive;
        ttsToggleBtn.classList.toggle('active', isTTSActive);
        
        if (isTTSActive) {
            // UNLOCK BROWSER AUTOPLAY via Web Audio API
            if (!audioCtx) {
                audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            }
            if (audioCtx.state === 'suspended') {
                audioCtx.resume();
            }
            // Play a silent buffer to explicitly register user interaction with Apple WebKit
            const silentBuffer = audioCtx.createBuffer(1, 1, 22050);
            const silentSource = audioCtx.createBufferSource();
            silentSource.buffer = silentBuffer;
            silentSource.connect(audioCtx.destination);
            silentSource.start(0);
        } else {
            stopTTS();
        }
    });
}

function stopTTS() {
    // ⚠ BUMP THE GENERATION FIRST. `processTTSFetch` shifts its text before
    // awaiting /api/tts, then pushes the blob and calls playNextAudio() —
    // against state this function just cleared. So an interrupted clip
    // played ANYWAY: the mic path's "full duplex interruption" is exactly
    // this, the agent talking over the user's recording (R3 lens B).
    _ttsGeneration += 1;
    // Revoke what we drop, or every interruption leaks a blob URL.
    for (const url of ttsAudioQueue) {
        try { URL.revokeObjectURL(url); } catch (e) { /* ignore */ }
    }
    ttsTextQueue = [];
    ttsAudioQueue = [];
    ttsBuffer = "";
    if (currentAudioSource) {
        try { currentAudioSource.stop(); } catch(e) {}
        try { currentAudioSource.disconnect(); } catch(e) {}
        try { currentAudioSource.pause(); } catch(e) {}
        currentAudioSource = null;
    }
    isPlayingTTS = false;
    isFetchingTTS = false;
    // Kill the analyser rAF loop and reset the sphere's audio level so
    // it decays immediately instead of drifting after a stop.
    if (typeof _stopTTSAudioPump === 'function') _stopTTSAudioPump();
}

// Bumped by stopTTS. Every await inside the TTS pipeline re-checks it, so
// an interrupted clip cannot resurrect itself after the queue was cleared.
let _ttsGeneration = 0;

function queueTTS(text) {
    ttsTextQueue.push(text);
    processTTSFetch();
}

async function processTTSFetch() {
    if (isFetchingTTS || ttsTextQueue.length === 0) return;
    isFetchingTTS = true;

    let text = ttsTextQueue.shift();
    const gen = _ttsGeneration;

    try {
        let res = await fetch('/api/tts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        });
        if (gen !== _ttsGeneration) {
            // stopTTS ran while this was in flight. Anything we do now
            // speaks over whatever replaced it.
            isFetchingTTS = false;
            return;
        }
        if (res.ok) {
            let blob = await res.blob();
            if (gen !== _ttsGeneration) { isFetchingTTS = false; return; }
            let audioUrl = URL.createObjectURL(blob);
            ttsAudioQueue.push(audioUrl);
            if (!isPlayingTTS) playNextAudio();
        } else {
            // Same as STT: /api/tts carries the reason (voice unavailable,
            // text too long, `say` timed out) and the user saw NOTHING at
            // all — console-only (R3 lens A).
            const _e = await _httpError(res, '🔊 Voice error');
            console.error("TTS Fetch Error:", _e.message);
            addMessage('system', _e.message);
        }
    } catch (e) {
        console.error("TTS Fetch Error:", e);
    }
    
    isFetchingTTS = false;
    // Recursively fetch the next chunk while audio is playing
    if (gen === _ttsGeneration && ttsTextQueue.length > 0) {
        processTTSFetch();
    }
}

// Route a freshly-created <audio> through the shared AudioContext with
// an AnalyserNode so the sphere visualizer can react to the voice.
// Fails safe: if any step throws (iOS Safari has been fussy about
// createMediaElementSource in the past), we silently fall back to
// plain element playback so TTS is never broken by the visualizer.
//
// Notes on Safari quirks:
//   * createMediaElementSource can only be called once per element.
//     We always pass a freshly-constructed Audio, so that's fine.
//   * Once an element is routed through AudioContext, its audio only
//     comes out via the context's destination — remember to connect
//     the analyser to audioCtx.destination.
//   * iOS requires the context to be resumed after a user gesture.
//     stopTTS/start paths already handle resume(); we defend again
//     here in case playback starts via a different code path.
let _ttsAnalyser = null;
let _ttsAudioDataArr = null;
let _ttsAudioRafId = null;

function _ensureTTSAnalyser() {
    if (_ttsAnalyser || !audioCtx) return _ttsAnalyser;
    try {
        _ttsAnalyser = audioCtx.createAnalyser();
        _ttsAnalyser.fftSize = 128;
        _ttsAudioDataArr = new Uint8Array(_ttsAnalyser.frequencyBinCount);
        _ttsAnalyser.connect(audioCtx.destination);
    } catch (e) {
        console.warn('AnalyserNode unavailable — audio-reactive sphere disabled', e);
        _ttsAnalyser = null;
    }
    return _ttsAnalyser;
}

function _pumpTTSAudioLevel() {
    if (!_ttsAnalyser || !_ttsAudioDataArr) return;
    _ttsAnalyser.getByteFrequencyData(_ttsAudioDataArr);
    let sum = 0;
    for (let i = 0; i < _ttsAudioDataArr.length; i++) sum += _ttsAudioDataArr[i];
    const avg = sum / (_ttsAudioDataArr.length * 255);  // 0..1
    if (activeFace && typeof activeFace.setAudioLevel === 'function') {
        activeFace.setAudioLevel(avg);
    }
    _ttsAudioRafId = requestAnimationFrame(_pumpTTSAudioLevel);
}

function _startTTSAudioPump() {
    if (_ttsAudioRafId !== null) return;
    _ttsAudioRafId = requestAnimationFrame(_pumpTTSAudioLevel);
}
function _stopTTSAudioPump() {
    if (_ttsAudioRafId !== null) {
        cancelAnimationFrame(_ttsAudioRafId);
        _ttsAudioRafId = null;
    }
    if (activeFace && typeof activeFace.setAudioLevel === 'function') {
        activeFace.setAudioLevel(0);
    }
}

function playNextAudio() {
    if (isPlayingTTS || ttsAudioQueue.length === 0) return;
    isPlayingTTS = true;

    let audioUrl = ttsAudioQueue.shift();

    currentAudioSource = new Audio(audioUrl);

    const cleanupAndNext = () => {
        URL.revokeObjectURL(audioUrl);
        isPlayingTTS = false;
        // Only halt the pump when the queue is fully drained — back-to-back
        // clips should keep the visualizer hot without resetting to 0.
        if (ttsAudioQueue.length === 0) _stopTTSAudioPump();
        playNextAudio();
    };

    currentAudioSource.onended = cleanupAndNext;
    currentAudioSource.onerror = cleanupAndNext;

    // Best-effort analyser wiring. Any failure here must not block
    // playback — the sphere visualizer is purely cosmetic.
    let wiredAnalyser = false;
    try {
        if (audioCtx && audioCtx.state === 'suspended') audioCtx.resume();
        const analyser = _ensureTTSAnalyser();
        if (analyser) {
            const src = audioCtx.createMediaElementSource(currentAudioSource);
            src.connect(analyser);
            wiredAnalyser = true;
            _startTTSAudioPump();
        }
    } catch (e) {
        // iOS Safari occasionally rejects createMediaElementSource for
        // cross-origin-ish blob URLs; swallow and play the element
        // natively. Next clip will try again fresh.
        console.warn('TTS analyser wiring skipped:', e && e.message);
    }

    currentAudioSource.play().catch(e => {
        console.error("Audio playback rejected:", e);
        cleanupAndNext();
    });
}

// --- Push-To-Talk Audio Engine (Mic) ---
const micBtn = document.getElementById('mic-btn');
let isRecording = false;

if (micBtn) {
    const startRecording = async (e) => {
        if (e.cancelable) e.preventDefault();
        if (isRecording) return;

        stopTTS(); // Full Duplex Interruption

        // Voice in → voice out. Holding the mic is both the intent signal
        // ("I'm hands-free") and the user gesture browsers require before
        // they will play audio, so we unlock autoplay HERE rather than
        // behind a separate TTS button (the header has no room for one —
        // see index.html). Typing a message switches speech back off.
        isTTSActive = true;
        try {
            if (!audioCtx) {
                audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            }
            if (audioCtx.state === 'suspended') audioCtx.resume();
            // A silent buffer is what actually registers the interaction
            // with Apple WebKit; resume() alone is not enough on iOS.
            const silentBuffer = audioCtx.createBuffer(1, 1, 22050);
            const silentSource = audioCtx.createBufferSource();
            silentSource.buffer = silentBuffer;
            silentSource.connect(audioCtx.destination);
            silentSource.start(0);
        } catch (_) { /* autoplay unlock is best-effort; STT still works */ }

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            addMessage('system', '🎙️ Error: Microphone access requires HTTPS or localhost context.');
            return;
        }

        // iOS Safari can silently hang on getUserMedia — give it a 5s ceiling.
        let permTimeout;
        try {
            const permPromise = navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: false,
                    sampleRate: 16000,
                    channelCount: 1
                }
            });
            const timeoutPromise = new Promise((_, reject) => {
                permTimeout = setTimeout(() => reject(new Error('Permission timeout')), 5000);
            });
            const stream = await Promise.race([permPromise, timeoutPromise]);
            clearTimeout(permTimeout);

            // iOS Safari does not encode audio/webm — pick a supported MIME type.
            const candidates = [
                'audio/mp4',
                'audio/mp4;codecs=mp4a.40.2',
                'audio/aac',
                'audio/webm;codecs=opus',
                'audio/webm',
                'audio/wav'
            ];
            let chosenMime = '';
            if (window.MediaRecorder && MediaRecorder.isTypeSupported) {
                chosenMime = candidates.find(t => MediaRecorder.isTypeSupported(t)) || '';
            }
            mediaRecorder = chosenMime
                ? new MediaRecorder(stream, { mimeType: chosenMime })
                : new MediaRecorder(stream);
            audioChunks = [];

            mediaRecorder.ondataavailable = event => {
                if (event.data.size > 0) audioChunks.push(event.data);
            };

            mediaRecorder.onstop = async () => {
                const blobType = mediaRecorder.mimeType || chosenMime || 'audio/webm';
                const audioBlob = new Blob(audioChunks, { type: blobType });
                const ext = blobType.includes('mp4') || blobType.includes('aac') ? 'm4a'
                    : blobType.includes('wav') ? 'wav' : 'webm';
                addMessage('system', '🎙️ Uploading audio for STT...');
                const formData = new FormData();
                formData.append('file', audioBlob, `voice_record.${ext}`);
                
                try {
                    // (Second orphaned updateActivityIcon call removed
                    // 2026-08-02 — see the upload path. This one was invisible
                    // while the mic button was deleted; restoring the button
                    // revived it and it broke STT with "can't find variable".)
                    activeFace.setWorkingState(true);

                    const res = await fetch('/api/stt', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (res.ok) {
                        const data = await res.json();
                        if (data.text) {
                            chatInput.value = data.text;
                            sendMessage();
                        } else {
                            addMessage('system', '🎙️ STT transcribed nothing.');
                        }
                    } else {
                        // voice.py raises real diagnoses ("'ffmpeg' not
                        // found on PATH… set GHOST_FFMPEG_BIN", "Audio is
                        // 22.3 min; the limit is 15 min", "thinking tokens
                        // consumed the entire budget — raise
                        // GHOST_STT_MAX_TOKENS") and stt_proxy returns each
                        // verbatim. Printing only the status made "the mic
                        // does nothing" indistinguishable from "the audio
                        // node is down" (R3 lens A).
                        const _e = await _httpError(res, '🎙️ STT Error');
                        addMessage('system', _e.message);
                    }
                } catch (err) {
                    addMessage('system', '🎙️ STT Upload Failed: ' + err.message);
                } finally {
                    activeFace.setWorkingState(false);
                }
                
                stream.getTracks().forEach(track => track.stop());
            };
            
            mediaRecorder.start();
            isRecording = true;
            micBtn.classList.add('recording');
            addMessage('system', '🎙️ Recording started...');
        } catch (err) {
            clearTimeout(permTimeout);
            let msg = err.message || String(err);
            if (err.name === 'NotAllowedError' || err.name === 'SecurityError') {
                msg = 'Microphone access denied. On iOS check Settings > Safari > Microphone.';
            } else if (err.name === 'NotFoundError') {
                msg = 'No microphone found on this device.';
            } else if (msg === 'Permission timeout') {
                msg = 'Microphone permission timed out. Try again.';
            }
            addMessage('system', '🎙️ ' + msg);
        }
    };

    const stopRecording = (e) => {
        if (e && e.cancelable) e.preventDefault();
        if (!isRecording || !mediaRecorder) return;
        
        mediaRecorder.stop();
        isRecording = false;
        micBtn.classList.remove('recording');
    };

    micBtn.addEventListener('mousedown', startRecording);
    micBtn.addEventListener('mouseup', stopRecording);
    micBtn.addEventListener('mouseleave', stopRecording);

    // For mobile
    micBtn.addEventListener('touchstart', startRecording, {passive: false});
    micBtn.addEventListener('touchend', stopRecording, {passive: false});
    micBtn.addEventListener('touchcancel', stopRecording, {passive: false});
    // iOS dispatches a context menu on long-press — suppress it for the mic button.
    micBtn.addEventListener('contextmenu', (e) => e.preventDefault());
}

// ═══════════════════════════════════════════════════════════════
//  Workspace bridge (2026-07-28)
//
//  app.js predates ES-module exports; the operator-console features
//  (sessions rail, notifications, status panel, command palette) live
//  in separate modules that talk to this core through window.GhostCore.
//  New functionality belongs in modules — this bridge is the seam, not
//  a dumping ground.
// ═══════════════════════════════════════════════════════════════
window.GhostCore = {
    events: new EventTarget(),
    addMessage,
    renderMarkdown,
    renderHistoryToLog,
    decorateCodeBlocks,
    decorateMessageActions,
    attachRenderButtons,
    stripInternalTags: _stripInternalTags,
    safeStorage,
    scrollToBottom,
    renderEmptyStateHero,
    dismissEmptyStateHero,
    activeFace,
    stopTTS,
    sendMessage,
    // Cancel THIS TAB's turn (resolving its request id first). Exposed so
    // the palette can stop a turn without the blind "whatever holds the
    // semaphore" cancel it used to issue (R3 lens B).
    cancelOwnTurn: (hard) => {
        if (currentChatController) { try { currentChatController.abort(); } catch (e) {} }
        return _cancelAgentTurn(currentReqId, _lastSentUserText, hard);
    },
    getChatHistory: () => chatHistory,
    setChatHistory: (h) => { chatHistory = Array.isArray(h) ? h : []; saveChatState(); },
    // Wire-shape helpers for modules that fetch/compare server histories
    // (sessions.js): strip client-only label keys for comparison, and
    // preserve them across adoption. See toWireMessage/mergeClientLabelKeys.
    toWireMessage,
    mergeClientLabelKeys,
    // Reset the visible conversation WITHOUT touching the session store —
    // the sessions module decides what happens next (new id, load, …).
    clearConversation: () => {
        chatLog.innerHTML = '';
        chatHistory = [];
        safeStorage.remove('ghost_chat_history');
        // Close the visualizer BEFORE revoking — it may be displaying one
        // of these blobs (session switches used to blank the open preview).
        closeRenderWindow();
        for (const url of _authedBlobCache.values()) {
            try { URL.revokeObjectURL(url); } catch (e) { /* ignore */ }
        }
        _authedBlobCache.clear();
        updateWorkspaceBtnState();
        renderEmptyStateHero();
    },
    isProcessing: () => isProcessingRequest,
    elements: { chatLog, chatInput, logsBtn },
    toggleLogConsole: () => { if (logsBtn) logsBtn.click(); },
};

import('./workspace.js?v=7.6').catch(e => {
    // ⚠ VISIBLE, not console-only. This module owns the sessions rail, and
    // with it `window.__ghostSessionId` — so when it fails to load, every
    // turn silently reverts to CLIENT-CARRIED history: no durable session,
    // no rail, no bell, no status strip, and nothing on screen saying so.
    // A stale `?v=` in a cached index.html is enough to cause it, and the
    // operator's only clue was an empty sidebar. Durable sessions are the
    // thing this console exists for; losing them cannot be a console.warn.
    console.warn('[Ghost] workspace modules failed to load:', e);
    try {
        addMessage('system',
            '⚠ Console modules failed to load — this tab has NO durable ' +
            'session (history is client-carried only), no sessions rail and ' +
            'no notifications. Hard-reload to retry; if it persists the ' +
            'static assets are stale or unreachable.');
    } catch (_) { /* addMessage unavailable — the warn above is all we have */ }
});

