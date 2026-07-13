import * as matrixGraphFace from './matrix_graph.js?v=3.4';

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
const activityIcon = document.getElementById('activity-icon');
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
const ICON_CLASS = {
    // --- lifecycle / status (accent) ---
    '⚡': 'accent',   // SYSTEM_BOOT
    '🚀': 'accent',   // SYSTEM_READY
    '💤': 'idle',     // SYSTEM_SHUT
    '😴': 'idle',
    '🎬': 'accent',   // REQ_START
    '🏁': 'accent',   // REQ_DONE
    '⏳': 'accent',   // REQ_WAIT
    '✅': 'accent',   // OK
    '❌': 'accent',   // FAIL
    '⚠️': 'accent',   // WARN
    '🛑': 'accent',   // STOP
    '🔄': 'accent',   // RETRY
    '💡': 'accent',   // IDEA
    '🐛': 'accent',   // BUG
    '🛡️': 'accent',   // SHIELD

    // --- tools (external action) ---
    '🌐': 'tool',     // TOOL_SEARCH
    '🔬': 'tool',     // TOOL_DEEP
    '🐍': 'tool',     // TOOL_CODE
    '🐚': 'tool',     // TOOL_SHELL
    '⬇️': 'tool',     // TOOL_DOWN
    '🎨': 'tool',     // Image Gen
    '📥': 'tool',     // Docker pull
    '📦': 'tool',     // Package install
    '⚙️': 'tool',     // Sandbox init

    // --- memory / filesystem ---
    '💾': 'memory',   // TOOL_FILE_W
    '📖': 'memory',   // TOOL_FILE_R
    '🔍': 'memory',   // TOOL_FILE_S
    '👀': 'memory',   // TOOL_FILE_I
    '📝': 'memory',   // MEM_SAVE
    '🔎': 'memory',   // MEM_READ
    '📍': 'memory',   // MEM_MATCH
    '📚': 'memory',   // MEM_INGEST
    '✂️': 'memory',   // MEM_SPLIT / CUT
    '🧬': 'memory',   // MEM_EMBED
    '🧹': 'memory',   // MEM_WIPE
    '🗒️': 'memory',   // MEM_SCRATCH
    '👤': 'memory',   // USER_ID

    // --- planning / routing ---
    '📋': 'plan',     // BRAIN_PLAN
    '🧩': 'plan',     // BRAIN_CTX
    '🧭': 'plan',     // BRAIN_ROUTE
    '🎯': 'plan',     // BRAIN_AIM

    // --- raw thinking (the floor) ---
    '🧠': 'think',    // BRAIN_SUM — the one that was drowning everything
    '💭': 'think',    // BRAIN_THINK
    '🗣️': 'think',    // LLM_ASK
    '🤖': 'think',    // LLM_REPLY

    // --- misc ---
    '🫥': 'accent',   // MODE_GHOST
    '🐘': 'memory',   // POSTGRES
    '🔥': 'accent',
};

// Minimum time an icon of a given class stays "locked" against lower
// priorities. So: a 🐍 (tool) sticks for 4s even if a 🧠 (think) arrives
// 500 ms later — after 4s, think can take over. Accent icons flash but
// don't lock (their job is to briefly assert, not dominate).
const ICON_DWELL_MS = {
    accent: 1400,
    tool:   4500,
    memory: 3000,
    plan:   2000,
    think:   600,
    idle:    800,
};

// Numeric priority for comparisons. Higher = harder to preempt.
const ICON_PRIORITY = {
    accent: 5, tool: 4, memory: 3, plan: 2, think: 1, idle: 0,
};

function _iconClass(icon) { return ICON_CLASS[icon] || 'think'; }
function _iconPriority(icon) { return ICON_PRIORITY[_iconClass(icon)]; }
function _iconDwell(icon) { return ICON_DWELL_MS[_iconClass(icon)]; }

// Used elsewhere (setWorkingState gating). "Working" = the agent is
// actively doing something a user cares to watch; true for everything
// except idle / SYSTEM_SHUT.
const WORKING_ICONS = new Set(Object.keys(ICON_CLASS).filter(i =>
    ICON_CLASS[i] !== 'idle'
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
let currentThinkingInterval = null;
let currentTTSMutedLength = 0;

const SEND_SVG = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>`;
const CANCEL_SVG = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="9" y1="9" x2="15" y2="15"></line><line x1="15" y1="9" x2="9" y2="15"></line></svg>`;

function toggleSendButtonUI(isProcessing) {
    if (sendBtn) {
        if (isProcessing) {
            sendBtn.innerHTML = CANCEL_SVG;
            sendBtn.classList.add('send-btn-cancel');
        } else {
            sendBtn.innerHTML = SEND_SVG;
            sendBtn.classList.remove('send-btn-cancel');
        }
    }
}

// --- Notification & Service Worker Setup ---
// iOS Safari: Notification API and Web Push are only available for installed PWAs.
// Skip on iOS unless running in standalone (home-screen) mode.
const isStandalonePWA = window.matchMedia('(display-mode: standalone)').matches ||
    window.navigator.standalone === true;
const notificationsSupported = ('Notification' in window) && (!isIOS || isStandalonePWA);

if ('serviceWorker' in navigator && !isIOS) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js')
            .then(reg => console.log('Service Worker registered successfully'))
            .catch(err => console.error('Service Worker registration failed:', err));
    });
}

document.addEventListener('click', () => {
    if (notificationsSupported && Notification.permission === "default") {
        try {
            const result = Notification.requestPermission();
            if (result && typeof result.catch === 'function') {
                result.catch(e => console.error('Notification permission error:', e));
            }
        } catch (e) {
            console.error('Notification permission error:', e);
        }
    }
}, { once: true });

async function notifyUser(message) {
    if (!notificationsSupported) return;
    if (Notification.permission !== "granted" || !document.hidden) return;

    try {
        if ('serviceWorker' in navigator) {
            const registration = await navigator.serviceWorker.ready;
            registration.showNotification("Ghost System", {
                body: message,
                icon: "/static/cyber_face.png"
            });
        } else {
            new Notification("Ghost System", { body: message, icon: "/static/cyber_face.png" });
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

function connectWebSocket() {
    setConnectionState('pending', 'CONNECTING…');
    ws = new WebSocket(wsUrl);
    ws.onopen = () => {
        setConnectionState('online', 'SYSTEM ONLINE');
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
                    updateActivityIcon(icon);
                    updateStateFromIcon(icon);
                    // Genuine failures only: ⚠️ warnings are routine (node
                    // fallbacks, heals) and kept tinting the face on
                    // ordinary healthy turns.
                    if (['❌', '🛑', '🔥'].includes(icon)) activeFace.triggerSpike();
                }
                flashActivityIcon();
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
        setTimeout(connectWebSocket, 3000);
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
});

// VisualViewport: track virtual keyboard height so the input area stays visible on iOS.
if ('visualViewport' in window) {
    const updateKeyboardOffset = () => {
        const vv = window.visualViewport;
        const keyboardHeight = Math.max(0, window.innerHeight - vv.height - vv.offsetTop);
        document.documentElement.style.setProperty('--keyboard-height', `${keyboardHeight}px`);
    };
    window.visualViewport.addEventListener('resize', updateKeyboardOffset);
    window.visualViewport.addEventListener('scroll', updateKeyboardOffset);
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
// tint. Jewel-tone family (2026-07-13, matching matrix_graph's palette
// wheel): saturated but dark-mid — these BLEND into the sphere's
// accent, so a full neon here would drag the whole theme bright again.
const _ICON_CLASS_COLOR = {
    accent_ok:   '#128a52',  // ✅ emerald
    accent_err:  '#a3123a',  // ❌ 🛑 ⚠️ crimson red
    accent:      '#2f55d4',  // ⚡ 🚀 🎬 🏁 electric blue
    tool:        '#6d28d9',  // vivid violet
    memory:      '#2f7d4a',  // sea green
    plan:        '#3644b8',  // indigo
    think:       '#5b21b6',  // violet
    idle:        '#232734',  // near-black slate
};

function getIconColor(icon) {
    // Error/success get special sub-colors — they're visually the
    // most meaningful accent events the user can see flash by.
    if (icon === '✅') return _ICON_CLASS_COLOR.accent_ok;
    if (icon === '❌' || icon === '🛑' || icon === '⚠️' || icon === '🔥') {
        return _ICON_CLASS_COLOR.accent_err;
    }
    return _ICON_CLASS_COLOR[_iconClass(icon)] || _ICON_CLASS_COLOR.think;
}

// ═══════════════════════════════════════════════════════════════
//  Activity icon — priority-gated replacement
//
//  Previously this function referenced a `resuming` variable that
//  only existed inside sendMessage's closure, so every call from the
//  WebSocket log handler threw ReferenceError (silently caught by
//  ws.onmessage). The activity icon was effectively only ever
//  updated from sendMessage, which hardcodes '🧠' — that's why the
//  user saw 🧠 almost all the time regardless of agent activity.
//
//  Replacement is priority-gated by dwell time: a high-priority icon
//  (tool, memory) locks the slot for its dwell window so a stream of
//  ubiquitous low-priority 🧠 logs can't clobber it before the user
//  sees it. Same-class icons always replace (so 🐍 → 🐚 swaps promptly).
// ═══════════════════════════════════════════════════════════════

let iconHideTimeout;
let _currentMainIcon = '';
let _currentMainClass = 'think';
let _currentMainSetAt = 0;

function updateActivityIcon(icon) {
    if (!activityIcon || !icon) return;

    const now = performance.now();
    const newClass = _iconClass(icon);
    const newPri = _iconPriority(icon);
    const curPri = ICON_PRIORITY[_currentMainClass] || 0;
    const elapsed = now - _currentMainSetAt;
    const dwell = _iconDwell(_currentMainIcon) || 0;

    // Priority gate:
    //   * Empty slot → always take it.
    //   * Same-or-higher priority → take it.
    //   * Lower priority → only after the current icon's dwell expired.
    let accept;
    if (!_currentMainIcon) accept = true;
    else if (newPri >= curPri) accept = true;
    else accept = elapsed >= dwell;

    if (!accept) return;

    _currentMainIcon = icon;
    _currentMainClass = newClass;
    _currentMainSetAt = now;
    activityIcon.textContent = icon;
    activityIcon.style.opacity = '1';
    // Color the main glow to match the class so the user can learn
    // the palette (magenta=tool, orange=memory, cyan=think, etc.).
    activityIcon.style.filter = `drop-shadow(0 0 15px ${getIconColor(icon)})`;
    clearTimeout(iconHideTimeout);

    if (!isProcessingRequest) {
        // Idle auto-fade: long enough that a WORKING icon is still
        // visible after the burst, short enough that stale icons
        // don't linger forever. No more 60s-on-a-🧠 dominance.
        const timeoutDuration = dwell + 1500;
        iconHideTimeout = setTimeout(() => {
            if (!isProcessingRequest) {
                activityIcon.style.opacity = '0';
                setTimeout(() => {
                    if (activityIcon.style.opacity === '0') {
                        activityIcon.textContent = '';
                        activityIcon.style.opacity = '1';
                        _currentMainIcon = '';
                        _currentMainClass = 'think';
                    }
                }, 300);
            }
        }, timeoutDuration);
    }
}

let workTimer;
function updateStateFromIcon(icon) {
    if (isProcessingRequest) return; // Prevent logs from turning off the active state during a request

    if (WORKING_ICONS.has(icon)) {
        activeFace.setWorkingState(true);
        if (activityIcon) activityIcon.classList.add('working');
        clearTimeout(workTimer);
        workTimer = setTimeout(() => {
            if (!isProcessingRequest) {
                activeFace.setWorkingState(false);
                if (activityIcon) activityIcon.classList.remove('working');
            }
        }, 60000);
    } else if (IDLE_ICONS.has(icon)) {
        activeFace.setWorkingState(false);
        if (activityIcon) activityIcon.classList.remove('working');
        clearTimeout(workTimer);
    }
}

let iconTimeout;
function flashActivityIcon() {
    if (activityIcon && !activityIcon.classList.contains('working')) {
        activityIcon.style.transform = "scale(1.2)";
        clearTimeout(iconTimeout);
        iconTimeout = setTimeout(() => { activityIcon.style.transform = "scale(1)"; }, 150);
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

        const header = document.createElement('div');
        header.className = 'code-header';

        const badge = document.createElement('span');
        badge.className = 'code-lang';
        badge.textContent = lang;
        header.appendChild(badge);

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
        header.appendChild(copyBtn);

        pre.insertBefore(header, pre.firstChild);
    });
}

// Empty-state hero: greet the user and suggest a few starter prompts
// instead of leaving the chat log as a silent black rectangle. Clicking
// a chip drops it into the input so the user can edit before sending.
const EXAMPLE_PROMPTS = [
    'What are you capable of?',
    'Summarise the latest files in my sandbox.',
    'Write a python script to sort CSV by a column.',
];
function renderEmptyStateHero() {
    if (!chatLog) return;
    // Don't stack heroes if one's already there.
    if (chatLog.querySelector('.empty-hero')) return;
    const hero = document.createElement('div');
    hero.className = 'empty-hero';
    const title = document.createElement('div');
    title.className = 'empty-hero-title';
    title.textContent = 'GHOST AGENT';
    const sub = document.createElement('div');
    sub.className = 'empty-hero-sub';
    sub.textContent = 'Ask anything. Use /clear to reset the session.';
    const chips = document.createElement('div');
    chips.className = 'empty-hero-chips';
    for (const p of EXAMPLE_PROMPTS) {
        const chip = document.createElement('button');
        chip.type = 'button';
        chip.className = 'empty-hero-chip';
        chip.textContent = p;
        chip.addEventListener('click', () => {
            chatInput.value = p;
            chatInput.focus();
            // Fire input so auto-expand runs.
            chatInput.dispatchEvent(new Event('input'));
        });
        chips.appendChild(chip);
    }
    hero.appendChild(title);
    hero.appendChild(sub);
    hero.appendChild(chips);
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

function addMessage(role, text) {
    dismissEmptyStateHero();
    const div = document.createElement('div');
    div.className = `message ${role}`;
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
    scrollToBottom();
    return div;
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

function loadChatState() {
    const saved = safeStorage.get('ghost_chat_history');
    if (saved) {
        try {
            chatHistory = JSON.parse(saved);
            _withHistoryRestore(() => {
                for (const msg of chatHistory) {
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
                            chatLog.appendChild(div);
                        }
                    }
                }
                scrollToBottom();
                if (typeof attachRenderButtons === 'function') attachRenderButtons();
            });
        } catch (e) {
            console.error("Failed to load chat history", e);
        }
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

async function sendMessage(isResume = false) {
    const resuming = isResume === true;
    const text = chatInput.value.trim();
    if (!resuming && (!text || isProcessingRequest)) return;

    if (text === '/clear' && !resuming) {
        chatInput.value = '';
        chatInput.style.height = '';
        chatLog.innerHTML = '';
        chatHistory = [];
        safeStorage.remove('ghost_chat_history');
        // Flush the authed-image blob cache too — the thumbnails those
        // URLs pointed at are gone from the DOM, so holding on to the
        // blobs is pure leak.
        for (const url of _authedBlobCache.values()) {
            try { URL.revokeObjectURL(url); } catch (e) { /* ignore */ }
        }
        _authedBlobCache.clear();
        if (typeof updateWorkspaceBtnState === 'function') updateWorkspaceBtnState();
        renderEmptyStateHero();
        const msg = addMessage('system', 'Context cleared');
        setTimeout(() => { msg.remove(); }, 2000);
        return;
    }

    if (!resuming) {
        chatInput.value = '';
        chatInput.style.height = ''; // Clear inline height → rows="1" takes over
        addMessage('user', text);
        
        currentTaskId = null;
        currentChunkIndex = 0;
        currentAccumulatedContent = "";
        
        chatHistory.push({ role: "user", content: text });
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
        scrollToBottom();
        currentThinkingInterval = null;
    } else {
        addMessage('system', 'Reconnected directly to Ghost Server.');
        setTimeout(scrollToBottom, 100);
    }

    // Explicitly lock the blob into an active state
    isProcessingRequest = true;
    toggleSendButtonUI(true);
    activeFace.setWorkingState(true);
    // Only flip the dot to 'busy' if the WS is actually up — otherwise
    // the 'error' state should stay so the user sees the real problem.
    if (ws && ws.readyState === WebSocket.OPEN) {
        setConnectionState('busy', 'PROCESSING…');
    }
    if (activityIcon && !resuming) {
        clearTimeout(iconHideTimeout);
        // Reset the priority gate so this turn's first WS event can
        // take over immediately — otherwise the 🧠 "think" floor we
        // set here would lock for its dwell time and suppress the
        // real first-event icon (🧭 route, 📋 plan, etc.).
        _currentMainIcon = '';
        _currentMainClass = 'think';
        _currentMainSetAt = 0;
        updateActivityIcon('🧠');
        activityIcon.classList.add('working');
    }

    try {
        // Deliberately omit `model`: the agent validates a supplied model
        // name against its single configured model and returns 404
        // ModelNotFound on any mismatch (see api/routes.py). Pinning a
        // name here means every model rename on the agent silently breaks
        // chat with a 404. With no `model`, the agent uses its configured
        // model (`requested_model or configured_model`), so the client
        // always tracks whatever the agent is running.
        const payload = { messages: chatHistory, stream: true };
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
            response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
                signal: currentChatController.signal
            });
            if (response.headers.has('X-Task-ID')) {
                currentTaskId = response.headers.get('X-Task-ID');
            }
        }

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.error || `HTTP ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let streamBuffer = "";

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

                    if (data.choices && data.choices[0] && data.choices[0].delta && data.choices[0].delta.content) {
                        chunkContent = data.choices[0].delta.content;
                    } else if (data.message && data.message.content) {
                        // Fallback for non-streaming formats that might be wrapped
                        chunkContent = data.message.content;
                    } else if (data.error) {
                        addMessage('system', `Error: ${data.error}`);
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
        // dropped, so a plain push is correct in both cases.
        if (currentAccumulatedContent) {
            chatHistory.push({ role: "assistant", content: currentAccumulatedContent });
        } else {
            currentAgentMessageDiv.textContent = "No response";
            chatHistory.push({ role: "assistant", content: "No response" });
        }
        if (typeof saveChatState === 'function') saveChatState();
        if (typeof notifyUser === 'function') notifyUser("Response complete.");
        currentTaskId = null; // Clear task id

        if (typeof updateWorkspaceBtnState === 'function') updateWorkspaceBtnState();
    } catch (e) {
        if (e.name === 'AbortError') {
            if (currentThinkingInterval) { clearInterval(currentThinkingInterval); currentThinkingInterval = null; }
            if (currentTaskId) {
                fetch(`/api/chat/cancel/${currentTaskId}`, { method: 'POST' }).catch(()=>{});
            }
            if (!resuming && currentAccumulatedContent === "" && currentAgentMessageDiv) currentAgentMessageDiv.remove();
            if (currentAccumulatedContent !== "") {
                chatHistory.push({ role: "assistant", content: currentAccumulatedContent + "\n\n*[Aborted]*" });
                addMessage('system', 'Request cancelled by user.');
            } else {
                addMessage('system', 'Request cancelled by user.');
            }
            currentTaskId = null;
            if (typeof saveChatState === 'function') saveChatState();
        } else {
            if (currentThinkingInterval) { clearInterval(currentThinkingInterval); currentThinkingInterval = null; }
            
            const errMsg = e.message.toLowerCase();
            if (errMsg.includes('load failed') || errMsg.includes('networkerror') || errMsg.includes('fetch')) {
                addMessage('system', 'Safari suspended the UI. Detached Ghost continues calculation...');
                if (document.visibilityState === 'visible') {
                    setTimeout(() => sendMessage(true), 1000);
                } else {
                    document.addEventListener('visibilitychange', function resumeOnVisible() {
                        if (document.visibilityState === 'visible') {
                            document.removeEventListener('visibilitychange', resumeOnVisible);
                            setTimeout(() => sendMessage(true), 1000);
                        }
                    });
                }
            } else {
                addMessage('system', `Network Error: ${e.message}`);
                activeFace.triggerSpike();
            }
        }
    } finally {
        if (currentThinkingInterval) { clearInterval(currentThinkingInterval); currentThinkingInterval = null; }
        // Cancel any throttled render still queued for a future frame —
        // otherwise it fires after this point and re-adds `streaming`.
        _cancelScheduledStreamRender();
        // Drop the streaming cursor glyph and any stale "thinking" class
        // so the message renders as a completed reply.
        if (currentAgentMessageDiv) {
            currentAgentMessageDiv.classList.remove('streaming', 'thinking');
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
        if (ws && ws.readyState === WebSocket.OPEN) {
            setConnectionState('online', 'SYSTEM ONLINE');
        }
        if (activityIcon && !resuming) {
            activityIcon.classList.remove('working');
            updateActivityIcon('✅');
        }
        setTimeout(scrollToBottom, 100);

        // Auto-hide planner monologue 2 seconds after reply
        monologueTimeout = setTimeout(hidePlannerMonologue, 2000);

        // Attach "Visualize" buttons to any renderable code blocks
        attachRenderButtons();
    }
}

sendBtn.addEventListener('click', () => {
    if (isProcessingRequest && currentChatController) {
        currentChatController.abort();
        toggleSendButtonUI(false);
    } else if (!isProcessingRequest) {
        sendMessage();
    }
});

chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { 
        e.preventDefault(); 
        if (!isProcessingRequest) {
            sendMessage();
        }
    }
});

if (fullscreenBtn) {
    fullscreenBtn.addEventListener('click', () => { document.body.classList.toggle('zen-mode'); });
}

// Global toggle for Zen Mode (Persistent Key)
document.addEventListener('keydown', (e) => {
    // Toggle with 'Z' unless user is typing in the chat input
    if (e.key.toLowerCase() === 'z' && document.activeElement !== chatInput) {
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
            
            try {
                const response = await fetch('/api/workspace/save', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ chat_history: chatHistory })
                });
                
                if (!response.ok) throw new Error('Failed to save workspace');
                
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
            }
        }
    });

    workspaceUploadInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        workspaceUploadInput.value = '';
        
        isProcessingRequest = true;
        activeFace.setWorkingState(true);
        addMessage('system', `Loading workspace from ${file.name}...`);
        
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch('/api/workspace/load', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) throw new Error(`Load failed with status ${response.status}`);
            
            const result = await response.json();
            if (result.error) throw new Error(result.error);
            
            chatHistory = result.chat_history || [];
            chatLog.innerHTML = '';

            _withHistoryRestore(() => {
                for (const msg of chatHistory) {
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
                            chatLog.appendChild(div);
                        }
                    }
                }
                if (typeof attachRenderButtons === 'function') attachRenderButtons();
            });

            addMessage('system', 'Workspace loaded successfully.');
            updateWorkspaceBtnState();

        } catch (error) {
            addMessage('system', `Workspace Load Error: ${error.message}`);
            activeFace.triggerSpike();
        } finally {
            isProcessingRequest = false;
            activeFace.setWorkingState(false);
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

        addMessage('system', `Uploading ${file.name} to sandbox...`);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Upload failed with status ${response.status}`);
            }

            const result = await response.json();
            if (result.error) {
                addMessage('system', `Upload Failed: ${result.error}`);
                activeFace.triggerSpike();
            } else {
                addMessage('system', `Successfully uploaded ${file.name}.`);
                updateActivityIcon('✅');
            }
        } catch (error) {
            addMessage('system', `Upload Error: ${error.message}`);
            activeFace.triggerSpike();
        } finally {
            isProcessingRequest = false;
            activeFace.setWorkingState(false);
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
                .then(res => {
                    if (!res.ok) throw new Error(`HTTP ${res.status}`);
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
    window.visualViewport.addEventListener('resize', () => {
        scrollToBottom();
        document.body.style.height = window.visualViewport.height + 'px';
        window.scrollTo(0, 0);
    });
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
loadChatState();
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

// --- Close button ---
renderCloseBtn.addEventListener('click', () => {
    renderWindow.classList.add('hidden');
    renderIframe.src = 'about:blank';
    // Tear down any agent code so it stops running while hidden.
    renderIframeSandboxed.removeAttribute('srcdoc');
    renderIframeSandboxed.style.display = 'none';
    mermaidContainer.innerHTML = '';
    if (currentChart) {
        currentChart.destroy();
        currentChart = null;
    }
    currentZoom = 1.0;
    currentRenderState = null;
    applyZoom();
});

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
            let filename = 'image.png';
            if (url.startsWith('http') && url.includes('/')) {
                const parts = url.split('/');
                filename = parts[parts.length - 1].split('?')[0] || 'image.png';
            }
            triggerDownload(url, filename);
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
        renderWindow.style.left = (origLeft + dx) + 'px';
        renderWindow.style.top = (origTop + dy) + 'px';
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

            // Show window & reset zoom
            renderWindow.classList.remove('hidden');
            currentZoom = 1.0;
            applyZoom();

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

// --- Mermaid renderer ---
function renderMermaid(codeText) {
    currentRenderState = { type: 'mermaid' };
    renderIframe.style.display = 'none';
    renderIframeSandboxed.style.display = 'none';
    chartContainer.style.display = 'none';
    mermaidContainer.style.display = 'flex';

    if (!window.mermaid) {
        mermaidContainer.innerHTML = `<pre style="color:#ff4444;">Diagram rendering unavailable (mermaid failed to load).</pre>`;
        return;
    }
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

// --- HTML / CSS / JS renderer ---
function renderHTMLContent(codeText, lang) {
    currentRenderState = { type: 'html', content: codeText, lang: lang };
    mermaidContainer.style.display = 'none';
    chartContainer.style.display = 'none';
    // Hide the same-origin iframe; agent code goes into the sandboxed one.
    renderIframe.style.display = 'none';
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

// --- CSV / Chart renderer ---
function renderCSV(codeText) {
    currentRenderState = { type: 'chart' };
    mermaidContainer.style.display = 'none';
    renderIframe.style.display = 'none';
    renderIframeSandboxed.style.display = 'none';
    chartContainer.style.display = 'block';

    if (currentChart) {
        currentChart.destroy();
        currentChart = null;
    }

    const parsed = Papa.parse(codeText.trim(), {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true
    });

    const fields = parsed.meta.fields;
    if (!fields || fields.length < 2) return;

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

// Stamp the iframe with a zoomed-image viewer. Builds the image via the
// iframe's own document APIs so the src is set through setAttribute
// (which escapes attribute context) rather than string interpolation.
// String templates that inject `src="${url}"` break out of the attribute
// on any embedded quote — `" onload="alert(1)` was the intended vector.
function writeImageIntoIframe(iframe, imgSrc) {
    if (!iframe) return;
    // The sandboxed code iframe may still be visible from a prior
    // "Visualize" — hide it so the image preview isn't covered.
    if (renderIframeSandboxed) renderIframeSandboxed.style.display = 'none';
    iframe.contentDocument.open();
    iframe.contentDocument.write(
        '<!DOCTYPE html><html><head><style>'
        + 'body{margin:0;display:flex;justify-content:center;align-items:center;'
        + 'height:100vh;background:#0f0505;}'
        + 'img{max-width:100%;max-height:100%;object-fit:contain;'
        + 'border-radius:8px;box-shadow:0 10px 40px rgba(0,0,0,0.5);}'
        + '</style></head><body></body></html>'
    );
    iframe.contentDocument.close();
    const doc = iframe.contentDocument;
    const img = doc.createElement('img');
    img.setAttribute('src', imgSrc);
    doc.body.appendChild(img);
}

chatLog.addEventListener('click', (e) => {
    if (e.target.tagName === 'IMG' && e.target.closest('.message')) {
        currentRenderState = { type: 'image', src: e.target.src };
        const renderWindow = document.getElementById('render-window');
        const renderIframe = document.getElementById('render-iframe');
        const mermaidContainer = document.getElementById('mermaid-container');
        const chartContainer = document.getElementById('chart-container');

        renderWindow.classList.remove('hidden');
        if (typeof currentZoom !== 'undefined') { currentZoom = 1.0; applyZoom(); }
        mermaidContainer.style.display = 'none';
        chartContainer.style.display = 'none';
        renderIframe.style.display = 'block';

        writeImageIntoIframe(renderIframe, e.target.src);
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
    while (_authedBlobCache.size > AUTHED_BLOB_CACHE_MAX) {
        const oldestKey = _authedBlobCache.keys().next().value;
        const oldestUrl = _authedBlobCache.get(oldestKey);
        _authedBlobCache.delete(oldestKey);
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
        if (mermaidContainer) mermaidContainer.style.display = 'none';
        if (chartContainer) chartContainer.style.display = 'none';
        if (renderIframe) renderIframe.style.display = 'block';

        if (typeof currentZoom !== 'undefined') { currentZoom = 1.0; applyZoom(); }
        currentRenderState = { type: 'image', src: img.src };

        writeImageIntoIframe(renderIframe, img.src);
        if (renderWindow) renderWindow.classList.remove('hidden');
    }

    const placeholder = document.createElement('button');
    placeholder.className = 'icon-btn';
    placeholder.textContent = '🖼️';
    placeholder.title = 'Open Image';
    placeholder.style.width = '100%';
    placeholder.style.borderRadius = '8px';
    placeholder.style.marginTop = '10px';
    placeholder.style.height = '40px';
    placeholder.onclick = () => {
        currentRenderState = { type: 'image', src: img.src };
        if (renderWindow) renderWindow.classList.remove('hidden');
        if (typeof currentZoom !== 'undefined') { currentZoom = 1.0; applyZoom(); }
        if (mermaidContainer) mermaidContainer.style.display = 'none';
        if (chartContainer) chartContainer.style.display = 'none';
        if (renderIframe) {
            renderIframe.style.display = 'block';
            writeImageIntoIframe(renderIframe, img.src);
        }
    };
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

// PDF report support (mirrors the image flow).
//
// The agent's `report_pdf` tool returns markdown of the form
//   [📄 Title (PDF)](/api/download/report_xxx.pdf)
// which marked renders as a plain <a href="/api/download/...">. Clicking
// that link navigates the browser straight at the proxy URL, which 401s
// because plain navigations don't carry the X-Ghost-Key header. We
// intercept the click, fetch via the authed wrapper, and:
//   1. open the resulting blob URL in the right-rail visualizer iframe
//      (browsers render application/pdf blobs natively), and
//   2. ALSO offer it as a true download via a hidden anchor.
function _writePdfIntoIframe(iframe, pdfBlobUrl) {
    if (!iframe) return;
    if (renderIframeSandboxed) renderIframeSandboxed.style.display = 'none';
    iframe.contentDocument.open();
    iframe.contentDocument.write(
        '<!DOCTYPE html><html><head><style>'
        + 'html,body{margin:0;height:100%;background:#0f0505;}'
        + 'iframe{border:0;width:100%;height:100%;}'
        + '</style></head><body></body></html>'
    );
    iframe.contentDocument.close();
    const inner = iframe.contentDocument.createElement('iframe');
    inner.setAttribute('src', pdfBlobUrl);
    iframe.contentDocument.body.appendChild(inner);
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
        try {
            const blobUrl = await _toAuthedBlobUrl(link.getAttribute('href') || link.href);

            // 1) Inline preview in the right-rail iframe.
            const renderWindow = document.getElementById('render-window');
            const renderIframe = document.getElementById('render-iframe');
            const mermaidContainer = document.getElementById('mermaid-container');
            const chartContainer = document.getElementById('chart-container');
            if (mermaidContainer) mermaidContainer.style.display = 'none';
            if (chartContainer) chartContainer.style.display = 'none';
            if (renderIframe) {
                renderIframe.style.display = 'block';
                _writePdfIntoIframe(renderIframe, blobUrl);
            }
            if (renderWindow) renderWindow.classList.remove('hidden');
            currentRenderState = { type: 'pdf', src: blobUrl };

            // 2) Trigger a real download alongside the preview so the
            //    user gets a file they can keep, not just a viewer tab.
            const a = document.createElement('a');
            a.href = blobUrl;
            const rawHref = link.getAttribute('href') || '';
            const filename = rawHref.split('/').pop().split('?')[0] || 'report.pdf';
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        } catch (e) {
            console.warn('PDF fetch failed', e);
            addMessage('system', `Could not load PDF: ${e.message || e}`);
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
window.addEventListener('beforeunload', () => {
    try { chatObserver.disconnect(); } catch (e) { /* ignore */ }
    try { if (ws) ws.close(); } catch (e) { /* ignore */ }
    // Revoke every cached blob URL so the browser releases the memory
    // promptly instead of waiting for GC on a navigated-away document.
    for (const url of _authedBlobCache.values()) {
        try { URL.revokeObjectURL(url); } catch (e) { /* ignore */ }
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

function queueTTS(text) {
    ttsTextQueue.push(text);
    processTTSFetch();
}

async function processTTSFetch() {
    if (isFetchingTTS || ttsTextQueue.length === 0) return;
    isFetchingTTS = true;
    
    let text = ttsTextQueue.shift();
    
    try {
        let res = await fetch('/api/tts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        });
        if (res.ok) {
            let blob = await res.blob();
            let audioUrl = URL.createObjectURL(blob);
            ttsAudioQueue.push(audioUrl);
            if (!isPlayingTTS) playNextAudio();
        } else {
            console.error("TTS Fetch Error:", res.status);
        }
    } catch (e) {
        console.error("TTS Fetch Error:", e);
    }
    
    isFetchingTTS = false;
    // Recursively fetch the next chunk while audio is playing
    if (ttsTextQueue.length > 0) {
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
                    activeFace.setWorkingState(true);
                    updateActivityIcon('🧠');
                    
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
                        addMessage('system', '🎙️ STT Error: HTTP ' + res.status);
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

