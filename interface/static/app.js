import * as imageFace from './image_face.js';
import * as matrixGraphFace from './matrix_graph.js';

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

const faces = { image: imageFace, matrix: matrixGraphFace };
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
const wsUrl = `ws://${window.location.host}/ws`;

const WORKING_ICONS = new Set(['🧠', '🔍', '⚙️', '🔨', '⚡', '💡', '📡', '💾', '🛡️', '🔑', '🔓', '🚀', '🔮', '🧬', '🔬', '🔭', '🩺', '🧩', '📈', '📊', '📋']);
const IDLE_ICONS = new Set(['✅', '❌', '🛑', '😴', '💤']);

let isProcessingRequest = false;

function connectWebSocket() {
    ws = new WebSocket(wsUrl);
    ws.onopen = () => {
        if (statusText) statusText.textContent = "SYSTEM ONLINE";
        if (connectionDot) {
            connectionDot.style.boxShadow = "0 0 10px #00ff9d";
            connectionDot.style.backgroundColor = "#00ff9d";
        }
    };
    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.type === 'log') {
                const icon = extractIcon(data.content);
                const flashColor = getIconColor(icon);

                const isMonologue = data.content.includes("PLANNER MONOLOGUE");
                if (isMonologue) {
                    activeFace.triggerSmallPulse();
                } else {
                    activeFace.triggerPulse(flashColor); // Always heartbeat on new log
                }

                if (icon) {
                    updateActivityIcon(icon);
                    updateStateFromIcon(icon);
                    if (['❌', '🛑', '⚠️', '🔥'].includes(icon)) activeFace.triggerSpike();
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
        if (statusText) statusText.textContent = "DISCONNECTED";
        if (connectionDot) {
            connectionDot.style.backgroundColor = "#ff2a2a";
            connectionDot.style.boxShadow = "none";
        }
        setTimeout(connectWebSocket, 3000);
    };
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

function getIconColor(icon) {
    if (['🧠', '💡', '🔮', '🧬', '🧩'].includes(icon)) return '#00f3ff';
    if (['✅', '🔧', '🔨', '⚙️', '🛡️', '🔓'].includes(icon)) return '#00ff9d';
    if (['🔍', '💾', '📈', '📊', '📋', '🔑'].includes(icon)) return '#ffaa00';
    if (['📡', '⚡', '🚀', '🔭'].includes(icon)) return '#1e90ff';
    if (['❌', '🛑', '⚠️', '🔥'].includes(icon)) return '#ff2a2a';
    if (['😴', '💤', '🩺', '🔬'].includes(icon)) return '#ffffff';
    return '#bd00ff';
}

let iconHideTimeout;
function updateActivityIcon(icon) {
    if (activityIcon) {
        activityIcon.textContent = icon;
        activityIcon.style.opacity = '1';
        clearTimeout(iconHideTimeout);

        if (!isProcessingRequest) {
            let timeoutDuration = WORKING_ICONS.has(icon) ? 60000 : 2000;
            iconHideTimeout = setTimeout(() => {
                if (!isProcessingRequest) {
                    activityIcon.style.opacity = '0';
                    setTimeout(() => {
                        // Ensure we don't clear an icon that just faded back in
                        if (activityIcon.style.opacity === '0') {
                            activityIcon.textContent = '';
                            activityIcon.style.opacity = '1';
                        }
                    }, 300);
                }
            }, timeoutDuration);
        }
    }
}

let workTimer;
function updateStateFromIcon(icon) {
    if (isProcessingRequest) return; // Prevent logs from turning off the active state during a request

    if (WORKING_ICONS.has(icon)) {
        activeFace.setWorkingState(true);
        activeFace.setWaitingState(true);
        if (activityIcon) activityIcon.classList.add('working');
        clearTimeout(workTimer);
        workTimer = setTimeout(() => {
            if (!isProcessingRequest) {
                activeFace.setWorkingState(false);
                activeFace.setWaitingState(false);
                if (activityIcon) activityIcon.classList.remove('working');
            }
        }, 60000);
    } else if (IDLE_ICONS.has(icon)) {
        activeFace.setWorkingState(false);
        activeFace.setWaitingState(false);
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

function addMessage(role, text) {
    const div = document.createElement('div');
    div.className = `message ${role}`;
    // Use marked.parse if available (it is added in index.html)
    if (window.marked) {
        div.innerHTML = marked.parse(text);
    } else {
        div.textContent = text;
    }
    chatLog.appendChild(div);
    scrollToBottom();
    return div;
}

function scrollToBottom() {
    requestAnimationFrame(() => { chatLog.scrollTo({ top: chatLog.scrollHeight, behavior: 'smooth' }); });
}

// Auto-expand textarea height organically
chatInput.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
    if (this.value === '') this.style.height = 'auto';
});

async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text || isProcessingRequest) return;

    chatInput.value = '';
    chatInput.style.height = 'auto'; // Reset height perfectly
    addMessage('user', text);

    if (text === '/clear') {
        chatLog.innerHTML = '';
        chatHistory = [];
        const msg = addMessage('system', 'Context cleared');
        setTimeout(() => { msg.remove(); }, 2000);

        return;
    }

    // Explicitly lock the blob into an active state
    isProcessingRequest = true;
    activeFace.setWorkingState(true);
    activeFace.setWaitingState(true);
    if (activityIcon) {
        clearTimeout(iconHideTimeout);
        activityIcon.style.opacity = '1';
        activityIcon.textContent = '🧠';
        activityIcon.classList.add('working');
    }

    try {
        chatHistory.push({ role: "user", content: text });
        const payload = { model: "qwen-3.5-9b", messages: chatHistory, stream: true };

        // Inject an empty message div for the agent's upcoming response
        const agentMessageDiv = addMessage('agent', 'Thinking.');
        let accumulatedContent = "";

        // Add animated thinking dots
        let dotCount = 1;
        const thinkingInterval = setInterval(() => {
            dotCount = (dotCount % 3) + 1;
            agentMessageDiv.textContent = 'Thinking' + '.'.repeat(dotCount);
        }, 400);

        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

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
                        if (accumulatedContent === "") {
                            clearInterval(thinkingInterval);
                            agentMessageDiv.textContent = ""; // Clear 'Thinking...'
                        }

                        const isAtBottom = Math.abs(chatLog.scrollHeight - chatLog.scrollTop - chatLog.clientHeight) <= 50;

                        accumulatedContent += chunkContent;

                        // --- Voice Intercept Logic ---
                        if (isTTSActive) {
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

                        let displayContent = accumulatedContent.replace(/<tool_call[\s\S]*?(?:<\/tool_call>|$)/gi, '').trim();

                        if (window.marked) {
                            agentMessageDiv.innerHTML = marked.parse(displayContent);
                        } else {
                            agentMessageDiv.textContent = displayContent;
                        }

                        if (isAtBottom) scrollToBottom();
                    }
                } catch (e) {
                    console.warn("Failed to parse SSE chunk:", dataStr, e);
                }
            }
        }

        // Push the final concatenated message to chat history
        if (accumulatedContent) {
            chatHistory.push({ role: "assistant", content: accumulatedContent });
        } else {
            agentMessageDiv.textContent = "No response";
            chatHistory.push({ role: "assistant", content: "No response" });
        }

    } catch (e) {
        chatHistory.pop();
        addMessage('system', `Network Error: ${e.message}`);
        activeFace.triggerSpike();
    } finally {
        if (typeof thinkingInterval !== 'undefined') clearInterval(thinkingInterval);

        if (isTTSActive && ttsBuffer.trim().length > 0) {
            let finalClean = ttsBuffer.replace(/(<think>[\s\S]*?<\/think>|<tool_call[\s\S]*?(?:<\/tool_call>|$)|<[^>]+>|\*|_|`|#|\[|\]|\(|\)|!\[.*?\]\(.*?\))/gi, "").trim();
            if (/[\p{L}\p{N}]/u.test(finalClean)) queueTTS(finalClean);
            ttsBuffer = "";
        }

        isProcessingRequest = false;
        activeFace.setWorkingState(false);
        activeFace.setWaitingState(false);
        if (activityIcon) {
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

sendBtn.addEventListener('click', sendMessage);
chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
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
        activeFace.setWaitingState(true);

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
            activeFace.setWaitingState(false);
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
            fetch(url)
                .then(res => res.blob())
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
                    console.error("Download fallback", err);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = filename.trim();
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
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

activeFace.init();
connectWebSocket();

// ═══════════════════════════════════════════════════════════════
//  Render Window – Visualizer Logic
// ═══════════════════════════════════════════════════════════════

// --- Mermaid init ---
mermaid.initialize({ startOnLoad: false, theme: 'dark' });

// --- Element references ---
const renderWindow = document.getElementById('render-window');
const renderHeader = document.getElementById('render-header');
const renderIframe = document.getElementById('render-iframe');
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
    mermaidContainer.style.transform = t;
    chartContainer.style.transform = t;
    renderIframe.style.transformOrigin = 'center center';
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
        renderIframe.style.pointerEvents = 'none'; // Prevent iframe event swallowing
        
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

const RENDERABLE_LANGS = new Set([
    'language-html', 'language-css', 'language-javascript', 'language-js',
    'language-mermaid', 'language-csv'
]);

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

        // Auto-open
        btn.click();
    });
}

// --- Mermaid renderer ---
function renderMermaid(codeText) {
    currentRenderState = { type: 'mermaid' };
    renderIframe.style.display = 'none';
    chartContainer.style.display = 'none';
    mermaidContainer.style.display = 'flex';

    mermaid.render('mermaid-graph-' + Date.now(), codeText).then(result => {
        mermaidContainer.innerHTML = result.svg;
    }).catch(err => {
        mermaidContainer.innerHTML = `<pre style="color:#ff4444;">${err}</pre>`;
    });
}

// --- HTML / CSS / JS renderer ---
function renderHTMLContent(codeText, lang) {
    currentRenderState = { type: 'html', content: codeText, lang: lang };
    mermaidContainer.style.display = 'none';
    chartContainer.style.display = 'none';
    renderIframe.style.display = 'block';

    let html = codeText;
    if (lang === 'language-css') {
        html = `<!DOCTYPE html><html><head><style>${codeText}</style></head><body></body></html>`;
    } else if (lang === 'language-javascript' || lang === 'language-js') {
        html = `<!DOCTYPE html><html><head></head><body><script>${codeText}<\/script></body></html>`;
    }

    renderIframe.contentDocument.open();
    renderIframe.contentDocument.write(html);
    renderIframe.contentDocument.close();
}

// --- CSV / Chart renderer ---
function renderCSV(codeText) {
    currentRenderState = { type: 'chart' };
    mermaidContainer.style.display = 'none';
    renderIframe.style.display = 'none';
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

        // Use the iframe to securely display the zoomed image
        let html = `<!DOCTYPE html><html><head><style>body { margin: 0; display: flex; justify-content: center; align-items: center; height: 100vh; background: #0f0505; } img { max-width: 100%; max-height: 100%; object-fit: contain; }</style></head><body><img src="${e.target.src}"></body></html>`;
        renderIframe.contentDocument.open();
        renderIframe.contentDocument.write(html);
        renderIframe.contentDocument.close();
    }
});


// Auto-open images in Visualizer (Robust against streaming DOM destruction)
const chatObserver = new MutationObserver(() => {
    document.querySelectorAll('#chat-log img').forEach(img => {
        if (!img.classList.contains('placeholder-added')) {
            img.classList.add('placeholder-added');

            const renderWindow = document.getElementById('render-window');

            const renderIframe = document.getElementById('render-iframe');
            const mermaidContainer = document.getElementById('mermaid-container');
            const chartContainer = document.getElementById('chart-container');

            if (mermaidContainer) mermaidContainer.style.display = 'none';
            if (chartContainer) chartContainer.style.display = 'none';
            if (renderIframe) renderIframe.style.display = 'block';

            if (typeof currentZoom !== 'undefined') { currentZoom = 1.0; applyZoom(); }
            currentRenderState = { type: 'image', src: img.src };

            let html = `<!DOCTYPE html><html><head><style>body { margin: 0; display: flex; justify-content: center; align-items: center; height: 100vh; background: #0f0505; } img { max-width: 100%; max-height: 100%; object-fit: contain; border-radius: 8px; box-shadow: 0 10px 40px rgba(0,0,0,0.5); }</style></head><body><img src="${img.src}"></body></html>`;
            if (renderIframe) {
                renderIframe.contentDocument.open();
                renderIframe.contentDocument.write(html);
                renderIframe.contentDocument.close();
            }
            if (renderWindow) renderWindow.classList.remove('hidden');

            // Add a placeholder to re-open easily if closed
            const placeholder = document.createElement('button');
            placeholder.className = 'icon-btn';
            placeholder.innerHTML = '🖼️';
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
                if (renderIframe) renderIframe.style.display = 'block';
                if (renderIframe) {
                    renderIframe.contentDocument.open();
                    renderIframe.contentDocument.write(html);
                    renderIframe.contentDocument.close();
                }
            };
            img.parentNode.insertBefore(placeholder, img);
            img.classList.add('renderable-hidden-img');
        }
    });
});
const chatLogElement = document.getElementById('chat-log');
if (chatLogElement) chatObserver.observe(chatLogElement, { childList: true, subtree: true });

document.addEventListener('DOMContentLoaded', () => {
    const faceSelector = document.getElementById('face-selector');
    if (faceSelector) {
        faceSelector.value = 'matrix';
        faceSelector.addEventListener('change', (e) => {
            if (activeFace && activeFace.destroy) {
                activeFace.destroy();
            } else {
                const container = document.getElementById('sphere-container');
                if (container) container.innerHTML = '';
            }
            activeFace = faces[e.target.value];
            if (activeFace && activeFace.init) activeFace.init();
        });
    }
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
        currentAudioSource = null;
    }
    isPlayingTTS = false;
    isFetchingTTS = false;
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
            let arrayBuffer = await res.arrayBuffer();
            if (!audioCtx) {
                audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            }
            try {
                // Safari WebKit requires the older callback syntax for decodeAudioData sometimes
                audioCtx.decodeAudioData(arrayBuffer, function(buffer) {
                    ttsAudioQueue.push(buffer);
                    playNextAudio();
                }, function(err) {
                    addMessage('system', `Safari Decoder Error: ${err}`);
                    console.error("Safari Audio Decode Error:", err);
                });
            } catch (err) {
                addMessage('system', `WebAudio Exception: ${err}`);
            }
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

function playNextAudio() {
    if (isPlayingTTS || ttsAudioQueue.length === 0) return;
    isPlayingTTS = true;
    
    let audioBuffer = ttsAudioQueue.shift();
    
    currentAudioSource = audioCtx.createBufferSource();
    currentAudioSource.buffer = audioBuffer;
    currentAudioSource.connect(audioCtx.destination);
    
    const cleanupAndNext = () => {
        try { currentAudioSource.disconnect(); } catch(e) {}
        currentAudioSource = null;
        isPlayingTTS = false;
        playNextAudio();
    };

    currentAudioSource.onended = cleanupAndNext;
    
    try {
        currentAudioSource.start(0);
    } catch (e) {
        console.error("Audio playback rejected:", e);
        cleanupAndNext();
    }
}

// --- STT Engine (Mic) ---
const micBtn = document.getElementById('mic-btn');
if (micBtn) {
    const startRecording = async (e) => {
        if(e.cancelable) e.preventDefault();
        stopTTS(); // Full Duplex Interruption
        
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            addMessage('system', '🎙️ Error: Microphone access requires HTTPS or localhost context.');
            return;
        }

        micBtn.classList.add('recording');
        try {
            let stream;
            try {
                stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: false,
                        autoGainControl: false,
                        sampleRate: 16000
                    } 
                });
            } catch (err) {
                console.warn("High-fidelity audio failed (likely iOS Safari), falling back to baseline constraints:", err);
                stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            }
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
            mediaRecorder.onstop = async () => {
                micBtn.classList.remove('recording');
                const audioBlob = new Blob(audioChunks);
                audioChunks = [];
                stream.getTracks().forEach(t => t.stop());
                
                const formData = new FormData();
                formData.append('file', audioBlob, 'voice_memo.webm');
                
                const sysMsg = addMessage('system', '🎙️ Transcribing audio...');
                let res = await fetch('/api/stt', { method: 'POST', body: formData });
                let data = await res.json();
                sysMsg.remove();
                console.log("STT Output:", data);
                
                if (data.text) {
                    chatInput.value = data.text;
                    sendMessage(); // Auto-send
                } else if (data.error) {
                    addMessage('system', `STT Error: ${data.error}`);
                } else {
                    addMessage('system', '🎙️ No speech detected or blank transcription.');
                }
            };
            mediaRecorder.start();
        } catch (err) {
            console.error(err);
            micBtn.classList.remove('recording');
            addMessage('system', 'Microphone access denied or failed.');
        }
    };

    const stopRecording = (e) => {
        if(e.cancelable) e.preventDefault();
        if (mediaRecorder && mediaRecorder.state === "recording") {
            mediaRecorder.stop();
        }
    };

    micBtn.addEventListener('mousedown', startRecording);
    micBtn.addEventListener('mouseup', stopRecording);
    micBtn.addEventListener('mouseleave', stopRecording);
    micBtn.addEventListener('touchstart', startRecording, {passive: false});
    micBtn.addEventListener('touchend', stopRecording);
}

