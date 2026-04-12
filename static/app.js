// ============================================================
// State Management
// ============================================================

const State = {
    IDLE: 'idle',
    LISTENING: 'listening',
    LISTENING_ACTIVE: 'listening_active',
    PROCESSING: 'processing',
    GENERATING: 'generating',
    SPEAKING: 'speaking'
};

let currentState = State.IDLE;
let ws = null;
let audioContext = null;
let micStream = null;
let workletNode = null;
let playbackQueue = [];
let isPlaying = false;
let currentAudio = null;
let currentAssistantBubble = null;
let reconnectDelay = 1000;
let reconnectTimer = null;
let micActive = false;

// ============================================================
// WebSocket Connection
// ============================================================

function connectWS() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
        return;
    }

    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${location.host}/ws`);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
        console.log('[WS] Connected');
        reconnectDelay = 1000;
        setState(State.LISTENING);
    };

    ws.onmessage = (event) => {
        if (typeof event.data === 'string') {
            handleTextMessage(event.data);
        } else {
            // Binary data = audio
            playbackQueue.push(event.data);
            if (!isPlaying) {
                playNext();
            }
        }
    };

    ws.onclose = (event) => {
        console.log('[WS] Closed:', event.code, event.reason);
        if (micActive) {
            scheduleReconnect();
        } else {
            setState(State.IDLE);
        }
    };

    ws.onerror = (err) => {
        console.error('[WS] Error:', err);
    };
}

function handleTextMessage(data) {
    let msg;
    try {
        msg = JSON.parse(data);
    } catch (e) {
        console.error('[WS] Invalid JSON:', data);
        return;
    }

    switch (msg.type) {
        case 'ready':
            console.log('[WS] Session ready:', msg.session_id);
            break;

        case 'state':
            setState(msg.state);
            break;

        case 'transcript':
            addChatBubble('user', msg.text);
            break;

        case 'transcript_chunk':
            appendToAssistantBubble(msg.text);
            break;

        case 'error':
            showError(msg.message || msg.text || 'Unknown error');
            break;

        case 'interrupt':
            clearPlayback();
            break;

        default:
            console.log('[WS] Unknown message type:', msg.type);
    }
}

function scheduleReconnect() {
    if (reconnectTimer) return;
    console.log(`[WS] Reconnecting in ${reconnectDelay}ms...`);
    reconnectTimer = setTimeout(() => {
        reconnectTimer = null;
        connectWS();
        reconnectDelay = Math.min(reconnectDelay * 2, 10000);
    }, reconnectDelay);
}

// ============================================================
// Audio Capture
// ============================================================

async function startMic() {
    try {
        audioContext = new AudioContext({ sampleRate: 48000 });
        micStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                sampleRate: 48000,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            }
        });

        await audioContext.audioWorklet.addModule('/static/audio-processor.js');
        const source = audioContext.createMediaStreamSource(micStream);
        workletNode = new AudioWorkletNode(audioContext, 'mic-processor');

        workletNode.port.onmessage = (event) => {
            if (ws && ws.readyState === WebSocket.OPEN &&
                (currentState === State.LISTENING ||
                 currentState === State.LISTENING_ACTIVE ||
                 currentState === State.SPEAKING)) {
                ws.send(event.data);
            }
        };

        source.connect(workletNode);
        workletNode.connect(audioContext.destination);

        micActive = true;
        document.getElementById('mic-btn').classList.add('active');
        console.log('[Mic] Started');
    } catch (err) {
        console.error('[Mic] Error:', err);
        showError('Microphone access denied or unavailable.');
        setState(State.IDLE);
    }
}

function stopMic() {
    micActive = false;
    document.getElementById('mic-btn').classList.remove('active');

    if (workletNode) {
        workletNode.disconnect();
        workletNode = null;
    }
    if (micStream) {
        micStream.getTracks().forEach(t => t.stop());
        micStream = null;
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
    if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
    }

    clearPlayback();
    console.log('[Mic] Stopped');
}

// ============================================================
// Audio Playback
// ============================================================

function playNext() {
    if (playbackQueue.length === 0) {
        isPlaying = false;
        currentAudio = null;
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'playback_done' }));
        }
        return;
    }

    isPlaying = true;
    const audioData = playbackQueue.shift();
    const blob = new Blob([audioData], { type: 'audio/wav' });
    const url = URL.createObjectURL(blob);

    const audio = new Audio(url);
    currentAudio = audio;

    audio.onended = () => {
        URL.revokeObjectURL(url);
        currentAudio = null;
        playNext();
    };

    audio.onerror = (err) => {
        console.error('[Playback] Error:', err);
        URL.revokeObjectURL(url);
        currentAudio = null;
        playNext();
    };

    audio.play().catch(err => {
        console.error('[Playback] Play failed:', err);
        URL.revokeObjectURL(url);
        currentAudio = null;
        playNext();
    });
}

function clearPlayback() {
    playbackQueue = [];
    if (currentAudio) {
        currentAudio.pause();
        currentAudio.currentTime = 0;
        currentAudio = null;
    }
    isPlaying = false;
}

// ============================================================
// UI Updates
// ============================================================

function setState(state) {
    currentState = state;

    const dot = document.getElementById('status-dot');
    const text = document.getElementById('status-text');
    const micBtn = document.getElementById('mic-btn');

    // Remove all state classes from dot
    dot.className = 'status-dot';

    switch (state) {
        case State.IDLE:
            text.textContent = 'Click mic to start';
            micBtn.classList.remove('active');
            break;
        case State.LISTENING:
            dot.classList.add('listening');
            text.textContent = 'Listening...';
            micBtn.classList.add('active');
            break;
        case State.LISTENING_ACTIVE:
            dot.classList.add('listening_active');
            text.textContent = 'Hearing you...';
            micBtn.classList.add('active');
            break;
        case State.PROCESSING:
            dot.classList.add('processing');
            text.textContent = 'Processing...';
            break;
        case State.GENERATING:
            dot.classList.add('generating');
            text.textContent = 'Generating response...';
            break;
        case State.SPEAKING:
            dot.classList.add('speaking');
            text.textContent = 'Speaking...';
            break;
    }
}

function isArabic(text) {
    const arabicPattern = /[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]/g;
    const matches = text.match(arabicPattern);
    if (!matches) return false;
    // If more than 25% of non-space characters are Arabic
    const nonSpace = text.replace(/\s/g, '');
    return matches.length / nonSpace.length > 0.25;
}

function addChatBubble(role, text) {
    const container = document.getElementById('chat-container');
    const bubble = document.createElement('div');
    bubble.className = `chat-bubble ${role}`;

    if (isArabic(text)) {
        bubble.setAttribute('dir', 'rtl');
    }

    const label = document.createElement('span');
    label.className = 'role-label';
    label.textContent = role === 'user' ? 'You' : 'Assistant';

    const content = document.createElement('span');
    content.className = 'bubble-text';
    content.textContent = text;

    bubble.appendChild(label);
    bubble.appendChild(content);
    container.appendChild(bubble);

    // Reset assistant bubble tracker for new user messages
    if (role === 'user') {
        currentAssistantBubble = null;
    }

    scrollToBottom();
}

function appendToAssistantBubble(text) {
    if (!currentAssistantBubble) {
        // Create a new assistant bubble
        const container = document.getElementById('chat-container');
        const bubble = document.createElement('div');
        bubble.className = 'chat-bubble assistant';

        const label = document.createElement('span');
        label.className = 'role-label';
        label.textContent = 'Assistant';

        const content = document.createElement('span');
        content.className = 'bubble-text';
        content.textContent = '';

        bubble.appendChild(label);
        bubble.appendChild(content);
        container.appendChild(bubble);

        currentAssistantBubble = bubble;
    }

    const contentEl = currentAssistantBubble.querySelector('.bubble-text');
    contentEl.textContent += text;

    // Re-check RTL
    if (isArabic(contentEl.textContent)) {
        currentAssistantBubble.setAttribute('dir', 'rtl');
    } else {
        currentAssistantBubble.removeAttribute('dir');
    }

    scrollToBottom();
}

function showError(message) {
    const container = document.getElementById('chat-container');
    const errDiv = document.createElement('div');
    errDiv.className = 'error-msg';
    errDiv.textContent = message;
    container.appendChild(errDiv);
    scrollToBottom();
}

function scrollToBottom() {
    const container = document.getElementById('chat-container');
    requestAnimationFrame(() => {
        container.scrollTop = container.scrollHeight;
    });
}

// ============================================================
// Controls
// ============================================================

function toggleMic() {
    if (currentState === State.IDLE) {
        startMic();
        connectWS();
    } else {
        stopMic();
        if (ws) ws.close();
        setState(State.IDLE);
    }
}

function clearChat() {
    document.getElementById('chat-container').innerHTML = '';
    currentAssistantBubble = null;
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'clear' }));
    }
}

// ============================================================
// Init
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('mic-btn').addEventListener('click', toggleMic);
    document.getElementById('clear-btn').addEventListener('click', clearChat);
    document.getElementById('asr-select').addEventListener('change', (e) => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'set_asr', engine: e.target.value }));
        }
    });
});
