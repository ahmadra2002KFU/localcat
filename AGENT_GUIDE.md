# Test Transcription Pipeline — Agent Guide

Complete reference for the real-time voice-to-voice AI pipeline running on neon.

## What This Is

A browser-based real-time voice assistant. The user opens `https://voice.ahmadgh.ovh/`, clicks the mic, talks naturally, and the system:
1. Auto-detects when they stop speaking (Silero VAD)
2. Transcribes speech (Cohere Transcribe or Whisper)
3. Generates a response (Gemma 4 26B via vLLM, streaming)
4. Speaks the response back (Chatterbox TTS, sentence-by-sentence)

Everything is 100% local. No cloud APIs.

## Architecture

```
Browser mic (48kHz) → AudioWorklet (downsample to 16kHz int16 PCM)
  → WebSocket → Server VAD (Silero ONNX, CPU)
  → ASR: Cohere Transcribe (GPU 2, port 6000) with tiny Whisper language detection (CPU)
       OR Whisper large-v3-turbo (GPU 1)
  → LLM: Gemma 4 26B via vLLM (GPU 2, port 8083, SSE streaming)
       thinking disabled via chat_template_kwargs
  → Sentence boundary detection → TTS per sentence
  → TTS: Chatterbox Multilingual (GPU 2, port 8001)
  → WAV audio chunks → WebSocket → Browser Audio() playback
```

## GPU Allocation

| GPU | Card | Usage |
|-----|------|-------|
| GPU 0 | RTX 2080 Ti (11GB) | Andexa (not used by this pipeline) |
| GPU 1 | RTX 2080 Ti (11GB) | Whisper large-v3-turbo ASR (~809MB) |
| GPU 2 | RTX 2080 Ti (11GB) | vLLM (Gemma 4, port 8083) + Chatterbox TTS (port 8001) + Cohere Transcribe (Docker, port 6000) |

## Files

### Backend (Python)

| File | Purpose |
|------|---------|
| `realtime_server.py` | FastAPI + WebSocket server. Routes, pipeline orchestration, static file serving. Port 7890. |
| `realtime_pipeline.py` | Streaming ASR → LLM → TTS. Handles Cohere and Whisper ASR, SSE token parsing, sentence-boundary TTS dispatch. |
| `realtime_session.py` | Per-WebSocket session state machine. VAD-driven audio buffering, conversation history, cancel/interrupt support. |
| `realtime_vad.py` | Silero VAD ONNX wrapper. Processes 512-sample chunks at 16kHz, detects speech start/end. Runs on CPU. |
| `run_realtime.sh` | Launch script: `CUDA_VISIBLE_DEVICES=1 python realtime_server.py --port 7890` |

### Frontend (static/)

| File | Purpose |
|------|---------|
| `index.html` | Single-page UI titled "Test Transcription Pipeline" |
| `app.js` | WebSocket client, AudioWorklet mic capture, WAV playback queue, UI state machine, barge-in support |
| `audio-processor.js` | AudioWorklet: captures 48kHz PCM, downsamples to 16kHz, converts to int16, sends ~100ms chunks |
| `style.css` | Dark theme with orange (#F97316) + green (#22C55E) palette. RTL Arabic support. No purple. |

### Legacy Files (not used by realtime pipeline)

| File | Purpose |
|------|---------|
| `web.py` | Old Gradio UI (replaced by realtime pipeline) |
| `bot.py` | CLI pipeline with interactive/file/demo modes |
| `voice_pipeline.py` | FastAPI REST server (no streaming) |
| `start.sh` | CLI launcher for bot.py |
| `run_web.sh` | Old Gradio launcher |

## External Services (must be running)

| Service | Port | Container/Process | Health Check |
|---------|------|-------------------|--------------|
| LLM (vLLM) | 8083 | `gemma4-server` Docker | `curl http://localhost:8083/health` |
| TTS (Chatterbox) | 8001 | Systemd or manual | `curl http://localhost:8001/health` |
| Cohere Transcribe | 6000 | `cohere-transcribe-vllm-transcribe-1` Docker | `curl http://localhost:6000/health` |
| Cloudflare Tunnel | — | `llm-stack-cloudflared-1` Docker | Routes `voice.ahmadgh.ovh` → `localhost:7890` |

All health checks combined: `curl http://localhost:7890/health`

## How to Start / Stop

```bash
# Start
cd /home/user/voice-pipeline
./run_realtime.sh

# Or manually
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 python realtime_server.py --port 7890

# Background
CUDA_VISIBLE_DEVICES=1 nohup python realtime_server.py --port 7890 > /tmp/realtime_server.log 2>&1 &

# Stop
pkill -f "python realtime_server.py"

# View logs
tail -f /tmp/realtime_server.log
```

## Key Technical Details

### Cohere Transcribe Integration
- Cohere does NOT auto-detect language — it needs a language hint
- We run a fast pre-pass with **Whisper tiny on CPU** (~100ms) just to detect the language
- Then send the audio + language hint to Cohere via `POST /v1/audio/transcriptions`
- The multipart form field must be named **`file`** (not `audio`) — OpenAI-compatible format
- The `model` field must be `CohereLabs/cohere-transcribe-03-2026`

### Gemma 4 Thinking Model
- Gemma 4 26B is a **thinking model** — it produces `reasoning_content` tokens before `content` tokens
- Thinking is disabled with `chat_template_kwargs: {"thinking": False}` in the API payload
- Even with thinking disabled, the model still produces ~200-300 reasoning tokens internally (counted against max_tokens)
- `max_tokens` is set to **1024** to ensure enough room for both reasoning and actual content
- Without this, the model burns all tokens on reasoning and returns empty content

### Sentence-Level TTS Streaming
- LLM tokens are accumulated until a sentence boundary is detected (`.!?؟\n`)
- Minimum 15 chars per sentence to avoid tiny TTS fragments
- Each sentence is immediately sent to TTS while the LLM continues generating
- The browser plays audio chunks sequentially — first audio starts playing while later sentences are still being synthesized

### VAD Parameters
- `threshold=0.5` — speech probability threshold
- `min_silence_ms=800` — silence duration to end utterance
- `min_speech_ms=250` — minimum speech duration to accept
- `pre_speech_pad_ms=300` — audio captured before speech start (ring buffer)
- Tunable in `realtime_vad.py` constructor

### Language Detection
- Arabic char ratio > 0.25 → Arabic, else English
- Uses Unicode range `\u0600-\u06FF`
- Applied per-sentence for TTS language selection

### WebSocket Protocol

**Client → Server:**
- Binary frames: PCM int16 LE, 16kHz mono, ~100ms chunks
- Text frames: `{"type": "interrupt"}`, `{"type": "clear"}`, `{"type": "playback_done"}`, `{"type": "set_asr", "engine": "cohere"|"whisper"}`

**Server → Client:**
- `{"type": "ready", "session_id": "..."}` — connection established
- `{"type": "state", "state": "listening|listening_active|processing|generating|speaking"}` — UI state
- `{"type": "transcript", "role": "user", "text": "..."}` — ASR result
- `{"type": "transcript_chunk", "role": "assistant", "text": "..."}` — one sentence of LLM output
- Binary frames: WAV audio for one TTS sentence
- `{"type": "error", "message": "..."}` — error info

### UI States
- **Listening** (green pulse): ready for speech
- **Listening Active** (orange pulse): speech detected, accumulating
- **Processing** (orange spin): running ASR
- **Generating** (orange glow): LLM streaming + TTS
- **Speaking** (green glow): playing audio response

### ASR Selector
- Dropdown in the UI footer: "Cohere" (default) or "Whisper"
- Sends `{"type": "set_asr", "engine": "..."}` to server
- Change takes effect on next utterance

## System Prompt

Bilingual Arabic/English, voice-optimized:
- 1-3 sentences max
- Conversational tone, no formal writing
- No bullet points, lists, markdown (everything is spoken aloud)
- Natural code-switching support
- No AI self-reference

## Latency Budget

| Stage | Typical Latency |
|-------|----------------|
| VAD silence detection | ~800ms after speech ends |
| Language detection (tiny Whisper CPU) | ~100ms |
| Cohere ASR | ~500ms |
| Whisper ASR (GPU) | 500-1500ms |
| LLM first content token | ~2s (includes hidden reasoning) |
| LLM full sentence | ~3-5s |
| TTS per sentence | ~2-4s |
| **Total to first audio** | **~5-8s with Cohere** |

## Dependencies

All in `.venv/` (Python 3.11):
- `fastapi`, `uvicorn`, `websockets` — web server
- `faster-whisper`, `ctranslate2` — Whisper ASR
- `aiohttp` — async HTTP for LLM/TTS/Cohere
- `onnxruntime` — Silero VAD
- `torch`, `numpy`, `soundfile`, `librosa` — audio processing

No new packages needed beyond what's already installed.

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| "Just listens, never processes" | VAD not triggering | Check mic permissions, try speaking louder, reduce `threshold` in `realtime_vad.py` |
| Processes but no text/audio | Cohere returns empty | Check field name is `file` not `audio`, verify `curl http://localhost:6000/health` |
| Text appears but no audio | TTS down | `curl http://localhost:8001/health` |
| LLM returns empty | max_tokens too low for thinking model | Must be ≥512, currently 1024 |
| WebSocket disconnects | Cloudflare tunnel timeout | Check `docker logs llm-stack-cloudflared-1` |
| Audio doesn't play | Browser autoplay policy | User must click mic button first (user interaction required) |
| Port 7890 in use | Old Gradio still running | `pkill -f "python web.py"` or `systemctl --user stop mufeed-voice` |
