# LocalCat — Deployment Guide

## Overview

LocalCat (Test Transcription Pipeline) is a real-time voice-to-voice AI assistant running entirely on-premise on the server **neon**. Users access it via `https://voice.ahmadgh.ovh/`. The system handles Arabic-English bilingual conversations with automatic language detection and code-switching.

## Server

- **Hostname**: neon
- **OS**: Ubuntu, Linux 6.8.0
- **GPUs**: 3x NVIDIA RTX 2080 Ti (11GB each)
- **Python**: 3.11 (venv at `/home/user/voice-pipeline/.venv/`)
- **Project path**: `/home/user/voice-pipeline/`
- **Git repo**: https://github.com/ahmadra2002KFU/localcat (branch: `main-dev`)

## Service Map

```
┌─────────────────────────────────────────────────────────────────┐
│                        INTERNET                                 │
│                           │                                     │
│               voice.ahmadgh.ovh (HTTPS)                         │
│                           │                                     │
│              ┌────────────▼────────────┐                        │
│              │  Cloudflare Tunnel      │                        │
│              │  (llm-stack-cloudflared) │                        │
│              │  Docker, host network   │                        │
│              └────────────┬────────────┘                        │
│                           │                                     │
│                    localhost:7890                                │
│                           │                                     │
├───────────────────────────▼─────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────┐         GPU 1         │
│  │  realtime_server.py (FastAPI)        │  ┌──────────────────┐ │
│  │  Port 7890                           │  │ Whisper v3-turbo │ │
│  │  WebSocket + static files            │  │ (faster-whisper) │ │
│  │                                      │  │ ~809MB           │ │
│  │  ├─ realtime_vad.py (Silero, CPU)    │  └──────────────────┘ │
│  │  ├─ realtime_session.py (state)      │                       │
│  │  └─ realtime_pipeline.py (orchestr.) │         GPU 2         │
│  └──────────────┬───────────────────────┘  ┌──────────────────┐ │
│                 │                          │ Gemma 4 26B      │ │
│      ┌──────────┼──────────┐               │ (vLLM Docker)    │ │
│      │          │          │               │ Port 8083        │ │
│      ▼          ▼          ▼               ├──────────────────┤ │
│  Cohere      vLLM      Chatterbox         │ Cohere Transcr.  │ │
│  ASR         LLM       TTS                │ (vLLM Docker)    │ │
│  :6000       :8083     :8001              │ Port 6000        │ │
│                                            ├──────────────────┤ │
│                                            │ Chatterbox TTS   │ │
│                                            │ Port 8001        │ │
│                                            └──────────────────┘ │
│                                                                 │
│  GPU 0: Reserved for Andexa (:8082)                             │
└─────────────────────────────────────────────────────────────────┘
```

## Running Processes

### 1. Realtime Voice Server (the main app)
- **Process**: `python realtime_server.py --port 7890`
- **Started via**: `./run_realtime.sh` or `nohup` (see below)
- **GPU**: 1 (set via `CUDA_VISIBLE_DEVICES=1`)
- **Logs**: `/tmp/realtime_server.log`
- **Loads**: Whisper large-v3-turbo (~809MB on GPU 1) + Whisper tiny (CPU, for language detection)

```bash
# Start
cd /home/user/voice-pipeline
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 nohup python realtime_server.py --port 7890 > /tmp/realtime_server.log 2>&1 &

# Stop
pkill -f "python realtime_server.py"

# Logs
tail -f /tmp/realtime_server.log
```

### 2. LLM — Gemma 4 26B (Docker)
- **Container**: `gemma4-server`
- **Image**: `vllm/vllm-openai`
- **Port**: 8083 → 8080
- **GPU**: 2
- **Model**: `gemma-4-26B-A4B-it-UD-Q5_K_XL.gguf`
- **Compose**: `/home/user/llm-stack/docker-compose.yml` (but uses a separate entry, not the qwen3 one)
- **Note**: This is a thinking model. The pipeline sends `chat_template_kwargs: {"thinking": False}` to disable visible reasoning, but the model still internally reasons (~200-300 tokens) before responding.

```bash
docker restart gemma4-server
docker logs -f gemma4-server
```

### 3. TTS — Chatterbox Multilingual
- **Port**: 8001
- **GPU**: 2
- **API**: `POST /tts` with `{"text": "...", "language": "ar"|"en"}`
- **Returns**: WAV audio bytes

```bash
curl http://localhost:8001/health
```

### 4. ASR — Cohere Transcribe (Docker)
- **Container**: `cohere-transcribe-vllm-transcribe-1`
- **Port**: 6000 → 8000
- **GPU**: 2
- **Model**: `CohereLabs/cohere-transcribe-03-2026`
- **API**: `POST /v1/audio/transcriptions` (OpenAI-compatible, multipart form)
- **Important**: Field name must be `file` (not `audio`). Requires a `language` hint — doesn't auto-detect.
- **Compose**: `/home/user/cohere-transcribe/docker-compose.yml` (presumably)

```bash
docker restart cohere-transcribe-vllm-transcribe-1
docker logs -f cohere-transcribe-vllm-transcribe-1
```

### 5. Cloudflare Tunnel (Docker)
- **Container**: `llm-stack-cloudflared-1`
- **Network**: host mode
- **Compose**: `/home/user/llm-stack/docker-compose.yml`
- **Routes** (managed via Cloudflare Zero Trust dashboard):
  - `voice.ahmadgh.ovh` → `localhost:7890` (this pipeline)
  - `llm.ahmadgh.ovh` → `localhost:8083`
  - `cohere.ahmadgh.ovh` → `localhost:7000`
  - `testingg.ahmadgh.ovh` → `localhost:8082`
- **WebSocket**: Supported natively through the tunnel

```bash
docker logs -f llm-stack-cloudflared-1
```

## Health Checks

```bash
# All-in-one (from the pipeline server)
curl http://localhost:7890/health
# Returns: {"asr": true, "llm": true, "tts": true, "cohere": true}

# Individual
curl http://localhost:8083/health    # LLM (Gemma 4)
curl http://localhost:8001/health    # TTS (Chatterbox)
curl http://localhost:6000/health    # Cohere Transcribe

# GPU usage
nvidia-smi
```

## Request Flow (step by step)

```
1. User opens https://voice.ahmadgh.ovh/, clicks mic
2. Browser captures audio via AudioWorklet (48kHz → downsampled to 16kHz int16 PCM)
3. ~100ms audio chunks sent over WebSocket to realtime_server.py
4. Server feeds chunks to Silero VAD (ONNX, CPU)
5. VAD detects speech end after 800ms of silence
6. Accumulated audio sent to ASR:
   a. Cohere mode (default):
      - Whisper tiny (CPU, ~100ms) detects language
      - Audio + language hint sent to Cohere Transcribe (GPU 2, port 6000)
   b. Whisper mode:
      - Whisper large-v3-turbo (GPU 1) transcribes directly
7. Transcript sent to user's browser (text bubble appears)
8. Transcript + conversation history sent to Gemma 4 LLM (GPU 2, port 8083)
   - Streaming SSE, thinking disabled
   - ~200-300 hidden reasoning tokens, then content tokens
9. Content tokens accumulated until sentence boundary detected (.!?؟\n)
10. Each complete sentence immediately sent to Chatterbox TTS (GPU 2, port 8001)
    - Language auto-detected per sentence (Arabic char ratio > 25%)
    - Returns WAV bytes
11. Sentence text + WAV audio sent to browser over WebSocket
12. Browser displays text bubble and plays audio
13. Steps 9-12 repeat for each sentence (pipeline streaming)
14. When all audio finishes playing, browser sends "playback_done"
15. Server returns to listening state
```

## Known Considerations

- **Gemma 4 thinking overhead**: Even with thinking disabled, the model uses ~200-300 tokens internally for reasoning before producing visible output. `max_tokens` is set to 1024 to accommodate this. If responses appear empty, this is likely the cause.
- **Cohere field name**: The multipart form field for audio upload must be `file`, not `audio`. Using `audio` returns a 400 error with empty transcription.
- **GPU 2 is shared**: LLM, TTS, and Cohere Transcribe all share GPU 2. Under heavy load, they may compete for VRAM. Monitor with `nvidia-smi`.
- **No persistence**: Conversation history lives in memory per WebSocket session. Disconnecting (page refresh, network drop) resets the conversation.
- **No systemd service yet**: The realtime server runs via nohup. Consider creating a systemd user service for auto-restart.
- **Old Gradio service**: `mufeed-voice.service` (systemd user service) runs the old `web.py` on port 7890. Must be stopped before starting the realtime server: `systemctl --user stop mufeed-voice`.

## File Reference

| File | What it does |
|------|-------------|
| `realtime_server.py` | FastAPI app — WebSocket endpoint, static files, health check |
| `realtime_pipeline.py` | ASR (Cohere + Whisper), streaming LLM (SSE), sentence-level TTS |
| `realtime_session.py` | Per-connection state machine, VAD integration, conversation memory |
| `realtime_vad.py` | Silero VAD ONNX wrapper, speech start/end detection |
| `static/app.js` | Frontend: WebSocket, mic capture, audio playback, UI state |
| `static/audio-processor.js` | AudioWorklet: 48kHz→16kHz downsampling, int16 conversion |
| `static/index.html` | Page layout |
| `static/style.css` | Dark theme, orange + green palette |
| `run_realtime.sh` | Launch script |
| `AGENT_GUIDE.md` | Detailed technical reference for AI agents working on this codebase |
| `REALTIME_SETUP.md` | Quick-start setup guide |
