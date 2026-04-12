# Test Transcription Pipeline — Setup Guide

## Overview

A real-time voice-to-voice AI web interface replacing the old Gradio-based `web.py`. Users open a browser, click the mic, talk naturally, and the system auto-detects when they stop speaking, transcribes their speech, generates an LLM response with streaming, and plays back TTS audio sentence-by-sentence — all with minimal latency.

## Architecture

```
Browser mic (16kHz PCM) → WebSocket → Silero VAD (auto speech detection)
  → ASR (faster-whisper large-v3-turbo, GPU 1)
  → Streaming LLM (vLLM SSE, Gemma 4 26B, port 8083, GPU 2)
  → Sentence boundary splitting
  → TTS per sentence (Chatterbox Multilingual, port 8001, GPU 2)
  → WAV chunks → WebSocket → Browser playback
```

## New Files (existing files untouched)

| File | Purpose |
|------|---------|
| `realtime_vad.py` | Silero VAD ONNX wrapper — processes 512-sample chunks, detects speech start/end |
| `realtime_session.py` | Per-WebSocket session state machine — VAD, audio buffering, conversation history |
| `realtime_pipeline.py` | Streaming pipeline — ASR, streaming LLM (SSE), sentence-boundary TTS |
| `realtime_server.py` | FastAPI + WebSocket server — routes, pipeline orchestration, static files |
| `static/index.html` | Single-page UI — "Test Transcription Pipeline" |
| `static/app.js` | Frontend logic — WebSocket, AudioWorklet mic, playback queue, UI states |
| `static/audio-processor.js` | AudioWorklet — captures raw PCM, downsamples 48kHz→16kHz, converts to int16 |
| `static/style.css` | Dark theme with orange + green color scheme, RTL Arabic support |
| `run_realtime.sh` | Launch script — activates venv, sets GPU, starts server |

## Prerequisites

These services must be running before starting the pipeline:

1. **LLM (vLLM)** on port 8083, GPU 2
   ```bash
   curl http://localhost:8083/health
   ```

2. **TTS (Chatterbox)** on port 8001, GPU 2
   ```bash
   curl http://localhost:8001/health
   ```

3. **ASR model** — large-v3-turbo (~809 MB, auto-downloaded from HuggingFace on first run)

4. **Silero VAD ONNX** — cached at `~/.cache/torch/hub/snakers4_silero-vad_master/src/silero_vad/data/silero_vad.onnx`

## How to Start

```bash
# Stop the old Gradio service if running
systemctl --user stop mufeed-voice  # if the old Gradio service is running

# Start the new real-time server
cd /home/user/voice-pipeline
./run_realtime.sh
```

This runs on **port 7890** (same port as the old Gradio UI).

### Manual start (without script)

```bash
cd /home/user/voice-pipeline
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 python realtime_server.py --port 7890
```

## How to Use

1. Open browser to `http://neon:7890` (or whatever the server hostname is)
2. Click the green mic button
3. Grant microphone permission when prompted
4. Start talking — the system auto-detects when you stop
5. Watch the status indicator:
   - **Green pulse** = Listening (ready for speech)
   - **Orange pulse** = Hearing you (speech detected, accumulating audio)
   - **Orange spin** = Processing (running ASR)
   - **Orange glow** = Generating (LLM streaming + TTS)
   - **Green glow** = Speaking (playing audio response)
6. **Barge-in**: Start talking while the AI is speaking to interrupt it
7. **Clear**: Click "Clear" to reset conversation history

## WebSocket Protocol

### Client → Server

| Type | Format | Description |
|------|--------|-------------|
| Audio | Binary (PCM int16 LE, 16kHz mono) | ~100ms chunks continuously while mic is active |
| Interrupt | JSON `{"type": "interrupt"}` | Cancel current AI response |
| Clear | JSON `{"type": "clear"}` | Reset conversation history |
| Playback done | JSON `{"type": "playback_done"}` | All audio chunks finished playing |

### Server → Client

| Type | Format | Description |
|------|--------|-------------|
| Ready | `{"type": "ready", "session_id": "..."}` | Connection established |
| State | `{"type": "state", "state": "..."}` | UI state transition |
| User transcript | `{"type": "transcript", "role": "user", "text": "..."}` | ASR result |
| AI text chunk | `{"type": "transcript_chunk", "role": "assistant", "text": "..."}` | One sentence of response |
| AI audio | Binary frame (WAV bytes) | TTS audio for one sentence |
| Error | `{"type": "error", "message": "..."}` | Error info |

## GPU Allocation

| GPU | Device | Usage |
|-----|--------|-------|
| GPU 0 | RTX 2080 Ti | Andexa (not used by this pipeline) |
| GPU 1 | RTX 2080 Ti | ASR (faster-whisper large-v3-turbo) — set via `CUDA_VISIBLE_DEVICES=1` |
| GPU 2 | RTX 2080 Ti | LLM (vLLM) + TTS (Chatterbox) — separate services |

## Dependencies

All already installed in `.venv/`:
- `fastapi`, `uvicorn`, `websockets` — web server
- `faster-whisper`, `ctranslate2` — ASR
- `aiohttp` — async HTTP client for LLM/TTS
- `onnxruntime` — Silero VAD inference
- `torch`, `numpy`, `soundfile`, `librosa` — audio processing

No new packages required.

## Latency Budget

| Stage | Expected Latency |
|-------|-----------------|
| VAD silence detection | ~800ms after speech ends |
| ASR (Whisper) | 500–1500ms |
| LLM first sentence | 700–2500ms |
| TTS first sentence | 500–1500ms |
| **Total to first audio** | **~2.5–4s** |

vs old Gradio pipeline: ~5–8s (waited for full LLM response before TTS).

## Key Configuration

Tunable parameters in source files:

- **VAD sensitivity**: `realtime_vad.py` → `threshold=0.5`, `min_silence_ms=800` (lower = more sensitive, shorter silence to end speech)
- **LLM tokens**: `realtime_pipeline.py` → `max_tokens` not set (vLLM default), `temperature` not set (vLLM default)
- **Sentence split minimum**: `realtime_pipeline.py` → `_MIN_SENTENCE_LEN = 15` chars
- **Language detection threshold**: `realtime_pipeline.py` → Arabic char ratio > 0.25

## Troubleshooting

- **No mic permission**: Browser requires HTTPS for `getUserMedia` in production. Works on `localhost` and `http://` on local network.
- **ASR slow**: Check GPU 1 is free with `nvidia-smi`. The large-v3-turbo model is ~809 MB.
- **No audio playback**: Check browser console for autoplay policy errors. The user must interact (click mic) before audio can play.
- **WebSocket disconnects**: Check server logs. The frontend auto-reconnects with exponential backoff.
- **LLM/TTS down**: Visit `http://neon:7890/health` to check service status.
