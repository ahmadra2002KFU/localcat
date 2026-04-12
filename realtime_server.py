"""FastAPI + WebSocket server for the real-time voice pipeline."""

import os
import json
import asyncio
import logging
import argparse
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

from realtime_session import RealtimeSession, SessionState
from realtime_pipeline import StreamingPipeline

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger("realtime-server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
pipeline: StreamingPipeline | None = None

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Test Transcription Pipeline")

_STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
async def index():
    return FileResponse(_STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/health")
async def health():
    if pipeline is None:
        return JSONResponse({"asr": False, "llm": False, "tts": False}, status_code=503)
    status = await pipeline.health()
    return JSONResponse(status)


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------
async def run_pipeline(ws: WebSocket, session: RealtimeSession, audio: np.ndarray):
    """Run the ASR -> LLM -> TTS pipeline for a single utterance."""
    try:
        # --- ASR ---
        text = await pipeline.transcribe(audio, engine=session.asr_engine)
        if not text or not text.strip():
            await _send_json(ws, {"type": "state", "state": "listening"})
            session.set_state(SessionState.LISTENING)
            return

        log.info("ASR result: %s", text)
        await _send_json(ws, {"type": "transcript", "role": "user", "text": text})
        session.add_message("user", text)

        # --- Streaming LLM + TTS ---
        log.info("Starting LLM streaming...")
        await _send_json(ws, {"type": "state", "state": "generating"})

        token_stream = pipeline.stream_llm(session.conversation, session.cancel_event)
        full_reply = ""

        async for sentence_text, wav_bytes in pipeline.stream_tts_by_sentence(
            token_stream, session.cancel_event
        ):
            if session.cancel_event.is_set():
                log.info("Pipeline cancelled")
                break

            full_reply += sentence_text + " "
            log.info("TTS sentence [%d bytes]: %s", len(wav_bytes), sentence_text[:80])
            await _send_json(
                ws,
                {"type": "transcript_chunk", "role": "assistant", "text": sentence_text},
            )
            try:
                await ws.send_bytes(wav_bytes)
            except (WebSocketDisconnect, RuntimeError):
                log.info("WebSocket closed during audio send")
                return

        full_reply = full_reply.strip()
        if full_reply and not session.cancel_event.is_set():
            log.info("Full reply: %s", full_reply[:120])
            session.add_message("assistant", full_reply)
            await _send_json(ws, {"type": "state", "state": "speaking"})
            session.set_state(SessionState.SPEAKING)
        else:
            log.info("Empty or cancelled reply, back to listening")
            await _send_json(ws, {"type": "state", "state": "listening"})
            session.set_state(SessionState.LISTENING)

    except (WebSocketDisconnect, RuntimeError):
        # Connection already closed — nothing to send.
        log.info("Session %s: WebSocket closed during pipeline run", session.session_id)
    except Exception as exc:
        log.exception("Pipeline error for session %s", session.session_id)
        await _send_json(ws, {"type": "error", "message": str(exc)})
        await _send_json(ws, {"type": "state", "state": "listening"})
        session.set_state(SessionState.LISTENING)


async def _send_json(ws: WebSocket, data: dict) -> None:
    """Send a JSON text frame, swallowing errors if the socket is already closed."""
    try:
        await ws.send_json(data)
    except (WebSocketDisconnect, RuntimeError):
        pass


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    session = RealtimeSession()
    pipeline_task: asyncio.Task | None = None

    log.info("Session %s: connected", session.session_id)
    await ws.send_json({"type": "ready", "session_id": session.session_id})

    try:
        while True:
            message = await ws.receive()

            # --- Binary frame: PCM audio data ---
            if "bytes" in message and message["bytes"] is not None:
                event = session.feed_audio(message["bytes"])

                if event == "speech_start":
                    await ws.send_json({"type": "state", "state": "listening_active"})

                elif event == "speech_end":
                    await ws.send_json({"type": "state", "state": "processing"})
                    audio = session.get_utterance()

                    # Cancel any in-flight pipeline before starting a new one
                    if pipeline_task is not None and not pipeline_task.done():
                        session.interrupt()
                        pipeline_task.cancel()
                        try:
                            await pipeline_task
                        except (asyncio.CancelledError, Exception):
                            pass
                        # Re-create cancel event (interrupt() already does this)
                        session.set_state(SessionState.PROCESSING)

                    pipeline_task = asyncio.create_task(
                        run_pipeline(ws, session, audio)
                    )

                elif event == "interrupt":
                    if pipeline_task is not None and not pipeline_task.done():
                        session.interrupt()
                        pipeline_task.cancel()
                        try:
                            await pipeline_task
                        except (asyncio.CancelledError, Exception):
                            pass
                    else:
                        session.interrupt()
                    await ws.send_json({"type": "interrupt"})

            # --- Text frame: JSON control messages ---
            elif "text" in message and message["text"] is not None:
                try:
                    msg = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue

                msg_type = msg.get("type")

                if msg_type == "interrupt":
                    if pipeline_task is not None and not pipeline_task.done():
                        session.interrupt()
                        pipeline_task.cancel()
                        try:
                            await pipeline_task
                        except (asyncio.CancelledError, Exception):
                            pass
                    else:
                        session.interrupt()
                    await ws.send_json({"type": "interrupt"})

                elif msg_type == "clear":
                    if pipeline_task is not None and not pipeline_task.done():
                        session.interrupt()
                        pipeline_task.cancel()
                        try:
                            await pipeline_task
                        except (asyncio.CancelledError, Exception):
                            pass
                    session.clear()
                    await ws.send_json({"type": "state", "state": "listening"})

                elif msg_type == "set_asr":
                    engine = msg.get("engine", "cohere")
                    if engine in ("cohere", "whisper"):
                        session.asr_engine = engine
                        log.info("Session %s: ASR engine → %s", session.session_id, engine)
                        await ws.send_json({"type": "asr_changed", "engine": engine})

                elif msg_type == "playback_done":
                    session.set_state(SessionState.LISTENING)
                    await ws.send_json({"type": "state", "state": "listening"})

    except (WebSocketDisconnect, RuntimeError):
        log.info("Session %s: disconnected", session.session_id)
    except Exception:
        log.exception("Session %s: unexpected error", session.session_id)
    finally:
        if pipeline_task is not None and not pipeline_task.done():
            session.interrupt()
            pipeline_task.cancel()
            try:
                await pipeline_task
            except (asyncio.CancelledError, Exception):
                pass
        log.info("Session %s: cleaned up", session.session_id)


# ---------------------------------------------------------------------------
# Lifecycle events
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    global pipeline
    log.info("Loading ASR model...")
    pipeline = StreamingPipeline()
    log.info("Pipeline ready")


@app.on_event("shutdown")
async def shutdown():
    if pipeline:
        await pipeline.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7890)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
