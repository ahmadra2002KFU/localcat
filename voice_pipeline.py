"""
Mufeed Voice Pipeline — fully local voice-to-voice AI assistant
Components:
  ASR: Arabic-Whisper-CodeSwitching-Edition (CTranslate2 int8) on GPU 0
  LLM: Gemma 4 26B-A4B via llama.cpp OpenAI-compatible API on port 8083
  TTS: Chatterbox Multilingual on port 8001 (GPU 2)
"""

import asyncio
import io
import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from faster_whisper import WhisperModel

# ─── Configuration ───────────────────────────────────────────────────────────

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("voice-pipeline")

# Service endpoints
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8083/v1")
TTS_BASE_URL = os.getenv("TTS_BASE_URL", "http://localhost:8001")
WHISPER_MODEL_PATH = os.getenv(
    "WHISPER_MODEL_PATH",
    os.path.expanduser("~/.cache/huggingface/hub/arabic-whisper-codeswitching-ct2"),
)
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_DEVICE_INDEX = int(os.getenv("WHISPER_DEVICE_INDEX", "0"))
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8_float16")

# LLM system prompt — bilingual Arabic/English assistant for Mufeed
SYSTEM_PROMPT = """أنت مساعد ذكي ثنائي اللغة يعمل في شركة مُفيد المتقدمة. 
You are a bilingual AI assistant working at Mufeed Advanced Company.

Rules:
- Respond in the same language the user speaks (Arabic or English, or a mix).
- Use الفصحى (Modern Standard Arabic) when writing Arabic, with full diacritics (تشكيل).
- Be concise and professional.
- If the user mixes Arabic and English in one message, respond in the same mixed style."""

# ─── ASR: Arabic-English Code-Switching Whisper ──────────────────────────────

class ArabicCodeSwitchingASR:
    """Speech recognition optimized for Arabic-English code-switching."""

    def __init__(self, model_path: str, device: str = "cuda",
                 device_index: int = 0, compute_type: str = "int8_float16"):
        logger.info(f"Loading Whisper model from {model_path}...")
        self.model = WhisperModel(
            model_path,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
        )
        logger.info("Whisper model loaded successfully")

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> dict:
        """Transcribe audio file. Returns {text, language, segments, duration}."""
        # Run two passes for code-switching: one Arabic-biased, one English-biased
        # Then pick the better result. This handles mixed Arabic-English speech.
        if language is None:
            return self._transcribe_dual_pass(audio_path, beam_size, vad_filter)

        segments_iter, info = self.model.transcribe(
            audio_path,
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            vad_parameters=dict(
                min_silence_duration_ms=300,
                speech_pad_ms=200,
            ),
        )

        segments = list(segments_iter)
        text = " ".join(s.text.strip() for s in segments if s.text.strip())
        detected_lang = info.language if info else "unknown"
        confidence = info.language_probability if info else 0.0
        duration = info.duration if info else 0.0

        logger.info(f"ASR: lang={language} duration={duration:.1f}s text={text[:100]}")

        return {
            "text": text,
            "language": detected_lang,
            "confidence": confidence,
            "segments": [
                {"start": s.start, "end": s.end, "text": s.text.strip()}
                for s in segments if s.text.strip()
            ],
            "duration": duration,
        }

    def _transcribe_dual_pass(
        self,
        audio_path: str,
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> dict:
        """Dual-pass transcription for Arabic-English code-switching.
        Runs with both Arabic and English hints, picks the better result."""
        results = {}
        for lang in ["ar", "en"]:
            try:
                segments_iter, info = self.model.transcribe(
                    audio_path,
                    language=lang,
                    beam_size=beam_size,
                    vad_filter=vad_filter,
                    vad_parameters=dict(
                        min_silence_duration_ms=300,
                        speech_pad_ms=200,
                    ),
                )
                segments = list(segments_iter)
                text = " ".join(s.text.strip() for s in segments if s.text.strip())
                results[lang] = {
                    "text": text,
                    "confidence": info.language_probability if info else 0.0,
                    "segments": segments,
                    "info": info,
                }
            except Exception as e:
                logger.warning(f"ASR pass {lang} failed: {e}")

        if not results:
            return {"text": "", "language": "unknown", "confidence": 0, "segments": [], "duration": 0}

        # Also try auto-detect as a third option
        try:
            segments_iter, info = self.model.transcribe(
                audio_path,
                language=None,
                beam_size=beam_size,
                vad_filter=vad_filter,
                vad_parameters=dict(
                    min_silence_duration_ms=300,
                    speech_pad_ms=200,
                ),
            )
            segments = list(segments_iter)
            text = " ".join(s.text.strip() for s in segments if s.text.strip())
            detected = info.language if info else "unknown"
            conf = info.language_probability if info else 0.0
            if detected in ["ar", "en"]:
                results[detected] = results.get(detected, {})
                # Use auto-detect only if it has higher confidence
                if conf > results[detected].get("confidence", 0):
                    results[detected] = {
                        "text": text, "confidence": conf,
                        "segments": segments, "info": info,
                    }
        except Exception:
            pass

        # Pick the result with higher confidence
        best_lang = max(results, key=lambda k: results[k].get("confidence", 0))
        best = results[best_lang]
        info = best.get("info")
        text = best["text"]

        # Detect actual language from content
        arabic_ratio = len(re.findall(r'[\u0600-\u06FF]', text)) / max(len(text), 1)
        detected_lang = "ar" if arabic_ratio > 0.3 else "en"

        logger.info(f"ASR dual-pass: best={best_lang} detected={detected_lang} "
                     f"conf={best['confidence']:.2f} text={text[:100]}")

        return {
            "text": text,
            "language": detected_lang,
            "confidence": best["confidence"],
            "segments": [
                {"start": s.start, "end": s.end, "text": s.text.strip()}
                for s in best.get("segments", []) if s.text.strip()
            ],
            "duration": info.duration if info else 0.0,
        }

class BilingualLLM:
    """Bilingual LLM client using OpenAI-compatible API (Gemma 4)."""

    def __init__(self, base_url: str, system_prompt: str = SYSTEM_PROMPT):
        self.base_url = base_url.rstrip("/")
        self.system_prompt = system_prompt
        self.client = httpx.AsyncClient(timeout=120.0)

    async def chat(self, message: str, history: list[dict] = None) -> str:
        """Send a message and get a reply. Supports conversation history."""
        messages = [{"role": "system", "content": self.system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": message})

        try:
            resp = await self.client.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": "gemma-4-26B-A4B-it-UD-Q5_K_XL.gguf",
                    "messages": messages,
                    "max_tokens": 1024,
                    "temperature": 0.7,
                    "stream": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            reply = data["choices"][0]["message"]["content"]
            logger.info(f"LLM reply ({len(reply)} chars): {reply[:100]}")
            return reply
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return f"عذراً، حدث خطأ في معالجة طلبك: {e}"


# ─── TTS: Chatterbox Multilingual ────────────────────────────────────────────

class ChatterboxTTS:
    """Text-to-speech via Chatterbox Multilingual server."""

    def __init__(self, base_url: str = TTS_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=60.0)

    def _detect_language(self, text: str) -> str:
        """Detect if text is primarily Arabic or English."""
        # Count Arabic characters
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        total_chars = len(re.findall(r'[\u0600-\u06FF\u0041-\u007A\s]', text))
        if total_chars > 0 and arabic_chars / total_chars > 0.3:
            return "ar"
        return "en"

    async def synthesize(self, text: str, language: Optional[str] = None) -> bytes:
        """Synthesize text to audio (WAV bytes)."""
        if language is None:
            language = self._detect_language(text)

        # For mixed Arabic-English text, split and synthesize per-segment
        is_mixed = bool(re.search(r'[\u0600-\u06FF]', text)) and bool(re.search(r'[a-zA-Z]{3,}', text))

        if is_mixed:
            return await self._synthesize_mixed(text)

        try:
            resp = await self.client.post(
                f"{self.base_url}/tts",
                json={"text": text, "language": language},
            )
            resp.raise_for_status()
            audio_bytes = resp.content
            logger.info(f"TTS: {language} {len(text)} chars → {len(audio_bytes)} bytes audio")
            return audio_bytes
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return b""

    async def _synthesize_mixed(self, text: str) -> bytes:
        """Handle code-switched text by splitting into language segments."""
        # Split text into Arabic and English segments
        segments = re.split(r'(?<=[\u0600-\u06FF\s])\s*(?=[a-zA-Z])|(?<=[a-zA-Z\s])\s*(?=[\u0600-\u06FF])', text)
        segments = [s.strip() for s in segments if s.strip()]

        if len(segments) <= 1:
            # Fallback to single language
            lang = self._detect_language(text)
            return await self.synthesize(text, language=lang)

        audio_chunks = []
        for seg in segments:
            lang = self._detect_language(seg)
            try:
                resp = await self.client.post(
                    f"{self.base_url}/tts",
                    json={"text": seg, "language": lang},
                )
                resp.raise_for_status()
                if resp.content:
                    audio_chunks.append(resp.content)
            except Exception as e:
                logger.warning(f"TTS mixed segment error: {e}")

        if not audio_chunks:
            return b""

        # Concatenate WAV files (simple PCM concatenation for same-format WAVs)
        return self._concatenate_wav(audio_chunks)

    def _concatenate_wav(self, wav_chunks: list[bytes]) -> bytes:
        """Concatenate multiple WAV files into one."""
        import wave

        output = io.BytesIO()
        with wave.open(output, 'wb') as out_wf:
            first = True
            for chunk in wav_chunks:
                chunk_io = io.BytesIO(chunk)
                with wave.open(chunk_io, 'rb') as wf:
                    if first:
                        out_wf.setnchannels(wf.getnchannels())
                        out_wf.setsampwidth(wf.getsampwidth())
                        out_wf.setframerate(wf.getframerate())
                        first = False
                    out_wf.writeframes(wf.readframes(wf.getnframes()))
        return output.getvalue()

    async def health(self) -> bool:
        try:
            resp = await self.client.get(f"{self.base_url}/health", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False


# ─── Audio Preprocessing ─────────────────────────────────────────────────────

async def preprocess_audio(input_path: str) -> str:
    """Convert any audio format to 16kHz mono WAV for Whisper."""
    output_path = input_path.replace(Path(input_path).suffix, "_processed.wav")

    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "16000",          # 16kHz (Whisper's expected sample rate)
        "-ac", "1",              # mono
        "-sample_fmt", "s16",    # 16-bit PCM
        output_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await proc.wait()

    if proc.returncode != 0:
        logger.error(f"ffmpeg failed: {await proc.stderr.read()}")
        return input_path  # return original as fallback

    return output_path


# ─── Main Pipeline ───────────────────────────────────────────────────────────

class VoicePipeline:
    """Orchestrates ASR → LLM → TTS."""

    def __init__(self):
        self.asr = ArabicCodeSwitchingASR(
            WHISPER_MODEL_PATH,
            device=WHISPER_DEVICE,
            device_index=WHISPER_DEVICE_INDEX,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
        self.llm = BilingualLLM(LLM_BASE_URL)
        self.tts = ChatterboxTTS(TTS_BASE_URL)

    async def process(
        self,
        audio_path: str,
        language: Optional[str] = None,
        skip_tts: bool = False,
    ) -> dict:
        """Process audio through the full pipeline.
        Returns {transcription, reply, reply_audio_path, timings}."""
        t0 = time.time()
        result = {"timings": {}}

        # Step 1: ASR
        t1 = time.time()
        asr_result = self.asr.transcribe(audio_path, language=language)
        result["transcription"] = asr_result
        result["timings"]["asr"] = round(time.time() - t1, 2)

        if not asr_result["text"].strip():
            result["reply"] = ""
            result["error"] = "No speech detected"
            return result

        # Step 2: LLM
        t2 = time.time()
        reply = await self.llm.chat(asr_result["text"])
        result["reply"] = reply
        result["timings"]["llm"] = round(time.time() - t2, 2)

        # Step 3: TTS
        if not skip_tts and reply:
            t3 = time.time()
            audio_bytes = await self.tts.synthesize(reply)
            result["timings"]["tts"] = round(time.time() - t3, 2)

            if audio_bytes:
                out_path = f"/tmp/voice_pipeline_reply_{int(time.time())}.wav"
                with open(out_path, "wb") as f:
                    f.write(audio_bytes)
                result["reply_audio_path"] = out_path
                result["reply_audio_size"] = len(audio_bytes)
        else:
            result["timings"]["tts"] = 0

        result["timings"]["total"] = round(time.time() - t0, 2)
        return result


# ─── FastAPI Server ──────────────────────────────────────────────────────────

app = FastAPI(
    title="Mufeed Voice Pipeline",
    description="Fully local voice-to-voice AI pipeline: Arabic/English ASR → Gemma 4 LLM → Chatterbox TTS",
    version="1.0.0",
)

# Global pipeline instance (loaded at startup)
pipeline: Optional[VoicePipeline] = None


@app.on_event("startup")
async def startup():
    global pipeline
    logger.info("Starting Mufeed Voice Pipeline...")
    pipeline = VoicePipeline()
    logger.info("Voice Pipeline ready!")


@app.get("/health")
async def health():
    """Check health of all components."""
    checks = {}
    checks["asr"] = pipeline is not None
    checks["tts"] = await pipeline.tts.health() if pipeline else False
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{LLM_BASE_URL}/models")
            checks["llm"] = r.status_code == 200
    except Exception:
        checks["llm"] = False
    checks["all_ok"] = all(checks.values())
    return checks


@app.post("/v1/voice")
async def process_voice(
    audio: UploadFile = File(...),
    language: Optional[str] = Form(None),
    skip_tts: bool = Form(False),
):
    """Process audio: transcribe → LLM reply → TTS audio response.
    Accepts: WAV, OGG, MP3, M4A, FLAC
    Returns: JSON with transcription, reply, and audio file path."""
    if not pipeline:
        return JSONResponse({"error": "Pipeline not initialized"}, status_code=503)

    # Save uploaded audio
    suffix = Path(audio.filename).suffix if audio.filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(await audio.read())
        input_path = f.name

    try:
        # Preprocess audio for Whisper
        processed_path = await preprocess_audio(input_path)

        # Run pipeline
        result = await pipeline.process(
            processed_path,
            language=language,
            skip_tts=skip_tts,
        )

        # Return result
        response = {
            "transcription": result.get("transcription", {}),
            "reply": result.get("reply", ""),
            "timings": result.get("timings", {}),
        }

        if "error" in result:
            response["error"] = result["error"]

        if result.get("reply_audio_path"):
            response["audio_url"] = f"/v1/audio/{Path(result['reply_audio_path']).name}"

        return response

    finally:
        # Cleanup
        for p in [input_path, input_path.replace(suffix, "_processed.wav")]:
            if os.path.exists(p):
                try:
                    os.unlink(p)
                except Exception:
                    pass


@app.post("/v1/transcribe")
async def transcribe_only(
    audio: UploadFile = File(...),
    language: Optional[str] = Form(None),
):
    """Transcribe audio only (no LLM or TTS)."""
    if not pipeline:
        return JSONResponse({"error": "Pipeline not initialized"}, status_code=503)

    suffix = Path(audio.filename).suffix if audio.filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(await audio.read())
        input_path = f.name

    try:
        processed_path = await preprocess_audio(input_path)
        result = pipeline.asr.transcribe(processed_path, language=language)
        return result
    finally:
        for p in [input_path, input_path.replace(suffix, "_processed.wav")]:
            if os.path.exists(p):
                try:
                    os.unlink(p)
                except Exception:
                    pass


@app.post("/v1/chat")
async def chat_only(
    message: str = Form(...),
    history: str = Form("[]"),
):
    """Send text to LLM only (no ASR or TTS)."""
    if not pipeline:
        return JSONResponse({"error": "Pipeline not initialized"}, status_code=503)

    try:
        history_list = json.loads(history)
    except json.JSONDecodeError:
        history_list = []

    reply = await pipeline.llm.chat(message, history=history_list)
    return {"reply": reply}


@app.get("/v1/audio/{filename}")
async def get_audio(filename: str):
    """Serve generated audio files."""
    path = f"/tmp/voice_pipeline_reply_{filename}"
    if not os.path.exists(path):
        # Try alternative path
        path = f"/tmp/{filename}"
    if os.path.exists(path):
        return FileResponse(path, media_type="audio/wav")
    return JSONResponse({"error": "Audio file not found"}, status_code=404)


@app.post("/v1/tts")
async def tts_only(
    text: str = Form(...),
    language: Optional[str] = Form(None),
):
    """Synthesize speech from text only."""
    if not pipeline:
        return JSONResponse({"error": "Pipeline not initialized"}, status_code=503)

    audio_bytes = await pipeline.tts.synthesize(text, language=language)
    if not audio_bytes:
        return JSONResponse({"error": "TTS synthesis failed"}, status_code=500)

    out_path = f"/tmp/voice_pipeline_tts_{int(time.time())}.wav"
    with open(out_path, "wb") as f:
        f.write(audio_bytes)

    return FileResponse(out_path, media_type="audio/wav")


# ─── CLI mode (for testing) ─────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Mufeed Voice Pipeline")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)
