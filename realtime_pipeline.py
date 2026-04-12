import os
import re
import io
import time
import json
import asyncio
import logging
from typing import AsyncGenerator

import numpy as np
import aiohttp
from faster_whisper import WhisperModel

log = logging.getLogger("realtime-pipeline")

# ---------------------------------------------------------------------------
# Config constants (mirrored from bot.py)
# ---------------------------------------------------------------------------
ASR_MODEL_CS = "/home/user/voice-pipeline/models/arabic-whisper-ct2"
ASR_MODEL_FAST = "large-v3-turbo"
ASR_COMPUTE_TYPE = "float16"
DEFAULT_ASR_MODEL = "turbo"

LLM_BASE_URL = "http://localhost:8083/v1"
LLM_MODEL = "gemma-4-26B-A4B-it-UD-Q5_K_XL.gguf"

TTS_URL = "http://localhost:8001/tts"
TTS_SAMPLE_RATE = 24000

COHERE_URL = "http://localhost:6000/v1/audio/transcriptions"
COHERE_MODEL = "CohereLabs/cohere-transcribe-03-2026"

# Regex for sentence boundary: punctuation followed by whitespace or end of string
_SENTENCE_BOUNDARY = re.compile(r"[.!?؟।。\n](?:\s|$)")
_MIN_SENTENCE_LEN = 15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_language(text: str) -> str:
    arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
    total_chars = sum(1 for c in text if c.isalpha())
    if total_chars == 0:
        return "en"
    return "ar" if arabic_chars / total_chars > 0.25 else "en"


# ---------------------------------------------------------------------------
# StreamingPipeline
# ---------------------------------------------------------------------------

class StreamingPipeline:
    """Streaming ASR -> LLM -> TTS pipeline orchestration."""

    def __init__(self, asr_model: str = DEFAULT_ASR_MODEL) -> None:
        t0 = time.monotonic()

        model_path = ASR_MODEL_CS if asr_model == "codeswitch" else ASR_MODEL_FAST
        device_index = 0 if os.environ.get("CUDA_VISIBLE_DEVICES") else 1

        self._whisper = WhisperModel(
            model_path,
            device="cuda",
            compute_type=ASR_COMPUTE_TYPE,
            device_index=device_index,
        )

        # Tiny Whisper on CPU for fast language detection (used by Cohere)
        self._whisper_tiny = WhisperModel("tiny", device="cpu", compute_type="int8")

        self._session: aiohttp.ClientSession | None = None

        elapsed = time.monotonic() - t0
        log.info("ASR model loaded in %.2fs (model=%s, device_index=%d)", elapsed, model_path, device_index)

    # -- internal -----------------------------------------------------------

    def _get_session(self) -> aiohttp.ClientSession:
        """Lazily create the aiohttp session inside a running event loop."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    # -- lifecycle ----------------------------------------------------------

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # -- ASR ----------------------------------------------------------------

    async def transcribe(self, audio: np.ndarray, engine: str = "cohere") -> str:
        """Transcribe audio. engine='cohere' or 'whisper'."""
        if engine == "cohere":
            return await self._transcribe_cohere(audio)
        return await self._transcribe_whisper(audio)

    async def _transcribe_whisper(self, audio: np.ndarray) -> str:
        def _run() -> str:
            segments, info = self._whisper.transcribe(
                audio,
                beam_size=5,
                language=None,
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": 300,
                    "speech_pad_ms": 200,
                },
            )
            return " ".join(seg.text.strip() for seg in segments)

        transcript = await asyncio.to_thread(_run)
        log.info("Whisper [%s]: %s", detect_language(transcript), transcript)
        return transcript

    async def _detect_language_tiny(self, audio: np.ndarray) -> str:
        """Fast language detection using tiny Whisper on CPU."""
        def _run() -> str:
            _, info = self._whisper_tiny.transcribe(
                audio, beam_size=1, language=None, vad_filter=False,
            )
            return info.language or "ar"
        return await asyncio.to_thread(_run)

    async def _transcribe_cohere(self, audio: np.ndarray) -> str:
        import soundfile as sf

        # Detect language first with tiny Whisper (CPU, ~100ms)
        lang = await self._detect_language_tiny(audio)
        log.info("Language detected (tiny): %s", lang)

        # Write audio to in-memory WAV
        buf = io.BytesIO()
        sf.write(buf, audio, 16000, format="WAV", subtype="PCM_16")
        buf.seek(0)

        session = self._get_session()
        form = aiohttp.FormData()
        form.add_field("file", buf, filename="audio.wav", content_type="audio/wav")
        form.add_field("model", COHERE_MODEL)
        form.add_field("language", lang)

        try:
            async with session.post(
                COHERE_URL, data=form, timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    log.error("Cohere HTTP %d: %s", resp.status, err[:200])
                    return ""
                data = await resp.json()
                transcript = data.get("text", "").strip()
                log.info("Cohere [%s]: %s", lang, transcript)
                return transcript
        except Exception:
            log.exception("Cohere transcription error")
            return ""

    # -- LLM (streaming SSE) ------------------------------------------------

    async def stream_llm(
        self, messages: list[dict], cancel: asyncio.Event
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from the LLM chat-completions endpoint."""
        session = self._get_session()
        payload = {
            "model": LLM_MODEL,
            "messages": messages,
            "stream": True,
            "max_tokens": 1024,
            "temperature": 0.7,
            "chat_template_kwargs": {"thinking": False},
        }
        timeout = aiohttp.ClientTimeout(total=60)

        try:
            async with session.post(
                f"{LLM_BASE_URL}/chat/completions",
                json=payload,
                timeout=timeout,
            ) as resp:
                async for raw_line in resp.content:
                    if cancel.is_set():
                        break

                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data: "):
                        continue

                    data = line[len("data: "):]
                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                        content = chunk["choices"][0]["delta"].get("content")
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
        except Exception:
            log.exception("LLM streaming error")
            yield "عذراً، حدث خطأ أثناء معالجة طلبك."

    # -- TTS (sentence-level streaming) -------------------------------------

    async def stream_tts_by_sentence(
        self,
        token_stream: AsyncGenerator[str, None],
        cancel: asyncio.Event,
    ) -> AsyncGenerator[tuple[str, bytes], None]:
        """Accumulate tokens into sentences, synthesise each via TTS."""
        session = self._get_session()
        buf = ""

        async for token in token_stream:
            if cancel.is_set():
                return

            buf += token

            # Try to split off complete sentences
            while True:
                match = _SENTENCE_BOUNDARY.search(buf)
                if match is None:
                    break

                end = match.end()
                # Enforce minimum sentence length before splitting
                if end < _MIN_SENTENCE_LEN:
                    break

                sentence = buf[:end].strip()
                buf = buf[end:]

                if not sentence:
                    continue

                result = await self._synthesise(session, sentence)
                if result is not None:
                    yield result

                if cancel.is_set():
                    return

        # Flush remaining buffer
        remaining = buf.strip()
        if remaining and not cancel.is_set():
            result = await self._synthesise(session, remaining)
            if result is not None:
                yield result

    async def _synthesise(
        self, session: aiohttp.ClientSession, sentence: str
    ) -> tuple[str, bytes] | None:
        lang = detect_language(sentence)
        try:
            async with session.post(
                TTS_URL,
                json={"text": sentence, "language": lang},
            ) as resp:
                wav_bytes = await resp.read()
            return (sentence, wav_bytes)
        except Exception:
            log.exception("TTS error for sentence: %.60s...", sentence)
            return None

    # -- health -------------------------------------------------------------

    async def health(self) -> dict:
        session = self._get_session()
        llm_ok = False
        tts_ok = False

        try:
            async with session.get(f"{LLM_BASE_URL}/models", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                llm_ok = resp.status == 200
        except Exception:
            pass

        try:
            async with session.get("http://localhost:8001/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                tts_ok = resp.status == 200
        except Exception:
            pass

        cohere_ok = False
        try:
            async with session.get("http://localhost:6000/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                cohere_ok = resp.status == 200
        except Exception:
            pass

        return {
            "asr": True,
            "llm": llm_ok,
            "tts": tts_ok,
            "cohere": cohere_ok,
        }
