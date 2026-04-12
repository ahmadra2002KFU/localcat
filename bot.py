#!/usr/bin/env python3
"""
Mufeed Voice Pipeline — fully local ASR + LLM + TTS
=====================================================
ASR:  Arabic-Whisper-CodeSwitching (CT2) or Whisper large-v3-turbo (faster-whisper)
LLM:  Gemma 4 via vLLM OpenAI-compatible API (port 8083)
TTS:  Chatterbox Multilingual (port 8001)

Usage:
  python bot.py interactive    — Real-time mic → speaker
  python bot.py file           — Drop .wav in input/ → get response in output/
  python bot.py demo           — Demo with sample conversation
  python bot.py asr-test       — Test ASR with microphone input only
"""

import asyncio
import json
import logging
import sys
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import aiohttp
import aiofiles

# ─── Config ───────────────────────────────────────────────────────────────────

# ASR
ASR_MODEL_CS = "/home/user/voice-pipeline/models/arabic-whisper-ct2"  # CodeSwitching (CT2)
ASR_MODEL_FAST = "large-v3-turbo"  # Fast Whisper (built-in)
ASR_DEVICE = "cuda"
ASR_DEVICE_INDEX = 1  # GPU 1 (most free VRAM)
ASR_COMPUTE_TYPE = "float16"
DEFAULT_ASR_MODEL = "codeswitch"  # "codeswitch" or "turbo"

# LLM
LLM_BASE_URL = "http://localhost:8083/v1"
LLM_MODEL = "gemma-4-26B-A4B-it-UD-Q5_K_XL.gguf"
LLM_API_KEY = "not-needed"

# TTS
TTS_URL = "http://localhost:8001/tts"
TTS_SAMPLE_RATE = 24000

# Audio
MIC_SAMPLE_RATE = 16000
BLOCK_SIZE_MS = 100

# VAD
SILENCE_THRESHOLD = 500  # RMS threshold
SILENCE_TIMEOUT_S = 1.5  # Seconds of silence to end utterance
MIN_UTTERANCE_S = 0.5  # Minimum recording length

# System Prompt
SYSTEM_PROMPT = """أنت مساعد ذكي يعمل في شركة مفيض (Mufeed Advanced Co). أنت تتحدث العربية الفصحى والإنجليزية بسلاسة. عندما يتحدث المستخدم بالعربية، رد بالعربية. عندما يتحدث بالإنجليزية، رد بالإنجليزية. كن محترفاً ومفيداً وأجب بإجابات مختصرة ومباشرة. لا تذكر أنك مساعد ذكاء اصطناعي.

You are a helpful AI assistant at Mufeed Advanced Co. You speak both Arabic (MSA) and English fluently. Match the user's language. Keep responses concise and professional. Do not mention you are an AI."""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mufeed-voice")


# ─── Language Detection ──────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    """Detect if text is primarily Arabic or English."""
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    total_chars = sum(1 for c in text if c.isalpha())
    if total_chars == 0:
        return "en"
    ratio = arabic_chars / total_chars
    return "ar" if ratio > 0.25 else "en"


# ─── ASR Engine ───────────────────────────────────────────────────────────────

class ASREngine:
    """Wraps faster-whisper for speech recognition."""

    def __init__(self, model_name=DEFAULT_ASR_MODEL):
        self.model_name = model_name
        self._model = None
        self._load()

    def _load(self):
        import time
        from faster_whisper import WhisperModel

        if self.model_name == "codeswitch":
            path = ASR_MODEL_CS
            log.info(f"Loading Arabic-Whisper CodeSwitching from {path}...")
        else:
            path = ASR_MODEL_FAST
            log.info(f"Loading Whisper {path}...")

        # CUDA_VISIBLE_DEVICES remaps device indices, so use 0 when env var is set
        device_index = 0 if os.environ.get("CUDA_VISIBLE_DEVICES") else ASR_DEVICE_INDEX

        start = time.time()
        self._model = WhisperModel(
            path,
            device=ASR_DEVICE,
            compute_type=ASR_COMPUTE_TYPE,
            device_index=device_index,
        )
        log.info(f"ASR model loaded in {time.time()-start:.1f}s")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe audio numpy array. Returns transcript string."""
        segments, info = self._model.transcribe(
            audio,
            beam_size=5,
            language=None,  # auto-detect for code-switching
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=300,
                speech_pad_ms=200,
            ),
        )
        texts = []
        for seg in segments:
            texts.append(seg.text.strip())
        text = " ".join(texts).strip()
        if text:
            lang = info.language
            prob = info.language_probability
            log.info(f"[ASR] lang={lang}({prob:.0%}): {text}")
        return text

    def switch_model(self, model_name: str):
        """Switch to a different ASR model at runtime."""
        if model_name != self.model_name:
            log.info(f"Switching ASR from {self.model_name} to {model_name}...")
            self.model_name = model_name
            self._load()


# ─── LLM Client ──────────────────────────────────────────────────────────────

class LLMClient:
    """OpenAI-compatible LLM client for vLLM."""

    def __init__(self):
        self._session = None
        self._conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

    async def start(self):
        import aiohttp
        self._session = aiohttp.ClientSession()

    async def stop(self):
        if self._session:
            await self._session.close()

    async def chat(self, user_message: str) -> str:
        """Send a message and get the response."""
        self._conversation.append({"role": "user", "content": user_message})

        payload = {
            "model": LLM_MODEL,
            "messages": self._conversation,
            "max_tokens": 256,
            "temperature": 0.7,
        }

        try:
            async with self._session.post(
                f"{LLM_BASE_URL}/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                data = await resp.json()
                reply = data["choices"][0]["message"]["content"].strip()
                self._conversation.append({"role": "assistant", "content": reply})
                log.info(f"[LLM] {reply[:100]}...")
                return reply
        except Exception as e:
            log.error(f"[LLM] Error: {e}")
            return "عذراً، حدث خطأ في النظام."

    def clear_history(self):
        self._conversation = [{"role": "system", "content": SYSTEM_PROMPT}]


# ─── TTS Client ───────────────────────────────────────────────────────────────

class TTSClient:
    """Chatterbox TTS client."""

    def __init__(self):
        self._session = None

    async def start(self):
        import aiohttp
        self._session = aiohttp.ClientSession()

    async def stop(self):
        if self._session:
            await self._session.close()

    async def synthesize(self, text: str) -> np.ndarray | None:
        """Synthesize text to audio. Returns float32 numpy array."""
        if not text.strip():
            return None

        lang = detect_language(text)

        try:
            async with self._session.post(
                TTS_URL,
                json={"text": text, "language": lang},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    log.error(f"[TTS] Error {resp.status}: {err[:200]}")
                    return None
                content = await resp.read()
                import io
                audio, sr = sf.read(io.BytesIO(content))
                if len(audio.shape) > 1:
                    audio = audio[:, 0]
                # Resample if needed
                if sr != TTS_SAMPLE_RATE:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=TTS_SAMPLE_RATE)
                return audio.astype(np.float32)
        except Exception as e:
            log.error(f"[TTS] Error: {e}")
            return None


# ─── Audio Player ─────────────────────────────────────────────────────────────

class AudioPlayer:
    """Plays audio through system speakers."""

    def __init__(self, sample_rate=TTS_SAMPLE_RATE):
        self._sample_rate = sample_rate
        self._stream = None

    def start(self):
        import sounddevice as sd
        self._stream = sd.OutputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype='float32',
        )
        self._stream.start()

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()

    def play(self, audio: np.ndarray):
        """Play audio (blocking)."""
        if self._stream and len(audio) > 0:
            self._stream.write(audio.reshape(-1, 1))

    async def play_async(self, audio: np.ndarray):
        """Play audio without blocking the event loop."""
        import sounddevice as sd
        sd.play(audio, samplerate=self._sample_rate)
        # Wait for playback to finish
        while sd.get_stream().active:
            await asyncio.sleep(0.1)


# ─── Microphone Input ────────────────────────────────────────────────────────

class MicRecorder:
    """Records from microphone with VAD."""

    def __init__(self, sample_rate=MIC_SAMPLE_RATE, block_size=BLOCK_SIZE_MS):
        self._sample_rate = sample_rate
        self._block_size = int(sample_rate * block_size / 1000)
        self._stream = None
        self._buffer = []

    def start(self):
        import sounddevice as sd
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype='int16',
            blocksize=self._block_size,
        )
        self._stream.start()

    def stop(self):
        if self._stream:
            self._stream.stop()

    def read_chunk(self) -> np.ndarray | None:
        """Read one chunk from mic. Returns float32 numpy or None."""
        if self._stream and self._stream.read_available >= self._block_size:
            data, _ = self._stream.read(self._block_size)
            audio = data[:, 0].astype(np.float32) / 32767.0
            return audio
        return None

    def reset_buffer(self):
        self._buffer = []

    def append(self, audio: np.ndarray):
        self._buffer.append(audio)

    def get_buffer(self) -> np.ndarray:
        return np.concatenate(self._buffer) if self._buffer else np.array([], dtype=np.float32)


# ─── Voice Pipeline ──────────────────────────────────────────────────────────

class VoicePipeline:
    """Main pipeline orchestrating ASR → LLM → TTS."""

    def __init__(self, asr_model="codeswitch"):
        self.asr = ASREngine(model_name=asr_model)
        self.llm = LLMClient()
        self.tts = TTSClient()
        self.player = AudioPlayer()
        self.mic = MicRecorder()

    async def start(self):
        await self.llm.start()
        await self.tts.start()
        self.player.start()
        self.mic.start()
        log.info("Voice pipeline started ✓")

    async def stop(self):
        self.mic.stop()
        self.player.stop()
        await self.tts.stop()
        await self.llm.stop()

    async def process_audio(self, audio: np.ndarray) -> str | None:
        """Process audio through full pipeline: ASR → LLM → TTS."""
        # ASR
        text = self.asr.transcribe(audio)
        if not text:
            return None

        # LLM
        reply = await self.llm.chat(text)
        if not reply:
            return None

        # TTS
        audio_out = await self.tts.synthesize(reply)
        if audio_out is not None:
            await self.player.play_async(audio_out)

        return reply


# ─── Interactive Mode ────────────────────────────────────────────────────────

async def run_interactive(asr_model="codeswitch"):
    """Real-time voice conversation."""
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║         Mufeed Voice Pipeline — Interactive Mode        ║")
    print("║                                                          ║")
    print("║  ASR: Arabic-Whisper CodeSwitching / Whisper v3-turbo   ║")
    print("║  LLM: Gemma 4  (vLLM)                                  ║")
    print("║  TTS: Chatterbox Multilingual                           ║")
    print("║                                                          ║")
    print("║  Commands:                                              ║")
    print("║    'switch turbo'     — Switch to Whisper v3-turbo      ║")
    print("║    'switch codeswitch'— Switch to CodeSwitching model   ║")
    print("║    'clear'           — Clear conversation history       ║")
    print("║    Ctrl+C            — Exit                             ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    pipeline = VoicePipeline(asr_model=asr_model)
    await pipeline.start()

    silence_blocks = 0
    is_speaking = False

    try:
        while True:
            chunk = pipeline.mic.read_chunk()
            if chunk is None:
                await asyncio.sleep(0.05)
                continue

            rms = np.sqrt(np.mean(chunk ** 2))

            if rms > SILENCE_THRESHOLD:
                if not is_speaking:
                    is_speaking = True
                    pipeline.mic.reset_buffer()
                    log.debug("🎤 User started speaking...")
                silence_blocks = 0
                pipeline.mic.append(chunk)
            elif is_speaking:
                silence_blocks += 1
                pipeline.mic.append(chunk)
                if silence_blocks > int(SILENCE_TIMEOUT_S / (BLOCK_SIZE_MS / 1000)):
                    is_speaking = False
                    audio = pipeline.mic.get_buffer()
                    duration = len(audio) / MIC_SAMPLE_RATE

                    if duration > MIN_UTTERANCE_S:
                        print(f"\n🎤 [{duration:.1f}s] Processing...")
                        try:
                            reply = await pipeline.process_audio(audio)
                            if reply:
                                print(f"🤖 {reply}\n")
                        except Exception as e:
                            log.error(f"Pipeline error: {e}")

                    pipeline.mic.reset_buffer()
                    silence_blocks = 0

            await asyncio.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        await pipeline.stop()


# ─── ASR Test Mode ────────────────────────────────────────────────────────────

async def run_asr_test(asr_model="codeswitch"):
    """Test ASR only - transcribe from mic, print text."""
    print(f"\n🎤 ASR Test Mode ({asr_model})")
    print("Speak into your microphone. Ctrl+C to exit.\n")

    asr = ASREngine(model_name=asr_model)

    import sounddevice as sd
    stream = sd.InputStream(samplerate=MIC_SAMPLE_RATE, channels=1, dtype='int16',
                           blocksize=int(MIC_SAMPLE_RATE * BLOCK_SIZE_MS / 1000))
    stream.start()

    buffer = []
    silence_blocks = 0
    is_speaking = False

    try:
        while True:
            if stream.read_available >= stream.blocksize:
                data, _ = stream.read(stream.blocksize)
                audio = data[:, 0].astype(np.float32) / 32767.0
                rms = np.sqrt(np.mean(audio ** 2))

                if rms > SILENCE_THRESHOLD:
                    if not is_speaking:
                        is_speaking = True
                        buffer = []
                    silence_blocks = 0
                    buffer.append(audio)
                elif is_speaking:
                    silence_blocks += 1
                    if silence_blocks > int(SILENCE_TIMEOUT_S / (BLOCK_SIZE_MS / 1000)):
                        is_speaking = False
                        full_audio = np.concatenate(buffer)
                        duration = len(full_audio) / MIC_SAMPLE_RATE
                        if duration > MIN_UTTERANCE_S:
                            text = asr.transcribe(full_audio)
                            if text:
                                print(f"[{duration:.1f}s] {text}")
                        buffer = []
                        silence_blocks = 0

            await asyncio.sleep(0.05)
    except KeyboardInterrupt:
        print("\nDone.")
    finally:
        stream.stop()


# ─── File Mode ────────────────────────────────────────────────────────────────

async def run_file_mode(asr_model="codeswitch"):
    """Process .wav files from input/ directory."""
    input_dir = Path("/home/user/voice-pipeline/input")
    output_dir = Path("/home/user/voice-pipeline/output")
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 File Mode — drop .wav files in {input_dir}")
    print(f"   Output will be in {output_dir}")
    print("   Ctrl+C to exit.\n")

    pipeline = VoicePipeline(asr_model=asr_model)
    await pipeline.start()

    processed = set()

    try:
        while True:
            for wav_file in sorted(input_dir.glob("*.wav")):
                if wav_file.name in processed:
                    continue
                processed.add(wav_file.name)

                print(f"📄 Processing: {wav_file.name}")
                audio, sr = sf.read(str(wav_file))
                if len(audio.shape) > 1:
                    audio = audio[:, 0]

                # Resample to 16kHz
                if sr != MIC_SAMPLE_RATE:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=MIC_SAMPLE_RATE)

                reply = await pipeline.process_audio(audio)
                if reply:
                    print(f"🤖 {reply}")

            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nDone.")
    finally:
        await pipeline.stop()


# ─── Demo Mode ────────────────────────────────────────────────────────────────

async def run_demo():
    """Run a demo with pre-recorded text inputs (no mic needed)."""
    print("\n🎮 Demo Mode — testing the full pipeline with sample inputs\n")

    pipeline = VoicePipeline()
    # Override ASR to just return the text (no audio needed)
    await pipeline.start()

    demos = [
        "مرحبا، كيف حالك؟",
        "What is machine learning?",
        "أخبرني عن شركة مفيض",
        "What can you help me with?",
    ]

    for text in demos:
        print(f"👤 {text}")
        reply = await pipeline.llm.chat(text)
        audio = await pipeline.tts.synthesize(reply)
        if audio is not None:
            print(f"🔊 Speaking ({len(audio)/TTS_SAMPLE_RATE:.1f}s)...")
            await pipeline.player.play_async(audio)
        print(f"🤖 {reply}")
        print()

    await pipeline.stop()
    print("Demo complete ✓")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "interactive"
    asr_model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_ASR_MODEL

    if mode == "interactive":
        asyncio.run(run_interactive(asr_model))
    elif mode == "file":
        asyncio.run(run_file_mode(asr_model))
    elif mode == "demo":
        asyncio.run(run_demo())
    elif mode == "asr-test":
        asyncio.run(run_asr_test(asr_model))
    else:
        print(f"Usage: {sys.argv[0]} [interactive|file|demo|asr-test] [codeswitch|turbo]")
